//! A bounded paired measurement of exact greedy action-average reuse.
//!
//! Only whole current-layer fields are copied. The immutable next snapshot is
//! read through the production file lookup, never pinned as a complete layer.
//! Both arms use the same total cache allowance and the same greedy traversal.

use super::*;
use crate::action_avg_cache::{ActionSource, ValidatedPlacements};
use crate::layer_store::FieldChainLayer;
use crate::loop1_whole_operator::InputGuard;
use crate::pre_reveal::PreRevealIndex;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;

const MIB: usize = 1 << 20;
const MAX_STATES: usize = 3_000_000;
const MAX_ARTIFACT_BYTES: u64 = 256 << 20;
// During atomic publication, the temporary and final hard-link names can both
// be counted by a logical directory-size monitor although they share an inode.
const SUMMARY_RESERVE_BYTES: u64 = 2 * MIB as u64;

struct Config {
    graph: PathBuf,
    pruned: PathBuf,
    values: PathBuf,
    work: PathBuf,
    layer: usize,
    maximum: usize,
    cycles: usize,
    cache_bytes: usize,
    memo_bytes: usize,
    start: Option<usize>,
}

fn number(args: &[String], at: usize, default: usize, name: &str) -> Result<usize> {
    args.get(at)
        .map(|value| value.parse::<usize>())
        .transpose()
        .with_context(|| format!("invalid {name}"))
        .map(|value| value.unwrap_or(default))
}

impl Config {
    fn parse(args: &[String]) -> Result<Self> {
        anyhow::ensure!((6..=12).contains(&args.len()),
            "usage: greedy-cache-probe <graph.bin> <pruned> <values> <work_parent> [layer=5] [max_current_states=2000000] [ABBA_cycles=2] [cache_mib=768] [memo_mib=1] [start_field_index=midpoint_for_L5_or_0_for_L2]");
        let layer = number(args, 6, 5, "layer")?;
        anyhow::ensure!(layer == 2 || layer == 5, "layer must be 2 or 5");
        let maximum = number(
            args,
            7,
            if layer == 2 { 512_000 } else { 2_000_000 },
            "max_current_states",
        )?;
        let cycles = number(args, 8, 2, "ABBA_cycles")?;
        let cache_mib = number(args, 9, 768, "cache_mib")?;
        let memo_mib = number(args, 10, 1, "memo_mib")?;
        let start = args
            .get(11)
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("invalid start_field_index")?;
        // Validate every numerical argument before opening or parsing sources.
        anyhow::ensure!(
            (1..=MAX_STATES).contains(&maximum),
            "max_current_states must be 1..={MAX_STATES}"
        );
        anyhow::ensure!((1..=4).contains(&cycles), "ABBA_cycles must be 1..=4");
        anyhow::ensure!(
            (64..=1024).contains(&cache_mib),
            "cache_mib must be 64..=1024"
        );
        anyhow::ensure!(
            (1..=256).contains(&memo_mib) && memo_mib < cache_mib,
            "memo_mib must be 1..=256 and less than cache_mib"
        );
        Ok(Self {
            graph: PathBuf::from(&args[2]),
            pruned: PathBuf::from(&args[3]),
            values: PathBuf::from(&args[4]),
            work: PathBuf::from(&args[5]),
            layer,
            maximum,
            cycles,
            cache_bytes: cache_mib * MIB,
            memo_bytes: memo_mib * MIB,
            start,
        })
    }
}

fn whole_fields(layer: &FieldChainLayer, start: usize, maximum: usize) -> Result<Range<usize>> {
    anyhow::ensure!(
        start < layer.metas.len(),
        "start_field_index outside current-layer membership"
    );
    let start = (start..layer.metas.len())
        .find(|&i| layer.metas[i].count != 0)
        .context("no nonempty field after start_field_index")?;
    let mut states = 0usize;
    let mut end = start;
    for meta in &layer.metas[start..] {
        let next = states
            .checked_add(meta.count as usize)
            .context("selected state count overflow")?;
        if next > maximum {
            break;
        }
        states = next;
        end += 1;
    }
    anyhow::ensure!(
        states > 0,
        "first whole field exceeds max_current_states={maximum}"
    );
    Ok(start..end)
}

fn create_new_file(path: &Path, bytes: u64) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create fresh probe output {}", path.display()))?;
    file.set_len(bytes)?;
    Ok(file)
}

fn fresh_directory(parent: &Path) -> Result<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for attempt in 0..100 {
        let path = parent.join(format!(
            "greedy-cache-probe-{}-{stamp}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("create owned greedy-cache probe directory"),
        }
    }
    anyhow::bail!("could not create a fresh greedy-cache probe directory")
}

fn copy_current(
    original: &FieldChainLayer,
    range: Range<usize>,
    owned: &Path,
) -> Result<FieldChainLayer> {
    let metas = &original.metas[range];
    let first = metas.first().context("empty selected field range")?;
    let last = metas.last().context("empty selected field range")?;
    let count = last
        .chain_offset
        .checked_add(u64::from(last.count))
        .and_then(|end| end.checked_sub(first.chain_offset))
        .context("current slab offset overflow")?;
    let prefix = owned.join("current");
    let mut index = BufWriter::new(create_new_file(
        &prefix.with_extension("field_index.bin"),
        0,
    )?);
    index.write_all(b"FCHNIDX1")?;
    index.write_all(&1u32.to_le_bytes())?;
    index.write_all(&u32::try_from(metas.len())?.to_le_bytes())?;
    index.write_all(&count.to_le_bytes())?;
    for meta in metas {
        index.write_all(&meta.field_id.to_le_bytes())?;
        index.write_all(&meta.count.to_le_bytes())?;
        index.write_all(&(meta.chain_offset - first.chain_offset).to_le_bytes())?;
    }
    index.flush()?;
    let input = File::open(&original.chains_path)?;
    let mut output = BufWriter::new(create_new_file(
        &prefix.with_extension("chains.u24.bin"),
        0,
    )?);
    let mut buffer = vec![0u8; 256 << 10];
    let mut copied = 0u64;
    let bytes = count
        .checked_mul(3)
        .context("current slab byte count overflow")?;
    let base = first
        .chain_offset
        .checked_mul(3)
        .context("current slab byte offset overflow")?;
    while copied < bytes {
        let length = usize::try_from((bytes - copied).min(buffer.len() as u64))?;
        read_exact_at_file(&input, &mut buffer[..length], base + copied)?;
        output.write_all(&buffer[..length])?;
        copied += length as u64;
    }
    output.flush()?;
    FieldChainLayer::open_index_only(prefix)
}

fn validate_metadata(layer: &FieldChainLayer, graph: &FieldGraph) -> Result<()> {
    anyhow::ensure!(
        layer
            .metas
            .iter()
            .all(|meta| (meta.field_id as usize) < graph.hashes.len()
                && meta.count as usize <= CHECK_QSTATE_COUNT),
        "field metadata exceeds graph/qstate domain"
    );
    Ok(())
}

fn validate_current_qids(layer: &FieldChainLayer) -> Result<()> {
    for bi in 0..layer.metas.len() {
        let qids = layer.read_block_chains(bi);
        anyhow::ensure!(
            qids.iter().all(|&qid| (qid as usize) < CHECK_QSTATE_COUNT)
                && qids.windows(2).all(|pair| pair[0] < pair[1]),
            "selected current field has unordered/out-of-domain qids"
        );
    }
    Ok(())
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 32 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn source_provenance(paths: &[PathBuf]) -> Result<String> {
    let mut sources = Vec::new();
    for path in paths {
        let metadata = fs::metadata(path)?;
        let canonical = fs::canonicalize(path)?;
        sources.push(format!("{{\"path\":{},\"canonical\":{},\"device\":{},\"inode\":{},\"bytes\":{},\"mtime_seconds\":{},\"mtime_nanoseconds\":{},\"ctime_seconds\":{},\"ctime_nanoseconds\":{}}}",
            json_string(&path.to_string_lossy()), json_string(&canonical.to_string_lossy()),
            metadata.dev(), metadata.ino(), metadata.len(), metadata.mtime(), metadata.mtime_nsec(),
            metadata.ctime(), metadata.ctime_nsec()));
    }
    Ok(format!("[{}]", sources.join(",")))
}

fn process_cpu_seconds() -> Result<f64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the initialized timespec is writable for this synchronous call.
    anyhow::ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time) } == 0,
        "cannot read process CPU clock: {}",
        std::io::Error::last_os_error()
    );
    Ok(time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9)
}

struct Kernel<'a> {
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    legal: &'a [[u8; 2]],
    current: &'a FieldChainLayer,
    next: &'a FieldChainLayer,
    values: &'a Path,
    terminal: &'a [[f32; PIECE_COUNT]],
    placements: &'a ValidatedPlacements<'a>,
    config: &'a Config,
    two_line: FieldId,
}

struct Sample {
    memo: bool,
    wall_seconds: f64,
    cpu_seconds: f64,
    stats: FullMdpLayerStats,
}

impl Kernel<'_> {
    fn apply(&self, memo: bool, output: &mut [f32], policy: &File) -> Result<Sample> {
        let cpu = process_cpu_seconds()?;
        let began = Instant::now();
        let result = compute_full_mdp_range_group_f32_rolling_sinks_with_action_cache(
            self.config.layer,
            &[0..self.current.metas.len()],
            self.current,
            None,
            Some(SharedF32Out {
                ptr: output.as_mut_ptr(),
                len: output.len(),
            }),
            Some(policy),
            self.graph,
            self.qfwd,
            self.legal,
            self.terminal,
            Some(self.next),
            Some(self.values),
            None,
            None,
            1.0,
            self.config.cache_bytes,
            CHECK_QSTATE_COUNT,
            65_536,
            self.two_line,
            false,
            false,
            true,
            true,
            None,
            memo.then_some(ActionSource {
                placements: self.placements,
                requested_bytes: self.config.memo_bytes,
            }),
        )?;
        let wall_seconds = began.elapsed().as_secs_f64();
        let cpu_seconds = process_cpu_seconds()? - cpu;
        anyhow::ensure!(
            result.blocks == self.current.metas.len()
                && result.stats.total_states == self.current.total_states,
            "kernel did not cover the selected whole fields"
        );
        if memo {
            anyhow::ensure!(
                result.stats.action_avg_hits + result.stats.action_avg_misses
                    == result.stats.action_count,
                "memo did not account for every greedy candidate"
            );
            anyhow::ensure!(
                result.stats.action_avg_cache_bytes > 0,
                "requested memo did not fit the total cache allowance"
            );
        }
        Ok(Sample {
            memo,
            wall_seconds,
            cpu_seconds,
            stats: result.stats,
        })
    }
}

fn compare_values(expected: &[f32], actual: &[f32]) -> Result<()> {
    anyhow::ensure!(expected.len() == actual.len(), "greedy value count changed");
    for (index, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite greedy value at {index}"
        );
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "greedy value bits changed at {index}: {a:?} vs {b:?}"
        );
    }
    Ok(())
}

fn compare_stats(expected: &FullMdpLayerStats, actual: &FullMdpLayerStats) -> Result<()> {
    anyhow::ensure!(
        [
            expected.total_states == actual.total_states,
            expected.nonzero == actual.nonzero,
            expected.action_count == actual.action_count,
            expected.missing_next_terms == actual.missing_next_terms,
            expected.terminal_2_edges == actual.terminal_2_edges,
            expected.terminal_4_edges == actual.terminal_4_edges,
            expected.max_v.to_bits() == actual.max_v.to_bits(),
            expected.sum_v.to_bits() == actual.sum_v.to_bits(),
            expected.max_delta.to_bits() == actual.max_delta.to_bits(),
            expected.sum_abs_delta.to_bits() == actual.sum_abs_delta.to_bits(),
        ]
        .into_iter()
        .all(|same| same),
        "greedy mathematical/action statistics changed"
    );
    Ok(())
}

fn compare_policy(expected: &[u8], path: &Path) -> Result<()> {
    let actual = fs::read(path)?;
    anyhow::ensure!(expected == actual, "greedy u16 policy bytes changed");
    Ok(())
}

fn sample_json(sample: &Sample, phase: &str, cycle: usize, position: usize) -> String {
    let s = &sample.stats;
    let current_bytes = s.total_states * 3;
    let next_bytes = s.dense_materialized_states * 7;
    format!(concat!("{{\"phase\":{},\"cycle\":{},\"position\":{},\"memo\":{},",
        "\"wall_seconds\":{:.9},\"process_cpu_seconds\":{:.9},\"states\":{},\"actions\":{},",
        "\"missing_zero_terms\":{},\"action_cache_hits\":{},\"action_cache_misses\":{},",
        "\"action_cache_collisions\":{},\"action_cache_saved_reveals\":{},\"action_cache_payload_bytes\":{},",
        "\"lookup_hits\":{},\"lookup_misses\":{},\"lookup_evictions\":{},\"lookup_materialized_states\":{},",
        "\"current_membership_requested_bytes\":{},\"next_membership_values_requested_bytes\":{},",
        "\"total_logical_requested_source_bytes\":{},\"physical_disk_bytes_measured\":false}}"),
        json_string(phase), cycle, position, sample.memo, sample.wall_seconds, sample.cpu_seconds,
        s.total_states, s.action_count, s.missing_next_terms, s.action_avg_hits, s.action_avg_misses,
        s.action_avg_collisions, s.action_avg_saved_reveals, s.action_avg_cache_bytes,
        s.dense_cache_hits, s.dense_cache_misses, s.dense_evictions, s.dense_materialized_states,
        current_bytes, next_bytes, current_bytes + next_bytes)
}

fn median(samples: &[f64]) -> Result<f64> {
    anyhow::ensure!(
        !samples.is_empty()
            && samples
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        "invalid timing samples"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let config = Config::parse(args)?;
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run greedy-cache-probe in release mode"
    );
    let setup = Instant::now();
    let current_prefix = q_pruned_prefix(&config.pruned, config.layer);
    let next_prefix = q_pruned_prefix(&config.pruned, config.layer + 1);
    let next_values = mdp_f32_value_path(&config.values, config.layer + 1);
    let source_paths = vec![
        config.graph.clone(),
        current_prefix.with_extension("field_index.bin"),
        current_prefix.with_extension("chains.u24.bin"),
        next_prefix.with_extension("field_index.bin"),
        next_prefix.with_extension("chains.u24.bin"),
        next_values.clone(),
    ];
    let guard = InputGuard::capture(source_paths.clone())?;
    let source_versions = source_provenance(&source_paths)?;
    let parent = fs::canonicalize(&config.work).context("work_parent must already exist")?;
    anyhow::ensure!(parent.is_dir(), "work_parent must be a directory");
    anyhow::ensure!(
        !parent.starts_with(fs::canonicalize(&config.pruned)?)
            && !parent.starts_with(fs::canonicalize(&config.values)?),
        "work_parent must be outside source dataset directories"
    );
    let graph = FieldGraph::load_graph_bin(&config.graph)?;
    let q_index = {
        let old = enumerate_states();
        build_q_index_from_old(&old)
    };
    anyhow::ensure!(
        q_index.states.len() == CHECK_QSTATE_COUNT,
        "unexpected quotient domain size"
    );
    let qfwd = build_qforward_graph(&q_index);
    let legal = build_qfwd_legal_placed_table(&qfwd, CHECK_QSTATE_COUNT);
    let original = FieldChainLayer::open_index_only(&current_prefix)?;
    let next = FieldChainLayer::open_index_only(&next_prefix)?;
    validate_metadata(&original, &graph)?;
    validate_metadata(&next, &graph)?;
    anyhow::ensure!(
        fs::metadata(&next_values)?.len()
            == next
                .total_states
                .checked_mul(4)
                .context("next value size overflow")?,
        "next snapshot byte count differs from next membership"
    );
    let range = whole_fields(
        &original,
        config.start.unwrap_or(if config.layer == 5 {
            original.metas.len() / 2
        } else {
            0
        }),
        config.maximum,
    )?;
    let first = &original.metas[range.start];
    let last = &original.metas[range.end - 1];
    let states = last.chain_offset + u64::from(last.count) - first.chain_offset;
    let artifact_bound = states
        .checked_mul(15)
        .and_then(|n| n.checked_add((range.len() as u64) * 16 + 24 + SUMMARY_RESERVE_BYTES))
        .context("probe artifact bound overflow")?;
    anyhow::ensure!(
        states <= config.maximum as u64 && artifact_bound < MAX_ARTIFACT_BYTES,
        "selected probe exceeds current-state or artifact limit"
    );
    guard.check()?;
    let owned = fresh_directory(&parent)?;
    let cur = copy_current(&original, range.clone(), &owned)?;
    drop(original);
    validate_current_qids(&cur)?;
    let placement_start = Instant::now();
    let pre = PreRevealIndex::build(&q_index)?;
    let placements = ValidatedPlacements::build(&q_index, &qfwd, &pre)?;
    let placement_seconds = placement_start.elapsed().as_secs_f64();
    let placement_bytes = placements.storage_bytes();
    // L2 and L5 have no terminal reward branches; the table is never read.
    let terminal = vec![[0.0f32; PIECE_COUNT]; CHECK_QSTATE_COUNT];
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("two-line field missing")?;
    let baseline_policy_path = owned.join("baseline.policy.u16.bin");
    let memo_policy_path = owned.join("memo.policy.u16.bin");
    let baseline_policy = create_new_file(&baseline_policy_path, states * 2)?;
    let memo_policy = create_new_file(&memo_policy_path, states * 2)?;
    let mut baseline = vec![0.0f32; usize::try_from(states)?];
    let mut cached = vec![0.0f32; baseline.len()];
    let kernel = Kernel {
        graph: &graph,
        qfwd: &qfwd,
        legal: &legal,
        current: &cur,
        next: &next,
        values: &next_values,
        terminal: &terminal,
        placements: &placements,
        config: &config,
        two_line,
    };
    guard.check()?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    println!("greedy-cache-probe setup layer={} scope=whole_current_fields kernel=production_greedy next_source=file_lookup full_next_layer_pinned=0 full_loop_or_Vstar_speedup_measured=0 original_block_start={} original_block_end={} fields={} states={} total_cache_bytes={} requested_memo_bytes={} placement_storage_bytes={} placement_build_seconds={placement_seconds:.9} setup_seconds={setup_seconds:.9} setup_excluded_from_apply_timing=1 output_dir={}",
        config.layer, range.start, range.end, cur.metas.len(), states, config.cache_bytes, config.memo_bytes, placement_bytes, owned.display());
    println!("greedy-cache-probe sources json={source_versions}");
    let mut samples_json = Vec::new();
    let reference = kernel.apply(false, &mut baseline, &baseline_policy)?;
    compare_values(&baseline, &baseline)?;
    let expected_values = baseline.clone();
    let expected_policy = fs::read(&baseline_policy_path)?;
    anyhow::ensure!(
        expected_policy.len() as u64 == states * 2,
        "baseline policy length changed"
    );
    samples_json.push(sample_json(&reference, "warmup", 0, 0));
    println!(
        "greedy-cache-probe warmup json={}",
        samples_json.last().unwrap()
    );
    guard.check()?;
    let warm_memo = kernel.apply(true, &mut cached, &memo_policy)?;
    compare_values(&expected_values, &cached)?;
    compare_policy(&expected_policy, &memo_policy_path)?;
    compare_stats(&reference.stats, &warm_memo.stats)?;
    samples_json.push(sample_json(&warm_memo, "warmup", 0, 1));
    println!(
        "greedy-cache-probe warmup json={}",
        samples_json.last().unwrap()
    );
    let mut base_wall = Vec::new();
    let mut memo_wall = Vec::new();
    let mut base_cpu = Vec::new();
    let mut memo_cpu = Vec::new();
    for cycle in 0..config.cycles {
        for (position, memo) in [false, true, true, false].into_iter().enumerate() {
            guard.check()?;
            let sample = kernel.apply(
                memo,
                if memo { &mut cached } else { &mut baseline },
                if memo { &memo_policy } else { &baseline_policy },
            )?;
            compare_values(&expected_values, if memo { &cached } else { &baseline })?;
            compare_policy(
                &expected_policy,
                if memo {
                    &memo_policy_path
                } else {
                    &baseline_policy_path
                },
            )?;
            compare_stats(&reference.stats, &sample.stats)?;
            guard.check()?;
            if memo {
                memo_wall.push(sample.wall_seconds);
                memo_cpu.push(sample.cpu_seconds);
            } else {
                base_wall.push(sample.wall_seconds);
                base_cpu.push(sample.cpu_seconds);
            }
            samples_json.push(sample_json(&sample, "measured", cycle, position));
            println!("greedy-cache-probe sample cycle={cycle} position={position} memo={} all_f32_bits_equal=1 all_u16_policy_bytes_equal=1 mathematical_stats_equal=1 json={}", memo as u8, samples_json.last().unwrap());
        }
    }
    let base_wall = median(&base_wall)?;
    let memo_wall = median(&memo_wall)?;
    let base_cpu = median(&base_cpu)?;
    let memo_cpu = median(&memo_cpu)?;
    anyhow::ensure!(
        base_wall > 0.0 && memo_wall > 0.0 && base_cpu > 0.0 && memo_cpu > 0.0,
        "zero timing median"
    );
    write_greedy_ram_snapshot(&owned.join("baseline.values.f32.bin"), &baseline)?;
    write_greedy_ram_snapshot(&owned.join("memo.values.f32.bin"), &cached)?;
    guard.check()?;
    let existing_bytes = fs::read_dir(&owned)?.try_fold(0u64, |total, entry| -> Result<u64> {
        Ok(total
            .checked_add(entry?.metadata()?.len())
            .context("output size overflow")?)
    })?;
    anyhow::ensure!(
        existing_bytes + SUMMARY_RESERVE_BYTES <= artifact_bound
            && existing_bytes + SUMMARY_RESERVE_BYTES < MAX_ARTIFACT_BYTES,
        "probe artifact budget exceeded"
    );
    let summary = format!(concat!("{{\"schema_version\":1,\"status\":\"PASS\",\"layer\":{},",
        "\"scope\":\"bounded whole-field greedy kernel comparison on an immutable existing next snapshot\",",
        "\"original_block_start\":{},\"original_block_end\":{},\"fields\":{},\"states\":{},",
        "\"order\":\"ABBA\",\"ABBA_cycles\":{},\"samples_per_arm\":{},\"sequential_kernel\":true,",
        "\"total_cache_bytes\":{},\"requested_memo_bytes\":{},\"equal_total_cache_budget\":true,",
        "\"placement_storage_bytes\":{},\"placement_build_seconds\":{:.9},\"setup_seconds\":{:.9},",
        "\"baseline_median_seconds\":{:.9},\"memo_median_seconds\":{:.9},\"wall_speedup\":{:.9},",
        "\"baseline_median_process_cpu_seconds\":{:.9},\"memo_median_process_cpu_seconds\":{:.9},\"cpu_speedup\":{:.9},",
        "\"all_states_bit_identical_f32\":true,\"all_policy_bytes_identical_u16\":true,\"mathematical_stats_equal\":true,",
        "\"source_versions_unchanged\":true,\"source_files_written\":false,\"source_payloads_hashed\":false,",
        "\"full_next_layer_pinned\":false,\"full_next_membership_validation\":false,\"next_source\":\"file_lookup\",",
        "\"setup_amortization_measured\":false,\"full_loop_speedup_measured\":false,\"Vstar_convergence_measured\":false,",
        "\"physical_disk_bytes_measured\":false,\"artifact_bound_bytes\":{},\"artifact_limit_bytes\":{},",
        "\"artifact_bytes_before_summary\":{},\"output_dir\":{},\"sources\":{},\"samples\":[{}]}}\n"),
        config.layer, range.start, range.end, cur.metas.len(), states, config.cycles, config.cycles * 2,
        config.cache_bytes, config.memo_bytes, placement_bytes, placement_seconds, setup_seconds,
        base_wall, memo_wall, base_wall / memo_wall, base_cpu, memo_cpu, base_cpu / memo_cpu,
        artifact_bound, MAX_ARTIFACT_BYTES, existing_bytes, json_string(&owned.to_string_lossy()),
        source_versions, samples_json.join(","));
    anyhow::ensure!(
        summary.len() < MIB,
        "probe summary exceeds reserved output allowance"
    );
    let temporary = owned.join(".summary.json.tmp");
    let mut file = create_new_file(&temporary, 0)?;
    file.write_all(summary.as_bytes())?;
    file.sync_all()?;
    guard.check()?;
    fs::hard_link(&temporary, owned.join("summary.json"))?;
    fs::remove_file(&temporary)?;
    println!("greedy-cache-probe summary order=ABBA cycles={} samples_per_arm={} baseline_median_seconds={base_wall:.9} memo_median_seconds={memo_wall:.9} wall_speedup={:.9} baseline_median_process_cpu_seconds={base_cpu:.9} memo_median_process_cpu_seconds={memo_cpu:.9} cpu_speedup={:.9} equal_total_cache_budget=1 all_states_bit_identical_f32=1 all_policy_bytes_identical_u16=1 full_loop_speedup_measured=0 Vstar_convergence_measured=0",
        config.cycles, config.cycles * 2, base_wall / memo_wall, base_cpu / memo_cpu);
    println!("greedy-cache-probe complete status=PASS source_versions_unchanged=1 source_files_written=0 artifacts_below_256MiB=1 summary={}", owned.join("summary.json").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<String> {
        [
            "zxcl",
            "greedy-cache-probe",
            "/absent/graph",
            "/absent/pruned",
            "/absent/values",
            "/absent/work",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn bounds_are_checked_without_opening_sources() {
        let base = arguments();
        assert!(Config::parse(&base).is_ok());
        for tail in [
            vec!["9"],
            vec!["5", "3000001"],
            vec!["5", "0"],
            vec!["5", "2000000", "5"],
            vec!["5", "2000000", "2", "63"],
            vec!["5", "2000000", "2", "768", "768"],
        ] {
            let mut args = base.clone();
            args.extend(tail.into_iter().map(str::to_owned));
            assert!(Config::parse(&args).is_err());
        }
    }

    #[test]
    fn exact_comparison_preserves_signed_zero_and_rejects_nonfinite() {
        assert!(compare_values(&[-0.0, -2.0, 3.0], &[-0.0, -2.0, 3.0]).is_ok());
        assert!(compare_values(&[0.0], &[-0.0]).is_err());
        assert!(compare_values(&[f32::NAN], &[f32::NAN]).is_err());
        assert!(compare_values(&[f32::INFINITY], &[f32::INFINITY]).is_err());
        let mut expected = FullMdpLayerStats::default();
        expected.action_count = 7;
        let mut same_math = expected.clone();
        same_math.dense_cache_hits = 99;
        assert!(compare_stats(&expected, &same_math).is_ok());
        same_math.missing_next_terms = 1;
        assert!(compare_stats(&expected, &same_math).is_err());
    }
}
