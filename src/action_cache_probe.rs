//! Bounded, matched fixed-policy action-average cache measurements on L5.
//!
//! This command reads the real L6 snapshot through the production file lookup;
//! it never pins the complete next layer. Both arms have the same total cache
//! allowance, with the action table carved out of the lookup allowance.

use super::*;
use crate::action_avg_cache::{ActionSource, ValidatedPlacements};
use crate::layer_store::FieldChainLayer;
use crate::pre_reveal::PreRevealIndex;
use std::collections::BTreeMap;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;

const LAYER: usize = 5;
const DEFAULT_STATES: usize = 2_000_000;
const MAX_STATES: usize = 3_000_000;
const MIB: usize = 1 << 20;

struct ScopedEnv(&'static str, Option<std::ffi::OsString>);

impl ScopedEnv {
    fn set(name: &'static str, value: &str) -> Self {
        let old = env::var_os(name);
        env::set_var(name, value);
        Self(name, old)
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        match &self.1 {
            Some(value) => env::set_var(self.0, value),
            None => env::remove_var(self.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Version {
    canonical: PathBuf,
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

fn version(path: &Path) -> Result<Version> {
    let canonical = fs::canonicalize(path)?;
    let meta = fs::metadata(path)?;
    anyhow::ensure!(
        meta.is_file(),
        "not a regular source file: {}",
        path.display()
    );
    Ok(Version {
        canonical,
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
        mtime: (meta.mtime(), meta.mtime_nsec()),
        ctime: (meta.ctime(), meta.ctime_nsec()),
    })
}

struct Sources(Vec<(PathBuf, Version)>);

impl Sources {
    fn capture(paths: Vec<PathBuf>) -> Result<Self> {
        paths
            .into_iter()
            .map(|path| Ok((path.clone(), version(&path)?)))
            .collect::<Result<Vec<_>>>()
            .map(Self)
    }

    fn check(&self) -> Result<()> {
        for (path, saved) in &self.0 {
            anyhow::ensure!(
                version(path)? == *saved,
                "source changed during action-cache probe: {}",
                path.display()
            );
        }
        Ok(())
    }
}

fn whole_fields(layer: &FieldChainLayer, start: usize, maximum: usize) -> Result<Range<usize>> {
    anyhow::ensure!(
        (1..=MAX_STATES).contains(&maximum),
        "max_l5_states must be 1..={MAX_STATES}"
    );
    anyhow::ensure!(
        start < layer.metas.len(),
        "L5 start field index outside membership"
    );
    let start = (start..layer.metas.len())
        .find(|&i| layer.metas[i].count != 0)
        .context("no nonempty L5 field after the requested start")?;
    let mut states = 0usize;
    let mut end = start;
    for meta in &layer.metas[start..] {
        let next = states
            .checked_add(meta.count as usize)
            .context("L5 slab count overflow")?;
        if next > maximum {
            break;
        }
        states = next;
        end += 1;
    }
    anyhow::ensure!(
        states != 0,
        "first whole L5 field exceeds max_l5_states={maximum}"
    );
    Ok(start..end)
}

fn copy_span(source: &Path, destination: &Path, offset: u64, bytes: u64) -> Result<()> {
    let input = File::open(source)?;
    let mut output = BufWriter::new(File::create(destination)?);
    let mut buffer = vec![0; 256 << 10];
    let mut copied = 0;
    while copied < bytes {
        let count = usize::try_from((bytes - copied).min(buffer.len() as u64))?;
        read_exact_at_file(&input, &mut buffer[..count], offset + copied)?;
        output.write_all(&buffer[..count])?;
        copied += count as u64;
    }
    output.flush()?;
    Ok(())
}

fn copied_slab(
    original: &FieldChainLayer,
    policy: &Path,
    range: Range<usize>,
    owned: &Path,
) -> Result<(FieldChainLayer, File)> {
    let metas = &original.metas[range];
    let base = metas[0].chain_offset;
    let last = metas.last().context("empty L5 slab")?;
    let count = last.chain_offset + u64::from(last.count) - base;
    let prefix = owned.join("current");
    let mut index = BufWriter::new(File::create(prefix.with_extension("field_index.bin"))?);
    index.write_all(b"FCHNIDX1")?;
    index.write_all(&1u32.to_le_bytes())?;
    index.write_all(&u32::try_from(metas.len())?.to_le_bytes())?;
    index.write_all(&count.to_le_bytes())?;
    for meta in metas {
        index.write_all(&meta.field_id.to_le_bytes())?;
        index.write_all(&meta.count.to_le_bytes())?;
        index.write_all(&(meta.chain_offset - base).to_le_bytes())?;
    }
    index.flush()?;
    copy_span(
        &original.chains_path,
        &prefix.with_extension("chains.u24.bin"),
        base * 3,
        count * 3,
    )?;
    let policy_copy = owned.join("current.policy.u16.bin");
    copy_span(policy, &policy_copy, base * 2, count * 2)?;
    Ok((
        FieldChainLayer::open_index_only(prefix)?,
        File::open(policy_copy)?,
    ))
}

fn bit_match(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "{label}: state count mismatch"
    );
    for (i, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "{label}: nonfinite state {i}"
        );
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "{label}: state {i} differs: expected={a:?}/0x{:08x} actual={b:?}/0x{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
    Ok(())
}

fn process_cpu_seconds() -> Result<f64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes one initialized timespec through this pointer.
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
    cur: &'a FieldChainLayer,
    policy: &'a File,
    next: &'a FieldChainLayer,
    values: &'a Path,
    terminal: &'a [[f32; PIECE_COUNT]],
    placements: &'a ValidatedPlacements<'a>,
    cache_bytes: usize,
    memo_bytes: usize,
    two_line: FieldId,
}

struct Applied {
    seconds: f64,
    cpu_seconds: f64,
    stats: FullMdpLayerStats,
}

impl Kernel<'_> {
    fn apply(
        &self,
        memo: bool,
        gamma: f32,
        range: Range<usize>,
        output: &mut [f32],
    ) -> Result<Applied> {
        let cpu = process_cpu_seconds()?;
        let began = Instant::now();
        let result = compute_fixed_policy_chunk_f32_with_action_cache(
            LAYER,
            range,
            self.cur,
            self.policy,
            None,
            Some(SharedF32Out {
                ptr: output.as_mut_ptr(),
                len: output.len(),
            }),
            self.graph,
            self.qfwd,
            self.terminal,
            Some(self.next),
            Some(self.values),
            None,
            gamma,
            self.cache_bytes,
            CHECK_QSTATE_COUNT,
            65_536,
            self.two_line,
            None,
            None,
            memo.then_some(ActionSource {
                placements: self.placements,
                requested_bytes: self.memo_bytes,
            }),
        )?;
        let seconds = began.elapsed().as_secs_f64();
        Ok(Applied {
            seconds,
            cpu_seconds: process_cpu_seconds()? - cpu,
            stats: result.stats,
        })
    }
}

fn describe(applied: &Applied) -> String {
    let stats = &applied.stats;
    // File lookup materializes exactly count u24 qids and count f32 values on
    // each nonempty block miss. These are application-requested bytes, including
    // rereads; they do not claim physical disk traffic or OS-cache misses.
    let current_bytes = stats.total_states * 5;
    let next_bytes = stats.dense_materialized_states * 7;
    let actions = stats.action_avg_hits + stats.action_avg_misses;
    format!("seconds={:.9} process_cpu_seconds={:.9} states={} action_cache_hits={} action_cache_misses={} action_cache_collisions={} action_cache_saved_reveals={} action_cache_payload_bytes={} action_cache_hit_fraction={:.9} lookup_hits={} lookup_misses={} lookup_evictions={} lookup_materialized_states={} zero_or_absent_next_terms={} current_membership_policy_requested_bytes={} next_membership_values_requested_bytes={} total_logical_requested_source_bytes={} physical_disk_bytes_measured=0",
        applied.seconds, applied.cpu_seconds, stats.total_states,
        stats.action_avg_hits, stats.action_avg_misses, stats.action_avg_collisions,
        stats.action_avg_saved_reveals, stats.action_avg_cache_bytes,
        if actions == 0 { 0.0 } else { stats.action_avg_hits as f64 / actions as f64 },
        stats.dense_cache_hits, stats.dense_cache_misses, stats.dense_evictions,
        stats.dense_materialized_states, stats.missing_next_terms,
        current_bytes, next_bytes, current_bytes + next_bytes)
}

fn median(values: &mut [f64]) -> Result<f64> {
    anyhow::ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite() && *v >= 0.0),
        "invalid benchmark timing samples"
    );
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

/// Exercise signed values and deliberately absent members using the real first
/// L5 field and its stored actions. Retain at most 256 selected actions' reveal
/// children, intersected with their original L6 membership. This is a small
/// synthetic correctness fixture, separate from the actual-snapshot timing.
fn signed_fixture(kernel: &Kernel<'_>, owned: &Path) -> Result<()> {
    let qids = kernel.cur.read_block_chains(0);
    let mut codes = vec![0u8; qids.len() * 2];
    read_exact_at_file(kernel.policy, &mut codes, 0)?;
    let field = kernel.cur.metas[0].field_id;
    let mut wanted = BTreeMap::<FieldId, Vec<u32>>::new();
    let mut selected = 0;
    for (&qid, code) in qids.iter().zip(codes.chunks_exact(2)) {
        let code = u16::from_le_bytes([code[0], code[1]]);
        if code == POLICY_NONE_U16 {
            continue;
        }
        let placed = (code >> 8) as u8;
        let target = *kernel
            .graph
            .edges(field, placed)
            .get((code & 255) as usize)
            .context("invalid signed-fixture policy edge")?;
        let children = wanted.entry(target).or_default();
        for (i, rec) in kernel.qfwd.records_for(qid, placed).iter().enumerate() {
            if (qid as usize + i) % 3 != 0 {
                children.push(rec.child_qid);
            }
        }
        selected += 1;
        if selected == 256 {
            break;
        }
    }
    anyhow::ensure!(
        selected != 0,
        "signed fixture requires a nonempty selected policy"
    );
    let prefix = owned.join("signed-next");
    let mut writer = layer_store::FieldChainWriter::create(&prefix)?;
    let value_path = owned.join("signed-next.values.f32.bin");
    let mut value_file = BufWriter::new(File::create(&value_path)?);
    let mut independent = BTreeMap::<FieldId, Vec<(u32, f32)>>::new();
    let mut retained_count = 0usize;
    for (target, mut children) in wanted {
        children.sort_unstable();
        children.dedup();
        let Ok(bi) = kernel
            .next
            .metas
            .binary_search_by_key(&target, |m| m.field_id)
        else {
            continue;
        };
        let original = kernel.next.read_block_chains(bi);
        let retained = independent.entry(target).or_default();
        for qid in children {
            if original.binary_search(&qid).is_err() {
                continue;
            }
            let hash = (u64::from(target) * 0x9e37_79b9).wrapping_add(u64::from(qid) * 0x85eb_ca6b);
            let magnitude = (((hash >> 7) & 1023) as f32 + 1.0) / 128.0;
            let value = if retained_count % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            retained_count += 1;
            writer.push_pair_key((u64::from(target) << 32) | u64::from(qid))?;
            value_file.write_all(&value.to_le_bytes())?;
            retained.push((qid, value));
        }
    }
    writer.finish()?;
    value_file.flush()?;
    let next = FieldChainLayer::open_index_only(prefix)?;
    anyhow::ensure!(
        next.total_states >= 2,
        "signed fixture needs two real membership children to exercise both signs"
    );
    let synthetic = Kernel {
        next: &next,
        values: &value_path,
        ..*kernel
    };
    let mut expected = Vec::with_capacity(qids.len());
    for gamma in [1.0f32, 0.625] {
        expected.clear();
        let mut scalar_missing = 0u64;
        for (&qid, code) in qids.iter().zip(codes.chunks_exact(2)) {
            let code = u16::from_le_bytes([code[0], code[1]]);
            if code == POLICY_NONE_U16 {
                expected.push(0.0);
                continue;
            }
            let placed = (code >> 8) as u8;
            let target = kernel.graph.edges(field, placed)[(code & 255) as usize];
            let recs = kernel.qfwd.records_for(qid, placed);
            let mut sum = 0.0f32;
            for rec in recs {
                let value = independent
                    .get(&target)
                    .and_then(|entries| {
                        entries
                            .binary_search_by_key(&rec.child_qid, |(qid, _)| *qid)
                            .ok()
                            .map(|i| entries[i].1)
                    })
                    .unwrap_or(0.0);
                scalar_missing += u64::from(value == 0.0);
                sum += value;
            }
            expected.push((if gamma == 1.0 { sum } else { gamma * sum }) / recs.len() as f32);
        }
        anyhow::ensure!(
            expected.iter().any(|v| *v != 0.0),
            "signed fixture produced no nonzero outputs"
        );
        anyhow::ensure!(
            scalar_missing > 0,
            "signed fixture did not exercise missing members"
        );
        for memo in [false, true] {
            let mut actual = vec![0.0; expected.len()];
            let result = synthetic.apply(memo, gamma, 0..1, &mut actual)?;
            bit_match(&expected, &actual, "independent signed scalar reference")?;
            anyhow::ensure!(
                result.stats.missing_next_terms == scalar_missing,
                "signed fixture missing-term count differs from independent scalar reference"
            );
            println!("action-cache-probe signed_validation gamma={gamma} memo={} whole_current_fields=1 states={} retained_next_states={} synthetic_values=1 intentionally_missing_members=1 scalar_reference=independent bit_identical_f32=1 {}",
                memo as u8, expected.len(), next.total_states, describe(&result));
        }
    }
    Ok(())
}

fn execute(
    graph_path: &Path,
    pruned: &Path,
    policy: &Path,
    values: &Path,
    maximum: usize,
    repetitions: Option<usize>,
    cache_mib: usize,
    memo_mib: usize,
    start: Option<usize>,
) -> Result<()> {
    anyhow::ensure!(
        (1..=MAX_STATES).contains(&maximum),
        "max_l5_states must be 1..={MAX_STATES}"
    );
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run the action-cache probe in release mode"
    );
    anyhow::ensure!(
        (64..=1024).contains(&cache_mib),
        "cache_mib must be 64..=1024"
    );
    anyhow::ensure!(
        (1..=256).contains(&memo_mib) && memo_mib < cache_mib,
        "memo_mib must be 1..=256 and smaller than cache_mib"
    );
    if let Some(repetitions) = repetitions {
        anyhow::ensure!(
            (3..=10).contains(&repetitions),
            "repetitions must be 3..=10 for matched ABBA timing"
        );
    }
    let _cache_stats = ScopedEnv::set("FULL_MDP_CACHE_STATS", "1");
    let _action_stats = ScopedEnv::set("FULL_MDP_ACTION_STATS", "1");
    let current_prefix = q_pruned_prefix(pruned, LAYER);
    let next_prefix = q_pruned_prefix(pruned, LAYER + 1);
    let policy_path = mdp_policy_u16_path(policy, LAYER);
    let value_path = mdp_f32_value_path(values, LAYER + 1);
    let sources = Sources::capture(vec![
        graph_path.to_owned(),
        current_prefix.with_extension("field_index.bin"),
        current_prefix.with_extension("chains.u24.bin"),
        next_prefix.with_extension("field_index.bin"),
        next_prefix.with_extension("chains.u24.bin"),
        policy_path.clone(),
        value_path.clone(),
    ])?;
    let setup = Instant::now();
    let graph = FieldGraph::load_graph_bin(graph_path)?;
    let q_index = {
        let old = enumerate_states();
        build_q_index_from_old(&old)
    };
    anyhow::ensure!(q_index.states.len() == CHECK_QSTATE_COUNT);
    let qfwd = build_qforward_graph(&q_index);
    let original = FieldChainLayer::open_index_only(current_prefix)?;
    let next = FieldChainLayer::open_index_only(next_prefix)?;
    anyhow::ensure!(
        fs::metadata(&policy_path)?.len() == original.total_states * 2,
        "L5 policy byte count differs from membership"
    );
    anyhow::ensure!(
        fs::metadata(&value_path)?.len() == next.total_states * 4,
        "L6 value byte count differs from membership"
    );
    let range = whole_fields(
        &original,
        start.unwrap_or(original.metas.len() / 2),
        maximum,
    )?;
    let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
    let (cur, policy_file) = copied_slab(&original, &policy_path, range.clone(), &scratch.path)?;
    let pre_build = Instant::now();
    let pre = PreRevealIndex::build(&q_index)?;
    let placements = ValidatedPlacements::build(&q_index, &qfwd, &pre)?;
    let placement_build_seconds = pre_build.elapsed().as_secs_f64();
    let terminal = vec![[0.0; PIECE_COUNT]; CHECK_QSTATE_COUNT];
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("two-line field missing")?;
    sources.check()?;
    println!("action-cache-probe scope=contiguous_whole_L5_fields full_loop1=0 kernel=production_fixed_policy execution=sequential next_source=file_lookup full_next_layer_pinned=0 original_block_start={} original_block_end={} fields={} states={} next_snapshot_bytes={} total_cache_bytes={} requested_memo_bytes={} placement_build_seconds={placement_build_seconds:.9} setup_seconds={:.9} setup_excluded_from_apply_timing=1 original_qforward_order=1",
        range.start, range.end, cur.metas.len(), cur.total_states, next.total_states * 4,
        cache_mib * MIB, memo_mib * MIB, setup.elapsed().as_secs_f64());
    let kernel = Kernel {
        graph: &graph,
        qfwd: &qfwd,
        cur: &cur,
        policy: &policy_file,
        next: &next,
        values: &value_path,
        terminal: &terminal,
        placements: &placements,
        cache_bytes: cache_mib * MIB,
        memo_bytes: memo_mib * MIB,
        two_line,
    };
    let mut baseline = vec![0.0; usize::try_from(cur.total_states)?];
    let mut cached = vec![0.0; baseline.len()];
    let mut missing_terms = None;
    for memo in [false, true] {
        sources.check()?;
        let result = kernel.apply(
            memo,
            1.0,
            0..cur.metas.len(),
            if memo { &mut cached } else { &mut baseline },
        )?;
        anyhow::ensure!(result.stats.total_states == cur.total_states);
        if memo {
            bit_match(&baseline, &cached, "warmup actual L6 values")?;
            anyhow::ensure!(
                missing_terms == Some(result.stats.missing_next_terms),
                "warmup missing-term count differs"
            );
        } else {
            missing_terms = Some(result.stats.missing_next_terms);
        }
        println!(
            "action-cache-probe warmup memo={} {}",
            memo as u8,
            describe(&result)
        );
    }
    let mut base_wall = Vec::new();
    let mut memo_wall = Vec::new();
    let mut base_cpu = Vec::new();
    let mut memo_cpu = Vec::new();
    for repetition in 0..repetitions.unwrap_or(0) {
        for memo in [false, true, true, false] {
            sources.check()?;
            let result = kernel.apply(
                memo,
                1.0,
                0..cur.metas.len(),
                if memo { &mut cached } else { &mut baseline },
            )?;
            anyhow::ensure!(result.stats.total_states == cur.total_states);
            bit_match(&baseline, &cached, "measured actual L6 values")?;
            anyhow::ensure!(
                missing_terms == Some(result.stats.missing_next_terms),
                "measured missing-term count differs"
            );
            if memo {
                memo_wall.push(result.seconds);
                memo_cpu.push(result.cpu_seconds);
            } else {
                base_wall.push(result.seconds);
                base_cpu.push(result.cpu_seconds);
            }
            println!("action-cache-probe measured repetition={repetition} memo={} bit_identical_f32=1 {}", memo as u8, describe(&result));
        }
    }
    if !base_wall.is_empty() {
        let base = median(&mut base_wall)?;
        let memo = median(&mut memo_wall)?;
        let base_cpu = median(&mut base_cpu)?;
        let memo_cpu = median(&mut memo_cpu)?;
        println!("action-cache-probe summary repetitions={} order=ABBA samples_per_arm={} baseline_median_seconds={base:.9} memo_median_seconds={memo:.9} wall_speedup={:.9} baseline_median_process_cpu_seconds={base_cpu:.9} memo_median_process_cpu_seconds={memo_cpu:.9} cpu_speedup={:.9} equal_total_cache_budget=1 current_group_shape_identical=1 all_states_bit_identical_f32=1 setup_amortization_measured=0 full_loop1_speedup_measured=0",
            repetitions.unwrap(), base_wall.len(), base / memo, base_cpu / memo_cpu);
    }
    signed_fixture(&kernel, &scratch.path)?;
    sources.check()?;
    println!("action-cache-probe complete actual_snapshot_bit_identical_f32=1 independent_signed_scalar_bit_identical_f32=1 source_versions_unchanged=1 source_files_written=0 default_solver_settings_changed=0");
    Ok(())
}

pub(super) fn run(args: &[String]) -> Result<()> {
    anyhow::ensure!((6..=11).contains(&args.len()),
        "usage: {} action-cache-probe <graph> <pruned> <policy> <values> [max_l5_states=2000000] [repetitions=3] [cache_mib=768] [memo_mib=16] [start_field_index=midpoint]", args[0]);
    let number = |at: usize, default: usize| -> Result<usize> {
        args.get(at)
            .map(|s| s.parse::<usize>())
            .transpose()
            .with_context(|| format!("invalid argument {at}"))
            .map(|value| value.unwrap_or(default))
    };
    let start = args
        .get(10)
        .map(|s| s.parse::<usize>())
        .transpose()
        .context("invalid start_field_index")?;
    execute(
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        Path::new(&args[5]),
        number(6, DEFAULT_STATES)?,
        Some(number(7, 3)?),
        number(8, 768)?,
        number(9, 16)?,
        start,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_preserves_signed_zero_and_rejects_nonfinite() {
        assert!(bit_match(&[-0.0, -2.0], &[-0.0, -2.0], "same").is_ok());
        assert!(bit_match(&[0.0], &[-0.0], "zero").is_err());
        assert!(bit_match(&[f32::NAN], &[f32::NAN], "nan").is_err());
        assert!(bit_match(&[f32::INFINITY], &[f32::INFINITY], "infinite").is_err());
    }

    #[test]
    fn median_requires_finite_nonnegative_samples() {
        assert_eq!(median(&mut [5.0, 1.0, 3.0]).unwrap(), 3.0);
        assert_eq!(median(&mut [8.0, 2.0, 6.0, 4.0]).unwrap(), 5.0);
        assert!(median(&mut []).is_err());
        assert!(median(&mut [f64::NAN]).is_err());
        assert!(median(&mut [-1.0]).is_err());
    }

    #[test]
    #[ignore = "requires explicit ZXCL_VALIDATION_GRAPH, PRUNED, POLICY, VALUES; release mode"]
    fn bounded_real_l5_action_cache_matches_original_and_signed_scalar() -> Result<()> {
        let _lock = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let input = |name: &str| {
            env::var_os(name)
                .map(PathBuf::from)
                .with_context(|| format!("explicit action-cache validation requires {name}"))
        };
        execute(
            &input("ZXCL_VALIDATION_GRAPH")?,
            &input("ZXCL_VALIDATION_PRUNED")?,
            &input("ZXCL_VALIDATION_POLICY")?,
            &input("ZXCL_VALIDATION_VALUES")?,
            DEFAULT_STATES,
            None,
            768,
            16,
            None,
        )
    }
}
