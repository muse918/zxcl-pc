//! Exact gamma=1 fixed-policy experiment over complete L9 and a contiguous
//! whole-field L8 slab. This is a two-layer measurement, not a boundary solve.

use super::*;
use crate::layer_store::FieldChainLayer;
use crate::loop1_stream::{CompiledL8, CompiledL9, Loop1Limits, PreparedL9Values};
use crate::loop1_terminal::TerminalPlan;
use crate::pre_reveal::PreRevealIndex;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;

const DEFAULT_STATES: usize = 2_000_000;
const MAX_STATES: usize = 4_000_000;
const CACHE_BYTES: usize = 32 << 20;

fn mib_setting(name: &str, default: u64) -> Result<u64> {
    let mib = match env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .with_context(|| format!("invalid {name}"))?,
        Err(env::VarError::NotPresent) => default,
        Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
    };
    mib.checked_mul(1 << 20)
        .with_context(|| format!("{name} byte size overflow"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileVersion {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

fn file_version(path: &Path) -> Result<FileVersion> {
    let m = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    anyhow::ensure!(
        m.is_file(),
        "source is not a regular file: {}",
        path.display()
    );
    Ok(FileVersion {
        dev: m.dev(),
        ino: m.ino(),
        len: m.len(),
        mtime: (m.mtime(), m.mtime_nsec()),
        ctime: (m.ctime(), m.ctime_nsec()),
    })
}

struct SourceVersions(Vec<(PathBuf, FileVersion)>);

impl SourceVersions {
    fn capture(paths: Vec<PathBuf>) -> Result<Self> {
        paths
            .into_iter()
            .map(|p| Ok((p.clone(), file_version(&p)?)))
            .collect::<Result<Vec<_>>>()
            .map(Self)
    }

    fn check(&self) -> Result<()> {
        for (path, version) in &self.0 {
            anyhow::ensure!(
                file_version(path)? == *version,
                "source changed after loop1 compilation: {}; rebuild the plan",
                path.display()
            );
        }
        Ok(())
    }
}

/// The cap never splits a field. Start at the first nonempty original block;
/// the returned range contains every intervening field, including empty ones.
fn whole_field_slab(layer: &FieldChainLayer, maximum: usize) -> Result<Range<usize>> {
    anyhow::ensure!(
        (1..=MAX_STATES).contains(&maximum),
        "max_l8_states must be 1..={MAX_STATES}"
    );
    let start = layer
        .metas
        .iter()
        .position(|m| m.count != 0)
        .context("empty L8 membership")?;
    anyhow::ensure!(layer.metas[start].count as usize <= maximum,
        "first complete L8 field has {} states, exceeding max_l8_states={maximum}; increase the cap (maximum {MAX_STATES})",
        layer.metas[start].count);
    let mut count = 0usize;
    let mut end = start;
    for meta in &layer.metas[start..] {
        let next = count
            .checked_add(meta.count as usize)
            .context("L8 slab count overflow")?;
        if next > maximum {
            break;
        }
        count = next;
        end += 1;
    }
    anyhow::ensure!(count > 0 && count <= maximum, "invalid bounded L8 slab");
    Ok(start..end)
}

fn copy_span(source: &Path, dest: &Path, offset: u64, bytes: u64) -> Result<()> {
    let input = File::open(source)?;
    let mut output = BufWriter::new(File::create(dest)?);
    let mut scratch = vec![0u8; 256 << 10];
    let mut done = 0;
    while done < bytes {
        let n = ((bytes - done) as usize).min(scratch.len());
        read_exact_at_file(&input, &mut scratch[..n], offset + done)?;
        output.write_all(&scratch[..n])?;
        done += n as u64;
    }
    output.flush()?;
    Ok(())
}

/// Rebase only this invocation's copied slab, so the unchanged production
/// output kernel can write a small zero-based RAM buffer.
fn slab_fixture(
    original: &FieldChainLayer,
    policy: &Path,
    range: Range<usize>,
    owned: &Path,
) -> Result<(FieldChainLayer, File)> {
    let metas = &original.metas[range];
    let base = metas[0].chain_offset;
    let last = metas.last().context("empty slab")?;
    let states = last.chain_offset + last.count as u64 - base;
    let prefix = owned.join("loop1-l8");
    let mut index = BufWriter::new(File::create(prefix.with_extension("field_index.bin"))?);
    index.write_all(b"FCHNIDX1")?;
    index.write_all(&1u32.to_le_bytes())?;
    index.write_all(&(metas.len() as u32).to_le_bytes())?;
    index.write_all(&states.to_le_bytes())?;
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
        states * 3,
    )?;
    let policy_copy = owned.join("loop1-l8.policy.u16.bin");
    copy_span(policy, &policy_copy, base * 2, states * 2)?;
    Ok((
        FieldChainLayer::open_index_only(prefix)?,
        File::open(policy_copy)?,
    ))
}

fn reset_vectors(layer: &FieldChainLayer, input: &str) -> Result<Vec<(&'static str, Vec<f32>)>> {
    let qids = layer0_reset_block_qids(layer)?;
    anyhow::ensure!(
        qids.len() == CHECK_QSTATE_COUNT && layer.total_states == CHECK_QSTATE_COUNT as u64,
        "loop1 probe requires one complete {CHECK_QSTATE_COUNT}-coordinate L0 reset block"
    );
    let raw = load_boundary_vector(input, layer.total_states)?;
    let mut dense = vec![0.0f32; CHECK_QSTATE_COUNT];
    let mut seen = vec![false; CHECK_QSTATE_COUNT];
    for (&qid, &value) in qids.iter().zip(&raw) {
        anyhow::ensure!(
            (qid as usize) < dense.len() && !seen[qid as usize],
            "invalid/duplicate reset qid {qid}"
        );
        anyhow::ensure!(value.is_finite(), "nonfinite L0 reset input at qid {qid}");
        dense[qid as usize] = value;
        seen[qid as usize] = true;
    }
    anyhow::ensure!(seen.iter().all(|v| *v), "incomplete reset qid membership");
    fn signed(seed: u64, scale: f32) -> Vec<f32> {
        (0..CHECK_QSTATE_COUNT)
            .map(|qid| {
                let mut x = (qid as u64).wrapping_add(seed);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                x ^= x >> 31;
                let value = ((x >> 16) & 0xffff) as i32 - 32768;
                (value as f32 / 32768.0) * scale
            })
            .collect()
    }
    Ok(vec![
        (if input == "zero" { "zero" } else { "actual_l0" }, dense),
        ("independent_signed", signed(0x17b4_d393_a987_22ab, 2.0)),
        (
            "independent_signed_krylov",
            signed(
                0x8973_cede_44b5_2231,
                1.0 / (CHECK_QSTATE_COUNT as f32).sqrt(),
            ),
        ),
    ])
}

fn bit_match(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "{label}: output length mismatch"
    );
    for (i, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "{label}: nonfinite value at state {i}"
        );
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "{label}: f32 bits differ at state {i}: legacy={a:?}/0x{:08x} compiled={b:?}/0x{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
    Ok(())
}

struct Legacy<'a> {
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    l9: &'a FieldChainLayer,
    l9_policy: &'a File,
    slab: &'a FieldChainLayer,
    slab_policy: &'a File,
    two_line: u32,
}

struct Applied {
    output: Vec<f32>,
    full_seconds: f64,
    leaf_seconds: f64,
}

impl Legacy<'_> {
    fn source<'a>(&'a self, table: &'a [[f32; 7]]) -> terminal_l9::Source<'a> {
        terminal_l9::Source {
            layer: self.l9,
            graph: self.graph,
            qfwd: self.qfwd,
            terminal_reset_avg: table,
            mode: terminal_l9::Mode::Fixed(self.l9_policy),
        }
    }

    fn l9_kernel(&self, table: &[[f32; 7]], mut output: Option<&mut [f32]>) -> Result<()> {
        let sink = output.as_mut().map(|out| SharedF32Out {
            ptr: out.as_mut_ptr(),
            len: out.len(),
        });
        let mut states = 0;
        // Contiguous production-kernel groups bound transient buffers without
        // adding a syscall per field to the baseline.
        let mut start = 0;
        while start < self.l9.metas.len() {
            let mut end = start + 1;
            let mut count = self.l9.metas[start].count as usize;
            while end < self.l9.metas.len()
                && count + self.l9.metas[end].count as usize <= DEFAULT_STATES
            {
                count += self.l9.metas[end].count as usize;
                end += 1;
            }
            states += compute_fixed_policy_chunk_f32_with_l9_source(
                9,
                start..end,
                self.l9,
                self.l9_policy,
                None,
                sink,
                self.graph,
                self.qfwd,
                table,
                None,
                None,
                None,
                1.0,
                CACHE_BYTES,
                CHECK_QSTATE_COUNT,
                50_000,
                self.two_line,
                None,
            )?
            .stats
            .total_states;
            start = end;
        }
        anyhow::ensure!(
            states == self.l9.total_states,
            "legacy L9 scan was incomplete"
        );
        Ok(())
    }

    fn l8_kernel(&self, table: &[[f32; 7]], values: Option<&[f32]>) -> Result<Vec<f32>> {
        self.l8_kernel_with_compiled(table, values, None)
    }

    fn l8_kernel_with_compiled(
        &self,
        table: &[[f32; 7]],
        values: Option<&[f32]>,
        compiled: Option<PreparedL9Values<'_>>,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.slab.total_states as usize];
        let sink = SharedF32Out {
            ptr: out.as_mut_ptr(),
            len: out.len(),
        };
        let direct = if values.is_none() && compiled.is_none() {
            Some(self.source(table))
        } else {
            None
        };
        let result = compute_fixed_policy_chunk_f32_with_sources(
            8,
            0..self.slab.metas.len(),
            self.slab,
            self.slab_policy,
            None,
            Some(sink),
            self.graph,
            self.qfwd,
            table,
            Some(self.l9),
            None,
            values,
            1.0,
            CACHE_BYTES,
            CHECK_QSTATE_COUNT,
            50_000,
            self.two_line,
            direct,
            compiled,
        )?;
        anyhow::ensure!(
            result.stats.total_states == self.slab.total_states,
            "legacy L8 scan was incomplete"
        );
        Ok(out)
    }

    fn apply(&self, reset: &[f32], reward: bool, materialized: bool) -> Result<Applied> {
        let full = Instant::now();
        let table =
            build_terminal_reset_avg_table_opt(self.qfwd, reset, 1.0, CHECK_QSTATE_COUNT, reward);
        let output;
        let leaf_seconds;
        if materialized {
            let mut next = vec![0.0; self.l9.total_states as usize];
            self.l9_kernel(&table, Some(&mut next))?;
            let leaf = Instant::now();
            output = self.l8_kernel(&table, Some(&next))?;
            leaf_seconds = leaf.elapsed().as_secs_f64();
        } else {
            // Matches FULL_MDP_L9_DIRECT's current per-application validation
            // and complete sinkless fixed-policy L9 traversal.
            terminal_l9::validate_membership(self.source(&table))?;
            self.l9_kernel(&table, None)?;
            let leaf = Instant::now();
            output = self.l8_kernel(&table, None)?;
            leaf_seconds = leaf.elapsed().as_secs_f64();
        }
        Ok(Applied {
            output,
            full_seconds: full.elapsed().as_secs_f64(),
            leaf_seconds,
        })
    }
}

fn compiled_apply(
    compiled: &CompiledL8<'_>,
    l9: &CompiledL9<'_>,
    plan: &TerminalPlan,
    sources: &SourceVersions,
    reset: &[f32],
    reward: bool,
) -> Result<Applied> {
    let full = Instant::now();
    sources.check()?;
    l9.check_inputs_unchanged()?;
    compiled.check_inputs_unchanged()?;
    let terminal = plan.evaluate_selected_f32(reset, reward, compiled.used_terminal_zids())?;
    let leaf = Instant::now();
    let output = compiled.evaluate_f32(&terminal)?;
    Ok(Applied {
        output,
        full_seconds: full.elapsed().as_secs_f64(),
        leaf_seconds: leaf.elapsed().as_secs_f64(),
    })
}

/// Production L9-only route: retain the original L8 policy/kernel and gather
/// the full terminal table needed by the unchanged lower-layer kernels.
fn production_apply(
    legacy: &Legacy<'_>,
    l9: &CompiledL9<'_>,
    plan: &TerminalPlan,
    pre: &PreRevealIndex,
    sources: &SourceVersions,
    reset: &[f32],
    reward: bool,
) -> Result<Applied> {
    production_apply_impl(legacy, l9, plan, pre, sources, reset, reward, true)
}

fn production_apply_impl(
    legacy: &Legacy<'_>,
    l9: &CompiledL9<'_>,
    plan: &TerminalPlan,
    pre: &PreRevealIndex,
    sources: &SourceVersions,
    reset: &[f32],
    reward: bool,
    materialize: bool,
) -> Result<Applied> {
    let full = Instant::now();
    sources.check()?;
    l9.check_inputs_unchanged()?;
    let terminal = plan.evaluate_f32(reset, reward)?;
    let table = crate::loop1_prepared::terminal_table(pre, &terminal)?;
    let source = l9.prepare_f32(&terminal)?;
    // The production sweep validates the full gathered table before invoking
    // any layer kernel, even though its values came from a checked plan.
    anyhow::ensure!(
        table.len() == CHECK_QSTATE_COUNT,
        "production terminal table shape mismatch"
    );
    anyhow::ensure!(
        table.iter().flatten().all(|v| v.is_finite()),
        "nonfinite production terminal table"
    );
    let next = if materialize {
        Some(l9.evaluate_f32(&terminal)?)
    } else {
        None
    };
    let leaf = Instant::now();
    let output = if let Some(values) = next.as_deref() {
        legacy.l8_kernel(&table, Some(values))?
    } else {
        legacy.l8_kernel_with_compiled(&table, None, Some(source))?
    };
    let leaf_seconds = leaf.elapsed().as_secs_f64();
    drop(next);
    Ok(Applied {
        output,
        full_seconds: full.elapsed().as_secs_f64(),
        leaf_seconds,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn peak_rss_bytes() -> Option<u64> {
    // Process lifetime high-water mark, including graph construction and the
    // validation baseline. It is deliberately not labelled an isolated peak.
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
}

fn execute(
    graph_path: &Path,
    pruned: &Path,
    policy: &Path,
    input: &str,
    max_states: usize,
    repetitions: Option<usize>,
) -> Result<()> {
    let setup = Instant::now();
    // Capture before loading: a version observed only after reading would not
    // bind the in-memory graph and metadata to the captured input generation.
    let policy8 = mdp_policy_u16_path(policy, 8);
    let policy9 = mdp_policy_u16_path(policy, 9);
    let mut paths = vec![graph_path.to_path_buf(), policy8.clone(), policy9.clone()];
    for layer in [0, 8, 9] {
        let prefix = q_pruned_prefix(pruned, layer);
        paths.push(prefix.with_extension("field_index.bin"));
        paths.push(prefix.with_extension("chains.u24.bin"));
    }
    if input != "zero" {
        paths.push(PathBuf::from(input));
    }
    let sources = SourceVersions::capture(paths)?;
    let graph = FieldGraph::load_graph_bin(graph_path)?;
    let q_index = build_q_index_from_old(&enumerate_states());
    anyhow::ensure!(
        q_index.states.len() == CHECK_QSTATE_COUNT,
        "unexpected quotient count"
    );
    let pre_build = Instant::now();
    let pre = PreRevealIndex::build(&q_index)?;
    let pre_index_build_seconds = pre_build.elapsed().as_secs_f64();
    let qfwd = build_qforward_graph(&q_index);
    let l0 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 0))?;
    let l8 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 8))?;
    let l9 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 9))?;
    anyhow::ensure!(l9.total_states > 0, "empty L9 membership");
    for (path, states) in [(&policy8, l8.total_states), (&policy9, l9.total_states)] {
        anyhow::ensure!(
            fs::metadata(path)?.len() == states.checked_mul(2).context("policy size overflow")?,
            "policy size mismatch {}",
            path.display()
        );
    }
    let range = whole_field_slab(&l8, max_states)?;
    let vectors = reset_vectors(&l0, input)?;
    let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
    let (slab, slab_policy) = slab_fixture(&l8, &policy8, range.clone(), &scratch.path)?;
    sources.check()?;
    let prep_seconds = setup.elapsed().as_secs_f64() - pre_index_build_seconds;
    let build = Instant::now();
    let plan_build = Instant::now();
    let plan = TerminalPlan::build(&q_index, &qfwd, &pre)?;
    let terminal_plan_build_seconds = plan_build.elapsed().as_secs_f64();
    let limits = Loop1Limits {
        max_mapping_bytes: mib_setting("PI_LOOP1_L9_MAX_MAPPING_MIB", 512)?,
        max_cache_bytes: usize::try_from(mib_setting("PI_LOOP1_L9_BUILD_CACHE_MIB", 32)?)?,
        ..Loop1Limits::default()
    };
    let rank_cap_bytes = mib_setting("PI_LOOP1_L9_RANK_MAX_MIB", 64)?;
    let mut compiled9 = CompiledL9::build(&graph, &q_index, &qfwd, &pre, &l9, &policy9, limits)?;
    // Match the production PreparedL9 setup. Rank construction and its source
    // reads belong to both the L9-only and complete experimental build costs.
    compiled9.build_rank_index(rank_cap_bytes, limits.max_cache_bytes)?;
    sources.check()?;
    let l9_only_b = pre_index_build_seconds + build.elapsed().as_secs_f64();
    let compiled8 = CompiledL8::build(
        &graph,
        &q_index,
        &qfwd,
        &pre,
        &l8,
        &policy8,
        range.clone(),
        &compiled9,
        limits,
    )?;
    sources.check()?;
    let b = pre_index_build_seconds + build.elapsed().as_secs_f64();
    let stats9 = compiled9.stats();
    let stats8 = compiled8.stats();
    anyhow::ensure!(
        stats9.scanned_states == l9.total_states && stats8.scanned_states == slab.total_states,
        "compiler did not cover the complete requested slab"
    );
    let exprs = 1usize + compiled9.used_terminal_zids().len() + stats8.unique_averages;
    let state_count = l9.total_states + slab.total_states;
    let table_bytes = stats9.table_bytes + stats8.table_bytes;
    let source_bytes = stats9.index_bytes_read
        + stats8.index_bytes_read
        + stats9.membership_bytes_read
        + stats9.policy_bytes_read
        + stats8.membership_bytes_read
        + stats8.policy_bytes_read;
    let retained_estimate = table_bytes.saturating_add(plan.storage_bytes() as u64);
    let construction_estimate = retained_estimate
        .saturating_add(
            stats9
                .cache_peak_bytes
                .max(stats8.cache_peak_bytes)
                .max(stats9.rank_build_peak_bytes) as u64,
        )
        .saturating_add(
            stats9
                .interner_peak_logical_bytes
                .max(stats8.interner_peak_logical_bytes),
        );
    let graph_payload = (graph.hashes.len() * std::mem::size_of::<u64>()
        + graph.offsets.len() * std::mem::size_of::<[u32; 8]>()
        + graph.targets.len() * std::mem::size_of::<u32>()) as u64;
    let quotient_payload = (q_index.states.len() * std::mem::size_of::<qstate::QChainState>()
        + q_index.old_to_qid.len() * std::mem::size_of::<u32>()
        + q_index.key_to_id.len() * std::mem::size_of::<(u32, usize)>())
        as u64;
    let qforward_payload = (qfwd.offsets.len() * std::mem::size_of::<[u32; 8]>()
        + qfwd.records.len() * std::mem::size_of::<qgraph::QForwardRec>())
        as u64;
    // PreRevealIndex retains states, qid_to_zid, reveal offsets/children,
    // placement pairs and one available-pair byte per quotient state.
    let pre_payload = (pre.len() * std::mem::size_of::<crate::pre_reveal::PreRevealState>()
        + (pre.len() + 1) * 4
        + CHECK_QSTATE_COUNT * 17) as u64;
    let metadata_payload = ((l0.metas.len() + l8.metas.len() + l9.metas.len() + slab.metas.len())
        * std::mem::size_of::<crate::layer_store::FieldChainBlockMeta>())
        as u64;
    let common_payload = graph_payload
        + quotient_payload
        + qforward_payload
        + pre_payload
        + metadata_payload
        + (vectors.len() * CHECK_QSTATE_COUNT * 4) as u64;
    // Validation holds both complete L9 outputs at once. Include the full
    // legacy terminal table and bounded kernel scratch/cache in this estimate.
    let validation_work = l9.total_states * 8
        + CHECK_QSTATE_COUNT as u64 * 28
        + pre.len() as u64 * 4
        + slab.total_states * 8
        + CACHE_BYTES as u64
        + DEFAULT_STATES as u64 * 9;
    let peak_logical_estimate = common_payload
        + retained_estimate
        + validation_work.max(construction_estimate - retained_estimate);
    println!("loop1-probe scope=complete_L9_plus_contiguous_L8_slab gamma=1 fixed_policy=1 full_boundary_solve=0 execution=sequential_existing_kernels max_l8_states={max_states} l8_block_start={} l8_block_end={} l8_fields={} l8_states={} l8_state_offset={} l9_states={} prep_seconds={prep_seconds:.9} prep_excludes_pre_index=1 pre_index_build_seconds={pre_index_build_seconds:.9} terminal_plan_build_seconds={terminal_plan_build_seconds:.9} B_seconds={b:.9} B_L9_only_seconds={l9_only_b:.9} B_includes_pre_index_plan_L9_L8=1 B_L9_only_includes_pre_index_plan_L9=1",
        range.start, range.end, range.len(), slab.total_states, l8.metas[range.start].chain_offset, l9.total_states);
    println!("loop1-probe build_source_bytes_scanned={source_bytes} build_membership_bytes={} build_policy_bytes={} build_index_bytes={} table_bytes={table_bytes} terminal_plan_bytes={} terminal_plan_edges={} unique_expressions={exprs} unique_l8_averages={} unique_terminal_expressions={} retained_expression_edges={} states_per_unique_expression={:.6} compiled_payload_bytes_per_input_state={:.6} retained_payload_estimate_bytes={retained_estimate} construction_payload_plus_cache_hash_estimate_bytes={construction_estimate} allocator_overhead_in_estimate=0 graph_qstate_qforward_pre_index_in_estimate=0",
        stats9.membership_bytes_read + stats8.membership_bytes_read,
        stats9.policy_bytes_read + stats8.policy_bytes_read,
        stats9.index_bytes_read + stats8.index_bytes_read,
        plan.storage_bytes(), plan.child_count(), stats8.unique_averages,
        compiled9.used_terminal_zids().len(), stats8.retained_edges,
        state_count as f64 / exprs as f64, retained_estimate as f64 / state_count as f64);
    println!("loop1-probe L9_membership_rank_bytes={} rank_construction_seconds={:.9} rank_retained_cap_bytes={rank_cap_bytes} rank_build_buffer_cap_bytes={} rank_build_peak_bytes={} L9_table_bytes_including_rank={} rank_build_in_B_and_source_bytes=1 P_lookup=materialized_f32_RAM R_lookup=persistent_qid_rank R_timing=validation_only",
        stats9.membership_rank_bytes, stats9.rank_construction_seconds, limits.max_cache_bytes, stats9.rank_build_peak_bytes, stats9.table_bytes);
    println!("loop1-probe graph_payload_bytes={graph_payload} quotient_payload_bytes={quotient_payload} qforward_payload_bytes={qforward_payload} pre_index_payload_estimate_bytes={pre_payload} terminal_validation_qforward_rows={} terminal_validation_qforward_records={} peak_resident_logical_estimate_bytes={peak_logical_estimate} peak_estimate_includes_full_L9_validation_outputs=1 allocator_and_hash_capacity_overhead_in_peak_estimate=0 source_graph_parse_peak_in_estimate=0", qfwd.offsets.len(), qfwd.records.len());
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("missing two-line field")?;
    let l9_policy = File::open(&policy9)?;
    let legacy = Legacy {
        graph: &graph,
        qfwd: &qfwd,
        l9: &l9,
        l9_policy: &l9_policy,
        slab: &slab,
        slab_policy: &slab_policy,
        two_line,
    };
    let materialized_timing = env::var("ZXCL_LOOP1_MATERIALIZED").is_ok_and(|v| v == "1");
    for (name, reset) in &vectors {
        for reward in [false, true] {
            sources.check()?;
            let table =
                build_terminal_reset_avg_table_opt(&qfwd, reset, 1.0, CHECK_QSTATE_COUNT, reward);
            let terminal = plan.evaluate_f32(reset, reward)?;
            // L8's compiled token does not consume this table. Check all of
            // its bits directly because production lower layers still do.
            let gathered = crate::loop1_prepared::terminal_table(&pre, &terminal)?;
            anyhow::ensure!(
                table.len() == gathered.len(),
                "gathered terminal table length mismatch"
            );
            for (legacy_row, gathered_row) in table.iter().zip(&gathered) {
                bit_match(
                    legacy_row,
                    gathered_row,
                    "production gathered terminal table",
                )?;
            }
            drop(gathered);
            let mut reference9 = vec![0.0; l9.total_states as usize];
            legacy.l9_kernel(&table, Some(&mut reference9))?;
            let actual9 = compiled9.evaluate_f32(&terminal)?;
            bit_match(&reference9, &actual9, "complete L9")?;
            drop(actual9);
            let expected = legacy.l8_kernel(&table, Some(&reference9))?;
            drop(reference9);
            let actual = compiled8.evaluate_f32(&terminal)?;
            bit_match(&expected, &actual, "complete L8 slab")?;
            let selected = compiled_apply(&compiled8, &compiled9, &plan, &sources, reset, reward)?;
            bit_match(
                &expected,
                &selected.output,
                "selected-terminal L8 application",
            )?;
            let production =
                production_apply(&legacy, &compiled9, &plan, &pre, &sources, reset, reward)?;
            bit_match(
                &expected,
                &production.output,
                "production compiled-L9 L8 application",
            )?;
            let rank_direct = production_apply_impl(
                &legacy, &compiled9, &plan, &pre, &sources, reset, reward, false,
            )?;
            bit_match(
                &expected,
                &rank_direct.output,
                "rank-direct compiled-L9 L8 application",
            )?;
            let direct = legacy.l8_kernel(&table, None)?;
            bit_match(&expected, &direct, "materialized versus direct L8 slab")?;
            drop((
                table,
                terminal,
                actual,
                selected,
                production,
                rank_direct,
                direct,
            ));
            println!("loop1-probe validation vector={name} include_reward={} l9_states={} l8_states={} bit_identical_f32=1 production_L9_gather_bit_identical_f32=1 rank_direct_L9_bit_identical_f32=1 finite=1", reward as u8, l9.total_states, slab.total_states);
            let Some(reps) = repetitions else {
                continue;
            };
            // Warm both complete application paths before the ABBA samples.
            let warm_e = compiled_apply(&compiled8, &compiled9, &plan, &sources, reset, reward)?;
            bit_match(&expected, &warm_e.output, "compiled warmup")?;
            let warm_s = legacy.apply(reset, reward, false)?;
            bit_match(&expected, &warm_s.output, "direct warmup")?;
            let warm_p =
                production_apply(&legacy, &compiled9, &plan, &pre, &sources, reset, reward)?;
            bit_match(&expected, &warm_p.output, "production L9-only warmup")?;
            let warm_m = legacy.apply(reset, reward, true)?;
            bit_match(&expected, &warm_m.output, "materialized warmup")?;
            drop((warm_e, warm_s, warm_p, warm_m));
            let mut e_times = Vec::new();
            let mut ek_times = Vec::new();
            let mut me_times = Vec::new();
            let mut mek_times = Vec::new();
            let mut s_times = Vec::new();
            let mut sk_times = Vec::new();
            let mut m_times = Vec::new();
            let mut mk_times = Vec::new();
            let mut p_times = Vec::new();
            let mut pk_times = Vec::new();
            let mut ps_times = Vec::new();
            let mut psk_times = Vec::new();
            for _ in 0..reps {
                // A=compiled E, B=existing S, always A B B A. Repeat the
                // same paired order separately for the optional RAM baseline.
                for materialized in [false, true] {
                    if materialized && !materialized_timing {
                        continue;
                    }
                    for is_e in [true, false, false, true] {
                        let measured = if is_e {
                            compiled_apply(&compiled8, &compiled9, &plan, &sources, reset, reward)?
                        } else {
                            sources.check()?;
                            legacy.apply(reset, reward, materialized)?
                        };
                        bit_match(&expected, &measured.output, "ABBA sample")?;
                        if is_e && materialized {
                            me_times.push(measured.full_seconds);
                            mek_times.push(measured.leaf_seconds);
                        } else if is_e {
                            e_times.push(measured.full_seconds);
                            ek_times.push(measured.leaf_seconds);
                        } else if materialized {
                            m_times.push(measured.full_seconds);
                            mk_times.push(measured.leaf_seconds);
                        } else {
                            s_times.push(measured.full_seconds);
                            sk_times.push(measured.leaf_seconds);
                        }
                    }
                }
                // P keeps the production L8 kernel. Pair it independently
                // with the same materialized S kernel in P S S P order.
                for is_p in [true, false, false, true] {
                    let measured = if is_p {
                        production_apply(&legacy, &compiled9, &plan, &pre, &sources, reset, reward)?
                    } else {
                        sources.check()?;
                        legacy.apply(reset, reward, true)?
                    };
                    bit_match(&expected, &measured.output, "production P/S ABBA sample")?;
                    if is_p {
                        p_times.push(measured.full_seconds);
                        pk_times.push(measured.leaf_seconds);
                    } else {
                        ps_times.push(measured.full_seconds);
                        psk_times.push(measured.leaf_seconds);
                    }
                }
            }
            let e = median(&mut e_times);
            let ek = median(&mut ek_times);
            let s = median(&mut s_times);
            let sk = median(&mut sk_times);
            println!("loop1-probe timing vector={name} include_reward={} repetitions={reps} order=ABBA E_direct_context_samples={} S_direct_samples={} E_seconds={e:.9} S_direct_seconds={s:.9} E_L8_apply_kernel_seconds={ek:.9} S_direct_L8_leaf_kernel_seconds={sk:.9} E_scope=version_checks_selected_terminal_generation_L8_expression_gather E_L8_kernel_scope=compiled_L8_apply_including_internal_version_and_finite_checks S_direct_scope=full_legacy_terminal_table_full_L9_validation_full_L9_policy_scan_L8_direct_lookup",
                reward as u8, e_times.len(), s_times.len());
            report_break_even(name, reward, "E", "direct", b, e, s);
            if materialized_timing {
                let m = median(&mut m_times);
                let mk = median(&mut mk_times);
                let me = median(&mut me_times);
                let mek = median(&mut mek_times);
                println!("loop1-probe timing vector={name} include_reward={} E_materialized_context_samples={} S_materialized_samples={} E_materialized_context_seconds={me:.9} S_materialized_seconds={m:.9} E_materialized_context_L8_apply_kernel_seconds={mek:.9} S_materialized_L8_leaf_kernel_seconds={mk:.9} E_L8_kernel_scope=compiled_L8_apply_including_internal_version_and_finite_checks S_materialized_scope=full_legacy_terminal_table_full_L9_materialization_L8_RAM_lookup L9_materialized_bytes={}",
                    reward as u8, me_times.len(), m_times.len(), l9.total_states * 4);
                report_break_even(name, reward, "E", "materialized", b, me, m);
            }
            let p = median(&mut p_times);
            let pk = median(&mut pk_times);
            let ps = median(&mut ps_times);
            let psk = median(&mut psk_times);
            println!("loop1-probe production_timing vector={name} include_reward={} repetitions={reps} order=PSSP P_samples={} P_context_S_materialized_samples={} P_seconds={p:.9} P_context_S_materialized_seconds={ps:.9} P_L8_leaf_kernel_seconds={pk:.9} P_context_S_materialized_L8_leaf_kernel_seconds={psk:.9} B_L9_only_seconds={l9_only_b:.9} P_scope=version_checks_full_terminal_generation_full_table_gather_prepare_token_full_table_finite_validation_full_L9_ID_value_gather_L8_RAM_lookup P_L9_values_materialized=1 P_L9_policy_scanned=0 P_kernel=original_L8_fixed_policy_RAM_lookup S_kernel=original_L8_fixed_policy_RAM_lookup",
                reward as u8, p_times.len(), ps_times.len());
            report_break_even(name, reward, "P", "materialized", l9_only_b, p, ps);
        }
    }
    sources.check()?;
    compiled9.check_inputs_unchanged()?;
    compiled8.check_inputs_unchanged()?;
    println!("loop1-probe complete bit_identical_f32=1 independent_input_vectors={} reward_modes=2 process_lifetime_VmHWM_bytes={} source_versions_unchanged=1 optimizer_defaults_changed=0", vectors.len(),
        peak_rss_bytes().map_or_else(|| "unavailable".to_string(), |n| n.to_string()));
    Ok(())
}

fn report_break_even(
    name: &str,
    reward: bool,
    algorithm: &str,
    baseline: &str,
    b: f64,
    e: f64,
    s: f64,
) {
    if e < s {
        println!("loop1-probe amortization vector={name} include_reward={} algorithm={algorithm} baseline={baseline} B_seconds={b:.9} B_over_S_minus_apply={:.6} break_even_applications={} cost_model=B+k{algorithm}_vs_kS applies_only_to_this_fixed_policy_L9_L8_slab=1", reward as u8, b / (s - e), (b / (s - e)).ceil());
    } else {
        println!("loop1-probe amortization vector={name} include_reward={} algorithm={algorithm} baseline={baseline} B_seconds={b:.9} break_even_applications=none_measured apply_ge_S=1 applies_only_to_this_fixed_policy_L9_L8_slab=1", reward as u8);
    }
}

pub(super) fn validate_real(
    graph: &Path,
    pruned: &Path,
    policy: &Path,
    values: &Path,
    maximum: usize,
) -> Result<()> {
    let input = values.to_str().context("non-UTF8 L0 input path")?;
    execute(graph, pruned, policy, input, maximum, None)
}

pub(super) fn run(args: &[String]) -> Result<()> {
    anyhow::ensure!((6..=8).contains(&args.len()),
        "usage: {} loop1-probe <graph> <pruned> <policy> <l0.f32|zero> [max_l8_states=2000000] [repetitions=5]", args[0]);
    let maximum = args
        .get(6)
        .map(|s| s.parse::<usize>())
        .transpose()
        .context("invalid max_l8_states")?
        .unwrap_or(DEFAULT_STATES);
    anyhow::ensure!(
        (1..=MAX_STATES).contains(&maximum),
        "max_l8_states must be 1..={MAX_STATES}"
    );
    let reps = args
        .get(7)
        .map(|s| s.parse::<usize>())
        .transpose()
        .context("invalid repetitions")?
        .unwrap_or(5);
    anyhow::ensure!(
        (3..=1000).contains(&reps),
        "repetitions must be 3..=1000 for ABBA timing"
    );
    execute(
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        &args[5],
        maximum,
        Some(reps),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_comparison_rejects_nonfinite_and_signed_zero_mismatch() {
        assert!(bit_match(&[f32::NAN], &[f32::NAN], "nan").is_err());
        assert!(bit_match(&[f32::INFINITY], &[f32::INFINITY], "infinite").is_err());
        assert!(bit_match(&[0.0], &[-0.0], "signed zero").is_err());
        assert!(bit_match(&[-0.0, -2.0], &[-0.0, -2.0], "equal").is_ok());
    }

    #[test]
    fn slab_cap_keeps_whole_contiguous_fields_and_rejects_oversized_first_field() -> Result<()> {
        let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
        let prefix = scratch.path.join("slab-cap");
        let mut writer = crate::layer_store::FieldChainWriter::create(&prefix)?;
        for (field, count) in [(2u32, 4u32), (5, 3), (9, 4)] {
            for qid in 0..count {
                writer.push_pair_key(((field as u64) << 32) | qid as u64)?;
            }
        }
        writer.finish()?;
        let layer = FieldChainLayer::open_index_only(prefix)?;
        assert_eq!(whole_field_slab(&layer, 6)?, 0..1);
        assert_eq!(whole_field_slab(&layer, 7)?, 0..2);
        assert_eq!(whole_field_slab(&layer, 11)?, 0..3);
        assert!(whole_field_slab(&layer, 3).is_err());
        Ok(())
    }
}
