//! Complete L9/L8 fixed-policy A/B experiment with two reusable L8 buffers.
//!
//! This command never writes source data or scratch value tables. Its caller
//! must run it inside the shared 32 GiB / four-CPU experiment limit. It measures
//! the production RAM kernel against complete compiled L8, not a full loop1.

use super::*;
use crate::layer_store::FieldChainLayer;
use crate::loop1_prepared::SourceBindings;
use crate::loop1_stream::{CompiledL8, CompiledL9, Loop1Limits};
use crate::loop1_terminal::TerminalPlan;
use crate::pre_reveal::PreRevealIndex;
use crate::sweep_schedule::{SweepPlan, SweepTuning};

const GIB: u64 = 1 << 30;
const MEMORY_LIMIT: u64 = 32 * GIB;
const COMPILER_HEAP_LIMIT: u64 = 16 * GIB;
const WORKERS: usize = 4;
const TASK_CACHE_BYTES: usize = 192 << 20;
const BUILD_CACHE_BYTES: usize = 32 << 20;
const SNAPSHOT_CHUNK_STATES: usize = 1 << 20;

/// Start mutation observation before taking metadata snapshots or parsing any
/// input. Metadata-only bindings can miss same-size writes within one clock
/// tick; unread watch queues retain those events for the whole experiment.
struct ProbeSources {
    bindings: SourceBindings,
    #[cfg(target_os = "linux")]
    mutation_watches: Vec<(PathBuf, crate::loop1_stream::InputMutationWatch)>,
}

impl ProbeSources {
    fn capture(paths: Vec<PathBuf>) -> Result<Self> {
        #[cfg(target_os = "linux")]
        let mutation_watches = paths
            .iter()
            .map(|path| {
                let canonical = fs::canonicalize(path)
                    .with_context(|| format!("canonicalize probe input {}", path.display()))?;
                let watch = crate::loop1_stream::InputMutationWatch::open(&canonical)?;
                Ok((path.clone(), watch))
            })
            .collect::<Result<Vec<_>>>()?;
        let bindings = SourceBindings::capture_paths(paths)?;
        let sources = Self {
            bindings,
            #[cfg(target_os = "linux")]
            mutation_watches,
        };
        sources.check()?;
        Ok(sources)
    }

    fn check(&self) -> Result<()> {
        self.bindings.check()?;
        #[cfg(target_os = "linux")]
        for (path, watch) in &self.mutation_watches {
            watch.check(path)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct Timed {
    total: f64,
    terminal: f64,
    l9: f64,
    l8: f64,
    unique: f64,
    gather: f64,
}

fn output_vec(states: u64) -> Result<Vec<f32>> {
    let count = usize::try_from(states).context("full L8 state count exceeds usize")?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .context("allocate full L8 probe output")?;
    out.resize(count, 0.0);
    Ok(out)
}

fn bit_match(
    pool: &rayon::ThreadPool,
    expected: &[f32],
    actual: &[f32],
    label: &str,
) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "{label}: output length mismatch"
    );
    let mismatch = pool.install(|| {
        expected
            .par_iter()
            .zip(actual.par_iter())
            .enumerate()
            .find_first(|(_, (a, b))| {
                !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits()
            })
    });
    if let Some((index, (&expected, &actual))) = mismatch {
        anyhow::bail!(
            "{label}: state {index}: original={expected:?}/0x{:08x} compiled={actual:?}/0x{:08x}",
            expected.to_bits(),
            actual.to_bits()
        );
    }
    Ok(())
}

fn cases(layer0: &FieldChainLayer, input: &str) -> Result<Vec<(&'static str, Vec<f32>, bool)>> {
    let qids = layer0_reset_block_qids(layer0)?;
    anyhow::ensure!(
        qids.len() == CHECK_QSTATE_COUNT && layer0.total_states == CHECK_QSTATE_COUNT as u64,
        "full-l8-probe requires one complete L0 reset block"
    );
    let raw = load_boundary_vector(input, layer0.total_states)?;
    let mut dense = vec![0.0; CHECK_QSTATE_COUNT];
    let mut seen = vec![false; CHECK_QSTATE_COUNT];
    for (&qid, &value) in qids.iter().zip(&raw) {
        anyhow::ensure!(
            (qid as usize) < dense.len() && !seen[qid as usize],
            "invalid/duplicate reset qid {qid}"
        );
        anyhow::ensure!(value.is_finite(), "nonfinite L0 value at qid {qid}");
        dense[qid as usize] = value;
        seen[qid as usize] = true;
    }
    anyhow::ensure!(
        seen.iter().all(|&value| value),
        "incomplete L0 reset membership"
    );
    let scale = 1.0 / (CHECK_QSTATE_COUNT as f32).sqrt();
    let signed = (0..CHECK_QSTATE_COUNT)
        .map(|qid| {
            let mut word = (qid as u64).wrapping_add(0x8973_cede_44b5_2231);
            word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            word ^= word >> 31;
            ((((word >> 16) & 0xffff) as i32 - 32768) as f32 / 32768.0) * scale
        })
        .collect();
    Ok(vec![
        (
            if input == "zero" {
                "zero_affine"
            } else {
                "actual_l0_affine"
            },
            dense,
            true,
        ),
        ("independent_signed_linear", signed, false),
    ])
}

/// A snapshot is only an optional lineage observation. A disagreement is
/// reported, never treated as evidence against the freshly computed reference.
fn compare_snapshot(file: &File, values: &[f32], layer: usize) -> Result<()> {
    anyhow::ensure!(
        file.metadata()?.len() == values.len() as u64 * 4,
        "Vstar L{layer} size differs from full membership"
    );
    let mut bytes = vec![0u8; SNAPSHOT_CHUNK_STATES * 4];
    let (mut differences, mut nonfinite, mut first) = (0u64, 0u64, None);
    for start in (0..values.len()).step_by(SNAPSHOT_CHUNK_STATES) {
        let count = (values.len() - start).min(SNAPSHOT_CHUNK_STATES);
        read_exact_at_file(file, &mut bytes[..count * 4], start as u64 * 4)?;
        for (offset, word) in bytes[..count * 4].chunks_exact(4).enumerate() {
            let saved = f32::from_le_bytes(word.try_into().expect("four-byte snapshot word"));
            let current = values[start + offset];
            if !saved.is_finite() || !current.is_finite() {
                nonfinite += 1;
            }
            if saved.to_bits() != current.to_bits() {
                differences += 1;
                first.get_or_insert(start + offset);
            }
        }
    }
    println!("full-l8-probe snapshot_lineage layer={layer} states={} bitwise_differences={differences} nonfinite={nonfinite} first_difference={first:?} exact_match={} snapshot_is_reference=0", values.len(), differences == 0 && nonfinite == 0);
    Ok(())
}

struct Legacy<'a> {
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    l8: &'a FieldChainLayer,
    l9: &'a FieldChainLayer,
    policy8: &'a File,
    policy9: &'a File,
    plan8: &'a SweepPlan,
    plan9: &'a SweepPlan,
    pool: &'a rayon::ThreadPool,
    sources: &'a ProbeSources,
    two_line: FieldId,
    sparse_limit: usize,
}

impl Legacy<'_> {
    fn kernel(
        &self,
        layer: usize,
        table: &[[f32; 7]],
        next: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        let (current, policy, plan) = if layer == 9 {
            (self.l9, self.policy9, self.plan9)
        } else {
            (self.l8, self.policy8, self.plan8)
        };
        anyhow::ensure!(
            out.len() as u64 == current.total_states,
            "original L{layer} output size mismatch"
        );
        let sink = SharedF32Out {
            ptr: out.as_mut_ptr(),
            len: out.len(),
        };
        let mut total_states = 0u64;
        for batch in plan.ranges.chunks(WORKERS * 4) {
            let results: Vec<Result<FullMdpChunkResult>> = self.pool.install(|| {
                batch
                    .par_iter()
                    .map(|range| {
                        compute_fixed_policy_chunk_f32_with_sources(
                            layer,
                            range.clone(),
                            current,
                            policy,
                            None,
                            Some(sink),
                            self.graph,
                            self.qfwd,
                            table,
                            if layer == 8 { Some(self.l9) } else { None },
                            None,
                            next,
                            1.0,
                            plan.per_task_cache_bytes,
                            CHECK_QSTATE_COUNT,
                            self.sparse_limit,
                            self.two_line,
                            None,
                            None,
                        )
                    })
                    .collect()
            });
            for result in results {
                total_states += result?.stats.total_states;
            }
        }
        anyhow::ensure!(
            total_states == current.total_states,
            "original L{layer} state scan incomplete"
        );
        Ok(())
    }

    fn apply(
        &self,
        reset: &[f32],
        reward: bool,
        out: &mut [f32],
        snapshot9: Option<&File>,
    ) -> Result<Timed> {
        let full = Instant::now();
        self.sources.check()?;
        let terminal = Instant::now();
        let table =
            build_terminal_reset_avg_table_opt(self.qfwd, reset, 1.0, CHECK_QSTATE_COUNT, reward);
        anyhow::ensure!(
            table.iter().flatten().all(|value| value.is_finite()),
            "nonfinite original terminal table"
        );
        let terminal_seconds = terminal.elapsed().as_secs_f64();
        let l9 = Instant::now();
        let mut next = output_vec(self.l9.total_states)?;
        self.kernel(9, &table, None, &mut next)?;
        let l9_seconds = l9.elapsed().as_secs_f64();
        let snapshot = Instant::now();
        if let Some(snapshot9) = snapshot9 {
            compare_snapshot(snapshot9, &next, 9)?;
        }
        let snapshot_seconds = snapshot.elapsed().as_secs_f64();
        let l8 = Instant::now();
        self.kernel(8, &table, Some(&next), out)?;
        let l8_seconds = l8.elapsed().as_secs_f64();
        drop(next);
        drop(table);
        self.sources.check()?;
        Ok(Timed {
            total: full.elapsed().as_secs_f64() - snapshot_seconds,
            terminal: terminal_seconds,
            l9: l9_seconds,
            l8: l8_seconds,
            ..Timed::default()
        })
    }
}

struct Compiled<'a> {
    pre: &'a PreRevealIndex,
    terminal: &'a TerminalPlan,
    l8: &'a CompiledL8<'a>,
    sources: &'a ProbeSources,
    pool: &'a rayon::ThreadPool,
}

impl Compiled<'_> {
    fn apply(&self, reset: &[f32], reward: bool, out: &mut [f32]) -> Result<Timed> {
        let full = Instant::now();
        self.sources.check()?;
        let terminal = Instant::now();
        // Match the production full solver: lower layers still need the full
        // terminal table, even though this measurement stops after L8.
        let terminals = self.terminal.evaluate_f32(reset, reward)?;
        let table = crate::loop1_prepared::terminal_table(self.pre, &terminals)?;
        anyhow::ensure!(
            table.iter().flatten().all(|value| value.is_finite()),
            "nonfinite compiled terminal table"
        );
        let terminal_seconds = terminal.elapsed().as_secs_f64();
        let unique = Instant::now();
        let averages = self
            .pool
            .install(|| self.l8.evaluate_averages_f32(&terminals))?;
        let unique_seconds = unique.elapsed().as_secs_f64();
        let gather = Instant::now();
        self.pool
            .install(|| self.l8.gather_range_into_f32(&averages, 0..out.len(), out))?;
        let gather_seconds = gather.elapsed().as_secs_f64();
        drop(averages);
        drop(table);
        drop(terminals);
        self.sources.check()?;
        Ok(Timed {
            total: full.elapsed().as_secs_f64(),
            terminal: terminal_seconds,
            unique: unique_seconds,
            gather: gather_seconds,
            ..Timed::default()
        })
    }
}

fn mean(samples: &[Timed], select: impl Fn(&Timed) -> f64) -> f64 {
    samples.iter().map(select).sum::<f64>() / samples.len() as f64
}

pub(super) fn run(args: &[String]) -> Result<()> {
    anyhow::ensure!((6..=8).contains(&args.len()),
        "usage: {} full-l8-probe <graph.bin> <pruned_dir> <policy_dir> <L0.f32|zero> [repetitions=3] [Vstar_dir]", args.first().map_or("zxcl", String::as_str));
    let graph_path = Path::new(&args[2]);
    let pruned = Path::new(&args[3]);
    let policy = Path::new(&args[4]);
    let input = &args[5];
    let repetitions = if args.len() >= 7 {
        args[6]
            .parse::<usize>()
            .context("invalid full-l8 repetitions")?
    } else {
        3
    };
    anyhow::ensure!(
        (1..=10).contains(&repetitions),
        "full-l8 repetitions must be 1..=10"
    );
    let snapshot_dir = args.get(7).map(Path::new);
    let policy8_path = mdp_policy_u16_path(policy, 8);
    let policy9_path = mdp_policy_u16_path(policy, 9);
    let mut source_paths = vec![
        graph_path.to_path_buf(),
        policy8_path.clone(),
        policy9_path.clone(),
    ];
    for layer in [0, 8, 9] {
        let prefix = q_pruned_prefix(pruned, layer);
        source_paths.extend([
            prefix.with_extension("field_index.bin"),
            prefix.with_extension("chains.u24.bin"),
        ]);
    }
    if input != "zero" {
        source_paths.push(PathBuf::from(input));
    }
    if let Some(directory) = snapshot_dir {
        source_paths.extend([
            mdp_f32_value_path(directory, 8),
            mdp_f32_value_path(directory, 9),
        ]);
    }
    let sources = ProbeSources::capture(source_paths)?;
    let setup = Instant::now();
    let graph = FieldGraph::load_graph_bin(graph_path)?;
    let q_index = {
        let old = enumerate_states();
        build_q_index_from_old(&old)
    };
    anyhow::ensure!(
        q_index.states.len() == CHECK_QSTATE_COUNT,
        "unexpected quotient domain"
    );
    let qfwd = build_qforward_graph(&q_index);
    let layer0 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 0))?;
    let layer8 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 8))?;
    let layer9 = FieldChainLayer::open_index_only(q_pruned_prefix(pruned, 9))?;
    anyhow::ensure!(
        layer8.total_states > 0 && layer9.total_states > 0,
        "full-l8-probe requires nonempty L8/L9"
    );
    let value8_bytes = layer8
        .total_states
        .checked_mul(4)
        .context("L8 value bytes overflow")?;
    let value9_bytes = layer9
        .total_states
        .checked_mul(4)
        .context("L9 value bytes overflow")?;
    for (path, states) in [
        (&policy8_path, layer8.total_states),
        (&policy9_path, layer9.total_states),
    ] {
        anyhow::ensure!(
            fs::metadata(path)?.len() == states.checked_mul(2).context("policy size overflow")?,
            "policy length differs from membership: {}",
            path.display()
        );
    }
    let input_cases = cases(&layer0, input)?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    // Before allocating any giant map, prove that the specified maximum
    // compiler heap plus both full outputs leave common and worker allowances.
    let conservative_preflight = COMPILER_HEAP_LIMIT
        .checked_add(
            value8_bytes
                .checked_mul(2)
                .context("two L8 buffers overflow")?,
        )
        .and_then(|bytes| bytes.checked_add(value9_bytes))
        .and_then(|bytes| bytes.checked_add(4 * GIB))
        .and_then(|bytes| {
            bytes.checked_add(WORKERS as u64 * (2 * TASK_CACHE_BYTES as u64 + (192 << 20)))
        })
        .context("full L8 peak memory estimate overflow")?;
    anyhow::ensure!(conservative_preflight <= MEMORY_LIMIT,
        "full-l8-probe conservative payload estimate {conservative_preflight} exceeds 32 GiB; no value buffers were allocated");
    sources.check()?;
    let build = Instant::now();
    let pre_start = Instant::now();
    let pre = PreRevealIndex::build(&q_index)?;
    let pre_seconds = pre_start.elapsed().as_secs_f64();
    let terminal_start = Instant::now();
    let terminal = TerminalPlan::build(&q_index, &qfwd, &pre)?;
    let terminal_seconds = terminal_start.elapsed().as_secs_f64();
    let limits = Loop1Limits {
        max_mapping_bytes: 4096 << 20,
        max_unique_averages: 32_000_000,
        max_cache_bytes: BUILD_CACHE_BYTES,
    };
    let l9_start = Instant::now();
    let mut compiled9 = CompiledL9::build(
        &graph,
        &q_index,
        &qfwd,
        &pre,
        &layer9,
        &policy9_path,
        limits,
    )?;
    compiled9.build_rank_index(64 << 20, BUILD_CACHE_BYTES)?;
    let l9_seconds = l9_start.elapsed().as_secs_f64();
    let l8_start = Instant::now();
    let compiled8 = CompiledL8::build_with_heap_limit(
        &graph,
        &q_index,
        &qfwd,
        &pre,
        &layer8,
        &policy8_path,
        0..layer8.metas.len(),
        &compiled9,
        limits,
        COMPILER_HEAP_LIMIT,
    )?;
    let l8_seconds = l8_start.elapsed().as_secs_f64();
    let build_seconds = build.elapsed().as_secs_f64();
    sources.check()?;
    anyhow::ensure!(
        compiled8.state_offset() == 0 && compiled8.state_count() as u64 == layer8.total_states,
        "compiled L8 does not cover the complete original layer"
    );
    let retained = compiled8
        .storage_bytes()
        .checked_add(compiled9.storage_bytes())
        .and_then(|bytes| bytes.checked_add(pre.storage_bytes()))
        .and_then(|bytes| bytes.checked_add(terminal.storage_bytes() as u64))
        .context("compiled retained capacity overflow")?;
    let tuning = SweepTuning::from_env(WORKERS * TASK_CACHE_BYTES)?;
    // Account for both persistent comparison outputs on every original pass.
    let plan9 = tuning.plan(
        &layer9.metas,
        value9_bytes,
        retained
            .checked_add(value8_bytes * 2)
            .context("L9 planner resident overflow")?,
        WORKERS,
        CHECK_QSTATE_COUNT,
    )?;
    let plan8 = tuning.plan(
        &layer8.metas,
        value8_bytes,
        retained
            .checked_add(value8_bytes)
            .and_then(|bytes| bytes.checked_add(value9_bytes))
            .context("L8 planner resident overflow")?,
        WORKERS,
        CHECK_QSTATE_COUNT,
    )?;
    anyhow::ensure!(
        plan8.workers == WORKERS && plan9.workers == WORKERS,
        "four-worker full-L8 comparison does not fit the current sweep memory plan"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKERS)
        .build()?;
    let policy8 = File::open(&policy8_path)?;
    let policy9 = File::open(&policy9_path)?;
    let snapshot8 = snapshot_dir
        .map(|directory| File::open(mdp_f32_value_path(directory, 8)))
        .transpose()?;
    let snapshot9 = snapshot_dir
        .map(|directory| File::open(mdp_f32_value_path(directory, 9)))
        .transpose()?;
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("missing two-line field")?;
    let legacy = Legacy {
        graph: &graph,
        qfwd: &qfwd,
        l8: &layer8,
        l9: &layer9,
        policy8: &policy8,
        policy9: &policy9,
        plan8: &plan8,
        plan9: &plan9,
        pool: &pool,
        sources: &sources,
        two_line,
        sparse_limit: env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(65_536),
    };
    let compiled = Compiled {
        pre: &pre,
        terminal: &terminal,
        l8: &compiled8,
        sources: &sources,
        pool: &pool,
    };
    println!("full-l8-probe scope=complete_L9_L8 production_original_kernel=1 full_loop1=0 boundary_solve=0 gamma=1 workers={WORKERS} repetitions={repetitions} order=ABBA output_buffers_reused=2 output_zero_initialization_in_timing=0 source_or_scratch_writes=0 setup_seconds={setup_seconds:.9} build_seconds={build_seconds:.9} pre_build_seconds={pre_seconds:.9} terminal_plan_build_seconds={terminal_seconds:.9} L9_build_seconds={l9_seconds:.9} L8_build_seconds={l8_seconds:.9}");
    println!("full-l8-probe L8_states={} L9_states={} unique_averages={} mapping_bytes={} retained_capacity_bytes={retained} compiler_peak_capacity_bytes={} compiler_heap_cap_bytes={COMPILER_HEAP_LIMIT} conservative_preflight_bytes={conservative_preflight} resource_limit_bytes={MEMORY_LIMIT} external_cgroup_required=1 L8_groups={} L9_groups={} L8_per_task_cache_bytes={} L9_per_task_cache_bytes={} L8_task_working_bytes={} L9_task_working_bytes={}",
        layer8.total_states, layer9.total_states, compiled8.averages().len(), compiled8.stats().mapping_bytes + compiled9.stats().mapping_bytes,
        compiled8.stats().compiler_peak_capacity_bytes, plan8.ranges.len(), plan9.ranges.len(),
        plan8.per_task_cache_bytes, plan9.per_task_cache_bytes, plan8.max_working_bytes, plan9.max_working_bytes);
    // These are the only two full L8 value vectors alive throughout the run.
    let mut reference = output_vec(layer8.total_states)?;
    let mut actual = output_vec(layer8.total_states)?;
    for (case_index, (name, reset, reward)) in input_cases.iter().enumerate() {
        println!(
            "full-l8-probe warmup case={name} include_reward={}",
            *reward as u8
        );
        legacy.apply(
            reset,
            *reward,
            &mut reference,
            if case_index == 0 {
                snapshot9.as_ref()
            } else {
                None
            },
        )?;
        if case_index == 0 {
            if let Some(snapshot8) = snapshot8.as_ref() {
                compare_snapshot(snapshot8, &reference, 8)?;
            }
        }
        compiled.apply(reset, *reward, &mut actual)?;
        bit_match(&pool, &reference, &actual, name)?;
        let (mut original_samples, mut compiled_samples) = (Vec::new(), Vec::new());
        for repetition in 0..repetitions {
            for (step, use_compiled) in [false, true, true, false].into_iter().enumerate() {
                let sample = if use_compiled {
                    let sample = compiled.apply(reset, *reward, &mut actual)?;
                    compiled_samples.push(sample);
                    sample
                } else {
                    let sample = legacy.apply(reset, *reward, &mut reference, None)?;
                    original_samples.push(sample);
                    sample
                };
                bit_match(&pool, &reference, &actual, name)?;
                sources.check()?;
                println!("full-l8-probe sample case={name} repetition={repetition} step={step} algorithm={} seconds={:.9} full_bitwise_finite_match=1 validation_outside_timing=1", if use_compiled { "compiled" } else { "original" }, sample.total);
            }
        }
        let original_seconds = mean(&original_samples, |sample| sample.total);
        let compiled_seconds = mean(&compiled_samples, |sample| sample.total);
        println!("full-l8-probe timing case={name} include_reward={} original_samples={} compiled_samples={} original_seconds={original_seconds:.9} compiled_seconds={compiled_seconds:.9} original_terminal_seconds={:.9} original_L9_seconds={:.9} original_L8_seconds={:.9} compiled_terminal_seconds={:.9} compiled_unique_seconds={:.9} compiled_gather_seconds={:.9} tail_speedup={:.9} scope=complete_L9_L8 whole_loop_speedup_measured=0",
            *reward as u8, original_samples.len(), compiled_samples.len(), mean(&original_samples, |sample| sample.terminal),
            mean(&original_samples, |sample| sample.l9), mean(&original_samples, |sample| sample.l8),
            mean(&compiled_samples, |sample| sample.terminal), mean(&compiled_samples, |sample| sample.unique),
            mean(&compiled_samples, |sample| sample.gather), original_seconds / compiled_seconds);
        if original_seconds > compiled_seconds {
            println!("full-l8-probe amortization case={name} build_seconds={build_seconds:.9} break_even_applications={:.0} scope=one_fixed_policy_complete_L9_L8", (build_seconds / (original_seconds - compiled_seconds)).ceil());
        } else {
            println!("full-l8-probe amortization case={name} build_seconds={build_seconds:.9} break_even_applications=none_measured");
        }
    }
    sources.check()?;
    println!("full-l8-probe complete all_L8_states_bitwise_validated=1 cases=2 full_loop1=0 outputs_written=0");
    Ok(())
}
