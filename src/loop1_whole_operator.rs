//! Opt-in fixed-policy boundary operator in the original L0 coordinates.
//!
//! The policy, complete expression plan and formula plan are separate CLI
//! locals borrowed here. No object owns a value that borrows another field of
//! itself. The final reference follows stored policies numerically, using the
//! original terminal-table builder and fresh bounded caches without a DAG or
//! compiled L9 shortcut.

use anyhow::{ensure, Context, Result};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::graph::FieldGraph;
use crate::layer_store::FieldChainLayer;
use crate::loop1_prepared::SourceBindings;
use crate::loop1_sparse::dag::{WholeDagLimits, WholeFixedPlan};
use crate::loop1_sparse::{SparseLimits, SparsePolicy};
use crate::loop1_stream::CompiledL9;
#[cfg(target_os = "linux")]
use crate::loop1_stream::InputMutationWatch;
use crate::loop1_terminal::TerminalPlan;
use crate::qgraph::QForwardGraph;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// Leave the other half of a 32 GiB run for the graph, owned expressions,
/// evaluation scratch and mapped-input residency. This is planning headroom;
/// the shared cgroup and the runner's supplemental RSS monitor enforce the
/// experiment's process and charged-memory limits.
pub(crate) const BASIS_CAP_BYTES: u64 = 16 * GIB;

fn enabled(name: &str, default: bool) -> bool {
    std::env::var(name).ok().map_or(default, |value| {
        value != "0" && !value.eq_ignore_ascii_case("false") && !value.eq_ignore_ascii_case("no")
    })
}

fn setting(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Settings {
    pub(crate) dag: WholeDagLimits,
    pub(crate) reference: SparseLimits,
    pub(crate) l9_tail: bool,
}

impl Settings {
    /// Validate before any graph loading or scratch/output creation.
    pub(crate) fn from_env(
        gamma: f32,
        compiled_l8: bool,
        action_cache_bytes: usize,
    ) -> Result<Option<Self>> {
        if !enabled("PI_LOOP1_WHOLE_DAG", false) {
            return Ok(None);
        }
        ensure!(gamma == 1.0, "PI_LOOP1_WHOLE_DAG requires gamma=1");
        ensure!(
            !compiled_l8,
            "PI_LOOP1_WHOLE_DAG conflicts with PI_LOOP1_COMPILED_L8"
        );
        ensure!(
            action_cache_bytes == 0,
            "PI_LOOP1_WHOLE_DAG conflicts with PI_LOOP1_ACTION_CACHE_MIB"
        );
        ensure!(
            cfg!(target_os = "linux"),
            "PI_LOOP1_WHOLE_DAG requires Linux mapped-input residency controls"
        );
        let memo_mib = setting("PI_LOOP1_WHOLE_MEMO_MIB", 4096)?;
        let reference_mib = setting("PI_LOOP1_WHOLE_REFERENCE_MEMO_MIB", 4096)?;
        let max_nodes = setting("PI_LOOP1_WHOLE_MAX_NODES", 5_000_000)?;
        let max_edges = setting("PI_LOOP1_WHOLE_MAX_EDGES", 25_000_000)?;
        ensure!(memo_mib <= 4096, "PI_LOOP1_WHOLE_MEMO_MIB must be <=4096");
        ensure!(
            reference_mib <= 8192,
            "PI_LOOP1_WHOLE_REFERENCE_MEMO_MIB must be <=8192"
        );
        ensure!(
            (1..=5_000_000).contains(&max_nodes),
            "PI_LOOP1_WHOLE_MAX_NODES must be in 1..=5000000"
        );
        ensure!(
            max_edges <= 25_000_000,
            "PI_LOOP1_WHOLE_MAX_EDGES must be <=25000000"
        );
        Ok(Some(Self {
            dag: WholeDagLimits {
                total_memo_bytes: memo_mib * MIB,
                max_nodes: usize::try_from(max_nodes)?,
                max_edges: usize::try_from(max_edges)?,
                progress: true,
            },
            reference: SparseLimits {
                workers: 4,
                total_memo_bytes: reference_mib * MIB,
                progress: true,
            },
            l9_tail: enabled("PI_LOOP1_WHOLE_L9_TAIL", true),
        }))
    }
}

/// Watches start before taking metadata snapshots or parsing any source. Every
/// input is covered, including graph, L0 reset, and an optional comparison file.
pub(crate) struct InputGuard {
    bindings: SourceBindings,
    #[cfg(target_os = "linux")]
    watches: Vec<(PathBuf, InputMutationWatch)>,
}

impl InputGuard {
    pub(crate) fn capture(paths: Vec<PathBuf>) -> Result<Self> {
        #[cfg(target_os = "linux")]
        let watches = paths
            .iter()
            .map(|path| {
                let canonical = fs::canonicalize(path)
                    .with_context(|| format!("canonicalize whole-DAG input {}", path.display()))?;
                Ok((path.clone(), InputMutationWatch::open(&canonical)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let bindings = SourceBindings::capture_paths(paths)?;
        let result = Self {
            bindings,
            #[cfg(target_os = "linux")]
            watches,
        };
        result.check()?;
        Ok(result)
    }

    pub(crate) fn check(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        for (path, watch) in &self.watches {
            watch.check(path)?;
        }
        self.bindings.check()
    }
}

pub(crate) fn validate_root_layout(
    graph: &FieldGraph,
    layers: &[FieldChainLayer],
    reset_qids: &[u32],
) -> Result<()> {
    ensure!(layers.len() == 10, "whole DAG requires exactly ten layers");
    let layer0 = &layers[0];
    let first = layer0
        .metas
        .first()
        .context("whole DAG requires nonempty L0 membership")?;
    ensure!(
        (first.field_id as usize) < graph.hashes.len() && graph.hash(first.field_id) == 0,
        "whole DAG requires the first L0 block to be physical-empty (hash=0)"
    );
    ensure!(
        !reset_qids.is_empty()
            && reset_qids.len() == first.count as usize
            && reset_qids.len() as u64 <= layer0.total_states,
        "whole DAG reset membership differs from the first L0 boundary block"
    );
    Ok(())
}

pub(crate) fn validated_policy<'p>(
    graph: &'p FieldGraph,
    qfwd: &'p QForwardGraph,
    pre: &'p crate::pre_reveal::PreRevealIndex,
    layers: &'p [FieldChainLayer],
    policy_dir: &std::path::Path,
    two_line_field: crate::graph::FieldId,
    guard: &InputGuard,
) -> Result<SparsePolicy<'p>> {
    guard.check()?;
    let start = Instant::now();
    let mut policy = SparsePolicy::open(graph, qfwd, pre, layers, policy_dir, two_line_field)?;
    let states = policy.validate_all(true)?;
    guard.check()?;
    eprintln!(
        "whole-dag validation complete states={states} seconds={:.6} full_policy=1",
        start.elapsed().as_secs_f64()
    );
    Ok(policy)
}

pub(crate) fn compile<'s, 'p>(
    policy: &'s SparsePolicy<'p>,
    settings: Settings,
    tail: Option<&'s CompiledL9<'p>>,
    guard: &InputGuard,
) -> Result<WholeFixedPlan<'s, 'p>> {
    guard.check()?;
    let start = Instant::now();
    let plan = WholeFixedPlan::compile(
        policy,
        None,
        settings.dag,
        if settings.l9_tail { tail } else { None },
    )?;
    ensure!(
        plan.root_count() as u64 == policy.root_count(),
        "whole DAG omitted boundary roots"
    );
    guard.check()?;
    eprintln!("whole-dag ready roots={} nodes={} edges={} seconds={:.6} boundary_reduced=0 full_policy=1 tail_l9={} basis_cap_bytes={BASIS_CAP_BYTES} stats={:?}",
        plan.root_count(), plan.node_count(), plan.edge_count(), start.elapsed().as_secs_f64(), settings.l9_tail && tail.is_some(), plan.stats);
    Ok(plan)
}

pub(crate) fn same_finite_bits(expected: &[f32], actual: &[f32]) -> Result<()> {
    ensure!(
        expected.len() == actual.len(),
        "whole DAG/reference boundary length mismatch"
    );
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        ensure!(
            expected.is_finite() && actual.is_finite(),
            "whole DAG/reference nonfinite value at boundary {index}"
        );
        ensure!(expected.to_bits() == actual.to_bits(),
            "whole DAG/reference bit mismatch at boundary {index}: dag=0x{:08x} numeric=0x{:08x}; output was not published",
            expected.to_bits(), actual.to_bits());
    }
    Ok(())
}

pub(crate) struct FullBoundaryDag<'a, 's, 'p> {
    plan: &'a WholeFixedPlan<'s, 'p>,
    policy: &'s SparsePolicy<'p>,
    terminal: &'a TerminalPlan,
    qfwd: &'p QForwardGraph,
    reset_qids: &'a [u32],
    guard: &'a InputGuard,
    reference: SparseLimits,
}

impl<'a, 's, 'p> FullBoundaryDag<'a, 's, 'p> {
    pub(crate) fn new(
        plan: &'a WholeFixedPlan<'s, 'p>,
        policy: &'s SparsePolicy<'p>,
        terminal: &'a TerminalPlan,
        qfwd: &'p QForwardGraph,
        reset_qids: &'a [u32],
        guard: &'a InputGuard,
        reference: SparseLimits,
    ) -> Result<Self> {
        ensure!(
            policy.fully_validated(),
            "whole DAG requires full stored-policy validation"
        );
        ensure!(
            plan.root_count() >= reset_qids.len()
                && plan.root_count() as u64 == policy.root_count(),
            "whole DAG boundary order/length mismatch"
        );
        let mut previous = None;
        for &qid in reset_qids {
            ensure!(
                (qid as usize) < qfwd.offsets.len() && previous.is_none_or(|old| old < qid),
                "whole DAG reset qids must be sorted, unique and in-domain"
            );
            previous = Some(qid);
        }
        let operator = Self {
            plan,
            policy,
            terminal,
            qfwd,
            reset_qids,
            guard,
            reference,
        };
        operator.check_inputs_unchanged()?;
        Ok(operator)
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.guard.check()?;
        self.plan.check_inputs_unchanged()?;
        self.policy.check_inputs_unchanged()
    }

    fn dense_reset(&self, values: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            values.len() == self.plan.root_count(),
            "whole DAG boundary input length mismatch"
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "nonfinite whole DAG boundary input"
        );
        let mut reset = vec![0.0; self.qfwd.offsets.len()];
        for (&qid, &value) in self.reset_qids.iter().zip(values) {
            ensure!(
                value.is_finite(),
                "nonfinite whole DAG boundary input at qid {qid}"
            );
            reset[qid as usize] = value;
        }
        Ok(reset)
    }

    pub(crate) fn apply(&self, values: &[f32], linear: bool) -> Result<Vec<f32>> {
        self.check_inputs_unchanged()?;
        let start = Instant::now();
        let reset = self.dense_reset(values)?;
        let terminals = self.terminal.evaluate_f32(&reset, !linear)?;
        let output = self.plan.apply(&terminals)?;
        ensure!(
            output.iter().all(|value| value.is_finite()),
            "nonfinite whole DAG output"
        );
        self.check_inputs_unchanged()?;
        eprintln!(
            "whole-dag apply roots={} linear={linear} seconds={:.6}",
            output.len(),
            start.elapsed().as_secs_f64()
        );
        Ok(output)
    }

    pub(crate) fn apply_verified(&self, values: &[f32], linear: bool) -> Result<Vec<f32>> {
        self.check_inputs_unchanged()?;
        let verification_start = Instant::now();
        let dag_start = Instant::now();
        let dag_values = self.apply(values, linear)?;
        let dag_seconds = dag_start.elapsed().as_secs_f64();
        let numeric_start = Instant::now();
        let dense_start = Instant::now();
        let reset = self.dense_reset(values)?;
        let dense_seconds = dense_start.elapsed().as_secs_f64();
        // Use the original terminal arithmetic, not TerminalPlan or a compiled
        // L9 source, so the final gate is independent of expression compilation.
        let terminal_start = Instant::now();
        let table = crate::build_terminal_reset_avg_table_opt(
            self.qfwd,
            &reset,
            1.0,
            self.qfwd.offsets.len(),
            !linear,
        );
        let terminal_seconds = terminal_start.elapsed().as_secs_f64();
        eprintln!(
            "whole-dag numeric-reference roots={} gamma=1 compiled_l9=0 dag=0",
            self.plan.root_count()
        );
        let policy_start = Instant::now();
        let reference = self.policy.apply(&table, 1.0, None, self.reference)?;
        let policy_wall_seconds = policy_start.elapsed().as_secs_f64();
        let numeric_wall_seconds = numeric_start.elapsed().as_secs_f64();
        eprintln!("whole-dag numeric-reference timing roots={} dense_reset_seconds={dense_seconds:.9} terminal_table_seconds={terminal_seconds:.9} policy_apply_wall_seconds={policy_wall_seconds:.9} policy_apply_reported_seconds={:.9} numeric_reference_wall_seconds={numeric_wall_seconds:.9} numeric_wall_scope=dense_reset_through_policy_return_includes_status_log excludes_DAG_bit_gate_final_guard=1 source_retire_seconds_is_worker_sum=1 stats={:?}",
            reference.values.len(), reference.stats.elapsed_seconds, reference.stats);
        let bit_start = Instant::now();
        same_finite_bits(&dag_values, &reference.values)?;
        let bit_seconds = bit_start.elapsed().as_secs_f64();
        self.check_inputs_unchanged()?;
        eprintln!(
            "whole-dag verified roots={} finite=1 bitwise_match=1 dag_seconds={dag_seconds:.9} bit_compare_seconds={bit_seconds:.9} verification_wall_seconds={:.9} verification_scope=after_initial_guard_through_final_guard",
            reference.values.len(), verification_start.elapsed().as_secs_f64()
        );
        Ok(reference.values)
    }
}

#[cfg(test)]
#[path = "loop1_whole_operator_tests.rs"]
mod tests;
