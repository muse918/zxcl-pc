//! Exact fixed-policy expressions for the original boundary coordinates.
//!
//! This is a child of `loop1_sparse` so construction uses the same immutable
//! mapped policy, validation contract and action decoder as the scalar path.
//! Memo collisions discard only construction work; every expression remains
//! globally interned. Leaves hold the ordered legacy terminal formula, with
//! the reward inside its sum. No boundary reduction or expression rewrite is
//! performed. Logical node/edge caps are separate from the fixed memo budget.
//! Read-only source mappings are periodically released as well: a cgroup's
//! charged file cache can differ from this process's mapped resident pages.

use super::*;
use crate::expression_dag::{ExprDag, NodeId};
use crate::loop1_stream::{CompiledL9, CompiledL9FormulaLookup, Loop1Limits};
use std::collections::HashMap;

const MAX_DAG_MEMO_BYTES: u64 = 4 << 30;
const NO_NODE: NodeId = NodeId::MAX;

#[derive(Clone, Copy, Debug)]
pub(crate) struct WholeDagLimits {
    pub total_memo_bytes: u64,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub progress: bool,
}

impl Default for WholeDagLimits {
    fn default() -> Self {
        Self {
            total_memo_bytes: 256 << 20,
            max_nodes: 5_000_000,
            max_edges: 25_000_000,
            progress: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct WholeDagStats {
    pub roots: u64,
    pub state_requests: u64,
    pub evaluated_states: u64,
    pub state_hits: u64,
    pub state_collisions: u64,
    pub action_hits: u64,
    pub action_collisions: u64,
    pub missing_children: u64,
    pub tail_lookups: u64,
    pub membership_bytes: u64,
    pub policy_bytes: u64,
    pub reachable_validation_bytes: u64,
    pub nodes: usize,
    pub edges: usize,
    pub memo_payload_bytes: u64,
    pub validated_field_bytes: u64,
    pub source_release_calls: u64,
    pub source_range_release_calls: u64,
    pub source_window_flushes: u64,
    pub source_advised_ranges: u64,
    pub source_window_storage_bytes: u64,
    pub source_retire_seconds: f64,
    pub source_phase_retire_seconds: f64,
    pub source_range_retire_seconds: f64,
    /// Conservative read charges, not measured physical bytes or RSS.
    pub source_read_charge_bytes: u64,
    pub construction_seconds: f64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct NodeSlot {
    key: u64,
    node: NodeId,
    _padding: u32,
}

struct NodeMemo(Vec<NodeSlot>);

impl NodeMemo {
    fn new(bytes: u64) -> Result<Self> {
        let maximum = usize::try_from(bytes / size_of::<NodeSlot>() as u64)
            .context("DAG memo size exceeds address space")?;
        let count = if maximum == 0 {
            0
        } else {
            1usize << maximum.ilog2()
        };
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .context("allocate bounded DAG memo")?;
        slots.resize(
            count,
            NodeSlot {
                key: EMPTY_KEY,
                node: NO_NODE,
                _padding: 0,
            },
        );
        Ok(Self(slots))
    }

    #[inline]
    fn get(&self, key: u64) -> Option<NodeId> {
        if self.0.is_empty() {
            return None;
        }
        let slot = self.0[hash(key) as usize & (self.0.len() - 1)];
        (slot.key == key).then_some(slot.node)
    }

    #[inline]
    fn insert(&mut self, key: u64, node: NodeId) -> bool {
        if self.0.is_empty() {
            return false;
        }
        let at = hash(key) as usize & (self.0.len() - 1);
        let old = self.0[at].key;
        self.0[at] = NodeSlot {
            key,
            node,
            _padding: 0,
        };
        old != EMPTY_KEY && old != key
    }

    fn bytes(&self) -> u64 {
        (self.0.capacity() * size_of::<NodeSlot>()) as u64
    }
}

/// Avoid constructing and allocating an ExprDag candidate on a repeated
/// average. Children include explicit zeros, and inactive slots stay zero.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SmallAverage {
    children: [NodeId; PIECE_COUNT],
    denominator: u8,
}

pub(crate) struct WholeFixedPlan<'s, 'p> {
    policy: &'s SparsePolicy<'p>,
    compiled_l9: Option<&'s CompiledL9<'p>>,
    dag: ExprDag,
    roots: Vec<NodeId>,
    pub(crate) stats: WholeDagStats,
}

fn check_tail_binding(policy: &SparsePolicy<'_>, tail: &CompiledL9<'_>) -> Result<()> {
    let layer = &policy.layers[9];
    tail.check_sparse_binding(
        policy.graph,
        policy.qfwd,
        policy.formulas,
        &layer.index.file,
        &layer.membership.file,
        &layer.policy.file,
    )
}

impl<'s, 'p> WholeFixedPlan<'s, 'p> {
    /// `None` selects every L0 member and requires full stored-policy
    /// validation. An explicit prefix is a probe, even if its size equals L0.
    pub(crate) fn compile(
        policy: &'s SparsePolicy<'p>,
        root_limit: Option<usize>,
        limits: WholeDagLimits,
        compiled_l9: Option<&'s CompiledL9<'p>>,
    ) -> Result<Self> {
        policy.check_inputs_unchanged()?;
        ensure!(
            limits.total_memo_bytes <= MAX_DAG_MEMO_BYTES,
            "DAG memo exceeds 4 GiB cap"
        );
        ensure!(
            root_limit.is_some() || policy.fully_validated,
            "all-root DAG compilation requires validate_all; reachable-only compilation is a probe"
        );
        let total = usize::try_from(policy.root_count()).context("L0 root count exceeds usize")?;
        ensure!(
            total <= 4_000_000,
            "DAG root output exceeds bounded experiment limit"
        );
        let count = root_limit.unwrap_or(total).min(total);
        let tail = if let Some(compiled) = compiled_l9 {
            check_tail_binding(policy, compiled)?;
            Some(compiled.formula_lookup()?)
        } else {
            None
        };
        // Do not carry pages from full validation or a previous application
        // into the constructor's anonymous memo allocation.
        let source_phase_start = Instant::now();
        policy.release_source_pages()?;
        let source_phase_seconds = source_phase_start.elapsed().as_secs_f64();
        let mut builder = Builder::new(policy, limits, tail)?;
        builder.stats.source_release_calls += 1;
        builder.stats.source_phase_retire_seconds += source_phase_seconds;
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(count)
            .context("allocate DAG root IDs")?;
        for offset in 0..count {
            let layer = &policy.layers[0];
            let block = layer
                .metas
                .partition_point(|m| m.chain_offset + u64::from(m.count) <= offset as u64);
            let meta = layer
                .metas
                .get(block)
                .context("DAG root offset outside metadata")?;
            builder.record_source_read(0, SourceKind::Membership, offset as u64 * 3)?;
            let qid = layer.qid(offset as u64);
            let node = builder.state(0, meta.field_id, qid).with_context(|| format!(
                "DAG compilation failed at root {offset}/{count}, field={}, qid={qid}, visits={}, evaluated={}, nodes={}, edges={}",
                meta.field_id, builder.stats.state_requests, builder.stats.evaluated_states,
                builder.dag.node_count(), builder.dag.edge_count()))?;
            roots.push(node);
            builder.stats.roots += 1;
            builder.report_progress(false);
        }
        policy.check_inputs_unchanged()?;
        if let Some(compiled) = compiled_l9 {
            check_tail_binding(policy, compiled)?;
        }
        builder.release_source_pages()?;
        builder.stats.nodes = builder.dag.node_count();
        builder.stats.edges = builder.dag.edge_count();
        builder.stats.construction_seconds = builder.started.elapsed().as_secs_f64();
        builder.report_progress(true);
        Ok(Self {
            policy,
            compiled_l9,
            dag: builder.dag,
            roots,
            stats: builder.stats,
        })
    }

    pub(crate) fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub(crate) fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.dag.edge_count()
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.policy.check_inputs_unchanged()?;
        if let Some(compiled) = self.compiled_l9 {
            check_tail_binding(self.policy, compiled)?;
        }
        Ok(())
    }

    /// Terminal entries are the coherent ordered formula values, not reduced
    /// boundary coordinates. The CLI obtains them from TerminalPlan and tests
    /// independently construct the same legacy `mean(reward + reset)` values.
    pub(crate) fn apply(&self, terminals: &[f32]) -> Result<Vec<f32>> {
        self.check_inputs_unchanged()?;
        // Evaluation reads only the owned DAG and terminals; its identity
        // checks do not touch the mapped policy payload. Construction and the
        // scalar reference release their own source pages at their boundaries.
        let values = self.dag.eval_f32(terminals)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.roots.len())
            .context("allocate DAG boundary output")?;
        for &root in &self.roots {
            output.push(values[root as usize]);
        }
        self.check_inputs_unchanged()?;
        Ok(output)
    }
}

struct Builder<'s, 'p> {
    policy: &'s SparsePolicy<'p>,
    tail: Option<CompiledL9FormulaLookup<'s>>,
    dag: ExprDag,
    states: NodeMemo,
    actions: NodeMemo,
    terminals: Vec<NodeId>,
    averages: HashMap<SmallAverage, NodeId>,
    validated: Vec<u8>,
    field_bases: [usize; 10],
    stats: WholeDagStats,
    progress: bool,
    started: Instant,
    last_report: Instant,
    source_pages: SourceTouchWindow,
}

impl<'s, 'p> Builder<'s, 'p> {
    fn new(
        policy: &'s SparsePolicy<'p>,
        limits: WholeDagLimits,
        tail: Option<CompiledL9FormulaLookup<'s>>,
    ) -> Result<Self> {
        let started = Instant::now();
        let dag = ExprDag::new(
            policy.formulas.terminal_count(),
            limits.max_nodes,
            limits.max_edges,
        )?;
        let actions = NodeMemo::new(limits.total_memo_bytes / 4)?;
        let states = NodeMemo::new(limits.total_memo_bytes - limits.total_memo_bytes / 4)?;
        let mut field_bases = [0; 10];
        let mut total_fields = 0usize;
        for (depth, layer) in policy.layers.iter().enumerate() {
            field_bases[depth] = total_fields;
            total_fields = total_fields
                .checked_add(layer.metas.len())
                .context("DAG validated-field index overflow")?;
        }
        let validation_bytes = if policy.fully_validated {
            0
        } else {
            total_fields.div_ceil(8)
        };
        let mut validated = Vec::new();
        validated
            .try_reserve_exact(validation_bytes)
            .context("allocate DAG validation bitmap")?;
        validated.resize(validation_bytes, 0);
        let mut terminals = Vec::new();
        terminals
            .try_reserve_exact(policy.formulas.terminal_count())
            .context("allocate DAG terminal-node IDs")?;
        terminals.resize(policy.formulas.terminal_count(), NO_NODE);
        let source_pages = SourceTouchWindow::new()?;
        let stats = WholeDagStats {
            memo_payload_bytes: actions.bytes() + states.bytes(),
            validated_field_bytes: validated.capacity() as u64,
            source_window_storage_bytes: source_pages.storage_bytes(),
            ..WholeDagStats::default()
        };
        Ok(Self {
            policy,
            tail,
            dag,
            states,
            actions,
            terminals,
            averages: HashMap::new(),
            validated,
            field_bases,
            stats,
            progress: limits.progress,
            started,
            last_report: started,
            source_pages,
        })
    }

    fn release_source_pages(&mut self) -> Result<()> {
        self.flush_source_window()?;
        let start = Instant::now();
        self.policy.release_source_pages()?;
        self.stats.source_phase_retire_seconds += start.elapsed().as_secs_f64();
        self.stats.source_release_calls += 1;
        Ok(())
    }

    fn flush_source_window(&mut self) -> Result<()> {
        self.source_pages.flush(self.policy)?;
        self.stats.source_window_flushes = self.source_pages.flush_count();
        self.stats.source_advised_ranges = self.source_pages.range_count();
        self.stats.source_retire_seconds = self.source_pages.retire_seconds();
        Ok(())
    }

    /// Track the bounded source buckets actually touched. Retiring those
    /// buckets avoids repeatedly walking every page table of the much larger
    /// input mappings. The shared window also accounts for fault-around pages.
    fn record_source_read(
        &mut self,
        depth: usize,
        kind: SourceKind,
        byte_offset: u64,
    ) -> Result<()> {
        self.source_pages
            .record(self.policy, depth, kind, byte_offset)?;
        self.stats.source_read_charge_bytes += SOURCE_SCALAR_READ_CHARGE_BYTES;
        self.stats.source_window_flushes = self.source_pages.flush_count();
        self.stats.source_advised_ranges = self.source_pages.range_count();
        self.stats.source_retire_seconds = self.source_pages.retire_seconds();
        Ok(())
    }

    fn report_progress(&mut self, force: bool) {
        if self.progress && (force || self.last_report.elapsed().as_secs() >= 10) {
            eprintln!("loop1-dag compile roots={} visits={} evaluated={} state_hits={} action_hits={} collisions={} tail_lookups={} nodes={} edges={} memo_bytes={} source_windows={} source_ranges={} source_retire_seconds={:.3} phase_retire_seconds={:.3} range_retire_seconds={:.3} seconds={:.1}",
                self.stats.roots, self.stats.state_requests, self.stats.evaluated_states,
                self.stats.state_hits, self.stats.action_hits,
                self.stats.state_collisions + self.stats.action_collisions,
                self.stats.tail_lookups, self.dag.node_count(), self.dag.edge_count(),
                self.stats.memo_payload_bytes, self.stats.source_window_flushes, self.stats.source_advised_ranges,
                self.stats.source_retire_seconds, self.stats.source_phase_retire_seconds,
                self.stats.source_range_retire_seconds, self.started.elapsed().as_secs_f64());
            self.last_report = Instant::now();
        }
    }

    fn terminal(&mut self, zid: u32) -> Result<NodeId> {
        let saved = *self
            .terminals
            .get(zid as usize)
            .context("DAG terminal ID outside domain")?;
        if saved != NO_NODE {
            return Ok(saved);
        }
        let node = self.dag.terminal(zid as usize)?;
        self.terminals[zid as usize] = node;
        Ok(node)
    }

    fn find(&mut self, depth: usize, field: FieldId, qid: u32) -> Result<Option<u64>> {
        // Copy the immutable policy reference so a residency release can take
        // `&mut self` while this layer's immutable mapped view remains live.
        let policy = self.policy;
        let layer = &policy.layers[depth];
        let Ok(block) = layer.metas.binary_search_by_key(&field, |m| m.field_id) else {
            return Ok(None);
        };
        let meta = &layer.metas[block];
        if !self.policy.fully_validated {
            let at = self.field_bases[depth] + block;
            if self.validated[at / 8] & (1 << (at % 8)) == 0 {
                let mut previous = None;
                let mut done = 0u64;
                while done < u64::from(meta.count) {
                    let count = (u64::from(meta.count) - done).min(65_536);
                    for i in 0..count {
                        let member = layer.qid(meta.chain_offset + done + i);
                        ensure!(
                            (member as usize) < self.policy.qstate_count
                                && previous.is_none_or(|p| p < member),
                            "invalid DAG reachable membership layer={depth} field={field}"
                        );
                        previous = Some(member);
                    }
                    // A single first-touch field scan can exceed the scalar
                    // request window. Release its contiguous chunks directly.
                    let retire_start = Instant::now();
                    policy.release_source_range(depth, meta.chain_offset + done, count)?;
                    self.stats.source_range_retire_seconds += retire_start.elapsed().as_secs_f64();
                    self.stats.source_range_release_calls += 1;
                    done += count;
                }
                self.stats.reachable_validation_bytes += u64::from(meta.count) * 3;
                self.validated[at / 8] |= 1 << (at % 8);
            }
        }
        let (mut lo, mut hi) = (0u64, u64::from(meta.count));
        while lo < hi {
            let middle = lo + (hi - lo) / 2;
            self.record_source_read(
                depth,
                SourceKind::Membership,
                (meta.chain_offset + middle) * 3,
            )?;
            let member = layer.qid(meta.chain_offset + middle);
            self.stats.membership_bytes += 3;
            if member < qid {
                lo = middle + 1;
            } else {
                hi = middle;
            }
        }
        if lo < u64::from(meta.count) {
            self.record_source_read(depth, SourceKind::Membership, (meta.chain_offset + lo) * 3)?;
            self.stats.membership_bytes += 3;
            if layer.qid(meta.chain_offset + lo) == qid {
                return Ok(Some(meta.chain_offset + lo));
            }
        }
        Ok(None)
    }

    fn state(&mut self, depth: usize, field: FieldId, qid: u32) -> Result<NodeId> {
        ensure!(
            depth < 10
                && (field == EMPTY_FIELD_ID || (field as usize) < self.policy.graph.offsets.len())
                && (qid as usize) < self.policy.qstate_count,
            "DAG state outside immutable domain"
        );
        self.stats.state_requests += 1;
        if self.stats.state_requests & ((1 << 16) - 1) == 0 {
            self.flush_source_window()?;
            self.report_progress(false);
        }
        if depth == 9 {
            if let Some(tail) = &mut self.tail {
                self.stats.tail_lookups += 1;
                return match tail.get(field, qid)? {
                    Some(zid) => self.terminal(zid),
                    None => Ok(self.dag.zero()),
                };
            }
        }
        let key = ((depth as u64) << 56) | (u64::from(field) << 32) | u64::from(qid);
        if let Some(node) = self.states.get(key) {
            self.stats.state_hits += 1;
            return Ok(node);
        }
        let Some(at) = self.find(depth, field, qid)? else {
            self.stats.missing_children += 1;
            let node = self.dag.zero();
            self.stats.state_collisions += u64::from(self.states.insert(key, node));
            return Ok(node);
        };
        self.stats.evaluated_states += 1;
        self.stats.policy_bytes += 2;
        self.record_source_read(depth, SourceKind::Policy, at * 2)?;
        let code = self.policy.layers[depth].code(at);
        let node = if code == u16::MAX {
            self.dag.zero()
        } else {
            let action = self.policy.decode(depth, field, qid, code)?;
            if action.terminal {
                self.terminal(action.zid)?
            } else {
                let action_key = ((depth as u64) << 56)
                    | (u64::from(action.target) << 32)
                    | u64::from(action.zid);
                if let Some(node) = self.actions.get(action_key) {
                    self.stats.action_hits += 1;
                    node
                } else {
                    let count = action.end - action.start;
                    let mut average = SmallAverage {
                        children: [0; PIECE_COUNT],
                        denominator: count as u8,
                    };
                    for slot in 0..count {
                        let child = self.policy.qfwd.records[action.start + slot].child_qid;
                        average.children[slot] = self.state(depth + 1, action.target, child)?;
                    }
                    let node = if let Some(&node) = self.averages.get(&average) {
                        node
                    } else {
                        self.averages
                            .try_reserve(1)
                            .context("allocate DAG small-average index")?;
                        let node = self.dag.average(&average.children[..count], count)?;
                        self.averages.insert(average, node);
                        node
                    };
                    self.stats.action_collisions +=
                        u64::from(self.actions.insert(action_key, node));
                    node
                }
            }
        };
        self.stats.state_collisions += u64::from(self.states.insert(key, node));
        Ok(node)
    }
}

fn same_finite_bits(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "DAG/scalar boundary length mismatch"
    );
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite DAG/scalar value at root {index}"
        );
        ensure!(
            a.to_bits() == b.to_bits(),
            "DAG/scalar bit mismatch at root {index}: {a:?} != {b:?}"
        );
    }
    Ok(())
}

fn setting(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// This command publishes a new experimental boundary vector only. A complete
/// original numeric evaluation is requested only by the explicit roots=0 mode.
/// Baseline mode `full` also checks every selected signed root. Mode `1`
/// keeps that additional signed check to a prefix of at most 32 roots.
pub(crate) fn run(args: &[String]) -> Result<()> {
    ensure!((7..=13).contains(&args.len()),
        "usage: loop1-dag-probe <graph> <pruned> <policy> <x.f32|zero> <new-y.f32> [roots=32;0=all] [memo_mib=256] [validation=reachable|full] [repetitions=5] [tail=l9|none] [baseline=full|1|0]");
    let setup_start = Instant::now();
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let initial = &args[5];
    let output = PathBuf::from(&args[6]);
    let roots: usize = args
        .get(7)
        .map_or(Ok(32), |value| value.parse())
        .context("invalid roots")?;
    let memo_mib: u64 = args
        .get(8)
        .map_or(Ok(256), |value| value.parse())
        .context("invalid memo_mib")?;
    let validation = args.get(9).map_or("reachable", String::as_str);
    let repetitions: usize = args
        .get(10)
        .map_or(Ok(5), |value| value.parse())
        .context("invalid repetitions")?;
    let tail = args.get(11).map_or("l9", String::as_str);
    let baseline = args.get(12).map_or("1", String::as_str);
    ensure!(
        validation == "reachable" || validation == "full",
        "validation must be reachable or full"
    );
    ensure!(
        roots != 0 || validation == "full",
        "roots=0 requires validation=full"
    );
    ensure!(tail == "l9" || tail == "none", "tail must be l9 or none");
    ensure!(
        baseline == "full" || baseline == "1" || baseline == "0",
        "baseline must be full, 1 or 0"
    );
    ensure!(
        (1..=10_000).contains(&repetitions),
        "repetitions must be 1..=10000"
    );
    ensure!(
        !output.exists(),
        "DAG probe output already exists: {}",
        output.display()
    );
    let memo_bytes = memo_mib
        .checked_mul(1 << 20)
        .context("DAG memo size overflow")?;
    ensure!(
        memo_bytes <= MAX_DAG_MEMO_BYTES,
        "DAG memo_mib must be <=4096"
    );
    let max_nodes = setting("PI_LOOP1_DAG_MAX_NODES", 5_000_000)?;
    let max_edges = setting("PI_LOOP1_DAG_MAX_EDGES", 25_000_000)?;
    ensure!(
        (1..=5_000_000).contains(&max_nodes) && max_edges <= 25_000_000,
        "DAG probe caps require 1..=5m nodes and at most 25m edges"
    );
    let mut paths = vec![graph_path.clone()];
    if initial != "zero" {
        paths.push(PathBuf::from(initial));
    }
    for depth in 0..10 {
        let prefix = crate::q_pruned_prefix(&pruned_dir, depth);
        paths.extend([
            prefix.with_extension("field_index.bin"),
            prefix.with_extension("chains.u24.bin"),
            crate::mdp_policy_u16_path(&policy_dir, depth),
        ]);
    }
    #[cfg(target_os = "linux")]
    let external_watches = std::iter::once(graph_path.clone())
        .chain((initial != "zero").then(|| PathBuf::from(initial)))
        .map(|path| Ok((path.clone(), InputMutationWatch::open(&path)?)))
        .collect::<Result<Vec<_>>>()?;
    let bindings = crate::loop1_prepared::SourceBindings::capture_paths(paths)?;
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old = crate::enumerate_states();
    let q_index = crate::build_q_index_from_old(&old);
    drop(old);
    let qfwd = crate::build_qforward_graph(&q_index);
    let formula_build_start = Instant::now();
    let pre = crate::pre_reveal::PreRevealIndex::build(&q_index)?;
    let terminal_plan = crate::loop1_terminal::TerminalPlan::build(&q_index, &qfwd, &pre)?;
    let formula_build_seconds = formula_build_start.elapsed().as_secs_f64();
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("missing TWO_LINE_HASH")?;
    let layers = (0..10)
        .map(|depth| FieldChainLayer::open_index_only(crate::q_pruned_prefix(&pruned_dir, depth)))
        .collect::<Result<Vec<_>>>()?;
    let mut policy = SparsePolicy::open(&graph, &qfwd, &pre, &layers, &policy_dir, two_line)?;
    ensure!(
        layers[0].metas.len() == 1 && graph.hash(layers[0].metas[0].field_id) == 0,
        "DAG probe requires one physical-empty L0 block"
    );
    let mut reset = vec![0.0f32; q_index.states.len()];
    let mut previous = None;
    for i in 0..policy.root_count() {
        let qid = policy.layers[0].qid(i);
        ensure!(
            (qid as usize) < reset.len() && previous.is_none_or(|p| p < qid),
            "DAG probe L0 membership must be sorted, unique and in-domain"
        );
        previous = Some(qid);
    }
    if initial != "zero" {
        let values = crate::f32_ram::load(Path::new(initial), policy.root_count())?;
        for (index, value) in values.into_iter().enumerate() {
            ensure!(value.is_finite(), "nonfinite DAG probe reset at {index}");
            reset[policy.layers[0].qid(index as u64) as usize] = value;
        }
    }
    let check_external = || -> Result<()> {
        bindings.check()?;
        #[cfg(target_os = "linux")]
        for (path, watch) in &external_watches {
            watch.check(path)?;
        }
        Ok(())
    };
    check_external()?;
    let validation_start = Instant::now();
    let validated_states = if validation == "full" {
        policy.validate_all(true)?
    } else {
        0
    };
    let validation_seconds = validation_start.elapsed().as_secs_f64();
    let build_start = Instant::now();
    let compiled_l9 = if tail == "l9" {
        let mut compiled = CompiledL9::build(
            &graph,
            &q_index,
            &qfwd,
            &pre,
            &layers[9],
            &crate::mdp_policy_u16_path(&policy_dir, 9),
            Loop1Limits::default(),
        )?;
        compiled.build_rank_index(64 << 20, 32 << 20)?;
        Some(compiled)
    } else {
        None
    };
    let plan = WholeFixedPlan::compile(
        &policy,
        (roots != 0).then_some(roots),
        WholeDagLimits {
            total_memo_bytes: memo_bytes,
            max_nodes,
            max_edges,
            progress: true,
        },
        compiled_l9.as_ref(),
    )?;
    check_external()?;
    let construction_seconds = build_start.elapsed().as_secs_f64() + formula_build_seconds;
    eprintln!("loop1-dag ready scope={} validation={} validated_states={} validation_seconds={:.6} tail={} roots={} nodes={} edges={} B_seconds={:.6} setup_seconds={:.6} stats={:?}",
        if roots == 0 { "all-ten-layer-boundary" } else { "bounded-root-probe" },
        validation, validated_states, validation_seconds, tail, plan.root_count(), plan.node_count(),
        plan.edge_count(), construction_seconds, setup_start.elapsed().as_secs_f64(), plan.stats);

    let numeric_limits = SparseLimits {
        workers: 4,
        total_memo_bytes: memo_bytes,
        progress: true,
    };
    let mut accepted_probe = None;
    let mut signed = vec![0.0f32; reset.len()];
    for i in 0..policy.root_count() {
        signed[policy.layers[0].qid(i) as usize] = match i % 7 {
            0 => 16_777_216.0,
            1 => 1.0,
            2 => -16_777_216.0,
            3 => -3.0,
            4 => -0.0,
            5 => 7.0,
            _ => -2.0,
        };
    }
    for (name, vector, reward) in [
        ("actual-affine", &reset, true),
        ("signed-linear", &signed, false),
    ] {
        check_external()?;
        let terminals = terminal_plan.evaluate_f32(vector, reward)?;
        let dag_values = plan.apply(&terminals)?;
        let mut scalar_seconds = None;
        let mut checked_roots = 0;
        if baseline != "0" {
            let table_start = Instant::now();
            check_external()?;
            let table = crate::build_terminal_reset_avg_table_opt(
                &qfwd,
                vector,
                1.0,
                q_index.states.len(),
                reward,
            );
            let table_seconds = table_start.elapsed().as_secs_f64();
            // The real and signed checks must use coherent legacy terminal
            // values; arbitrary independent per-placement tables are not a
            // legal input to this shared-expression operator.
            for (qid, row) in table.iter().enumerate() {
                for (placed, value) in row.iter().enumerate() {
                    if let Some(zid) = pre.zid_for_placement(qid as u32, placed as u8) {
                        ensure!(
                            value.to_bits() == terminals[zid as usize].to_bits(),
                            "terminal-plan/legacy table bit mismatch qid={qid} placed={placed}"
                        );
                    }
                }
            }
            let selected = if name == "actual-affine" || baseline == "full" {
                (roots != 0).then_some(roots)
            } else {
                Some(plan.root_count().min(32))
            };
            let scalar_start = Instant::now();
            let scalar = policy.apply(&table, 1.0, selected, numeric_limits)?;
            check_external()?;
            let scalar_apply_seconds = scalar_start.elapsed().as_secs_f64();
            checked_roots = scalar.values.len();
            same_finite_bits(&dag_values[..checked_roots], &scalar.values)?;
            scalar_seconds = Some(table_seconds + scalar_apply_seconds);
            eprintln!(
                "loop1-dag scalar case={name} roots={checked_roots} seconds={:.6} stats={:?}",
                scalar_seconds.unwrap(),
                scalar.stats
            );
        }
        let mut samples = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let start = Instant::now();
            check_external()?;
            let terminals = terminal_plan.evaluate_f32(vector, reward)?;
            let values = plan.apply(&terminals)?;
            check_external()?;
            samples.push(start.elapsed().as_secs_f64());
            same_finite_bits(&values, &dag_values)?;
        }
        let evaluation_seconds = median(&mut samples);
        let break_even = scalar_seconds
            .filter(|_| checked_roots == plan.root_count())
            .filter(|scalar| *scalar > evaluation_seconds)
            .map(|scalar| {
                ((construction_seconds + validation_seconds) / (scalar - evaluation_seconds)).ceil()
            });
        eprintln!("loop1-dag result case={name} scope={} roots={} finite=1 scalar_bitchecked_roots={} scalar_seconds={:?} E_median_seconds={:.9} B_seconds={:.9} break_even_applications={:?} repetitions={} total_build_plus_validation_seconds={:.9}",
            if roots == 0 { "all-ten-layer-boundary" } else { "bounded-root-probe" },
            plan.root_count(), checked_roots, scalar_seconds, evaluation_seconds,
            construction_seconds, break_even, repetitions, construction_seconds + validation_seconds);
        if name == "actual-affine" {
            accepted_probe = Some(dag_values);
        }
    }
    let values = accepted_probe.context("missing actual DAG probe result")?;
    check_external()?;
    plan.check_inputs_unchanged()?;
    let temporary = output.with_extension(format!("dag-tmp-{}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let publication = (|| -> Result<()> {
        let mut writer = BufWriter::new(file);
        for value in &values {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        check_external()?;
        plan.check_inputs_unchanged()?;
        fs::hard_link(&temporary, &output)
            .with_context(|| format!("publish new DAG probe output {}", output.display()))?;
        Ok(())
    })();
    let cleanup = fs::remove_file(&temporary);
    publication?;
    cleanup?;
    eprintln!(
        "loop1-dag experimental_output={} full_solve=0",
        output.display()
    );
    Ok(())
}

#[cfg(test)]
#[path = "loop1_sparse_dag_tests.rs"]
mod tests;
