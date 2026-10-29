//! Bounded, demand-driven compilation of the ten-layer, gamma=1 boundary map.
//!
//! Membership always comes from the original pruned layer, including for a
//! requested root. A terminal edge ends before its final random reveal: its
//! variable is already a pre-reveal average. This is an algebraic boundary
//! change and does not promise the old f32 sweep's rounding.

use anyhow::{bail, ensure, Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use crate::expression_dag::{ExprDag, NodeId};
use crate::graph::{FieldGraph, EMPTY_FIELD_ID};
use crate::layer_store::FieldChainLayer;
use crate::piece::PIECE_COUNT;
use crate::pre_reveal::PreRevealIndex;
use crate::qgraph::{QForwardGraph, QForwardRec};
use crate::qstate::{
    decode_avail_pair, q_forward_after_reveal, QStateIndex, AVAIL_PAIR_COUNT, TAIL5_COUNT,
};

pub(crate) const POLICY_NONE: u16 = u16::MAX;
pub(crate) type StateKey = (usize, u32, u32);

/// Allows small, independent compiler fixtures as well as the production index.
/// Each variable must average the same reveal support as its qforward placement
/// range. PreRevealIndex with build_qforward_graph guarantees this contract for
/// the complete production quotient index.
pub(crate) trait BoundaryPlacementIndex {
    fn terminal_count(&self) -> usize;
    fn placement_zid(&self, qid: u32, placed: u8) -> Option<u32>;
}

impl BoundaryPlacementIndex for PreRevealIndex {
    fn terminal_count(&self) -> usize {
        self.len()
    }
    fn placement_zid(&self, qid: u32, placed: u8) -> Option<u32> {
        self.zid_for_placement(qid, placed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileMode {
    Fixed,
    Greedy,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompileLimits {
    pub max_states: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    /// Bound for resident encoded membership/policy blocks. Oversized blocks
    /// use positional binary searches, requiring only a three-byte scratch.
    pub field_cache_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompileStats {
    pub construction_seconds: f64,
    pub states: usize,
    pub missing_states: usize,
    pub memo_hits: usize,
    pub action_records: usize,
    pub terminal_actions: usize,
    pub nodes: usize,
    pub edges: usize,
    pub field_cache_hits: usize,
    pub field_cache_misses: usize,
    pub field_cache_peak_bytes: usize,
    pub uncached_lookups: usize,
    pub membership_bytes_read: u64,
    pub policy_bytes_read: u64,
}

impl CompileStats {
    /// Uses measured seconds per scalar application and DAG application. None
    /// means no measured amortization advantage; it is not a speedup claim.
    pub fn break_even_applications(&self, scalar_seconds: f64, dag_seconds: f64) -> Option<f64> {
        if !scalar_seconds.is_finite()
            || !dag_seconds.is_finite()
            || dag_seconds < 0.0
            || scalar_seconds <= dag_seconds
        {
            None
        } else {
            Some((self.construction_seconds / (scalar_seconds - dag_seconds)).ceil())
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActionNode {
    pub code: u16,
    pub node: NodeId,
}

#[derive(Clone, Debug)]
pub(crate) struct StateActions {
    pub state: StateKey,
    pub node: NodeId,
    /// Strict policy order: ascending placed piece, then original edge index.
    pub actions: Vec<ActionNode>,
}

pub(crate) struct CompiledDag {
    pub dag: ExprDag,
    /// Same order and multiplicity as the requested roots.
    pub roots: Vec<NodeId>,
    pub actions: Vec<StateActions>,
    pub stats: CompileStats,
}

impl CompiledDag {
    /// Extract actions from an evaluation of this same DAG. Zero and all ties
    /// preserve NONE or the first improving original action, respectively.
    pub fn greedy_policy_f64(&self, values: &[f64]) -> Result<Vec<(StateKey, u16)>> {
        ensure!(
            values.len() == self.dag.node_count(),
            "DAG evaluation length mismatch"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite DAG evaluation"
        );
        Ok(self
            .actions
            .iter()
            .map(|state| {
                let mut best = 0.0;
                let mut code = POLICY_NONE;
                for action in &state.actions {
                    let value = values[action.node as usize];
                    if value > best {
                        best = value;
                        code = action.code;
                    }
                }
                (state.state, code)
            })
            .collect())
    }

    pub fn greedy_policy_f32(&self, values: &[f32]) -> Result<Vec<(StateKey, u16)>> {
        ensure!(
            values.len() == self.dag.node_count(),
            "DAG evaluation length mismatch"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite DAG evaluation"
        );
        Ok(self
            .actions
            .iter()
            .map(|state| {
                let mut best = 0.0f32;
                let mut code = POLICY_NONE;
                for action in &state.actions {
                    let value = values[action.node as usize];
                    if value > best {
                        best = value;
                        code = action.code;
                    }
                }
                (state.state, code)
            })
            .collect())
    }
}

struct LayerFiles {
    membership: File,
    policy: Option<File>,
}

struct CachedBlock {
    layer: usize,
    block: usize,
    qids: Box<[u8]>,
    policy: Box<[u8]>,
    used: u64,
}

impl CachedBlock {
    fn bytes(&self) -> usize {
        self.qids.len() + self.policy.len()
    }
    fn qid(&self, i: usize) -> u32 {
        decode_u24(&self.qids[i * 3..i * 3 + 3])
    }
    fn find(&self, qid: u32) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.qids.len() / 3);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.qid(mid).cmp(&qid) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

pub(crate) struct Compiler<'a> {
    graph: &'a FieldGraph,
    q_index: &'a QStateIndex,
    qfwd: &'a QForwardGraph,
    layers: &'a [FieldChainLayer],
    pre: &'a dyn BoundaryPlacementIndex,
    files: Vec<LayerFiles>,
    mode: CompileMode,
    limits: CompileLimits,
    two_line_field: u32,
    dag: ExprDag,
    memo: HashMap<StateKey, NodeId>,
    actions: Vec<StateActions>,
    cache: Vec<CachedBlock>,
    cache_bytes: usize,
    validated_blocks: HashSet<(usize, usize)>,
    clock: u64,
    stats: CompileStats,
    started: Instant,
}

impl<'a> Compiler<'a> {
    pub fn new(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        qfwd: &'a QForwardGraph,
        layers: &'a [FieldChainLayer],
        pre: &'a dyn BoundaryPlacementIndex,
        policy_dir: Option<&Path>,
        mode: CompileMode,
        limits: CompileLimits,
        two_line_field: u32,
    ) -> Result<Self> {
        let started = Instant::now();
        ensure!(
            layers.len() == 10,
            "boundary compiler requires exactly ten layers"
        );
        ensure!(
            q_index.states.len() == qfwd.offsets.len(),
            "qstate/qforward size mismatch"
        );
        ensure!(
            (two_line_field as usize) < graph.offsets.len(),
            "two-line field outside graph"
        );
        ensure!(
            mode != CompileMode::Fixed || policy_dir.is_some(),
            "fixed compilation requires a policy directory"
        );
        let dag = ExprDag::new(pre.terminal_count(), limits.max_nodes, limits.max_edges)?;
        let mut files = Vec::with_capacity(10);
        for (layer, input) in layers.iter().enumerate() {
            // This must precede any graph accessor, including when a malformed
            // field belongs to a root that would otherwise return NONE.
            let mut previous = None;
            let mut total = 0u64;
            for meta in &input.metas {
                ensure!(
                    (meta.field_id as usize) < graph.offsets.len(),
                    "layer {layer} membership field {} outside graph",
                    meta.field_id
                );
                ensure!(
                    previous.map_or(true, |p| p < meta.field_id),
                    "unsorted layer {layer} fields"
                );
                ensure!(
                    meta.chain_offset == total,
                    "noncontiguous layer {layer} membership"
                );
                total = total
                    .checked_add(meta.count as u64)
                    .context("membership offset overflow")?;
                previous = Some(meta.field_id);
            }
            ensure!(
                total == input.total_states,
                "layer {layer} membership total mismatch"
            );
            let membership = File::open(&input.chains_path)
                .with_context(|| format!("open {}", input.chains_path.display()))?;
            ensure!(
                membership.metadata()?.len()
                    == total.checked_mul(3).context("membership size overflow")?,
                "layer {layer} membership file size mismatch"
            );
            let policy = if mode == CompileMode::Fixed {
                let path = policy_dir
                    .unwrap()
                    .join(format!("layer{layer}.policy.u16.bin"));
                let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
                ensure!(
                    file.metadata()?.len()
                        == total.checked_mul(2).context("policy size overflow")?,
                    "layer {layer} policy file size mismatch"
                );
                Some(file)
            } else {
                None
            };
            files.push(LayerFiles { membership, policy });
        }
        Ok(Self {
            graph,
            q_index,
            qfwd,
            layers,
            pre,
            files,
            mode,
            limits,
            two_line_field,
            dag,
            memo: HashMap::new(),
            actions: Vec::new(),
            cache: Vec::new(),
            cache_bytes: 0,
            validated_blocks: HashSet::new(),
            clock: 0,
            stats: CompileStats::default(),
            started,
        })
    }

    /// Consuming compilation prevents a cap failure from publishing a partial
    /// graph. Caller-owned output is untouched until this returns successfully.
    pub fn compile_roots(mut self, roots: &[StateKey]) -> Result<CompiledDag> {
        ensure!(
            roots.len() <= self.limits.max_states,
            "root count exceeds state cap"
        );
        let mut compiled = Vec::with_capacity(roots.len());
        for &(layer, field, qid) in roots {
            compiled.push(self.state(layer, field, qid)?);
        }
        self.stats.construction_seconds = self.started.elapsed().as_secs_f64();
        self.stats.nodes = self.dag.node_count();
        self.stats.edges = self.dag.edge_count();
        Ok(CompiledDag {
            dag: self.dag,
            roots: compiled,
            actions: self.actions,
            stats: self.stats,
        })
    }

    fn state(&mut self, layer: usize, field: u32, qid: u32) -> Result<NodeId> {
        ensure!(layer < 10, "state layer {layer} outside ten-layer graph");
        ensure!(
            (field as usize) < self.graph.offsets.len(),
            "state field {field} outside graph"
        );
        ensure!(
            (qid as usize) < self.q_index.states.len(),
            "state qid {qid} outside qstate index"
        );
        let key = (layer, field, qid);
        if let Some(&node) = self.memo.get(&key) {
            self.stats.memo_hits += 1;
            return Ok(node);
        }
        ensure!(
            self.stats.states < self.limits.max_states,
            "boundary DAG state cap {} exceeded",
            self.limits.max_states
        );
        self.stats.states += 1;
        let Some(policy) = self.membership(layer, field, qid)? else {
            self.stats.missing_states += 1;
            let node = self.dag.zero();
            self.memo.insert(key, node);
            return Ok(node);
        };
        let node = if self.mode == CompileMode::Fixed {
            let code = policy.context("fixed membership lacks policy code")?;
            if code == POLICY_NONE {
                self.dag.zero()
            } else {
                let placed = (code >> 8) as u8;
                let ei = (code & 255) as usize;
                ensure!(
                    (placed as usize) < PIECE_COUNT,
                    "invalid policy code {code:#06x} at {key:?}"
                );
                let edges = self.edges(field, placed)?;
                let target = *edges
                    .get(ei)
                    .with_context(|| format!("policy edge {ei} out of range at {key:?}"))?;
                self.action(layer, field, qid, placed, target)?
            }
        } else {
            let mut candidates = vec![self.dag.zero()];
            let mut actions = Vec::new();
            // Same order as qfwd_legal_placed: p=0..7, then graph edge order.
            for placed in 0..PIECE_COUNT as u8 {
                if self.records(qid, placed)?.is_empty() {
                    continue;
                }
                let edge_count = self.edges(field, placed)?.len();
                ensure!(
                    edge_count <= 256,
                    "field {field} piece {placed} cannot encode edge indices"
                );
                for ei in 0..edge_count {
                    let target = self.edges(field, placed)?[ei];
                    if layer == 9 && target != EMPTY_FIELD_ID {
                        continue;
                    }
                    ensure!(
                        self.stats.action_records < self.limits.max_edges,
                        "boundary DAG action metadata cap {} exceeded",
                        self.limits.max_edges
                    );
                    self.stats.action_records += 1;
                    let node = self.action(layer, field, qid, placed, target)?;
                    actions.push(ActionNode {
                        code: ((placed as u16) << 8) | ei as u16,
                        node,
                    });
                    candidates.push(node);
                }
            }
            let node = self.dag.maximum(&candidates)?;
            self.actions.push(StateActions {
                state: key,
                node,
                actions,
            });
            node
        };
        self.memo.insert(key, node);
        Ok(node)
    }

    fn edges(&self, field: u32, placed: u8) -> Result<&[u32]> {
        ensure!(
            (placed as usize) < PIECE_COUNT,
            "invalid placed piece {placed}"
        );
        let offsets = self
            .graph
            .offsets
            .get(field as usize)
            .context("field outside graph")?;
        let (a, b) = (
            offsets[placed as usize] as usize,
            offsets[placed as usize + 1] as usize,
        );
        ensure!(
            a <= b && b <= self.graph.targets.len(),
            "invalid graph offsets at field {field} piece {placed}"
        );
        Ok(&self.graph.targets[a..b])
    }

    fn records(&self, qid: u32, placed: u8) -> Result<&[QForwardRec]> {
        ensure!(
            (placed as usize) < PIECE_COUNT,
            "invalid placed piece {placed}"
        );
        let offsets = self
            .qfwd
            .offsets
            .get(qid as usize)
            .context("qid outside qforward graph")?;
        let (a, b) = (
            offsets[placed as usize] as usize,
            offsets[placed as usize + 1] as usize,
        );
        ensure!(
            a <= b && b <= self.qfwd.records.len(),
            "invalid qforward offsets at qid {qid} piece {placed}"
        );
        Ok(&self.qfwd.records[a..b])
    }

    fn action(
        &mut self,
        layer: usize,
        field: u32,
        qid: u32,
        placed: u8,
        target: u32,
    ) -> Result<NodeId> {
        ensure!(
            (target as usize) < self.graph.offsets.len() || target == EMPTY_FIELD_ID,
            "target field {target} outside graph at layer {layer} field {field}"
        );
        let parent = self.q_index.states[qid as usize];
        ensure!(
            (parent.avail as usize) < AVAIL_PAIR_COUNT
                && (parent.tail5 as usize) < TAIL5_COUNT
                && parent.bag_mask > 0
                && parent.bag_mask < 128,
            "invalid qstate {qid}"
        );
        let (a, b) = decode_avail_pair(parent.avail);
        ensure!(
            placed == a || placed == b,
            "illegal policy placement {placed} for qid {qid}"
        );
        let recs = self.records(qid, placed)?;
        ensure!(
            !recs.is_empty(),
            "placement {placed} has no legal reveals at layer {layer} field {field} qid {qid}"
        );
        let mut previous = None;
        for rec in recs {
            ensure!(
                (rec.reveal as usize) < PIECE_COUNT && previous.map_or(true, |p| p < rec.reveal),
                "qforward reveals must be distinct and ascending for qid {qid} piece {placed}"
            );
            let expected = q_forward_after_reveal(parent, placed, rec.reveal)
                .context("qforward record has illegal reveal")?;
            ensure!(
                self.q_index.states.get(rec.child_qid as usize) == Some(&expected),
                "qforward child mismatch for qid {qid} piece {placed} reveal {}",
                rec.reveal
            );
            previous = Some(rec.reveal);
        }
        let terminal = (layer == 4 && target == self.two_line_field)
            || (layer == 9 && target == EMPTY_FIELD_ID);
        if terminal {
            let zid = self
                .pre
                .placement_zid(qid, placed)
                .context("terminal placement lacks pre-reveal state")?;
            self.stats.terminal_actions += 1;
            return self.dag.terminal(zid as usize);
        }
        ensure!(
            layer < 9,
            "layer 9 policy targets nonterminal field {target}"
        );
        ensure!(
            (target as usize) < self.graph.offsets.len(),
            "nonterminal target field outside graph"
        );
        let denominator = recs.len();
        // The list is at most seven reveals; keep its original order and
        // multiplicity. Every child probes original layer membership.
        let mut children = Vec::with_capacity(denominator);
        for i in 0..denominator {
            let child_qid = self.records(qid, placed)?[i].child_qid;
            children.push(self.state(layer + 1, target, child_qid)?);
        }
        self.dag.average(&children, denominator)
    }

    /// Some(None) means present in a greedy layer, Some(Some(code)) in fixed.
    fn membership(&mut self, layer: usize, field: u32, qid: u32) -> Result<Option<Option<u16>>> {
        let Ok(block) = self.layers[layer]
            .metas
            .binary_search_by_key(&field, |m| m.field_id)
        else {
            return Ok(None);
        };
        self.clock = self.clock.checked_add(1).context("cache clock overflow")?;
        if let Some(entry) = self
            .cache
            .iter_mut()
            .find(|c| c.layer == layer && c.block == block)
        {
            self.stats.field_cache_hits += 1;
            entry.used = self.clock;
            return Ok(entry.find(qid).map(|i| {
                if self.mode == CompileMode::Fixed {
                    Some(u16::from_le_bytes([
                        entry.policy[i * 2],
                        entry.policy[i * 2 + 1],
                    ]))
                } else {
                    None
                }
            }));
        }
        self.stats.field_cache_misses += 1;
        let meta = &self.layers[layer].metas[block];
        let count = meta.count as usize;
        let qbytes = count
            .checked_mul(3)
            .context("membership block size overflow")?;
        let pbytes = if self.mode == CompileMode::Fixed {
            count.checked_mul(2).context("policy block size overflow")?
        } else {
            0
        };
        let bytes = qbytes
            .checked_add(pbytes)
            .context("field cache size overflow")?;
        if bytes > 0 && bytes <= self.limits.field_cache_bytes {
            while self.cache_bytes > self.limits.field_cache_bytes - bytes {
                let oldest = self
                    .cache
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, c)| c.used)
                    .unwrap()
                    .0;
                self.cache_bytes -= self.cache.swap_remove(oldest).bytes();
            }
            let mut qids = vec![0; qbytes].into_boxed_slice();
            read_exact_at(
                &self.files[layer].membership,
                &mut qids,
                meta.chain_offset
                    .checked_mul(3)
                    .context("membership offset overflow")?,
            )?;
            self.stats.membership_bytes_read += qbytes as u64;
            let mut prev = None;
            for chunk in qids.chunks_exact(3) {
                let q = decode_u24(chunk);
                ensure!(
                    (q as usize) < self.q_index.states.len() && prev.map_or(true, |p| p < q),
                    "invalid or unsorted membership qids at layer {layer} field {field}"
                );
                prev = Some(q);
            }
            self.validated_blocks.insert((layer, block));
            let mut policy = vec![0; pbytes].into_boxed_slice();
            if pbytes > 0 {
                read_exact_at(
                    self.files[layer].policy.as_ref().unwrap(),
                    &mut policy,
                    meta.chain_offset
                        .checked_mul(2)
                        .context("policy offset overflow")?,
                )?;
                self.stats.policy_bytes_read += pbytes as u64;
            }
            let entry = CachedBlock {
                layer,
                block,
                qids,
                policy,
                used: self.clock,
            };
            let result = entry.find(qid).map(|i| {
                if self.mode == CompileMode::Fixed {
                    Some(u16::from_le_bytes([
                        entry.policy[i * 2],
                        entry.policy[i * 2 + 1],
                    ]))
                } else {
                    None
                }
            });
            self.cache_bytes += bytes;
            self.stats.field_cache_peak_bytes =
                self.stats.field_cache_peak_bytes.max(self.cache_bytes);
            self.cache.push(entry);
            Ok(result)
        } else {
            self.stats.uncached_lookups += 1;
            if !self.validated_blocks.contains(&(layer, block)) {
                // Bounded first-touch validation prevents an unsorted block
                // from silently turning a present child into a pruned zero.
                // This scratch is independent of the resident field cache.
                let mut scratch = [0u8; 3 * 1024];
                let mut previous = None;
                let mut done = 0usize;
                while done < count {
                    let n = (count - done).min(scratch.len() / 3);
                    let index = meta
                        .chain_offset
                        .checked_add(done as u64)
                        .context("membership index overflow")?;
                    read_exact_at(
                        &self.files[layer].membership,
                        &mut scratch[..n * 3],
                        index.checked_mul(3).context("membership offset overflow")?,
                    )?;
                    self.stats.membership_bytes_read += (n * 3) as u64;
                    for raw in scratch[..n * 3].chunks_exact(3) {
                        let value = decode_u24(raw);
                        ensure!(
                            (value as usize) < self.q_index.states.len()
                                && previous.map_or(true, |p| p < value),
                            "invalid or unsorted membership qids at layer {layer} field {field}"
                        );
                        previous = Some(value);
                    }
                    done += n;
                }
                self.validated_blocks.insert((layer, block));
            }
            let (mut lo, mut hi) = (0usize, count);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let mut raw = [0; 3];
                let index = meta
                    .chain_offset
                    .checked_add(mid as u64)
                    .context("membership index overflow")?;
                read_exact_at(
                    &self.files[layer].membership,
                    &mut raw,
                    index.checked_mul(3).context("membership offset overflow")?,
                )?;
                self.stats.membership_bytes_read += 3;
                let value = decode_u24(&raw);
                ensure!(
                    (value as usize) < self.q_index.states.len(),
                    "membership qid {value} outside qstate index"
                );
                match value.cmp(&qid) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => {
                        let policy = if let Some(file) = &self.files[layer].policy {
                            let mut raw = [0; 2];
                            read_exact_at(
                                file,
                                &mut raw,
                                index.checked_mul(2).context("policy offset overflow")?,
                            )?;
                            self.stats.policy_bytes_read += 2;
                            Some(u16::from_le_bytes(raw))
                        } else {
                            None
                        };
                        return Ok(Some(policy));
                    }
                }
            }
            Ok(None)
        }
    }
}

fn decode_u24(raw: &[u8]) -> u32 {
    raw[0] as u32 | (raw[1] as u32) << 8 | (raw[2] as u32) << 16
}

fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<()> {
    let mut done = 0;
    while done < bytes.len() {
        let at = offset
            .checked_add(done as u64)
            .context("positional read offset overflow")?;
        match file.read_at(&mut bytes[done..], at) {
            Ok(0) => bail!("unexpected EOF in boundary compiler at byte {at}"),
            Ok(n) => done += n,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error).context("boundary compiler positional read"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_store::FieldChainWriter;
    use crate::qstate::{encode_avail_pair, encode_tail5, qchain_key, QChainState};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TinyBoundary {
        count: usize,
    }
    impl BoundaryPlacementIndex for TinyBoundary {
        fn terminal_count(&self) -> usize {
            self.count
        }
        fn placement_zid(&self, qid: u32, placed: u8) -> Option<u32> {
            Some(qid * PIECE_COUNT as u32 + placed as u32)
        }
    }

    struct Fixture {
        dir: PathBuf,
        graph: FieldGraph,
        qindex: QStateIndex,
        qfwd: QForwardGraph,
        pre: TinyBoundary,
        layers: Vec<FieldChainLayer>,
        members: Vec<Vec<Vec<u32>>>,
        codes: Vec<Vec<Vec<u16>>>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn fixture() -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "boundary-dag-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        // A deliberately small qforward subgraph. Every recorded child is an
        // actual queue/bag transition, checked independently by the compiler.
        let root = QChainState {
            avail: encode_avail_pair(0, 1),
            tail5: encode_tail5([0; 5]),
            bag_mask: 3,
        };
        let mut states = vec![root];
        let mut key_to_id = HashMap::from([(qchain_key(root), 0)]);
        let mut rows: Vec<[Vec<QForwardRec>; PIECE_COUNT]> =
            vec![std::array::from_fn(|_| Vec::new())];
        let mut frontier = vec![0u32];
        for depth in 0..10 {
            let mut next = Vec::new();
            for qid in frontier {
                let q = states[qid as usize];
                let (a, b) = decode_avail_pair(q.avail);
                let placed = if depth == 0 && a != b {
                    vec![a, b]
                } else {
                    vec![a]
                };
                let reveals: Vec<u8> = if depth == 0 {
                    (0..7).filter(|p| q.bag_mask & (1 << p) != 0).collect()
                } else {
                    vec![q.bag_mask.trailing_zeros() as u8]
                };
                for p in placed {
                    for &reveal in &reveals {
                        let child = q_forward_after_reveal(q, p, reveal).unwrap();
                        let key = qchain_key(child);
                        let child_qid = if let Some(&id) = key_to_id.get(&key) {
                            id as u32
                        } else {
                            let id = states.len();
                            states.push(child);
                            rows.push(std::array::from_fn(|_| Vec::new()));
                            key_to_id.insert(key, id);
                            id as u32
                        };
                        let rec = QForwardRec { reveal, child_qid };
                        if !rows[qid as usize][p as usize].contains(&rec) {
                            rows[qid as usize][p as usize].push(rec);
                        }
                        next.push(child_qid);
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            frontier = next;
        }
        let mut records = Vec::new();
        let mut qoffsets = Vec::new();
        for row in &mut rows {
            let mut off = [0; PIECE_COUNT + 1];
            off[0] = records.len() as u32;
            for p in 0..PIECE_COUNT {
                row[p].sort_unstable_by_key(|r| r.reveal);
                records.extend_from_slice(&row[p]);
                off[p + 1] = records.len() as u32;
            }
            qoffsets.push(off);
        }
        let qfwd = QForwardGraph {
            offsets: qoffsets,
            records,
            records_by_piece: [0; 7],
            max_records_per_piece: [7; 7],
            skipped_invalid_child: 0,
        };
        let qindex = QStateIndex {
            states,
            key_to_id,
            old_to_qid: Vec::new(),
            count_by_popcount: [0; 8],
        };
        let mut targets = Vec::new();
        let mut offsets = Vec::new();
        // Two independent paths: 0..9 stops at L4 into field5; 10..19
        // traverses all ten layers and stops at L9 into EMPTY_FIELD_ID.
        for field in 0..20 {
            let mut off = [0; 8];
            off[0] = targets.len() as u32;
            for p in 0..7 {
                let target = if field == 9 || field == 19 {
                    EMPTY_FIELD_ID
                } else {
                    field + 1
                };
                targets.extend_from_slice(&[target, target]);
                off[p + 1] = targets.len() as u32;
            }
            offsets.push(off);
        }
        let graph = FieldGraph {
            hashes: (0..20).map(|v| v as u64).collect(),
            offsets,
            targets,
        };
        let all: Vec<u32> = (0..qindex.states.len() as u32).collect();
        let mut members = vec![vec![all; 20]; 10];
        // One root has two reveal children, only one original member survives.
        let missing = qfwd.records_for(0, 0)[1].child_qid;
        members[1][11].retain(|&q| q != missing);
        // The L4 two-line edge is terminal even with no corresponding L5
        // field block. L9 likewise has no next layer at all.
        members[5][5].clear();
        let mut codes = vec![vec![Vec::new(); 20]; 10];
        let mut layers = Vec::new();
        for layer in 0..10 {
            let prefix = dir.join(format!("layer{layer}"));
            let mut writer = FieldChainWriter::create(&prefix).unwrap();
            let mut policy = Vec::new();
            for field in 0..20 {
                for &qid in &members[layer][field] {
                    writer
                        .push_pair_key((field as u64) << 32 | qid as u64)
                        .unwrap();
                    let (a, _) = decode_avail_pair(qindex.states[qid as usize].avail);
                    let code = (a as u16) << 8;
                    codes[layer][field].push(code);
                    policy.extend_from_slice(&code.to_le_bytes());
                }
            }
            writer.finish().unwrap();
            layers.push(FieldChainLayer::open_index_only(prefix).unwrap());
            fs::write(dir.join(format!("layer{layer}.policy.u16.bin")), policy).unwrap();
        }
        let pre = TinyBoundary {
            count: qindex.states.len() * PIECE_COUNT,
        };
        Fixture {
            dir,
            graph,
            qindex,
            qfwd,
            pre,
            layers,
            members,
            codes,
        }
    }

    fn limits(cache: usize) -> CompileLimits {
        CompileLimits {
            max_states: 50_000,
            max_nodes: 50_000,
            max_edges: 500_000,
            field_cache_bytes: cache,
        }
    }
    fn compile(
        f: &Fixture,
        mode: CompileMode,
        roots: &[StateKey],
        cache: usize,
    ) -> Result<CompiledDag> {
        Compiler::new(
            &f.graph,
            &f.qindex,
            &f.qfwd,
            &f.layers,
            &f.pre,
            Some(&f.dir),
            mode,
            limits(cache),
            5,
        )?
        .compile_roots(roots)
    }
    // Independent numerical Bellman recursion, without DAG or disk lookup.
    fn scalar(f: &Fixture, state: StateKey, u: &[f64], greedy: bool) -> f64 {
        let (layer, field, qid) = state;
        let Ok(rank) = f.members[layer][field as usize].binary_search(&qid) else {
            return 0.0;
        };
        let eval = |placed: u8, target: u32| {
            let recs = f.qfwd.records_for(qid, placed);
            if (layer == 4 && target == 5) || (layer == 9 && target == EMPTY_FIELD_ID) {
                u[qid as usize * 7 + placed as usize]
            } else {
                recs.iter()
                    .map(|r| scalar(f, (layer + 1, target, r.child_qid), u, greedy))
                    .sum::<f64>()
                    / recs.len() as f64
            }
        };
        if greedy {
            let mut best = 0.0;
            for p in 0..7u8 {
                if f.qfwd.records_for(qid, p).is_empty() {
                    continue;
                }
                for &target in f.graph.edges(field, p) {
                    if layer == 9 && target != EMPTY_FIELD_ID {
                        continue;
                    }
                    let value = eval(p, target);
                    if value > best {
                        best = value;
                    }
                }
            }
            best
        } else {
            let code = f.codes[layer][field as usize][rank];
            if code == POLICY_NONE {
                0.0
            } else {
                let p = (code >> 8) as u8;
                eval(p, f.graph.edges(field, p)[(code & 255) as usize])
            }
        }
    }

    #[test]
    fn ten_layer_fixed_and_greedy_match_independent_signed_reference() {
        let f = fixture();
        let roots = [(0, 10, 0), (0, 0, 0), (4, 4, 0), (9, 19, 0), (0, 10, 0)];
        let u: Vec<f64> = (0..f.pre.count)
            .map(|i| ((i % 17) as f64 - 12.0) / 8.0)
            .collect();
        for mode in [CompileMode::Fixed, CompileMode::Greedy] {
            for cache in [0, 128, 1 << 20] {
                let compiled = compile(&f, mode, &roots, cache).unwrap();
                let values = compiled.dag.eval_f64(&u).unwrap();
                for (&root, &state) in compiled.roots.iter().zip(&roots) {
                    assert_eq!(
                        values[root as usize],
                        scalar(&f, state, &u, mode == CompileMode::Greedy)
                    );
                }
                assert_eq!(compiled.roots[0], compiled.roots[4]);
                assert!(compiled.stats.missing_states > 0);
                assert!(compiled.stats.field_cache_peak_bytes <= cache);
                assert!(compiled.stats.states <= limits(cache).max_states);
                let floats: Vec<f32> = u.iter().map(|&v| v as f32).collect();
                let values32 = compiled.dag.eval_f32(&floats).unwrap();
                for &root in &compiled.roots {
                    assert_eq!(values32[root as usize] as f64, values[root as usize]);
                }
            }
        }
    }

    #[test]
    fn pruned_child_keeps_denominator_and_fixed_signed_values() {
        let f = fixture();
        let compiled = compile(
            &f,
            CompileMode::Fixed,
            &[(0, 10, 0), (4, 4, 0), (9, 19, 0)],
            0,
        )
        .unwrap();
        let values = compiled.dag.eval_f64(&vec![-2.0; f.pre.count]).unwrap();
        assert_eq!(values[compiled.roots[0] as usize], -1.0);
        assert_eq!(values[compiled.roots[1] as usize], -2.0);
        assert_eq!(values[compiled.roots[2] as usize], -2.0);
        assert!(compiled.stats.uncached_lookups > 0);
    }

    #[test]
    fn greedy_retains_zero_actions_and_uses_original_tie_order() {
        let f = fixture();
        let roots = [(4, 4, 0), (9, 19, 0)];
        let compiled = compile(&f, CompileMode::Greedy, &roots, 128).unwrap();
        let zero = compiled.dag.eval_f64(&vec![0.0; f.pre.count]).unwrap();
        assert!(compiled
            .greedy_policy_f64(&zero)
            .unwrap()
            .iter()
            .all(|(_, code)| *code == POLICY_NONE));
        assert!(compiled.actions.iter().all(|s| s.actions.len() == 4));
        let positive = compiled.dag.eval_f64(&vec![1.0; f.pre.count]).unwrap();
        assert!(compiled
            .greedy_policy_f64(&positive)
            .unwrap()
            .iter()
            .all(|(_, code)| *code == 0));
        for &root in &compiled.roots {
            assert_eq!(positive[root as usize], 1.0);
        }
    }

    fn replace_code(f: &mut Fixture, layer: usize, field: u32, qid: u32, code: u16) {
        let block = f.layers[layer]
            .metas
            .binary_search_by_key(&field, |m| m.field_id)
            .unwrap();
        let rank = f.members[layer][field as usize]
            .binary_search(&qid)
            .unwrap();
        f.codes[layer][field as usize][rank] = code;
        let path = f.dir.join(format!("layer{layer}.policy.u16.bin"));
        let mut bytes = fs::read(&path).unwrap();
        let offset = (f.layers[layer].metas[block].chain_offset as usize + rank) * 2;
        bytes[offset..offset + 2].copy_from_slice(&code.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn none_and_invalid_policy_codes_and_caps_are_checked() {
        let mut f = fixture();
        replace_code(&mut f, 0, 10, 0, POLICY_NONE);
        let compiled = compile(&f, CompileMode::Fixed, &[(0, 10, 0)], 0).unwrap();
        assert_eq!(compiled.roots, vec![compiled.dag.zero()]);
        for code in [7 << 8, 2, 6 << 8] {
            replace_code(&mut f, 0, 10, 0, code);
            assert!(compile(&f, CompileMode::Fixed, &[(0, 10, 0)], 128).is_err());
        }
        replace_code(&mut f, 0, 10, 0, 0);
        let tiny = CompileLimits {
            max_states: 1,
            ..limits(0)
        };
        assert!(Compiler::new(
            &f.graph,
            &f.qindex,
            &f.qfwd,
            &f.layers,
            &f.pre,
            Some(&f.dir),
            CompileMode::Fixed,
            tiny,
            5
        )
        .unwrap()
        .compile_roots(&[(0, 10, 0)])
        .is_err());
        let tiny = CompileLimits {
            max_edges: 1,
            ..limits(0)
        };
        assert!(Compiler::new(
            &f.graph,
            &f.qindex,
            &f.qfwd,
            &f.layers,
            &f.pre,
            None,
            CompileMode::Greedy,
            tiny,
            5
        )
        .unwrap()
        .compile_roots(&[(4, 4, 0)])
        .is_err());
        assert!(compile(&f, CompileMode::Fixed, &[(9, 0, 0)], 0).is_err());
    }

    #[test]
    fn corrupt_membership_and_reveals_are_rejected_in_both_cache_paths() {
        let mut f = fixture();
        let path = f.layers[0].chains_path.clone();
        let original = fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        corrupt[..6].rotate_left(3);
        fs::write(&path, corrupt).unwrap();
        for cap in [0, 1 << 20] {
            assert!(compile(&f, CompileMode::Fixed, &[(0, 0, 0)], cap).is_err());
        }
        fs::write(path, original).unwrap();
        f.qfwd.records[0].reveal = 6;
        assert!(compile(&f, CompileMode::Fixed, &[(4, 4, 0)], 0).is_err());
        f.layers[0].metas[0].field_id = 20;
        assert!(Compiler::new(
            &f.graph,
            &f.qindex,
            &f.qfwd,
            &f.layers,
            &f.pre,
            Some(&f.dir),
            CompileMode::Fixed,
            limits(0),
            5
        )
        .is_err());
    }
}
