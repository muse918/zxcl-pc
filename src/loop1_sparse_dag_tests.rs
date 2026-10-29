use super::super::SparsePolicy;
use super::*;
use crate::graph::{FieldGraph, EMPTY_FIELD_ID, PIECE_COUNT};
use crate::layer_store::{FieldChainLayer, FieldChainWriter};
use crate::loop1_stream::{CompiledL9, Loop1Limits, TerminalFormulaIndex};
use crate::qgraph::{QForwardGraph, QForwardRec};
use crate::qstate::{encode_avail_pair, QChainState, QStateIndex};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

const QCOUNT: usize = 6;
const TWO_LINE: u32 = 11;
static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Formulas {
    children: Vec<Vec<u32>>,
}

impl TerminalFormulaIndex for Formulas {
    fn terminal_count(&self) -> usize {
        self.children.len()
    }

    fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        if qid as usize >= QCOUNT || placed > 1 {
            return None;
        }
        if qid == 0 && placed == 1 {
            return Some(2);
        }
        Some(match qid {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => 4,
        })
    }

    fn reveal_qids(&self, zid: u32) -> &[u32] {
        &self.children[zid as usize]
    }
}

fn formulas() -> Formulas {
    Formulas {
        children: vec![vec![0, 1, 2], vec![2, 0, 1], vec![3, 4], vec![4], vec![5]],
    }
}

fn qforward(pre: &Formulas) -> QForwardGraph {
    let mut offsets = Vec::new();
    let mut records = Vec::new();
    let mut records_by_piece = [0u64; PIECE_COUNT];
    let mut max_records_per_piece = [0usize; PIECE_COUNT];
    for qid in 0..QCOUNT as u32 {
        let mut row = [records.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if let Some(zid) = pre.zid_for_placement(qid, placed as u8) {
                let children = pre.reveal_qids(zid);
                for (reveal, &child_qid) in children.iter().enumerate() {
                    records.push(QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
                records_by_piece[placed] += children.len() as u64;
                max_records_per_piece[placed] = max_records_per_piece[placed].max(children.len());
            }
            row[placed + 1] = records.len() as u32;
        }
        offsets.push(row);
    }
    QForwardGraph {
        offsets,
        records,
        records_by_piece,
        max_records_per_piece,
        skipped_invalid_child: 0,
    }
}

fn graph() -> FieldGraph {
    let mut offsets = Vec::new();
    let mut targets = Vec::new();
    for field in 0..12u32 {
        let mut row = [targets.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if placed < 2 && field < 10 {
                if field == 9 {
                    targets.push(EMPTY_FIELD_ID);
                    if placed == 1 {
                        targets.push(0);
                    }
                } else if field == 4 && placed == 1 {
                    targets.push(TWO_LINE);
                } else {
                    // Distinct policy edge indices resolve to one field.
                    targets.extend_from_slice(&[field + 1, field + 1]);
                }
            }
            row[placed + 1] = targets.len() as u32;
        }
        offsets.push(row);
    }
    FieldGraph {
        hashes: (0..12).collect(),
        offsets,
        targets,
    }
}

struct Fixture {
    dir: PathBuf,
    layers: Vec<FieldChainLayer>,
    codes: Vec<Vec<u16>>,
}

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "zxcl-whole-dag-test-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let mut layers = Vec::new();
        let mut codes = Vec::new();
        for depth in 0..10 {
            let prefix = dir.join(format!("layer{depth}q_pruned"));
            let mut writer = FieldChainWriter::create(&prefix).unwrap();
            let mut policy = Vec::new();
            for qid in 0..QCOUNT as u32 {
                if (depth == 6 && qid == 2) || (depth == 8 && qid == 1) || (depth == 9 && qid == 2)
                {
                    continue;
                }
                writer
                    .push_pair_key(((depth as u64) << 32) | u64::from(qid))
                    .unwrap();
                policy.push(if qid == 4 || (depth == 7 && qid == 5) {
                    u16::MAX
                } else if depth == 4 && qid == 0 {
                    1 << 8
                } else if qid == 1 && depth != 9 {
                    1
                } else {
                    0
                });
            }
            if depth == 0 || depth == 5 {
                // A second root field tests file order. At L5 this field is
                // unreachable, so validating only traversed states is weaker.
                writer.push_pair_key((10u64 << 32) | 4).unwrap();
                policy.push(u16::MAX);
            }
            writer.finish().unwrap();
            fs::write(
                crate::mdp_policy_u16_path(&dir, depth),
                policy
                    .iter()
                    .flat_map(|code| code.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            layers.push(FieldChainLayer::open_index_only(prefix).unwrap());
            codes.push(policy);
        }
        Self { dir, layers, codes }
    }

    fn rewrite_code(&mut self, depth: usize, position: usize, code: u16) {
        self.codes[depth][position] = code;
        fs::write(
            crate::mdp_policy_u16_path(&self.dir, depth),
            self.codes[depth]
                .iter()
                .flat_map(|code| code.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn limits(bytes: u64) -> WholeDagLimits {
    WholeDagLimits {
        total_memo_bytes: bytes,
        max_nodes: 100_000,
        max_edges: 500_000,
        progress: false,
    }
}

/// Build the terminal leaves independently, preserving sum(reward + x) order.
fn terminals(pre: &Formulas, reset: &[f32; QCOUNT], reward: bool) -> Vec<f32> {
    pre.children
        .iter()
        .map(|children| {
            let mut sum = 0.0f32;
            for &child in children {
                sum += if reward {
                    1.0 + reset[child as usize]
                } else {
                    reset[child as usize]
                };
            }
            sum / children.len() as f32
        })
        .collect()
}

/// Independent bottom-up scalar recurrence over every stored state. It uses
/// neither the sparse evaluator nor DAG nodes, memo keys, or membership search.
fn scalar_bottom_up(
    fixture: &Fixture,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    pre: &Formulas,
    terminals: &[f32],
) -> Vec<f32> {
    scalar_bottom_up_with_order(fixture, graph, qfwd, pre, terminals, false)
}

fn scalar_bottom_up_with_order(
    fixture: &Fixture,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    pre: &Formulas,
    terminals: &[f32],
    reverse_children: bool,
) -> Vec<f32> {
    let mut next = BTreeMap::new();
    for depth in (0..10).rev() {
        let mut current = BTreeMap::new();
        let mut ordered = Vec::new();
        let layer = &fixture.layers[depth];
        for (block, meta) in layer.metas.iter().enumerate() {
            for (i, qid) in layer.read_block_chains(block).into_iter().enumerate() {
                let code = fixture.codes[depth][meta.chain_offset as usize + i];
                let value = if code == u16::MAX {
                    0.0f32
                } else {
                    let placed = (code >> 8) as u8;
                    let target = graph.edges(meta.field_id, placed)[(code & 255) as usize];
                    if (depth == 4 && target == TWO_LINE)
                        || (depth == 9 && target == EMPTY_FIELD_ID)
                    {
                        terminals[pre.zid_for_placement(qid, placed).unwrap() as usize]
                    } else {
                        let records = qfwd.records_for(qid, placed);
                        let mut sum = 0.0f32;
                        for at in 0..records.len() {
                            let rec = &records[if reverse_children {
                                records.len() - at - 1
                            } else {
                                at
                            }];
                            sum += next.get(&(target, rec.child_qid)).copied().unwrap_or(0.0);
                        }
                        sum / records.len() as f32
                    }
                };
                current.insert((meta.field_id, qid), value);
                ordered.push(value);
            }
        }
        if depth == 0 {
            return ordered;
        }
        next = current;
    }
    unreachable!()
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f32 bits at root {i}: {a} / {b}");
    }
}

#[test]
fn all_roots_match_independent_scalar_for_reused_reward_and_signed_leaves() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    let resets = [
        [16_777_216.0, 1.0, -16_777_216.0, -3.0, -0.0, 7.0],
        [-9.0, 0.5, 13.0, -11.0, -0.0, 4.0],
        [0.0, -0.0, 0.0, -0.0, -0.0, -0.0],
        [1.0, 1.0, 1.0, -33_554_432.0, 0.0, -3.0],
    ];
    for bytes in [0, 16, 64, 4096] {
        let plan = WholeFixedPlan::compile(&policy, None, limits(bytes), None).unwrap();
        assert_eq!(plan.root_count(), QCOUNT + 1);
        assert!(plan.stats.memo_payload_bytes <= bytes);
        assert!(plan.stats.nodes > 1 && plan.stats.edges > 0);
        for reset in resets {
            for reward in [false, true] {
                let leaves = terminals(&pre, &reset, reward);
                assert_bits(
                    &plan.apply(&leaves).unwrap(),
                    &scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves),
                );
            }
        }
    }
}

#[test]
fn tiny_memos_evict_without_changing_expressions_or_root_file_order() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    let leaves = terminals(
        &pre,
        &[16_777_216.0, 1.0, -16_777_216.0, -3.0, -0.0, 7.0],
        false,
    );
    assert_ne!(leaves[0].to_bits(), leaves[1].to_bits());
    let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves);
    let uncached = WholeFixedPlan::compile(&policy, None, limits(0), None).unwrap();
    let tiny = WholeFixedPlan::compile(&policy, None, limits(64), None).unwrap();
    let cached = WholeFixedPlan::compile(&policy, None, limits(4096), None).unwrap();
    assert_bits(&uncached.apply(&leaves).unwrap(), &expected);
    assert_bits(&tiny.apply(&leaves).unwrap(), &expected);
    assert_eq!(uncached.stats.nodes, tiny.stats.nodes);
    assert_eq!(uncached.stats.edges, tiny.stats.edges);
    assert_eq!(uncached.stats.nodes, cached.stats.nodes);
    assert_eq!(uncached.stats.edges, cached.stats.edges);
    assert!(tiny.stats.state_collisions + tiny.stats.action_collisions > 0);
    assert!(cached.stats.action_hits > 0);
    assert!(tiny.stats.missing_children > 0);
    assert_eq!(expected[4].to_bits(), 0.0f32.to_bits());
    assert_eq!(expected[QCOUNT].to_bits(), 0.0f32.to_bits());
    for count in [0, 1, 3, QCOUNT + 1, QCOUNT + 100] {
        let prefix = WholeFixedPlan::compile(&policy, Some(count), limits(64), None).unwrap();
        let n = count.min(expected.len());
        assert_eq!(prefix.root_count(), n);
        assert_bits(&prefix.apply(&leaves).unwrap(), &expected[..n]);
    }
}

#[test]
fn coherent_cancellation_leaves_detect_reordered_nonterminal_sums() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    let leaves = terminals(&pre, &[1.0, 1.0, 1.0, -33_554_432.0, 0.0, -3.0], false);
    let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves);
    let reordered = scalar_bottom_up_with_order(&fixture, &graph, &qfwd, &pre, &leaves, true);
    // These terminal leaves come from one reset vector. The difference is
    // caused by the nonterminal sums that the DAG compiler constructs.
    assert_ne!(expected[0].to_bits(), reordered[0].to_bits());
    assert_eq!(expected[0].to_bits(), (-5_592_405.5f32).to_bits());
    for bytes in [0, 16, 64, 4096] {
        let plan = WholeFixedPlan::compile(&policy, None, limits(bytes), None).unwrap();
        assert_bits(&plan.apply(&leaves).unwrap(), &expected);
    }
}

#[test]
fn nonterminal_empty_field_target_is_missing_membership_and_positive_zero() {
    let fixture = Fixture::new();
    let mut graph = graph();
    graph.targets[graph.offsets[0][0] as usize] = EMPTY_FIELD_ID;
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    let leaves = terminals(&pre, &[-9.0, 0.5, 13.0, -11.0, -0.0, 4.0], false);
    let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves);
    assert_eq!(expected[0].to_bits(), 0.0f32.to_bits());
    for bytes in [0, 64] {
        let plan = WholeFixedPlan::compile(&policy, None, limits(bytes), None).unwrap();
        assert!(plan.stats.missing_children >= 3);
        assert_bits(&plan.apply(&leaves).unwrap(), &expected);
    }
}

#[test]
fn all_root_compilation_requires_complete_policy_validation() {
    let mut fixture = Fixture::new();
    let unreachable = fixture.codes[5].len() - 1;
    fixture.rewrite_code(5, unreachable, 5 << 8);
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    assert!(WholeFixedPlan::compile(&policy, None, limits(64), None).is_err());
    assert!(WholeFixedPlan::compile(&policy, Some(1), limits(64), None).is_ok());
    assert!(!policy.fully_validated());
    assert!(policy.validate_all(false).is_err());
    assert!(!policy.fully_validated());
    assert!(WholeFixedPlan::compile(&policy, None, limits(64), None).is_err());
}

#[test]
fn invalid_reachable_policy_and_membership_fail_during_bounded_compile() {
    for malformed_policy in [true, false] {
        let mut fixture = Fixture::new();
        if malformed_policy {
            fixture.rewrite_code(0, 0, 6 << 8);
        } else {
            let path = &fixture.layers[0].chains_path;
            let mut bytes = fs::read(path).unwrap();
            bytes[3..6].copy_from_slice(&[0, 0, 0]);
            fs::write(path, bytes).unwrap();
        }
        let graph = graph();
        let pre = formulas();
        let qfwd = qforward(&pre);
        let policy =
            SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE)
                .unwrap();
        assert!(WholeFixedPlan::compile(&policy, Some(1), limits(64), None).is_err());
    }
}

#[test]
fn source_writes_and_replacements_reject_compile_and_existing_plan_apply() {
    for replace in [false, true] {
        for membership in [false, true] {
            let fixture = Fixture::new();
            let graph = graph();
            let pre = formulas();
            let qfwd = qforward(&pre);
            let policy =
                SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE)
                    .unwrap();
            let plan = WholeFixedPlan::compile(&policy, Some(2), limits(64), None).unwrap();
            let path = if membership {
                fixture.layers[6].chains_path.clone()
            } else {
                crate::mdp_policy_u16_path(&fixture.dir, 3)
            };
            let bytes = fs::read(&path).unwrap();
            if replace {
                let replacement = fixture.dir.join("replacement");
                fs::write(&replacement, bytes).unwrap();
                fs::rename(&replacement, &path).unwrap();
            } else {
                // Equal bytes still represent a forbidden source mutation.
                fs::write(&path, bytes).unwrap();
            }
            assert!(WholeFixedPlan::compile(&policy, Some(2), limits(64), None).is_err());
            assert!(plan.apply(&[1.0; 5]).is_err());
        }
    }
}

#[test]
fn malformed_formula_order_is_rejected_before_action_sharing() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let mut qfwd = qforward(&pre);
    qfwd.records[0].child_qid = 1;
    qfwd.records[1].child_qid = 0;
    assert!(
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE,).is_err()
    );
}

#[test]
fn node_and_edge_caps_fail_without_mutating_sources() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    for (max_nodes, max_edges) in [(0, 100), (1, 100), (100, 0), (100, 1)] {
        let mut capped = limits(64);
        capped.max_nodes = max_nodes;
        capped.max_edges = max_edges;
        assert!(WholeFixedPlan::compile(&policy, None, capped, None).is_err());
        policy.check_inputs_unchanged().unwrap();
    }
    assert!(WholeFixedPlan::compile(&policy, None, limits(64), None).is_ok());
}

#[test]
fn finite_terminal_contract_rejects_wrong_length_and_nonfinite_unused_leaves() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    let plan = WholeFixedPlan::compile(&policy, Some(1), limits(64), None).unwrap();
    assert!(plan.apply(&[]).is_err());
    assert!(plan.apply(&[1.0; 6]).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut leaves = [1.0; 5];
        leaves[4] = invalid;
        assert!(plan.apply(&leaves).is_err());
    }
    assert!(plan.apply(&[-1.0, 2.0, -0.0, 4.0, 0.0]).is_ok());
}

fn qindex() -> QStateIndex {
    QStateIndex {
        states: (0..QCOUNT)
            .map(|qid| QChainState {
                avail: encode_avail_pair(0, 1),
                tail5: qid as u32,
                bag_mask: 1,
            })
            .collect(),
        key_to_id: HashMap::new(),
        old_to_qid: Vec::new(),
        count_by_popcount: [0; 8],
    }
}

#[test]
fn compiled_l9_tail_matches_independent_all_root_results_and_checks_domain() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let qindex = qindex();
    let mut compiled = CompiledL9::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &fixture.layers[9],
        &crate::mdp_policy_u16_path(&fixture.dir, 9),
        Loop1Limits::default(),
    )
    .unwrap();
    compiled.build_rank_index(1 << 20, 1 << 20).unwrap();
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    assert!(WholeFixedPlan::compile(&policy, None, limits(64), Some(&compiled)).is_err());
    policy.validate_all(false).unwrap();
    let plan = WholeFixedPlan::compile(&policy, None, limits(64), Some(&compiled)).unwrap();
    assert!(plan.stats.tail_lookups > 0);
    for reward in [false, true] {
        let leaves = terminals(
            &pre,
            &[16_777_216.0, 1.0, -16_777_216.0, -3.0, -0.0, 7.0],
            reward,
        );
        assert_bits(
            &plan.apply(&leaves).unwrap(),
            &scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves),
        );
    }
    let other_fixture = Fixture::new();
    let other_policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &pre,
        &other_fixture.layers,
        &other_fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(WholeFixedPlan::compile(&other_policy, Some(1), limits(64), Some(&compiled)).is_err());
    let cloned_graph = graph.clone();
    let other_domain = SparsePolicy::open(
        &cloned_graph,
        &qfwd,
        &pre,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(WholeFixedPlan::compile(&other_domain, Some(1), limits(64), Some(&compiled)).is_err());
    let path = crate::mdp_policy_u16_path(&fixture.dir, 9);
    fs::write(&path, fs::read(&path).unwrap()).unwrap();
    assert!(plan.apply(&[1.0; 5]).is_err());
}

#[test]
fn bounded_source_read_window_releases_without_changing_inputs_or_expressions() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    let mut builder = Builder::new(&policy, limits(0), None).unwrap();
    assert!(SOURCE_SCALAR_READ_CHARGE_BYTES > 0);
    assert!(SOURCE_SCALAR_READ_CHARGE_BYTES <= SOURCE_RESIDENCY_BUDGET_BYTES);
    let accesses = SOURCE_RESIDENCY_BUDGET_BYTES / SOURCE_SCALAR_READ_CHARGE_BYTES;
    for _ in 0..accesses {
        builder
            .record_source_read(0, SourceKind::Membership, 0)
            .unwrap();
        assert!(builder.source_pages.pending_charge_bytes() <= SOURCE_RESIDENCY_BUDGET_BYTES);
    }
    // Repeated accesses to one bucket retain one bounded residency charge.
    // The cumulative read-charge statistic is work accounting, not the
    // current unique-bucket residency budget.
    assert_eq!(builder.stats.source_release_calls, 0);
    assert_eq!(builder.stats.source_window_flushes, 0);
    builder
        .record_source_read(0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(builder.stats.source_release_calls, 0);
    assert_eq!(builder.stats.source_window_flushes, 0);
    assert_eq!(
        builder.source_pages.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    assert!(builder.stats.source_retire_seconds.is_finite());
    assert!(builder.stats.source_read_charge_bytes > SOURCE_RESIDENCY_BUDGET_BYTES);
    // Even the last-bucket fast path must reject a truncated u24 read whose
    // start is in that same valid bucket.
    let end = policy.layers[0].membership.bytes().len() as u64;
    assert!(builder
        .record_source_read(0, SourceKind::Membership, end - 1)
        .is_err());
    // A distinct mapped source has a distinct bucket even at the same offset.
    builder
        .record_source_read(0, SourceKind::Policy, 0)
        .unwrap();
    assert_eq!(
        builder.source_pages.pending_charge_bytes(),
        2 * SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    // Bounds must be checked before a same-bucket hit can bypass insertion.
    assert!(builder
        .record_source_read(0, SourceKind::Membership, end)
        .is_err());
    // A first-touch scan independently releases its contiguous range, even
    // when it has not consumed the random-access charging window.
    assert_eq!(builder.find(0, 0, 0).unwrap(), Some(0));
    assert_eq!(builder.stats.source_range_release_calls, 1);
    builder.flush_source_window().unwrap();
    assert_eq!(builder.source_pages.pending_charge_bytes(), 0);
    assert_eq!(builder.stats.source_window_flushes, 1);
    assert_eq!(builder.stats.source_release_calls, 0);
    // A completed flush resets the exact seen set and last-bucket fast path.
    builder
        .record_source_read(0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(
        builder.source_pages.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    builder.flush_source_window().unwrap();
    assert_eq!(builder.stats.source_window_flushes, 2);
    policy.check_inputs_unchanged().unwrap();
    let plan = WholeFixedPlan::compile(&policy, Some(QCOUNT), limits(64), None).unwrap();
    assert!(plan.stats.source_release_calls >= 2);
    assert!(plan.stats.source_range_release_calls > 0);
    let leaves = terminals(&pre, &[1.0, 1.0, 1.0, -33_554_432.0, 0.0, -3.0], true);
    let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &pre, &leaves);
    assert_bits(&plan.apply(&leaves).unwrap(), &expected[..QCOUNT]);
    policy.release_source_pages().unwrap();
    assert_bits(&plan.apply(&leaves).unwrap(), &expected[..QCOUNT]);
    policy.check_inputs_unchanged().unwrap();
}
