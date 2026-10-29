use super::*;
use crate::layer_store::FieldChainWriter;
use crate::loop1_stream::{CompiledL9, Loop1Limits};
use crate::qgraph::QForwardRec;
use crate::qstate::{encode_avail_pair, QChainState, QStateIndex};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::AtomicUsize;

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

fn compiled_l9<'a>(
    fixture: &Fixture,
    graph: &'a FieldGraph,
    qindex: &'a QStateIndex,
    qfwd: &'a QForwardGraph,
    pre: &'a Formulas,
) -> CompiledL9<'a> {
    let mut compiled = CompiledL9::build(
        graph,
        qindex,
        qfwd,
        pre,
        &fixture.layers[9],
        &crate::mdp_policy_u16_path(&fixture.dir, 9),
        Loop1Limits::default(),
    )
    .unwrap();
    compiled.build_rank_index(1 << 20, 1 << 20).unwrap();
    compiled
}

fn terminal_leaves(table: &[[f32; PIECE_COUNT]]) -> Vec<f32> {
    vec![
        table[0][0],
        table[2][0],
        table[3][0],
        table[4][0],
        table[5][0],
    ]
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
                for (reveal, &child_qid) in pre.reveal_qids(zid).iter().enumerate() {
                    records.push(QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
                records_by_piece[placed] += pre.reveal_qids(zid).len() as u64;
                max_records_per_piece[placed] =
                    max_records_per_piece[placed].max(pre.reveal_qids(zid).len());
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
                    // Duplicate edge indices deliberately resolve to one field.
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
            "zxcl-sparse-test-{}-{}",
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
                let code = if qid == 4 || (depth == 7 && qid == 5) {
                    u16::MAX
                } else if depth == 4 && qid == 0 {
                    1 << 8
                } else if qid == 1 && depth != 9 {
                    1 // Same resolved target and formula as qid0, another edge.
                } else {
                    0
                };
                policy.push(code);
            }
            if depth == 5 {
                // Unreachable field, whose policy must still be validated.
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

fn terminal(qfwd: &QForwardGraph, gamma: f32, reward: bool) -> Vec<[f32; PIECE_COUNT]> {
    let reset = [16_777_216.0, 1.0, -16_777_216.0, -3.0, -0.0, 7.0];
    crate::build_terminal_reset_avg_table_opt(qfwd, &reset, gamma, QCOUNT, reward)
}

/// Independent scalar bottom-up recurrence, retaining all stored states. It
/// does not use the sparse evaluator, formula IDs, action caches or membership
/// binary search.
fn scalar_bottom_up(
    fixture: &Fixture,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; PIECE_COUNT]],
    gamma: f32,
) -> Vec<f32> {
    let mut next = BTreeMap::new();
    for depth in (0..10).rev() {
        let mut current = BTreeMap::new();
        let mut ordered = Vec::new();
        let layer = &fixture.layers[depth];
        for (block, meta) in layer.metas.iter().enumerate() {
            let qids = layer.read_block_chains(block);
            for (i, qid) in qids.into_iter().enumerate() {
                let code = fixture.codes[depth][meta.chain_offset as usize + i];
                let value = if code == u16::MAX {
                    0.0f32
                } else {
                    let placed = (code >> 8) as u8;
                    let target = graph.edges(meta.field_id, placed)[(code & 255) as usize];
                    if (depth == 4 && target == TWO_LINE)
                        || (depth == 9 && target == EMPTY_FIELD_ID)
                    {
                        terminal[qid as usize][placed as usize]
                    } else {
                        let records = qfwd.records_for(qid, placed);
                        let mut sum = 0.0f32;
                        for rec in records {
                            sum += next.get(&(target, rec.child_qid)).copied().unwrap_or(0.0);
                        }
                        (if gamma == 1.0 { sum } else { gamma * sum }) / records.len() as f32
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

/// Actual pre-existing fixed-policy kernel, ten layers, RAM-backed values.
fn production_bottom_up(
    fixture: &Fixture,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; PIECE_COUNT]],
    gamma: f32,
) -> Vec<f32> {
    let mut next: Option<Vec<f32>> = None;
    for depth in (0..10).rev() {
        let layer = &fixture.layers[depth];
        let mut output = vec![0.0f32; layer.total_states as usize];
        let policy = File::open(crate::mdp_policy_u16_path(&fixture.dir, depth)).unwrap();
        crate::compute_fixed_policy_chunk_f32(
            depth,
            0..layer.metas.len(),
            layer,
            &policy,
            None,
            Some(crate::SharedF32Out {
                ptr: output.as_mut_ptr(),
                len: output.len(),
            }),
            graph,
            qfwd,
            terminal,
            fixture.layers.get(depth + 1),
            None,
            next.as_deref(),
            gamma,
            0,
            QCOUNT,
            64,
            TWO_LINE,
        )
        .unwrap();
        next = Some(output);
    }
    next.unwrap()
}

fn assert_bits(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (i, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "f32 bits differ at {i}: {left} / {right}"
        );
    }
}

#[test]
fn all_ten_layers_match_scalar_and_existing_kernel_with_signed_values() {
    let fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert_eq!(
        policy.validate_all(false).unwrap(),
        fixture.layers.iter().map(|l| l.total_states).sum()
    );
    for gamma in [1.0, 0.625, 0.9375] {
        for reward in [false, true] {
            let terminal = terminal(&qfwd, gamma, reward);
            let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &terminal, gamma);
            assert_bits(
                &production_bottom_up(&fixture, &graph, &qfwd, &terminal, gamma),
                &expected,
            );
            for (workers, bytes) in [(1, 0), (1, 64), (1, 1 << 20), (4, 1 << 20)] {
                let result = policy
                    .apply(
                        &terminal,
                        gamma,
                        None,
                        SparseLimits {
                            workers,
                            total_memo_bytes: bytes,
                            progress: false,
                        },
                    )
                    .unwrap();
                assert_bits(&result.values, &expected);
                assert_eq!(result.stats.roots as usize, QCOUNT);
                assert!(result.stats.memo_payload_bytes <= bytes);
            }
        }
    }
}

#[test]
fn action_keys_share_only_exact_order_and_resolved_target() {
    let fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    policy.validate_all(false).unwrap();
    let result = policy
        .apply(
            &terminal(&qfwd, 1.0, false),
            1.0,
            None,
            SparseLimits {
                workers: 1,
                total_memo_bytes: 1 << 20,
                progress: false,
            },
        )
        .unwrap();
    assert!(result.stats.action_hits > 0);
    assert!(result.values.iter().any(|v| *v < 0.0));
    assert_eq!(result.values[4].to_bits(), 0.0f32.to_bits());
    // The two ordered cancellation expressions actually differ in this fixture.
    let terminal = terminal(&qfwd, 1.0, false);
    assert_ne!(terminal[0][0].to_bits(), terminal[2][0].to_bits());
}

#[test]
fn terminal_values_remain_raw_per_qid_even_when_formula_ids_are_shared() {
    let fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    policy.validate_all(false).unwrap();
    let mut table = vec![[0.0f32; PIECE_COUNT]; QCOUNT];
    for (qid, row) in table.iter_mut().enumerate() {
        for (placed, value) in row.iter_mut().enumerate() {
            *value = (qid * PIECE_COUNT + placed) as f32 - 17.0;
        }
    }
    table[0][1] = -0.0;
    assert_ne!(table[0][0], table[1][0]);
    for gamma in [1.0, 0.9375] {
        let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &table, gamma);
        assert_bits(
            &production_bottom_up(&fixture, &graph, &qfwd, &table, gamma),
            &expected,
        );
        let result = policy
            .apply(
                &table,
                gamma,
                None,
                SparseLimits {
                    workers: 1,
                    total_memo_bytes: 1 << 20,
                    progress: false,
                },
            )
            .unwrap();
        assert_bits(&result.values, &expected);
    }
}

#[test]
fn compiled_l9_cut_matches_raw_sparse_and_original_ten_layer_kernel() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qindex = qindex();
    let qfwd = qforward(&pre);
    let compiled = compiled_l9(&fixture, &graph, &qindex, &qfwd, &pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    for reward in [false, true] {
        let table = terminal(&qfwd, 1.0, reward);
        let leaves = terminal_leaves(&table);
        let source = compiled.prepare_f32(&leaves).unwrap();
        let expected = production_bottom_up(&fixture, &graph, &qfwd, &table, 1.0);
        assert_bits(
            &scalar_bottom_up(&fixture, &graph, &qfwd, &table, 1.0),
            &expected,
        );
        for (workers, bytes) in [(1, 0), (1, 64), (1, 1 << 20), (4, 1 << 20)] {
            let limits = SparseLimits {
                workers,
                total_memo_bytes: bytes,
                progress: false,
            };
            let raw = policy.apply(&table, 1.0, None, limits).unwrap();
            let result = policy
                .apply_impl(&table, 1.0, None, limits, Some(source))
                .unwrap();
            assert_bits(&result.values, &raw.values);
            assert_bits(&result.values, &expected);
            assert!(result.stats.compiled_l9_averages > 0);
            assert!(result.stats.compiled_l9_reveals > 0);
            assert_eq!(result.stats.compiled_l9_membership_bytes, 0);
            assert!(result.stats.memo_payload_bytes <= bytes);
            if bytes == 0 {
                assert!(result.stats.evaluated_states < raw.stats.evaluated_states);
                assert!(result.stats.state_requests < raw.stats.state_requests);
            }
        }
    }
}

#[test]
fn compiled_cut_still_requires_full_validation_and_rejects_gamma_or_other_sources() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qindex = qindex();
    let qfwd = qforward(&pre);
    let compiled = compiled_l9(&fixture, &graph, &qindex, &qfwd, &pre);
    let table = terminal(&qfwd, 1.0, true);
    let leaves = terminal_leaves(&table);
    let source = compiled.prepare_f32(&leaves).unwrap();
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    let limits = SparseLimits {
        workers: 1,
        total_memo_bytes: 0,
        progress: false,
    };
    assert!(policy
        .apply_impl(&table, 1.0, None, limits, Some(source))
        .is_err());
    assert!(policy
        .apply_impl(&table, 0.9375, Some(1), limits, Some(source))
        .is_err());
    policy.validate_all(false).unwrap();
    assert!(policy
        .apply_impl(&table, 1.0, None, limits, Some(source))
        .is_ok());

    // Equal contents and dimensions do not authorize a different fixed policy.
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
    assert!(other_policy
        .apply_impl(&table, 1.0, Some(1), limits, Some(source))
        .is_err());
    let other_graph = graph.clone();
    let other_domain = SparsePolicy::open(
        &other_graph,
        &qfwd,
        &pre,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(other_domain
        .apply_impl(&table, 1.0, Some(1), limits, Some(source))
        .is_err());
}

#[test]
fn compiled_cut_rejects_l9_mutation_and_a_second_apply_uses_fresh_values() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qindex = qindex();
    let qfwd = qforward(&pre);
    let compiled = compiled_l9(&fixture, &graph, &qindex, &qfwd, &pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    let limits = SparseLimits {
        workers: 1,
        total_memo_bytes: 1 << 20,
        progress: false,
    };
    for reward in [true, false, true] {
        let table = terminal(&qfwd, 1.0, reward);
        let leaves = terminal_leaves(&table);
        let source = compiled.prepare_f32(&leaves).unwrap();
        let result = policy
            .apply_impl(&table, 1.0, None, limits, Some(source))
            .unwrap();
        assert_bits(
            &result.values,
            &production_bottom_up(&fixture, &graph, &qfwd, &table, 1.0),
        );
    }
    let table = terminal(&qfwd, 1.0, true);
    let leaves = terminal_leaves(&table);
    let source = compiled.prepare_f32(&leaves).unwrap();
    let path = crate::mdp_policy_u16_path(&fixture.dir, 9);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, bytes).unwrap(); // Same-size/same-tick writes must reject.
    assert!(policy
        .apply_impl(&table, 1.0, Some(1), limits, Some(source))
        .is_err());
}

#[test]
fn full_validation_rejects_unreachable_bad_policy_and_all_roots_need_it() {
    let mut fixture = Fixture::new();
    let last = fixture.codes[5].len() - 1;
    fixture.rewrite_code(5, last, 5 << 8);
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    let table = terminal(&qfwd, 1.0, true);
    let limits = SparseLimits {
        workers: 1,
        total_memo_bytes: 0,
        progress: false,
    };
    assert!(policy.apply(&table, 1.0, None, limits).is_err());
    let reachable = policy.apply(&table, 1.0, Some(1), limits).unwrap();
    assert_eq!(reachable.values.len(), 1);
    assert!(!policy.fully_validated());
    assert!(policy.validate_all(false).is_err());
    assert!(!policy.fully_validated());
}

#[test]
fn source_changes_and_replacements_invalidate_an_existing_policy() {
    let mut fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    {
        let policy = SparsePolicy::open(
            &graph,
            &qfwd,
            &formulas,
            &fixture.layers,
            &fixture.dir,
            TWO_LINE,
        )
        .unwrap();
        let path = crate::mdp_policy_u16_path(&fixture.dir, 3);
        let old = fs::read(&path).unwrap();
        fs::write(&path, &old).unwrap();
        assert!(policy.check_inputs_unchanged().is_err());
        assert!(policy
            .apply(
                &terminal(&qfwd, 1.0, true),
                1.0,
                Some(1),
                SparseLimits {
                    workers: 1,
                    total_memo_bytes: 0,
                    progress: false
                }
            )
            .is_err());
    }
    fixture.rewrite_code(3, 0, 0);
    let policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    let path = crate::mdp_policy_u16_path(&fixture.dir, 2);
    let replacement = fixture.dir.join("replacement");
    fs::write(&replacement, fs::read(&path).unwrap()).unwrap();
    fs::rename(replacement, &path).unwrap();
    assert!(policy.check_inputs_unchanged().is_err());
}

#[test]
fn changed_symlink_binding_and_membership_order_are_rejected() {
    let fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let path = crate::mdp_policy_u16_path(&fixture.dir, 2);
    let target_a = fixture.dir.join("target-a");
    let target_b = fixture.dir.join("target-b");
    fs::rename(&path, &target_a).unwrap();
    fs::write(&target_b, fs::read(&target_a).unwrap()).unwrap();
    std::os::unix::fs::symlink(&target_a, &path).unwrap();
    let policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(&target_b, &path).unwrap();
    assert!(policy.check_inputs_unchanged().is_err());
    drop(policy);
    let chains = &fixture.layers[0].chains_path;
    let mut bytes = fs::read(chains).unwrap();
    bytes[3..6].copy_from_slice(&[0, 0, 0]); // duplicate root qid
    fs::write(chains, bytes).unwrap();
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(policy.validate_all(false).is_err());
    assert!(policy
        .apply(
            &terminal(&qfwd, 1.0, true),
            1.0,
            Some(1),
            SparseLimits {
                workers: 1,
                total_memo_bytes: 0,
                progress: false
            }
        )
        .is_err());
}

#[test]
fn malformed_graph_placement_and_l9_target_fail_checked_validation() {
    let mut fixture = Fixture::new();
    let mut graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    graph.offsets[1][2] = u32::MAX;
    assert!(SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE
    )
    .is_err());
    let graph = super::tests::graph();
    let mut bad = qfwd.clone();
    bad.records[0].child_qid = 1;
    assert!(SparsePolicy::open(
        &graph,
        &bad,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE
    )
    .is_err());
    fixture.rewrite_code(9, 0, (1 << 8) | 1);
    let mut policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(policy.validate_all(false).is_err());
}

#[test]
fn caps_invalid_terminals_and_empty_root_limit_are_checked() {
    let fixture = Fixture::new();
    let graph = graph();
    let formulas = formulas();
    let qfwd = qforward(&formulas);
    let policy = SparsePolicy::open(
        &graph,
        &qfwd,
        &formulas,
        &fixture.layers,
        &fixture.dir,
        TWO_LINE,
    )
    .unwrap();
    let table = terminal(&qfwd, 1.0, true);
    for limits in [
        SparseLimits {
            workers: 0,
            total_memo_bytes: 0,
            progress: false,
        },
        SparseLimits {
            workers: 5,
            total_memo_bytes: 0,
            progress: false,
        },
        SparseLimits {
            workers: 1,
            total_memo_bytes: MAX_MEMO_BYTES + 1,
            progress: false,
        },
    ] {
        assert!(policy.apply(&table, 1.0, Some(1), limits).is_err());
    }
    let limits = SparseLimits {
        workers: 1,
        total_memo_bytes: 0,
        progress: false,
    };
    assert!(policy.apply(&table, f32::NAN, Some(1), limits).is_err());
    let mut bad = table.clone();
    bad[5][6] = f32::NAN;
    assert!(policy.apply(&bad, 1.0, Some(1), limits).is_err());
    assert!(policy
        .apply(&table, 1.0, Some(0), limits)
        .unwrap()
        .values
        .is_empty());
}

#[test]
fn direct_map_collisions_only_evict_and_cached_negative_zero_is_exact() {
    let mut cache = DirectCache::new(size_of::<Slot>() as u64).unwrap();
    assert!(!cache.insert(10, -0.0));
    assert_eq!(cache.get(10).unwrap().to_bits(), (-0.0f32).to_bits());
    assert!(cache.insert(11, -9.0));
    assert!(cache.get(10).is_none());
    assert_eq!(cache.get(11), Some(-9.0));
    let mut zero = DirectCache::new(0).unwrap();
    assert!(!zero.insert(1, -3.0));
    assert!(zero.get(1).is_none());
}

#[test]
fn full_validation_checks_none_qids_and_unavailable_pieces_with_valid_graph_edges() {
    for (code, message) in [
        (7 << 8, "invalid policy piece"),
        (255, "invalid policy edge"),
    ] {
        let mut fixture = Fixture::new();
        fixture.rewrite_code(5, 0, code);
        let graph = graph();
        let pre = formulas();
        let qfwd = qforward(&pre);
        let mut policy =
            SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE)
                .unwrap();
        assert!(policy
            .validate_all(false)
            .unwrap_err()
            .to_string()
            .contains(message));
        assert!(!policy.fully_validated());
    }
    // A NONE policy does not exempt even an unreachable qid from the domain
    // check. Change the one-state unreachable block before capturing inputs.
    {
        let fixture = Fixture::new();
        let graph = graph();
        let pre = formulas();
        let qfwd = qforward(&pre);
        let layer = &fixture.layers[5];
        let meta = layer.metas.iter().find(|meta| meta.field_id == 10).unwrap();
        let at = meta.chain_offset as usize * 3;
        let mut bytes = fs::read(&layer.chains_path).unwrap();
        bytes[at..at + 3].copy_from_slice(&[QCOUNT as u8, 0, 0]);
        fs::write(&layer.chains_path, bytes).unwrap();
        let mut policy =
            SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE)
                .unwrap();
        let error = policy.validate_all(false).unwrap_err();
        assert!(error.to_string().contains("invalid membership"));
        assert!(!policy.fully_validated());
    }

    // Piece 2 has a real field edge, but the quotient state cannot place it.
    // This specifically exercises the precomputed availability mask rather
    // than failing the simpler graph-degree or piece-domain checks.
    let mut fixture = Fixture::new();
    let mut graph = graph();
    let inserted = graph.offsets[5][3] as usize;
    graph.targets.insert(inserted, 6);
    for (field, row) in graph.offsets.iter_mut().enumerate() {
        for (boundary, offset) in row.iter_mut().enumerate() {
            if field > 5 || (field == 5 && boundary > 2) {
                *offset += 1;
            }
        }
    }
    fixture.rewrite_code(5, 0, 2 << 8);
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    assert_eq!(graph.edges(5, 2), &[6]);
    let error = policy.validate_all(false).unwrap_err();
    assert!(error.to_string().contains("no reveal records"));
    assert!(!policy.fully_validated());
}

#[test]
fn validation_ranges_cover_empty_small_and_large_field_sequences_exactly_once() {
    assert!(validation_ranges(&[]).is_empty());
    let mut offset = 0u64;
    let metas: Vec<_> = (0..3075u32)
        .map(|field_id| {
            let count = if field_id == 2048 {
                5 << 20
            } else if field_id % 3 == 0 {
                0
            } else {
                3
            };
            let meta = FieldChainBlockMeta {
                field_id,
                count,
                chain_offset: offset,
            };
            offset += u64::from(count);
            meta
        })
        .collect();
    let ranges = validation_ranges(&metas);
    let actual: Vec<_> = ranges.iter().cloned().flatten().collect();
    assert_eq!(actual, (0..metas.len()).collect::<Vec<_>>());
    assert!(ranges.len() >= 4);
    assert!(ranges
        .iter()
        .all(|range| !range.is_empty() && range.len() <= 1024));
    let oversized = ranges.iter().find(|range| range.contains(&2048)).unwrap();
    assert_eq!(oversized.end, 2049);
}

#[test]
fn repeated_full_validation_clears_certificate_before_detecting_changed_inputs() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    assert!(policy.fully_validated());
    let path = crate::mdp_policy_u16_path(&fixture.dir, 5);
    fs::write(&path, fs::read(&path).unwrap()).unwrap();
    assert!(policy.validate_all(false).is_err());
    assert!(!policy.fully_validated());
}

#[test]
fn read_only_page_retirement_checks_ranges_and_preserves_concurrent_refaults() {
    let fixture = Fixture::new();
    let path = fixture.dir.join("readonly-page-retirement.bin");
    let expected: Vec<u8> = (0..(1 << 20)).map(|i| (i * 17 + i / 4096) as u8).collect();
    fs::write(&path, &expected).unwrap();
    let mapped = BoundMap::open(&path, expected.len() as u64).unwrap();
    assert_eq!(mapped.bytes(), expected.as_slice());
    mapped.release_pages(17, 64 << 10).unwrap();
    let boundary = SOURCE_SCALAR_READ_CHARGE_BYTES as usize;
    let straddling = expected[boundary - 1..boundary + 2].to_vec();
    mapped
        .release_pages_with_fringe(0, SOURCE_SCALAR_READ_CHARGE_BYTES)
        .unwrap();
    assert_eq!(
        &mapped.bytes()[boundary - 1..boundary + 2],
        straddling.as_slice()
    );
    mapped.release_pages(expected.len() as u64, 0).unwrap();
    assert!(mapped.release_pages(expected.len() as u64, 1).is_err());
    assert!(mapped.release_pages(u64::MAX, 2).is_err());
    std::thread::scope(|scope| {
        for lane in 0..4usize {
            let mapped = &mapped;
            let expected = &expected;
            scope.spawn(move || {
                for iteration in 0..16usize {
                    mapped.release_pages(0, expected.len() as u64).unwrap();
                    for index in (lane + iteration..expected.len()).step_by(4093) {
                        assert_eq!(mapped.bytes()[index], expected[index]);
                    }
                }
            });
        }
    });
    mapped.check().unwrap();
    assert_eq!(mapped.bytes(), expected.as_slice());
    assert_eq!(fs::read(path).unwrap(), expected);
}

#[test]
fn numeric_source_window_deduplicates_repeated_bucket_without_changing_values() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    policy.validate_all(false).unwrap();
    assert!(policy.release_source_range(10, 0, 0).is_err());
    assert!(policy
        .release_source_range(0, policy.root_count(), 1)
        .is_err());
    policy
        .release_source_range(0, policy.root_count(), 0)
        .unwrap();
    let table = terminal(&qfwd, 1.0, true);
    let progress = Progress::default();
    let mut worker = Worker {
        policy: &policy,
        terminal: &table,
        gamma: 1.0,
        caches: WorkerCaches::new(0).unwrap(),
        stats: SparseStats::default(),
        published: SparseStats::default(),
        progress: &progress,
        compiled_l9: None,
        source_touch_window: SourceTouchWindow::new().unwrap(),
    };
    let reads = 2 * SOURCE_RESIDENCY_BUDGET_BYTES / SOURCE_SCALAR_READ_CHARGE_BYTES + 1;
    for _ in 0..reads {
        worker
            .record_source_read(0, SourceKind::Membership, 0)
            .unwrap();
    }
    assert_eq!(
        worker.source_touch_window.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    assert_eq!(worker.source_touch_window.flush_count(), 0);
    assert_eq!(worker.source_touch_window.record_count(), reads);
    assert_eq!(worker.source_touch_window.unique_record_count(), 1);
    assert_eq!(worker.source_touch_window.dedup_hit_count(), reads - 1);
    assert_eq!(
        worker.source_touch_window.record_count(),
        worker.source_touch_window.unique_record_count()
            + worker.source_touch_window.dedup_hit_count()
    );
    worker.source_touch_window.flush(&policy).unwrap();
    assert_eq!(worker.source_touch_window.flush_count(), 1);
    assert_eq!(worker.source_touch_window.range_count(), 1);
    assert_eq!(worker.source_touch_window.pending_charge_bytes(), 0);
    worker
        .record_source_read(0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(worker.source_touch_window.unique_record_count(), 2);
    assert_eq!(
        worker.source_touch_window.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    let expected = scalar_bottom_up(&fixture, &graph, &qfwd, &table, 1.0);
    for (qid, expected) in expected.iter().enumerate() {
        assert_eq!(
            worker.eval(0, 0, qid as u32).unwrap().to_bits(),
            expected.to_bits()
        );
    }
    policy.release_source_pages().unwrap();
    policy.check_inputs_unchanged().unwrap();
}

#[test]
fn targeted_source_window_deduplicates_within_maps_and_rejects_invalid_offsets() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    let mut window = SourceTouchWindow::new().unwrap();
    assert_eq!(window.pending_charge_bytes(), 0);
    window.flush(&policy).unwrap();
    assert_eq!(window.flush_count(), 0);
    assert!(window
        .record(&policy, 10, SourceKind::Membership, 0)
        .is_err());
    assert!(window
        .record(&policy, 0, SourceKind::Membership, policy.root_count() * 3)
        .is_err());
    window
        .record(&policy, 0, SourceKind::Membership, 0)
        .unwrap();
    window
        .record(&policy, 0, SourceKind::Membership, 3)
        .unwrap();
    window.record(&policy, 0, SourceKind::Policy, 0).unwrap();
    window
        .record(&policy, 1, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(
        window.pending_charge_bytes(),
        3 * SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    let storage = window.storage_bytes();
    window.flush(&policy).unwrap();
    assert_eq!(window.flush_count(), 1);
    assert_eq!(window.record_count(), 4);
    assert_eq!(window.unique_record_count(), 3);
    assert_eq!(window.dedup_hit_count(), 1);
    assert_eq!(window.range_count(), 3); // Membership/policy and layers never merge.
    assert_eq!(window.pending_charge_bytes(), 0);
    assert_eq!(window.storage_bytes(), storage);
    assert!(window.retire_seconds().is_finite());
    window
        .record(&policy, 0, SourceKind::Membership, 0)
        .unwrap();
    let other =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    assert!(window.flush(&other).is_err());
    assert!(window.record(&other, 0, SourceKind::Membership, 0).is_err());
    assert!(window
        .record(&policy, 10, SourceKind::Membership, 0)
        .is_err());
    assert!(window
        .record(
            &policy,
            0,
            SourceKind::Membership,
            policy.root_count() * 3 - 1
        )
        .is_err());
    assert_eq!(
        window.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    window.flush(&policy).unwrap();
    policy.check_inputs_unchanged().unwrap();
}

#[test]
fn unique_source_window_rollover_keeps_exact_capacity_and_clears_seen_state() {
    let fixture = Fixture::new();
    let graph = graph();
    let pre = formulas();
    let qfwd = qforward(&pre);
    let mut policy =
        SparsePolicy::open(&graph, &qfwd, &pre, &fixture.layers, &fixture.dir, TWO_LINE).unwrap();
    // This test exercises only source-window geometry. A sparse read-only file
    // provides 2,049 valid bucket offsets without generating or solving MDP
    // states and without filling 256 MiB of disk or anonymous memory.
    let path = fixture.dir.join("source-window-geometry.bin");
    let size = (SourceTouchWindow::MAX_RECORDS as u64 + 1) * SOURCE_SCALAR_READ_CHARGE_BYTES + 3;
    File::create(&path).unwrap().set_len(size).unwrap();
    policy.layers[0].membership = BoundMap::open(&path, size).unwrap();
    let mut window = SourceTouchWindow::new().unwrap();
    let storage = window.storage_bytes();
    for bucket in 0..SourceTouchWindow::MAX_RECORDS as u64 {
        window
            .record(
                &policy,
                0,
                SourceKind::Membership,
                bucket * SOURCE_SCALAR_READ_CHARGE_BYTES,
            )
            .unwrap();
    }
    assert_eq!(window.buckets.len(), 2048);
    assert_eq!(
        window.seen.iter().filter(|&&key| key != EMPTY_KEY).count(),
        2048
    );
    assert_eq!(window.pending_charge_bytes(), SOURCE_RESIDENCY_BUDGET_BYTES);
    assert_eq!(window.flush_count(), 0);
    // A non-last duplicate in a full window must find the exact set entry,
    // rather than flushing or losing the bucket because of a hash collision.
    window
        .record(&policy, 0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(window.flush_count(), 0);
    assert_eq!(window.dedup_hit_count(), 1);
    let next = SourceTouchWindow::MAX_RECORDS as u64 * SOURCE_SCALAR_READ_CHARGE_BYTES;
    window
        .record(&policy, 0, SourceKind::Membership, next)
        .unwrap();
    assert_eq!(window.flush_count(), 1);
    assert_eq!(window.range_count(), 1); // Adjacent buckets coalesce in one map.
    assert_eq!(window.buckets.len(), 1);
    assert_eq!(
        window.seen.iter().filter(|&&key| key != EMPTY_KEY).count(),
        1
    );
    assert_eq!(
        window.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    window
        .record(&policy, 0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(window.unique_record_count(), 2050);
    assert_eq!(window.buckets.len(), 2); // The pre-flush seen key was cleared.
    assert_eq!(window.storage_bytes(), storage);
    window.flush(&policy).unwrap();
    assert!(window.seen.iter().all(|&key| key == EMPTY_KEY));
    assert_eq!(window.last_bucket, EMPTY_KEY);
    window
        .record(&policy, 0, SourceKind::Membership, 0)
        .unwrap();
    assert_eq!(window.unique_record_count(), 2051);
    assert_eq!(
        window.pending_charge_bytes(),
        SOURCE_SCALAR_READ_CHARGE_BYTES
    );
    assert_eq!(
        SourceTouchWindow::MAX_RECORDS as u64 * 3 * SOURCE_SCALAR_READ_CHARGE_BYTES,
        768 << 20
    );
    assert_eq!(
        MAX_WORKERS as u64
            * SourceTouchWindow::MAX_RECORDS as u64
            * 3
            * SOURCE_SCALAR_READ_CHARGE_BYTES,
        3 << 30
    );
}

#[test]
fn exact_seen_set_distinguishes_forced_hash_collisions_and_bounds_probes() {
    let mut window = SourceTouchWindow::new().unwrap();
    let key = 0u64; // Map 0, bucket 0 is a valid key, not the empty sentinel.
    let (_, slot) = window.seen_slot(key).unwrap();
    let collision = (1u64..1_000_000)
        .find(|&candidate| hash(candidate) as usize & (SourceTouchWindow::SEEN_SLOTS - 1) == slot)
        .expect("bounded collision fixture");
    window.seen[slot] = key;
    let (present, collision_slot) = window.seen_slot(collision).unwrap();
    assert!(!present);
    assert_ne!(slot, collision_slot);
    window.seen[collision_slot] = collision;
    assert_eq!(window.seen_slot(key).unwrap(), (true, slot));
    assert_eq!(window.seen_slot(collision).unwrap(), (true, collision_slot));
    // A corrupted/full table must return an error instead of probing forever.
    window.seen.fill(0);
    assert!(window.seen_slot(u64::MAX - 1).is_err());
}
