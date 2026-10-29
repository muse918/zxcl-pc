use crate::loop1_stream::{CompiledL8, CompiledL9, Loop1Limits, TerminalFormulaIndex};
use crate::{
    compute_fixed_policy_chunk_f32, encode_policy_u16,
    graph::FieldGraph,
    layer_store::{FieldChainLayer, FieldChainWriter},
    qgraph::{QForwardGraph, QForwardRec},
    qstate::{encode_avail_pair, QChainState, QStateIndex},
    GreedyEmitScratch, SharedF32Out, EMPTY_FIELD_ID, PIECE_COUNT, POLICY_NONE_U16,
};
use std::{collections::HashMap, env, fs, ops::Range, path::Path};

const QCOUNT: usize = 6;

struct TinyTerminalIndex {
    placement: Vec<[Option<u32>; 2]>,
    children: Vec<Vec<u32>>,
}

impl TerminalFormulaIndex for TinyTerminalIndex {
    fn terminal_count(&self) -> usize {
        self.children.len()
    }

    fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        *self.placement.get(qid as usize)?.get(placed as usize)?
    }

    fn reveal_qids(&self, zid: u32) -> &[u32] {
        &self.children[zid as usize]
    }
}

fn terminal_index() -> TinyTerminalIndex {
    TinyTerminalIndex {
        placement: vec![
            [Some(0), Some(1)],
            [Some(2), Some(3)],
            [Some(4), Some(5)],
            [Some(6), Some(1)],
            [None, Some(7)],
            [None, None],
        ],
        // The first and third lists have identical children in different
        // orders. Their f32 averages must retain that order.
        children: vec![
            vec![0, 1, 2],
            vec![1, 3],
            vec![2, 0, 1],
            vec![3, 4],
            vec![2],
            vec![0, 4, 2],
            vec![4, 1],
            vec![4],
        ],
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

fn qforward(pre: &TinyTerminalIndex) -> QForwardGraph {
    let mut offsets = Vec::new();
    let mut records = Vec::new();
    let mut records_by_piece = [0; PIECE_COUNT];
    let mut max_records_per_piece = [0; PIECE_COUNT];
    for qid in 0..QCOUNT {
        let mut row = [records.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if let Some(zid) = pre.zid_for_placement(qid as u32, placed as u8) {
                let children = pre.reveal_qids(zid);
                records_by_piece[placed] += children.len() as u64;
                max_records_per_piece[placed] = max_records_per_piece[placed].max(children.len());
                for (reveal, &child_qid) in children.iter().enumerate() {
                    records.push(QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
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
    let rows: [[&[u32]; 2]; 7] = [
        [&[3, 4, 3], &[4, 5]],
        [&[4], &[3, 5]],
        [&[6], &[4]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID, 0]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
        [&[EMPTY_FIELD_ID, EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
        [&[], &[]],
    ];
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    for row in rows {
        let mut offsets_row = [targets.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if placed < 2 {
                targets.extend_from_slice(row[placed]);
            }
            offsets_row[placed + 1] = targets.len() as u32;
        }
        offsets.push(offsets_row);
    }
    FieldGraph {
        hashes: (0..rows.len() as u64).collect(),
        offsets,
        targets,
    }
}

fn layer(prefix: &Path, blocks: &[(u32, &[u32])]) -> FieldChainLayer {
    let mut writer = FieldChainWriter::create(prefix).unwrap();
    for &(field, qids) in blocks {
        for &qid in qids {
            writer
                .push_pair_key((field as u64) << 32 | qid as u64)
                .unwrap();
        }
    }
    writer.finish().unwrap();
    FieldChainLayer::open_index_only(prefix).unwrap()
}

fn l9_layer(prefix: &Path) -> FieldChainLayer {
    layer(
        prefix,
        &[
            (3, &[0, 1, 2, 3, 4, 5]),
            (4, &[0, 2, 4]),
            (5, &[0, 1, 3, 5]),
        ],
    )
}

fn l8_layer(prefix: &Path) -> FieldChainLayer {
    layer(
        prefix,
        &[
            (0, &[0, 1, 2, 3, 4, 5]),
            (1, &[0, 1, 2, 3, 4, 5]),
            (2, &[0, 1, 2, 3, 4, 5]),
        ],
    )
}

fn l9_codes() -> Vec<u16> {
    vec![
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 1),
        POLICY_NONE_U16,
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
    ]
}

fn l8_codes() -> Vec<u16> {
    vec![
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 1),
        encode_policy_u16(1, 1),
        POLICY_NONE_U16,
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        encode_policy_u16(1, 1),
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
    ]
}

fn write_policy(path: &Path, codes: &[u16]) {
    fs::write(
        path,
        codes
            .iter()
            .flat_map(|code| code.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

fn limits(cache: usize) -> Loop1Limits {
    Loop1Limits {
        max_mapping_bytes: 1 << 20,
        max_unique_averages: 1 << 10,
        max_cache_bytes: cache,
    }
}

fn terminal_values(pre: &TinyTerminalIndex, reset: &[f32], reward: f32) -> Vec<f32> {
    pre.children
        .iter()
        .map(|children| {
            let mut sum = 0.0f32;
            for &qid in children {
                sum += reward + reset[qid as usize];
            }
            sum / children.len() as f32
        })
        .collect()
}

fn terminal_table(pre: &TinyTerminalIndex, dense: &[f32]) -> Vec<[f32; PIECE_COUNT]> {
    (0..QCOUNT)
        .map(|qid| {
            let mut row = [0.0; PIECE_COUNT];
            for (placed, value) in row.iter_mut().enumerate() {
                if let Some(zid) = pre.zid_for_placement(qid as u32, placed as u8) {
                    *value = dense[zid as usize];
                }
            }
            row
        })
        .collect()
}

fn legacy_values(
    layer_number: usize,
    range: Range<usize>,
    cur: &FieldChainLayer,
    policy_path: &Path,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; PIECE_COUNT]],
    next: Option<(&FieldChainLayer, &[f32])>,
) -> Vec<f32> {
    let policy = fs::File::open(policy_path).unwrap();
    let mut output = vec![f32::NAN; cur.total_states as usize];
    let out = SharedF32Out {
        ptr: output.as_mut_ptr(),
        len: output.len(),
    };
    compute_fixed_policy_chunk_f32(
        layer_number,
        range.clone(),
        cur,
        &policy,
        None,
        Some(out),
        graph,
        qfwd,
        terminal,
        next.map(|(membership, _)| membership),
        None,
        next.map(|(_, values)| values),
        1.0,
        0,
        QCOUNT,
        0,
        0,
    )
    .unwrap();
    let start = cur.metas[range.start].chain_offset as usize;
    let last = &cur.metas[range.end - 1];
    let end = (last.chain_offset + last.count as u64) as usize;
    output[start..end].to_vec()
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "state {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn streamed_l9_and_l8_match_legacy_ordered_f32_for_signed_reset_and_reward_zero() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("l9-policy");
    let p8 = owned.path.join("l8-policy");
    write_policy(&p9, &l9_codes());
    write_policy(&p8, &l8_codes());
    for cache in [16, 64, 1 << 20] {
        let compiled9 =
            CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(cache)).unwrap();
        for range in [0..3, 1..3, 1..2] {
            let compiled8 = CompiledL8::build(
                &graph,
                &qindex,
                &qfwd,
                &pre,
                &l8,
                &p8,
                range.clone(),
                &compiled9,
                limits(cache),
            )
            .unwrap();
            assert!(compiled8.stats().cache_peak_bytes <= cache);
            assert_eq!(compiled8.state_offset(), (range.start * QCOUNT) as u64);
            assert_eq!(compiled8.range(), range);
            for reset in [
                [16_777_216.0, -16_777_216.0, 1.0, -0.0, -1.25, -1.0],
                [-2.5, -0.0, 0.5, 3.25, -4.0, 1.0],
            ] {
                for reward in [0.0, 1.0] {
                    let terminal = terminal_values(&pre, &reset, reward);
                    let table = terminal_table(&pre, &terminal);
                    let legacy9 =
                        legacy_values(9, 0..l9.metas.len(), &l9, &p9, &graph, &qfwd, &table, None);
                    assert_bits(&compiled9.evaluate_f32(&terminal).unwrap(), &legacy9);
                    let legacy8 = legacy_values(
                        8,
                        range.clone(),
                        &l8,
                        &p8,
                        &graph,
                        &qfwd,
                        &table,
                        Some((&l9, &legacy9)),
                    );
                    assert_bits(&compiled8.evaluate_f32(&terminal).unwrap(), &legacy8);
                    let mut output = vec![f32::NAN; 3];
                    let mut scratch = vec![f32::NAN; 2];
                    compiled8
                        .evaluate_into_f32(&terminal, &mut output, &mut scratch)
                        .unwrap();
                    assert_bits(&output, &legacy8);
                }
            }
        }
        assert!(compiled9.stats().cache_peak_bytes <= cache);
        let expected_reads = if cache == 16 { 5 } else { 3 };
        assert_eq!(compiled9.stats().membership_read_calls, expected_reads);
        assert_eq!(compiled9.stats().policy_read_calls, expected_reads);
    }
}

#[test]
fn missing_children_and_none_retain_the_full_reveal_denominator_and_resolved_targets() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &l9_codes());
    write_policy(&p8, &l8_codes());
    let compiled9 = CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(64)).unwrap();
    let compiled8 = CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        limits(64),
    )
    .unwrap();
    let terminal = [3.0, 12.0, 99.0, 6.0, 9.0, 99.0, -4.0, -2.0];
    let output = compiled8.evaluate_f32(&terminal).unwrap();
    // Both actions encode p0/e0. Resolving the edge in each parent field
    // produces two different ordered leaf lists.
    assert_eq!(output[0].to_bits(), 3.0f32.to_bits()); // (3 + 6 + NONE) / 3
    assert_eq!(output[6].to_bits(), 7.0f32.to_bits()); // (12 + missing + 9) / 3
    assert_ne!(output[0].to_bits(), output[6].to_bits());
    assert_eq!(output[5].to_bits(), 0.0f32.to_bits());
    assert_eq!(output[11].to_bits(), 0.0f32.to_bits());
    assert_eq!(output[12].to_bits(), 0.0f32.to_bits()); // Target field has no L9 block.
    assert_eq!(output[14].to_bits(), 0.0f32.to_bits());
    assert_eq!(output[15].to_bits(), 0.0f32.to_bits());
    assert_eq!(output[17].to_bits(), 0.0f32.to_bits());
    assert_eq!(compiled9.used_terminal_zids(), &[0, 1, 3, 4, 6, 7]);
    assert!(compiled8.stats().unique_averages > 1);
    let first = &compiled8.averages()[compiled8.state_ids()[0] as usize - 1];
    assert_eq!(first.denominator, 3);
    assert_eq!(&first.children[..3], &[1, 4, 0]);
    let second = &compiled8.averages()[compiled8.state_ids()[6] as usize - 1];
    assert_eq!(second.denominator, 3);
    assert_eq!(&second.children[..3], &[2, 0, 5]);
    assert_ne!(compiled8.state_ids()[0], compiled8.state_ids()[1]);
    let ordered_terminal = [3.0, 16_777_216.0, 99.0, 6.0, -16_777_216.0, 99.0, -4.0, 1.0];
    let ordered_output = compiled8.evaluate_f32(&ordered_terminal).unwrap();
    // field0/q2 visits child qids [0,4,2], hence terminal zids [1,7,4].
    // Sorting those zids would change (2^24 + 1 - 2^24) / 3 from 0 to 1/3.
    assert_eq!(ordered_output[2].to_bits(), 0.0f32.to_bits());
}

#[test]
fn terminal_leaves_preserve_signed_zero_and_evaluation_validates_input() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &l9_codes());
    write_policy(&p8, &l8_codes());
    let compiled9 = CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(64)).unwrap();
    let compiled8 = CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        limits(64),
    )
    .unwrap();
    let terminal = [-0.0; 8];
    let values9 = compiled9.evaluate_f32(&terminal).unwrap();
    assert_eq!(values9[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(values9[2].to_bits(), 0.0f32.to_bits());
    assert!(compiled9.evaluate_f32(&terminal[..7]).is_err());
    assert!(compiled8.evaluate_f32(&terminal[..7]).is_err());
    let mut invalid = terminal;
    invalid[0] = f32::NAN;
    assert!(compiled9.evaluate_f32(&invalid).is_err());
    assert!(compiled8.evaluate_f32(&invalid).is_err());
    invalid[0] = f32::INFINITY;
    assert!(compiled9.evaluate_f32(&invalid).is_err());
    assert!(compiled8.evaluate_f32(&invalid).is_err());
}

#[test]
fn invalid_l9_policy_actions_are_rejected_even_when_their_value_would_be_zero() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("p9");
    for (state, code) in [
        (0, 0x0700),
        (0, 0xff00),
        (0, encode_policy_u16(0, 3)),
        (4, encode_policy_u16(0, 0)), // qid4 has no p0 reveal records.
        (5, encode_policy_u16(1, 0)), // qid5 has no reveal records at all.
        (0, encode_policy_u16(1, 1)), // The selected L9 edge is nonterminal.
    ] {
        let mut codes = l9_codes();
        codes[state] = code;
        write_policy(&policy, &codes);
        assert!(
            CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).is_err(),
            "state {state} code {code:#06x}"
        );
    }
    fs::write(&policy, [0u8]).unwrap();
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).is_err());
}

#[test]
fn invalid_l8_policy_actions_and_terminal_formula_disagreements_are_rejected() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &l9_codes());
    let compiled9 = CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(64)).unwrap();
    for (state, code) in [
        (0, 0x0700),
        (0, 0xff00),
        (0, encode_policy_u16(0, 3)),
        (4, encode_policy_u16(0, 0)),
        (5, encode_policy_u16(1, 0)),
    ] {
        let mut codes = l8_codes();
        codes[state] = code;
        write_policy(&p8, &codes);
        assert!(
            CompiledL8::build(
                &graph,
                &qindex,
                &qfwd,
                &pre,
                &l8,
                &p8,
                0..3,
                &compiled9,
                limits(64)
            )
            .is_err(),
            "state {state} code {code:#06x}"
        );
    }
    let mut wrong = terminal_index();
    wrong.children[0].swap(0, 1);
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &wrong, &l9, &p9, limits(64)).is_err());
    wrong = terminal_index();
    wrong.placement[0][0] = None;
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &wrong, &l9, &p9, limits(64)).is_err());
    wrong = terminal_index();
    wrong.placement[0][0] = Some(99);
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &wrong, &l9, &p9, limits(64)).is_err());
}

#[test]
fn every_membership_qid_is_validated_including_none_policies() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    let valid_l9 = l9_layer(&owned.path.join("valid-l9"));
    write_policy(&p9, &l9_codes());
    let compiled9 =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &valid_l9, &p9, limits(64)).unwrap();
    for (name, raw) in [
        ("duplicate", [0, 0, 0, 0, 0, 0]),
        ("descending", [1, 0, 0, 0, 0, 0]),
        ("outside-domain", [0, 0, 0, QCOUNT as u8, 0, 0]),
    ] {
        for layer_number in [8, 9] {
            let prefix = owned.path.join(format!("{name}-{layer_number}"));
            let membership = layer(&prefix, &[(if layer_number == 9 { 3 } else { 0 }, &[0, 1])]);
            fs::write(&membership.chains_path, raw).unwrap();
            let membership = FieldChainLayer::open_index_only(&prefix).unwrap();
            let policy = if layer_number == 9 { &p9 } else { &p8 };
            // Use a separate L9 policy path so the already compiled parent
            // remains correctly bound while testing malformed L8 membership.
            let bad_policy = policy.with_extension(format!("{name}-{layer_number}"));
            write_policy(&bad_policy, &[POLICY_NONE_U16; 2]);
            let rejected = if layer_number == 9 {
                CompiledL9::build(
                    &graph,
                    &qindex,
                    &qfwd,
                    &pre,
                    &membership,
                    &bad_policy,
                    limits(64),
                )
                .is_err()
            } else {
                CompiledL8::build(
                    &graph,
                    &qindex,
                    &qfwd,
                    &pre,
                    &membership,
                    &bad_policy,
                    0..1,
                    &compiled9,
                    limits(64),
                )
                .is_err()
            };
            assert!(rejected, "{name} layer {layer_number}");
        }
    }
}

#[test]
fn mapping_average_and_cache_limits_fail_before_unbounded_construction() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &l9_codes());
    write_policy(&p8, &l8_codes());
    let mut tiny = limits(64);
    tiny.max_mapping_bytes = 1;
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, tiny).is_err());
    let mut tiny = limits(0);
    tiny.max_mapping_bytes = 1 << 20;
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, tiny).is_err());
    let compiled9 = CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(64)).unwrap();
    let mut exact = limits(64);
    exact.max_mapping_bytes = 13 * 4;
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, exact).is_ok());
    exact.max_mapping_bytes -= 1;
    assert!(CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, exact).is_err());
    exact.max_mapping_bytes = (13 + 18) * 4;
    let exact8 = CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        exact,
    )
    .unwrap();
    exact.max_unique_averages = exact8.stats().unique_averages;
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        exact
    )
    .is_ok());
    exact.max_unique_averages -= 1;
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        exact
    )
    .is_err());
    exact.max_unique_averages += 1;
    exact.max_mapping_bytes -= 1;
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        exact
    )
    .is_err());
    let mut tiny = limits(64);
    tiny.max_mapping_bytes = 1;
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        tiny
    )
    .is_err());
    let mut tiny = limits(64);
    tiny.max_unique_averages = 1;
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        tiny
    )
    .is_err());
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..3,
        &compiled9,
        limits(0)
    )
    .is_err());
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        0..4,
        &compiled9,
        limits(64)
    )
    .is_err());
}

#[test]
fn construction_statistics_count_membership_and_policy_bytes_without_evaluation_reads() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let l8 = l8_layer(&owned.path.join("l8"));
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &l9_codes());
    write_policy(&p8, &l8_codes());
    let compiled9 =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(8192)).unwrap();
    let compiled8 = CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &pre,
        &l8,
        &p8,
        1..3,
        &compiled9,
        limits(8192),
    )
    .unwrap();
    assert_eq!(compiled9.stats().scanned_states, 13);
    assert_eq!(compiled9.stats().membership_bytes_read, 13 * 3);
    assert_eq!(compiled9.stats().policy_bytes_read, 13 * 2);
    assert!(compiled9.stats().membership_read_calls > 0);
    assert!(compiled9.stats().policy_read_calls > 0);
    assert_eq!(compiled9.stats().unique_terminal_leaves, 6);
    assert_eq!(compiled8.stats().scanned_states, 12);
    assert!(compiled8.stats().membership_bytes_read >= 12 * 3);
    assert!(compiled8.stats().policy_bytes_read >= 12 * 2);
    assert!(compiled8.stats().cache_peak_bytes <= 8192);
    let before = (
        compiled8.stats().membership_bytes_read,
        compiled8.stats().policy_bytes_read,
        compiled8.stats().membership_read_calls,
        compiled8.stats().policy_read_calls,
    );
    for _ in 0..3 {
        compiled8.evaluate_f32(&[1.0; 8]).unwrap();
    }
    assert_eq!(
        before,
        (
            compiled8.stats().membership_bytes_read,
            compiled8.stats().policy_bytes_read,
            compiled8.stats().membership_read_calls,
            compiled8.stats().policy_read_calls,
        )
    );
}

#[cfg(unix)]
#[test]
fn evaluation_rejects_in_place_mutation_replacement_and_symlink_retargeting() {
    use std::os::unix::fs::symlink;

    for source in ["p9", "l9-chains", "l9-index", "p8", "l8-chains", "l8-index"] {
        for mutation in ["write", "replace", "retarget"] {
            let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
            let graph = graph();
            let qindex = qindex();
            let pre = terminal_index();
            let qfwd = qforward(&pre);
            let l9 = l9_layer(&owned.path.join("l9"));
            let l8 = l8_layer(&owned.path.join("l8"));
            let p9 = owned.path.join("p9");
            let p8 = owned.path.join("p8");
            write_policy(&p9, &l9_codes());
            write_policy(&p8, &l8_codes());
            let watched = match source {
                "p9" => &p9,
                "l9-chains" => &l9.chains_path,
                "l9-index" => &l9.index_path,
                "p8" => &p8,
                "l8-chains" => &l8.chains_path,
                "l8-index" => &l8.index_path,
                _ => unreachable!(),
            };
            let original = fs::read(watched).unwrap();
            if mutation == "retarget" {
                let target = owned.path.join("original-target");
                fs::rename(watched, &target).unwrap();
                symlink(&target, watched).unwrap();
            }
            let compiled9 =
                CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &p9, limits(64)).unwrap();
            let compiled8 = CompiledL8::build(
                &graph,
                &qindex,
                &qfwd,
                &pre,
                &l8,
                &p8,
                0..3,
                &compiled9,
                limits(64),
            )
            .unwrap();
            match mutation {
                "write" => {
                    let mut changed = original;
                    changed[0] ^= 1;
                    fs::write(watched, changed).unwrap();
                }
                "replace" => {
                    let replacement = owned.path.join("replacement");
                    fs::write(&replacement, original).unwrap();
                    fs::rename(replacement, watched).unwrap();
                }
                "retarget" => {
                    let replacement = owned.path.join("replacement");
                    fs::write(&replacement, original).unwrap();
                    fs::remove_file(watched).unwrap();
                    symlink(&replacement, watched).unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                compiled8.evaluate_f32(&[1.0; 8]).is_err(),
                "L8 accepted {mutation} of {source}"
            );
            if source == "p9" || source.starts_with("l9-") {
                assert!(
                    compiled9.evaluate_f32(&[1.0; 8]).is_err(),
                    "L9 accepted {mutation} of {source}"
                );
            }
        }
    }
}

#[test]
fn distinct_zero_sized_terminal_index_types_cannot_share_a_compiled_domain() {
    struct First;
    struct Second;
    macro_rules! implement_index {
        ($index:ty) => {
            impl TerminalFormulaIndex for $index {
                fn terminal_count(&self) -> usize {
                    1
                }
                fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
                    (qid == 0 && placed == 0).then_some(0)
                }
                fn reveal_qids(&self, zid: u32) -> &[u32] {
                    assert_eq!(zid, 0);
                    &[0]
                }
            }
        };
    }
    implement_index!(First);
    implement_index!(Second);
    #[repr(C)]
    struct Indices {
        first: First,
        second: Second,
    }
    let indices = Indices {
        first: First,
        second: Second,
    };
    assert_eq!(
        &indices.first as *const First as *const (),
        &indices.second as *const Second as *const (),
    );
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let mut qindex = qindex();
    qindex.states.truncate(1);
    let qfwd = QForwardGraph {
        offsets: vec![[0, 1, 1, 1, 1, 1, 1, 1]],
        records: vec![QForwardRec {
            reveal: 0,
            child_qid: 0,
        }],
        records_by_piece: [0; PIECE_COUNT],
        max_records_per_piece: [1; PIECE_COUNT],
        skipped_invalid_child: 0,
    };
    let l9 = layer(&owned.path.join("l9"), &[(3, &[0])]);
    let l8 = layer(&owned.path.join("l8"), &[(0, &[0])]);
    let p9 = owned.path.join("p9");
    let p8 = owned.path.join("p8");
    write_policy(&p9, &[encode_policy_u16(0, 0)]);
    write_policy(&p8, &[encode_policy_u16(0, 0)]);
    let compiled9 =
        CompiledL9::build(&graph, &qindex, &qfwd, &indices.first, &l9, &p9, limits(64)).unwrap();
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &indices.first,
        &l8,
        &p8,
        0..1,
        &compiled9,
        limits(64)
    )
    .is_ok());
    assert!(CompiledL8::build(
        &graph,
        &qindex,
        &qfwd,
        &indices.second,
        &l8,
        &p8,
        0..1,
        &compiled9,
        limits(64)
    )
    .is_err());
}

#[test]
fn prepared_l9_lookup_matches_scalar_values_with_bounded_and_positional_caches() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("policy");
    write_policy(&policy, &l9_codes());
    let compiled =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    let terminals = [1.375, -0.0, 99.0, -1.5, -3.0, 99.0, 0.125, -2.25];
    let table = terminal_table(&pre, &terminals);
    let scalar = legacy_values(
        9,
        0..l9.metas.len(),
        &l9,
        &policy,
        &graph,
        &qfwd,
        &table,
        None,
    );
    assert_bits(&compiled.evaluate_f32(&terminals).unwrap(), &scalar);
    let prepared = compiled.prepare_f32(&terminals).unwrap();
    for cap in [0, 1, 9, 18, 8192] {
        let mut lookup = prepared.lookup(cap, true).unwrap();
        assert_eq!(lookup.cache_bytes(), 0);
        assert_eq!(lookup.cache_cap_bytes(), cap);
        assert_eq!(lookup.stats_snapshot().materialized_states, 0);
        assert_eq!(lookup.io_stats().membership_read_calls, 0);
        assert_eq!(lookup.io_stats().policy_read_calls, 0);
        assert_eq!(lookup.io_stats().index_read_calls, 0);
        lookup.prefetch_fields(&[3, 4, 5, 99]).unwrap();
        for _ in 0..2 {
            for (block, meta) in l9.metas.iter().enumerate() {
                lookup.prewarm(meta.field_id).unwrap();
                let members = l9.read_block_chains(block);
                for qid in 0..QCOUNT as u32 {
                    let expected = members.binary_search(&qid).map_or(0.0, |position| {
                        scalar[meta.chain_offset as usize + position]
                    });
                    assert_eq!(
                        lookup.get(meta.field_id, qid).unwrap().to_bits(),
                        expected.to_bits(),
                        "cap {cap} field {} qid {qid}",
                        meta.field_id
                    );
                    assert!(lookup.cache_bytes() <= cap);
                    assert!(lookup.cache_peak_bytes() <= cap);
                }
            }
        }
        assert_eq!(lookup.get(99, 0).unwrap().to_bits(), 0.0f32.to_bits());
        assert!(lookup.get(3, QCOUNT as u32).is_err());
        assert!(lookup.get(99, u32::MAX).is_err());
        assert!(lookup.io_stats().membership_read_calls > 0);
        assert_eq!(lookup.io_stats().policy_read_calls, 0);
        assert_eq!(lookup.io_stats().index_read_calls, 0);
        if cap == 0 {
            assert_eq!(lookup.cache_bytes(), 0);
            assert_eq!(lookup.cache_peak_bytes(), 0);
            assert_eq!(lookup.stats_snapshot().materialized_states, 0);
        }
    }
    assert!(compiled.storage_bytes() >= compiled.stats().mapping_bytes);
}

#[test]
fn prepared_l9_sums_keep_reveal_order_and_count_all_zero_terms_like_legacy() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("policy");
    write_policy(&policy, &l9_codes());
    let compiled =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    let terminals = [3.0, 16_777_216.0, 99.0, 6.0, -16_777_216.0, 99.0, -4.0, 1.0];
    let prepared = compiled.prepare_f32(&terminals).unwrap();
    let records = |qids: &[u32]| {
        qids.iter()
            .enumerate()
            .map(|(reveal, &child_qid)| QForwardRec {
                reveal: reveal as u8,
                child_qid,
            })
            .collect::<Vec<_>>()
    };
    let all_qids = records(&[0, 1, 2, 3, 4, 5]);
    let ordered = records(&[0, 4, 2]);
    let reordered = records(&[0, 2, 4]);
    for cap in [0, 9, 8192] {
        let mut lookup = prepared.lookup(cap, true).unwrap();
        assert_eq!(lookup.sum_recs(4, &ordered).unwrap(), (0.0, 0));
        assert_eq!(lookup.sum_recs(4, &reordered).unwrap(), (1.0, 0));
        assert_eq!(lookup.sum_recs(4, &all_qids).unwrap(), (1.0, 3));
        assert_eq!(lookup.sum_recs(3, &all_qids).unwrap(), (6.0, 2));
        assert_eq!(lookup.sum_recs(99, &all_qids).unwrap(), (0.0, 6));
        assert_eq!(lookup.sum_recs(99, &[]).unwrap(), (0.0, 0));
        assert_eq!(lookup.sum_recs(4, &records(&[4, 4])).unwrap(), (2.0, 0));
        let invalid = records(&[QCOUNT as u32]);
        assert!(lookup.sum_recs(4, &invalid).is_err());
        assert!(lookup.sum_recs(99, &invalid).is_err());
        assert!(lookup.cache_peak_bytes() <= cap);
    }
    let mut numeric_zero = terminals;
    numeric_zero[1] = -0.0;
    let prepared_zero = compiled.prepare_f32(&numeric_zero).unwrap();
    let mut lookup = prepared_zero.lookup(0, true).unwrap();
    assert_eq!(lookup.get(4, 0).unwrap().to_bits(), (-0.0f32).to_bits());
    assert_eq!(lookup.sum_recs(4, &records(&[0, 1])).unwrap(), (0.0, 2));
    let huge = [f32::MAX; 8];
    let prepared_huge = compiled.prepare_f32(&huge).unwrap();
    let mut huge_lookup = prepared_huge.lookup(0, true).unwrap();
    assert!(huge_lookup.sum_recs(3, &records(&[0, 1])).is_err());
}

#[test]
fn prepared_l9_cache_prewarming_and_eviction_obey_the_payload_cap() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("policy");
    write_policy(&policy, &l9_codes());
    let compiled =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    let terminals = [1.0; 8];
    let prepared = compiled.prepare_f32(&terminals).unwrap();
    let mut lookup = prepared.lookup(18, true).unwrap();
    let initial = lookup.stats_snapshot();
    assert_eq!(
        (
            initial.cache_hits,
            initial.cache_misses,
            initial.materialized_states
        ),
        (0, 0, 0)
    );
    lookup.prewarm(3).unwrap();
    assert!(lookup.cache_bytes() > 0);
    let bytes_after_prewarm = lookup.io_stats().membership_bytes_read;
    lookup.get(3, 0).unwrap();
    lookup.get(3, 1).unwrap();
    assert_eq!(lookup.io_stats().membership_bytes_read, bytes_after_prewarm);
    lookup.prefetch_fields(&[4, 5, 3]).unwrap();
    lookup.get(3, 0).unwrap();
    let stats = lookup.stats_snapshot();
    assert!(stats.cache_hits > 0);
    assert!(stats.cache_misses > 0);
    assert!(stats.evictions > 0);
    assert!(stats.materialized_states > 0);
    assert!(lookup.cache_bytes() <= 18);
    assert!(lookup.cache_peak_bytes() <= 18);
    let mut quiet = prepared.lookup(18, false).unwrap();
    quiet.prewarm(3).unwrap();
    quiet.get(3, 0).unwrap();
    quiet.prefetch_fields(&[4, 5, 3]).unwrap();
    let quiet_stats = quiet.stats_snapshot();
    assert_eq!(
        (
            quiet_stats.cache_hits,
            quiet_stats.cache_misses,
            quiet_stats.evictions
        ),
        (0, 0, 0)
    );
}

#[test]
fn preparing_l9_values_validates_terminals_and_bound_inputs_before_worker_lookups() {
    for changed in ["policy", "membership", "index"] {
        let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
        let graph = graph();
        let qindex = qindex();
        let pre = terminal_index();
        let qfwd = qforward(&pre);
        let l9 = l9_layer(&owned.path.join("l9"));
        let policy = owned.path.join("policy");
        write_policy(&policy, &l9_codes());
        let compiled =
            CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
        assert!(compiled.prepare_f32(&[1.0; 7]).is_err());
        let mut invalid = [1.0; 8];
        invalid[0] = f32::NAN;
        assert!(compiled.prepare_f32(&invalid).is_err());
        invalid[0] = f32::INFINITY;
        assert!(compiled.prepare_f32(&invalid).is_err());
        let terminal = [1.0; 8];
        let prepared = compiled.prepare_f32(&terminal).unwrap();
        for _ in 0..4 {
            let lookup = prepared.lookup(18, true).unwrap();
            let stats = lookup.stats_snapshot();
            assert_eq!(
                (
                    stats.cache_hits,
                    stats.cache_misses,
                    stats.materialized_states
                ),
                (0, 0, 0)
            );
            assert_eq!(lookup.cache_bytes(), 0);
            assert_eq!(lookup.io_stats().membership_read_calls, 0);
            assert_eq!(lookup.io_stats().policy_read_calls, 0);
            assert_eq!(lookup.io_stats().index_read_calls, 0);
        }
        let watched = match changed {
            "policy" => &policy,
            "membership" => &l9.chains_path,
            "index" => &l9.index_path,
            _ => unreachable!(),
        };
        let replacement = owned.path.join("replacement");
        fs::write(&replacement, fs::read(watched).unwrap()).unwrap();
        fs::rename(replacement, watched).unwrap();
        assert!(
            compiled.prepare_f32(&terminal).is_err(),
            "prepare accepted replacement of {changed}"
        );
    }
}

#[test]
fn shared_l9_rank_index_preserves_values_and_order_without_worker_membership_reads() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("policy");
    write_policy(&policy, &l9_codes());
    let mut compiled =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    let terminals = [
        -0.0,
        16_777_216.0,
        99.0,
        6.0,
        -16_777_216.0,
        99.0,
        -4.0,
        1.0,
    ];
    let expected = compiled.evaluate_f32(&terminals).unwrap();
    let original_table_bytes = compiled.storage_bytes();
    compiled.build_rank_index(1 << 20, 64).unwrap();
    let ranked = compiled.stats().clone();
    assert!(ranked.membership_rank_bytes > 0);
    assert!(ranked.rank_construction_seconds >= 0.0);
    assert!(ranked.rank_build_peak_bytes <= 64);
    assert_eq!(
        compiled.storage_bytes(),
        original_table_bytes + ranked.membership_rank_bytes
    );
    compiled.build_rank_index(1 << 20, 64).unwrap();
    assert_eq!(
        compiled.stats().membership_bytes_read,
        ranked.membership_bytes_read
    );
    assert_eq!(
        compiled.stats().membership_read_calls,
        ranked.membership_read_calls
    );
    assert_eq!(compiled.stats().policy_bytes_read, ranked.policy_bytes_read);
    assert_eq!(compiled.stats().index_bytes_read, ranked.index_bytes_read);
    assert_eq!(
        compiled.storage_bytes(),
        original_table_bytes + ranked.membership_rank_bytes
    );
    let records = |qids: &[u32]| {
        qids.iter()
            .enumerate()
            .map(|(reveal, &child_qid)| QForwardRec {
                reveal: reveal as u8,
                child_qid,
            })
            .collect::<Vec<_>>()
    };
    let prepared = compiled.prepare_f32(&terminals).unwrap();
    let mut lookup = prepared.lookup(0, true).unwrap();
    lookup.prefetch_fields(&[3, 4, 5, 99]).unwrap();
    for (block, meta) in l9.metas.iter().enumerate() {
        let members = l9.read_block_chains(block);
        let mut expected_sum = 0.0f32;
        let mut expected_missing = 0;
        for qid in 0..QCOUNT as u32 {
            let value = members.binary_search(&qid).map_or(0.0, |position| {
                expected[meta.chain_offset as usize + position]
            });
            assert_eq!(
                lookup.get(meta.field_id, qid).unwrap().to_bits(),
                value.to_bits()
            );
            expected_sum += value;
            expected_missing += u64::from(value == 0.0);
        }
        assert_eq!(
            lookup
                .sum_recs(meta.field_id, &records(&[0, 1, 2, 3, 4, 5]))
                .unwrap(),
            (expected_sum, expected_missing)
        );
    }
    assert_eq!(lookup.sum_recs(4, &records(&[0, 4, 2])).unwrap(), (0.0, 0));
    assert_eq!(lookup.sum_recs(4, &records(&[0, 2, 4])).unwrap(), (1.0, 0));
    assert_eq!(lookup.sum_recs(99, &records(&[0, 1])).unwrap(), (0.0, 2));
    assert_eq!(lookup.get(99, 0).unwrap(), 0.0);
    assert!(lookup.get(3, QCOUNT as u32).is_err());
    assert_eq!(lookup.cache_bytes(), 0);
    assert_eq!(lookup.cache_peak_bytes(), 0);
    assert_eq!(lookup.io_stats().membership_read_calls, 0);
    assert_eq!(lookup.io_stats().policy_read_calls, 0);
    assert_eq!(lookup.io_stats().index_read_calls, 0);
}

#[test]
fn shared_l9_rank_index_enforces_retained_and_build_buffer_budgets() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qindex = qindex();
    let pre = terminal_index();
    let qfwd = qforward(&pre);
    let l9 = l9_layer(&owned.path.join("l9"));
    let policy = owned.path.join("policy");
    write_policy(&policy, &l9_codes());
    let mut compiled =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    let original_table_bytes = compiled.storage_bytes();
    assert!(compiled.build_rank_index(1, 64).is_err());
    assert!(compiled.build_rank_index(1 << 20, 1).is_err());
    assert_eq!(compiled.stats().membership_rank_bytes, 0);
    assert_eq!(compiled.storage_bytes(), original_table_bytes);
    let terminals = [1.0; 8];
    {
        let prepared = compiled.prepare_f32(&terminals).unwrap();
        let mut lookup = prepared.lookup(0, true).unwrap();
        assert_eq!(lookup.get(3, 0).unwrap(), 1.0);
        assert!(lookup.io_stats().membership_read_calls > 0);
    }
    compiled.build_rank_index(1 << 20, 64).unwrap();
    let rank_bytes = compiled.stats().membership_rank_bytes;
    let mut exact =
        CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
    assert!(exact.build_rank_index(rank_bytes - 1, 64).is_err());
    assert_eq!(exact.stats().membership_rank_bytes, 0);
    // The largest field needs 6 decoded u32 qids plus one encoded u24.
    assert!(exact.build_rank_index(rank_bytes, 26).is_err());
    exact.build_rank_index(rank_bytes, 27).unwrap();
    assert_eq!(exact.stats().membership_rank_bytes, rank_bytes);
    assert_eq!(exact.stats().rank_build_peak_bytes, 27);
}

#[test]
fn shared_l9_rank_index_keeps_input_version_checks_before_build_and_prepare() {
    for rank_first in [false, true] {
        for changed in ["policy", "membership", "index"] {
            let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
            let graph = graph();
            let qindex = qindex();
            let pre = terminal_index();
            let qfwd = qforward(&pre);
            let l9 = l9_layer(&owned.path.join("l9"));
            let policy = owned.path.join("policy");
            write_policy(&policy, &l9_codes());
            let mut compiled =
                CompiledL9::build(&graph, &qindex, &qfwd, &pre, &l9, &policy, limits(64)).unwrap();
            if rank_first {
                compiled.build_rank_index(1 << 20, 64).unwrap();
            }
            let watched = match changed {
                "policy" => &policy,
                "membership" => &l9.chains_path,
                "index" => &l9.index_path,
                _ => unreachable!(),
            };
            let replacement = owned.path.join("replacement");
            fs::write(&replacement, fs::read(watched).unwrap()).unwrap();
            fs::rename(replacement, watched).unwrap();
            assert!(
                compiled.build_rank_index(1 << 20, 64).is_err(),
                "rank build accepted replacement of {changed}; ranked={rank_first}"
            );
            assert!(
                compiled.prepare_f32(&[1.0; 8]).is_err(),
                "prepare accepted replacement of {changed}; ranked={rank_first}"
            );
        }
    }
}
