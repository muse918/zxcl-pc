use super::*;
use crate::{
    build_qfwd_legal_placed_table, compute_fixed_policy_chunk_f32_with_l9_source,
    compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source, GreedyEmitScratch, SharedF32Out,
};
use std::{env, fs, fs::OpenOptions, path::Path};

const QCOUNT: usize = 4;

fn graph_fixture() -> FieldGraph {
    let rows: [[&[u32]; 2]; 4] = [
        [&[1], &[2]],
        [&[EMPTY_FIELD_ID], &[]],
        [&[1, EMPTY_FIELD_ID, EMPTY_FIELD_ID], &[EMPTY_FIELD_ID, 2]],
        [&[EMPTY_FIELD_ID, EMPTY_FIELD_ID], &[EMPTY_FIELD_ID, 0]],
    ];
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    for row in rows {
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = targets.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                targets.extend_from_slice(row[p]);
            }
            off[p + 1] = targets.len() as u32;
        }
        offsets.push(off);
    }
    FieldGraph {
        hashes: vec![0, 1, 2, 3],
        offsets,
        targets,
    }
}

fn qforward_fixture() -> QForwardGraph {
    let rows: [[&[u32]; 2]; QCOUNT] =
        [[&[0, 1], &[2, 3]], [&[3, 0], &[1]], [&[], &[2]], [&[], &[]]];
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    for row in rows {
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                for (reveal, &child_qid) in row[p].iter().enumerate() {
                    records.push(QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
            }
            off[p + 1] = records.len() as u32;
        }
        offsets.push(off);
    }
    QForwardGraph {
        offsets,
        records,
        records_by_piece: [0; PIECE_COUNT],
        max_records_per_piece: [2; PIECE_COUNT],
        skipped_invalid_child: 0,
    }
}

fn layer_fixture(prefix: &Path, blocks: &[(FieldId, &[u32])]) -> FieldChainLayer {
    let mut writer = crate::layer_store::FieldChainWriter::create(prefix).unwrap();
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

fn membership_fixture(prefix: &Path) -> FieldChainLayer {
    layer_fixture(
        prefix,
        &[
            (0, &[0, 1, 2, 3]),
            (1, &[0, 2]),
            (2, &[0, 1, 2, 3]),
            (3, &[0, 2]),
        ],
    )
}

fn terminal_fixture() -> Vec<[f32; PIECE_COUNT]> {
    let mut terminal = vec![[0.0; PIECE_COUNT]; QCOUNT];
    terminal[0][0] = 1.375;
    terminal[0][1] = 1.375;
    terminal[1][0] = 1.0625;
    terminal[1][1] = 0.5;
    terminal[2][0] = 99.0; // No reveal records: this piece must be ignored.
    terminal[2][1] = 0.875;
    terminal[3] = [99.0; PIECE_COUNT]; // Neither piece is available.
    terminal
}

fn write_policy(path: &Path, codes: &[u16]) -> File {
    fs::write(
        path,
        codes
            .iter()
            .flat_map(|code| code.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    File::open(path).unwrap()
}

fn fixed_codes() -> Vec<u16> {
    vec![
        POLICY_NONE_U16,
        POLICY_NONE_U16,
        POLICY_NONE_U16,
        POLICY_NONE_U16,
        encode_policy_u16(0, 0),
        POLICY_NONE_U16,
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 1),
        encode_policy_u16(1, 0),
        POLICY_NONE_U16,
        encode_policy_u16(1, 0),
        encode_policy_u16(1, 0),
    ]
}

fn output_file(path: &Path, bytes: u64) -> File {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    file.set_len(bytes).unwrap();
    file
}

#[test]
fn terminal_masks_and_greedy_actions_preserve_ties_and_availability_order() {
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let mut legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let mut terminal = terminal_fixture();
    assert_eq!(
        (0..4)
            .map(|field| finish_mask(&graph, field).unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 3, 3]
    );
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 2, 0).unwrap(),
        (1.375, encode_policy_u16(0, 1))
    );
    legal[0] = [1, 0];
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 2, 0).unwrap(),
        (1.375, encode_policy_u16(1, 0))
    );
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 3, 2).unwrap(),
        (0.875, encode_policy_u16(1, 0))
    );
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 3, 3).unwrap(),
        (0.0, POLICY_NONE_U16)
    );
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 0, 0).unwrap(),
        (0.0, POLICY_NONE_U16)
    );
    terminal[0][0] = -0.0;
    terminal[0][1] = -2.0;
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 3, 0).unwrap(),
        (0.0, POLICY_NONE_U16)
    );
}

#[test]
fn missing_membership_keeps_zero_terms_and_the_original_reveal_denominator() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let layer = membership_fixture(&owned.path.join("l9"));
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let source = Source {
        layer: &layer,
        graph: &graph,
        qfwd: &qfwd,
        terminal_reset_avg: &terminal,
        mode: Mode::Greedy(&legal),
    };
    let recs = (0..4)
        .map(|child_qid| QForwardRec {
            reveal: child_qid as u8,
            child_qid,
        })
        .collect::<Vec<_>>();
    for cap in [0, 8192] {
        let mut lookup = Lookup::new(source, cap, true).unwrap();
        lookup.prewarm(3).unwrap();
        assert_eq!(
            lookup.get(3, 1).unwrap(),
            0.0,
            "absent qid has a positive terminal candidate"
        );
        assert_eq!(lookup.get(99, 0).unwrap(), 0.0);
        assert!(lookup.get(3, u32::MAX).is_err());
        assert!(lookup.get(99, QCOUNT as u32).is_err());
        let invalid_rec = [QForwardRec {
            reveal: 0,
            child_qid: QCOUNT as u32,
        }];
        assert!(lookup.sum_recs(3, &invalid_rec).is_err());
        assert!(lookup.sum_recs(99, &invalid_rec).is_err());
        let (sum, missing) = lookup.sum_recs(3, &recs).unwrap();
        assert_eq!(sum.to_bits(), 2.25f32.to_bits());
        assert_eq!(missing, 2);
        assert_eq!((sum / recs.len() as f32).to_bits(), 0.5625f32.to_bits());
        assert_eq!(lookup.sum_recs(99, &recs).unwrap(), (0.0, 4));
        assert_eq!(lookup.sum_recs(3, &[]).unwrap(), (0.0, 0));
        assert!(lookup.stats_snapshot().missing_fields > 0);
    }
}

#[test]
fn fixed_terminal_values_follow_stored_actions_and_reject_invalid_actions() {
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let mut terminal = terminal_fixture();
    terminal[0][0] = 3.0;
    terminal[0][1] = 1.0;
    assert_eq!(
        fixed_value(&graph, &qfwd, &terminal, 2, 0, encode_policy_u16(1, 0)).unwrap(),
        1.0
    );
    assert_eq!(
        greedy_action(&graph, &qfwd, &legal, &terminal, 2, 0)
            .unwrap()
            .0,
        3.0
    );
    terminal[0][0] = -2.0;
    terminal[0][1] = -4.25;
    assert_eq!(
        fixed_value(&graph, &qfwd, &terminal, 2, 0, encode_policy_u16(1, 0)).unwrap(),
        -4.25
    );
    terminal[0][1] = -0.0;
    assert_eq!(
        fixed_value(&graph, &qfwd, &terminal, 2, 0, encode_policy_u16(1, 0))
            .unwrap()
            .to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        fixed_value(&graph, &qfwd, &terminal, 2, 0, POLICY_NONE_U16).unwrap(),
        0.0
    );
    for (field, qid, code, message) in [
        (2, 0, 0x0700, "corrupt policy code"),
        (2, 0, encode_policy_u16(0, 3), "out of range"),
        (2, 2, encode_policy_u16(0, 1), "no reveal records"),
        (0, 0, encode_policy_u16(0, 0), "non-terminal field"),
    ] {
        let error = fixed_value(&graph, &qfwd, &terminal, field, qid, code).unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    }
}

#[test]
fn positional_fallback_and_eviction_match_cached_greedy_and_fixed_values() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let layer = membership_fixture(&owned.path.join("l9"));
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let mut terminal = terminal_fixture();
    terminal[0][0] = -1.375;
    terminal[0][1] = 0.625;
    terminal[1][0] = -1.0625;
    terminal[2][1] = -0.875;
    let codes = fixed_codes();
    let policy = write_policy(&owned.path.join("policy.u16"), &codes);
    for mode in [Mode::Greedy(&legal), Mode::Fixed(&policy)] {
        let source = Source {
            layer: &layer,
            graph: &graph,
            qfwd: &qfwd,
            terminal_reset_avg: &terminal,
            mode,
        };
        let mut cached = Lookup::new(source, 8192, true).unwrap();
        for cap in [0, 1100] {
            let mut other = Lookup::new(source, cap, true).unwrap();
            for _ in 0..2 {
                for field in [3, 0, 2, 1, 99] {
                    other.prewarm(field).unwrap();
                    for qid in [0, 1, 2, 3] {
                        assert_eq!(
                            other.get(field, qid).unwrap().to_bits(),
                            cached.get(field, qid).unwrap().to_bits(),
                            "cap={cap} field={field} qid={qid}"
                        );
                        assert!(other.cache_bytes <= other.cache_cap);
                    }
                }
            }
            if cap == 0 {
                assert!(other.cache.is_empty());
                assert_eq!(other.stats_snapshot().materialized_states, 0);
            } else {
                assert!(other.stats_snapshot().evictions > 0);
            }
        }
        if matches!(mode, Mode::Fixed(_)) {
            assert_eq!(cached.get(1, 0).unwrap(), -1.375);
            assert_eq!(cached.get(2, 0).unwrap(), 0.625);
            assert_eq!(cached.get(2, 1).unwrap(), -1.0625);
            assert_eq!(cached.get(2, 2).unwrap(), -0.875);
        }
    }
    let short = write_policy(&owned.path.join("short.u16"), &[POLICY_NONE_U16]);
    let source = Source {
        layer: &layer,
        graph: &graph,
        qfwd: &qfwd,
        terminal_reset_avg: &terminal,
        mode: Mode::Fixed(&short),
    };
    assert!(Lookup::new(source, 8192, false)
        .unwrap_err()
        .to_string()
        .contains("policy size mismatch"));
}

#[test]
fn helpers_and_lookup_reject_invalid_domains_before_indexing() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let layer = membership_fixture(&owned.path.join("l9"));
    let bad_field_layer = layer_fixture(&owned.path.join("bad-field"), &[(99, &[0])]);
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let source = Source {
        layer: &layer,
        graph: &graph,
        qfwd: &qfwd,
        terminal_reset_avg: &terminal,
        mode: Mode::Greedy(&legal),
    };
    validate_membership(source).unwrap();
    assert!(finish_mask(&graph, graph.offsets.len() as FieldId).is_err());
    for (field, qid) in [(99, 0), (2, QCOUNT as u32), (99, u32::MAX)] {
        assert!(greedy_action(&graph, &qfwd, &legal, &terminal, field, qid).is_err());
        assert!(fixed_value(&graph, &qfwd, &terminal, field, qid, POLICY_NONE_U16).is_err());
    }
    let mut invalid_legal = legal.clone();
    invalid_legal[0] = [PIECE_COUNT as u8, 255];
    assert!(greedy_action(&graph, &qfwd, &invalid_legal, &terminal, 2, 0).is_err());
    assert!(greedy_action(&graph, &qfwd, &legal[..0], &terminal, 2, 0).is_err());
    assert!(greedy_action(&graph, &qfwd, &legal, &terminal[..0], 2, 0).is_err());
    assert!(fixed_value(&graph, &qfwd, &terminal[..0], 2, 0, POLICY_NONE_U16).is_err());
    let mut bad_graph = graph.clone();
    bad_graph.offsets[2][1] = u32::MAX;
    assert!(finish_mask(&bad_graph, 2).is_err());
    assert!(greedy_action(&bad_graph, &qfwd, &legal, &terminal, 2, 0).is_err());
    let mut bad_qfwd = qfwd.clone();
    bad_qfwd.offsets[0][1] = u32::MAX;
    assert!(greedy_action(&graph, &bad_qfwd, &legal, &terminal, 2, 0).is_err());
    assert!(fixed_value(&graph, &bad_qfwd, &terminal, 2, 0, encode_policy_u16(0, 1)).is_err());
    for cap in [0, 8192] {
        let mut bad_field_lookup = Lookup::new(
            Source {
                layer: &bad_field_layer,
                ..source
            },
            cap,
            false,
        )
        .unwrap();
        // Construction does not scan unrelated metadata for every task group.
        // The one full validation pass rejects it globally; a queried corrupt
        // field is still rejected before accessing graph or qforward arrays.
        assert_eq!(bad_field_lookup.get(2, 0).unwrap(), 0.0);
        assert!(bad_field_lookup.get(99, 0).is_err());
        assert!(Lookup::new(
            Source {
                terminal_reset_avg: &terminal[..1],
                ..source
            },
            cap,
            false
        )
        .is_err());
        assert!(Lookup::new(
            Source {
                mode: Mode::Greedy(&legal[..1]),
                ..source
            },
            cap,
            false
        )
        .is_err());
        let mut lookup = Lookup::new(source, cap, false).unwrap();
        assert!(lookup.get(2, QCOUNT as u32).is_err());
        assert!(lookup.get(99, QCOUNT as u32).is_err());
        for field in [2, 99] {
            let recs = [QForwardRec {
                reveal: 0,
                child_qid: u32::MAX,
            }];
            assert!(lookup.sum_recs(field, &recs).is_err());
        }
    }
    assert!(validate_membership(Source {
        layer: &bad_field_layer,
        ..source
    })
    .is_err());
}

#[test]
fn corrupted_membership_is_rejected_by_cached_and_positional_paths() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let policy = write_policy(&owned.path.join("policy.u16"), &[POLICY_NONE_U16; 3]);
    for (name, malformed) in [
        ("out-of-domain", [0u32, 1, QCOUNT as u32]),
        ("unsorted", [0u32, 2, 1]),
        ("duplicate", [0u32, 1, 1]),
    ] {
        let layer = layer_fixture(&owned.path.join(name), &[(1, &[0, 1, 2])]);
        let bytes = malformed
            .iter()
            .flat_map(|qid| qid.to_le_bytes()[..3].to_vec())
            .collect::<Vec<_>>();
        fs::write(&layer.chains_path, bytes).unwrap();
        for mode in [Mode::Greedy(&legal), Mode::Fixed(&policy)] {
            let source = Source {
                layer: &layer,
                graph: &graph,
                qfwd: &qfwd,
                terminal_reset_avg: &terminal,
                mode,
            };
            assert!(validate_membership(source).is_err(), "{name} full scan");
            for cap in [0, 8192] {
                // qid 0 is valid and appears before every corruption. A lookup
                // must validate the complete block before accepting that hit.
                let actual =
                    Lookup::new(source, cap, false).and_then(|mut lookup| lookup.get(1, 0));
                assert!(actual.is_err(), "{name} cap={cap} mode={mode:?}");
                let recs = [QForwardRec {
                    reveal: 0,
                    child_qid: 0,
                }];
                let actual = Lookup::new(source, cap, false)
                    .and_then(|mut lookup| lookup.sum_recs(1, &recs));
                assert!(actual.is_err(), "{name} sum cap={cap} mode={mode:?}");
            }
        }
    }
}

#[test]
fn direct_l8_kernels_match_materialized_l9_for_greedy_and_signed_fixed_inputs() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let l9 = membership_fixture(&root.join("l9"));
    let l8 = layer_fixture(&root.join("l8"), &[(0, &[0, 1, 2, 3])]);
    let l9_policy = write_policy(&root.join("l9.u16"), &fixed_codes());
    let l8_policy = write_policy(
        &root.join("l8.u16"),
        &[
            encode_policy_u16(0, 0),
            encode_policy_u16(1, 0),
            encode_policy_u16(1, 0),
            POLICY_NONE_U16,
        ],
    );
    for signed in [false, true] {
        let mut terminal = terminal_fixture();
        if signed {
            terminal[0][0] = -1.375;
            terminal[0][1] = 0.625;
            terminal[1][0] = -1.0625;
            terminal[2][1] = -0.875;
        }
        let greedy = |layer: usize,
                      cur: &FieldChainLayer,
                      next_values: Option<&[f32]>,
                      source: Option<Source<'_>>,
                      cap: usize,
                      prewarm: bool| {
            let mut values = vec![f32::NAN; cur.total_states as usize];
            let policy = output_file(&root.join("greedy-output.u16"), cur.total_states * 2);
            compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source(
                layer,
                &[0..cur.metas.len()],
                cur,
                None,
                Some(SharedF32Out {
                    ptr: values.as_mut_ptr(),
                    len: values.len(),
                }),
                Some(&policy),
                &graph,
                &qfwd,
                &legal,
                &terminal,
                (layer == 8).then_some(&l9),
                None,
                next_values,
                None,
                0.9375,
                cap,
                QCOUNT,
                65_536,
                3,
                prewarm,
                false,
                true,
                true,
                source,
            )
            .unwrap();
            (values, fs::read(root.join("greedy-output.u16")).unwrap())
        };
        let fixed = |layer: usize,
                     cur: &FieldChainLayer,
                     next_values: Option<&[f32]>,
                     source: Option<Source<'_>>,
                     cap: usize| {
            let mut values = vec![f32::NAN; cur.total_states as usize];
            compute_fixed_policy_chunk_f32_with_l9_source(
                layer,
                0..cur.metas.len(),
                cur,
                if layer == 9 { &l9_policy } else { &l8_policy },
                None,
                Some(SharedF32Out {
                    ptr: values.as_mut_ptr(),
                    len: values.len(),
                }),
                &graph,
                &qfwd,
                &terminal,
                (layer == 8).then_some(&l9),
                None,
                next_values,
                0.9375,
                cap,
                QCOUNT,
                65_536,
                3,
                source,
            )
            .unwrap();
            values
        };
        let greedy_l9 = greedy(9, &l9, None, None, 8192, false).0;
        let fixed_l9 = fixed(9, &l9, None, None, 8192);
        let expected_greedy = greedy(8, &l8, Some(&greedy_l9), None, 8192, false);
        let expected_fixed = fixed(8, &l8, Some(&fixed_l9), None, 8192);
        assert!(expected_greedy.0.iter().any(|value| *value > 0.0));
        if signed {
            assert!(expected_fixed.iter().any(|value| *value < 0.0));
        }
        for cap in [0, 1100, 8192] {
            let source = Source {
                layer: &l9,
                graph: &graph,
                qfwd: &qfwd,
                terminal_reset_avg: &terminal,
                mode: Mode::Greedy(&legal),
            };
            for prewarm in [false, true] {
                let actual = greedy(8, &l8, None, Some(source), cap, prewarm);
                assert_eq!(
                    actual
                        .0
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_greedy
                        .0
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "greedy signed={signed} cap={cap} prewarm={prewarm}"
                );
                assert_eq!(actual.1, expected_greedy.1);
            }
            let source = Source {
                mode: Mode::Fixed(&l9_policy),
                ..source
            };
            let actual = fixed(8, &l8, None, Some(source), cap);
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_fixed
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "fixed signed={signed} cap={cap}"
            );
        }
    }
}
