use super::*;

const QCOUNT: usize = 4;
const TWO_LINE: FieldId = 3;

fn graph_fixture() -> FieldGraph {
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    let edges: [[&[u32]; 2]; 4] = [
        [&[1, 1, 2], &[2, 1]],
        [&[2], &[1]],
        [&[TWO_LINE], &[1, 2]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
    ];
    for row in edges {
        let mut off = [0u32; PIECE_COUNT + 1];
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
    let children: [[&[u32]; 2]; QCOUNT] =
        [[&[0, 1], &[2, 3]], [&[3, 0], &[1]], [&[], &[2]], [&[], &[]]];
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    for row in children {
        let mut off = [0u32; PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                for (reveal, &child_qid) in row[p].iter().enumerate() {
                    records.push(qgraph::QForwardRec {
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

fn layer_fixture(prefix: &Path, blocks: &[(u32, &[u32])]) -> layer_store::FieldChainLayer {
    let mut writer = layer_store::FieldChainWriter::create(prefix).unwrap();
    for (field, qids) in blocks {
        for &qid in *qids {
            writer
                .push_pair_key((*field as u64) << 32 | qid as u64)
                .unwrap();
        }
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
}

fn current_fixture(prefix: &Path) -> layer_store::FieldChainLayer {
    layer_fixture(
        prefix,
        &[
            (0, &[0, 1, 2, 3]),
            (1, &[0, 1, 2, 3]),
            (2, &[0, 1, 2, 3]),
            (3, &[0, 1, 2, 3]),
        ],
    )
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

fn terminal_fixture() -> Vec<[f32; PIECE_COUNT]> {
    vec![
        [1.375; PIECE_COUNT],
        [1.0625; PIECE_COUNT],
        [0.875; PIECE_COUNT],
        [0.0; PIECE_COUNT],
    ]
}

#[test]
fn grouped_greedy_values_and_policies_match_original_for_file_ram_and_mmap() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let cur = current_fixture(&root.join("cur"));
    let next = layer_fixture(
        &root.join("next"),
        &[(1, &[0, 1, 3]), (2, &[1, 2]), (3, &[0, 3])],
    );
    let next_values = vec![0.4f32, 0.4, 0.0, 0.7, 0.1, 1.3, 0.0];
    let next_path = root.join("next.f32");
    write_greedy_ram_snapshot(&next_path, &next_values).unwrap();
    let next_file = File::open(&next_path).unwrap();
    let next_mmap = unsafe { memmap2::Mmap::map(&next_file) }.unwrap();
    let next_mapped_values =
        unsafe { std::slice::from_raw_parts(next_mmap.as_ptr() as *const f32, next_values.len()) };
    let mut observed_missing = false;
    let mut observed_actions = false;
    for layer in [3, 4, 9] {
        for gamma in [1.0f32, 0.9375] {
            let expected_path = root.join("expected.f32");
            let expected_policy_path = root.join("expected.u16");
            let expected_file = output_file(&expected_path, cur.total_states * 4);
            let expected_policy = output_file(&expected_policy_path, cur.total_states * 2);
            let expected_result = compute_full_mdp_chunk_f32(
                layer,
                0..cur.metas.len(),
                &cur,
                None,
                &expected_file,
                Some(&expected_policy),
                &graph,
                &qfwd,
                &legal,
                &terminal,
                Some(&next),
                Some(&next_path),
                Some(&next_values),
                gamma,
                1024 * 1024,
                QCOUNT,
                65_536,
                TWO_LINE,
                false,
                false,
                true,
                true,
            )
            .unwrap();
            let expected = fs::read(&expected_path).unwrap();
            let expected_policy_bytes = fs::read(&expected_policy_path).unwrap();
            // These independent assertions pin the intended semantics in addition
            // to comparing kernels: strict > keeps the first tied edge/piece, and
            // missing reveals stay in the denominator with a zero contribution.
            let action = |field: usize, qid: usize| {
                let off = (field * QCOUNT + qid) * 2;
                u16::from_le_bytes(expected_policy_bytes[off..off + 2].try_into().unwrap())
            };
            if layer == 3 {
                assert_eq!(
                    action(0, 0),
                    encode_policy_u16(0, 0),
                    "first duplicate edge wins"
                );
                let off = QCOUNT * 4;
                let value = f32::from_le_bytes(expected[off..off + 4].try_into().unwrap());
                assert_eq!(
                    value.to_bits(),
                    ((gamma * 0.7f32) / 2.0).to_bits(),
                    "pruned reveal contributes zero without changing denominator"
                );
            } else if layer == 4 {
                assert_eq!(action(2, 0), encode_policy_u16(0, 0));
            } else {
                assert_eq!(
                    action(3, 0),
                    encode_policy_u16(0, 0),
                    "first terminal piece wins a tie"
                );
            }
            for source in 0..3 {
                for sink in 0..3 {
                    for prewarm in [false, true] {
                        let actual_path = root.join("actual.f32");
                        let actual_policy_path = root.join("actual.u16");
                        let actual_file = output_file(&actual_path, cur.total_states * 4);
                        let actual_policy = output_file(&actual_policy_path, cur.total_states * 2);
                        let mut ram = vec![f32::NAN; cur.total_states as usize];
                        let mut mapped = if sink == 2 {
                            Some(unsafe { memmap2::MmapMut::map_mut(&actual_file) }.unwrap())
                        } else {
                            None
                        };
                        let ram_out = match sink {
                            1 => Some(SharedF32Out {
                                ptr: ram.as_mut_ptr(),
                                len: ram.len(),
                            }),
                            2 => Some(SharedF32Out {
                                ptr: mapped.as_mut().unwrap().as_mut_ptr() as *mut f32,
                                len: ram.len(),
                            }),
                            _ => None,
                        };
                        let next_ram = match source {
                            0 => None,
                            1 => Some(next_values.as_slice()),
                            _ => Some(next_mapped_values),
                        };
                        let result = compute_full_mdp_range_group_f32_rolling_sinks(
                            layer,
                            &[0..1, 1..3, 3..4],
                            &cur,
                            if sink == 0 { Some(&actual_file) } else { None },
                            ram_out,
                            Some(&actual_policy),
                            &graph,
                            &qfwd,
                            &legal,
                            &terminal,
                            Some(&next),
                            Some(&next_path),
                            next_ram,
                            None,
                            gamma,
                            1024 * 1024,
                            QCOUNT,
                            65_536,
                            TWO_LINE,
                            prewarm,
                            false,
                            true,
                            true,
                        )
                        .unwrap();
                        let actual = if sink == 1 {
                            ram.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
                        } else if let Some(m) = mapped.as_ref() {
                            m.to_vec()
                        } else {
                            fs::read(&actual_path).unwrap()
                        };
                        assert_eq!(actual, expected, "values layer={layer} gamma={gamma} source={source} sink={sink} prewarm={prewarm}");
                        assert_eq!(
                            fs::read(&actual_policy_path).unwrap(),
                            expected_policy_bytes,
                            "policies layer={layer} source={source} sink={sink}"
                        );
                        assert_eq!(result.stats.total_states, cur.total_states);
                        assert_eq!(
                            result.stats.action_count,
                            expected_result.stats.action_count
                        );
                        assert_eq!(
                            result.stats.missing_next_terms,
                            expected_result.stats.missing_next_terms
                        );
                        assert_eq!(
                            result.stats.terminal_2_edges,
                            expected_result.stats.terminal_2_edges
                        );
                        assert_eq!(
                            result.stats.terminal_4_edges,
                            expected_result.stats.terminal_4_edges
                        );
                        observed_missing |= result.stats.missing_next_terms > 0;
                        observed_actions |= result.stats.action_count > 0;
                        for block in 0..4 {
                            let off = (block * 4 + 3) * 2;
                            assert_eq!(
                                u16::from_le_bytes(
                                    expected_policy_bytes[off..off + 2].try_into().unwrap()
                                ),
                                POLICY_NONE_U16
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        observed_missing,
        "fixture must exercise pruned reveal children"
    );
    assert!(observed_actions, "action statistics must be wired");
}

fn config(policy_only: bool, ram: bool, mmap: bool) -> GreedyEmitConfig {
    GreedyEmitConfig {
        policy_only,
        ram_next_layer: ram,
        ram_next_layer_max_bytes: 1024 * 1024,
        ram_pair_max_bytes: 2 * 1024 * 1024,
        mmap_spill: mmap,
        prewarm_field_targets: true,
        collect_cache_stats: true,
        collect_action_stats: true,
        sync_layer_outputs: true,
        sparse_lookup_max_count: 65_536,
        tuning: sweep_schedule::SweepTuning::from_env(1024 * 1024).unwrap(),
    }
}

#[test]
fn policy_only_rolling_sweeps_match_full_snapshots_and_preserve_unowned_files() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let layers: Vec<_> = (0..10)
        .map(|l| current_fixture(&root.join(format!("membership{l}"))))
        .collect();
    let baseline = root.join("baseline");
    fs::create_dir(&baseline).unwrap();
    let baseline_policy = root.join("baseline-policy");
    fs::create_dir(&baseline_policy).unwrap();
    for layer in (0..10).rev() {
        let out_path = mdp_f32_value_path(&baseline, layer);
        let policy_path = mdp_policy_u16_path(&baseline_policy, layer);
        let output = output_file(&out_path, layers[layer].total_states * 4);
        let policy = output_file(&policy_path, layers[layer].total_states * 2);
        let next_path = mdp_f32_value_path(&baseline, layer + 1);
        compute_full_mdp_chunk_f32(
            layer,
            0..layers[layer].metas.len(),
            &layers[layer],
            None,
            &output,
            Some(&policy),
            &graph,
            &qfwd,
            &legal,
            &terminal,
            layers.get(layer + 1),
            Some(&next_path),
            None,
            0.9375,
            1024 * 1024,
            QCOUNT,
            65_536,
            TWO_LINE,
            false,
            false,
            false,
            false,
        )
        .unwrap();
    }
    for policy_only in [false, true] {
        for mode in 0..4 {
            let output = root.join(format!("out-{policy_only}-{mode}"));
            fs::create_dir(&output).unwrap();
            let policies = root.join(format!("policy-{policy_only}-{mode}"));
            fs::create_dir(&policies).unwrap();
            let sentinel = mdp_f32_value_path(&output, 7);
            if policy_only {
                fs::write(&sentinel, b"user-owned snapshot").unwrap();
            }
            let mut cfg = config(policy_only, mode != 0, mode == 2);
            // Force a file sink followed by a read-only mmap pin for large layers.
            if mode == 3 {
                cfg.ram_next_layer_max_bytes = 0;
            }
            emit_greedy_mdp_sweep(
                &graph,
                &qfwd,
                &legal,
                &terminal,
                &layers,
                &output,
                Some(&policies),
                QCOUNT,
                TWO_LINE,
                0.9375,
                &cfg,
            )
            .unwrap();
            for layer in 0..10 {
                assert_eq!(
                    fs::read(mdp_policy_u16_path(&policies, layer)).unwrap(),
                    fs::read(mdp_policy_u16_path(&baseline_policy, layer)).unwrap()
                );
                if !policy_only || layer == 0 {
                    assert_eq!(
                        fs::read(mdp_f32_value_path(&output, layer)).unwrap(),
                        fs::read(mdp_f32_value_path(&baseline, layer)).unwrap()
                    );
                } else if layer != 7 {
                    assert!(!mdp_f32_value_path(&output, layer).exists());
                }
            }
            if policy_only {
                assert_eq!(fs::read(&sentinel).unwrap(), b"user-owned snapshot");
            }
            assert!(fs::read_dir(&output).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".policy-rolling-")));
        }
    }
    let failed = root.join("failed");
    fs::create_dir(&failed).unwrap();
    let policy_dir = root.join("failed-policy");
    fs::create_dir(&policy_dir).unwrap();
    let sentinel = failed.join("retain-me");
    fs::write(&sentinel, b"keep").unwrap();
    assert!(emit_greedy_mdp_sweep(
        &graph,
        &qfwd,
        &legal,
        &terminal,
        &layers,
        &failed,
        Some(&policy_dir),
        QCOUNT - 1,
        TWO_LINE,
        1.0,
        &config(true, false, false)
    )
    .is_err());
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    assert!(fs::read_dir(&failed).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".policy-rolling-")));
}

#[test]
fn greedy_group_reuses_one_lookup_and_optional_stats_do_not_change_output() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let cur = current_fixture(&root.join("cur"));
    let next = layer_fixture(
        &root.join("next"),
        &[(1, &[0, 1, 3]), (2, &[1, 2]), (3, &[0, 3])],
    );
    let next_values = vec![0.4f32, 0.4, 0.0, 0.7, 0.1, 1.3, 0.0];
    let next_path = root.join("next.f32");
    write_greedy_ram_snapshot(&next_path, &next_values).unwrap();
    let ranges = [0..1, 1..2, 2..3, 3..4];
    for ram_source in [false, true] {
        let next_ram = if ram_source {
            Some(next_values.as_slice())
        } else {
            None
        };
        let grouped_path = root.join("grouped.f32");
        let split_path = root.join("split.f32");
        let silent_path = root.join("silent.f32");
        let grouped = output_file(&grouped_path, cur.total_states * 4);
        let split = output_file(&split_path, cur.total_states * 4);
        let silent = output_file(&silent_path, cur.total_states * 4);
        let run = |ranges: &[std::ops::Range<usize>], file: &File, stats: bool| {
            compute_full_mdp_range_group_f32_rolling_sinks(
                3,
                ranges,
                &cur,
                Some(file),
                None,
                None,
                &graph,
                &qfwd,
                &legal,
                &terminal,
                Some(&next),
                Some(&next_path),
                next_ram,
                None,
                0.9375,
                1024 * 1024,
                QCOUNT,
                65_536,
                TWO_LINE,
                false,
                false,
                stats,
                stats,
            )
            .unwrap()
        };
        let grouped_stats = run(&ranges, &grouped, true).stats;
        let mut split_stats = FullMdpLayerStats::default();
        for range in &ranges {
            split_stats.add(&run(std::slice::from_ref(range), &split, true).stats);
        }
        let silent_stats = run(&ranges, &silent, false).stats;
        assert_eq!(
            fs::read(&grouped_path).unwrap(),
            fs::read(&split_path).unwrap()
        );
        assert_eq!(
            fs::read(&grouped_path).unwrap(),
            fs::read(&silent_path).unwrap()
        );
        assert_eq!(grouped_stats.dense_cache_misses, 3);
        assert_eq!(split_stats.dense_cache_misses, 7);
        assert_eq!(grouped_stats.dense_materialized_states, 7);
        assert_eq!(split_stats.dense_materialized_states, 17);
        assert!(grouped_stats.action_count > 0);
        assert!(grouped_stats.missing_next_terms > 0);
        assert_eq!(silent_stats.action_count, 0);
        assert_eq!(silent_stats.missing_next_terms, 0);
        assert_eq!(silent_stats.dense_cache_hits, 0);
        assert_eq!(silent_stats.dense_cache_misses, 0);
        assert_eq!(silent_stats.dense_missing_fields, 0);
        assert_eq!(silent_stats.dense_materialized_states, 0);
        assert_eq!(silent_stats.total_states, grouped_stats.total_states);
    }
}

#[test]
fn policy_only_requires_policy_destination_before_creating_outputs() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let layers: Vec<_> = (0..10)
        .map(|l| current_fixture(&root.join(format!("membership{l}"))))
        .collect();
    let out = root.join("out");
    fs::create_dir(&out).unwrap();
    let sentinel = mdp_f32_value_path(&out, 9);
    fs::write(&sentinel, b"existing snapshot").unwrap();
    let qfwd = qforward_fixture();
    let error = emit_greedy_mdp_sweep(
        &graph_fixture(),
        &qfwd,
        &build_qfwd_legal_placed_table(&qfwd, QCOUNT),
        &terminal_fixture(),
        &layers,
        &out,
        None,
        QCOUNT,
        TWO_LINE,
        1.0,
        &config(true, true, false),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("FULL_MDP_POLICY_ONLY=1 requires policy_out_dir"));
    assert_eq!(fs::read(&sentinel).unwrap(), b"existing snapshot");
    assert_eq!(fs::read_dir(&out).unwrap().count(), 1);
}

#[test]
fn greedy_empty_layers_support_all_sinks_and_cleanup() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let layers: Vec<_> = (0..10)
        .map(|l| layer_fixture(&root.join(format!("empty{l}")), &[]))
        .collect();
    let qfwd = qforward_fixture();
    for policy_only in [false, true] {
        for mode in 0..3 {
            let out = root.join(format!("out-{policy_only}-{mode}"));
            fs::create_dir(&out).unwrap();
            let policies = root.join(format!("policies-{policy_only}-{mode}"));
            fs::create_dir(&policies).unwrap();
            emit_greedy_mdp_sweep(
                &graph_fixture(),
                &qfwd,
                &build_qfwd_legal_placed_table(&qfwd, QCOUNT),
                &terminal_fixture(),
                &layers,
                &out,
                Some(&policies),
                QCOUNT,
                TWO_LINE,
                1.0,
                &config(policy_only, mode != 0, mode == 2),
            )
            .unwrap();
            assert_eq!(
                fs::read_dir(&out).unwrap().count(),
                if policy_only { 1 } else { 10 }
            );
            for layer in 0..10 {
                assert_eq!(
                    fs::metadata(mdp_policy_u16_path(&policies, layer))
                        .unwrap()
                        .len(),
                    0
                );
                if !policy_only || layer == 0 {
                    assert_eq!(
                        fs::metadata(mdp_f32_value_path(&out, layer)).unwrap().len(),
                        0
                    );
                }
            }
        }
    }
}

#[test]
fn greedy_preserves_reveal_sum_order_and_applies_gamma_after_sum() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let records = (0..3)
        .map(|child_qid| qgraph::QForwardRec {
            reveal: child_qid as u8,
            child_qid,
        })
        .collect();
    let mut offsets = vec![[3; PIECE_COUNT + 1]; QCOUNT];
    offsets[0][0] = 0;
    let qfwd = QForwardGraph {
        offsets,
        records,
        records_by_piece: [3, 0, 0, 0, 0, 0, 0],
        max_records_per_piece: [3, 0, 0, 0, 0, 0, 0],
        skipped_invalid_child: 0,
    };
    let cur = layer_fixture(&root.join("cur"), &[(0, &[0])]);
    let next = layer_fixture(&root.join("next"), &[(1, &[0, 1, 2])]);
    let values = [16_777_216.0f32, 1.0, 1.0];
    let next_path = root.join("next.f32");
    write_greedy_ram_snapshot(&next_path, &values).unwrap();
    let next_file = File::open(&next_path).unwrap();
    let mapped = unsafe { memmap2::Mmap::map(&next_file) }.unwrap();
    let mapped_values =
        unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, values.len()) };
    let gamma = 0.9375f32;
    let expected = (gamma * values[0]) / 3.0;
    let regrouped = (gamma * (values[0] + (values[1] + values[2]))) / 3.0;
    let scaled_terms = ((gamma * values[0] + gamma * values[1]) + gamma * values[2]) / 3.0;
    assert_ne!(
        expected.to_bits(),
        regrouped.to_bits(),
        "fixture must expose reassociation"
    );
    assert_ne!(
        expected.to_bits(),
        scaled_terms.to_bits(),
        "fixture must expose moving gamma into the sum"
    );
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    for source in 0..3 {
        let next_ram = match source {
            0 => None,
            1 => Some(values.as_slice()),
            _ => Some(mapped_values),
        };
        let policy_path = root.join("policy.u16");
        let policy = output_file(&policy_path, 2);
        let mut actual = [f32::NAN];
        compute_full_mdp_range_group_f32_rolling_sinks(
            3,
            &[0..1],
            &cur,
            None,
            Some(SharedF32Out {
                ptr: actual.as_mut_ptr(),
                len: actual.len(),
            }),
            Some(&policy),
            &graph,
            &qfwd,
            &legal,
            &terminal_fixture(),
            Some(&next),
            Some(&next_path),
            next_ram,
            None,
            gamma,
            1024 * 1024,
            QCOUNT,
            65_536,
            TWO_LINE,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(actual[0].to_bits(), expected.to_bits(), "source={source}");
        assert_eq!(
            fs::read(&policy_path).unwrap(),
            encode_policy_u16(0, 0).to_le_bytes()
        );
    }
}
