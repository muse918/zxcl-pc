//! Greedy T* regression tests. The reference kernels below never use the action
//! cache. The discounted ten-layer fixture is deliberately tiny; it is not a
//! certificate for the production SSP or its optimal value.

use super::*;

const QCOUNT: usize = 8;
const TWO_LINE: FieldId = 4;

fn graph_from_rows(rows: &[[&[u32]; 2]]) -> FieldGraph {
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
        hashes: (0..rows.len() as u64).collect(),
        offsets,
        targets,
    }
}

fn qforward_from_rows(rows: &[[&[u32]; 2]]) -> QForwardGraph {
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    let mut records_by_piece = [0; PIECE_COUNT];
    let mut max_records_per_piece = [0; PIECE_COUNT];
    for row in rows {
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                for (reveal, &child_qid) in row[p].iter().enumerate() {
                    records.push(qgraph::QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
                records_by_piece[p] += row[p].len() as u64;
                max_records_per_piece[p] = max_records_per_piece[p].max(row[p].len());
            }
            off[p + 1] = records.len() as u32;
        }
        offsets.push(off);
    }
    QForwardGraph {
        offsets,
        records,
        records_by_piece,
        max_records_per_piece,
        skipped_invalid_child: 0,
    }
}

fn graph_fixture() -> FieldGraph {
    graph_from_rows(&[
        [&[2, 3, 2], &[2, TWO_LINE]],
        [&[2, 3, 2], &[2, TWO_LINE]],
        [&[EMPTY_FIELD_ID, EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
        [&[EMPTY_FIELD_ID, EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
    ])
}

fn qforward_fixture() -> QForwardGraph {
    qforward_from_rows(&[
        [&[0, 1, 2], &[0, 1, 2]],
        [&[0, 1, 2], &[2, 0, 1]],
        [&[2, 0, 1], &[0, 1, 2]],
        [&[0, 3], &[0, 3]],
        [&[0, 1], &[0, 1]],
        [&[2], &[2]],
        [&[], &[]],
        [&[0, 1, 2], &[0, 1, 2]],
    ])
}

fn terminal_fixture() -> Vec<[f32; PIECE_COUNT]> {
    let mut values = vec![[-0.25; PIECE_COUNT]; QCOUNT];
    for row in &mut values {
        row[0] = 1.0625;
    }
    values[0][0] = 1.375;
    values[0][1] = 1.375;
    values[1][0] = -3.0;
    values[1][1] = 0.0;
    values[2][0] = 0.875;
    values[2][1] = 1.75;
    values
}

fn layer_fixture(prefix: &Path, rows: &[(u32, &[u32])]) -> layer_store::FieldChainLayer {
    let mut writer = layer_store::FieldChainWriter::create(prefix).unwrap();
    for (field, qids) in rows {
        for &qid in *qids {
            writer
                .push_pair_key((u64::from(*field) << 32) | u64::from(qid))
                .unwrap();
        }
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
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

#[derive(Debug)]
struct Evaluation {
    values: Vec<f32>,
    policy: Vec<u16>,
    stats: FullMdpLayerStats,
}

fn read_policy(path: &Path) -> Vec<u16> {
    fs::read(path)
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_equal(expected: &Evaluation, actual: &Evaluation) {
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&expected.values), bits(&actual.values));
    assert_eq!(expected.policy, actual.policy);
    assert_eq!(expected.stats.total_states, actual.stats.total_states);
    assert_eq!(expected.stats.action_count, actual.stats.action_count);
    assert_eq!(
        expected.stats.missing_next_terms,
        actual.stats.missing_next_terms
    );
    assert_eq!(
        expected.stats.terminal_2_edges,
        actual.stats.terminal_2_edges
    );
    assert_eq!(
        expected.stats.terminal_4_edges,
        actual.stats.terminal_4_edges
    );
}

// A scalar oracle uses an explicit membership map, seven-piece scan and direct
// ordered f32 sums. It does not share NextValueLookup or action-average helpers.
#[allow(clippy::too_many_arguments)]
fn scalar_reference(
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; PIECE_COUNT]],
    next: &layer_store::FieldChainLayer,
    values: &[f32],
    gamma: f32,
) -> Evaluation {
    let mut membership = std::collections::BTreeMap::new();
    for (bi, meta) in next.metas.iter().enumerate() {
        for (i, qid) in next.read_block_chains(bi).into_iter().enumerate() {
            membership.insert((meta.field_id, qid), values[meta.chain_offset as usize + i]);
        }
    }
    let mut result = Evaluation {
        values: Vec::new(),
        policy: Vec::new(),
        stats: FullMdpLayerStats::default(),
    };
    for (bi, meta) in cur.metas.iter().enumerate() {
        for qid in cur.read_block_chains(bi) {
            let mut best = 0.0f32;
            let mut code = POLICY_NONE_U16;
            result.stats.total_states += 1;
            for placed in 0..PIECE_COUNT as u8 {
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                for (ei, &field) in graph.edges(meta.field_id, placed).iter().enumerate() {
                    if layer == 9 && field != EMPTY_FIELD_ID {
                        continue;
                    }
                    result.stats.action_count += 1;
                    let value = if layer == 9 {
                        result.stats.terminal_4_edges += 1;
                        terminal[qid as usize][placed as usize]
                    } else if layer == 4 && field == TWO_LINE {
                        result.stats.terminal_2_edges += 1;
                        terminal[qid as usize][placed as usize]
                    } else {
                        let mut sum = 0.0f32;
                        for rec in recs {
                            let v = membership
                                .get(&(field, rec.child_qid))
                                .copied()
                                .unwrap_or(0.0);
                            sum += v;
                            result.stats.missing_next_terms += u64::from(v == 0.0);
                        }
                        let scaled = if gamma == 1.0 { sum } else { gamma * sum };
                        scaled / recs.len() as f32
                    };
                    if value > best {
                        best = value;
                        code = encode_policy_u16(placed, ei);
                    }
                }
            }
            result.values.push(best);
            result.policy.push(code);
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn original(
    root: &Path,
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal: &[[u8; 2]],
    terminal: &[[f32; PIECE_COUNT]],
    next: Option<&layer_store::FieldChainLayer>,
    next_path: &Path,
    gamma: f32,
    two_line: FieldId,
) -> Evaluation {
    let values_path = root.join("original.f32");
    let policy_path = root.join("original.u16");
    let file = output_file(&values_path, cur.total_states * 4);
    let policy = output_file(&policy_path, cur.total_states * 2);
    let result = compute_full_mdp_chunk_f32(
        layer,
        0..cur.metas.len(),
        cur,
        None,
        &file,
        Some(&policy),
        graph,
        qfwd,
        legal,
        terminal,
        next,
        Some(next_path),
        None,
        gamma,
        1024 * 1024,
        qfwd.offsets.len(),
        65_536,
        two_line,
        false,
        false,
        true,
        true,
    )
    .unwrap();
    Evaluation {
        values: load_f32_values_vec(&values_path, cur.total_states).unwrap(),
        policy: read_policy(&policy_path),
        stats: result.stats,
    }
}

#[allow(clippy::too_many_arguments)]
fn cached(
    root: &Path,
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal: &[[u8; 2]],
    terminal: &[[f32; PIECE_COUNT]],
    next: &layer_store::FieldChainLayer,
    next_path: &Path,
    next_values: Option<&[f32]>,
    gamma: f32,
    requested_bytes: usize,
    total_bytes: usize,
    placements: &action_avg_cache::ValidatedPlacements<'_>,
    direct_l9: Option<terminal_l9::Source<'_>>,
) -> Evaluation {
    let values_path = root.join("cached.f32");
    let policy_path = root.join("cached.u16");
    let file = output_file(&values_path, cur.total_states * 4);
    let policy = output_file(&policy_path, cur.total_states * 2);
    let mut ram = vec![f32::NAN; cur.total_states as usize];
    let use_ram = next_values.is_some() || direct_l9.is_some();
    let ram_out = use_ram.then(|| SharedF32Out {
        ptr: ram.as_mut_ptr(),
        len: ram.len(),
    });
    let result = compute_full_mdp_range_group_f32_rolling_sinks_with_action_cache(
        layer,
        &[0..1, 1..cur.metas.len()],
        cur,
        if use_ram { None } else { Some(&file) },
        ram_out,
        Some(&policy),
        graph,
        qfwd,
        legal,
        terminal,
        Some(next),
        Some(next_path),
        next_values,
        None,
        gamma,
        total_bytes,
        qfwd.offsets.len(),
        65_536,
        TWO_LINE,
        false,
        false,
        true,
        true,
        direct_l9,
        Some(action_avg_cache::ActionSource {
            placements,
            requested_bytes,
        }),
    )
    .unwrap();
    Evaluation {
        values: if use_ram {
            ram
        } else {
            load_f32_values_vec(&values_path, cur.total_states).unwrap()
        },
        policy: read_policy(&policy_path),
        stats: result.stats,
    }
}

#[test]
fn greedy_cache_preserves_order_ties_missing_terms_and_fresh_next_values() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let placements = action_avg_cache::ValidatedPlacements::from_fixture(&qfwd).unwrap();
    let terminal = terminal_fixture();
    let qids: Vec<_> = (0..QCOUNT as u32).collect();
    let cur = layer_fixture(
        &root.join("current"),
        &[(0, &qids), (1, &qids), (4, &qids), (5, &qids)],
    );
    let next = layer_fixture(&root.join("next"), &[(2, &[0, 1, 2]), (3, &[0, 1, 2])]);
    let next_path = root.join("next.f32");
    let a = 16_777_216.0f32;
    let inputs = [
        vec![a, -a, 1.0, a, -a, 1.0],
        vec![-7.0, 0.0, 1.25, 2.0, -9.0, 0.0],
        vec![-1.0, -1.0, -1.0, -2.0, -2.0, -2.0],
        vec![-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
    ];
    let mut hits = 0;
    let mut collisions = 0;
    let mut changed_values = Vec::new();
    for (input, values) in inputs.iter().enumerate() {
        write_greedy_ram_snapshot(&next_path, values).unwrap();
        for layer in [3, 4, 9] {
            for gamma in [1.0f32, 0.9375, 0.0] {
                let scalar =
                    scalar_reference(layer, &cur, &graph, &qfwd, &terminal, &next, values, gamma);
                let reference = original(
                    root,
                    layer,
                    &cur,
                    &graph,
                    &qfwd,
                    &legal,
                    &terminal,
                    Some(&next),
                    &next_path,
                    gamma,
                    TWO_LINE,
                );
                assert_equal(&scalar, &reference);
                if layer == 3 && gamma == 1.0 {
                    changed_values.push(
                        reference
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                    if input == 0 {
                        assert_eq!(
                            reference.policy[0],
                            encode_policy_u16(0, 0),
                            "first duplicate edge and piece win ties"
                        );
                        assert_eq!(
                            reference.policy[2],
                            encode_policy_u16(1, 0),
                            "ordered cancellation changes the winning piece"
                        );
                    } else if input == 1 {
                        assert_eq!(
                            reference.values[3].to_bits(),
                            1.0f32.to_bits(),
                            "missing reveal retains denominator two"
                        );
                        assert_eq!(reference.policy[3], encode_policy_u16(0, 1));
                    } else {
                        assert!(reference.policy.iter().all(|&p| p == POLICY_NONE_U16));
                        assert!(reference.values.iter().all(|v| v.to_bits() == 0));
                    }
                }
                if layer == 9 {
                    assert_eq!(reference.policy[2 * QCOUNT], encode_policy_u16(0, 0));
                    assert_eq!(reference.policy[2 * QCOUNT + 1], POLICY_NONE_U16);
                    assert_eq!(reference.policy[2 * QCOUNT + 2], encode_policy_u16(1, 0));
                }
                for ram_source in [false, true] {
                    for (request, budget) in [(0, 0), (16, 64), (16, 4096), (65_536, 1_048_576)] {
                        let result = cached(
                            root,
                            layer,
                            &cur,
                            &graph,
                            &qfwd,
                            &legal,
                            &terminal,
                            &next,
                            &next_path,
                            ram_source.then_some(values.as_slice()),
                            gamma,
                            request,
                            budget,
                            &placements,
                            None,
                        );
                        assert_equal(&reference, &result);
                        assert!(result.stats.action_avg_cache_bytes <= request.min(budget));
                        if layer == 9 || budget < 512 {
                            assert_eq!(result.stats.action_avg_cache_bytes, 0);
                        }
                        hits += result.stats.action_avg_hits;
                        collisions += result.stats.action_avg_collisions;
                    }
                }
            }
        }
    }
    assert_ne!(
        changed_values[0], changed_values[1],
        "successive calls must exercise changing numeric sources"
    );
    assert_ne!(changed_values[1], changed_values[2]);
    assert!(hits > 0, "test must exercise actual cache reuse");
    assert!(collisions > 0, "one-slot cache must exercise replacement");
}

#[test]
fn greedy_cache_matches_materialized_l9_when_l8_uses_terminal_source() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let placements = action_avg_cache::ValidatedPlacements::from_fixture(&qfwd).unwrap();
    let qids: Vec<_> = (0..QCOUNT as u32).collect();
    let cur = layer_fixture(&root.join("current"), &[(0, &qids), (1, &qids)]);
    let l9 = layer_fixture(
        &root.join("l9"),
        &[(2, &[0, 1, 2]), (3, &[0, 1, 2]), (4, &[0, 1, 2])],
    );
    let l9_path = root.join("l9.f32");
    let mut hits = 0;
    let mut observed = Vec::new();
    for scale in [1.0f32, 0.375] {
        let mut terminal = terminal_fixture();
        for row in &mut terminal {
            for value in row {
                *value *= scale;
            }
        }
        let materialized = original(
            root, 9, &l9, &graph, &qfwd, &legal, &terminal, None, &l9_path, 1.0, TWO_LINE,
        );
        write_greedy_ram_snapshot(&l9_path, &materialized.values).unwrap();
        let direct = terminal_l9::Source {
            layer: &l9,
            graph: &graph,
            qfwd: &qfwd,
            terminal_reset_avg: &terminal,
            mode: terminal_l9::Mode::Greedy(&legal),
        };
        terminal_l9::validate_membership(direct).unwrap();
        for gamma in [1.0f32, 0.9375] {
            let reference = original(
                root,
                8,
                &cur,
                &graph,
                &qfwd,
                &legal,
                &terminal,
                Some(&l9),
                &l9_path,
                gamma,
                TWO_LINE,
            );
            assert_equal(
                &scalar_reference(
                    8,
                    &cur,
                    &graph,
                    &qfwd,
                    &terminal,
                    &l9,
                    &materialized.values,
                    gamma,
                ),
                &reference,
            );
            observed.push(
                reference
                    .values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
            for (request, budget) in [(0, 0), (16, 4096), (65_536, 1_048_576)] {
                let result = cached(
                    root,
                    8,
                    &cur,
                    &graph,
                    &qfwd,
                    &legal,
                    &terminal,
                    &l9,
                    &l9_path,
                    None,
                    gamma,
                    request,
                    budget,
                    &placements,
                    Some(direct),
                );
                assert_equal(&reference, &result);
                hits += result.stats.action_avg_hits;
            }
        }
    }
    assert_ne!(
        observed[0], observed[2],
        "terminal table changes must invalidate all numeric averages between calls"
    );
    assert!(hits > 0);
}

#[derive(Debug, PartialEq)]
struct Iteration {
    boundary_bits: Vec<u32>,
    layer_value_bits: Vec<Vec<u8>>,
    policies: Vec<Vec<u8>>,
}

#[allow(clippy::too_many_arguments)]
fn discounted_iterations(
    root: &Path,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal: &[[u8; 2]],
    layers: &[layer_store::FieldChainLayer],
    config: &GreedyEmitConfig,
    source: Option<action_avg_cache::ActionSource<'_>>,
) -> Vec<Iteration> {
    let out = root.join("values");
    let policies = root.join("policies");
    fs::create_dir_all(&out).unwrap();
    fs::create_dir_all(&policies).unwrap();
    let gamma = 0.9375f32;
    let mut boundary = vec![-1000.0f32; layers[0].total_states as usize];
    let mut trajectory = Vec::new();
    for _ in 0..128 {
        let terminal = build_terminal_reset_avg_table(qfwd, &boundary[..4], gamma, 4);
        if let Some(source) = source {
            // This is the production T* sweep/mapper, with all ten memberships
            // and a fresh cache per task and per changing next-value layer.
            emit_greedy_mdp_sweep_with_action_cache(
                graph,
                qfwd,
                legal,
                &terminal,
                layers,
                &out,
                Some(&policies),
                4,
                3,
                gamma,
                config,
                Some(source),
            )
            .unwrap();
        } else {
            for layer in (0..10).rev() {
                let path = mdp_f32_value_path(&out, layer);
                let policy_path = mdp_policy_u16_path(&policies, layer);
                let output = output_file(&path, layers[layer].total_states * 4);
                let policy = output_file(&policy_path, layers[layer].total_states * 2);
                let next_path = mdp_f32_value_path(&out, layer + 1);
                compute_full_mdp_chunk_f32(
                    layer,
                    0..layers[layer].metas.len(),
                    &layers[layer],
                    None,
                    &output,
                    Some(&policy),
                    graph,
                    qfwd,
                    legal,
                    &terminal,
                    layers.get(layer + 1),
                    Some(&next_path),
                    None,
                    gamma,
                    1_048_576,
                    4,
                    65_536,
                    3,
                    false,
                    false,
                    true,
                    true,
                )
                .unwrap();
            }
        }
        let next =
            load_f32_values_vec(&mdp_f32_value_path(&out, 0), layers[0].total_states).unwrap();
        assert!(next.iter().all(|v| v.is_finite()));
        let max_delta = next
            .iter()
            .zip(&boundary)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_delta = next
            .iter()
            .zip(&boundary)
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / next.len() as f64;
        trajectory.push(Iteration {
            boundary_bits: next.iter().map(|v| v.to_bits()).collect(),
            layer_value_bits: (0..10)
                .map(|layer| fs::read(mdp_f32_value_path(&out, layer)).unwrap())
                .collect(),
            policies: (0..10)
                .map(|layer| fs::read(mdp_policy_u16_path(&policies, layer)).unwrap())
                .collect(),
        });
        boundary = next;
        if max_delta <= 1e-5 && mean_delta <= 1e-6 {
            return trajectory;
        }
    }
    panic!("tiny discounted fixture did not reach its stated absolute stopping criterion");
}

#[test]
fn tiny_discounted_ten_layer_optimal_iteration_has_identical_policy_trajectory() {
    let _lock = TEST_ENV_LOCK.lock().unwrap();
    let _settings = loop1_prepared_tests::EnvRestore::set(&[
        ("FULL_MDP_MEMORY_MAX_GIB", "1"),
        ("FULL_MDP_MEMORY_RESERVE_GIB", "0"),
        ("FULL_MDP_BLOCK_CHUNK", "1"),
        ("FULL_MDP_LOOKUP_GROUP_CHUNKS", "4"),
        ("FULL_MDP_TASK_MAX_STATES", "100"),
        ("FULL_MDP_TASK_MAX_MIB", "16"),
    ]);
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_from_rows(&[
        [&[1, 1, 2], &[2, 1]],
        [&[2], &[1]],
        [&[3], &[1, 2]],
        [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
    ]);
    let qfwd = qforward_from_rows(&[
        [&[0, 1, 2], &[0, 1, 2]],
        [&[2, 0, 1], &[2]],
        [&[0, 3], &[2, 3]],
        [&[0, 1, 2], &[0, 1, 2]],
    ]);
    let legal = build_qfwd_legal_placed_table(&qfwd, 4);
    let placements = action_avg_cache::ValidatedPlacements::from_fixture(&qfwd).unwrap();
    let layers: Vec<_> = (0..10)
        .map(|layer| {
            layer_fixture(
                &root.join(format!("membership{layer}")),
                &[
                    (0, &[0, 1, 2, 3]),
                    (1, &[0, 1, 2]),
                    (2, &[0, 2, 3]),
                    (3, &[0, 1, 2, 3]),
                ],
            )
        })
        .collect();
    let config = GreedyEmitConfig {
        policy_only: false,
        ram_next_layer: true,
        ram_next_layer_max_bytes: 1_048_576,
        ram_pair_max_bytes: 2_097_152,
        mmap_spill: false,
        prewarm_field_targets: false,
        collect_cache_stats: true,
        collect_action_stats: true,
        sync_layer_outputs: false,
        sparse_lookup_max_count: 65_536,
        tuning: sweep_schedule::SweepTuning::from_env(1_048_576).unwrap(),
    };
    let reference = discounted_iterations(
        &root.join("reference"),
        &graph,
        &qfwd,
        &legal,
        &layers,
        &config,
        None,
    );
    let cached = discounted_iterations(
        &root.join("cached"),
        &graph,
        &qfwd,
        &legal,
        &layers,
        &config,
        Some(action_avg_cache::ActionSource {
            placements: &placements,
            requested_bytes: 65_536,
        }),
    );
    assert_eq!(
        reference.len(),
        cached.len(),
        "absolute convergence iteration must agree"
    );
    assert_eq!(reference, cached, "every boundary, all ten value layers and the full greedy policy trajectory must agree bitwise");
    assert!(
        reference.len() > 3,
        "fixture must exercise multiple nonstationary sweeps"
    );
    assert!(
        reference[0].policies.iter().flatten().all(|&b| b == 0xff),
        "negative warm start must first choose NONE everywhere"
    );
    assert_ne!(
        reference[0].policies, reference[1].policies,
        "competing actions must replace the initial NONE policy"
    );
    assert!(reference
        .last()
        .unwrap()
        .boundary_bits
        .iter()
        .any(|&b| f32::from_bits(b) > 0.0));
}

#[test]
fn invalid_greedy_cache_setting_rejects_before_loading_or_creating_outputs() {
    const NAME: &str = "FULL_MDP_GREEDY_ACTION_CACHE_MIB";
    let _lock = TEST_ENV_LOCK.lock().unwrap();
    let _settings =
        loop1_prepared_tests::EnvRestore::set(&[(NAME, "0"), ("FULL_MDP_POLICY_ONLY", "0")]);
    assert_eq!(greedy_action_cache_bytes_from_env().unwrap(), 0);
    env::remove_var(NAME);
    assert_eq!(greedy_action_cache_bytes_from_env().unwrap(), 0);
    env::set_var(NAME, "1");
    assert_eq!(greedy_action_cache_bytes_from_env().unwrap(), 1_048_576);
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let out = root.join("must-not-create-values");
    let policy = root.join("must-not-create-policies");
    let args = vec![
        "solver".to_owned(),
        "emit-full-mdp-layers-from-l0".to_owned(),
        root.join("missing-graph.bin")
            .to_string_lossy()
            .into_owned(),
        root.join("missing-memberships")
            .to_string_lossy()
            .into_owned(),
        root.join("missing-boundary.f32")
            .to_string_lossy()
            .into_owned(),
        out.to_string_lossy().into_owned(),
        "0.9375".to_owned(),
        "0".to_owned(),
        policy.to_string_lossy().into_owned(),
    ];
    for value in [
        "".to_owned(),
        "-1".to_owned(),
        "abc".to_owned(),
        "1.5".to_owned(),
        usize::MAX.to_string(),
    ] {
        env::set_var(NAME, &value);
        assert!(
            greedy_action_cache_bytes_from_env().is_err(),
            "invalid MiB setting {value:?}"
        );
        let error = run_emit_full_mdp_layers_from_l0(&args).unwrap_err();
        assert!(
            format!("{error:#}").contains(NAME),
            "setting must reject before nonexistent input I/O: {error:#}"
        );
        assert!(!out.exists());
        assert!(!policy.exists());
    }
}
