//! Scheduling regressions for the legacy seed/value-iteration sweep drivers.
use super::*;

const QCOUNT: usize = 4;
const TWO_LINE: FieldId = 3;

fn graph_fixture(fields: usize) -> FieldGraph {
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    for field in 0..fields {
        let edges: [&[u32]; 2] = match field {
            0 => [&[1, 1, 2], &[2, 1]],
            1 => [&[2], &[1]],
            2 => [&[TWO_LINE], &[1, 2]],
            3 => [&[EMPTY_FIELD_ID], &[EMPTY_FIELD_ID]],
            _ => [&[1, 2], &[2, 1]],
        };
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = targets.len() as u32;
        for piece in 0..PIECE_COUNT {
            if piece < 2 {
                targets.extend_from_slice(edges[piece]);
            }
            off[piece + 1] = targets.len() as u32;
        }
        offsets.push(off);
    }
    FieldGraph {
        hashes: (0..fields as u64).collect(),
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
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for piece in 0..PIECE_COUNT {
            if piece < 2 {
                for (reveal, &child_qid) in row[piece].iter().enumerate() {
                    records.push(qgraph::QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
            }
            off[piece + 1] = records.len() as u32;
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
    for &(field, qids) in blocks {
        for &qid in qids {
            writer
                .push_pair_key((field as u64) << 32 | qid as u64)
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

fn assert_semantic_stats(actual: &FullMdpLayerStats, expected: &FullMdpLayerStats) {
    assert_eq!(actual.total_states, expected.total_states);
    assert_eq!(actual.nonzero, expected.nonzero);
    assert_eq!(actual.terminal_2_edges, expected.terminal_2_edges);
    assert_eq!(actual.terminal_4_edges, expected.terminal_4_edges);
    assert_eq!(actual.action_count, expected.action_count);
    assert_eq!(actual.missing_next_terms, expected.missing_next_terms);
    assert_eq!(actual.sum_v, expected.sum_v);
    assert_eq!(actual.max_v, expected.max_v);
    assert_eq!(actual.max_delta, expected.max_delta);
    assert_eq!(actual.sum_abs_delta, expected.sum_abs_delta);
    // Cache counters intentionally describe the selected grouping/backend and
    // need not equal the single-task reference's cache hit/miss counts.
}

#[test]
fn scheduled_discount_chunks_preserve_values_ties_and_previous_value_residuals() {
    let _env_lock = TEST_ENV_LOCK.lock().unwrap();
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    // Default scheduling creates several grouped tasks, exercising pool/cache
    // wiring without a production graph, q-domain or full solve.
    let fields = 4097usize;
    let graph = graph_fixture(fields);
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = vec![
        [1.375; PIECE_COUNT],
        [1.0625; PIECE_COUNT],
        [0.875; PIECE_COUNT],
        [0.0; PIECE_COUNT],
    ];
    let qids = [0, 1, 2, 3];
    let blocks: Vec<_> = (0..fields as u32)
        .map(|field| (field, qids.as_slice()))
        .collect();
    let cur = layer_fixture(&root.join("current"), &blocks);
    let next = layer_fixture(
        &root.join("next"),
        &[(1, &[0, 1, 3]), (2, &[1, 2]), (3, &[0, 3])],
    );
    let next_values = vec![0.4f32, 0.4, 0.0, 0.7, 0.1, 1.3, 0.0];
    let next_path = root.join("next.f32");
    write_greedy_ram_snapshot(&next_path, &next_values).unwrap();
    let prev_path = root.join("prev.f32");
    let prev_values: Vec<_> = (0..cur.total_states)
        .map(|i| (i % 11) as f32 * 0.125)
        .collect();
    write_greedy_ram_snapshot(&prev_path, &prev_values).unwrap();
    let tuning = sweep_schedule::SweepTuning::from_env(1024 * 1024).unwrap();
    for gamma in [1.0f32, 0.9375] {
        let reference_path = root.join("reference.f32");
        let reference_policy_path = root.join("reference.u16");
        let reference = output_file(&reference_path, cur.total_states * 4);
        let reference_policy = output_file(&reference_policy_path, cur.total_states * 2);
        let expected = compute_full_mdp_chunk_f32(
            4,
            0..cur.metas.len(),
            &cur,
            Some(&prev_path),
            &reference,
            Some(&reference_policy),
            &graph,
            &qfwd,
            &legal,
            &terminal,
            Some(&next),
            Some(&next_path),
            None,
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
        for ram_source in [false, true] {
            let actual_path = root.join("actual.f32");
            let actual_policy_path = root.join("actual.u16");
            let actual = output_file(&actual_path, cur.total_states * 4);
            let actual_policy = output_file(&actual_policy_path, cur.total_states * 2);
            let ram_values = ram_source.then_some(next_values.as_slice());
            let pinned_bytes = ram_values.map_or(0, |values| std::mem::size_of_val(values) as u64);
            let plan = tuning.plan(&cur.metas, 0, pinned_bytes, 4, QCOUNT).unwrap();
            assert!(plan.workers <= 4);
            assert_eq!(plan.ranges.first().unwrap().start, 0);
            assert_eq!(plan.ranges.last().unwrap().end, cur.metas.len());
            assert!(plan
                .ranges
                .windows(2)
                .all(|pair| pair[0].end == pair[1].start));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(plan.workers)
                .build()
                .unwrap();
            let mut stats = FullMdpLayerStats::default();
            let mut blocks_done = 0;
            for batch in plan.ranges.chunks(plan.workers.max(1)) {
                let results: Vec<_> = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|range| {
                            compute_full_mdp_chunk_f32(
                                4,
                                range.clone(),
                                &cur,
                                Some(&prev_path),
                                &actual,
                                Some(&actual_policy),
                                &graph,
                                &qfwd,
                                &legal,
                                &terminal,
                                Some(&next),
                                Some(&next_path),
                                ram_values,
                                gamma,
                                plan.per_task_cache_bytes,
                                QCOUNT,
                                65_536,
                                TWO_LINE,
                                true,
                                false,
                                true,
                                true,
                            )
                        })
                        .collect()
                });
                for result in results {
                    let result = result.unwrap();
                    stats.add(&result.stats);
                    blocks_done += result.blocks;
                }
            }
            assert_eq!(blocks_done, cur.metas.len());
            assert_eq!(
                fs::read(&actual_path).unwrap(),
                fs::read(&reference_path).unwrap()
            );
            assert_eq!(
                fs::read(&actual_policy_path).unwrap(),
                fs::read(&reference_policy_path).unwrap()
            );
            assert_semantic_stats(&stats, &expected.stats);
            assert!(
                stats.missing_next_terms > 0,
                "fixture must include absent reveal children"
            );
            assert!(stats.action_count > 0);
            assert!(
                stats.sum_abs_delta > 0.0,
                "previous-value residuals must be exercised"
            );
        }
    }
}

#[test]
fn scheduled_l0_groups_preserve_dense_boundary_residuals_for_file_and_ram() {
    let _env_lock = TEST_ENV_LOCK.lock().unwrap();
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture(4);
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = vec![
        [1.375; PIECE_COUNT],
        [1.0625; PIECE_COUNT],
        [0.875; PIECE_COUNT],
        [0.0; PIECE_COUNT],
    ];
    let cur = layer_fixture(&root.join("current"), &[(0, &[0, 1, 2, 3])]);
    let next = layer_fixture(
        &root.join("next"),
        &[(1, &[0, 1, 3]), (2, &[1, 2]), (3, &[0, 3])],
    );
    let next_values = vec![0.4f32, 0.4, 0.0, 0.7, 0.1, 1.3, 0.0];
    let next_path = root.join("next.f32");
    write_greedy_ram_snapshot(&next_path, &next_values).unwrap();
    let old_boundary = [0.125f32, 0.5, 0.25, 0.75];
    let prev_path = root.join("prev.f32");
    write_greedy_ram_snapshot(&prev_path, &old_boundary).unwrap();
    let reference_path = root.join("reference.f32");
    let reference = output_file(&reference_path, cur.total_states * 4);
    let expected = compute_full_mdp_chunk_f32(
        0,
        0..1,
        &cur,
        Some(&prev_path),
        &reference,
        None,
        &graph,
        &qfwd,
        &legal,
        &terminal,
        Some(&next),
        Some(&next_path),
        None,
        1.0,
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
    let tuning = sweep_schedule::SweepTuning::from_env(1024 * 1024).unwrap();
    for ram_source in [false, true] {
        let actual_path = root.join("actual.f32");
        let actual = output_file(&actual_path, cur.total_states * 4);
        let ram_values = ram_source.then_some(next_values.as_slice());
        let pinned_bytes = ram_values.map_or(0, |values| std::mem::size_of_val(values) as u64);
        let plan = tuning.plan(&cur.metas, 0, pinned_bytes, 4, QCOUNT).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(plan.workers)
            .build()
            .unwrap();
        let results: Vec<_> = pool.install(|| {
            plan.ranges
                .par_iter()
                .map(|range| {
                    compute_full_mdp_range_group_f32_rolling(
                        0,
                        std::slice::from_ref(range),
                        &cur,
                        &actual,
                        &graph,
                        &qfwd,
                        &legal,
                        &terminal,
                        Some(&next),
                        Some(&next_path),
                        ram_values,
                        Some(&old_boundary),
                        1.0,
                        plan.per_task_cache_bytes,
                        QCOUNT,
                        65_536,
                        TWO_LINE,
                        true,
                        false,
                        true,
                        true,
                    )
                })
                .collect()
        });
        let mut stats = FullMdpLayerStats::default();
        for result in results {
            stats.add(&result.unwrap().stats);
        }
        assert_eq!(
            fs::read(&actual_path).unwrap(),
            fs::read(&reference_path).unwrap()
        );
        assert_semantic_stats(&stats, &expected.stats);
        assert!(stats.sum_abs_delta > 0.0);
    }
}
