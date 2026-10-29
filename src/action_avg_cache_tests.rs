use super::*;
use crate::*;

#[test]
fn cache_partition_preserves_lookup_floor_and_exact_payload_cap() {
    assert_eq!(ActionAverageCache::entry_bytes(), 16);
    for qcount in [0usize, 8, CHECK_QSTATE_COUNT, usize::MAX] {
        for cap in [0usize, 511, 1_024, 1_048_576, 64 * 1_048_576] {
            for requested in [0usize, 1, 15, 16, 31, 1_048_576, usize::MAX] {
                let (cache, lookup_cap) = ActionAverageCache::partition(requested, cap, qcount);
                let payload = cache.as_ref().map_or(0, |c| c.stats().payload_bytes);
                assert!(payload <= requested);
                assert_eq!(payload + lookup_cap, cap);
                if payload > 0 {
                    assert!(lookup_cap >= qcount.saturating_mul(16).saturating_add(512));
                    assert!(payload / 16 == (payload / 16).next_power_of_two());
                }
            }
        }
    }
}

#[test]
fn one_slot_collisions_keep_field_formula_zero_sign_and_missing_identity() {
    let (cache, _) = ActionAverageCache::partition(16, 4_096, 8);
    let mut cache = cache.unwrap();
    assert!(cache.get(0, 0, 3).is_none());
    cache.insert(
        0,
        0,
        CachedAverage {
            value: -0.0,
            missing_terms: 2,
        },
    );
    let cached = cache.get(0, 0, 3).unwrap();
    assert_eq!(cached.value.to_bits(), (-0.0f32).to_bits());
    assert_eq!(cached.missing_terms, 2);
    assert!(cache.get(1, 0, 3).is_none());
    cache.insert(
        1,
        0,
        CachedAverage {
            value: -7.25,
            missing_terms: 1,
        },
    );
    assert!(cache.get(0, 0, 3).is_none());
    assert!(cache.get(1, 1, 2).is_none());
    let stats = cache.stats();
    assert_eq!(
        (
            stats.hits,
            stats.misses,
            stats.collisions,
            stats.saved_reveals
        ),
        (1, 4, 3, 3)
    );
    // Another invocation must start with no values, even for the same key.
    let (fresh, _) = ActionAverageCache::partition(16, 4_096, 8);
    assert!(fresh.unwrap().get(1, 0, 3).is_none());
}

fn qforward() -> QForwardGraph {
    let children: [&[u32]; 8] = [
        &[0, 1, 2],
        &[0, 1, 2],
        &[2, 0, 1],
        &[0, 1, 2],
        &[0, 3],
        &[0, 3],
        &[0, 1, 2],
        &[],
    ];
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    for (qid, list) in children.iter().enumerate() {
        let mut row = [records.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if (qid == 3 && placed == 1) || (qid != 3 && placed == 0) {
                records.extend(list.iter().enumerate().map(|(reveal, &child_qid)| {
                    qgraph::QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    }
                }));
            }
            row[placed + 1] = records.len() as u32;
        }
        offsets.push(row);
    }
    QForwardGraph {
        offsets,
        records,
        records_by_piece: [0; PIECE_COUNT],
        max_records_per_piece: [3; PIECE_COUNT],
        skipped_invalid_child: 0,
    }
}

fn graph() -> FieldGraph {
    let mut offsets = Vec::new();
    let mut targets = Vec::new();
    for field in 0..5 {
        let mut row = [targets.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if field < 2 {
                if placed == 0 {
                    targets.extend([2, 3, 2]);
                }
                if placed == 1 {
                    targets.extend([2, 4]);
                }
            }
            row[placed + 1] = targets.len() as u32;
        }
        offsets.push(row);
    }
    FieldGraph {
        hashes: (0..5).collect(),
        offsets,
        targets,
    }
}

fn layer(prefix: &Path, fields: &[(u32, &[u32])]) -> layer_store::FieldChainLayer {
    let mut writer = layer_store::FieldChainWriter::create(prefix).unwrap();
    for &(field, qids) in fields {
        for &qid in qids {
            writer
                .push_pair_key((u64::from(field) << 32) | u64::from(qid))
                .unwrap();
        }
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
}

fn policies() -> Vec<u16> {
    let mut codes = vec![
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 2),
        encode_policy_u16(0, 0),
        encode_policy_u16(1, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 0),
        encode_policy_u16(0, 1),
        POLICY_NONE_U16,
    ];
    let mut second = codes.clone();
    second[3] = encode_policy_u16(1, 1); // Layer-4 terminal must bypass sharing.
    codes.extend(second);
    codes
}

fn reference(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; 7]],
    codes: &[u16],
    next: &[f32],
    gamma: f32,
    layer: usize,
) -> (Vec<f32>, u64) {
    let mut values = Vec::new();
    let mut missing = 0;
    for field in 0..2 {
        for qid in 0..8 {
            let code = codes[field * 8 + qid];
            if code == POLICY_NONE_U16 {
                values.push(0.0);
                continue;
            }
            let placed = (code >> 8) as usize;
            let target = graph.edges(field as u32, placed as u8)[(code & 255) as usize];
            if layer == 4 && target == 4 {
                values.push(terminal[qid][placed]);
                continue;
            }
            let row = &qfwd.offsets[qid];
            let recs = &qfwd.records[row[placed] as usize..row[placed + 1] as usize];
            let mut sum = 0.0f32;
            for rec in recs {
                let value = if (target == 2 || target == 3) && rec.child_qid < 3 {
                    next[(target as usize - 2) * 3 + rec.child_qid as usize]
                } else {
                    0.0
                };
                missing += u64::from(value == 0.0);
                sum += value;
            }
            let scaled = if gamma == 1.0 { sum } else { gamma * sum };
            values.push(scaled / recs.len() as f32);
        }
    }
    (values, missing)
}

struct ActionStatsEnv(Option<std::ffi::OsString>);
impl Drop for ActionStatsEnv {
    fn drop(&mut self) {
        match self.0.take() {
            Some(old) => env::set_var("FULL_MDP_ACTION_STATS", old),
            None => env::remove_var("FULL_MDP_ACTION_STATS"),
        }
    }
}

#[test]
fn kernel_matches_independent_ordered_signed_reference_and_restarts_each_call() {
    let _lock = TEST_ENV_LOCK.lock().unwrap();
    let _stats_env = ActionStatsEnv(env::var_os("FULL_MDP_ACTION_STATS"));
    env::set_var("FULL_MDP_ACTION_STATS", "1");
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qfwd = qforward();
    let placements = ValidatedPlacements::from_fixture(&qfwd).unwrap();
    let cur = layer(
        &owned.path.join("cur"),
        &[
            (0, &[0, 1, 2, 3, 4, 5, 6, 7]),
            (1, &[0, 1, 2, 3, 4, 5, 6, 7]),
        ],
    );
    let next = layer(
        &owned.path.join("next"),
        &[(2, &[0, 1, 2]), (3, &[0, 1, 2])],
    );
    let codes = policies();
    let policy_path = owned.path.join("policy");
    fs::write(
        &policy_path,
        codes
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let policy = File::open(policy_path).unwrap();
    let terminal: Vec<[f32; 7]> = (0..8)
        .map(|q| std::array::from_fn(|p| (q * 7 + p) as f32 + 0.125))
        .collect();
    let mut total_hits = 0;
    let mut total_collisions = 0;
    // The first vector exposes sum reordering: [a,-a,1] and [1,a,-a]
    // have different f32 sums. The second changes every cached numeric value.
    for values in [
        [16_777_216.0f32, -16_777_216.0, 1.0, 0.5, -0.5, -0.0],
        [-7.0, 0.0, 1.25, 2.0, -9.0, 0.0],
    ] {
        let next_path = owned.path.join("values");
        fs::write(
            &next_path,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for gamma in [1.0f32, 0.9375, 0.0] {
            for layer_number in [3, 4] {
                let (expected, expected_missing) = reference(
                    &graph,
                    &qfwd,
                    &terminal,
                    &codes,
                    &values,
                    gamma,
                    layer_number,
                );
                for source in [0, 1] {
                    for requested_bytes in [0, 16, 256, 64 * 1_024] {
                        let mut actual = vec![f32::NAN; 16];
                        let result = compute_fixed_policy_chunk_f32_with_action_cache(
                            layer_number,
                            0..2,
                            &cur,
                            &policy,
                            None,
                            Some(SharedF32Out {
                                ptr: actual.as_mut_ptr(),
                                len: actual.len(),
                            }),
                            &graph,
                            &qfwd,
                            &terminal,
                            Some(&next),
                            Some(&next_path),
                            if source == 1 { Some(&values) } else { None },
                            gamma,
                            1_048_576,
                            8,
                            if source == 0 { 0 } else { 64 },
                            4,
                            None,
                            None,
                            Some(ActionSource {
                                placements: &placements,
                                requested_bytes,
                            }),
                        )
                        .unwrap();
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(result.stats.missing_next_terms, expected_missing);
                        assert!(result.stats.action_avg_cache_bytes <= requested_bytes);
                        total_hits += result.stats.action_avg_hits;
                        total_collisions += result.stats.action_avg_collisions;
                    }
                }
            }
        }
    }
    assert!(total_hits > 0);
    assert!(total_collisions > 0);
    let other_graph = qforward();
    assert!(placements.check_graph(&other_graph, 8).is_err());
    assert!(placements.check_graph(&qfwd, 7).is_err());
}
