use super::*;
use crate::qgraph::QRevClassRec;
use std::cmp::Reverse;

fn graph_fixture(qids: usize) -> QRevClassGraph {
    let mut offsets = Vec::new();
    let mut records = Vec::new();
    for qid in 0..qids {
        let mut off = [0; crate::piece::PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for piece in 0..crate::piece::PIECE_COUNT {
            for i in 0..((qid + piece) % 4) {
                records.push(QRevClassRec {
                    bag_tail: ((qid * 13 + piece * 11 + i) % 127) as u32,
                    avail_mask: (1 << ((qid + i) % 28)) | (1 << ((piece + i) % 28)),
                });
            }
            off[piece + 1] = records.len() as u32;
        }
        offsets.push(off);
    }
    QRevClassGraph {
        offsets,
        records,
        records_by_piece: [0; crate::piece::PIECE_COUNT],
        masks_popcount_by_piece: [0; crate::piece::PIECE_COUNT],
        max_records_per_piece: [0; crate::piece::PIECE_COUNT],
        max_parent_avails_per_record: 2,
    }
}

// The previous child loop, kept independent from the optimized helper.
fn collect_reference(
    qids: &[u32],
    graph: &QRevClassGraph,
    parents: &[u64; crate::piece::PIECE_COUNT],
) -> ([HashMap<u32, u32>; crate::piece::PIECE_COUNT], u64) {
    let mut maps = std::array::from_fn(|_| HashMap::with_capacity(4096));
    let mut raw = 0;
    for &qid in qids {
        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let records = graph.records_for(qid, placed);
            if records.is_empty() {
                continue;
            }
            let parent_count = parents[placed as usize];
            if parent_count == 0 {
                continue;
            }
            for rec in records {
                raw += rec.avail_mask.count_ones() as u64 * parent_count;
                *maps[placed as usize].entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
            }
        }
    }
    (maps, raw)
}

#[test]
fn reused_child_maps_match_allocating_reference() {
    let graph = graph_fixture(257);
    let mut maps = std::array::from_fn(|_| HashMap::with_capacity(4096));
    for child in 0..101 {
        let qids: Vec<_> = (0..(child * 7 % 80))
            .map(|i| ((child * 31 + i) % 257) as u32)
            .collect();
        let parents = std::array::from_fn(|piece| ((child + piece) % 5) as u64);
        let (expected, raw) = collect_reference(&qids, &graph, &parents);
        let mut stats = QLiveStats::default();
        qlive_collect_child_classes(&qids, &graph, &parents, &mut maps, &mut stats);
        assert_eq!(maps, expected, "child {child}");
        assert_eq!(stats.raw, raw);
        for map in &mut maps {
            qlive_reset_piece_classes(map);
        }
    }
}

#[test]
fn oversized_child_workspace_is_not_retained() {
    let mut ordinary = HashMap::with_capacity(4096);
    ordinary.insert(1, 3);
    let capacity = ordinary.capacity();
    qlive_reset_piece_classes(&mut ordinary);
    assert!(ordinary.is_empty());
    assert_eq!(ordinary.capacity(), capacity);
    let mut oversized = HashMap::with_capacity(QLIVE_RETAINED_PIECE_CLASSES + 1);
    oversized.insert(9, 7);
    qlive_reset_piece_classes(&mut oversized);
    assert!(oversized.is_empty());
    assert!(oversized.capacity() <= QLIVE_RETAINED_PIECE_CLASSES);
}

#[test]
fn child_read_buffers_release_only_oversized_allocations() {
    let mut qids = Vec::with_capacity(128);
    let mut bytes = Vec::with_capacity(384);
    qids.push(7);
    bytes.push(9);
    let capacities = (qids.capacity(), bytes.capacity());
    qlive_release_oversized_child_buffers(&mut qids, &mut bytes);
    assert_eq!(qids, [7]);
    assert_eq!(bytes, [9]);
    assert_eq!((qids.capacity(), bytes.capacity()), capacities);

    let mut qids = Vec::with_capacity(QLIVE_RETAINED_CHILD_QIDS + 1);
    let mut bytes = Vec::with_capacity(QLIVE_RETAINED_CHILD_QIDS * 3 + 1);
    qlive_release_oversized_child_buffers(&mut qids, &mut bytes);
    assert_eq!((qids.capacity(), bytes.capacity()), (0, 0));
}

fn accumulators(fields: usize) -> HashMap<FieldId, HashMap<u32, u32>> {
    (0..fields)
        .map(|field| {
            let classes = (0..((field * 13 + 3) % 17))
                .map(|class| (class as u32, (1 << (class % 28)) | 1))
                .collect();
            (field as FieldId, classes)
        })
        .collect()
}

#[test]
fn batch_spill_order_matches_repeated_largest_scan() {
    // Exercise ties, empty fields, ineligible fields, and max_spills == 0.
    for max_spills in 0..4 {
        let mut reference = accumulators(1024);
        let counts: Vec<_> = (0..1024).map(|i| (i % 4) as u32).collect();
        let mut batch = qlive_spill_candidates(&reference, &counts, max_spills);
        let eligible = reference
            .iter()
            .filter(|(f, a)| !a.is_empty() && counts[**f as usize] < max_spills)
            .count();
        assert_eq!(batch.len(), eligible);
        while let Some((count, Reverse(field))) = batch.pop() {
            let (&expected, acc) = reference
                .iter()
                .filter(|(f, a)| !a.is_empty() && counts[**f as usize] < max_spills)
                .max_by_key(|(f, a)| (a.len(), Reverse(**f)))
                .unwrap();
            assert_eq!(field, expected);
            assert_eq!(count, acc.len());
            reference.remove(&field);
        }
        assert!(!reference
            .iter()
            .any(|(f, a)| !a.is_empty() && counts[*f as usize] < max_spills));
    }
}

#[test]
fn capped_spill_ignores_absent_empty_and_ineligible_candidates() -> Result<()> {
    let root = std::env::temp_dir().join(format!("zxcl-frontier-capped-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let mut spill = QLiveSpillWriter::create(&root.join("spill.bin"))?;
    let mut accs: HashMap<_, _> = [
        (0, HashMap::new()),
        (1, [(3, 1)].into_iter().collect()),
        (2, [(5, 2)].into_iter().collect()),
    ]
    .into_iter()
    .collect();
    let mut live = 2;
    let mut counts = [0, 1, 0, 0];
    let mut cold = [0, 1, 0, 0];
    for field in [0, 1, 3] {
        assert_eq!(
            qlive_spill_field_capped(
                field,
                &mut accs,
                &mut spill,
                &mut live,
                &mut counts,
                &mut cold,
                1
            )?,
            (0, false)
        );
    }
    assert_eq!(
        qlive_spill_field_capped(
            2,
            &mut accs,
            &mut spill,
            &mut live,
            &mut counts,
            &mut cold,
            0
        )?,
        (0, false)
    );
    assert_eq!(live, 2);
    assert_eq!(spill.total_records, 0);
    assert_eq!(spill.run_count, 0);
    assert_eq!(counts, [0, 1, 0, 0]);
    assert_eq!(cold, [0, 1, 0, 0]);
    assert_eq!(
        qlive_spill_field_capped(
            2,
            &mut accs,
            &mut spill,
            &mut live,
            &mut counts,
            &mut cold,
            1
        )?,
        (1, true)
    );
    assert_eq!(live, 1);
    assert_eq!(counts, [0, 1, 1, 0]);
    assert_eq!(cold, [0, 1, 1, 0]);
    assert_eq!(spill.run_count, 1);
    drop(spill);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn spill_roundtrip_matches_reference_union() -> Result<()> {
    let root = std::env::temp_dir().join(format!("zxcl-frontier-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let mut spill = QLiveSpillWriter::create(&root.join("spill.bin"))?;
    let mut accs = accumulators(64);
    let mut expected = accs.clone();
    let mut live: u64 = accs.values().map(|a| a.len() as u64).sum();
    let mut counts = vec![0; 64];
    let mut cold = vec![0; 64];
    let mut candidates = qlive_spill_candidates(&accs, &counts, 1);
    while let Some((count, Reverse(field))) = candidates.pop() {
        let (written, became_cold) = qlive_spill_field_capped(
            field,
            &mut accs,
            &mut spill,
            &mut live,
            &mut counts,
            &mut cold,
            1,
        )?;
        assert_eq!(written, count as u64);
        assert!(became_cold);
        // Later cold contributions contain both an overlapping class with a
        // new mask bit and a new class; finalization must union every run.
        let direct: HashMap<_, _> = [(3, 1 << 27), (17 + field % 3, 1 << 26)]
            .into_iter()
            .collect();
        spill.spill_pairs_ref(field, direct.iter())?;
        for (class, mask) in direct {
            *expected.get_mut(&field).unwrap().entry(class).or_insert(0) |= mask;
        }
    }
    assert_eq!(live, 0);
    let lookup: Vec<_> = (0..20 * AVAIL_PAIR_COUNT).map(|i| i as u32).collect();
    let mut spool = QLiveSpoolWriter::create(&root.join("spool.bin"))?;
    let mut qids = Vec::new();
    let mut expected_states = 0;
    for (&field, classes) in &expected {
        let count = qlive_finalize_field_with_spills(
            field,
            accs.remove(&field).unwrap_or_default(),
            &mut spill,
            &lookup,
            &mut qids,
            &mut spool,
        )?;
        let reference_count: u64 = classes.values().map(|m| m.count_ones() as u64).sum();
        assert_eq!(count, reference_count);
        expected_states += reference_count;
    }
    assert_eq!(spool.total_qids, expected_states);
    spool.flush()?;
    let bytes = std::fs::read(root.join("spool.bin"))?;
    for &(field, count, offset) in &spool.metas {
        let mut reference_qids = Vec::new();
        for (&bag_tail, &mask) in &expected[&field] {
            for avail in 0..28 {
                if mask & (1 << avail) != 0 {
                    reference_qids.push(bag_tail * AVAIL_PAIR_COUNT as u32 + avail);
                }
            }
        }
        reference_qids.sort_unstable();
        let actual: Vec<_> = (0..count as u64)
            .map(|i| crate::layer_store::read_u24_le_at(&bytes, offset + i))
            .collect();
        assert_eq!(actual, reference_qids, "field {field}");
    }
    drop(spool);
    drop(spill);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

// This intentionally uses the production entrypoints, including ordering,
// field closure, direct cold spills, spool assembly, and both RAM/index-only reads. It is
// ignored in the small unit suite because those entrypoints allocate the full
// NUM_FIELDS metadata and compact qid lookup even for this tiny fixture.
#[test]
#[ignore = "release end-to-end frontier validation; existing full field/qid metadata uses about 500 MiB"]
fn frontier_entrypoints_match_unoptimized_reference() -> Result<()> {
    use crate::qstate::QChainState;
    use std::collections::BTreeSet;

    let root =
        std::env::temp_dir().join(format!("zxcl-frontier-entrypoints-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let qgraph = graph_fixture(257);
    let mut rgraph = ReverseGraph {
        offsets: Vec::new(),
        sources: Vec::new(),
    };
    for child in 0..12 {
        let mut offsets = [0; crate::piece::PIECE_COUNT + 1];
        offsets[0] = rgraph.sources.len() as u32;
        for piece in 0..crate::piece::PIECE_COUNT {
            // Child 10 has no contributing parents; child 11 closes an isolated
            // parent. Shared parents 0..23 receive overlapping classes/masks.
            if child == 11 && piece == 0 {
                rgraph.sources.push(31);
            } else if child < 10 && (child + piece) % 3 != 0 {
                for parent in 0..24 {
                    if (parent + child * 3 + piece) % 4 < 2 {
                        rgraph.sources.push(parent as FieldId);
                    }
                }
            }
            offsets[piece + 1] = rgraph.sources.len() as u32;
        }
        rgraph.offsets.push(offsets);
    }
    let mut expected = HashMap::<FieldId, BTreeSet<u32>>::new();
    let prefix = root.join("input");
    let mut writer = FieldChainWriter::create(&prefix)?;
    for child in 0..12 {
        let mut qids: Vec<_> = (0..((child * 7 + 3) % 43 + 1))
            .map(|i| ((child * 19 + i) % 257) as u32)
            .collect();
        qids.sort_unstable();
        qids.dedup();
        for qid in qids {
            writer.push_pair_key(q_pair_key(child as FieldId, qid))?;
            // Independent allocating/reference union: no optimized collector,
            // retained maps, heap, spill helper, or output materializer.
            for piece in 0..crate::piece::PIECE_COUNT as u8 {
                for rec in qgraph.records_for(qid, piece) {
                    for &parent in rgraph.edges(child as FieldId, piece) {
                        for avail in 0..AVAIL_PAIR_COUNT as u8 {
                            if rec.avail_mask & (1 << avail) != 0 {
                                expected
                                    .entry(parent)
                                    .or_default()
                                    .insert(rec.bag_tail * AVAIL_PAIR_COUNT as u32 + avail as u32);
                            }
                        }
                    }
                }
            }
        }
    }
    writer.finish()?;
    let input = FieldChainLayer::open_index_only(&prefix)?;
    let input_in_ram = FieldChainLayer::open(&prefix)?;
    let q_index = QStateIndex {
        states: (0..127)
            .flat_map(|tail5| {
                (0..AVAIL_PAIR_COUNT as u8).map(move |avail| QChainState {
                    tail5,
                    avail,
                    bag_mask: 0,
                })
            })
            .collect(),
        key_to_id: HashMap::new(),
        old_to_qid: Vec::new(),
        count_by_popcount: [0; 8],
    };
    type Step = fn(
        &ReverseGraph,
        &FieldChainLayer,
        &QStateIndex,
        &QRevClassGraph,
        &Path,
        &Path,
        QLiveFrontierSpillConfig,
    ) -> Result<u64>;
    let steps: [(&str, Step); 4] = [
        ("live", |rg, input, qi, qg, out, work, _cfg| {
            external_step_q_live(rg, input, qi, qg, out, work)
        }),
        ("spill", external_step_q_live_frontier_spill),
        ("deferred", external_step_q_live_frontier_spill_deferred),
        ("strong", external_step_q_live_frontier_spill_strong),
    ];
    let configs = [
        QLiveFrontierSpillConfig {
            global_live_class_limit: 0,
            per_field_class_limit: 0,
            max_spills_per_parent: 0,
        },
        QLiveFrontierSpillConfig {
            global_live_class_limit: 0,
            per_field_class_limit: 13,
            max_spills_per_parent: 1,
        },
        QLiveFrontierSpillConfig {
            global_live_class_limit: 17,
            per_field_class_limit: 0,
            max_spills_per_parent: 2,
        },
    ];
    let expected_count: u64 = expected.values().map(|qids| qids.len() as u64).sum();
    for (name, step) in steps {
        for (case, cfg) in configs.into_iter().enumerate() {
            if name == "live" && case != 0 {
                continue;
            }
            // The normal CLI currently opens chains in RAM; retain index-only
            // coverage as well, since the reused read buffer serves both.
            let selected_input = if case % 2 == 0 { &input_in_ram } else { &input };
            let output = root.join(format!("{name}-{case}"));
            let work = root.join(format!("work-{name}-{case}"));
            let count = step(
                &rgraph,
                selected_input,
                &q_index,
                &qgraph,
                &output,
                &work,
                cfg,
            )?;
            assert_eq!(count, expected_count, "{name} case {case}");
            let actual = FieldChainLayer::open(&output)?;
            assert_eq!(actual.metas.len(), expected.len(), "{name} case {case}");
            for (bi, meta) in actual.metas.iter().enumerate() {
                let wanted: Vec<_> = expected[&meta.field_id].iter().copied().collect();
                assert_eq!(
                    actual.read_block_chains(bi),
                    wanted,
                    "{name} case {case} field {}",
                    meta.field_id
                );
            }
        }
    }
    drop(input);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "bounded release-mode microbenchmark; run explicitly with --ignored --nocapture"]
fn benchmark_frontier_workspaces_and_spill_selection() {
    let graph = graph_fixture(257);
    let qids: Vec<_> = (0..64).collect();
    let parents = [0, 3, 0, 1, 9, 1, 4];
    let repetitions = 3000;
    let start = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(collect_reference(&qids, &graph, &parents));
    }
    let old_collect = start.elapsed();
    let mut maps = std::array::from_fn(|_| HashMap::with_capacity(4096));
    let start = Instant::now();
    for _ in 0..repetitions {
        qlive_collect_child_classes(
            &qids,
            &graph,
            &parents,
            &mut maps,
            &mut QLiveStats::default(),
        );
        std::hint::black_box(&maps);
        for map in &mut maps {
            qlive_reset_piece_classes(map);
        }
    }
    let new_collect = start.elapsed();
    let original = accumulators(20_000);
    let counts = vec![0; 20_000];
    let mut old = original.clone();
    let start = Instant::now();
    for _ in 0..4000 {
        let field = *old
            .iter()
            .filter(|(_, a)| !a.is_empty())
            .max_by_key(|(f, a)| (a.len(), Reverse(**f)))
            .unwrap()
            .0;
        std::hint::black_box(old.remove(&field));
    }
    let old_select = start.elapsed();
    let mut new = original;
    let start = Instant::now();
    let mut candidates = qlive_spill_candidates(&new, &counts, 1);
    let metadata_bytes = candidates.capacity() * std::mem::size_of::<(usize, Reverse<FieldId>)>();
    for _ in 0..4000 {
        let (_, Reverse(field)) = candidates.pop().unwrap();
        std::hint::black_box(new.remove(&field));
    }
    let new_select = start.elapsed();
    eprintln!("child collection: allocating={old_collect:?} reused={new_collect:?}; global spill selection: repeated_scan={old_select:?} heap={new_select:?}; heap_metadata_bytes={metadata_bytes}");
}
