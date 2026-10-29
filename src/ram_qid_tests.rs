use super::*;

fn fixture(prefix: &Path, blocks: &[(FieldId, &[u32])]) -> layer_store::FieldChainLayer {
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

fn configure(lookup: &mut RamNextValueLookup<'_>, mode: usize) {
    lookup.bitmap_min_count = if mode == 0 { usize::MAX } else { 1 };
    lookup.dense_min_count = if mode == 2 { 1 } else { usize::MAX };
    lookup.dense_heat_min = 1;
    lookup.dense_max_blocks = 1;
}

fn reference_sum(qids: &[u32], values: &[f32], recs: &[qgraph::QForwardRec]) -> (f32, u64) {
    let mut sum = 0.0f32;
    let mut missing = 0;
    for rec in recs {
        let value = qids
            .binary_search(&rec.child_qid)
            .ok()
            .map_or(0.0, |i| values[i]);
        sum += value;
        if value == 0.0 {
            missing += 1;
        }
    }
    (sum, missing)
}

#[test]
fn sparse_bitmap_and_heat_promoted_dense_match_get_and_sum_bits() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let qids = (0..1_030u32).filter(|qid| qid % 3 != 1).collect::<Vec<_>>();
    let layer = fixture(&owned.path.join("next"), &[(5, &qids)]);
    let values = (0..qids.len())
        .map(|i| match i % 7 {
            0 => -0.0,
            1 => 0.0,
            2 => 1.0e20,
            3 => -1.0e20,
            4 => 3.125,
            5 => -0.25,
            _ => 0.0625,
        })
        .collect::<Vec<_>>();
    let children = [0, 1, 63, 64, 65, 127, 128, 129, 1_029, 1_030, u32::MAX];
    for mode in 0..3 {
        let mut lookup =
            RamNextValueLookup::new(&layer, &values, 8 * 1024 * 1024, true, true).unwrap();
        configure(&mut lookup, mode);
        lookup.prewarm(5).unwrap();
        // In dense mode this first hit crosses the heat gate. Other modes retain
        // their indexes, including gaps that must keep contributing exact zero.
        lookup.get(5, 0).unwrap();
        let block = &lookup.cache.get(&5).unwrap().block;
        match (&block.index, mode) {
            (RamQidIndex::Sparse(_), 0)
            | (RamQidIndex::Bitmap(_), 1)
            | (RamQidIndex::Dense(_), 2) => {}
            _ => panic!("wrong index selected for mode={mode}"),
        }
        assert_eq!(lookup.cache_bytes, block.cache_bytes());
        assert_eq!(lookup.dense_live, usize::from(mode == 2));
        for qid in (0..1_200).chain([CHECK_QSTATE_COUNT as u32, u32::MAX]) {
            let expected = qids.binary_search(&qid).ok().map_or(0.0, |i| values[i]);
            assert_eq!(
                lookup.get(5, qid).unwrap().to_bits(),
                expected.to_bits(),
                "mode={mode} qid={qid}"
            );
        }
        for n in [0usize, 1, 7, 8, children.len()] {
            let recs = children[..n]
                .iter()
                .enumerate()
                .map(|(i, &child_qid)| qgraph::QForwardRec {
                    reveal: i as u8,
                    child_qid,
                })
                .collect::<Vec<_>>();
            let expected = reference_sum(&qids, &values, &recs);
            let actual = lookup.sum_recs(5, &recs).unwrap();
            assert_eq!(
                actual.0.to_bits(),
                expected.0.to_bits(),
                "mode={mode} n={n}"
            );
            assert_eq!(actual.1, expected.1, "mode={mode} n={n}");
            assert_eq!(lookup.sum_recs(99, &recs).unwrap(), (0.0, n as u64));
        }
        assert_eq!(lookup.const_value(5).unwrap(), None);
        let stats = lookup.stats_snapshot();
        assert_eq!(stats.sparse_blocks, u64::from(mode == 0));
        assert_eq!(stats.bitmap_blocks, u64::from(mode != 0));
        assert_eq!(stats.dense_blocks, u64::from(mode == 2));
    }
}

#[test]
fn bitmap_selection_requires_benefit_and_boxed_charge_covers_allocated_payload() {
    let mut oversized_vec = Vec::with_capacity(8_192);
    oversized_vec.extend([0, 64, 128]);
    let sparse = RamQidBlock::new(oversized_vec, 17, 1).unwrap();
    assert!(matches!(sparse.index, RamQidIndex::Sparse(_)));
    assert_eq!(sparse.cache_bytes(), 256 + 3 * 4);
    assert_eq!(sparse.value_index(64), Some(18));
    let qids = (0..512u32).filter(|qid| qid % 5 != 0).collect::<Vec<_>>();
    let bitmap = RamQidBlock::new(qids.clone(), 9, 1).unwrap();
    assert!(matches!(bitmap.index, RamQidIndex::Bitmap(_)));
    assert_eq!(
        bitmap.cache_bytes(),
        256 + qid_rank::QidRankIndex::estimated_heap_bytes(&qids)
    );
    let disabled = RamQidBlock::new(qids.clone(), 9, usize::MAX).unwrap();
    assert!(matches!(disabled.index, RamQidIndex::Sparse(_)));
    assert_eq!(disabled.cache_bytes(), 256 + qids.len() * 4);
}

#[test]
fn bitmap_rank_and_dense_promotion_keep_original_nonzero_chain_offset() {
    let qids = (61..513u32).filter(|qid| qid % 7 != 4).collect::<Vec<_>>();
    let offset = 19;
    let mut values = vec![12345.0f32; offset];
    values.extend(
        qids.iter()
            .enumerate()
            .map(|(i, _)| (i as f32 + 1.0) / 32.0),
    );
    let mut block = RamQidBlock::new(qids.clone(), offset, 1).unwrap();
    assert!(matches!(block.index, RamQidIndex::Bitmap(_)));
    for (i, &qid) in qids.iter().enumerate() {
        assert_eq!(block.value_index(qid), Some(offset + i));
    }
    block.promote(&values);
    let dense = block.dense_values().unwrap();
    for qid in 0..600u32 {
        let expected = qids
            .binary_search(&qid)
            .ok()
            .map_or(0.0, |i| values[offset + i]);
        assert_eq!(
            dense[qid as usize].to_bits(),
            expected.to_bits(),
            "qid={qid}"
        );
    }
    assert_eq!(block.cache_bytes(), CHECK_QSTATE_COUNT * 4 + 256);
}

#[test]
fn independent_lookup_values_and_evictions_do_not_reuse_old_dense_copies() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let qids = (0..256u32).filter(|qid| qid % 3 != 1).collect::<Vec<_>>();
    let layer = fixture(
        &owned.path.join("next"),
        &[(1, &qids), (2, &qids), (3, &qids)],
    );
    let first = (0..layer.total_states as usize)
        .map(|i| (i as f32 + 1.0) / 1_024.0)
        .collect::<Vec<_>>();
    let second = first.iter().map(|value| value + 0.5).collect::<Vec<_>>();
    // A single dense array exceeds this budget: established behavior retains that
    // sole oversized block, evicts it on the next field, and charges its full size.
    for values in [&first, &second] {
        let mut lookup = RamNextValueLookup::new(&layer, values, 2_048, true, false).unwrap();
        configure(&mut lookup, 2);
        for field in [1, 2, 3, 1, 3, 2] {
            let offset = layer.metas[(field - 1) as usize].chain_offset as usize;
            let field_values = &values[offset..offset + qids.len()];
            assert_eq!(lookup.get(field, 0).unwrap(), field_values[0]);
            assert_eq!(lookup.get(field, 0).unwrap(), field_values[0]);
            assert_eq!(lookup.get(field, 1).unwrap(), 0.0);
            let recs = [0, 1, 63, 64, 65, 127, 128]
                .into_iter()
                .enumerate()
                .map(|(i, child_qid)| qgraph::QForwardRec {
                    reveal: i as u8,
                    child_qid,
                })
                .collect::<Vec<_>>();
            let expected = reference_sum(&qids, field_values, &recs);
            let actual = lookup.sum_recs(field, &recs).unwrap();
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1, expected.1);
            assert_eq!(lookup.cache.len(), 1);
            assert_eq!(lookup.dense_live, 1);
            assert_eq!(lookup.cache_bytes, CHECK_QSTATE_COUNT * 4 + 256);
        }
        assert!(lookup.stats_snapshot().evictions >= 5);
    }
}

#[test]
fn empty_meta_and_absent_fields_do_not_allocate_cache_entries() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let mut layer = fixture(&owned.path.join("next"), &[]);
    layer.metas.push(layer_store::FieldChainBlockMeta {
        field_id: 7,
        count: 0,
        chain_offset: 0,
    });
    let mut lookup = RamNextValueLookup::new(&layer, &[], 0, true, false).unwrap();
    for _ in 0..20 {
        assert_eq!(lookup.get(7, 0).unwrap(), 0.0);
        assert_eq!(lookup.get(8, 0).unwrap(), 0.0);
    }
    assert!(lookup.cache.is_empty());
    assert!(lookup.cache_order.is_empty());
    assert_eq!(lookup.cache_bytes, 0);
}

/// Includes positional qid reads, index construction, cache insertion/eviction and
/// actual pinned-value reads. Run explicitly in release mode; results complement,
/// rather than replace, representative Bellman sweep measurements.
#[test]
#[ignore]
fn benchmark_ram_qid_cold_and_warm_release() {
    use std::hint::black_box;
    use std::time::Instant;
    assert!(!cfg!(debug_assertions), "run this benchmark with --release");
    const FIELDS: usize = 16;
    const REPEATS: usize = 3;
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    for count in [512usize, 40_000, 100_000] {
        // The small case is clustered and byte-beneficial; the larger cases span
        // the entire production qstate domain and exercise the crossover region.
        let qids = (0..count)
            .map(|i| {
                if count == 512 {
                    i as u32
                } else {
                    (i * CHECK_QSTATE_COUNT / count) as u32
                }
            })
            .collect::<Vec<_>>();
        let blocks = (1..=FIELDS as u32)
            .map(|field| (field, qids.as_slice()))
            .collect::<Vec<_>>();
        let layer = fixture(&owned.path.join(format!("next-{count}")), &blocks);
        let values = (0..layer.total_states as usize)
            .map(|i| (i % 1009) as f32 / 1024.0)
            .collect::<Vec<_>>();
        for cache_cap in [256 * 1024usize, 16 * 1024 * 1024] {
            for queries_per_field in [3usize, 7, 1000] {
                let queries = (0..queries_per_field)
                    .map(|i| {
                        // Include present and absent children with a deterministic mix.
                        if i % 3 == 1 {
                            ((i * 197 + 31) % CHECK_QSTATE_COUNT) as u32
                        } else {
                            qids[(i * 137 + 5) % count]
                        }
                    })
                    .collect::<Vec<_>>();
                let mut checksums = [0.0f64; 2];
                for mode in 0..2 {
                    let start = Instant::now();
                    let mut checksum = 0.0f64;
                    let mut evictions = 0;
                    let mut materialized = 0;
                    let mut bitmap_blocks = 0;
                    for _ in 0..REPEATS {
                        let mut lookup =
                            RamNextValueLookup::new(&layer, &values, cache_cap, true, false)
                                .unwrap();
                        configure(&mut lookup, mode);
                        for field in 1..=FIELDS as u32 {
                            for &qid in black_box(&queries) {
                                checksum += black_box(lookup.get(field, qid).unwrap()) as f64;
                            }
                        }
                        let stats = lookup.stats_snapshot();
                        evictions += stats.evictions;
                        materialized += stats.materialized_states;
                        bitmap_blocks += stats.bitmap_blocks;
                    }
                    let elapsed = start.elapsed();
                    checksums[mode] = black_box(checksum);
                    eprintln!("ram-qid-build-bench count={count} queries_per_field={queries_per_field} cache_cap_bytes={cache_cap} mode={} fields={} materialized={materialized} bitmap_blocks={bitmap_blocks} evictions={evictions} elapsed_ms={:.3} us_per_field={:.3}",
                        if mode == 0 { "sparse" } else { "bitmap" }, FIELDS * REPEATS,
                        elapsed.as_secs_f64() * 1000.0, elapsed.as_secs_f64() * 1.0e6 / (FIELDS * REPEATS) as f64);
                }
                assert_eq!(
                    checksums[0].to_bits(),
                    checksums[1].to_bits(),
                    "count={count} queries={queries_per_field} cap={cache_cap}"
                );
            }
        }
    }
}
