use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn graph() -> FieldGraph {
    let rows: [[&[u32]; 2]; 6] = [
        [&[3, 3, 4, EMPTY_FIELD_ID], &[4, 3]],
        [&[4, 5], &[3]],
        [&[], &[]],
        [&[], &[]],
        [&[3, 5], &[5]],
        [&[], &[]],
    ];
    let mut offsets = Vec::new();
    let mut targets = Vec::new();
    for row in rows {
        let mut off = [targets.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if placed < 2 {
                targets.extend_from_slice(row[placed]);
            }
            off[placed + 1] = targets.len() as u32;
        }
        offsets.push(off);
    }
    FieldGraph {
        hashes: (0..6).collect(),
        offsets,
        targets,
    }
}

fn qforward() -> QForwardGraph {
    let children: [[&[u32]; 2]; 4] = [[&[0, 1], &[2, 3]], [&[3, 0], &[1]], [&[], &[2]], [&[], &[]]];
    let mut offsets = Vec::new();
    let mut records = Vec::new();
    for row in children {
        let mut off = [records.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if placed < 2 {
                records.extend(row[placed].iter().enumerate().map(|(reveal, &child_qid)| {
                    qgraph::QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    }
                }));
            }
            off[placed + 1] = records.len() as u32;
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

fn layer(prefix: &Path, members: &[(u32, &[u32])]) -> layer_store::FieldChainLayer {
    let mut writer = layer_store::FieldChainWriter::create(prefix).unwrap();
    for &(field, qids) in members {
        for &qid in qids {
            writer.push_pair_key(q_pair_key(field, qid)).unwrap();
        }
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
}

fn scalar_members(
    depth: usize,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    current: &[(u32, &[u32])],
    backward: &[(u32, &[u32])],
) -> BTreeMap<u32, BTreeSet<u32>> {
    let mut expected = BTreeMap::<u32, BTreeSet<u32>>::new();
    for &(field, qids) in current {
        if depth == 5 && field == 4 {
            continue;
        }
        for &qid in qids {
            for placed in 0..PIECE_COUNT {
                let offsets = &qfwd.offsets[qid as usize];
                let recs = &qfwd.records[offsets[placed] as usize..offsets[placed + 1] as usize];
                for &target in graph.edges(field, placed as u8) {
                    if depth == 4 && target == 4 {
                        continue;
                    }
                    if depth == 9 {
                        continue;
                    }
                    let Some((_, allowed)) = backward.iter().find(|(f, _)| *f == target) else {
                        continue;
                    };
                    for rec in recs {
                        if allowed.contains(&rec.child_qid) {
                            expected.entry(target).or_default().insert(rec.child_qid);
                        }
                    }
                }
            }
        }
    }
    expected
}

/// Build the on-disk reference directly from the format contract, independently
/// of both the shard encoder and FieldChainWriter's metadata bookkeeping.
fn file_reference(members: &BTreeMap<u32, BTreeSet<u32>>) -> (Vec<u8>, Vec<u8>) {
    let mut index = b"FCHNIDX1".to_vec();
    index.extend_from_slice(&1u32.to_le_bytes());
    index.extend_from_slice(&(members.len() as u32).to_le_bytes());
    index.extend_from_slice(&(members.values().map(|q| q.len() as u64).sum::<u64>()).to_le_bytes());
    let mut chains = Vec::new();
    let mut offset = 0u64;
    for (&field, qids) in members {
        index.extend_from_slice(&field.to_le_bytes());
        index.extend_from_slice(&(qids.len() as u32).to_le_bytes());
        index.extend_from_slice(&offset.to_le_bytes());
        for &qid in qids {
            chains.extend_from_slice(&qid.to_le_bytes()[..3]);
        }
        offset += qids.len() as u64;
    }
    (index, chains)
}

#[test]
fn compact_target_owned_generation_preserves_scalar_membership_and_both_final_files() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qfwd = qforward();
    let current: [(u32, &[u32]); 3] = [(0, &[0, 1, 2, 3]), (1, &[0, 2]), (4, &[0, 1, 3])];
    let backward: [(u32, &[u32]); 3] = [(3, &[0, 1, 3]), (4, &[1, 2]), (5, &[0, 2])];
    let cur = layer(&owned.path.join("current"), &current);
    let next = layer(&owned.path.join("backward"), &backward);
    let sources = [0, 1, 2];
    for depth in [3usize, 4, 5, 9] {
        let expected = scalar_members(depth, &graph, &qfwd, &current, &backward);
        let (expected_index, expected_chains) = file_reference(&expected);
        for format in [
            TargetOwnedShardFormat::PairsU64,
            TargetOwnedShardFormat::CompactU24,
        ] {
            let prefix = owned.path.join(format!("final-{depth}-{format:?}"));
            let mut writer = layer_store::FieldChainWriter::create(&prefix).unwrap();
            let mut total = 0;
            for (block, &(field, _)) in backward.iter().enumerate() {
                let path = owned.path.join(format!("shard-{depth}-{field}-{format:?}"));
                // NUM_FIELDS shards makes this tiny fixture's target field ID
                // its shard ID, without allocating a NUM_FIELDS-sized vector.
                let shard = process_forward_target_owned_range_shard_with_format(
                    depth,
                    field as usize,
                    &sources,
                    &[block],
                    &cur,
                    &graph,
                    &qfwd,
                    &next,
                    4,
                    NUM_FIELDS,
                    1,
                    &path,
                    format,
                )
                .unwrap();
                let width = if format == TargetOwnedShardFormat::PairsU64 {
                    8
                } else {
                    3
                };
                assert_eq!(
                    fs::metadata(&path).unwrap().len(),
                    shard.output_pairs * width
                );
                assert_eq!(
                    shard.metas.iter().map(|m| u64::from(m.count)).sum::<u64>(),
                    shard.output_pairs
                );
                total += append_target_owned_output_to_writer(&shard, &mut writer).unwrap();
            }
            assert_eq!(writer.finish().unwrap(), total);
            assert_eq!(
                fs::read(prefix.with_extension("field_index.bin")).unwrap(),
                expected_index
            );
            assert_eq!(
                fs::read(prefix.with_extension("chains.u24.bin")).unwrap(),
                expected_chains
            );
            let reopened = layer_store::FieldChainLayer::open_index_only(&prefix).unwrap();
            assert_eq!(reopened.total_states, total);
        }
    }
}

fn compact_output(
    path: &Path,
    metas: Vec<layer_store::FieldChainBlockMeta>,
    pairs: u64,
) -> TargetOwnedShardOutput {
    TargetOwnedShardOutput {
        shard: 0,
        stats: ForwardPrefilterChunkStats::default(),
        output_fields: metas.len(),
        output_pairs: pairs,
        path: path.to_path_buf(),
        format: TargetOwnedShardFormat::CompactU24,
        metas,
    }
}

#[test]
fn multi_field_target_shard_rebases_every_field_exactly() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let graph = graph();
    let qfwd = qforward();
    let current: [(u32, &[u32]); 3] = [(0, &[0, 1, 2, 3]), (1, &[0, 2]), (4, &[0, 1, 3])];
    let backward: [(u32, &[u32]); 3] = [(3, &[0, 1, 3]), (4, &[1, 2]), (5, &[0, 2])];
    let cur = layer(&owned.path.join("current"), &current);
    let next = layer(&owned.path.join("backward"), &backward);
    let expected = scalar_members(3, &graph, &qfwd, &current, &backward);
    let (expected_index, expected_chains) = file_reference(&expected);
    assert_eq!(expected.len(), 3);
    for format in [
        TargetOwnedShardFormat::PairsU64,
        TargetOwnedShardFormat::CompactU24,
    ] {
        let path = owned.path.join(format!("multi-shard-{format:?}"));
        let shard = process_forward_target_owned_range_shard_with_format(
            3,
            0,
            &[0, 1, 2],
            &[0, 1, 2],
            &cur,
            &graph,
            &qfwd,
            &next,
            4,
            1,
            1,
            &path,
            format,
        )
        .unwrap();
        let mut offset = 0u64;
        for (meta, (&field, qids)) in shard.metas.iter().zip(&expected) {
            assert_eq!(meta.field_id, field);
            assert_eq!(meta.count as usize, qids.len());
            assert_eq!(meta.chain_offset, offset);
            offset += qids.len() as u64;
        }
        assert_eq!(shard.output_fields, expected.len());
        assert_eq!(shard.output_pairs, offset);
        let prefix = owned.path.join(format!("multi-final-{format:?}"));
        let mut writer = layer_store::FieldChainWriter::create(&prefix).unwrap();
        assert_eq!(
            append_target_owned_output_to_writer(&shard, &mut writer).unwrap(),
            offset
        );
        assert_eq!(writer.finish().unwrap(), offset);
        assert_eq!(
            fs::read(prefix.with_extension("field_index.bin")).unwrap(),
            expected_index
        );
        assert_eq!(
            fs::read(prefix.with_extension("chains.u24.bin")).unwrap(),
            expected_chains
        );
    }
}

#[test]
fn corrupt_compact_shard_cannot_publish_a_final_index() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    for kind in 0..7 {
        let path = owned.path.join(format!("bad-shard-{kind}"));
        let bytes = match kind {
            0 => vec![0, 0],             // partial u24
            1 => vec![0, 0, 0, 1, 0, 0], // metadata gap
            2 => vec![1, 0, 0, 1, 0, 0], // duplicate qid
            3 => vec![2, 0, 0, 1, 0, 0], // descending qid
            6 => vec![],                 // nonempty metadata with a truncated file
            _ => vec![0, 0, 0, 1, 0, 0],
        };
        fs::write(&path, bytes).unwrap();
        let mut metas = vec![layer_store::FieldChainBlockMeta {
            field_id: 3,
            count: 2,
            chain_offset: 0,
        }];
        if kind == 1 {
            metas[0].chain_offset = 1;
        }
        if kind == 4 {
            metas[0].count = 1;
        } // stream longer than metadata
        if kind == 5 {
            metas = vec![
                layer_store::FieldChainBlockMeta {
                    field_id: 4,
                    count: 1,
                    chain_offset: 0,
                },
                layer_store::FieldChainBlockMeta {
                    field_id: 3,
                    count: 1,
                    chain_offset: 1,
                },
            ];
        }
        let output = compact_output(&path, metas, 2);
        let prefix = owned.path.join(format!("bad-final-{kind}"));
        let mut writer = layer_store::FieldChainWriter::create(&prefix).unwrap();
        assert!(append_target_owned_output_to_writer(&output, &mut writer).is_err());
        assert!(
            writer.finish().is_err(),
            "a caller ignoring the append error cannot publish an index"
        );
        assert!(!prefix.with_extension("field_index.bin").exists());
    }
}

#[test]
fn shard_scratch_and_failed_publication_preserve_unrelated_files() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let unrelated = owned.path.join("unrelated");
    fs::write(&unrelated, b"preserve me").unwrap();
    let scratch = TargetOwnedShardScratch::create(&owned.path).unwrap();
    let scratch_path = scratch.path.clone();
    fs::write(scratch.path.join("pending-worker"), b"owned temporary").unwrap();
    drop(scratch);
    assert!(!scratch_path.exists());
    assert_eq!(fs::read(&unrelated).unwrap(), b"preserve me");

    let graph = graph();
    let qfwd = qforward();
    let cur = layer(&owned.path.join("current"), &[(0, &[0, 1])]);
    let next = layer(&owned.path.join("backward"), &[(3, &[0, 1, 3])]);
    let destination = owned.path.join("existing-directory");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("accepted"), b"old accepted shard").unwrap();
    assert!(process_forward_target_owned_range_shard(
        3,
        3,
        &[0],
        &[0],
        &cur,
        &graph,
        &qfwd,
        &next,
        4,
        NUM_FIELDS,
        1,
        &destination,
    )
    .is_err());
    assert_eq!(
        fs::read(destination.join("accepted")).unwrap(),
        b"old accepted shard"
    );
    assert!(fs::read_dir(&owned.path).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".pending.")));
}
