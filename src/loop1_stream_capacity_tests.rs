use super::*;

struct TinyFormulas;

impl TerminalFormulaIndex for TinyFormulas {
    fn terminal_count(&self) -> usize {
        3
    }
    fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        (qid < 3 && placed == 0).then_some(qid)
    }
    fn reveal_qids(&self, zid: u32) -> &[u32] {
        match zid {
            0 => &[0],
            1 => &[1],
            2 => &[2],
            _ => &[],
        }
    }
}

fn with_tiny_l8(check: impl FnOnce(&CompiledL8<'_>)) {
    with_tiny_l8_ranked(false, check)
}

fn with_tiny_l8_ranked(ranked: bool, check: impl FnOnce(&CompiledL8<'_>)) {
    use crate::layer_store::FieldChainWriter;
    use crate::qstate::{encode_avail_pair, QChainState};
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let graph = FieldGraph {
        hashes: vec![0, 1],
        offsets: vec![[0, 1, 1, 1, 1, 1, 1, 1], [1, 2, 2, 2, 2, 2, 2, 2]],
        targets: vec![1, EMPTY_FIELD_ID],
    };
    let q_index = QStateIndex {
        states: (0..4)
            .map(|qid| QChainState {
                avail: encode_avail_pair(0, 1),
                tail5: qid,
                bag_mask: 1,
            })
            .collect(),
        key_to_id: HashMap::new(),
        old_to_qid: Vec::new(),
        count_by_popcount: [0; 8],
    };
    let qfwd = QForwardGraph {
        offsets: (0..4)
            .map(|i| [i, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1])
            .collect(),
        records: (0..4)
            .map(|child_qid| QForwardRec {
                reveal: 0,
                child_qid,
            })
            .collect(),
        records_by_piece: [4, 0, 0, 0, 0, 0, 0],
        max_records_per_piece: [1, 0, 0, 0, 0, 0, 0],
        skipped_invalid_child: 0,
    };
    let mut layers = Vec::new();
    let mut policies = Vec::new();
    for (number, field) in [(8, 0u32), (9, 1u32)] {
        let prefix = owned.path.join(format!("l{number}"));
        let mut writer = FieldChainWriter::create(&prefix).unwrap();
        for qid in 0..3u32 {
            writer
                .push_pair_key((field as u64) << 32 | qid as u64)
                .unwrap();
        }
        writer.finish().unwrap();
        layers.push(FieldChainLayer::open_index_only(&prefix).unwrap());
        let policy = owned.path.join(format!("p{number}"));
        fs::write(
            &policy,
            [
                0,
                0,
                if number == 9 { 255 } else { 0 },
                if number == 9 { 255 } else { 0 },
                0,
                0,
            ],
        )
        .unwrap();
        policies.push(policy);
    }
    let limits = Loop1Limits {
        max_mapping_bytes: 1 << 20,
        max_unique_averages: 32,
        max_cache_bytes: 1_024,
    };
    let pre = TinyFormulas;
    let mut l9 = CompiledL9::build(
        &graph,
        &q_index,
        &qfwd,
        &pre,
        &layers[1],
        &policies[1],
        limits,
    )
    .unwrap();
    if ranked {
        l9.build_rank_index(1 << 20, 1 << 20).unwrap();
    }
    let l8 = CompiledL8::build_with_heap_limit(
        &graph,
        &q_index,
        &qfwd,
        &pre,
        &layers[0],
        &policies[0],
        0..1,
        &l9,
        limits,
        1 << 20,
    )
    .unwrap();
    check(&l8);
}

fn reserve_unique(
    count: usize,
    cap: u64,
) -> (
    Result<()>,
    Loop1Stats,
    Vec<AverageSignature>,
    HashMap<AverageSignature, u32>,
) {
    let budget = CompilerHeapBudget {
        max_bytes: cap,
        fixed_bytes: 1_024,
    };
    let mut stats = Loop1Stats::default();
    let mut averages = Vec::new();
    let mut interner = HashMap::new();
    let result = (|| {
        for i in 0..count {
            budget.reserve_average(&mut averages, &mut interner, &mut stats)?;
            let signature = AverageSignature {
                denominator: 1,
                children: [i as u32, 0, 0, 0, 0, 0, 0],
            };
            averages.push(signature);
            interner.insert(signature, i as u32 + 1);
            stats.scanned_states += 1;
            stats.unique_averages = averages.len();
        }
        Ok(())
    })();
    (result, stats, averages, interner)
}

#[test]
fn compiler_capacity_cap_precedes_mapping_and_unique_node_growth() {
    let (result, stats, averages, interner) = reserve_unique(10_000, u64::MAX);
    result.unwrap();
    assert_eq!(averages.len(), 10_000);
    assert_eq!(interner.len(), 10_000);
    assert!(
        stats.interner_peak_capacity_bytes >= 10_000 * size_of::<(AverageSignature, u32)>() as u64
    );
    assert!(
        stats.compiler_peak_capacity_bytes
            > 1_024 + vec_capacity_bytes(&averages) + stats.interner_peak_capacity_bytes
    );

    let exact = stats.compiler_peak_capacity_bytes;
    reserve_unique(10_000, exact).0.unwrap();
    let (result, stopped, averages, interner) = reserve_unique(10_000, exact - 1);
    let error = result.unwrap_err().to_string();
    assert!(error.contains("before"), "{error}");
    assert!(error.contains("scanned="), "{error}");
    assert!(error.contains("unique="), "{error}");
    assert!(stopped.compiler_peak_capacity_bytes < exact);
    assert!(averages.len() < 10_000);
    assert_eq!(averages.len(), interner.len());

    let (result, _, averages, interner) = reserve_unique(1, 0);
    assert!(result.is_err());
    assert_eq!(averages.capacity(), 0);
    assert_eq!(interner.capacity(), 0);
}

#[test]
fn conservative_hash_reservation_covers_actual_std_capacity_at_each_rehash() {
    let mut table = HashMap::<AverageSignature, u32>::new();
    for i in 0..10_000 {
        let needed = table.len() + 1;
        if needed > table.capacity() {
            let upper = hash_reserve_upper_bytes::<AverageSignature, u32>(needed).unwrap();
            table.try_reserve(1).unwrap();
            assert!(
                hash_capacity_bytes::<AverageSignature, u32>(table.capacity()).unwrap() <= upper
            );
        }
        table.insert(
            AverageSignature {
                denominator: 1,
                children: [i, 0, 0, 0, 0, 0, 0],
            },
            i,
        );
    }
    assert!(hash_reserve_upper_bytes::<AverageSignature, u32>(usize::MAX).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn bound_input_notifications_reject_writes_even_when_metadata_snapshot_matches() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let path = owned.path.join("same-tick-source");
    fs::write(&path, [1u8, 2, 3]).unwrap();
    let mut bound = BoundFile::open(&path).unwrap();
    bound.check().unwrap();
    fs::write(&path, [3u8, 2, 1]).unwrap();
    // Make the metadata check uninformative deliberately. This models coarse
    // filesystem timestamps without sleeping or weakening the public check.
    bound.stamp = FileStamp::of(&bound.file.metadata().unwrap());
    bound.request_stamp = FileStamp::of(&fs::symlink_metadata(&path).unwrap());
    for _ in 0..3 {
        assert!(bound
            .check()
            .unwrap_err()
            .to_string()
            .contains("mutation observed"));
    }
}

#[test]
fn sparse_l9_binding_requires_shared_rank_indexes_and_original_handles_and_domain() {
    for ranked in [false, true] {
        with_tiny_l8_ranked(ranked, |l8| {
            let compiled = l8.l9;
            let values = compiled.prepare_f32(&[1.0, -0.0, -2.0]).unwrap();
            let domain = &compiled.domain;
            let index = compiled.inputs.index.file.try_clone().unwrap();
            let membership = compiled.inputs.membership.file.try_clone().unwrap();
            let policy = compiled.inputs.policy.file.try_clone().unwrap();
            let original = values.check_sparse_binding(
                domain.graph,
                domain.qfwd,
                domain.pre,
                &index,
                &membership,
                &policy,
            );
            if !ranked {
                assert!(original.unwrap_err().to_string().contains("rank"));
                return;
            }
            original.unwrap();
            values.check_inputs_unchanged().unwrap();
            let other_graph = FieldGraph {
                hashes: domain.graph.hashes.clone(),
                offsets: domain.graph.offsets.clone(),
                targets: domain.graph.targets.clone(),
            };
            assert!(values
                .check_sparse_binding(
                    &other_graph,
                    domain.qfwd,
                    domain.pre,
                    &index,
                    &membership,
                    &policy
                )
                .unwrap_err()
                .to_string()
                .contains("domain"));
            let parent = compiled.inputs.index.request.parent().unwrap();
            for (name, source) in [
                ("index", &compiled.inputs.index),
                ("membership", &compiled.inputs.membership),
                ("policy", &compiled.inputs.policy),
            ] {
                let other_path = parent.join(format!("other-{name}"));
                fs::copy(&source.request, &other_path).unwrap();
                let other = File::open(other_path).unwrap();
                let (actual_index, actual_membership, actual_policy) = match name {
                    "index" => (&other, &membership, &policy),
                    "membership" => (&index, &other, &policy),
                    _ => (&index, &membership, &other),
                };
                let error = values
                    .check_sparse_binding(
                        domain.graph,
                        domain.qfwd,
                        domain.pre,
                        actual_index,
                        actual_membership,
                        actual_policy,
                    )
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(&format!("binding mismatch: {name}")),
                    "{error}"
                );
            }
        });
    }
}

#[test]
fn l9_formula_lookup_retains_policy_ids_missing_qids_and_none_without_numeric_values() {
    with_tiny_l8_ranked(false, |l8| {
        assert!(l8.l9.formula_lookup().is_err());
    });
    with_tiny_l8_ranked(true, |l8| {
        let compiled = l8.l9;
        let mut lookup = compiled.formula_lookup().unwrap();
        assert_eq!(lookup.get(1, 0).unwrap(), Some(0));
        assert_eq!(lookup.get(1, 1).unwrap(), None); // Stored POLICY_NONE.
        assert_eq!(lookup.get(1, 2).unwrap(), Some(2));
        assert_eq!(lookup.get(1, 3).unwrap(), None); // Valid qid absent from L9.
        assert_eq!(lookup.get(0, 0).unwrap(), None); // Field absent from L9.
        assert!(lookup.get(1, 4).is_err()); // Outside immutable qid domain.
        assert_eq!(lookup.io_stats().membership_read_calls, 0);
        assert_eq!(lookup.io_stats().membership_bytes_read, 0);
        assert_eq!(lookup.io_stats().policy_read_calls, 0);

        let domain = &compiled.domain;
        compiled
            .check_sparse_binding(
                domain.graph,
                domain.qfwd,
                domain.pre,
                &compiled.inputs.index.file,
                &compiled.inputs.membership.file,
                &compiled.inputs.policy.file,
            )
            .unwrap();
        let mut changed = fs::read(&compiled.inputs.policy.request).unwrap();
        changed[0] ^= 1;
        fs::write(&compiled.inputs.policy.request, changed).unwrap();
        assert!(compiled.formula_lookup().is_err());
        assert!(compiled.check_inputs_unchanged().is_err());
        assert!(compiled
            .check_sparse_binding(
                domain.graph,
                domain.qfwd,
                domain.pre,
                &compiled.inputs.index.file,
                &compiled.inputs.membership.file,
                &compiled.inputs.policy.file,
            )
            .is_err());
    });
}

#[test]
fn parallel_unique_evaluation_and_bounded_gather_preserve_bits_and_validate_ranges() {
    with_tiny_l8(|l8| {
        let terminals = [16_777_216.0, -3.25, -16_777_216.0];
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let averages = one
            .install(|| l8.evaluate_averages_f32(&terminals))
            .unwrap();
        assert_eq!(
            bits(&averages),
            bits(
                &four
                    .install(|| l8.evaluate_averages_f32(&terminals))
                    .unwrap()
            )
        );
        let expected = [16_777_216.0, 0.0, -16_777_216.0];
        assert_eq!(bits(&l8.evaluate_f32(&terminals).unwrap()), bits(&expected));
        let mut tail = [f32::NAN; 2];
        four.install(|| l8.gather_range_into_f32(&averages, 1..3, &mut tail))
            .unwrap();
        assert_eq!(bits(&tail), bits(&expected[1..]));
        let mut chunks = Vec::new();
        l8.gather_chunks_f32(&averages, 2, |offset, values| {
            assert_eq!(offset, chunks.len());
            assert!(values.len() <= 2);
            chunks.extend_from_slice(values);
            Ok(())
        })
        .unwrap();
        assert_eq!(bits(&chunks), bits(&expected));
        assert!(l8.gather_chunks_f32(&averages, 0, |_, _| Ok(())).is_err());
        assert!(l8
            .gather_range_into_f32(&averages[..averages.len() - 1], 0..0, &mut [])
            .is_err());
        assert!(l8.gather_range_into_f32(&averages, 2..1, &mut []).is_err());
        assert!(l8
            .gather_range_into_f32(&averages, 0..4, &mut [0.0; 4])
            .is_err());
        assert!(l8
            .gather_range_into_f32(&averages, 0..3, &mut tail)
            .is_err());
        let mut invalid = averages.clone();
        invalid[l8.state_ids()[0] as usize - 1] = f32::NAN;
        assert!(l8
            .gather_range_into_f32(&invalid, 0..1, &mut [0.0])
            .is_err());
        assert!(l8.storage_bytes() >= l8.stats().table_bytes);
        assert_eq!(l8.storage_bytes(), l8.stats().retained_capacity_bytes);
    });
}
