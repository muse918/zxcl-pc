//! The complete L8 compiled path feeds the unchanged L7..L0 kernels. A full
//! quotient fixture keeps the ten-placement chain nonzero while extra L8 NONE
//! states and omitted reveal children exercise gather order and missing mass.

use super::*;
use crate::loop1_prepared::PreparedL9;
use crate::loop1_prepared_tests::{
    assert_bits, membership, scalar_path, sweep, write_policy, EnvRestore, FullPolicyFixture,
};
use crate::loop1_stream::CompiledL8;

fn compiled_sweep<'a>(
    fixture: &'a FullPolicyFixture,
    prepared_l9: &'a PreparedL9<'a>,
    prepared_l8: &'a CompiledL8<'a>,
    scratch: &'a Path,
    ram: bool,
) -> FixedPolicySweep<'a> {
    let mut sweep = sweep(
        &fixture.graph,
        &fixture.qfwd,
        &fixture.layers,
        &fixture.policy_dir,
        scratch,
        &fixture.reset_qids,
        Some(prepared_l9),
        ram,
    );
    sweep.prepared_l8 = Some(prepared_l8);
    sweep
}

// Independent ordered f32 arithmetic at the last two placements. This does not
// use compiled IDs, an interner, or any of the new evaluator/gather methods.
fn scalar_l8(fixture: &FullPolicyFixture, reset: &[f32], include_reward: bool) -> Vec<f32> {
    let qfwd = &fixture.qfwd;
    let reward = if include_reward { 1.0f32 } else { 0.0f32 };
    let terminal_records = qfwd.records_for(fixture.qids[9], fixture.placed[9]);
    let mut terminal = 0.0f32;
    for rec in terminal_records {
        let x = fixture
            .reset_qids
            .binary_search(&rec.child_qid)
            .ok()
            .map_or(0.0, |index| reset[index]);
        terminal += reward + x;
    }
    terminal /= terminal_records.len() as f32;
    let records = qfwd.records_for(fixture.qids[8], fixture.placed[8]);
    let mut value = 0.0f32;
    for rec in records {
        value += if rec.child_qid == fixture.qids[9] {
            terminal
        } else {
            0.0
        };
    }
    value /= records.len() as f32;
    fixture.layers[8]
        .read_block_chains(0)
        .iter()
        .map(|&qid| if qid == fixture.qids[8] { value } else { 0.0 })
        .collect()
}

fn dense_reset(fixture: &FullPolicyFixture, values: &[f32]) -> Vec<f32> {
    let mut dense = vec![0.0f32; fixture.quotient.states.len()];
    for (&qid, &value) in fixture.reset_qids.iter().zip(values) {
        dense[qid as usize] = value;
    }
    dense
}

#[test]
fn full_compiled_l8_production_sweep_matches_original_all_sinks_and_signed_modes() {
    let _lock = TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _settings = EnvRestore::set(&[
        ("FULL_MDP_L9_DIRECT", "0"),
        ("FULL_MDP_MEMORY_MAX_GIB", "1"),
        ("FULL_MDP_MEMORY_RESERVE_GIB", "0"),
        ("FULL_MDP_RAM_PAIR_MAX_GIB", "1"),
        ("FULL_MDP_LOOKUP_GROUP_CHUNKS", "2"),
        ("FULL_MDP_TASK_MAX_STATES", "100"),
        ("PI_LOOP1_L8_MAX_MAPPING_MIB", "16"),
        ("PI_LOOP1_L8_BUILD_CACHE_MIB", "1"),
        ("PI_LOOP1_L8_BUILD_MAX_MIB", "256"),
        ("PI_LOOP1_L8_MAX_UNIQUE_AVERAGES", "100"),
    ]);
    let mut fixture = FullPolicyFixture::with_l8_padding();
    let root = fixture.root().to_path_buf();
    {
        let prepared_l9 = PreparedL9::build(
            &fixture.graph,
            &fixture.quotient,
            &fixture.qfwd,
            &fixture.pre,
            &fixture.layers,
            &fixture.policy_dir,
            fixture.bindings(),
        )
        .unwrap();
        let prepared_l8 = loop1_prepared_l8::build(
            &fixture.graph,
            &fixture.quotient,
            &fixture.qfwd,
            &fixture.pre,
            &fixture.layers[8],
            &mdp_policy_u16_path(&fixture.policy_dir, 8),
            &prepared_l9.compiled,
        )
        .unwrap();
        assert_eq!(prepared_l8.state_offset(), 0);
        assert_eq!(
            prepared_l8.state_count() as u64,
            fixture.layers[8].total_states
        );
        assert!(
            prepared_l8.state_count() > 3,
            "fixture needs uneven gather chunks"
        );
        assert!(
            prepared_l8.state_ids().contains(&0),
            "fixture needs NONE states"
        );

        let inputs = [
            vec![-2.25; fixture.reset_qids.len()],
            (0..fixture.reset_qids.len())
                .map(|index| {
                    if index % 2 == 0 {
                        16_777_216.0
                    } else {
                        -16_777_216.0
                    }
                })
                .collect(),
        ];
        let mut saw_signed_result = false;
        for (input_index, values) in inputs.iter().enumerate() {
            for linear in [false, true] {
                let terminals = prepared_l9
                    .terminal_values(&dense_reset(&fixture, values), !linear)
                    .unwrap();
                let expected = scalar_l8(&fixture, values, !linear);
                assert_bits(
                    &prepared_l8.evaluate_f32(&terminals).unwrap(),
                    &expected,
                    "complete compiled evaluator against scalar",
                );
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    pool.install(|| {
                        let averages = prepared_l8.evaluate_averages_f32(&terminals).unwrap();
                        if input_index == 0 && !linear && threads == 1 {
                            let mut untouched =
                                vec![f32::from_bits(0x3f123456); prepared_l8.state_count()];
                            assert!(prepared_l8
                                .gather_range_into_f32(
                                    &averages,
                                    0..prepared_l8.state_count() + 1,
                                    &mut untouched
                                )
                                .is_err());
                            assert!(prepared_l8
                                .gather_range_into_f32(&averages, 0..1, &mut untouched)
                                .is_err());
                            assert!(prepared_l8
                                .gather_range_into_f32(
                                    &[],
                                    0..prepared_l8.state_count(),
                                    &mut untouched
                                )
                                .is_err());
                            assert!(untouched.iter().all(|value| value.to_bits() == 0x3f123456));
                            let mut called = false;
                            assert!(prepared_l8
                                .gather_chunks_f32(&averages, 0, |_, _| {
                                    called = true;
                                    Ok(())
                                })
                                .is_err());
                            assert!(!called);
                        }
                        let mut ranges = vec![f32::NAN; prepared_l8.state_count()];
                        for (index, chunk) in ranges.chunks_mut(2).enumerate() {
                            let start = index * 2;
                            prepared_l8
                                .gather_range_into_f32(&averages, start..start + chunk.len(), chunk)
                                .unwrap();
                        }
                        assert_bits(&ranges, &expected, "uneven range gather");
                        let mut chunks = Vec::new();
                        prepared_l8
                            .gather_chunks_f32(&averages, 3, |offset, values| {
                                assert_eq!(offset, chunks.len());
                                chunks.extend_from_slice(values);
                                Ok(())
                            })
                            .unwrap();
                        assert_bits(&chunks, &expected, "uneven bounded file gather");
                    });
                }
                let expected_boundary = scalar_path(
                    &fixture.qfwd,
                    &fixture.qids,
                    &fixture.placed,
                    &fixture.reset_qids,
                    values,
                    !linear,
                );
                saw_signed_result |= linear && expected_boundary.iter().any(|value| *value < 0.0);
                for mode in 0..3 {
                    let _mode = EnvRestore::set(&[(
                        "FULL_MDP_MMAP_SPILL",
                        if mode == 2 { "1" } else { "0" },
                    )]);
                    let original_dir =
                        root.join(format!("full8-original-{input_index}-{linear}-{mode}"));
                    let compiled_dir =
                        root.join(format!("full8-compiled-{input_index}-{linear}-{mode}"));
                    fs::create_dir(&original_dir).unwrap();
                    fs::create_dir(&compiled_dir).unwrap();
                    let original = sweep(
                        &fixture.graph,
                        &fixture.qfwd,
                        &fixture.layers,
                        &fixture.policy_dir,
                        &original_dir,
                        &fixture.reset_qids,
                        None,
                        mode != 0,
                    );
                    let compiled = compiled_sweep(
                        &fixture,
                        &prepared_l9,
                        &prepared_l8,
                        &compiled_dir,
                        mode != 0,
                    );
                    assert_bits(
                        &original.apply(values, linear).unwrap(),
                        &expected_boundary,
                        "original ten-layer scalar",
                    );
                    assert_bits(
                        &compiled.apply(values, linear).unwrap(),
                        &expected_boundary,
                        "full L8 ten-layer scalar",
                    );
                    assert_bits(
                        &compiled.apply(values, linear).unwrap(),
                        &expected_boundary,
                        "repeated full L8 application",
                    );
                    assert!(
                        !mdp_f32_value_path(&compiled_dir, 9).exists(),
                        "compiled L8 should not materialize L9"
                    );
                    if mode == 1 {
                        assert!(
                            !mdp_f32_value_path(&compiled_dir, 8).exists(),
                            "RAM gather should stay in RAM"
                        );
                    } else {
                        assert_bits(
                            &load_f32_values_vec(
                                &mdp_f32_value_path(&compiled_dir, 8),
                                fixture.layers[8].total_states,
                            )
                            .unwrap(),
                            &expected,
                            "complete L8 file/mmap sink",
                        );
                    }
                }
            }
        }
        assert!(
            saw_signed_result,
            "fixture must exercise nonzero signed output"
        );

        // A caller-defined terminal table does not imply equal values for
        // placements sharing a formula ID. This API must still run raw kernels.
        {
            let _mode = EnvRestore::set(&[("FULL_MDP_MMAP_SPILL", "0")]);
            let original_dir = root.join("full8-custom-original");
            let compiled_dir = root.join("full8-custom-compiled");
            fs::create_dir(&original_dir).unwrap();
            fs::create_dir(&compiled_dir).unwrap();
            let original = sweep(
                &fixture.graph,
                &fixture.qfwd,
                &fixture.layers,
                &fixture.policy_dir,
                &original_dir,
                &fixture.reset_qids,
                None,
                false,
            );
            let compiled =
                compiled_sweep(&fixture, &prepared_l9, &prepared_l8, &compiled_dir, false);
            let arbitrary: Vec<[f32; PIECE_COUNT]> = (0..fixture.quotient.states.len())
                .map(|qid| std::array::from_fn(|p| ((qid * 13 + p * 7) % 41) as f32 - 19.0))
                .collect();
            assert_bits(
                &compiled.apply_terminal(&arbitrary).unwrap(),
                &original.apply_terminal(&arbitrary).unwrap(),
                "arbitrary terminal table remains raw",
            );
        }
        {
            let _mode =
                EnvRestore::set(&[("FULL_MDP_L9_DIRECT", "1"), ("FULL_MDP_MMAP_SPILL", "0")]);
            let scratch = root.join("full8-independent-original");
            fs::create_dir(&scratch).unwrap();
            let compiled = compiled_sweep(&fixture, &prepared_l9, &prepared_l8, &scratch, false);
            let reset = vec![-2.25; fixture.reset_qids.len()];
            for linear in [false, true] {
                let expected = scalar_path(
                    &fixture.qfwd,
                    &fixture.qids,
                    &fixture.placed,
                    &fixture.reset_qids,
                    &reset,
                    !linear,
                );
                assert_bits(
                    &compiled.apply(&reset, linear).unwrap(),
                    &expected,
                    "all optional L9/L8 paths enabled",
                );
                assert!(!mdp_f32_value_path(&scratch, 9).exists());
                assert_bits(
                    &compiled.apply_original(&reset, linear).unwrap(),
                    &expected,
                    "original gate bypasses both compiled layers and direct flag",
                );
                assert!(mdp_f32_value_path(&scratch, 9).exists());
                assert_bits(
                    &load_f32_values_vec(
                        &mdp_f32_value_path(&scratch, 8),
                        fixture.layers[8].total_states,
                    )
                    .unwrap(),
                    &scalar_l8(&fixture, &reset, !linear),
                    "original gate materializes L8 through raw arithmetic",
                );
                fs::remove_file(mdp_f32_value_path(&scratch, 9)).unwrap();
                fs::remove_file(mdp_f32_value_path(&scratch, 8)).unwrap();
            }
        }

        // Even an already evaluated shared average cannot be gathered after a
        // fixed-policy source changes. Apply rejects before touching any sink.
        let scratch = root.join("full8-source-change");
        fs::create_dir(&scratch).unwrap();
        let marker = b"accepted scratch must survive source changes";
        for layer in 0..10 {
            fs::write(mdp_f32_value_path(&scratch, layer), marker).unwrap();
        }
        let terminals = prepared_l9
            .terminal_values(&dense_reset(&fixture, &inputs[0]), true)
            .unwrap();
        let averages = prepared_l8.evaluate_averages_f32(&terminals).unwrap();
        let policy = mdp_policy_u16_path(&fixture.policy_dir, 8);
        let original = fs::read(&policy).unwrap();
        let mut changed = original.clone();
        changed[0] ^= 1;
        fs::write(&policy, changed).unwrap();
        let mut untouched = vec![f32::from_bits(0x3f123456); prepared_l8.state_count()];
        let gather_error = prepared_l8
            .gather_range_into_f32(&averages, 0..prepared_l8.state_count(), &mut untouched)
            .unwrap_err();
        assert!(
            format!("{gather_error:#}").contains("chang"),
            "{gather_error:#}"
        );
        assert!(untouched.iter().all(|value| value.to_bits() == 0x3f123456));
        let compiled = compiled_sweep(&fixture, &prepared_l9, &prepared_l8, &scratch, false);
        let error = compiled.apply(&inputs[0], false).unwrap_err();
        assert!(format!("{error:#}").contains("chang"), "{error:#}");
        for layer in 0..10 {
            assert_eq!(
                fs::read(mdp_f32_value_path(&scratch, layer)).unwrap(),
                marker
            );
        }
        fs::write(&policy, original).unwrap();
    }

    // A complete but empty L8 is valid: every L7 continuation is missing and
    // therefore zero. All sink modes need an empty placeholder rather than an
    // invalid mmap of length zero or a stale previous-layer pin.
    fixture.layers[8] = membership(&root.join("empty-layer8"), 8, &[]);
    write_policy(&mdp_policy_u16_path(&fixture.policy_dir, 8), &[]);
    let prepared_l9 = PreparedL9::build(
        &fixture.graph,
        &fixture.quotient,
        &fixture.qfwd,
        &fixture.pre,
        &fixture.layers,
        &fixture.policy_dir,
        fixture.bindings(),
    )
    .unwrap();
    let prepared_l8 = loop1_prepared_l8::build(
        &fixture.graph,
        &fixture.quotient,
        &fixture.qfwd,
        &fixture.pre,
        &fixture.layers[8],
        &mdp_policy_u16_path(&fixture.policy_dir, 8),
        &prepared_l9.compiled,
    )
    .unwrap();
    assert_eq!(prepared_l8.state_count(), 0);
    assert_eq!(prepared_l8.field_range(), 0..0);
    let reset = vec![-2.25; fixture.reset_qids.len()];
    let expected = vec![0.0f32; fixture.reset_qids.len()];
    for mode in 0..3 {
        let _mode = EnvRestore::set(&[("FULL_MDP_MMAP_SPILL", if mode == 2 { "1" } else { "0" })]);
        let scratch = root.join(format!("full8-empty-{mode}"));
        fs::create_dir(&scratch).unwrap();
        let compiled = compiled_sweep(&fixture, &prepared_l9, &prepared_l8, &scratch, mode != 0);
        for linear in [false, true] {
            assert_bits(
                &compiled.apply(&reset, linear).unwrap(),
                &expected,
                "empty complete L8",
            );
        }
        assert_eq!(
            fs::metadata(mdp_f32_value_path(&scratch, 8)).unwrap().len(),
            0
        );
    }
}
