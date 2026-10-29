//! Acceptance and source-version guards around the production whole operator.

use super::*;
use crate::loop1_prepared_tests::{assert_bits, scalar_path, FullPolicyFixture, TWO_LINE};
use std::fs;

#[test]
fn whole_reference_gate_requires_finite_bitwise_equality() {
    let finite = [-0.0f32, 0.0, -2.25, f32::MIN_POSITIVE, f32::MAX];
    same_finite_bits(&finite, &finite).unwrap();
    same_finite_bits(&[], &[]).unwrap();
    assert!(same_finite_bits(&[1.0], &[]).is_err());
    for (dag, numeric) in [
        (0.0f32, -0.0f32),
        (1.0, f32::from_bits(1.0f32.to_bits() + 1)),
        (-2.25, 2.25),
    ] {
        let error = same_finite_bits(&[0.0, dag], &[0.0, numeric]).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("boundary 1"), "{message}");
        assert!(message.contains("bit mismatch"), "{message}");
        assert!(message.contains("output was not published"), "{message}");
    }
    // Identical encodings of infinities and NaNs are still invalid values.
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc01234)] {
        for (dag, numeric) in [(value, value), (value, 0.0), (0.0, value)] {
            let error = same_finite_bits(&[dag], &[numeric]).unwrap_err();
            assert!(format!("{error:#}").contains("nonfinite"), "{error:#}");
        }
    }
}

#[test]
fn whole_input_guard_allows_reads_and_rejects_same_byte_writes_to_every_input() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let paths: Vec<_> = ["graph", "membership", "policy", "initial", "comparison"]
        .iter()
        .map(|name| owned.path.join(name))
        .collect();
    for path in &paths {
        fs::write(path, b"stable source bytes").unwrap();
    }
    for changed in &paths {
        let guard = InputGuard::capture(paths.clone()).unwrap();
        for path in &paths {
            assert_eq!(fs::read(path).unwrap(), b"stable source bytes");
        }
        guard.check().unwrap();
        guard.check().unwrap();
        // The same length and same contents must not hide a mutation that can
        // race a parsed snapshot or a compiled plan.
        fs::write(changed, b"stable source bytes").unwrap();
        assert!(
            guard.check().is_err(),
            "source write accepted: {}",
            changed.display()
        );
        assert!(guard.check().is_err(), "mutation notification was consumed");
    }
}

#[test]
fn whole_input_guard_rejects_same_byte_replacements() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let source = owned.path.join("policy");
    fs::write(&source, b"same bytes").unwrap();
    let guard = InputGuard::capture(vec![source.clone()]).unwrap();
    let replacement = owned.path.join("replacement");
    fs::write(&replacement, b"same bytes").unwrap();
    fs::rename(&replacement, &source).unwrap();
    assert!(guard.check().is_err());
    assert!(guard.check().is_err());
}

#[cfg(unix)]
#[test]
fn whole_input_guard_rejects_retargeted_symlink() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let first = owned.path.join("first");
    let second = owned.path.join("second");
    let request = owned.path.join("policy-link");
    fs::write(&first, b"same bytes").unwrap();
    fs::write(&second, b"same bytes").unwrap();
    std::os::unix::fs::symlink(&first, &request).unwrap();
    let guard = InputGuard::capture(vec![request.clone()]).unwrap();
    guard.check().unwrap();
    fs::remove_file(&request).unwrap();
    std::os::unix::fs::symlink(&second, &request).unwrap();
    assert!(guard.check().is_err());
}

#[test]
fn whole_source_change_at_publication_callback_preserves_output_and_removes_pending_file() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let source = owned.path.join("source.fixture");
    let output = owned.path.join("accepted.f32.bin");
    fs::write(&source, b"stable input").unwrap();
    let accepted = b"previously accepted output";
    for gmres in [false, true] {
        fs::write(&output, accepted).unwrap();
        let guard = InputGuard::capture(vec![source.clone()]).unwrap();
        let called = std::cell::Cell::new(false);
        let check = || -> Result<()> {
            called.set(true);
            // This callback is the last check after the temporary candidate is
            // flushed, before rename. Mutate here instead of using a race.
            assert!(fs::read_dir(&owned.path)?.any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("pending")));
            assert_eq!(fs::read(&output)?, accepted);
            fs::write(&source, b"stable input")?;
            guard.check()
        };
        let result = if gmres {
            crate::gmres_accept_and_publish_checked(&output, &[2.0], &[2.0], 1e-3, check)
                .map(|_| ())
        } else {
            crate::publish_finite_f32_atomic_checked(&output, &[2.0], check)
        };
        assert!(result.is_err());
        assert!(called.get(), "publication skipped its final source check");
        assert_eq!(fs::read(&output).unwrap(), accepted);
        assert!(!fs::read_dir(&owned.path).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("pending")));
    }
}

#[test]
fn whole_production_wrapper_matches_scalar_with_additional_l0_field_and_rejects_stale_source() {
    let _lock = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut fixture = FullPolicyFixture::new();
    let root = fixture.root().to_path_buf();
    let prefix = root.join("whole-multi-field-l0");
    let mut writer = crate::layer_store::FieldChainWriter::create(&prefix).unwrap();
    for &qid in &fixture.reset_qids {
        writer.push_pair_key(qid as u64).unwrap();
    }
    writer.push_pair_key(1u64 << 32).unwrap();
    writer.finish().unwrap();
    fixture.layers[0] = crate::layer_store::FieldChainLayer::open_index_only(&prefix).unwrap();
    let policy0 = crate::mdp_policy_u16_path(&fixture.policy_dir, 0);
    let mut codes = fs::read(&policy0).unwrap();
    codes.extend_from_slice(&crate::POLICY_NONE_U16.to_le_bytes());
    fs::write(&policy0, codes).unwrap();
    validate_root_layout(&fixture.graph, &fixture.layers, &fixture.reset_qids).unwrap();

    let mut paths = vec![fixture.graph_path.clone()];
    for (depth, layer) in fixture.layers.iter().enumerate() {
        paths.extend([
            layer.index_path.clone(),
            layer.chains_path.clone(),
            crate::mdp_policy_u16_path(&fixture.policy_dir, depth),
        ]);
    }
    let guard = InputGuard::capture(paths).unwrap();
    let policy = validated_policy(
        &fixture.graph,
        &fixture.qfwd,
        &fixture.pre,
        &fixture.layers,
        &fixture.policy_dir,
        TWO_LINE,
        &guard,
    )
    .unwrap();
    let settings = Settings {
        dag: WholeDagLimits {
            total_memo_bytes: 64 << 10,
            max_nodes: 10_000,
            max_edges: 50_000,
            progress: false,
        },
        reference: SparseLimits {
            workers: 2,
            total_memo_bytes: 64 << 10,
            progress: false,
        },
        l9_tail: false,
    };
    let plan = compile(&policy, settings, None, &guard).unwrap();
    let terminal = TerminalPlan::build(&fixture.quotient, &fixture.qfwd, &fixture.pre).unwrap();
    let operator = FullBoundaryDag::new(
        &plan,
        &policy,
        &terminal,
        &fixture.qfwd,
        &fixture.reset_qids,
        &guard,
        settings.reference,
    )
    .unwrap();
    assert_eq!(plan.root_count(), fixture.reset_qids.len() + 1);
    let mut saw_signed = false;
    for cancellation in [false, true] {
        let mut values: Vec<f32> = (0..fixture.reset_qids.len())
            .map(|index| {
                if !cancellation {
                    -2.25
                } else if index % 2 == 0 {
                    16_777_216.0
                } else {
                    -16_777_216.0
                }
            })
            .collect();
        values.push(91.0); // This non-reset boundary coordinate must not feed reset qids.
        for linear in [false, true] {
            let mut expected = scalar_path(
                &fixture.qfwd,
                &fixture.qids,
                &fixture.placed,
                &fixture.reset_qids,
                &values[..fixture.reset_qids.len()],
                !linear,
            );
            expected.push(0.0); // The additional field has a NONE policy.
            saw_signed |= linear && expected.iter().any(|value| *value < 0.0);
            assert_bits(
                &operator.apply(&values, linear).unwrap(),
                &expected,
                "whole production scalar",
            );
            assert_bits(
                &operator.apply_verified(&values, linear).unwrap(),
                &expected,
                "whole independent numeric gate",
            );
        }
        assert!(operator.apply(&values[..values.len() - 1], false).is_err());
        let mut nonfinite = values;
        *nonfinite.last_mut().unwrap() = f32::NAN;
        assert!(
            operator.apply(&nonfinite, false).is_err(),
            "all boundary coordinates require finite inputs"
        );
    }
    assert!(
        saw_signed,
        "fixture must produce a nonzero signed homogeneous value"
    );
    let source = crate::mdp_policy_u16_path(&fixture.policy_dir, 5);
    fs::write(&source, fs::read(&source).unwrap()).unwrap();
    let values = vec![-2.25; plan.root_count()];
    assert!(operator.apply(&values, false).is_err());
    assert!(operator.apply_verified(&values, false).is_err());
}
