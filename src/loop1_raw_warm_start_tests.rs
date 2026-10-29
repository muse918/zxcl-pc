//! Independent raw boundary evaluation before any whole-DAG construction.

use super::*;
use crate::loop1_prepared_tests::{
    assert_bits, scalar_path, EnvRestore, FullPolicyFixture, TWO_LINE,
};
use std::fs;

#[test]
fn raw_warm_start_flag_is_explicit_and_requires_the_whole_policy_path() {
    let _lock = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _restore = EnvRestore::set(&[("PI_GMRES_RAW_WARM_START", "0")]);
    std::env::remove_var("PI_GMRES_RAW_WARM_START");
    assert!(!raw_warm_start_enabled(false).unwrap());
    assert!(!raw_warm_start_enabled(true).unwrap());
    for value in ["0", "false", "FALSE", "no", "No"] {
        std::env::set_var("PI_GMRES_RAW_WARM_START", value);
        assert!(!raw_warm_start_enabled(false).unwrap(), "value={value}");
        assert!(!raw_warm_start_enabled(true).unwrap(), "value={value}");
    }
    for value in ["1", "true", "TRUE", "yes", "YeS"] {
        std::env::set_var("PI_GMRES_RAW_WARM_START", value);
        assert!(raw_warm_start_enabled(true).unwrap(), "value={value}");
        let error = raw_warm_start_enabled(false).unwrap_err();
        assert!(format!("{error:#}").contains("PI_LOOP1_WHOLE_DAG"));
    }
    for value in ["", "2", "sometimes", " true "] {
        std::env::set_var("PI_GMRES_RAW_WARM_START", value);
        assert!(raw_warm_start_enabled(true).is_err(), "value={value}");
        assert!(raw_warm_start_enabled(false).is_err(), "value={value}");
    }
}

#[test]
fn raw_warm_start_uses_every_original_boundary_root_and_rejects_invalid_or_stale_inputs() {
    let _lock = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut fixture = FullPolicyFixture::new();
    let root = fixture.root().to_path_buf();

    // The reset vector is only the first field's prefix of the complete L0
    // boundary. A distinct extra coordinate must be evaluated, length-checked,
    // and checked for finiteness without leaking into the reset distribution.
    let prefix = root.join("raw-warm-start-multi-field-l0");
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
    let limits = SparseLimits {
        workers: 2,
        total_memo_bytes: 64 << 10,
        progress: false,
    };
    let roots = fixture.reset_qids.len() + 1;
    assert_eq!(policy.root_count() as usize, roots);

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
        values.push(if cancellation { -91.0 } else { 91.0 });
        for linear in [false, true] {
            let mut expected = scalar_path(
                &fixture.qfwd,
                &fixture.qids,
                &fixture.placed,
                &fixture.reset_qids,
                &values[..fixture.reset_qids.len()],
                !linear,
            );
            expected.push(0.0);
            saw_signed |= linear && expected.iter().any(|value| *value < 0.0);
            let raw = raw_boundary_apply(
                &policy,
                &fixture.qfwd,
                &fixture.reset_qids,
                &guard,
                &values,
                linear,
                limits,
            )
            .unwrap();
            assert_bits(&raw.values, &expected, "raw all-root independent scalar");
            assert_eq!(raw.stats.roots as usize, roots);
            assert!(raw.stats.evaluated_states > 0);
            assert_eq!(raw.stats.compiled_l9_averages, 0);
            assert_eq!(raw.stats.compiled_l9_reveals, 0);
            assert_eq!(raw.stats.compiled_l9_membership_bytes, 0);
        }

        for malformed in [values[..roots - 1].to_vec(), {
            let mut longer = values.clone();
            longer.push(0.0);
            longer
        }] {
            assert!(raw_boundary_apply(
                &policy,
                &fixture.qfwd,
                &fixture.reset_qids,
                &guard,
                &malformed,
                false,
                limits,
            )
            .is_err());
        }
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut malformed = values.clone();
            *malformed.last_mut().unwrap() = nonfinite;
            assert!(
                raw_boundary_apply(
                    &policy,
                    &fixture.qfwd,
                    &fixture.reset_qids,
                    &guard,
                    &malformed,
                    false,
                    limits,
                )
                .is_err(),
                "the non-reset boundary coordinate still requires a finite value"
            );
        }
    }
    assert!(
        saw_signed,
        "the homogeneous fixture must produce a negative value"
    );

    let values = vec![-2.25; roots];
    let unvalidated = SparsePolicy::open(
        &fixture.graph,
        &fixture.qfwd,
        &fixture.pre,
        &fixture.layers,
        &fixture.policy_dir,
        TWO_LINE,
    )
    .unwrap();
    assert!(raw_boundary_apply(
        &unvalidated,
        &fixture.qfwd,
        &fixture.reset_qids,
        &guard,
        &values,
        false,
        limits,
    )
    .is_err());

    let duplicate = vec![fixture.reset_qids[0]; 2];
    let mut descending = fixture.reset_qids.clone();
    descending.reverse();
    let outside_domain = vec![fixture.qfwd.offsets.len() as u32];
    for malformed_reset in [duplicate, descending, outside_domain] {
        assert!(raw_boundary_apply(
            &policy,
            &fixture.qfwd,
            &malformed_reset,
            &guard,
            &values,
            false,
            limits,
        )
        .is_err());
    }

    // A stale source must fail before any caller reaches atomic publication.
    // A same-byte policy write still invalidates its earlier validation.
    let accepted = root.join("raw-warm-start-accepted.f32.bin");
    let previous = b"previously accepted output";
    fs::write(&accepted, previous).unwrap();
    fs::write(&policy0, fs::read(&policy0).unwrap()).unwrap();
    let publication_reached = std::cell::Cell::new(false);
    let result = raw_boundary_apply(
        &policy,
        &fixture.qfwd,
        &fixture.reset_qids,
        &guard,
        &values,
        false,
        limits,
    )
    .and_then(|raw| {
        publication_reached.set(true);
        crate::gmres_accept_and_publish_checked(&accepted, &values, &raw.values, 3e-3, || {
            guard.check()
        })
    });
    assert!(result.is_err());
    assert!(!publication_reached.get());
    assert_eq!(fs::read(&accepted).unwrap(), previous);
    assert!(!fs::read_dir(&root).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains("pending")));
}
