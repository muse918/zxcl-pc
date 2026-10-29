//! Production fixed-policy sweeps with a compiled L9 and an independent scalar
//! ten-placement path. The quotient domain is complete; layer membership stays
//! small and deliberately omits most random-reveal children.

use super::*;
use crate::loop1_prepared::{PreparedL9, SourceBindings};

pub(crate) const TWO_LINE: FieldId = 10;

pub(crate) struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvRestore {
    pub(crate) fn set(settings: &[(&'static str, &str)]) -> Self {
        let previous = settings
            .iter()
            .map(|&(name, value)| {
                let old = env::var_os(name);
                env::set_var(name, value);
                (name, old)
            })
            .collect();
        Self(previous)
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..).rev() {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
    }
}

fn path_graph() -> FieldGraph {
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    for field in 0..=TWO_LINE {
        let mut row = [targets.len() as u32; PIECE_COUNT + 1];
        for placed in 0..PIECE_COUNT {
            if field < 9 {
                targets.push(field + 1);
                if field == 4 {
                    targets.push(TWO_LINE);
                }
            } else if field == 9 {
                targets.push(EMPTY_FIELD_ID);
            }
            row[placed + 1] = targets.len() as u32;
        }
        offsets.push(row);
    }
    FieldGraph {
        hashes: (0..=TWO_LINE as u64).collect(),
        offsets,
        targets,
    }
}

pub(crate) fn membership(
    prefix: &Path,
    field: FieldId,
    qids: &[u32],
) -> layer_store::FieldChainLayer {
    let mut writer = layer_store::FieldChainWriter::create(prefix).unwrap();
    for &qid in qids {
        writer
            .push_pair_key((field as u64) << 32 | qid as u64)
            .unwrap();
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
}

pub(crate) fn write_policy(path: &Path, codes: &[u16]) {
    fs::write(
        path,
        codes
            .iter()
            .flat_map(|code| code.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

pub(crate) fn assert_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context} state={index}"
        );
    }
}

pub(crate) fn sweep<'a>(
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    layers: &'a [layer_store::FieldChainLayer],
    policy_dir: &'a Path,
    scratch: &'a Path,
    reset_qids: &'a [u32],
    prepared: Option<&'a PreparedL9<'a>>,
    ram: bool,
) -> FixedPolicySweep<'a> {
    FixedPolicySweep {
        graph,
        qfwd,
        full_layers: layers,
        policy_dir,
        scratch_dir: scratch,
        two_line_field: TWO_LINE,
        qstate_count: qfwd.offsets.len(),
        gamma: 1.0,
        cache_cap_bytes: 1024 * 1024,
        sparse_lookup_max_count: 65_536,
        block_chunk: 1,
        batch_chunks_env: Some(2),
        ram_next_layer: ram,
        ram_next_layer_max_bytes: 1024 * 1024,
        l0_reset_qids: reset_qids,
        verbose_layers: false,
        prepared_l9: prepared,
        prepared_l8: None,
        action_source: None,
        operator_bindings: None,
        action_resident_bytes: 0,
    }
}

pub(crate) fn scalar_path(
    qfwd: &QForwardGraph,
    qids: &[u32; 10],
    placed: &[u8; 10],
    reset_qids: &[u32],
    x: &[f32],
    include_reward: bool,
) -> Vec<f32> {
    let reward = if include_reward { 1.0f32 } else { 0.0f32 };
    let mut value = 0.0f32;
    let recs = qfwd.records_for(qids[9], placed[9]);
    for rec in recs {
        let reset = reset_qids
            .binary_search(&rec.child_qid)
            .ok()
            .map_or(0.0, |i| x[i]);
        value += reward + reset;
    }
    value /= recs.len() as f32;
    for stage in (0..9).rev() {
        let recs = qfwd.records_for(qids[stage], placed[stage]);
        let mut sum = 0.0f32;
        for rec in recs {
            sum += if rec.child_qid == qids[stage + 1] {
                value
            } else {
                0.0
            };
        }
        value = sum / recs.len() as f32;
    }
    let mut output = vec![0.0; reset_qids.len()];
    output[reset_qids.binary_search(&qids[0]).unwrap()] = value;
    output
}

pub(crate) struct FullPolicyFixture {
    owned: GreedyEmitScratch,
    pub(crate) graph: FieldGraph,
    pub(crate) quotient: QStateIndex,
    pub(crate) qfwd: QForwardGraph,
    pub(crate) pre: pre_reveal::PreRevealIndex,
    pub(crate) graph_path: PathBuf,
    pub(crate) policy_dir: PathBuf,
    pub(crate) layers: Vec<layer_store::FieldChainLayer>,
    pub(crate) qids: [u32; 10],
    pub(crate) placed: [u8; 10],
    pub(crate) reset_qids: Vec<u32>,
}

impl FullPolicyFixture {
    pub(crate) fn new() -> Self {
        Self::build(false)
    }

    pub(crate) fn with_l8_padding() -> Self {
        Self::build(true)
    }

    fn build(pad_l8: bool) -> Self {
        let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
        let root = &owned.path;
        let old = enumerate_states();
        let quotient = enumerate_qstates(&old);
        drop(old);
        let qfwd = build_qforward_graph(&quotient);
        let pre = pre_reveal::PreRevealIndex::build(&quotient).unwrap();
        let graph = path_graph();
        // The graph object is synthetic; this file exercises the same immutable
        // graph-source binding as the CLI without creating the huge binary graph.
        let graph_path = root.join("graph.fixture");
        fs::write(&graph_path, b"immutable synthetic graph fixture").unwrap();
        let policy_dir = root.join("policy");
        fs::create_dir(&policy_dir).unwrap();

        let mut qids = [0u32; 10];
        let mut placed = [0u8; 10];
        for stage in 0..10 {
            placed[stage] = (0..PIECE_COUNT as u8)
                .find(|&p| !qfwd.records_for(qids[stage], p).is_empty())
                .unwrap();
            if stage < 9 {
                qids[stage + 1] = qfwd.records_for(qids[stage], placed[stage])[0].child_qid;
            }
        }
        let mut reset_qids = vec![qids[0]];
        reset_qids.extend(
            qfwd.records_for(qids[9], placed[9])
                .iter()
                .map(|r| r.child_qid),
        );
        reset_qids.sort_unstable();
        reset_qids.dedup();
        let layers: Vec<_> = (0..10)
            .map(|stage| {
                let mut members = if stage == 0 {
                    reset_qids.clone()
                } else {
                    vec![qids[stage]]
                };
                if stage == 8 && pad_l8 {
                    members.extend(
                        (1..7).map(|offset| (qids[stage] + offset) % quotient.states.len() as u32),
                    );
                    members.sort_unstable();
                    members.dedup();
                }
                let layer = membership(&root.join(format!("layer{stage}")), stage as u32, &members);
                let codes: Vec<_> = members
                    .iter()
                    .map(|&qid| {
                        if qid != qids[stage] {
                            POLICY_NONE_U16
                        } else {
                            encode_policy_u16(placed[stage], 0)
                        }
                    })
                    .collect();
                write_policy(&mdp_policy_u16_path(&policy_dir, stage), &codes);
                layer
            })
            .collect();
        Self {
            owned,
            graph,
            quotient,
            qfwd,
            pre,
            graph_path,
            policy_dir,
            layers,
            qids,
            placed,
            reset_qids,
        }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.owned.path
    }

    pub(crate) fn bindings(&self) -> SourceBindings {
        SourceBindings::capture(&self.graph_path, &self.layers, &self.policy_dir).unwrap()
    }
}

#[test]
fn compiled_l9_production_sweep_matches_scalar_signed_modes_and_rejects_source_changes() {
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
    ]);
    let fixture = FullPolicyFixture::new();
    let root = fixture.root();
    let graph = &fixture.graph;
    let quotient = &fixture.quotient;
    let qfwd = &fixture.qfwd;
    let pre = &fixture.pre;
    let graph_path = &fixture.graph_path;
    let policy_dir = &fixture.policy_dir;
    let layers = &fixture.layers;
    let qids = fixture.qids;
    let placed = fixture.placed;
    let reset_qids = &fixture.reset_qids;
    let bindings = SourceBindings::capture(&graph_path, &layers, &policy_dir).unwrap();
    let prepared = PreparedL9::build(
        &graph,
        &quotient,
        &qfwd,
        &pre,
        &layers,
        &policy_dir,
        bindings,
    )
    .unwrap();

    let mut saw_nonzero_signed = false;
    for mode in 0..3 {
        let _mode = EnvRestore::set(&[("FULL_MDP_MMAP_SPILL", if mode == 2 { "1" } else { "0" })]);
        let legacy_scratch = root.join(format!("legacy-{mode}"));
        let compiled_scratch = root.join(format!("compiled-{mode}"));
        fs::create_dir(&legacy_scratch).unwrap();
        fs::create_dir(&compiled_scratch).unwrap();
        let legacy = sweep(
            &graph,
            &qfwd,
            &layers,
            &policy_dir,
            &legacy_scratch,
            &reset_qids,
            None,
            mode != 0,
        );
        let compiled = sweep(
            &graph,
            &qfwd,
            &layers,
            &policy_dir,
            &compiled_scratch,
            &reset_qids,
            Some(&prepared),
            mode != 0,
        );
        for values in [
            vec![-2.25; reset_qids.len()],
            (0..reset_qids.len())
                .map(|i| {
                    if i % 2 == 0 {
                        16_777_216.0
                    } else {
                        -16_777_216.0
                    }
                })
                .collect(),
        ] {
            for linear in [false, true] {
                let expected = scalar_path(&qfwd, &qids, &placed, &reset_qids, &values, !linear);
                saw_nonzero_signed |= linear && expected.iter().any(|&value| value < 0.0);
                let raw = legacy.apply(&values, linear).unwrap();
                let optimized = compiled.apply(&values, linear).unwrap();
                assert_bits(&raw, &expected, "original scalar sweep");
                assert_bits(
                    &optimized,
                    &expected,
                    &format!("compiled mode={mode} linear={linear}"),
                );
                assert_bits(
                    &compiled.apply(&values, linear).unwrap(),
                    &expected,
                    "repeated compiled sweep",
                );
            }
        }
        // apply_terminal accepts a caller-defined table. Entries sharing a
        // formula ID intentionally disagree, so the prepared path must retain
        // this API's raw table semantics.
        let arbitrary: Vec<[f32; PIECE_COUNT]> = (0..quotient.states.len())
            .map(|qid| std::array::from_fn(|p| ((qid * 13 + p * 7) % 41) as f32 - 19.0))
            .collect();
        assert_bits(
            &compiled.apply_terminal(&arbitrary).unwrap(),
            &legacy.apply_terminal(&arbitrary).unwrap(),
            "caller-defined terminal table",
        );
    }
    assert!(
        saw_nonzero_signed,
        "fixture must exercise a nonzero signed homogeneous result"
    );

    // The acceptance path must independently run the original arithmetic even
    // when both optional L9 accelerators are present. File-only sinks expose
    // the distinction: compiled applies have no L9 sink, the original has one.
    {
        let _original_mode =
            EnvRestore::set(&[("FULL_MDP_L9_DIRECT", "1"), ("FULL_MDP_MMAP_SPILL", "0")]);
        let scratch = root.join("forced-original");
        fs::create_dir(&scratch).unwrap();
        let compiled = sweep(
            &graph,
            &qfwd,
            &layers,
            &policy_dir,
            &scratch,
            &reset_qids,
            Some(&prepared),
            false,
        );
        let reset = vec![-2.25; reset_qids.len()];
        for linear in [false, true] {
            let expected = scalar_path(&qfwd, &qids, &placed, &reset_qids, &reset, !linear);
            assert_bits(
                &compiled.apply(&reset, linear).unwrap(),
                &expected,
                "prepared apply with direct flag enabled",
            );
            assert!(!mdp_f32_value_path(&scratch, 9).exists());
            assert_bits(
                &compiled.apply_original(&reset, linear).unwrap(),
                &expected,
                "forced original with both accelerators enabled",
            );
            let terminal = if linear { -2.25f32 } else { -1.25f32 };
            assert_eq!(
                fs::read(mdp_f32_value_path(&scratch, 9)).unwrap(),
                terminal.to_le_bytes(),
            );
            fs::remove_file(mdp_f32_value_path(&scratch, 9)).unwrap();
        }
    }

    // A policy outside L9 is also part of the fixed operator. Changing it must
    // reject the prepared plan before any existing scratch value is touched.
    let scratch = root.join("mutation-scratch");
    fs::create_dir(&scratch).unwrap();
    let accepted = b"preserve existing scratch bytes";
    for stage in 0..10 {
        fs::write(mdp_f32_value_path(&scratch, stage), accepted).unwrap();
    }
    let policy0 = mdp_policy_u16_path(&policy_dir, 0);
    let original = fs::read(&policy0).unwrap();
    let mut changed = original.clone();
    changed[0] ^= 1;
    fs::write(&policy0, changed).unwrap();
    let compiled = sweep(
        &graph,
        &qfwd,
        &layers,
        &policy_dir,
        &scratch,
        &reset_qids,
        Some(&prepared),
        false,
    );
    let error = compiled
        .apply(&vec![-2.25; reset_qids.len()], false)
        .unwrap_err();
    assert!(format!("{error:#}").contains("chang"), "{error:#}");
    for stage in 0..10 {
        assert_eq!(
            fs::read(mdp_f32_value_path(&scratch, stage)).unwrap(),
            accepted
        );
    }
}
