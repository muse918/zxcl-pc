use super::*;

const QCOUNT: usize = 4;
const TWO_LINE: FieldId = 3;

fn graph_fixture() -> FieldGraph {
    // Distinct targets, a repeated edge and two legal placed pieces expose code
    // indexing mistakes independently of whether the resulting value is zero.
    let rows: [[&[u32]; 2]; 4] = [
        [&[1, 2, 1], &[2, 1]],
        [&[2], &[1, 2]],
        [&[TWO_LINE], &[2, 1]],
        [&[EMPTY_FIELD_ID, 0], &[EMPTY_FIELD_ID, 0]],
    ];
    let mut targets = Vec::new();
    let mut offsets = Vec::new();
    for row in rows {
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = targets.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                targets.extend_from_slice(row[p]);
            }
            off[p + 1] = targets.len() as u32;
        }
        offsets.push(off);
    }
    FieldGraph {
        hashes: vec![0, 1, 2, 3],
        offsets,
        targets,
    }
}

fn qforward_fixture() -> QForwardGraph {
    let rows: [[&[u32]; 2]; QCOUNT] =
        [[&[0, 1], &[2, 3]], [&[3, 0], &[1]], [&[], &[2]], [&[], &[]]];
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    for row in rows {
        let mut off = [0; PIECE_COUNT + 1];
        off[0] = records.len() as u32;
        for p in 0..PIECE_COUNT {
            if p < 2 {
                for (reveal, &child_qid) in row[p].iter().enumerate() {
                    records.push(qgraph::QForwardRec {
                        reveal: reveal as u8,
                        child_qid,
                    });
                }
            }
            off[p + 1] = records.len() as u32;
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
    for (field, qids) in blocks {
        for &qid in *qids {
            writer
                .push_pair_key((*field as u64) << 32 | qid as u64)
                .unwrap();
        }
    }
    writer.finish().unwrap();
    layer_store::FieldChainLayer::open_index_only(prefix).unwrap()
}

fn uneven_fixture(prefix: &Path) -> layer_store::FieldChainLayer {
    layer_fixture(
        prefix,
        &[(0, &[0, 1, 2, 3]), (1, &[1]), (2, &[0, 2, 3]), (3, &[0, 1])],
    )
}

fn terminal_fixture() -> Vec<[f32; PIECE_COUNT]> {
    let mut terminal = vec![[0.0; PIECE_COUNT]; QCOUNT];
    for (q, row) in terminal.iter_mut().enumerate() {
        for (p, value) in row.iter_mut().enumerate() {
            *value = 1.375 + q as f32 * 0.125 + p as f32 * 0.25;
        }
    }
    terminal
}

fn chosen_policy(layer: usize) -> Vec<u16> {
    if layer == 9 {
        let mut codes = vec![POLICY_NONE_U16; 10];
        codes[8] = encode_policy_u16(0, 0);
        codes[9] = encode_policy_u16(1, 0);
        codes
    } else {
        vec![
            encode_policy_u16(0, 0),
            encode_policy_u16(0, 1),
            encode_policy_u16(1, 0),
            POLICY_NONE_U16,
            encode_policy_u16(1, 0),
            encode_policy_u16(0, 0),
            encode_policy_u16(1, 1),
            POLICY_NONE_U16,
            encode_policy_u16(0, 1),
            encode_policy_u16(1, 1),
        ]
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_policy(path: &Path, codes: &[u16]) {
    fs::write(
        path,
        codes
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
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

// Each source/sink can independently use positional files, ordinary RAM or an
// mmap. All output bytes start poisoned, so an omitted range cannot pass as zero.
fn kernel_output(
    root: &Path,
    layer: usize,
    ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    policy: &File,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal: &[[f32; PIECE_COUNT]],
    next: &layer_store::FieldChainLayer,
    next_values: &[f32],
    gamma: f32,
    source: usize,
    sink: usize,
    sparse_threshold: usize,
) -> Result<(Vec<u8>, FullMdpLayerStats)> {
    let next_path = root.join("kernel-next.f32");
    fs::write(&next_path, f32_bytes(next_values))?;
    let next_file = File::open(&next_path)?;
    let next_map = unsafe { memmap2::Mmap::map(&next_file) }?;
    let next_mapped_values =
        unsafe { std::slice::from_raw_parts(next_map.as_ptr() as *const f32, next_values.len()) };
    let actual_path = root.join("kernel-actual.f32");
    fs::write(
        &actual_path,
        f32_bytes(&vec![f32::NAN; cur.total_states as usize]),
    )?;
    let actual_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&actual_path)?;
    let mut ram = vec![f32::NAN; cur.total_states as usize];
    let mut mapped = if sink == 2 {
        Some(unsafe { memmap2::MmapMut::map_mut(&actual_file) }?)
    } else {
        None
    };
    let ram_out = match sink {
        1 => Some(SharedF32Out {
            ptr: ram.as_mut_ptr(),
            len: ram.len(),
        }),
        2 => Some(SharedF32Out {
            ptr: mapped.as_mut().unwrap().as_mut_ptr() as *mut f32,
            len: ram.len(),
        }),
        _ => None,
    };
    let next_ram = match source {
        0 => None,
        1 => Some(next_values),
        _ => Some(next_mapped_values),
    };
    let mut stats = FullMdpLayerStats::default();
    for range in ranges {
        let result = compute_fixed_policy_chunk_f32(
            layer,
            range.clone(),
            cur,
            policy,
            if sink == 0 { Some(&actual_file) } else { None },
            ram_out,
            graph,
            qfwd,
            terminal,
            Some(next),
            Some(&next_path),
            next_ram,
            gamma,
            1024 * 1024,
            QCOUNT,
            sparse_threshold,
            TWO_LINE,
        )?;
        assert_eq!(result.blocks, range.len());
        stats.add(&result.stats);
    }
    let bytes = if sink == 1 {
        f32_bytes(&ram)
    } else if let Some(map) = mapped.as_ref() {
        map.to_vec()
    } else {
        fs::read(actual_path)?
    };
    Ok((bytes, stats))
}

#[test]
fn fixed_policy_group_reads_match_single_blocks_for_all_sources_and_sinks() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let terminal = terminal_fixture();
    let cur = uneven_fixture(&root.join("cur"));
    let next = layer_fixture(
        &root.join("next"),
        &[(1, &[0, 1, 3]), (2, &[1, 2]), (3, &[0, 3])],
    );
    let next_values = [0.5, 0.25, 0.0, 0.75, 0.125, 1.5, 0.0];
    let gamma = 0.9375;
    for layer in [3, 4, 9] {
        let policy_path = root.join("policy.u16");
        write_policy(&policy_path, &chosen_policy(layer));
        let policy = File::open(policy_path).unwrap();
        let mut expected = vec![0.0; cur.total_states as usize];
        if layer == 9 {
            expected[8] = terminal[0][0];
            expected[9] = terminal[1][1];
        } else {
            expected[0] = (gamma * 0.75) / 2.0;
            expected[2] = gamma * 0.125;
            expected[4] = gamma * 0.25;
            expected[5] = if layer == 4 {
                terminal[0][0]
            } else {
                (gamma * 1.5) / 2.0
            };
        }
        let reference = kernel_output(
            root,
            layer,
            &[0..1, 1..2, 2..3, 3..4],
            &cur,
            &policy,
            &graph,
            &qfwd,
            &terminal,
            &next,
            &next_values,
            gamma,
            0,
            0,
            65_536,
        )
        .unwrap()
        .0;
        assert_eq!(
            reference,
            f32_bytes(&expected),
            "explicit reveal probabilities layer={layer}"
        );
        // 1..4 starts at chain offset 4, so absolute policy offsets and block-local
        // offsets are both exercised by a single batched policy read.
        for ranges in [&[0..4][..], &[0..1, 1..4][..]] {
            for source in 0..3 {
                for sink in 0..3 {
                    for sparse in [0, 65_536] {
                        let (actual, stats) = kernel_output(
                            root,
                            layer,
                            ranges,
                            &cur,
                            &policy,
                            &graph,
                            &qfwd,
                            &terminal,
                            &next,
                            &next_values,
                            gamma,
                            source,
                            sink,
                            sparse,
                        )
                        .unwrap();
                        assert_eq!(actual, reference,
                            "layer={layer} source={source} sink={sink} sparse={sparse} ranges={ranges:?}");
                        assert_eq!(stats.total_states, cur.total_states);
                        assert_eq!(stats.terminal_2_edges, if layer == 4 { 1 } else { 0 });
                        assert_eq!(stats.terminal_4_edges, if layer == 9 { 2 } else { 0 });
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_policy_uses_the_first_greedy_tie_choice_per_source() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let legal = build_qfwd_legal_placed_table(&qfwd, QCOUNT);
    let terminal = terminal_fixture();
    let cur = uneven_fixture(&root.join("cur"));
    let next = layer_fixture(
        &root.join("next"),
        &[
            (0, &[0, 1, 2, 3]),
            (1, &[0, 1, 2, 3]),
            (2, &[0, 1, 2, 3]),
            (3, &[0, 1, 2, 3]),
        ],
    );
    let values = vec![1.0; next.total_states as usize];
    let next_path = root.join("next.f32");
    fs::write(&next_path, f32_bytes(&values)).unwrap();
    let expected_path = root.join("greedy.f32");
    let expected_file = output_file(&expected_path, cur.total_states * 4);
    let policy_path = root.join("policy.u16");
    let policy = output_file(&policy_path, cur.total_states * 2);
    compute_full_mdp_chunk_f32(
        3,
        0..4,
        &cur,
        None,
        &expected_file,
        Some(&policy),
        &graph,
        &qfwd,
        &legal,
        &terminal,
        Some(&next),
        Some(&next_path),
        Some(&values),
        1.0,
        1024 * 1024,
        QCOUNT,
        65_536,
        TWO_LINE,
        false,
        false,
        false,
        false,
    )
    .unwrap();
    let codes: Vec<u16> = fs::read(&policy_path)
        .unwrap()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        codes,
        vec![
            0,
            0,
            0x0100,
            POLICY_NONE_U16,
            0,
            0,
            0x0100,
            POLICY_NONE_U16,
            1,
            1
        ]
    );
    let actual = kernel_output(
        root,
        3,
        &[0..2, 2..4],
        &cur,
        &policy,
        &graph,
        &qfwd,
        &terminal,
        &next,
        &values,
        1.0,
        1,
        1,
        65_536,
    )
    .unwrap()
    .0;
    assert_eq!(actual, fs::read(expected_path).unwrap());
}

#[test]
fn fixed_policy_rejects_invalid_codes_and_incomplete_policy_reads() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let terminal = terminal_fixture();
    let cur = uneven_fixture(&root.join("cur"));
    let next = uneven_fixture(&root.join("next"));
    let values = vec![1.0; next.total_states as usize];
    let cases = [
        (3, 0, 0x0700u16, "corrupt policy code"),
        (3, 0, encode_policy_u16(0, 3), "policy edge index"),
        (3, 3, encode_policy_u16(0, 0), "no reveal records"),
        (9, 0, encode_policy_u16(0, 0), "non-terminal field"),
    ];
    for (layer, at, code, message) in cases {
        let mut codes = chosen_policy(layer);
        codes[at] = code;
        let policy_path = root.join("policy.u16");
        write_policy(&policy_path, &codes);
        let policy = File::open(&policy_path).unwrap();
        let err = kernel_output(
            root,
            layer,
            &[0..4],
            &cur,
            &policy,
            &graph,
            &qfwd,
            &terminal,
            &next,
            &values,
            1.0,
            1,
            1,
            65_536,
        )
        .unwrap_err();
        assert!(err.to_string().contains(message), "{err:#}");
    }
    let policy_path = root.join("truncated.u16");
    write_policy(&policy_path, &chosen_policy(3)[..9]);
    let policy = File::open(policy_path).unwrap();
    assert!(kernel_output(
        root,
        3,
        &[0..4],
        &cur,
        &policy,
        &graph,
        &qfwd,
        &terminal,
        &next,
        &values,
        1.0,
        0,
        0,
        65_536
    )
    .is_err());
}

#[test]
fn fixed_policy_loader_dependency_rejects_non_contiguous_blocks() {
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let prefix = owned.path.join("cur");
    let cur = uneven_fixture(&prefix);
    let original = fs::read(&cur.index_path).unwrap();
    drop(cur);
    for offset in [0u64, 3, 5, u64::MAX] {
        let mut corrupt = original.clone();
        // Header=24, each meta=16, chain_offset follows field/count (8).
        corrupt[24 + 16 + 8..24 + 16 + 16].copy_from_slice(&offset.to_le_bytes());
        fs::write(prefix.with_extension("field_index.bin"), corrupt).unwrap();
        for result in [
            layer_store::FieldChainLayer::open(&prefix),
            layer_store::FieldChainLayer::open_index_only(&prefix),
        ] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("not contiguous"), "{error}");
        }
    }
}

// Deliberately simple reference: locate every reveal directly in membership,
// without NextValueLookup, chunk batching, terminal tables or rolling buffers.
fn scalar_sweep(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    layers: &[layer_store::FieldChainLayer],
    policies: &[Vec<u16>],
    x: &[f32],
    gamma: f32,
    linear: bool,
) -> Vec<f32> {
    let reset_qids = layers[0].read_block_chains(0);
    let mut reset = [0.0f32; QCOUNT];
    for (i, q) in reset_qids.into_iter().enumerate() {
        reset[q as usize] = x[i];
    }
    let mut next_values = Vec::<f32>::new();
    for layer in (0..10).rev() {
        let mut values = Vec::new();
        for (bi, meta) in layers[layer].metas.iter().enumerate() {
            for (i, qid) in layers[layer].read_block_chains(bi).into_iter().enumerate() {
                let code = policies[layer][meta.chain_offset as usize + i];
                if code == POLICY_NONE_U16 {
                    values.push(0.0);
                    continue;
                }
                let placed = (code >> 8) as u8;
                let target = graph.edges(meta.field_id, placed)[(code & 255) as usize];
                let reveals = qfwd.records_for(qid, placed);
                let terminal =
                    (layer == 4 && target == TWO_LINE) || (layer == 9 && target == EMPTY_FIELD_ID);
                let mut sum = 0.0f32;
                for reveal in reveals {
                    if terminal {
                        sum += (if linear { 0.0 } else { 1.0 })
                            + gamma * reset[reveal.child_qid as usize];
                    } else {
                        let child_value = layers[layer + 1]
                            .metas
                            .iter()
                            .enumerate()
                            .find(|(_, block)| block.field_id == target)
                            .and_then(|(next_bi, block)| {
                                layers[layer + 1]
                                    .read_block_chains(next_bi)
                                    .iter()
                                    .position(|&q| q == reveal.child_qid)
                                    .map(|at| next_values[block.chain_offset as usize + at])
                            })
                            .unwrap_or(0.0);
                        sum += child_value;
                    }
                }
                if !terminal {
                    sum *= gamma;
                }
                values.push(sum / reveals.len() as f32);
            }
        }
        next_values = values;
    }
    next_values
}

struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvRestore {
    fn set(settings: &[(&'static str, &str)]) -> Self {
        let mut previous = Vec::new();
        for &(name, value) in settings {
            previous.push((name, env::var_os(name)));
            env::set_var(name, value);
        }
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

#[test]
fn fixed_policy_ten_layer_affine_and_linear_match_scalar_in_file_ram_and_mmap() {
    // Integration gate runs env-mutating tests with --test-threads=1.
    let _env_lock = super::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _settings = EnvRestore::set(&[
        ("FULL_MDP_BLOCK_CHUNK", "1"),
        ("FULL_MDP_LOOKUP_GROUP_CHUNKS", "2"),
        ("FULL_MDP_TASK_MAX_STATES", "5"),
        ("FULL_MDP_MEMORY_MAX_GIB", "1"),
        ("FULL_MDP_MEMORY_RESERVE_GIB", "0"),
        ("FULL_MDP_RAM_PAIR_MAX_GIB", "1"),
    ]);
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    let layers: Vec<_> = (0..10)
        .map(|layer| uneven_fixture(&root.join(format!("membership{layer}"))))
        .collect();
    let policies: Vec<_> = (0..10).map(chosen_policy).collect();
    let policy_dir = root.join("policies");
    fs::create_dir(&policy_dir).unwrap();
    for (layer, codes) in policies.iter().enumerate() {
        write_policy(&mdp_policy_u16_path(&policy_dir, layer), codes);
    }
    let reset_qids = layer0_reset_block_qids(&layers[0]).unwrap();
    let x = [0.25, -0.5, 1.75, -2.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0];
    for gamma in [1.0, 0.5, 0.9] {
        for linear in [false, true] {
            let expected = scalar_sweep(&graph, &qfwd, &layers, &policies, &x, gamma, linear);
            assert!(
                expected.iter().any(|v| *v != 0.0),
                "nontrivial reference required"
            );
            for mode in 0..4 {
                let _mode = EnvRestore::set(&[
                    ("FULL_MDP_MMAP_SPILL", if mode == 2 { "1" } else { "0" }),
                    (
                        "FULL_MDP_RAM_PAIR_MAX_GIB",
                        if mode == 3 { "0" } else { "1" },
                    ),
                ]);
                let scratch = root.join(format!("scratch-{gamma}-{linear}-{mode}"));
                fs::create_dir(&scratch).unwrap();
                // Existing garbage must not leak into mmap or file output.
                for layer in 0..10 {
                    fs::write(mdp_f32_value_path(&scratch, layer), vec![0xff; 123]).unwrap();
                }
                let sweep = FixedPolicySweep {
                    graph: &graph,
                    qfwd: &qfwd,
                    full_layers: &layers,
                    policy_dir: &policy_dir,
                    scratch_dir: &scratch,
                    two_line_field: TWO_LINE,
                    qstate_count: QCOUNT,
                    gamma,
                    cache_cap_bytes: 1024 * 1024,
                    sparse_lookup_max_count: 65_536,
                    block_chunk: 1,
                    batch_chunks_env: Some(2),
                    ram_next_layer: mode != 0,
                    ram_next_layer_max_bytes: 1024 * 1024,
                    l0_reset_qids: &reset_qids,
                    verbose_layers: false,
                    prepared_l9: None,
                    prepared_l8: None,
                    action_source: None,
                    operator_bindings: None,
                    action_resident_bytes: 0,
                };
                let actual = sweep.apply(&x, linear).unwrap_or_else(|error| {
                    panic!("gamma={gamma} linear={linear} mode={mode}: {error:#}")
                });
                assert_eq!(
                    f32_bytes(&actual),
                    f32_bytes(&expected),
                    "gamma={gamma} linear={linear} mode={mode}"
                );
                let again = sweep.apply(&x, linear).unwrap();
                assert_eq!(
                    f32_bytes(&again),
                    f32_bytes(&expected),
                    "reused scratch mode={mode}"
                );
            }
        }
    }
}

#[test]
fn fixed_policy_empty_layers_match_scalar_without_zero_length_mmaps() {
    let _env_lock = super::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _settings = EnvRestore::set(&[
        ("FULL_MDP_BLOCK_CHUNK", "1"),
        ("FULL_MDP_LOOKUP_GROUP_CHUNKS", "2"),
        ("FULL_MDP_TASK_MAX_STATES", "5"),
        ("FULL_MDP_MEMORY_MAX_GIB", "1"),
        ("FULL_MDP_MEMORY_RESERVE_GIB", "0"),
    ]);
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let root = &owned.path;
    let graph = graph_fixture();
    let qfwd = qforward_fixture();
    // L7 exercises empty output followed by a nonempty predecessor. Empty L9
    // additionally exercises the first output, where there is no pinned layer.
    for empty_layer in [7, 9] {
        let fixture = root.join(format!("empty-{empty_layer}"));
        fs::create_dir(&fixture).unwrap();
        let layers: Vec<_> = (0..10)
            .map(|layer| {
                let prefix = fixture.join(format!("membership{layer}"));
                if layer == empty_layer {
                    layer_fixture(&prefix, &[])
                } else {
                    uneven_fixture(&prefix)
                }
            })
            .collect();
        let policies: Vec<_> = (0..10)
            .map(|layer| {
                if layer == empty_layer {
                    Vec::new()
                } else {
                    chosen_policy(layer)
                }
            })
            .collect();
        let policy_dir = fixture.join("policies");
        fs::create_dir(&policy_dir).unwrap();
        for (layer, codes) in policies.iter().enumerate() {
            write_policy(&mdp_policy_u16_path(&policy_dir, layer), codes);
        }
        let reset_qids = layer0_reset_block_qids(&layers[0]).unwrap();
        let x = [0.25, -0.5, 1.75, -2.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0];
        for linear in [false, true] {
            let expected = scalar_sweep(&graph, &qfwd, &layers, &policies, &x, 0.5, linear);
            assert!(expected.iter().any(|&v| v != 0.0));
            for mode in 0..4 {
                let _mode = EnvRestore::set(&[
                    ("FULL_MDP_MMAP_SPILL", if mode == 2 { "1" } else { "0" }),
                    (
                        "FULL_MDP_RAM_PAIR_MAX_GIB",
                        if mode == 3 { "0" } else { "1" },
                    ),
                ]);
                let scratch = fixture.join(format!("scratch-{linear}-{mode}"));
                fs::create_dir(&scratch).unwrap();
                let sweep = FixedPolicySweep {
                    graph: &graph,
                    qfwd: &qfwd,
                    full_layers: &layers,
                    policy_dir: &policy_dir,
                    scratch_dir: &scratch,
                    two_line_field: TWO_LINE,
                    qstate_count: QCOUNT,
                    gamma: 0.5,
                    cache_cap_bytes: 1024 * 1024,
                    sparse_lookup_max_count: 65_536,
                    block_chunk: 1,
                    batch_chunks_env: Some(2),
                    ram_next_layer: mode != 0,
                    ram_next_layer_max_bytes: 1024 * 1024,
                    l0_reset_qids: &reset_qids,
                    verbose_layers: false,
                    prepared_l9: None,
                    prepared_l8: None,
                    action_source: None,
                    operator_bindings: None,
                    action_resident_bytes: 0,
                };
                let actual = sweep.apply(&x, linear).unwrap_or_else(|error| {
                    panic!("empty_layer={empty_layer} linear={linear} mode={mode}: {error:#}")
                });
                assert_eq!(
                    f32_bytes(&actual),
                    f32_bytes(&expected),
                    "empty_layer={empty_layer} linear={linear} mode={mode}"
                );
            }
        }
    }
}

#[test]
fn fixed_policy_coalesced_output_flushes_large_ranges_without_touching_gaps() {
    const MIB: usize = 1024 * 1024;
    const GUARD: u8 = 0xa5;
    const PREFIX: u64 = 4096;
    const GAP: u64 = 8192;
    const TRAILER: u64 = 4096;
    let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
    let sizes = [
        3 * MIB + 4,
        2 * MIB - 8,
        3 * MIB + 20,
        MIB + 12,
        2 * MIB + 4,
    ];
    let mut blocks = Vec::new();
    let mut cursor = PREFIX;
    for (index, &bytes) in sizes.iter().enumerate() {
        if index == 4 {
            cursor += GAP;
        }
        blocks.push((cursor, bytes, 17u8 + index as u8 * 29));
        cursor += bytes as u64;
    }
    let file_len = cursor + TRAILER;
    let file = output_file(&owned.path.join("coalesced.f32"), file_len);
    // Stream the sentinel and verification rather than allocating a second
    // 11 MiB expected output. The coalescing buffer plus one block stays small.
    let guard = vec![GUARD; 64 * 1024];
    for offset in (0..file_len).step_by(guard.len()) {
        let len = ((file_len - offset) as usize).min(guard.len());
        write_all_at_file(&file, &guard[..len], offset).unwrap();
    }
    let pattern = |position: usize, seed: u8| seed ^ (position % 251) as u8;
    let mut start = None;
    let mut next = 0;
    let mut buffer = Vec::new();
    for (index, &(offset, bytes, seed)) in blocks.iter().enumerate() {
        let payload: Vec<_> = (0..bytes).map(|i| pattern(i, seed)).collect();
        let meta = layer_store::FieldChainBlockMeta {
            field_id: index as u32,
            count: (bytes / 4) as u32,
            chain_offset: offset / 4,
        };
        append_block_output_and_maybe_flush(
            &file,
            &meta,
            &payload,
            &mut start,
            &mut next,
            &mut buffer,
        )
        .unwrap();
        if index == 2 {
            assert!(
                buffer.is_empty(),
                "8 MiB threshold must flush the coalesced range"
            );
            assert!(
                start.is_none(),
                "the next block must establish a fresh write offset"
            );
        }
    }
    flush_chunk_output(&file, start, &mut buffer).unwrap();
    assert!(buffer.is_empty());
    assert_eq!(file.metadata().unwrap().len(), file_len);
    let mut actual = vec![0; 64 * 1024];
    let check = |offset: u64, bytes: usize, seed: Option<u8>, actual: &mut [u8]| {
        for at in (0..bytes).step_by(actual.len()) {
            let len = (bytes - at).min(actual.len());
            read_exact_at_file(&file, &mut actual[..len], offset + at as u64).unwrap();
            for (i, &value) in actual[..len].iter().enumerate() {
                let expected = seed.map_or(GUARD, |seed| pattern(at + i, seed));
                assert_eq!(value, expected, "byte offset {}", offset + (at + i) as u64);
            }
        }
    };
    check(0, PREFIX as usize, None, &mut actual);
    for &(offset, bytes, seed) in &blocks {
        check(offset, bytes, Some(seed), &mut actual);
    }
    check(
        blocks[3].0 + blocks[3].1 as u64,
        GAP as usize,
        None,
        &mut actual,
    );
    check(cursor, TRAILER as usize, None, &mut actual);
}
