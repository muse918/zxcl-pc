//! Explicitly invoked, bounded checks against existing read-only solver datasets.
//!
//! Inputs are selected using ZXCL_VALIDATION_GRAPH, ZXCL_VALIDATION_PRUNED,
//! ZXCL_VALIDATION_VALUES and ZXCL_VALIDATION_POLICY. No source file is opened
//! for writing. Only whole current-layer blocks are copied into temporary files.

use super::*;

struct ScopedEnv(&'static str, Option<std::ffi::OsString>);

impl ScopedEnv {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let old = env::var_os(name);
        match value {
            Some(value) => env::set_var(name, value),
            None => env::remove_var(name),
        }
        Self(name, old)
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        match &self.1 {
            Some(value) => env::set_var(self.0, value),
            None => env::remove_var(self.0),
        }
    }
}

fn input(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("explicit real-data validation requires {name}"))
}

fn output(path: &Path, bytes: u64) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len(bytes)?;
    Ok(file)
}

fn bounded_range(
    layer: &layer_store::FieldChainLayer,
    start: usize,
    max_states: u64,
) -> Result<std::ops::Range<usize>> {
    anyhow::ensure!(start < layer.metas.len(), "empty real-data sample");
    let mut states = 0u64;
    let mut end = start;
    for meta in layer.metas.iter().skip(start).take(4096) {
        if states + u64::from(meta.count) > max_states {
            break;
        }
        states += u64::from(meta.count);
        end += 1;
    }
    anyhow::ensure!(end > start, "first whole block exceeds sample state limit");
    Ok(start..end)
}

fn copy_current_sample(
    original: &layer_store::FieldChainLayer,
    range: std::ops::Range<usize>,
    prefix: &Path,
) -> Result<layer_store::FieldChainLayer> {
    let mut writer = layer_store::FieldChainWriter::create(prefix)?;
    for bi in range {
        let field = original.metas[bi].field_id;
        for qid in original.read_block_chains(bi) {
            writer.push_pair_key((u64::from(field) << 32) | u64::from(qid))?;
        }
    }
    writer.finish()?;
    layer_store::FieldChainLayer::open_index_only(prefix)
}

fn describe_stats(stats: &FullMdpLayerStats) -> String {
    format!(
        "states={} cache_hit={} cache_miss={} evictions={} materialized_states={} bitmap_blocks={}",
        stats.total_states,
        stats.dense_cache_hits,
        stats.dense_cache_misses,
        stats.dense_evictions,
        stats.dense_materialized_states,
        stats.bitmap_blocks
    )
}

fn same_bytes(expected: &[u8], actual: &[u8], context: &str) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "{context}: byte count {} != {}",
        expected.len(),
        actual.len()
    );
    if let Some(at) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        anyhow::bail!(
            "{context}: first differing byte at {at}: {} != {}",
            expected[at],
            actual[at]
        );
    }
    Ok(())
}

#[test]
#[ignore = "bounded real-data benchmark; requires explicit ZXCL_VALIDATION_* paths and release mode"]
fn bounded_real_data_grouped_kernels_match_production_chunks() -> Result<()> {
    let _lock = TEST_ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let _cache_env = ScopedEnv::set("FULL_MDP_CACHE_STATS", Some("1"));
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run this benchmark in release mode"
    );
    let graph_path = input("ZXCL_VALIDATION_GRAPH")?;
    let pruned = input("ZXCL_VALIDATION_PRUNED")?;
    let values = input("ZXCL_VALIDATION_VALUES")?;
    let policies = input("ZXCL_VALIDATION_POLICY")?;
    let max_states: u64 = env::var("ZXCL_VALIDATION_MAX_STATES")
        .unwrap_or_else(|_| "2200000".into())
        .parse()?;
    anyhow::ensure!(
        max_states <= 3_000_000,
        "sample limit must be at most three million states"
    );
    let repetitions: usize = env::var("ZXCL_VALIDATION_REPETITIONS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    anyhow::ensure!((1..=5).contains(&repetitions), "repetitions must be 1..=5");
    let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
    let setup_start = Instant::now();
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let qfwd = {
        let old_index = enumerate_states();
        let q_index = build_q_index_from_old(&old_index);
        anyhow::ensure!(q_index.states.len() == CHECK_QSTATE_COUNT);
        build_qforward_graph(&q_index)
    };
    let legal = build_qfwd_legal_placed_table(&qfwd, CHECK_QSTATE_COUNT);
    // These sample layers have no terminal transitions. Keep the terminal table
    // shape valid, while avoiding an irrelevant layer-0 read in setup.
    let terminal = vec![[0.0f32; PIECE_COUNT]; CHECK_QSTATE_COUNT];
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("two-line field missing")?;
    eprintln!(
        "real-data validation setup_seconds={:.6}",
        setup_start.elapsed().as_secs_f64()
    );
    for layer in [2usize, 5] {
        let original =
            layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned, layer))?;
        let start = if layer == 5 {
            original.metas.len() / 2
        } else {
            0
        };
        let range = bounded_range(
            &original,
            start,
            if layer == 2 {
                max_states.min(512_000)
            } else {
                max_states
            },
        )?;
        let original_chain_start = original.metas[range.start].chain_offset;
        let cur = copy_current_sample(
            &original,
            range.clone(),
            &scratch.path.join(format!("current-{layer}")),
        )?;
        let expected_policy_bytes = original
            .total_states
            .checked_mul(2)
            .context("policy size overflow")?;
        drop(original);
        let next =
            layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned, layer + 1))?;
        let next_path = mdp_f32_value_path(&values, layer + 1);
        anyhow::ensure!(fs::metadata(&next_path)?.len() == next.total_states * 4);
        let source_policy = File::open(mdp_policy_u16_path(&policies, layer))?;
        anyhow::ensure!(
            source_policy.metadata()?.len() == expected_policy_bytes,
            "source policy and membership sizes differ"
        );
        let mut policy_bytes = vec![0; usize::try_from(cur.total_states * 2)?];
        read_exact_at_file(&source_policy, &mut policy_bytes, original_chain_start * 2)?;
        let policy_path = scratch.path.join(format!("input-policy-{layer}"));
        fs::write(&policy_path, policy_bytes)?;
        let policy = File::open(&policy_path)?;
        // Match the original production driver's 512-block chunks. The grouped
        // path retains one lookup cache across those exact same chunk boundaries.
        let ranges: Vec<_> = (0..cur.metas.len())
            .step_by(512)
            .map(|bi| bi..(bi + 512).min(cur.metas.len()))
            .collect();
        let baseline_path = scratch.path.join(format!("baseline-{layer}.f32"));
        let grouped_path = scratch.path.join(format!("grouped-{layer}.f32"));
        let baseline_policy_path = scratch.path.join(format!("baseline-{layer}.u16"));
        let grouped_policy_path = scratch.path.join(format!("grouped-{layer}.u16"));
        let baseline = output(&baseline_path, cur.total_states * 4)?;
        let grouped = output(&grouped_path, cur.total_states * 4)?;
        let baseline_policy = output(&baseline_policy_path, cur.total_states * 2)?;
        let grouped_policy = output(&grouped_policy_path, cur.total_states * 2)?;
        // Approximate the reference 16 GiB / 20-worker production cache. A tiny
        // cache here creates dense-field eviction thrash rather than measuring
        // chunk reuse under the intended operating conditions.
        let cache_mib: usize = env::var("ZXCL_VALIDATION_CACHE_MIB")
            .unwrap_or_else(|_| "768".into())
            .parse()?;
        anyhow::ensure!((64..=1024).contains(&cache_mib));
        let cache_bytes = cache_mib * 1024 * 1024;
        let kernel = |fixed: bool, group: bool| -> Result<(f64, FullMdpLayerStats)> {
            let file = if group { &grouped } else { &baseline };
            let policy_out = if group {
                &grouped_policy
            } else {
                &baseline_policy
            };
            let began = Instant::now();
            let mut stats = FullMdpLayerStats::default();
            if fixed {
                let selected = if group {
                    std::iter::once(0..cur.metas.len()).collect()
                } else {
                    ranges.clone()
                };
                for chunk in selected {
                    let result = compute_fixed_policy_chunk_f32(
                        layer,
                        chunk,
                        &cur,
                        &policy,
                        Some(file),
                        None,
                        &graph,
                        &qfwd,
                        &terminal,
                        Some(&next),
                        Some(&next_path),
                        None,
                        1.0,
                        cache_bytes,
                        CHECK_QSTATE_COUNT,
                        65_536,
                        two_line,
                    )?;
                    stats.add(&result.stats);
                }
            } else if group {
                stats = compute_full_mdp_range_group_f32_rolling_sinks(
                    layer,
                    &ranges,
                    &cur,
                    Some(file),
                    None,
                    Some(policy_out),
                    &graph,
                    &qfwd,
                    &legal,
                    &terminal,
                    Some(&next),
                    Some(&next_path),
                    None,
                    None,
                    1.0,
                    cache_bytes,
                    CHECK_QSTATE_COUNT,
                    65_536,
                    two_line,
                    false,
                    false,
                    true,
                    false,
                )?
                .stats;
            } else {
                for chunk in &ranges {
                    let result = compute_full_mdp_chunk_f32(
                        layer,
                        chunk.clone(),
                        &cur,
                        None,
                        file,
                        Some(policy_out),
                        &graph,
                        &qfwd,
                        &legal,
                        &terminal,
                        Some(&next),
                        Some(&next_path),
                        None,
                        1.0,
                        cache_bytes,
                        CHECK_QSTATE_COUNT,
                        65_536,
                        two_line,
                        false,
                        false,
                        true,
                        false,
                    )?;
                    stats.add(&result.stats);
                }
            }
            Ok((began.elapsed().as_secs_f64(), stats))
        };
        eprintln!(
            "real-data layer={layer} original_blocks={range:?} sampled_blocks={} states={} next_bytes={} cache_bytes={cache_bytes}",
            cur.metas.len(), cur.total_states, next.total_states * 4
        );
        for fixed in [false, true] {
            let kind = if fixed { "fixed" } else { "greedy" };
            for group in [false, true] {
                let (seconds, stats) = kernel(fixed, group)?;
                eprintln!("real-data warmup layer={layer} kind={kind} grouped={group} seconds={seconds:.6} {}", describe_stats(&stats));
            }
            same_bytes(
                &fs::read(&baseline_path)?,
                &fs::read(&grouped_path)?,
                &format!("warmup values layer={layer} kind={kind}"),
            )?;
            if !fixed {
                same_bytes(
                    &fs::read(&baseline_policy_path)?,
                    &fs::read(&grouped_policy_path)?,
                    &format!("warmup policy layer={layer}"),
                )?;
            }
            for repetition in 0..repetitions {
                for group in [false, true, true, false] {
                    let (seconds, stats) = kernel(fixed, group)?;
                    anyhow::ensure!(stats.total_states == cur.total_states);
                    eprintln!("real-data measured layer={layer} kind={kind} repetition={repetition} grouped={group} seconds={seconds:.6} {}", describe_stats(&stats));
                    same_bytes(
                        &fs::read(&baseline_path)?,
                        &fs::read(&grouped_path)?,
                        &format!("measured values layer={layer} kind={kind}"),
                    )?;
                    if !fixed {
                        same_bytes(
                            &fs::read(&baseline_policy_path)?,
                            &fs::read(&grouped_policy_path)?,
                            &format!("measured policy layer={layer}"),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "bounded real-data bitmap A/B; requires explicit ZXCL_VALIDATION_* paths and release mode"]
fn bounded_real_data_ram_bitmap_modes_match() -> Result<()> {
    let _lock = TEST_ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let _stats = ScopedEnv::set("FULL_MDP_CACHE_STATS", Some("1"));
    let _bitmap = ScopedEnv::set("FULL_MDP_RAM_BITMAP_MIN_COUNT", None);
    let _dense = ScopedEnv::set("FULL_MDP_RAM_DENSE_MAX_BLOCKS", Some("160"));
    let _dense_min = ScopedEnv::set("FULL_MDP_RAM_DENSE_MIN_COUNT", Some("512"));
    let _dense_heat = ScopedEnv::set("FULL_MDP_RAM_DENSE_HEAT_MIN", Some("64"));
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run this benchmark in release mode"
    );
    let graph = FieldGraph::load_graph_bin(input("ZXCL_VALIDATION_GRAPH")?)?;
    let pruned = input("ZXCL_VALIDATION_PRUNED")?;
    let values = input("ZXCL_VALIDATION_VALUES")?;
    let policies = input("ZXCL_VALIDATION_POLICY")?;
    let repetitions: usize = env::var("ZXCL_VALIDATION_REPETITIONS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    anyhow::ensure!((1..=5).contains(&repetitions), "repetitions must be 1..=5");
    let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
    let qfwd = {
        let old_index = enumerate_states();
        let q_index = build_q_index_from_old(&old_index);
        anyhow::ensure!(q_index.states.len() == CHECK_QSTATE_COUNT);
        build_qforward_graph(&q_index)
    };
    let layer = 8usize;
    let original = layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned, layer))?;
    anyhow::ensure!(original.metas.len() >= 8, "need eight layer-8 field blocks");
    let cur = copy_current_sample(&original, 0..8, &scratch.path.join("ram-current"))?;
    let expected_policy_bytes = original
        .total_states
        .checked_mul(2)
        .context("policy size overflow")?;
    drop(original);
    anyhow::ensure!(
        cur.total_states <= 1_000_000,
        "RAM comparison sample is too large"
    );
    let next = layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned, layer + 1))?;
    let next_path = mdp_f32_value_path(&values, layer + 1);
    anyhow::ensure!(
        next.total_states * 4 <= 128 * 1024 * 1024,
        "next RAM layer exceeds 128 MiB validation bound"
    );
    let next_values = load_f32_values_vec(&next_path, next.total_states)?;
    let source_policy = File::open(mdp_policy_u16_path(&policies, layer))?;
    anyhow::ensure!(
        source_policy.metadata()?.len() == expected_policy_bytes,
        "source policy and membership sizes differ"
    );
    let mut policy_bytes = vec![0; usize::try_from(cur.total_states * 2)?];
    read_exact_at_file(&source_policy, &mut policy_bytes, 0)?;
    let policy_path = scratch.path.join("ram-input-policy");
    fs::write(&policy_path, policy_bytes)?;
    let policy = File::open(policy_path)?;
    let reference_path = scratch.path.join("ram-file-reference.f32");
    let measured_path = scratch.path.join("ram-measured.f32");
    let reference = output(&reference_path, cur.total_states * 4)?;
    let measured = output(&measured_path, cur.total_states * 4)?;
    let terminal = vec![[0.0f32; PIECE_COUNT]; CHECK_QSTATE_COUNT];
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("two-line field missing")?;
    let cache_bytes = 512 * 1024 * 1024;
    compute_fixed_policy_chunk_f32(
        layer,
        0..cur.metas.len(),
        &cur,
        &policy,
        Some(&reference),
        None,
        &graph,
        &qfwd,
        &terminal,
        Some(&next),
        Some(&next_path),
        None,
        1.0,
        cache_bytes,
        CHECK_QSTATE_COUNT,
        65_536,
        two_line,
    )?;
    let expected = fs::read(&reference_path)?;
    eprintln!("real-data bitmap layer={layer} blocks={} states={} next_ram_bytes={} cache_bytes={cache_bytes}", cur.metas.len(), cur.total_states, next.total_states * 4);
    let kernel = |bitmap: bool| -> Result<(f64, FullMdpLayerStats)> {
        env::set_var(
            "FULL_MDP_RAM_BITMAP_MIN_COUNT",
            if bitmap { "128" } else { "0" },
        );
        let began = Instant::now();
        let result = compute_fixed_policy_chunk_f32(
            layer,
            0..cur.metas.len(),
            &cur,
            &policy,
            Some(&measured),
            None,
            &graph,
            &qfwd,
            &terminal,
            Some(&next),
            Some(&next_path),
            Some(&next_values),
            1.0,
            cache_bytes,
            CHECK_QSTATE_COUNT,
            65_536,
            two_line,
        )?;
        let seconds = began.elapsed().as_secs_f64();
        same_bytes(
            &expected,
            &fs::read(&measured_path)?,
            &format!("RAM bitmap={bitmap}"),
        )?;
        anyhow::ensure!(result.stats.total_states == cur.total_states);
        anyhow::ensure!(
            (result.stats.bitmap_blocks > 0) == bitmap,
            "bitmap A/B did not exercise the requested membership representation"
        );
        Ok((seconds, result.stats))
    };
    for dense in [160, 0] {
        env::set_var("FULL_MDP_RAM_DENSE_MAX_BLOCKS", dense.to_string());
        for bitmap in [false, true] {
            let (seconds, stats) = kernel(bitmap)?;
            eprintln!(
                "real-data bitmap warmup dense_max={dense} bitmap={bitmap} seconds={seconds:.6} {}",
                describe_stats(&stats)
            );
        }
        for repetition in 0..repetitions {
            for bitmap in [false, true, true, false] {
                let (seconds, stats) = kernel(bitmap)?;
                eprintln!("real-data bitmap measured dense_max={dense} repetition={repetition} bitmap={bitmap} seconds={seconds:.6} {}", describe_stats(&stats));
            }
        }
    }
    Ok(())
}
