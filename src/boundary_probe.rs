//! Read-only, bounded samples of the compiled boundary operator. This command
//! measures construction/application and point-box branch pruning; it does not
//! solve or certify the complete optimality fixed point.

use super::*;
use crate::boundary_dag::{CompileLimits, CompileMode, Compiler, StateKey};
use crate::certified_bounds::{self, Interval};
use crate::layer_store::FieldChainLayer;
use crate::pre_reveal::PreRevealIndex;

fn setting(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

fn decode_qid(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | u32::from(bytes[1]) << 8 | u32::from(bytes[2]) << 16
}

fn checked_edges(graph: &FieldGraph, field: u32, placed: u8) -> Result<&[u32]> {
    anyhow::ensure!((placed as usize) < PIECE_COUNT, "invalid placed piece");
    let offsets = graph
        .offsets
        .get(field as usize)
        .context("sample field outside graph")?;
    graph
        .targets
        .get(offsets[placed as usize] as usize..offsets[placed as usize + 1] as usize)
        .context("invalid sample field edge offsets")
}

fn greedy_action_exists(
    graph: &FieldGraph,
    q_index: &QStateIndex,
    qfwd: &QForwardGraph,
    state: StateKey,
) -> Result<bool> {
    let (layer, field, qid) = state;
    let q = q_index
        .states
        .get(qid as usize)
        .context("sample qid outside quotient index")?;
    let offsets = qfwd
        .offsets
        .get(qid as usize)
        .context("sample qid outside qforward graph")?;
    let (a, b) = decode_avail_pair(q.avail);
    for placed in [a, b] {
        let start = offsets[placed as usize] as usize;
        let end = offsets[placed as usize + 1] as usize;
        let records = qfwd
            .records
            .get(start..end)
            .context("invalid sample qforward offsets")?;
        if records.is_empty() {
            continue;
        }
        let edges = checked_edges(graph, field, placed)?;
        if edges
            .iter()
            .any(|&target| layer != 9 || target == EMPTY_FIELD_ID)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// One root per spaced field interval. At most sixteen fields are considered
/// in an interval. Fixed sampling reads at most one bounded policy block at a
/// time; greedy sampling considers at most 64 spaced qids in a field.
fn sample_roots(
    graph: &FieldGraph,
    q_index: &QStateIndex,
    qfwd: &QForwardGraph,
    layer: usize,
    original: &FieldChainLayer,
    policy: Option<&File>,
    wanted: usize,
) -> Result<Vec<StateKey>> {
    let intervals = wanted.min(original.metas.len());
    anyhow::ensure!(intervals > 0, "sample layer has no field blocks");
    let membership = File::open(&original.chains_path)?;
    let mut roots = Vec::with_capacity(intervals);
    let mut raw_policy = Vec::new();
    for slot in 0..intervals {
        let lo = slot * original.metas.len() / intervals;
        let hi = (slot + 1) * original.metas.len() / intervals;
        let mut selected = None;
        for probe in 0..(hi - lo).min(16) {
            let block = lo + ((hi - lo) / 2 + probe) % (hi - lo);
            let meta = &original.metas[block];
            anyhow::ensure!(
                (meta.field_id as usize) < graph.offsets.len(),
                "sample membership field outside graph"
            );
            let count = meta.count as usize;
            anyhow::ensure!(
                count <= q_index.states.len(),
                "sample membership block exceeds quotient domain"
            );
            if count == 0 {
                continue;
            }
            let start = (slot * 7919 + probe * 104729) % count;
            if let Some(policy) = policy {
                raw_policy.resize(
                    count
                        .checked_mul(2)
                        .context("sample policy size overflow")?,
                    0,
                );
                read_exact_at_file(
                    policy,
                    &mut raw_policy,
                    meta.chain_offset
                        .checked_mul(2)
                        .context("sample policy offset overflow")?,
                )?;
                for step in 0..count {
                    let rank = (start + step) % count;
                    let code = u16::from_le_bytes([raw_policy[rank * 2], raw_policy[rank * 2 + 1]]);
                    if code == POLICY_NONE_U16 {
                        continue;
                    }
                    let mut raw = [0; 3];
                    let index = meta
                        .chain_offset
                        .checked_add(rank as u64)
                        .context("sample membership index overflow")?;
                    read_exact_at_file(
                        &membership,
                        &mut raw,
                        index
                            .checked_mul(3)
                            .context("sample membership offset overflow")?,
                    )?;
                    let qid = decode_qid(&raw);
                    anyhow::ensure!(
                        (qid as usize) < q_index.states.len(),
                        "sample membership qid outside quotient domain"
                    );
                    selected = Some((layer, meta.field_id, qid));
                    break;
                }
            } else {
                let probes = count.min(64);
                for step in 0..probes {
                    let rank = (start + step * count / probes) % count;
                    let mut raw = [0; 3];
                    let index = meta
                        .chain_offset
                        .checked_add(rank as u64)
                        .context("sample membership index overflow")?;
                    read_exact_at_file(
                        &membership,
                        &mut raw,
                        index
                            .checked_mul(3)
                            .context("sample membership offset overflow")?,
                    )?;
                    let state = (layer, meta.field_id, decode_qid(&raw));
                    if greedy_action_exists(graph, q_index, qfwd, state)? {
                        selected = Some(state);
                        break;
                    }
                }
            }
            if selected.is_some() {
                break;
            }
        }
        if let Some(state) = selected {
            roots.push(state);
        }
    }
    roots.sort_unstable();
    roots.dedup();
    anyhow::ensure!(
        !roots.is_empty(),
        "bounded sampling found no non-NONE or legal-action roots"
    );
    Ok(roots)
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

pub(super) fn run(args: &[String]) -> Result<()> {
    anyhow::ensure!((8..=9).contains(&args.len()),
        "usage: {} boundary-dag-probe <graph> <pruned> <policy_dir|greedy> <layer 7|8|9> <max_roots> <l0.f32> [repetitions=10]", args[0]);
    let graph_path = Path::new(&args[2]);
    let pruned_dir = Path::new(&args[3]);
    let mode = if args[4] == "greedy" {
        CompileMode::Greedy
    } else {
        CompileMode::Fixed
    };
    let policy_dir = if mode == CompileMode::Fixed {
        Some(Path::new(&args[4]))
    } else {
        None
    };
    let layer: usize = args[5].parse().context("invalid sample layer")?;
    anyhow::ensure!((7..=9).contains(&layer), "probe layer must be 7, 8, or 9");
    let wanted: usize = args[6].parse().context("invalid sample root count")?;
    anyhow::ensure!(
        (1..=10_000).contains(&wanted),
        "max_roots must be 1..=10000"
    );
    let repetitions: usize = args.get(8).map(|v| v.parse()).transpose()?.unwrap_or(10);
    anyhow::ensure!(
        (1..=10_000).contains(&repetitions),
        "repetitions must be 1..=10000"
    );
    let limits = CompileLimits {
        max_states: setting("PI_DAG_MAX_STATES", 2_000_000)?,
        max_nodes: setting("PI_DAG_MAX_NODES", 5_000_000)?,
        max_edges: setting("PI_DAG_MAX_EDGES", 25_000_000)?,
        field_cache_bytes: setting("PI_DAG_FIELD_CACHE_MIB", 64)?
            .checked_mul(1usize << 20)
            .context("PI_DAG_FIELD_CACHE_MIB byte size overflow")?,
    };
    anyhow::ensure!(
        limits.max_states > 0 && limits.max_nodes > 0,
        "DAG state/node caps must be positive"
    );
    let setup = Instant::now();
    eprintln!("boundary-dag-probe sample_only=1 full_solve=0 global_certificate=0 layer={layer} mode={mode:?} wanted_roots={wanted} max_states={} max_nodes={} max_edges={} field_cache_bytes={}",
        limits.max_states, limits.max_nodes, limits.max_edges, limits.field_cache_bytes);
    let graph = FieldGraph::load_graph_bin(graph_path)?;
    let old = enumerate_states();
    let q_index = build_q_index_from_old(&old);
    drop(old);
    anyhow::ensure!(
        q_index.states.len() == CHECK_QSTATE_COUNT,
        "unexpected quotient state count"
    );
    let pre = PreRevealIndex::build(&q_index)?;
    let qfwd = build_qforward_graph(&q_index);
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("TWO_LINE_HASH not found in graph")?;
    let layers = (0..10)
        .map(|l| FieldChainLayer::open_index_only(q_pruned_prefix(pruned_dir, l)))
        .collect::<Result<Vec<_>>>()?;
    let qids = layer0_reset_block_qids(&layers[0])?;
    let old_values = load_f32_values_vec(Path::new(&args[7]), layers[0].total_states)?;
    anyhow::ensure!(
        qids.len() == q_index.states.len() && qids.len() == old_values.len(),
        "probe requires one complete L0 reset block and its file-order values"
    );
    let mut seen = vec![false; q_index.states.len()];
    let mut full_values = vec![0.0f64; q_index.states.len()];
    for (&qid, &value) in qids.iter().zip(&old_values) {
        let target = full_values
            .get_mut(qid as usize)
            .context("L0 reset qid outside quotient domain")?;
        anyhow::ensure!(!seen[qid as usize], "duplicate L0 reset qid {qid}");
        anyhow::ensure!(value.is_finite(), "nonfinite input L0 value");
        seen[qid as usize] = true;
        *target = value as f64;
    }
    anyhow::ensure!(seen.iter().all(|&v| v), "incomplete L0 reset membership");
    let mut u = pre.average_f64(&full_values)?;
    drop(full_values);
    drop(old_values);
    drop(qids);
    drop(seen);
    for value in &mut u {
        *value += 1.0;
    }
    anyhow::ensure!(
        u.iter().all(|v| v.is_finite()),
        "nonfinite pre-reveal input"
    );
    anyhow::ensure!(
        mode != CompileMode::Greedy || u.iter().all(|v| *v >= 0.0),
        "greedy point-box pruning requires nonnegative pre-reveal inputs"
    );
    let terminal_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    anyhow::ensure!(
        terminal_f32.iter().all(|v| v.is_finite()),
        "pre-reveal input overflows f32"
    );
    let policy = policy_dir
        .map(|dir| File::open(mdp_policy_u16_path(dir, layer)))
        .transpose()?;
    if let Some(policy) = &policy {
        anyhow::ensure!(
            policy.metadata()?.len()
                == layers[layer]
                    .total_states
                    .checked_mul(2)
                    .context("sample policy length overflow")?,
            "sample policy file size mismatch"
        );
    }
    let roots = sample_roots(
        &graph,
        &q_index,
        &qfwd,
        layer,
        &layers[layer],
        policy.as_ref(),
        wanted,
    )?;
    eprintln!("boundary-dag-probe setup_seconds={:.9} qstates={} pre_reveal_states={} roots={} root_selection={} terminal_input=1+R(file_order_L0_values)",
        setup.elapsed().as_secs_f64(), q_index.states.len(), pre.len(), roots.len(),
        if mode == CompileMode::Fixed { "spaced_non_NONE_policy" } else { "spaced_legal_action_members" });
    let compiled = Compiler::new(
        &graph, &q_index, &qfwd, &layers, &pre, policy_dir, mode, limits, two_line,
    )?
    .compile_roots(&roots)?;
    let baseline = compiled.dag.eval_f64(&u)?;
    let baseline32 = compiled.dag.eval_f32(&terminal_f32)?;
    let mut f64_times = Vec::with_capacity(repetitions);
    let mut f32_times = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        for use_f64 in [true, false, false, true] {
            let began = Instant::now();
            if use_f64 {
                let values = compiled.dag.eval_f64(&u)?;
                f64_times.push(began.elapsed().as_secs_f64());
                std::hint::black_box(&values);
            } else {
                let values = compiled.dag.eval_f32(&terminal_f32)?;
                f32_times.push(began.elapsed().as_secs_f64());
                std::hint::black_box(&values);
            }
        }
    }
    let nonzero_roots = compiled
        .roots
        .iter()
        .filter(|&&node| baseline[node as usize] != 0.0)
        .count();
    let max_precision_difference = compiled
        .roots
        .iter()
        .map(|&node| (baseline[node as usize] - baseline32[node as usize] as f64).abs())
        .fold(0.0f64, f64::max);
    println!("boundary-dag-probe sample_only=1 full_solve=0 global_certificate=0 layer={layer} mode={mode:?} roots={} nonzero_roots={nonzero_roots} states={} missing_states={} nodes={} edges={} actions={} B_seconds={:.9} E_f64_median_seconds={:.9} E_f32_median_seconds={:.9} repetitions={repetitions} evaluations_per_precision={} field_cache_peak_bytes={} f32_f64_root_difference={:.9e} scalar_seconds=unmeasured break_even_applications=unmeasured",
        roots.len(), compiled.stats.states, compiled.stats.missing_states, compiled.stats.nodes, compiled.stats.edges, compiled.stats.action_records,
        compiled.stats.construction_seconds, median(&mut f64_times), median(&mut f32_times), repetitions * 2,
        compiled.stats.field_cache_peak_bytes, max_precision_difference);
    if mode == CompileMode::Greedy {
        let points = u
            .iter()
            .map(|&value| Interval::point(value))
            .collect::<Result<Vec<_>>>()?;
        let began = Instant::now();
        let pruned = certified_bounds::prune(&compiled.dag, &compiled.roots, &points)?;
        let pruning_seconds = began.elapsed().as_secs_f64();
        let pruned_values = pruned.dag.eval_f64(&u)?;
        for (index, (&old, &new)) in compiled.roots.iter().zip(&pruned.roots).enumerate() {
            anyhow::ensure!(
                baseline[old as usize].to_bits() == pruned_values[new as usize].to_bits(),
                "point-box pruning changed sampled f64 root {index}"
            );
        }
        println!("boundary-dag-probe-pruning sample_only=1 full_solve=0 global_certificate=0 validity=supplied_exact_point_box precision=f64 removed_branches={} original_nodes={} pruned_nodes={} original_edges={} pruned_edges={} pruning_seconds={:.9} sampled_roots_match=1 ties_retained=1",
            pruned.removed_branches, pruned.original_node_count, pruned.dag.node_count(), pruned.original_edge_count, pruned.dag.edge_count(), pruning_seconds);
    }
    Ok(())
}
