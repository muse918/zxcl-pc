//! Production entry points for gamma=1 policy evaluation before the last reveal.
//! Accepted values always retain the original layer-zero file order and pass
//! the original affine f32 sweep before they replace a previous accepted file.

use super::*;

use crate::boundary_dag::{CompileLimits, CompileMode, Compiler};
use crate::pre_reveal::PreRevealIndex;
use crate::reduced_solver::{self, Config, ReducedPolicyOperator};

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

fn checked_units(value: usize, unit: usize, label: &str) -> Result<usize> {
    value
        .checked_mul(unit)
        .with_context(|| format!("{label} byte size overflow"))
}

fn env_choice(name: &str, default: &str, choices: &[&str]) -> Result<String> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => default.to_owned(),
        Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
    };
    anyhow::ensure!(
        choices.contains(&value.as_str()),
        "invalid {name}={value:?}; expected {}",
        choices.join(" or ")
    );
    Ok(value)
}

fn finite_f64(values: &[f64], expected: usize, label: &str) -> Result<()> {
    anyhow::ensure!(
        values.len() == expected,
        "{label} length {}, expected {expected}",
        values.len()
    );
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite {label}"
    );
    Ok(())
}

fn finite_f32(values: &[f64], expected: usize, label: &str) -> Result<Vec<f32>> {
    finite_f64(values, expected, label)?;
    let converted: Vec<_> = values.iter().map(|&value| value as f32).collect();
    anyhow::ensure!(
        converted.iter().all(|value| value.is_finite()),
        "{label} cannot be represented as finite f32 values"
    );
    Ok(converted)
}

fn validate_complete_qids(qids: &[u32], count: usize) -> Result<()> {
    anyhow::ensure!(
        qids.len() == count,
        "incomplete layer0 reset membership: {} qids, expected all {count}",
        qids.len()
    );
    let mut seen = vec![false; count];
    for (index, &qid) in qids.iter().enumerate() {
        let slot = seen.get_mut(qid as usize).with_context(|| {
            format!("layer0 reset qid {qid} at file index {index} is out of range")
        })?;
        anyhow::ensure!(
            !*slot,
            "duplicate layer0 reset qid {qid} at file index {index}"
        );
        *slot = true;
    }
    anyhow::ensure!(
        seen.iter().all(|&present| present),
        "incomplete layer0 reset membership"
    );
    Ok(())
}

fn validate_reset_field(graph: &FieldGraph, field: FieldId) -> Result<()> {
    let hash = graph
        .hashes
        .get(field as usize)
        .with_context(|| format!("layer0 reset field {field} outside graph"))?;
    // The physical empty start field has hash zero. EMPTY_FIELD_ID denotes the
    // completed four-line target, which is distinct from that start geometry.
    anyhow::ensure!(
        *hash == 0,
        "reduced policy evaluation requires a physically empty layer0 reset field (hash=0), got field={field} hash={hash:#x}"
    );
    Ok(())
}

fn complete_reset_qids(
    graph: &FieldGraph,
    layer0: &layer_store::FieldChainLayer,
    count: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        layer0.metas.len() == 1,
        "reduced policy evaluation requires exactly one complete layer0 reset field block"
    );
    let reset = &layer0.metas[0];
    validate_reset_field(graph, reset.field_id)?;
    anyhow::ensure!(
        reset.chain_offset == 0
            && reset.count as usize == count
            && layer0.total_states == count as u64,
        "incomplete layer0 reset membership: expected all {count} unique quotient states"
    );
    // Read fallibly here: the legacy block helper turns a late I/O error into
    // a panic. Preserve the encoded order rather than treating offsets as qids.
    let bytes = fs::read(&layer0.chains_path)
        .with_context(|| format!("read {}", layer0.chains_path.display()))?;
    anyhow::ensure!(
        bytes.len() == checked_units(count, 3, "layer0 membership")?,
        "layer0 reset membership byte length mismatch"
    );
    let qids: Vec<_> = bytes
        .chunks_exact(3)
        .map(|bytes| graph::read_u24_le(bytes))
        .collect();
    validate_complete_qids(&qids, count)?;
    Ok(qids)
}

fn file_values_to_full_q(values: &[f64], qids: &[u32], count: usize) -> Result<Vec<f64>> {
    finite_f64(values, qids.len(), "file-order quotient values")?;
    let mut full = vec![0.0; count];
    let mut seen = vec![false; count];
    for (&qid, &value) in qids.iter().zip(values) {
        let index = qid as usize;
        anyhow::ensure!(index < count, "quotient qid {qid} outside capacity {count}");
        anyhow::ensure!(!seen[index], "duplicate quotient qid {qid}");
        seen[index] = true;
        full[index] = value;
    }
    Ok(full)
}

/// K(u): terminal values are already before revelation and contain their reward.
fn sweep_reconstruct(
    sweep: &FixedPolicySweep<'_>,
    pre: &PreRevealIndex,
    input: &[f64],
) -> Result<Vec<f64>> {
    let input = finite_f32(input, pre.len(), "reduced terminal input")?;
    let mut table = vec![[0.0; PIECE_COUNT]; sweep.qstate_count];
    for (qid, row) in table.iter_mut().enumerate() {
        for (placed, value) in row.iter_mut().enumerate() {
            if let Some(zid) = pre.zid_for_placement(qid as u32, placed as u8) {
                *value = input[zid as usize];
            }
        }
    }
    let output = sweep.apply_terminal(&table)?;
    anyhow::ensure!(
        output.len() == sweep.l0_reset_qids.len() && output.iter().all(|value| value.is_finite()),
        "invalid reduced sweep root values"
    );
    Ok(output.into_iter().map(|value| value as f64).collect())
}

fn sweep_apply_p(
    sweep: &FixedPolicySweep<'_>,
    pre: &PreRevealIndex,
    input: &[f64],
) -> Result<Vec<f64>> {
    let roots = sweep_reconstruct(sweep, pre, input)?;
    let full = file_values_to_full_q(&roots, sweep.l0_reset_qids, sweep.qstate_count)?;
    pre.average_f64(&full)
}

/// Fresh residual of the actual reduced operator, independent of GMRES reports.
fn check_reduced_residual(
    input: &[f64],
    applied: &[f64],
    config: &Config,
    accept_tol: f64,
) -> Result<(f64, f64)> {
    finite_f64(input, applied.len(), "reduced solution")?;
    finite_f64(applied, input.len(), "reduced applied solution")?;
    anyhow::ensure!(!input.is_empty(), "empty reduced solution");
    let mut l2 = 0.0f64;
    let mut inf = 0.0f64;
    for (&value, &next) in input.iter().zip(applied) {
        let residual = value - next - 1.0;
        anyhow::ensure!(residual.is_finite(), "nonfinite true reduced residual");
        l2 = l2.hypot(residual);
        inf = inf.max(residual.abs());
    }
    let tolerance = config
        .abs_tol
        .max(config.rel_tol * (input.len() as f64).sqrt());
    let inf_tolerance = accept_tol.max(config.abs_tol);
    anyhow::ensure!(l2.is_finite() && tolerance.is_finite() && l2 <= tolerance && inf <= inf_tolerance,
        "pi-policy-eval-reduced NOT ACCEPTED: true |(I-P)u-1|_2={l2:.6e} (tol={tolerance:.6e}), |_inf={inf:.6e} (tol={inf_tolerance:.6e}); output was not published");
    Ok((l2, inf))
}

// Resolve the existing ancestor as well as an existing file, so symlinked
// scratch directories cannot alias a supposedly protected accepted output.
fn resolved_destination(path: &Path) -> Result<PathBuf> {
    match fs::canonicalize(path) {
        Ok(resolved) => return Ok(resolved),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("resolve policy output path {}", path.display()))
        }
    }
    // canonicalize cannot resolve a dangling symlink. Treating its link name
    // as an ordinary new destination would miss writes through the link to a
    // protected, not-yet-created output. Parent recursion performs the same
    // check for dangling directory links.
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            anyhow::bail!("cannot resolve dangling symlink while checking path aliases: {}", path.display());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error)
            .with_context(|| format!("stat policy output path {}", path.display())),
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let parent = absolute
        .parent()
        .context("policy output path has no parent")?;
    let name = absolute
        .file_name()
        .context("policy output path has no filename")?;
    Ok(resolved_destination(parent)?.join(name))
}

fn metadata_if_present(path: &Path) -> Result<Option<fs::Metadata>> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("stat policy value path {}", path.display()))
        }
    }
}

pub(super) fn paths_alias(left: &Path, right: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let same_inode = match (metadata_if_present(left)?, metadata_if_present(right)?) {
        (Some(left), Some(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
        _ => false,
    };
    Ok(same_inode || resolved_destination(left)? == resolved_destination(right)?)
}

pub(super) fn reject_scratch_output_aliases(output: &Path, scratch_dir: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let destination = resolved_destination(output)?;
    let accepted_metadata = metadata_if_present(output)?;
    for layer in 0..10 {
        let scratch = mdp_f32_value_path(scratch_dir, layer);
        let scratch_destination = resolved_destination(&scratch)?;
        let scratch_metadata = metadata_if_present(&scratch)?;
        let same_inode = match (&accepted_metadata, &scratch_metadata) {
            (Some(accepted), Some(scratch)) => {
                accepted.dev() == scratch.dev() && accepted.ino() == scratch.ino()
            }
            _ => false,
        };
        anyhow::ensure!(
            destination != scratch_destination && !same_inode,
            "accepted output must differ from fixed-policy scratch value files (path or inode alias at layer {layer})"
        );
    }
    Ok(())
}

pub(super) fn run_pre_reveal_check(args: &[String]) -> Result<()> {
    anyhow::ensure!(args.len() == 2, "usage: {} pre-reveal-check", args[0]);
    let started = Instant::now();
    let old_index = enumerate_states();
    let old_count = old_index.states.len();
    let q_index = build_q_index_from_old(&old_index);
    drop(old_index);
    anyhow::ensure!(
        q_index.states.len() == CHECK_QSTATE_COUNT,
        "unexpected qstate count {} != {CHECK_QSTATE_COUNT}",
        q_index.states.len()
    );
    let pre = PreRevealIndex::build(&q_index)?;
    let reveal_children: usize = (0..pre.len())
        .map(|zid| pre.reveal_qids(zid as u32).len())
        .sum();
    anyhow::ensure!(
        reveal_children == q_index.states.len(),
        "pre-reveal partition size mismatch"
    );
    println!("pre-reveal-check states={} qstates={} pre_reveal_states={} reveal_children={} elapsed={:.3}s",
        old_count, q_index.states.len(), pre.len(), reveal_children, started.elapsed().as_secs_f64());
    Ok(())
}

pub(super) fn run_pi_policy_eval_reduced(args: &[String]) -> Result<()> {
    anyhow::ensure!((8..=12).contains(&args.len()),
        "usage: {} pi-policy-eval-reduced <graph.bin> <pruned_layer_dir> <policy_dir> <x0.f32|zero> <out_v.f32> <scratch_dir> [gamma=1.0] [max_iters=300] [rel_tol=1e-6] [dense_cache_gib=16]", args[0]);
    let started = Instant::now();
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let x0_arg = &args[5];
    let out_v_path = PathBuf::from(&args[6]);
    let scratch_dir = PathBuf::from(&args[7]);
    let gamma: f64 = if args.len() >= 9 {
        args[8].parse().context("invalid gamma")?
    } else {
        1.0
    };
    anyhow::ensure!(
        gamma == 1.0,
        "pi-policy-eval-reduced supports only gamma=1, got {gamma}"
    );
    let max_iters: usize = if args.len() >= 10 {
        args[9].parse().context("invalid max_iters")?
    } else {
        300
    };
    let rel_tol: f64 = if args.len() >= 11 {
        args[10].parse().context("invalid rel_tol")?
    } else {
        1e-6
    };
    anyhow::ensure!(
        rel_tol.is_finite() && rel_tol >= 0.0,
        "rel_tol must be finite and nonnegative"
    );
    let cache_gib: usize = if args.len() >= 12 {
        args[11].parse().context("invalid dense_cache_gib")?
    } else {
        16
    };
    let cache_cap_bytes = checked_units(cache_gib, 1usize << 30, "dense_cache_gib")?;
    let backend = env_choice("PI_REDUCED_OPERATOR", "sweep", &["sweep", "dag"])?;
    let method = env_choice("PI_REDUCED_METHOD", "gmres", &["gmres", "deflated"])?;
    anyhow::ensure!(method != "deflated" || backend == "dag",
        "PI_REDUCED_METHOD=deflated requires PI_REDUCED_OPERATOR=dag for its exact transpose and certificate");
    let accept_tol: f64 = match env::var("PI_ACCEPT_BELLMAN_INF") {
        Ok(value) => value.parse().context("invalid PI_ACCEPT_BELLMAN_INF")?,
        Err(env::VarError::NotPresent) => 1e-3,
        Err(error) => return Err(error).context("invalid PI_ACCEPT_BELLMAN_INF"),
    };
    anyhow::ensure!(
        accept_tol.is_finite() && accept_tol >= 0.0,
        "PI_ACCEPT_BELLMAN_INF must be finite and nonnegative"
    );
    let config = Config {
        max_iters,
        restart: env_usize("PI_REDUCED_RESTART", 40)?,
        rel_tol,
        abs_tol: 1e-8,
        max_basis_bytes: checked_units(
            env_usize("PI_GMRES_BASIS_MAX_GIB", 24)?,
            1usize << 30,
            "PI_GMRES_BASIS_MAX_GIB",
        )?,
    };
    // Parse caps before doing expensive work. Compilation is in memory and a
    // hard cap error never falls back to an unbounded representation.
    let limits = if backend == "dag" {
        Some(CompileLimits {
            max_states: env_usize("PI_DAG_MAX_STATES", 2_000_000)?,
            max_nodes: env_usize("PI_DAG_MAX_NODES", 5_000_000)?,
            max_edges: env_usize("PI_DAG_MAX_EDGES", 25_000_000)?,
            field_cache_bytes: checked_units(
                env_usize("PI_DAG_FIELD_CACHE_MIB", 64)?,
                1usize << 20,
                "PI_DAG_FIELD_CACHE_MIB",
            )?,
        })
    } else {
        None
    };
    reject_scratch_output_aliases(&out_v_path, &scratch_dir)?;
    fs::create_dir_all(&scratch_dir)?;
    eprintln!("pi-policy-eval-reduced operator={} method={} gamma=1 graph={} membership={} policy={} max_iters={} restart={} rel_tol={:.3e} basis_cap_bytes={}",
        backend, method, graph_path.display(), pruned_dir.display(), policy_dir.display(), max_iters, config.restart, rel_tol, config.max_basis_bytes);
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    drop(old_index);
    let qstate_count = q_index.states.len();
    anyhow::ensure!(
        qstate_count == CHECK_QSTATE_COUNT,
        "unexpected qstate count {qstate_count} != {CHECK_QSTATE_COUNT}"
    );
    let pre = PreRevealIndex::build(&q_index)?;
    let qfwd = build_qforward_graph(&q_index);
    let two_line_field = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("TWO_LINE_HASH not found in graph")?;
    let full_layers: Vec<_> = (0..10)
        .map(|layer| {
            layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer))
        })
        .collect::<Result<Vec<_>>>()?;
    for (layer, input) in full_layers.iter().enumerate() {
        for meta in &input.metas {
            anyhow::ensure!(
                (meta.field_id as usize) < graph.offsets.len(),
                "layer {layer} membership field {} outside graph",
                meta.field_id
            );
        }
    }
    let l0_reset_qids = complete_reset_qids(&graph, &full_layers[0], qstate_count)?;
    let l0_reset_field = full_layers[0].metas[0].field_id;
    let old_values = load_boundary_vector(x0_arg, full_layers[0].total_states)?;
    anyhow::ensure!(
        old_values.iter().all(|value| value.is_finite()),
        "nonfinite reduced policy initial value"
    );
    let old_values: Vec<_> = old_values.into_iter().map(|value| value as f64).collect();
    let full_old_values = file_values_to_full_q(&old_values, &l0_reset_qids, qstate_count)?;
    let mut initial = pre.average_f64(&full_old_values)?;
    for value in &mut initial {
        *value += 1.0;
    }
    finite_f64(&initial, pre.len(), "reduced warm start")?;
    drop(full_old_values);
    drop(old_values);
    let sweep = FixedPolicySweep {
        graph: &graph,
        qfwd: &qfwd,
        full_layers: &full_layers,
        policy_dir: &policy_dir,
        scratch_dir: &scratch_dir,
        two_line_field,
        qstate_count,
        gamma: 1.0,
        cache_cap_bytes,
        sparse_lookup_max_count: env_usize("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT", 65_536)?,
        block_chunk: env_usize("FULL_MDP_BLOCK_CHUNK", 512)?.max(1),
        batch_chunks_env: match env::var("FULL_MDP_BATCH_CHUNKS") {
            Ok(value) => Some(value.parse().context("invalid FULL_MDP_BATCH_CHUNKS")?),
            Err(env::VarError::NotPresent) => None,
            Err(error) => return Err(error).context("invalid FULL_MDP_BATCH_CHUNKS"),
        },
        ram_next_layer: env::var("FULL_MDP_RAM_NEXT_LAYER")
            .ok()
            .map(|value| {
                value != "0"
                    && !value.eq_ignore_ascii_case("false")
                    && !value.eq_ignore_ascii_case("no")
            })
            .unwrap_or(true),
        ram_next_layer_max_bytes: u64::try_from(checked_units(
            env_usize("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB", 96)?,
            1usize << 30,
            "FULL_MDP_RAM_NEXT_LAYER_MAX_GIB",
        )?)?,
        l0_reset_qids: &l0_reset_qids,
        verbose_layers: false,
        prepared_l9: None,
        prepared_l8: None,
        action_source: None,
        operator_bindings: None,
        action_resident_bytes: 0,
    };
    eprintln!("pi-policy-eval-reduced quotient_dimension={} reduced_dimension={} reset_membership=complete reset_field={} reset_hash=0 file_order=preserved", qstate_count, pre.len(), l0_reset_field);
    let compiled = if let Some(limits) = limits {
        eprintln!("pi-policy-eval-reduced compile in_memory=true max_states={} max_nodes={} max_edges={} field_cache_bytes={}",
            limits.max_states, limits.max_nodes, limits.max_edges, limits.field_cache_bytes);
        let roots: Vec<_> = l0_reset_qids
            .iter()
            .map(|&qid| (0, l0_reset_field, qid))
            .collect();
        let compiled = Compiler::new(
            &graph,
            &q_index,
            &qfwd,
            &full_layers,
            &pre,
            Some(&policy_dir),
            CompileMode::Fixed,
            limits,
            two_line_field,
        )?
        .compile_roots(&roots)?;
        let stats = &compiled.stats;
        eprintln!("pi-policy-eval-reduced compiled states={} nodes={} edges={} construction_seconds={:.3} membership_bytes_read={} policy_bytes_read={} field_cache_peak_bytes={} persistent_cache=false",
            stats.states, stats.nodes, stats.edges, stats.construction_seconds, stats.membership_bytes_read, stats.policy_bytes_read, stats.field_cache_peak_bytes);
        Some(compiled)
    } else {
        None
    };
    let operator = compiled
        .as_ref()
        .map(|compiled| {
            ReducedPolicyOperator::new(&compiled.dag, &compiled.roots, &l0_reset_qids, &pre)
        })
        .transpose()?;
    let solve_started = Instant::now();
    let (solution, iterations, converged) = if let Some(operator) = operator.as_ref() {
        if method == "deflated" {
            let compiled = compiled
                .as_ref()
                .expect("DAG operator lacks compiled storage");
            let failure = crate::failure_deflation::direct_failure(
                &pre,
                &l0_reset_qids,
                &compiled.roots,
                &compiled.dag,
            )?;
            let mu = crate::failure_deflation::uniform_mu(pre.len())?;
            let deflation_config = crate::failure_deflation::DeflationConfig {
                gmres: config.clone(),
                residual_tolerance: config.abs_tol.max(config.rel_tol),
                normalization_tolerance: 1e-9,
            };
            let report = crate::failure_deflation::solve_deflated(
                |input| operator.apply_p(input),
                |input| operator.apply_pt(input),
                &failure,
                &mu,
                &deflation_config,
            )?;
            eprintln!("pi-policy-eval-reduced deflation C={:.17e} stationary_iters={} relative_iters={} p_applications={} pt_applications={} original_residual_inf={:.6e}",
                report.constant, report.stationary_solve.iters, report.relative_solve.iters,
                report.stats.p_applications, report.stats.pt_applications, report.residuals.original);
            // The numerical systems supply a candidate, not a properness proof.
            // Scaling its positive solution leaves ample slack for an outward
            // H(w)<=w check; a closed class fails this independent certificate.
            let upper: Vec<_> = report.solution.iter().map(|&value| 2.0 * value).collect();
            let certificate = crate::certified_bounds::certify_supersolution(
                &compiled.dag, &compiled.roots, &l0_reset_qids, &pre, &upper, None)
                .context("deflated fixed policy lacks a verified finite positive supersolution; output was not published")?;
            eprintln!("pi-policy-eval-reduced fixed_policy_certificate=verified margin_lower={:.6e} contraction_gap_lower={:.6e} contraction_factor_upper={:.17e}",
                certificate.min_margin_lower, certificate.contraction_gap_lower, certificate.contraction_factor_upper);
            (
                report.solution,
                report.stationary_solve.iters + report.relative_solve.iters,
                true,
            )
        } else {
            let report = operator.solve(&initial, &config)?;
            (report.solution, report.iters, report.converged)
        }
    } else {
        let rhs = vec![1.0; pre.len()];
        let report = reduced_solver::gmres(
            |input| {
                let mut output = sweep_apply_p(&sweep, &pre, input)?;
                for (value, &input) in output.iter_mut().zip(input) {
                    *value = input - *value;
                }
                Ok(output)
            },
            &rhs,
            &initial,
            &config,
        )?;
        (report.solution, report.iters, report.converged)
    };
    let solve_seconds = solve_started.elapsed().as_secs_f64();
    finite_f64(&solution, pre.len(), "reduced solution")?;
    // A single fresh K evaluation both reconstructs the candidate and supplies
    // the true P=RK residual; do not spend another ten-layer sweep for the same K.
    let values = if let Some(operator) = operator.as_ref() {
        operator.reconstruct(&solution)?
    } else {
        sweep_reconstruct(&sweep, &pre, &solution)?
    };
    let full_values = file_values_to_full_q(&values, &l0_reset_qids, qstate_count)?;
    let applied = pre.average_f64(&full_values)?;
    let (reduced_l2, reduced_inf) =
        check_reduced_residual(&solution, &applied, &config, accept_tol)?;
    drop(full_values);
    let values = finite_f32(&values, qstate_count, "reconstructed policy values")?;
    eprintln!("pi-policy-eval-reduced true_reduced_residual_l2={:.6e} true_reduced_residual_inf={:.6e} final verification affine sweep T_pi(v)", reduced_l2, reduced_inf);
    let verification_started = Instant::now();
    let legacy_applied = sweep.apply(&values, false)?;
    let legacy_seconds = verification_started.elapsed().as_secs_f64();
    let bellman = gmres_accept_and_publish(&out_v_path, &values, &legacy_applied, accept_tol)?;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = values.iter().map(|&value| value as f64).sum();
    let negatives = values.iter().filter(|&&value| value < 0.0).count();
    eprintln!("pi-policy-eval-reduced done v={} status={} krylov_iters={} solve_seconds={:.3} legacy_verification_seconds={:.3} elapsed={:.3}s |T(v)-v|_inf={:.6e} accept_tol={:.3e} v_min={:.8} v_max={:.8} v_avg={:.8} negatives={}",
        out_v_path.display(), if converged { "CONVERGED" } else { "MAX_ITERS" }, iterations,
        solve_seconds, legacy_seconds, started.elapsed().as_secs_f64(), bellman, accept_tol,
        min, max, sum / values.len() as f64, negatives);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_alias_guard_rejects_hard_links_and_symlinks_but_accepts_distinct_files() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            env::temp_dir().join(format!("zxcl-reduced-alias-{}-{stamp}", std::process::id()));
        let scratch_dir = root.join("scratch");
        fs::create_dir_all(&scratch_dir).unwrap();
        let accepted = root.join("accepted.f32.bin");
        let accepted_bytes = 123.0f32.to_le_bytes();
        fs::write(&accepted, accepted_bytes).unwrap();

        let hard_link = mdp_f32_value_path(&scratch_dir, 9);
        fs::hard_link(&accepted, &hard_link).unwrap();
        let error = reject_scratch_output_aliases(&accepted, &scratch_dir)
            .unwrap_err()
            .to_string();
        assert!(error.contains("path or inode alias at layer 9"));
        assert_eq!(fs::read(&accepted).unwrap(), accepted_bytes);
        fs::remove_file(&hard_link).unwrap();

        let symbolic_link = mdp_f32_value_path(&scratch_dir, 4);
        std::os::unix::fs::symlink(&accepted, &symbolic_link).unwrap();
        assert!(reject_scratch_output_aliases(&accepted, &scratch_dir).is_err());
        assert_eq!(fs::read(&accepted).unwrap(), accepted_bytes);
        fs::remove_file(&symbolic_link).unwrap();

        fs::write(mdp_f32_value_path(&scratch_dir, 1), accepted_bytes).unwrap();
        assert!(reject_scratch_output_aliases(&accepted, &scratch_dir).is_ok());
        assert!(
            reject_scratch_output_aliases(&root.join("new-output"), &root.join("new-scratch"))
                .is_ok()
        );
        // A non-directory component is an I/O error, not an absent file.
        assert!(reject_scratch_output_aliases(&accepted, &accepted.join("scratch")).is_err());
        assert_eq!(fs::read(&accepted).unwrap(), accepted_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reset_field_uses_empty_geometry_instead_of_a_terminal_or_literal_id() {
        let graph = FieldGraph {
            hashes: vec![0xff_ffff_ffff, 0, 0x0f_ffff],
            offsets: vec![[0; PIECE_COUNT + 1]; 3],
            targets: Vec::new(),
        };
        assert!(validate_reset_field(&graph, 1).is_ok());
        assert!(validate_reset_field(&graph, 0).is_err());
        assert!(validate_reset_field(&graph, 2).is_err());
        assert!(validate_reset_field(&graph, 3).is_err());
        assert!(validate_reset_field(&graph, EMPTY_FIELD_ID).is_err());
    }

    #[test]
    fn full_reset_membership_is_unique_complete_and_keeps_file_order() {
        assert!(validate_complete_qids(&[2, 0, 3, 1], 4).is_ok());
        assert!(validate_complete_qids(&[0, 1, 2], 4).is_err());
        assert!(validate_complete_qids(&[0, 1, 1, 3], 4).is_err());
        assert!(validate_complete_qids(&[0, 1, 2, 4], 4).is_err());
        assert_eq!(
            file_values_to_full_q(&[-2.0, 4.0, -3.0], &[2, 0, 1], 4).unwrap(),
            [4.0, -3.0, -2.0, 0.0]
        );
        assert!(file_values_to_full_q(&[1.0, 2.0], &[1, 1], 4).is_err());
        assert!(file_values_to_full_q(&[f64::NAN], &[0], 4).is_err());
    }

    #[test]
    fn true_reduced_gate_rejects_unconverged_and_nonfinite_candidates() {
        let config = Config {
            rel_tol: 1e-6,
            abs_tol: 1e-8,
            ..Config::default()
        };
        assert_eq!(
            check_reduced_residual(&[2.0, 3.0], &[1.0, 2.0], &config, 1e-3).unwrap(),
            (0.0, 0.0)
        );
        assert!(check_reduced_residual(&[2.0], &[0.0], &config, 1e-3).is_err());
        assert!(check_reduced_residual(&[f64::NAN], &[0.0], &config, 1e-3).is_err());
        assert!(check_reduced_residual(&[1.0], &[f64::INFINITY], &config, 1e-3).is_err());
        assert!(check_reduced_residual(&[], &[], &config, 1e-3).is_err());
    }

    #[test]
    fn conversion_preserves_signed_values_and_checks_overflow() {
        assert_eq!(
            finite_f32(&[-3.0, 0.0, 2.0], 3, "test").unwrap(),
            [-3.0, 0.0, 2.0]
        );
        assert!(finite_f32(&[f64::MAX], 1, "test").is_err());
        assert!(checked_units(usize::MAX, 1 << 30, "test").is_err());
        assert_eq!(checked_units(64, 1 << 20, "test").unwrap(), 67_108_864);
    }

    #[test]
    fn invalid_cli_choices_and_gamma_fail_before_loading_graph() {
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let args = vec![
            "zxcl",
            "pi-policy-eval-reduced",
            "missing-graph",
            "missing-pruned",
            "missing-policy",
            "zero",
            "out",
            "scratch",
            "0.99",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        assert!(run_pi_policy_eval_reduced(&args)
            .unwrap_err()
            .to_string()
            .contains("only gamma=1"));
        let mut args = args;
        args[8] = "1".to_owned();
        let old_backend = env::var_os("PI_REDUCED_OPERATOR");
        let old_method = env::var_os("PI_REDUCED_METHOD");
        env::set_var("PI_REDUCED_OPERATOR", "sweep");
        env::set_var("PI_REDUCED_METHOD", "deflated");
        let error = run_pi_policy_eval_reduced(&args).unwrap_err().to_string();
        match old_backend {
            Some(value) => env::set_var("PI_REDUCED_OPERATOR", value),
            None => env::remove_var("PI_REDUCED_OPERATOR"),
        }
        match old_method {
            Some(value) => env::set_var("PI_REDUCED_METHOD", value),
            None => env::remove_var("PI_REDUCED_METHOD"),
        }
        assert!(error.contains("requires PI_REDUCED_OPERATOR=dag"));
    }
}
