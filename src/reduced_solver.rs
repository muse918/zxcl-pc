//! Matrix-free policy evaluation at the boundary before a random reveal.
//!
//! Krylov vectors are signed f64 vectors. Convergence is always decided from
//! a fresh application of the original operator, rather than its small
//! Hessenberg approximation. This module never publishes a policy value file.

use anyhow::{anyhow, bail, ensure, Context, Result};

use crate::expression_dag::{ExprDag, Node};
use crate::pre_reveal::PreRevealIndex;

#[derive(Clone, Debug)]
pub struct Config {
    /// Maximum total Arnoldi steps across all restarts.
    pub max_iters: usize,
    pub restart: usize,
    pub rel_tol: f64,
    pub abs_tol: f64,
    /// Cap for the Arnoldi basis, Hessenberg matrix, and small work arrays.
    /// Operator allocations and solution/residual vectors are additional.
    pub max_basis_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iters: 300,
            restart: 40,
            rel_tol: 1e-10,
            abs_tol: 1e-12,
            max_basis_bytes: 1usize << 30,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub solution: Vec<f64>,
    /// The L2 norm of the freshly evaluated residual b - A x.
    pub residual_norm: f64,
    /// residual_norm / ||b||, or residual_norm when b is zero.
    pub relative_residual: f64,
    pub iters: usize,
    pub converged: bool,
}

/// Number of bytes in the stored Krylov basis and small work arrays.
/// Callers should use min(restart, max_iters, dimension) as the cycle length.
pub fn basis_bytes(dimension: usize, cycle: usize) -> Result<usize> {
    let rows = cycle.checked_add(1).context("GMRES basis size overflow")?;
    let vectors = dimension
        .checked_mul(rows)
        .context("GMRES basis size overflow")?;
    let hessenberg = cycle
        .checked_mul(rows)
        .context("GMRES Hessenberg size overflow")?;
    // Two Givens arrays, transformed RHS, and back-substitution coefficients.
    let small = cycle
        .checked_mul(4)
        .and_then(|n| n.checked_add(1))
        .context("GMRES work array size overflow")?;
    let payload = vectors
        .checked_add(hessenberg)
        .and_then(|n| n.checked_add(small))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .context("GMRES workspace byte size overflow")?;
    let descriptors = rows
        .checked_mul(std::mem::size_of::<Vec<f64>>())
        .context("GMRES basis descriptor size overflow")?;
    payload
        .checked_add(descriptors)
        .context("GMRES workspace byte size overflow")
}

/// Restarted, unpreconditioned GMRES with two-pass modified Gram-Schmidt.
/// `apply` represents A, not I-A. Iteration counts exclude residual evaluations.
/// A nonconverged report is returned on iteration exhaustion or Arnoldi
/// breakdown; malformed inputs, nonfinite arithmetic, and budget violations
/// return errors. An already converged initial vector needs no Krylov budget.
pub fn gmres<F>(mut apply: F, rhs: &[f64], initial: &[f64], config: &Config) -> Result<Report>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
{
    ensure!(
        initial.len() == rhs.len(),
        "GMRES initial length {}, expected {}",
        initial.len(),
        rhs.len()
    );
    check_finite(rhs, "GMRES right-hand side")?;
    check_finite(initial, "GMRES initial vector")?;
    ensure!(
        config.rel_tol.is_finite() && config.rel_tol >= 0.0,
        "GMRES relative tolerance must be finite and nonnegative"
    );
    ensure!(
        config.abs_tol.is_finite() && config.abs_tol >= 0.0,
        "GMRES absolute tolerance must be finite and nonnegative"
    );
    let rhs_norm = norm(rhs)?;
    let tolerance = config.abs_tol.max(config.rel_tol * rhs_norm);
    ensure!(
        tolerance.is_finite(),
        "GMRES convergence threshold is nonfinite"
    );
    let mut x = initial.to_vec();
    if rhs.is_empty() {
        return Ok(report(x, 0.0, rhs_norm, 0, true));
    }
    let (mut residual, mut residual_norm) = true_residual(&mut apply, rhs, &x)?;
    if residual_norm <= tolerance || config.max_iters == 0 {
        return Ok(report(
            x,
            residual_norm,
            rhs_norm,
            0,
            residual_norm <= tolerance,
        ));
    }
    ensure!(
        config.restart > 0,
        "GMRES restart must be positive when iterations are needed"
    );
    let cycle = config.restart.min(config.max_iters).min(rhs.len());
    let required = basis_bytes(rhs.len(), cycle)?;
    ensure!(
        required <= config.max_basis_bytes,
        "GMRES Krylov workspace needs {required} bytes, exceeding basis cap {}",
        config.max_basis_bytes
    );

    let stride = cycle + 1;
    let mut h = zeros(cycle * stride, "Hessenberg matrix")?;
    let mut cosines = zeros(cycle, "Givens cosines")?;
    let mut sines = zeros(cycle, "Givens sines")?;
    let mut g = zeros(stride, "transformed right-hand side")?;
    let mut coefficients = zeros(cycle, "back-substitution coefficients")?;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    basis
        .try_reserve_exact(stride)
        .context("allocating GMRES basis descriptors")?;
    let mut iters = 0;

    loop {
        h.fill(0.0);
        g.fill(0.0);
        basis.clear();
        g[0] = residual_norm;
        let first: Vec<f64> = residual
            .iter()
            .map(|&value| value / residual_norm)
            .collect();
        check_finite(&first, "GMRES normalized residual")?;
        basis.push(first);
        let steps = cycle.min(config.max_iters - iters);
        let mut restarted = false;

        for column in 0..steps {
            let mut work = checked_apply(&mut apply, &basis[column], rhs.len())?;
            iters += 1;
            let original_norm = norm(&work)?;
            let offset = column * stride;
            // Reorthogonalization matters when I-P has a very slow mode.
            for _ in 0..2 {
                for row in 0..=column {
                    let projection = dot(&basis[row], &work)?;
                    h[offset + row] += projection;
                    ensure!(
                        h[offset + row].is_finite(),
                        "nonfinite GMRES Hessenberg entry"
                    );
                    axpy(&mut work, -projection, &basis[row])?;
                }
            }
            let next_norm = norm(&work)?;
            h[offset + column + 1] = next_norm;
            // Relative to ||A v||: a small but nonsingular A is not a breakdown.
            let breakdown = next_norm == 0.0 || next_norm <= f64::EPSILON * original_norm;
            if !breakdown {
                for value in &mut work {
                    *value /= next_norm;
                }
                check_finite(&work, "GMRES normalized Arnoldi vector")?;
                basis.push(work);
            }

            for row in 0..column {
                let top = h[offset + row];
                let bottom = h[offset + row + 1];
                h[offset + row] = cosines[row] * top + sines[row] * bottom;
                h[offset + row + 1] = -sines[row] * top + cosines[row] * bottom;
            }
            let diagonal = h[offset + column];
            let subdiagonal = h[offset + column + 1];
            let radius = diagonal.hypot(subdiagonal);
            ensure!(radius.is_finite(), "nonfinite GMRES Givens rotation");
            if radius == 0.0 {
                cosines[column] = 1.0;
                sines[column] = 0.0;
            } else {
                cosines[column] = diagonal / radius;
                sines[column] = subdiagonal / radius;
            }
            h[offset + column] = radius;
            h[offset + column + 1] = 0.0;
            g[column + 1] = -sines[column] * g[column];
            g[column] *= cosines[column];
            check_finite(&g[..column + 2], "GMRES transformed residual")?;

            let end_cycle = column + 1 == steps;
            if breakdown || end_cycle || g[column + 1].abs() <= tolerance {
                // On singular breakdown retain the last nonsingular projected
                // problem instead of dividing a zero pivot or claiming success.
                let mut width = column + 1;
                while width > 0 && h[(width - 1) * stride + width - 1] == 0.0 {
                    width -= 1;
                }
                coefficients[..width].copy_from_slice(&g[..width]);
                for row in (0..width).rev() {
                    let mut value = coefficients[row];
                    for col in row + 1..width {
                        value -= h[col * stride + row] * coefficients[col];
                    }
                    let pivot = h[row * stride + row];
                    ensure!(pivot != 0.0, "zero pivot inside GMRES projected problem");
                    coefficients[row] = value / pivot;
                }
                check_finite(&coefficients[..width], "GMRES correction coefficients")?;
                let mut candidate = x.clone();
                for row in 0..width {
                    axpy(&mut candidate, coefficients[row], &basis[row])?;
                }
                let (next_residual, true_norm) = true_residual(&mut apply, rhs, &candidate)?;
                if true_norm <= tolerance {
                    return Ok(report(candidate, true_norm, rhs_norm, iters, true));
                }
                if iters == config.max_iters {
                    return Ok(report(candidate, true_norm, rhs_norm, iters, false));
                }
                // A finite precision Arnoldi breakdown can leave a small true
                // residual when I-P is ill-conditioned. Restart from that
                // residual if the cycle made progress; this is also iterative
                // refinement of the projected solution. Exact stagnation must
                // still return a nonconverged report for singular operators.
                if breakdown && true_norm >= residual_norm {
                    return Ok(report(candidate, true_norm, rhs_norm, iters, false));
                }
                if breakdown || end_cycle {
                    x = candidate;
                    residual = next_residual;
                    residual_norm = true_norm;
                    restarted = true;
                    break;
                }
            }
        }
        ensure!(restarted, "GMRES cycle ended without a residual update");
    }
}

fn report(
    solution: Vec<f64>,
    residual_norm: f64,
    rhs_norm: f64,
    iters: usize,
    converged: bool,
) -> Report {
    Report {
        solution,
        residual_norm,
        relative_residual: if rhs_norm > 0.0 {
            residual_norm / rhs_norm
        } else {
            residual_norm
        },
        iters,
        converged,
    }
}

fn zeros(length: usize, label: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .with_context(|| format!("allocating GMRES {label}"))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn check_finite(values: &[f64], label: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        bail!("nonfinite {label} at index {index}");
    }
    Ok(())
}

fn norm(values: &[f64]) -> Result<f64> {
    let value = values.iter().fold(0.0f64, |norm, &value| norm.hypot(value));
    ensure!(value.is_finite(), "nonfinite GMRES vector norm");
    Ok(value)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64> {
    // Compensate summation error while retaining signed products.
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let product = left * right;
        let corrected = product - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }
    ensure!(sum.is_finite(), "nonfinite GMRES dot product");
    Ok(sum)
}

fn axpy(target: &mut [f64], scale: f64, vector: &[f64]) -> Result<()> {
    for (target, &value) in target.iter_mut().zip(vector) {
        *target += scale * value;
    }
    check_finite(target, "GMRES vector update")
}

fn checked_apply<F>(apply: &mut F, input: &[f64], dimension: usize) -> Result<Vec<f64>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
{
    let output = apply(input)?;
    ensure!(
        output.len() == dimension,
        "GMRES operator output length {}, expected {dimension}",
        output.len()
    );
    check_finite(&output, "GMRES operator output")?;
    Ok(output)
}

fn true_residual<F>(apply: &mut F, rhs: &[f64], x: &[f64]) -> Result<(Vec<f64>, f64)>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
{
    let mut residual = checked_apply(apply, x, rhs.len())?;
    for (value, &rhs) in residual.iter_mut().zip(rhs) {
        *value = rhs - *value;
    }
    check_finite(&residual, "GMRES true residual")?;
    let residual_norm = norm(&residual)?;
    Ok((residual, residual_norm))
}

/// K maps pre-reveal continuation values to quotient policy roots. R averages
/// the full quotient boundary, so the reduced transition is P = R K. Roots and
/// qids have the same file order; quotient states absent from qids carry zero.
pub struct ReducedPolicyOperator<'a> {
    pub dag: &'a ExprDag,
    pub roots: &'a [u32],
    pub qids: &'a [u32],
    pub pre: &'a PreRevealIndex,
}

impl<'a> ReducedPolicyOperator<'a> {
    pub fn new(
        dag: &'a ExprDag,
        roots: &'a [u32],
        qids: &'a [u32],
        pre: &'a PreRevealIndex,
    ) -> Result<Self> {
        ensure!(
            dag.terminal_count() == pre.len(),
            "reduced DAG terminal count {}, expected {}",
            dag.terminal_count(),
            pre.len()
        );
        ensure!(
            roots.len() == qids.len(),
            "reduced roots/qids length mismatch"
        );
        let mut seen = vec![false; pre.qid_to_zid.len()];
        for (index, &qid) in qids.iter().enumerate() {
            let present = seen.get_mut(qid as usize).ok_or_else(|| {
                anyhow!(
                    "reduced qid {qid} at index {index} is outside full quotient capacity {}",
                    pre.qid_to_zid.len()
                )
            })?;
            ensure!(!*present, "duplicate reduced qid {qid}");
            *present = true;
            ensure!(
                (roots[index] as usize) < dag.node_count(),
                "reduced root {} does not exist",
                roots[index]
            );
        }
        // The fixed-policy compiler only emits linear nodes. Avoid allocating
        // a full transpose workspace just to validate that common case; if the
        // arena also holds maxima, check root reachability even at zero weight.
        if dag
            .nodes()
            .iter()
            .any(|node| matches!(node, Node::Maximum(_)))
        {
            dag.linear_transpose(roots, &vec![0.0; roots.len()])?;
        }
        Ok(Self {
            dag,
            roots,
            qids,
            pre,
        })
    }

    pub fn dimension(&self) -> usize {
        self.pre.len()
    }

    pub fn full_capacity(&self) -> usize {
        self.pre.qid_to_zid.len()
    }

    /// Apply K and retain the roots' file order.
    pub fn reconstruct(&self, input: &[f64]) -> Result<Vec<f64>> {
        let values = self.dag.eval_f64(input)?;
        self.roots
            .iter()
            .map(|&root| {
                values
                    .get(root as usize)
                    .copied()
                    .ok_or_else(|| anyhow!("reduced root {root} does not exist"))
            })
            .collect()
    }

    pub fn apply_p(&self, input: &[f64]) -> Result<Vec<f64>> {
        let roots = self.reconstruct(input)?;
        let mut full = vec![0.0; self.full_capacity()];
        for (&qid, value) in self.qids.iter().zip(roots) {
            let slot = full
                .get_mut(qid as usize)
                .ok_or_else(|| anyhow!("reduced qid {qid} is outside full quotient capacity"))?;
            *slot = value;
        }
        self.pre.average_f64(&full)
    }

    /// Apply K^T R^T. R^T divides each group's value by its reveal count;
    /// missing quotient roots contribute zero without changing that divisor.
    pub fn apply_pt(&self, input: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            input.len() == self.dimension(),
            "reduced transpose input length {}, expected {}",
            input.len(),
            self.dimension()
        );
        check_finite(input, "reduced transpose input")?;
        let mut weights = Vec::with_capacity(self.qids.len());
        for &qid in self.qids {
            let &zid =
                self.pre.qid_to_zid.get(qid as usize).ok_or_else(|| {
                    anyhow!("reduced qid {qid} is outside full quotient capacity")
                })?;
            let children = self.pre.reveal_qids(zid);
            ensure!(!children.is_empty(), "empty pre-reveal group {zid}");
            weights.push(input[zid as usize] / children.len() as f64);
        }
        self.dag.linear_transpose(self.roots, &weights)
    }

    /// Transform an accepted full quotient value vector to u = 1 + R v_old.
    pub fn warm_start(&self, full_q_values: &[f64]) -> Result<Vec<f64>> {
        let mut values = self.pre.average_f64(full_q_values)?;
        for value in &mut values {
            *value += 1.0;
        }
        check_finite(&values, "reduced warm start")?;
        Ok(values)
    }

    /// Solve (I-P)u = 1. Reconstruction and the original sweep acceptance gate
    /// remain the caller's responsibility.
    pub fn solve(&self, initial: &[f64], config: &Config) -> Result<Report> {
        let rhs = vec![1.0; self.dimension()];
        gmres(
            |input| {
                let mut output = self.apply_p(input)?;
                for (value, &input) in output.iter_mut().zip(input) {
                    *value = input - *value;
                }
                check_finite(&output, "reduced I-P output")?;
                Ok(output)
            },
            &rhs,
            initial,
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            max_iters: 100,
            restart: 4,
            rel_tol: 1e-12,
            abs_tol: 1e-13,
            max_basis_bytes: 1 << 20,
        }
    }

    fn matrix_apply(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>> {
        Ok(matrix
            .iter()
            .map(|row| row.iter().zip(vector).map(|(&a, &b)| a * b).sum())
            .collect())
    }

    #[test]
    fn nonsymmetric_substochastic_policy_matches_analytic_solution() {
        // P = [[1/2, 1/4], [1/10, 1/5]], so (I-P)^-1 1 = [2.8, 1.6].
        let a = vec![vec![0.5, -0.25], vec![-0.1, 0.8]];
        let solved = gmres(|x| matrix_apply(&a, x), &[1.0, 1.0], &[0.0, 0.0], &config()).unwrap();
        assert!(solved.converged, "{solved:?}");
        assert!((solved.solution[0] - 2.8).abs() < 1e-11);
        assert!((solved.solution[1] - 1.6).abs() < 1e-11);
        assert!(solved.residual_norm < 1e-12);
        assert!(solved.iters <= 2);
    }

    #[test]
    fn near_unit_mode_and_small_nonsingular_operator_do_not_break_down_early() {
        let diagonal = [1e-9, 0.4, 0.75];
        let expected = [1e9, -3.0, 2.0];
        let rhs: Vec<_> = diagonal.iter().zip(expected).map(|(&a, x)| a * x).collect();
        let solved = gmres(
            |x| Ok(diagonal.iter().zip(x).map(|(&a, &x)| a * x).collect()),
            &rhs,
            &[0.0; 3],
            &config(),
        )
        .unwrap();
        assert!(solved.converged, "{solved:?}");
        for (&actual, &expected) in solved.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-8);
        }
        let solved = gmres(
            |x| Ok(x.iter().map(|&x| 1e-14 * x).collect()),
            &[1.0],
            &[0.0],
            &config(),
        )
        .unwrap();
        assert!(solved.converged, "{solved:?}");
        assert!((solved.solution[0] / 1e14 - 1.0).abs() < 1e-12);

        // Exercise the actual cancellation in I-P and a nonzero warm vector,
        // rather than only multiplying directly by the small eigenvalue.
        let probability = 1.0 - 1e-9;
        let solved = gmres(
            |x| Ok(vec![x[0] - probability * x[0]]),
            &[1.0],
            &[7.0],
            &config(),
        )
        .unwrap();
        assert!(solved.converged, "{solved:?}");
        assert_eq!(
            solved.residual_norm,
            (1.0 - (solved.solution[0] - probability * solved.solution[0])).abs()
        );
    }

    #[test]
    fn restart_reaches_signed_solution_and_callback_sees_signed_vectors() {
        let a = vec![
            vec![3.0, 1.0, 0.0],
            vec![0.0, 2.0, 0.5],
            vec![0.25, 0.0, 1.0],
        ];
        let expected = [-2.0, 3.0, -4.0];
        let rhs = matrix_apply(&a, &expected).unwrap();
        let mut cfg = config();
        cfg.restart = 2;
        let mut saw_negative = false;
        let solved = gmres(
            |x| {
                saw_negative |= x.iter().any(|&x| x < 0.0);
                matrix_apply(&a, x)
            },
            &rhs,
            &[0.0; 3],
            &cfg,
        )
        .unwrap();
        assert!(solved.converged, "{solved:?}");
        assert!(solved.iters > cfg.restart);
        assert!(saw_negative);
        for (&actual, &expected) in solved.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn breakdown_and_iteration_limit_never_claim_convergence() {
        let failed = gmres(|x| Ok(vec![0.0; x.len()]), &[1.0], &[0.0], &config()).unwrap();
        assert!(!failed.converged);
        assert_eq!(failed.iters, 1);
        assert_eq!(failed.solution, [0.0]);
        assert_eq!(failed.residual_norm, 1.0);
        let inconsistent = gmres(|x| Ok(vec![x[0], 0.0]), &[1.0; 2], &[0.0; 2], &config()).unwrap();
        assert!(!inconsistent.converged);
        assert!(inconsistent.residual_norm.is_finite());
        assert!(inconsistent.residual_norm >= 1.0);
        let mut cfg = config();
        cfg.max_iters = 1;
        let failed = gmres(|x| Ok(vec![x[0], 2.0 * x[1]]), &[1.0; 2], &[0.0; 2], &cfg).unwrap();
        assert!(!failed.converged);
        assert_eq!(failed.iters, 1);
        let true_norm = (1.0 - failed.solution[0]).hypot(1.0 - 2.0 * failed.solution[1]);
        assert_eq!(failed.residual_norm, true_norm);
    }

    #[test]
    fn warm_convergence_and_zero_iterations_need_no_basis_budget() {
        let mut cfg = config();
        cfg.max_basis_bytes = 0;
        let solved = gmres(|x| Ok(x.to_vec()), &[2.0, -3.0], &[2.0, -3.0], &cfg).unwrap();
        assert!(solved.converged);
        assert_eq!(solved.iters, 0);
        assert!(gmres(|x| Ok(x.to_vec()), &[1.0], &[0.0], &cfg).is_err());
        cfg.max_iters = 0;
        let failed = gmres(|x| Ok(x.to_vec()), &[1.0], &[0.0], &cfg).unwrap();
        assert!(!failed.converged);
        assert_eq!(failed.iters, 0);
        assert_eq!(failed.residual_norm, 1.0);
        cfg.restart = 0;
        assert!(
            gmres(|x| Ok(x.to_vec()), &[0.0], &[0.0], &cfg)
                .unwrap()
                .converged
        );
    }

    #[test]
    fn basis_budget_includes_small_arrays_and_rejects_overflow() {
        let mut cfg = config();
        cfg.restart = 2;
        cfg.max_basis_bytes = basis_bytes(2, 2).unwrap();
        assert!(
            gmres(|x| Ok(x.to_vec()), &[1.0; 2], &[0.0; 2], &cfg)
                .unwrap()
                .converged
        );
        cfg.max_basis_bytes -= 1;
        assert!(gmres(|x| Ok(x.to_vec()), &[1.0; 2], &[0.0; 2], &cfg).is_err());
        assert!(basis_bytes(usize::MAX, 1).is_err());
    }

    #[test]
    fn nonfinite_inputs_outputs_and_invalid_dimensions_are_errors() {
        let cfg = config();
        assert!(gmres(|x| Ok(x.to_vec()), &[f64::NAN], &[0.0], &cfg).is_err());
        assert!(gmres(|x| Ok(x.to_vec()), &[1.0], &[f64::INFINITY], &cfg).is_err());
        assert!(gmres(|_| Ok(vec![f64::NEG_INFINITY]), &[1.0], &[0.0], &cfg).is_err());
        assert!(gmres(|_| Ok(vec![]), &[1.0], &[0.0], &cfg).is_err());
        assert!(gmres(|x| Ok(x.to_vec()), &[1.0], &[], &cfg).is_err());
        let mut cfg = cfg;
        cfg.rel_tol = f64::NAN;
        assert!(gmres(|x| Ok(x.to_vec()), &[1.0], &[0.0], &cfg).is_err());
    }

    #[test]
    fn empty_system_and_zero_rhs_have_defined_relative_residual() {
        let solved = gmres(
            |_| panic!("empty system does not apply operator"),
            &[],
            &[],
            &config(),
        )
        .unwrap();
        assert!(solved.converged);
        assert_eq!(solved.relative_residual, 0.0);
        let solved = gmres(|x| Ok(x.to_vec()), &[0.0], &[1.0], &config()).unwrap();
        assert!(solved.converged);
        assert_eq!(solved.relative_residual, solved.residual_norm);
    }

    #[test]
    fn reduced_operator_uses_full_reveal_divisors_file_order_and_true_transpose() {
        // Use the real complete partition: absent file qids must retain zero
        // mass while every denominator still uses the entire reveal group.
        let old = crate::state::enumerate_states();
        let quotient = crate::qstate::enumerate_qstates(&old);
        drop(old);
        let pre = PreRevealIndex::build(&quotient).unwrap();
        assert_eq!(pre.len(), 205_170);
        assert_eq!(pre.qid_to_zid.len(), 817_740);
        let z0 = (0..pre.len())
            .find(|&zid| pre.reveal_qids(zid as u32).len() == 2)
            .unwrap();
        let z1 = (0..pre.len())
            .find(|&zid| pre.reveal_qids(zid as u32).len() == 3)
            .unwrap();
        let mut dag = ExprDag::new(pre.len(), 32, 32).unwrap();
        let t0 = dag.terminal(z0).unwrap();
        let t1 = dag.terminal(z1).unwrap();
        let half = dag.average(&[t0], 2).unwrap();
        let quarter = dag.average(&[t0, t1], 4).unwrap();
        let roots = [half, t1, quarter];
        let qids = [
            pre.reveal_qids(z1 as u32)[0],
            pre.reveal_qids(z0 as u32)[1],
            pre.reveal_qids(z0 as u32)[0],
        ];
        let op = ReducedPolicyOperator::new(&dag, &roots, &qids, &pre).unwrap();
        let mut input = vec![0.0; pre.len()];
        input[z0] = -2.0;
        input[z1] = 3.0;
        assert_eq!(op.reconstruct(&input).unwrap(), [-1.0, 3.0, 0.25]);
        let product = op.apply_p(&input).unwrap();
        assert_eq!(product[z0], 1.625);
        assert_eq!(product[z1], -1.0 / 3.0);
        assert!(product
            .iter()
            .enumerate()
            .all(|(zid, &value)| zid == z0 || zid == z1 || value == 0.0));
        let mut left = vec![0.0; pre.len()];
        left[z0] = 4.0;
        left[z1] = -6.0;
        let transposed = op.apply_pt(&left).unwrap();
        assert_eq!(transposed[z0], -0.5);
        assert_eq!(transposed[z1], 2.5);
        assert_eq!(
            dot(&left, &product).unwrap(),
            dot(&transposed, &input).unwrap()
        );
        assert_eq!(
            op.warm_start(&vec![2.0; op.full_capacity()]).unwrap(),
            vec![3.0; pre.len()]
        );

        let mut cfg = config();
        cfg.max_basis_bytes = 64 << 20;
        let solved = op.solve(&vec![1.0; pre.len()], &cfg).unwrap();
        assert!(solved.converged, "{solved:?}");
        assert!((solved.solution[z0] - 78.0 / 37.0).abs() < 1e-10);
        assert!((solved.solution[z1] - 50.0 / 37.0).abs() < 1e-10);
        let roots_v = op.reconstruct(&solved.solution).unwrap();
        let mut full_v = vec![0.0; op.full_capacity()];
        for (&qid, &value) in qids.iter().zip(&roots_v) {
            full_v[qid as usize] = value;
        }
        let lifted = op.warm_start(&full_v).unwrap();
        assert!(lifted
            .iter()
            .zip(&solved.solution)
            .all(|(&a, &b)| (a - b).abs() < 1e-10));

        assert!(
            ReducedPolicyOperator::new(&dag, &roots, &[qids[0], qids[0], qids[2]], &pre).is_err()
        );
        assert!(
            ReducedPolicyOperator::new(&dag, &roots, &[u32::MAX, qids[1], qids[2]], &pre).is_err()
        );
        assert!(ReducedPolicyOperator::new(&dag, &[u32::MAX], &[qids[0]], &pre).is_err());
        assert!(ReducedPolicyOperator::new(&dag, &roots[..2], &qids, &pre).is_err());
        let maximum = dag.maximum(&[t0, t1]).unwrap();
        assert!(ReducedPolicyOperator::new(&dag, &[maximum], &qids[..1], &pre).is_err());
    }
}
