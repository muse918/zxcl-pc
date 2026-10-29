//! Regenerative failure deflation for a fixed, substochastic reduced policy.
//!
//! This produces a numerically checked *candidate*. A small linear residual is
//! not a properness certificate or a bound on the value error. Publication must
//! still pass the independent policy/properness and original Bellman checks.

use anyhow::{bail, ensure, Context, Result};

use crate::expression_dag::{ExprDag, Node, NodeId};
use crate::pre_reveal::PreRevealIndex;
use crate::reduced_solver::{gmres, Config, Report};

#[derive(Clone, Debug)]
pub(crate) struct DeflationConfig {
    pub(crate) gmres: Config,
    /// Absolute infinity-norm tolerance, including for `(I-P)u - 1`.
    pub(crate) residual_tolerance: f64,
    /// Absolute tolerance for `sum(nu)-1` and `mu^T h`.
    pub(crate) normalization_tolerance: f64,
}

impl Default for DeflationConfig {
    fn default() -> Self {
        Self {
            gmres: Config::default(),
            residual_tolerance: 1e-8,
            normalization_tolerance: 1e-9,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DeflationResiduals {
    pub(crate) row_mass: f64,
    pub(crate) stationary_system: f64,
    pub(crate) stationarity: f64,
    pub(crate) stationary_normalization: f64,
    pub(crate) relative_system: f64,
    pub(crate) relative_equation: f64,
    pub(crate) relative_normalization: f64,
    pub(crate) original: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct DeflationStats {
    pub(crate) p_applications: usize,
    pub(crate) pt_applications: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct DeflationReport {
    pub(crate) solution: Vec<f64>,
    pub(crate) constant: f64,
    pub(crate) stationary: Vec<f64>,
    pub(crate) relative: Vec<f64>,
    /// Raw GMRES reports, before the final normalization of their solutions.
    pub(crate) stationary_solve: Report,
    pub(crate) relative_solve: Report,
    pub(crate) residuals: DeflationResiduals,
    pub(crate) stats: DeflationStats,
}

pub(crate) fn uniform_mu(n: usize) -> Result<Vec<f64>> {
    ensure!(n > 0, "failure deflation requires at least one state");
    Ok(vec![1.0 / n as f64; n])
}

/// Compute failure directly, without subtracting a nearly unit survival value.
/// Only reachable nodes are inspected; a reachable maximum is not a fixed
/// linear policy and is rejected even when its numeric weight would be zero.
pub(crate) fn failure_at_roots(dag: &ExprDag, roots: &[NodeId]) -> Result<Vec<f64>> {
    let nodes = dag.nodes();
    let mut reachable = vec![false; nodes.len()];
    let mut stack = Vec::with_capacity(roots.len());
    for &root in roots {
        ensure!(
            (root as usize) < nodes.len(),
            "failure root node {root} does not exist"
        );
        stack.push(root);
    }
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id as usize], true) {
            continue;
        }
        match &nodes[id as usize] {
            Node::Average { children, .. } => stack.extend_from_slice(children),
            Node::Maximum(_) => bail!("failure propagation reached Maximum node {id}"),
            Node::Zero | Node::Terminal(_) => {}
        }
    }

    // ExprDag guarantees topological IDs. Keep the stored child addition order.
    let mut failure = vec![0.0; nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        if !reachable[id] {
            continue;
        }
        let value = match node {
            Node::Zero => 1.0,
            Node::Terminal(_) => 0.0,
            Node::Average {
                children,
                denominator,
            } => {
                let denominator = *denominator as usize;
                ensure!(
                    denominator > 0 && children.len() <= denominator,
                    "invalid average at node {id} in failure propagation"
                );
                let mut sum = 0.0;
                for &child in children {
                    sum += failure[child as usize];
                }
                (sum + (denominator - children.len()) as f64) / denominator as f64
            }
            Node::Maximum(_) => unreachable!("reachable maxima were rejected"),
        };
        ensure!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "invalid direct failure {value} at node {id}"
        );
        failure[id] = value;
    }
    Ok(roots.iter().map(|&root| failure[root as usize]).collect())
}

/// Apply the reveal average R to the direct failures of K's quotient roots.
/// An omitted quotient root represents the zero survival operator, hence has
/// failure one. Callers needing homogeneous reset rewards must independently
/// require complete reset membership.
pub(crate) fn direct_failure(
    pre: &PreRevealIndex,
    qids: &[u32],
    roots: &[NodeId],
    dag: &ExprDag,
) -> Result<Vec<f64>> {
    ensure!(
        qids.len() == roots.len(),
        "failure root/qid length mismatch"
    );
    ensure!(
        dag.terminal_count() == pre.len(),
        "failure DAG terminal/pre-reveal dimension mismatch"
    );
    let root_failure = failure_at_roots(dag, roots)?;
    let mut full_failure = vec![1.0; pre.qid_to_zid.len()];
    let mut seen = vec![false; full_failure.len()];
    for ((&qid, &failure), root_index) in qids.iter().zip(&root_failure).zip(0..) {
        let qid = qid as usize;
        ensure!(
            qid < full_failure.len(),
            "failure qid {qid} is out of range"
        );
        ensure!(
            !seen[qid],
            "duplicate failure qid {qid} at root {root_index}"
        );
        seen[qid] = true;
        full_failure[qid] = failure;
    }
    pre.average_f64(&full_failure)
}

/// Solve the two rank-one stabilized systems for Q=P+d mu^T, then check their
/// true residuals and the original equation. P and P^T must describe the same
/// fixed, nonnegative linear operator. Their structural validation and a
/// separate properness certificate remain the caller's responsibility.
pub(crate) fn solve_deflated<P, PT>(
    mut apply_p: P,
    mut apply_pt: PT,
    direct_failure: &[f64],
    mu: &[f64],
    config: &DeflationConfig,
) -> Result<DeflationReport>
where
    P: FnMut(&[f64]) -> Result<Vec<f64>>,
    PT: FnMut(&[f64]) -> Result<Vec<f64>>,
{
    let n = direct_failure.len();
    ensure!(
        n > 0 && mu.len() == n,
        "invalid failure deflation dimensions"
    );
    ensure!(
        config.residual_tolerance.is_finite() && config.residual_tolerance > 0.0,
        "failure deflation residual tolerance must be positive and finite"
    );
    ensure!(
        config.normalization_tolerance.is_finite() && config.normalization_tolerance > 0.0,
        "failure deflation normalization tolerance must be positive and finite"
    );
    for (i, (&d, &weight)) in direct_failure.iter().zip(mu).enumerate() {
        ensure!(
            d.is_finite() && (0.0..=1.0).contains(&d),
            "invalid direct failure at {i}"
        );
        ensure!(
            weight.is_finite() && weight > 0.0,
            "mu must be strictly positive at {i}"
        );
    }
    let mu_sum = finite_sum(mu)?;
    ensure!(
        (mu_sum - 1.0).abs() <= config.normalization_tolerance,
        "mu is not normalized: sum={mu_sum:.17e}"
    );
    // Normalize the represented weights once; Q then uses exactly these same
    // weights in its forward, transpose, and normalization operations.
    let mu: Vec<_> = mu.iter().map(|&value| value / mu_sum).collect();
    let ones = vec![1.0; n];
    let zeros = vec![0.0; n];
    let mut p_applications = 1;
    let mut pt_applications = 0;
    let p_one = checked_output(apply_p(&ones)?, n, "P one")?;
    let mut row_mass: f64 = 0.0;
    for (i, (&survival, &failure)) in p_one.iter().zip(direct_failure).enumerate() {
        ensure!(
            survival >= -config.residual_tolerance && survival <= 1.0 + config.residual_tolerance,
            "P is not substochastic at row {i}"
        );
        row_mass = row_mass.max((survival + failure - 1.0).abs());
    }
    check_residual(
        "direct failure/row mass",
        row_mass,
        config.residual_tolerance,
    )?;

    // (I-Q^T+mu 1^T)nu=mu. Never form Q or its dense rank-one term.
    let stationary_solve = gmres(
        |x| {
            pt_applications += 1;
            let mut out = checked_output(apply_pt(x)?, n, "P transpose")?;
            let regenerate = finite_dot(direct_failure, x)?;
            let normalization = finite_sum(x)?;
            for i in 0..n {
                out[i] = x[i] - out[i] - mu[i] * regenerate + mu[i] * normalization;
            }
            checked_output(out, n, "stationary system")
        },
        &mu,
        &mu,
        &config.gmres,
    )
    .context("stationary failure-deflation solve")?;
    check_gmres_convergence("stationary", &stationary_solve, &mu, &config.gmres)?;
    let mut stationary =
        checked_output(stationary_solve.solution.clone(), n, "stationary solution")?;
    let stationary_sum = finite_sum(&stationary)?;
    ensure!(
        stationary_sum > 0.0,
        "stationary solution has no positive mass"
    );
    for value in &mut stationary {
        *value /= stationary_sum;
        ensure!(
            *value > 0.0 && value.is_finite(),
            "stationary solution is not strictly positive"
        );
    }
    pt_applications += 1;
    let pt_nu = checked_output(apply_pt(&stationary)?, n, "stationary P transpose")?;
    let failure_rate = finite_dot(&stationary, direct_failure)?;
    ensure!(
        failure_rate > 0.0 && failure_rate.is_finite(),
        "stationary failure rate is zero or invalid; a closed class cannot be deflated"
    );
    let constant = failure_rate.recip();
    ensure!(
        constant.is_finite() && constant > 0.0,
        "nonfinite failure-deflation constant"
    );
    let stationary_sum = finite_sum(&stationary)?;
    let stationary_normalization = (stationary_sum - 1.0).abs();
    let mut stationarity: f64 = 0.0;
    let mut stationary_system: f64 = 0.0;
    for i in 0..n {
        let residual = stationary[i] - pt_nu[i] - mu[i] * failure_rate;
        stationarity = stationarity.max(residual.abs());
        stationary_system =
            stationary_system.max((residual + mu[i] * (stationary_sum - 1.0)).abs());
    }
    check_residual("stationarity", stationarity, config.residual_tolerance)?;
    check_residual(
        "stationary system",
        stationary_system,
        config.residual_tolerance,
    )?;
    check_residual(
        "stationary normalization",
        stationary_normalization,
        config.normalization_tolerance,
    )?;

    let rhs: Vec<_> = direct_failure.iter().map(|&d| 1.0 - constant * d).collect();
    checked_finite(&rhs, "relative right-hand side")?;
    // (I-Q+one mu^T)h=one-Cd, with mu^T h=0.
    let relative_solve = gmres(
        |x| {
            p_applications += 1;
            let mut out = checked_output(apply_p(x)?, n, "P relative vector")?;
            let mean = finite_dot(&mu, x)?;
            for i in 0..n {
                out[i] = x[i] - out[i] - direct_failure[i] * mean + mean;
            }
            checked_output(out, n, "relative system")
        },
        &rhs,
        &zeros,
        &config.gmres,
    )
    .context("relative failure-deflation solve")?;
    check_gmres_convergence("relative", &relative_solve, &rhs, &config.gmres)?;
    let mut relative = checked_output(relative_solve.solution.clone(), n, "relative solution")?;
    // Recenter along the constant null direction of I-Q, then check the actual
    // equations again. This cannot turn a failed original equation into a pass.
    let mean = finite_dot(&mu, &relative)?;
    for value in &mut relative {
        *value -= mean;
    }
    checked_finite(&relative, "normalized relative solution")?;
    let mean = finite_dot(&mu, &relative)?;
    let relative_normalization = mean.abs();
    p_applications += 1;
    let p_h = checked_output(apply_p(&relative)?, n, "P normalized relative solution")?;
    let mut relative_equation: f64 = 0.0;
    let mut relative_system: f64 = 0.0;
    for i in 0..n {
        let residual = relative[i] - p_h[i] - direct_failure[i] * mean - rhs[i];
        relative_equation = relative_equation.max(residual.abs());
        relative_system = relative_system.max((residual + mean).abs());
    }
    check_residual(
        "relative equation",
        relative_equation,
        config.residual_tolerance,
    )?;
    check_residual(
        "relative system",
        relative_system,
        config.residual_tolerance,
    )?;
    check_residual(
        "relative normalization",
        relative_normalization,
        config.normalization_tolerance,
    )?;

    let solution: Vec<_> = relative.iter().map(|&h| constant + h).collect();
    for (i, &value) in solution.iter().enumerate() {
        ensure!(
            value.is_finite() && value > 0.0,
            "invalid deflated solution at state {i}"
        );
    }
    p_applications += 1;
    let p_u = checked_output(apply_p(&solution)?, n, "P deflated solution")?;
    let mut original: f64 = 0.0;
    for (&u, &pu) in solution.iter().zip(&p_u) {
        original = original.max((u - pu - 1.0).abs());
    }
    check_residual("original (I-P)u=one", original, config.residual_tolerance)?;

    Ok(DeflationReport {
        solution,
        constant,
        stationary,
        relative,
        stationary_solve,
        relative_solve,
        residuals: DeflationResiduals {
            row_mass,
            stationary_system,
            stationarity,
            stationary_normalization,
            relative_system,
            relative_equation,
            relative_normalization,
            original,
        },
        stats: DeflationStats {
            p_applications,
            pt_applications,
        },
    })
}

fn checked_finite(values: &[f64], name: &str) -> Result<()> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite {name}"
    );
    Ok(())
}

fn check_gmres_convergence(
    name: &str,
    report: &Report,
    rhs: &[f64],
    config: &Config,
) -> Result<()> {
    let rhs_norm = rhs.iter().fold(0.0f64, |norm, &value| norm.hypot(value));
    let threshold = config.abs_tol.max(config.rel_tol * rhs_norm);
    ensure!(
        report.converged,
        "{name} failure-deflation GMRES did not converge after {} iterations: true L2 residual {:.17e}, requested L2 threshold {threshold:.17e}, relative residual {:.17e}",
        report.iters,
        report.residual_norm,
        report.relative_residual,
    );
    Ok(())
}

fn checked_output(values: Vec<f64>, n: usize, name: &str) -> Result<Vec<f64>> {
    ensure!(
        values.len() == n,
        "{name} dimension mismatch: {} != {n}",
        values.len()
    );
    checked_finite(&values, name)?;
    Ok(values)
}

fn check_residual(name: &str, residual: f64, tolerance: f64) -> Result<()> {
    ensure!(
        residual.is_finite() && residual <= tolerance,
        "failure deflation {name} residual {residual:.17e} exceeds {tolerance:.17e}"
    );
    Ok(())
}

/// Neumaier summation keeps normalization checks useful for signed h and large
/// state counts. Products and their sums must remain finite.
fn finite_sum(values: &[f64]) -> Result<f64> {
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    for &value in values {
        ensure!(value.is_finite(), "nonfinite summand in failure deflation");
        let next = sum + value;
        ensure!(next.is_finite(), "sum overflow in failure deflation");
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    let result = sum + correction;
    ensure!(
        result.is_finite(),
        "corrected sum overflow in failure deflation"
    );
    Ok(result)
}

fn finite_dot(left: &[f64], right: &[f64]) -> Result<f64> {
    ensure!(
        left.len() == right.len(),
        "failure deflation dot-product dimension mismatch"
    );
    // Avoid another state-sized temporary in every matrix-free application.
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let product = left * right;
        ensure!(
            product.is_finite(),
            "nonfinite dot product in failure deflation"
        );
        let next = sum + product;
        ensure!(
            next.is_finite(),
            "dot-product overflow in failure deflation"
        );
        correction += if sum.abs() >= product.abs() {
            (sum - next) + product
        } else {
            (product - next) + sum
        };
        sum = next;
    }
    let result = sum + correction;
    ensure!(
        result.is_finite(),
        "corrected dot-product overflow in failure deflation"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DeflationConfig {
        DeflationConfig {
            gmres: Config {
                max_iters: 40,
                restart: 4,
                rel_tol: 1e-13,
                abs_tol: 1e-14,
                max_basis_bytes: 1024 * 1024,
            },
            residual_tolerance: 1e-8,
            normalization_tolerance: 1e-9,
        }
    }

    fn matvec(matrix: &[[f64; 2]; 2], x: &[f64], transpose: bool) -> Result<Vec<f64>> {
        Ok((0..2)
            .map(|i| {
                (0..2).map(|j| if transpose { matrix[j][i] } else { matrix[i][j] } * x[j]).sum()
            })
            .collect())
    }

    fn solve(matrix: [[f64; 2]; 2], failure: [f64; 2]) -> Result<DeflationReport> {
        solve_deflated(
            |x| matvec(&matrix, x, false),
            |x| matvec(&matrix, x, true),
            &failure,
            &uniform_mu(2)?,
            &config(),
        )
    }

    #[test]
    fn nonsymmetric_policy_matches_exact_original_solution() {
        let report = solve([[0.3, 0.4], [0.1, 0.2]], [0.3, 0.7]).unwrap();
        // Invert [[0.7,-0.4],[-0.1,0.8]] independently of deflation.
        let expected = [1.2 / 0.52, 0.8 / 0.52];
        for (&actual, &expected) in report.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((report.constant - 1.0 / 0.52).abs() < 1e-12);
        assert!((report.stationary[0] - 0.45).abs() < 1e-12);
        assert!((report.stationary[1] - 0.55).abs() < 1e-12);
        assert!(report.residuals.original < 1e-12);
        assert!(report.residuals.relative_normalization < 1e-12);
        assert!(report.stationary_solve.converged && report.relative_solve.converged);
    }

    #[test]
    fn very_near_unit_dominant_mode_stays_finite() {
        let d = 2f64.powi(-40);
        let s = 1.0 - d;
        let report = solve([[0.75 * s, 0.25 * s], [0.5 * s, 0.5 * s]], [d, d]).unwrap();
        for &u in &report.solution {
            assert!((u * d - 1.0).abs() < 1e-12);
        }
        assert!(report.residuals.original <= 1e-8);
    }

    #[test]
    fn multiple_slow_modes_match_diagonal_lifetimes() {
        let d0 = 2f64.powi(-18);
        let d1 = 2f64.powi(-20);
        let matrix = [[1.0 - d0, 0.0], [0.0, 1.0 - d1]];
        let mut config = config();
        // h has components near +/-393216, so evaluating h-P h incurs
        // absolute f64 rounding on that scale, even with exact dyadic P.
        // Request a representable internal residual. The independent original
        // equation/publication tolerance remains the same absolute 1e-8.
        config.gmres.abs_tol = f64::EPSILON * d0.recip().max(d1.recip());
        assert_eq!(config.residual_tolerance, 1e-8);
        let report = solve_deflated(
            |x| matvec(&matrix, x, false),
            |x| matvec(&matrix, x, true),
            &[d0, d1],
            &uniform_mu(2).unwrap(),
            &config,
        )
        .unwrap();
        assert!((report.solution[0] * d0 - 1.0).abs() < 1e-8);
        assert!((report.solution[1] * d1 - 1.0).abs() < 1e-8);
        assert!(report.residuals.original <= config.residual_tolerance);
        assert!(report.relative_solve.converged);
    }

    #[test]
    fn multiple_slow_modes_reject_unattainable_internal_precision_with_diagnostics() {
        let d0 = 2f64.powi(-18);
        let d1 = 2f64.powi(-20);
        let error = solve([[1.0 - d0, 0.0], [0.0, 1.0 - d1]], [d0, d1]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("relative failure-deflation GMRES did not converge"));
        assert!(message.contains("true L2 residual"));
        assert!(message.contains("requested L2 threshold"));
    }

    #[test]
    fn closed_class_and_zero_failure_are_rejected() {
        assert!(solve([[0.5, 0.0], [0.0, 1.0]], [0.5, 0.0]).is_err());
        assert!(solve([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]).is_err());
    }

    #[test]
    fn inconsistent_failure_and_invalid_mu_are_rejected() {
        assert!(solve([[0.3, 0.4], [0.1, 0.2]], [0.4, 0.7]).is_err());
        let matrix = [[0.3, 0.4], [0.1, 0.2]];
        assert!(solve_deflated(
            |x| matvec(&matrix, x, false),
            |x| matvec(&matrix, x, true),
            &[0.3, 0.7],
            &[0.0, 1.0],
            &config(),
        )
        .is_err());
    }

    #[test]
    fn direct_failure_survives_cancellation_of_one_minus_survival() {
        let mut dag = ExprDag::new(1, 100, 1000).unwrap();
        let terminal = dag.terminal(0).unwrap();
        let mut root = dag.zero();
        for _ in 0..24 {
            root = dag
                .average(
                    &[
                        root, terminal, terminal, terminal, terminal, terminal, terminal,
                    ],
                    7,
                )
                .unwrap();
        }
        let d = failure_at_roots(&dag, &[root]).unwrap()[0];
        let survival = dag.eval_f64(&[1.0]).unwrap()[root as usize];
        assert!(d > 0.0 && d < f64::EPSILON);
        assert!((d / 7f64.powi(-24) - 1.0).abs() < 1e-14);
        assert_eq!(1.0 - survival, 0.0);
    }

    #[test]
    fn direct_failure_accounts_for_missing_children_and_rejects_reachable_maximum() {
        let mut dag = ExprDag::new(1, 20, 30).unwrap();
        let terminal = dag.terminal(0).unwrap();
        let root = dag.average(&[terminal, dag.zero()], 4).unwrap();
        assert_eq!(failure_at_roots(&dag, &[root]).unwrap(), vec![0.75]);
        let maximum = dag.maximum(&[terminal, dag.zero()]).unwrap();
        assert_eq!(failure_at_roots(&dag, &[terminal]).unwrap(), vec![0.0]);
        assert!(failure_at_roots(&dag, &[maximum]).is_err());
        assert!(failure_at_roots(&dag, &[u32::MAX]).is_err());
    }
}
