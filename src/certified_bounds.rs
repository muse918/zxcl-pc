//! Outward bounds and checked pruning for the gamma=1 optimality operator.
//!
//! The operator is `H(u) = 1 + R F(u)`: `F` is the ordered expression DAG
//! and `R` averages its *post-reveal* roots. In particular, a Maximum below a
//! reveal average is never moved outside that average. Missing children retain
//! their original denominator and contribute zero.
//!
//! A finite positive supersolution is essential. A finite horizon or a small
//! residual alone proves neither finiteness nor a value-error bound for an SSP.
//! Here a verified `1 + R F(w) <= w`, together with an outward contraction
//! coefficient strictly below one, supplies that missing certificate. The
//! max/average DAG is monotone, positively homogeneous, and subadditive, so
//! `|R F(x)-R F(y)| <= R F(|x-y|)`. This proves contraction in the weighted
//! infinity norm with weight `w`, including all choices represented by Maximum.

use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

use crate::expression_dag::{ExprDag, Node, NodeId};
use crate::pre_reveal::PreRevealIndex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn new(lower: f64, upper: f64) -> Result<Self> {
        let interval = Self { lower, upper };
        interval.validate()?;
        Ok(interval)
    }

    pub fn point(value: f64) -> Result<Self> {
        Self::new(value, value)
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.lower.is_finite() && self.upper.is_finite(),
            "interval endpoints must be finite"
        );
        ensure!(
            self.lower >= 0.0 && self.lower <= self.upper,
            "optimality intervals require 0 <= lower <= upper"
        );
        Ok(())
    }
}

const ZERO: Interval = Interval {
    lower: 0.0,
    upper: 0.0,
};

/// Portable adjacent-float operations, including signed zero and subnormals.
/// These helpers do not change the process rounding mode or require `next_up`
/// support from a particular Rust version.
fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn finite(value: f64) -> Result<f64> {
    ensure!(
        value.is_finite(),
        "interval arithmetic overflow; no finite bound was verified"
    );
    Ok(value)
}

fn add_lower(a: f64, b: f64) -> Result<f64> {
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    Ok(next_down(finite(a + b)?).max(0.0))
}

fn add_upper(a: f64, b: f64) -> Result<f64> {
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    finite(next_up(finite(a + b)?))
}

fn divide_lower(a: f64, b: f64) -> Result<f64> {
    ensure!(b.is_finite() && b > 0.0, "positive finite divisor required");
    if a == 0.0 || b == 1.0 {
        return Ok(a);
    }
    Ok(next_down(finite(a / b)?).max(0.0))
}

fn divide_upper(a: f64, b: f64) -> Result<f64> {
    ensure!(b.is_finite() && b > 0.0, "positive finite divisor required");
    if a == 0.0 || b == 1.0 {
        return Ok(a);
    }
    finite(next_up(finite(a / b)?))
}

fn subtract_lower(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 {
        return Ok(a);
    }
    if a == b {
        return Ok(0.0);
    }
    finite(next_down(finite(a - b)?))
}

fn absolute_difference_upper(a: f64, b: f64) -> Result<f64> {
    if a == b {
        return Ok(0.0);
    }
    finite(next_up(finite((a - b).abs())?))
}

fn multiply_upper(a: f64, b: f64) -> Result<f64> {
    if a == 0.0 || b == 0.0 {
        return Ok(0.0);
    }
    if a == 1.0 {
        return Ok(b);
    }
    if b == 1.0 {
        return Ok(a);
    }
    finite(next_up(finite(a * b)?))
}

fn add_intervals(a: Interval, b: Interval) -> Result<Interval> {
    Ok(Interval {
        lower: add_lower(a.lower, b.lower)?,
        upper: add_upper(a.upper, b.upper)?,
    })
}

fn divide_interval(value: Interval, denominator: usize) -> Result<Interval> {
    ensure!(
        (1..=255).contains(&denominator),
        "average denominator must be in 1..=255"
    );
    Ok(Interval {
        lower: divide_lower(value.lower, denominator as f64)?,
        upper: divide_upper(value.upper, denominator as f64)?,
    })
}

fn check_gradual_underflow() -> Result<()> {
    // One adjacent float encloses ordinary IEEE rounding, including subnormal
    // results. A caller that enables flush-to-zero/denormals-are-zero can move
    // an arithmetic result by many representable floats, invalidating that
    // argument. Black boxes keep these checks from being folded at compile time.
    let tiny = std::hint::black_box(f64::from_bits(1));
    let doubled = tiny + std::hint::black_box(tiny);
    let half_min_normal = std::hint::black_box(f64::MIN_POSITIVE) / std::hint::black_box(2.0);
    ensure!(
        doubled.to_bits() == 2 && half_min_normal.to_bits() == (1u64 << 51),
        "certified interval arithmetic requires gradual f64 underflow"
    );
    Ok(())
}

/// Enclose every expression over a nonnegative terminal box. All addition and
/// division endpoints are rounded outward in the stored child order.
pub fn evaluate_intervals(dag: &ExprDag, terminals: &[Interval]) -> Result<Vec<Interval>> {
    check_gradual_underflow()?;
    ensure!(
        terminals.len() == dag.terminal_count(),
        "terminal interval count mismatch"
    );
    for (index, &interval) in terminals.iter().enumerate() {
        interval
            .validate()
            .with_context(|| format!("terminal interval {index}"))?;
    }
    let mut values = Vec::<Interval>::with_capacity(dag.node_count());
    for (id, node) in dag.nodes().iter().enumerate() {
        let value = match node {
            Node::Zero => ZERO,
            Node::Terminal(index) => *terminals
                .get(*index as usize)
                .context("invalid terminal node")?,
            Node::Average {
                children,
                denominator,
            } => {
                ensure!(
                    children.len() <= *denominator as usize,
                    "average has more children than its full denominator"
                );
                let mut sum = ZERO;
                for &child in children {
                    let value = *values
                        .get(child as usize)
                        .with_context(|| format!("node {id} has a non-topological child"))?;
                    sum =
                        add_intervals(sum, value).with_context(|| format!("average node {id}"))?;
                }
                divide_interval(sum, *denominator as usize)?
            }
            Node::Maximum(children) => {
                ensure!(!children.is_empty(), "empty Maximum at node {id}");
                let mut best = ZERO;
                for &child in children {
                    let value = *values
                        .get(child as usize)
                        .with_context(|| format!("node {id} has a non-topological child"))?;
                    best.lower = best.lower.max(value.lower);
                    best.upper = best.upper.max(value.upper);
                }
                best
            }
        };
        values.push(value);
    }
    Ok(values)
}

/// Full reveal support and denominator for each pre-reveal state. Quotient IDs
/// refer to roots through the explicitly supplied file-order `qids` mapping.
/// An omitted root has zero value; its probability mass is never redistributed.
pub trait RevealGroups {
    fn group_count(&self) -> usize;
    fn quotient_count(&self) -> usize;
    fn reveal_group(&self, zid: usize) -> (&[u32], usize);
}

impl RevealGroups for PreRevealIndex {
    fn group_count(&self) -> usize {
        self.len()
    }
    fn quotient_count(&self) -> usize {
        self.qid_to_zid.len()
    }
    fn reveal_group(&self, zid: usize) -> (&[u32], usize) {
        (
            self.reveal_qids(zid as u32),
            self.states[zid].bag_mask.count_ones() as usize,
        )
    }
}

fn averaged_root_intervals<G: RevealGroups + ?Sized>(
    dag: &ExprDag,
    roots: &[NodeId],
    qids: &[u32],
    groups: &G,
    terminals: &[Interval],
) -> Result<Vec<Interval>> {
    ensure!(
        groups.group_count() == dag.terminal_count(),
        "pre-reveal group/terminal count mismatch"
    );
    ensure!(roots.len() == qids.len(), "root/qid count mismatch");
    let values = evaluate_intervals(dag, terminals)?;
    let mut quotient = vec![ZERO; groups.quotient_count()];
    let mut seen = vec![false; quotient.len()];
    for (&root, &qid) in roots.iter().zip(qids) {
        let value = *values.get(root as usize).context("invalid root node")?;
        let slot = quotient
            .get_mut(qid as usize)
            .context("root quotient ID out of range")?;
        ensure!(!seen[qid as usize], "duplicate root quotient ID {qid}");
        seen[qid as usize] = true;
        *slot = value;
    }
    let mut averaged = Vec::with_capacity(groups.group_count());
    for zid in 0..groups.group_count() {
        let (children, denominator) = groups.reveal_group(zid);
        ensure!(
            children.len() <= denominator,
            "reveal group {zid} has excess probability mass"
        );
        let mut sum = ZERO;
        for &qid in children {
            sum = add_intervals(
                sum,
                *quotient
                    .get(qid as usize)
                    .context("reveal quotient ID out of range")?,
            )?;
        }
        averaged.push(divide_interval(sum, denominator)?);
    }
    Ok(averaged)
}

/// Outward evaluation of `1 + R F` without changing where revelation occurs.
pub fn evaluate_h_intervals<G: RevealGroups + ?Sized>(
    dag: &ExprDag,
    roots: &[NodeId],
    qids: &[u32],
    groups: &G,
    terminals: &[Interval],
) -> Result<Vec<Interval>> {
    averaged_root_intervals(dag, roots, qids, groups, terminals)?
        .into_iter()
        .map(|value| {
            let mut interval = add_intervals(
                Interval {
                    lower: 1.0,
                    upper: 1.0,
                },
                value,
            )?;
            // H is analytically at least one. Adding a positive value below
            // half an ulp must not invalidate the all-ones subsolution through
            // conservative downward rounding.
            interval.lower = interval.lower.max(1.0);
            Ok(interval)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct OptimalityCertificate {
    /// This box contains the unique optimal gamma=1 fixed point.
    pub terminal_bounds: Vec<Interval>,
    /// Lower bound on `min_i (w_i - (R F(w))_i)`, strictly positive.
    pub min_margin_lower: f64,
    /// Verified upper bound on `max_i (R F(w))_i / w_i`, strictly below 1.
    pub contraction_factor_upper: f64,
    /// Strictly positive lower bound on `min_i (w_i - (R F(w))_i)/w_i`.
    /// Residual bounds divide by this directly, avoiding cancellation in
    /// subtracting a rounded contraction factor from one.
    pub contraction_gap_lower: f64,
    /// Lower bound on the smallest component of `w - H(w)`; may be zero.
    pub supersolution_slack_lower: f64,
}

/// Certify a *full*, finite, strictly positive supersolution. A smaller supplied
/// lower vector is accepted only after checking its subsolution inequality.
/// Positivity and contraction make this subsolution a valid global lower bound;
/// a subsolution test by itself would be insufficient for an improper SSP.
pub fn certify_supersolution<G: RevealGroups + ?Sized>(
    dag: &ExprDag,
    roots: &[NodeId],
    qids: &[u32],
    groups: &G,
    upper: &[f64],
    proposed_lower: Option<&[f64]>,
) -> Result<OptimalityCertificate> {
    ensure!(
        !upper.is_empty(),
        "cannot certify an empty optimality operator"
    );
    ensure!(
        upper.len() == dag.terminal_count(),
        "supersolution dimension mismatch"
    );
    let mut points = Vec::with_capacity(upper.len());
    for (index, &value) in upper.iter().enumerate() {
        ensure!(
            value.is_finite() && value > 0.0,
            "supersolution {index} must be finite and strictly positive"
        );
        points.push(Interval::point(value)?);
    }
    let averaged = averaged_root_intervals(dag, roots, qids, groups, &points)?;
    let mut min_margin_lower = f64::INFINITY;
    let mut contraction_factor_upper = 0.0f64;
    let mut contraction_gap_lower = f64::INFINITY;
    let mut supersolution_slack_lower = f64::INFINITY;
    for (index, (&weight, value)) in upper.iter().zip(&averaged).enumerate() {
        let h_upper = add_upper(1.0, value.upper)?;
        ensure!(h_upper <= weight, "no finite supersolution verified at terminal {index}: outward H(w)={h_upper} > w={weight}");
        let margin_lower = subtract_lower(weight, value.upper)?;
        min_margin_lower = min_margin_lower.min(margin_lower);
        contraction_gap_lower = contraction_gap_lower.min(divide_lower(margin_lower, weight)?);
        contraction_factor_upper = contraction_factor_upper.max(divide_upper(value.upper, weight)?);
        supersolution_slack_lower = supersolution_slack_lower.min(subtract_lower(weight, h_upper)?);
    }
    ensure!(
        min_margin_lower > 0.0,
        "no strictly positive transience margin verified"
    );
    ensure!(
        contraction_factor_upper < 1.0,
        "no contraction factor strictly below one verified"
    );
    ensure!(
        contraction_gap_lower > 0.0,
        "no positive contraction gap verified"
    );

    let default_lower = vec![1.0; upper.len()];
    let lower = proposed_lower.unwrap_or(&default_lower);
    ensure!(lower.len() == upper.len(), "subsolution dimension mismatch");
    let mut lower_points = Vec::with_capacity(lower.len());
    for (index, (&lo, &hi)) in lower.iter().zip(upper).enumerate() {
        ensure!(
            lo.is_finite() && lo >= 0.0 && lo <= hi,
            "invalid subsolution bound at terminal {index}"
        );
        lower_points.push(Interval::point(lo)?);
    }
    let h_lower = evaluate_h_intervals(dag, roots, qids, groups, &lower_points)?;
    for (index, (&lo, value)) in lower.iter().zip(&h_lower).enumerate() {
        ensure!(
            lo <= value.lower,
            "subsolution inequality not verified at terminal {index}"
        );
    }
    Ok(OptimalityCertificate {
        terminal_bounds: lower
            .iter()
            .zip(upper)
            .map(|(&lower, &upper)| Interval { lower, upper })
            .collect(),
        min_margin_lower,
        contraction_factor_upper,
        contraction_gap_lower,
        supersolution_slack_lower,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualErrorBound {
    pub residual_inf_upper: f64,
    pub weighted_residual_upper: f64,
    /// Error in the infinity norm weighted by the certified supersolution.
    pub weighted_error_upper: f64,
    /// Error in the ordinary infinity norm of pre-reveal values.
    pub absolute_error_upper: f64,
}

/// Reverify the certificate against this exact operator before using its weight
/// for a residual bound. This prevents accidentally reusing a proof for a
/// different DAG, qid mapping, grouping, or publicly modified certificate.
pub fn residual_error_bound<G: RevealGroups + ?Sized>(
    dag: &ExprDag,
    roots: &[NodeId],
    qids: &[u32],
    groups: &G,
    certificate: &OptimalityCertificate,
    candidate: &[f64],
) -> Result<ResidualErrorBound> {
    let upper: Vec<_> = certificate
        .terminal_bounds
        .iter()
        .map(|v| v.upper)
        .collect();
    let checked = certify_supersolution(dag, roots, qids, groups, &upper, None)?;
    ensure!(
        candidate.len() == upper.len(),
        "residual candidate dimension mismatch"
    );
    let points: Result<Vec<_>> = candidate
        .iter()
        .map(|&value| Interval::point(value))
        .collect();
    let h = evaluate_h_intervals(dag, roots, qids, groups, &points?)?;
    let mut residual_inf_upper = 0.0f64;
    let mut weighted_residual_upper = 0.0f64;
    let mut max_weight = 0.0f64;
    for ((&value, &weight), interval) in candidate.iter().zip(&upper).zip(h) {
        let residual = absolute_difference_upper(interval.lower, value)?
            .max(absolute_difference_upper(interval.upper, value)?);
        residual_inf_upper = residual_inf_upper.max(residual);
        weighted_residual_upper = weighted_residual_upper.max(divide_upper(residual, weight)?);
        max_weight = max_weight.max(weight);
    }
    let gap = checked.contraction_gap_lower;
    ensure!(gap > 0.0, "no positive contraction gap verified");
    let weighted_error_upper = divide_upper(weighted_residual_upper, gap)?;
    Ok(ResidualErrorBound {
        residual_inf_upper,
        weighted_residual_upper,
        weighted_error_upper,
        absolute_error_upper: multiply_upper(weighted_error_upper, max_weight)?,
    })
}

#[derive(Clone, Debug)]
pub struct DominationProof {
    pub maximum: NodeId,
    pub removed_position: usize,
    pub removed_child: NodeId,
    pub witness_child: NodeId,
    pub removed_upper: f64,
    pub witness_lower: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruningValidity {
    /// Equality is proved only for terminal vectors inside the supplied box.
    SuppliedBox,
    /// The box contains the unique optimal fixed point, and the pruned operator
    /// has that same unique fixed point by the verified global contraction.
    CertifiedOptimalFixedPoint,
}

#[derive(Clone, Debug)]
pub struct PrunedDag {
    pub dag: ExprDag,
    pub roots: Vec<NodeId>,
    pub original_node_count: usize,
    pub original_edge_count: usize,
    /// Removed child occurrences across unique original Maximum nodes.
    pub removed_branches: usize,
    pub proofs: Vec<DominationProof>,
    pub terminal_bounds: Vec<Interval>,
    pub validity: PruningValidity,
}

/// Remove only a strictly dominated Maximum child. Ties, order, multiplicity,
/// Average denominators, and unsimplified structure are retained. Rebuilding
/// follows retained edges from the roots and omits orphaned subgraphs, while
/// always keeping the zero node. Structural interning never merges expressions
/// by their numerical values. Proof node IDs refer to the original DAG.
pub fn prune(dag: &ExprDag, roots: &[NodeId], terminal_bounds: &[Interval]) -> Result<PrunedDag> {
    let values = evaluate_intervals(dag, terminal_bounds)?;
    for &root in roots {
        ensure!(
            (root as usize) < values.len(),
            "invalid pruning root {root}"
        );
    }
    // Store replacement edge lists only for changed Maximum nodes. Average
    // edges and unchanged maxima can be borrowed from the original arena.
    let mut retained_maxima = HashMap::<NodeId, Vec<NodeId>>::new();
    let mut proofs = Vec::new();
    for (id, node) in dag.nodes().iter().enumerate() {
        let Node::Maximum(children) = node else {
            continue;
        };
        let mut witness = children[0];
        for &child in &children[1..] {
            if values[child as usize].lower > values[witness as usize].lower {
                witness = child;
            }
        }
        let witness_lower = values[witness as usize].lower;
        let mut retained = Vec::with_capacity(children.len());
        for (position, &child) in children.iter().enumerate() {
            if values[child as usize].upper < witness_lower {
                proofs.push(DominationProof {
                    maximum: id as NodeId,
                    removed_position: position,
                    removed_child: child,
                    witness_child: witness,
                    removed_upper: values[child as usize].upper,
                    witness_lower,
                });
            } else {
                retained.push(child);
            }
        }
        // The witness always survives: its upper is at least its lower.
        if retained.len() != children.len() {
            retained_maxima.insert(id as NodeId, retained);
        }
    }
    drop(values);

    // Children precede parents, so a descending pass marks the transitive
    // retained closure without a traversal stack or copying Average edges.
    let mut reachable = vec![false; dag.node_count()];
    reachable[dag.zero() as usize] = true;
    for &root in roots {
        reachable[root as usize] = true;
    }
    for id in (0..dag.node_count()).rev() {
        if !reachable[id] {
            continue;
        }
        let children: &[NodeId] = match &dag.nodes()[id] {
            Node::Zero | Node::Terminal(_) => &[],
            Node::Average { children, .. } => children,
            Node::Maximum(children) => retained_maxima
                .get(&(id as NodeId))
                .map(Vec::as_slice)
                .unwrap_or(children),
        };
        for &child in children {
            reachable[child as usize] = true;
        }
    }

    let mut rebuilt = ExprDag::new(
        dag.terminal_count(),
        dag.node_count()
            .checked_add(1)
            .context("pruning node limit overflow")?,
        dag.edge_count()
            .checked_add(1)
            .context("pruning edge limit overflow")?,
    )?;
    let mut mapped = vec![NodeId::MAX; dag.node_count()];
    for (id, node) in dag.nodes().iter().enumerate() {
        if !reachable[id] {
            continue;
        }
        mapped[id] = match node {
            Node::Zero => rebuilt.zero(),
            Node::Terminal(index) => rebuilt.terminal(*index as usize)?,
            Node::Average {
                children,
                denominator,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|&child| mapped[child as usize])
                    .collect();
                rebuilt.average(&children, *denominator as usize)?
            }
            Node::Maximum(children) => {
                let retained = retained_maxima
                    .get(&(id as NodeId))
                    .map(Vec::as_slice)
                    .unwrap_or(children);
                let children: Vec<_> = retained
                    .iter()
                    .map(|&child| mapped[child as usize])
                    .collect();
                rebuilt.maximum(&children)?
            }
        };
    }
    Ok(PrunedDag {
        dag: rebuilt,
        roots: roots.iter().map(|&root| mapped[root as usize]).collect(),
        original_node_count: dag.node_count(),
        original_edge_count: dag.edge_count(),
        removed_branches: proofs.len(),
        proofs,
        terminal_bounds: terminal_bounds.to_vec(),
        validity: PruningValidity::SuppliedBox,
    })
}

#[derive(Clone, Debug)]
pub struct CertifiedPruning {
    pub pruned: PrunedDag,
    pub certificate: OptimalityCertificate,
}

/// Verify and consume the enclosure in the same call. No certificate can be
/// silently applied to a different expression or reveal mapping.
pub fn certify_and_prune<G: RevealGroups + ?Sized>(
    dag: &ExprDag,
    roots: &[NodeId],
    qids: &[u32],
    groups: &G,
    upper: &[f64],
    lower: Option<&[f64]>,
) -> Result<CertifiedPruning> {
    let certificate = certify_supersolution(dag, roots, qids, groups, upper, lower)?;
    let mut pruned = prune(dag, roots, &certificate.terminal_bounds)?;
    pruned.validity = PruningValidity::CertifiedOptimalFixedPoint;
    Ok(CertifiedPruning {
        pruned,
        certificate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Groups {
        quotient_count: usize,
        groups: Vec<(Vec<u32>, usize)>,
    }
    impl RevealGroups for Groups {
        fn group_count(&self) -> usize {
            self.groups.len()
        }
        fn quotient_count(&self) -> usize {
            self.quotient_count
        }
        fn reveal_group(&self, zid: usize) -> (&[u32], usize) {
            (&self.groups[zid].0, self.groups[zid].1)
        }
    }

    fn dag(terminals: usize) -> ExprDag {
        ExprDag::new(terminals, 1000, 10000).unwrap()
    }
    fn contains(interval: Interval, value: f64) {
        assert!(
            interval.lower <= value && value <= interval.upper,
            "{interval:?} does not contain {value}"
        );
    }

    fn compare_small_rational(
        value: f64,
        numerator: u128,
        denominator: u128,
    ) -> std::cmp::Ordering {
        assert!(value.is_finite() && value >= 0.0 && denominator > 0);
        if value == 0.0 {
            return 0u128.cmp(&numerator);
        }
        let bits = value.to_bits();
        let exponent_bits = (bits >> 52) & 0x7ff;
        // The small rational fixture below stays in the normal range and its
        // exact cross-products fit u128. Decode the endpoint rather than using
        // another floating operation as the containment oracle.
        assert!(exponent_bits > 0);
        let significand = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as u128;
        let exponent = exponent_bits as i32 - 1023 - 52;
        let (left, right) = if exponent >= 0 {
            (
                significand
                    .checked_shl(exponent as u32)
                    .unwrap()
                    .checked_mul(denominator)
                    .unwrap(),
                numerator,
            )
        } else {
            (
                significand.checked_mul(denominator).unwrap(),
                numerator.checked_shl((-exponent) as u32).unwrap(),
            )
        };
        left.cmp(&right)
    }

    fn contains_small_rational(interval: Interval, numerator: u128, denominator: u128) {
        assert!(
            compare_small_rational(interval.lower, numerator, denominator)
                != std::cmp::Ordering::Greater
        );
        assert!(
            compare_small_rational(interval.upper, numerator, denominator)
                != std::cmp::Ordering::Less
        );
    }

    #[test]
    fn adjacent_float_helpers_cover_zero_subnormal_and_signs() {
        let tiny = f64::from_bits(1);
        assert_eq!(next_up(0.0), tiny);
        assert_eq!(next_up(-0.0), tiny);
        assert_eq!(next_down(0.0), -tiny);
        assert_eq!(next_down(tiny), 0.0);
        assert_eq!(next_up(-tiny), -0.0);
        assert!(next_up(-1.0) > -1.0);
        assert!(next_down(-1.0) < -1.0);
    }

    #[test]
    fn intervals_contain_exact_small_rationals_and_missing_mass() {
        let mut dag = dag(3);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let c = dag.terminal(2).unwrap();
        let average = dag.average(&[a, b, a], 7).unwrap();
        let nested = dag.average(&[average, c], 3).unwrap();
        let maximum = dag.maximum(&[average, nested]).unwrap();
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..5 {
                    let values = evaluate_intervals(
                        &dag,
                        &[
                            Interval::point(x as f64).unwrap(),
                            Interval::point(y as f64).unwrap(),
                            Interval::point(z as f64).unwrap(),
                        ],
                    )
                    .unwrap();
                    contains_small_rational(values[average as usize], (2 * x + y) as u128, 7);
                    contains_small_rational(
                        values[nested as usize],
                        (2 * x + y + 7 * z) as u128,
                        21,
                    );
                    contains_small_rational(
                        values[maximum as usize],
                        ((3 * (2 * x + y)) as u128).max((2 * x + y + 7 * z) as u128),
                        21,
                    );
                }
            }
        }
        let missing = dag.average(&[a], 7).unwrap();
        let values = evaluate_intervals(&dag, &[Interval::point(7.0).unwrap(); 3]).unwrap();
        contains(values[missing as usize], 1.0);
        assert!(values[missing as usize].upper < 2.0);
    }

    #[test]
    fn ties_and_numerically_equal_structures_are_retained() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let explicit = dag.average(&[a], 1).unwrap();
        let maximum = dag.maximum(&[a, explicit, a]).unwrap();
        let pruned = prune(&dag, &[maximum], &[Interval::new(1.0, 2.0).unwrap()]).unwrap();
        assert_eq!(pruned.removed_branches, 0);
        let Node::Maximum(children) = &pruned.dag.nodes()[pruned.roots[0] as usize] else {
            panic!("maximum was simplified");
        };
        assert_eq!(children.len(), 3);
        assert_ne!(children[0], children[1]);
        assert_eq!(children[0], children[2]);
        assert_eq!(pruned.validity, PruningValidity::SuppliedBox);
    }

    #[test]
    fn pruned_roots_equal_original_for_vectors_inside_the_box() {
        let mut dag = dag(2);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let small = dag.average(&[a], 7).unwrap();
        let maximum = dag.maximum(&[small, b, b, dag.zero()]).unwrap();
        let root = dag.average(&[maximum, a, maximum], 7).unwrap();
        let pruned = prune(
            &dag,
            &[root],
            &[
                Interval::new(1.0, 2.0).unwrap(),
                Interval::new(2.0, 4.0).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(pruned.removed_branches, 2);
        assert!(pruned
            .proofs
            .iter()
            .all(|proof| proof.removed_upper < proof.witness_lower));
        let mut seed = 17u64;
        for _ in 0..100 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = 1.0 + (seed >> 11) as f64 / (1u64 << 53) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = 2.0 + 2.0 * (seed >> 11) as f64 / (1u64 << 53) as f64;
            let original = dag.eval_f64(&[x, y]).unwrap()[root as usize];
            let rebuilt = pruned.dag.eval_f64(&[x, y]).unwrap()[pruned.roots[0] as usize];
            assert_eq!(original.to_bits(), rebuilt.to_bits());
        }
        let Node::Average { denominator, .. } = pruned.dag.nodes()[pruned.roots[0] as usize] else {
            panic!("average changed");
        };
        assert_eq!(denominator, 7);
    }

    #[test]
    fn strict_dominance_compacts_orphan_subgraphs() {
        let mut dag = dag(2);
        let a = dag.terminal(0).unwrap();
        let half = dag.average(&[a], 2).unwrap();
        let orphan = dag.average(&[half], 3).unwrap();
        let b = dag.terminal(1).unwrap();
        let root = dag.maximum(&[orphan, b]).unwrap();
        let bounds = [
            Interval::new(1.0, 2.0).unwrap(),
            Interval::new(2.0, 4.0).unwrap(),
        ];
        let pruned = prune(&dag, &[root], &bounds).unwrap();
        assert_eq!(pruned.removed_branches, 1);
        assert_eq!(pruned.proofs[0].removed_child, orphan);
        assert_eq!(pruned.dag.node_count(), 3); // Zero, Terminal(1), Maximum.
        assert_eq!(pruned.original_node_count - pruned.dag.node_count(), 3);
        assert_eq!(pruned.dag.edge_count(), 1);
        assert_eq!(pruned.dag.terminal_count(), 2);
        assert!(pruned
            .dag
            .nodes()
            .iter()
            .all(|node| !matches!(node, Node::Terminal(0) | Node::Average { .. })));
        assert_eq!(pruned.dag.nodes()[pruned.dag.zero() as usize], Node::Zero);

        let mut seed = 29u64;
        for _ in 0..100 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = 1.0 + (seed >> 11) as f64 / (1u64 << 53) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = 2.0 + 2.0 * (seed >> 11) as f64 / (1u64 << 53) as f64;
            assert_eq!(
                dag.eval_f64(&[x, y]).unwrap()[root as usize].to_bits(),
                pruned.dag.eval_f64(&[x, y]).unwrap()[pruned.roots[0] as usize].to_bits(),
            );
        }

        // A removed edge does not orphan a subgraph used by another root.
        let shared = prune(&dag, &[root, orphan], &bounds).unwrap();
        assert_eq!(shared.dag.node_count(), dag.node_count());
        let original = dag.eval_f64(&[1.75, 3.5]).unwrap();
        let rebuilt = shared.dag.eval_f64(&[1.75, 3.5]).unwrap();
        assert_eq!(
            original[orphan as usize].to_bits(),
            rebuilt[shared.roots[1] as usize].to_bits()
        );

        let empty = prune(&dag, &[], &bounds).unwrap();
        assert_eq!(empty.dag.node_count(), 1);
        assert_eq!(empty.dag.edge_count(), 0);
        assert!(empty.roots.is_empty());
    }

    #[test]
    fn proper_operator_has_certificate_and_residual_error_bound() {
        let mut dag = dag(1);
        let terminal = dag.terminal(0).unwrap();
        let root = dag.average(&[terminal], 2).unwrap();
        let groups = Groups {
            quotient_count: 1,
            groups: vec![(vec![0], 1)],
        };
        let certificate =
            certify_supersolution(&dag, &[root], &[0], &groups, &[4.0], None).unwrap();
        assert!(certificate.min_margin_lower > 1.0);
        assert!(certificate.contraction_factor_upper < 0.51);
        contains(certificate.terminal_bounds[0], 2.0);
        for candidate in [1.5, 2.5] {
            let bound =
                residual_error_bound(&dag, &[root], &[0], &groups, &certificate, &[candidate])
                    .unwrap();
            assert!(bound.absolute_error_upper >= 0.5);
            assert!(bound.absolute_error_upper < 0.500000000001);
        }
        let result = certify_and_prune(&dag, &[root], &[0], &groups, &[4.0], None).unwrap();
        assert_eq!(
            result.pruned.validity,
            PruningValidity::CertifiedOptimalFixedPoint
        );
        assert!(certify_supersolution(&dag, &[root], &[0], &groups, &[4.0], Some(&[3.0])).is_err());
    }

    #[test]
    fn closed_class_and_small_residual_do_not_certify_finiteness() {
        let mut dag = dag(1);
        let terminal = dag.terminal(0).unwrap();
        let groups = Groups {
            quotient_count: 1,
            groups: vec![(vec![0], 1)],
        };
        for weight in [1.0, 1.0e10, 1.0e100, f64::MAX] {
            assert!(
                certify_supersolution(&dag, &[terminal], &[0], &groups, &[weight], None).is_err()
            );
        }
        // One proper action does not justify a certificate when another action
        // can remain in a closed class forever.
        let closed_maximum = dag.maximum(&[dag.zero(), terminal]).unwrap();
        assert!(
            certify_supersolution(&dag, &[closed_maximum], &[0], &groups, &[16.0], None).is_err()
        );
        // A contraction close to one turns a small relative residual into a
        // large absolute error. Repeating [254/255] four times gives an exact
        // rational fixed point and demonstrates the required conditioning term.
        let mut root = terminal;
        for _ in 0..4 {
            root = dag.average(&vec![root; 254], 255).unwrap();
        }
        let ratio = (254.0f64 / 255.0).powi(4);
        let exact = 1.0 / (1.0 - ratio);
        let certificate =
            certify_supersolution(&dag, &[root], &[0], &groups, &[2.0 * exact], None).unwrap();
        let candidate = exact - 1.0e-5;
        let bound =
            residual_error_bound(&dag, &[root], &[0], &groups, &certificate, &[candidate]).unwrap();
        assert!(bound.residual_inf_upper < 2.0e-7);
        assert!(bound.absolute_error_upper > 1.0e-5);
    }

    #[test]
    fn weighted_certificate_handles_unweighted_row_sum_one() {
        let mut dag = dag(2);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let a4 = dag.average(&[a], 4).unwrap();
        let groups = Groups {
            quotient_count: 2,
            groups: vec![(vec![0], 1), (vec![1], 1)],
        };
        let certificate =
            certify_supersolution(&dag, &[b, a4], &[0, 1], &groups, &[8.0, 4.0], None).unwrap();
        assert!(certificate.contraction_factor_upper < 0.51);
        assert!(certificate.contraction_gap_lower > 0.49);
        contains(certificate.terminal_bounds[0], 8.0 / 3.0);
        contains(certificate.terminal_bounds[1], 5.0 / 3.0);
        let error =
            residual_error_bound(&dag, &[b, a4], &[0, 1], &groups, &certificate, &[2.0, 1.0])
                .unwrap();
        assert!(error.absolute_error_upper >= 2.0 / 3.0);
    }

    #[test]
    fn subnormal_averages_remain_outward() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let half = dag.average(&[a], 2).unwrap();
        let tiny = f64::from_bits(1);
        let values = evaluate_intervals(&dag, &[Interval::point(tiny).unwrap()]).unwrap();
        assert_eq!(values[half as usize].lower, 0.0);
        assert!(values[half as usize].upper >= tiny);
        let triple = f64::from_bits(3);
        let values = evaluate_intervals(&dag, &[Interval::point(triple).unwrap()]).unwrap();
        assert!(values[half as usize].lower <= tiny);
        assert!(values[half as usize].upper >= f64::from_bits(2));
    }

    #[test]
    fn tiny_continuation_retains_the_analytic_all_ones_lower_bound() {
        let mut dag = dag(1);
        let mut root = dag.terminal(0).unwrap();
        for _ in 0..7 {
            root = dag.average(&[root], 255).unwrap();
        }
        let groups = Groups {
            quotient_count: 1,
            groups: vec![(vec![0], 1)],
        };
        let certificate =
            certify_supersolution(&dag, &[root], &[0], &groups, &[2.0], None).unwrap();
        assert_eq!(certificate.terminal_bounds[0].lower, 1.0);
        let h = evaluate_h_intervals(
            &dag,
            &[root],
            &[0],
            &groups,
            &[Interval::point(1.0).unwrap()],
        )
        .unwrap();
        assert_eq!(h[0].lower, 1.0);
    }

    #[test]
    fn reveal_average_retains_missing_mass_and_observed_maxima() {
        let mut dag = dag(2);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let a4 = dag.average(&[a], 4).unwrap();
        let a8 = dag.average(&[a], 8).unwrap();
        let b4 = dag.average(&[b], 4).unwrap();
        let b8 = dag.average(&[b], 8).unwrap();
        let first = dag.maximum(&[a4, b8]).unwrap();
        let second = dag.maximum(&[a8, b4]).unwrap();
        let groups = Groups {
            quotient_count: 3,
            groups: vec![(vec![0, 1], 2), (vec![2], 2)],
        };
        let h = evaluate_h_intervals(
            &dag,
            &[first, second, a],
            &[0, 1, 2],
            &groups,
            &[Interval::point(2.0).unwrap(); 2],
        )
        .unwrap();
        contains(h[0], 1.5);
        assert!(h[0].lower > 1.375); // max of action averages would be wrong.
        contains(h[1], 2.0); // one missing reveal retains denominator two.
        let reordered = evaluate_h_intervals(
            &dag,
            &[a, second, first],
            &[2, 1, 0],
            &groups,
            &[Interval::point(2.0).unwrap(); 2],
        )
        .unwrap();
        assert_eq!(reordered, h); // Explicit qids, never assumed file offsets.
        let missing_root = evaluate_h_intervals(
            &dag,
            &[first, second],
            &[0, 1],
            &groups,
            &[Interval::point(2.0).unwrap(); 2],
        )
        .unwrap();
        assert_eq!(missing_root[1], Interval::point(1.0).unwrap());
    }

    #[test]
    fn invalid_bounds_mapping_and_overflow_are_rejected() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let sum = dag.average(&[a, a], 2).unwrap();
        assert!(evaluate_intervals(
            &dag,
            &[Interval {
                lower: -1.0,
                upper: 1.0
            }]
        )
        .is_err());
        assert!(evaluate_intervals(
            &dag,
            &[Interval {
                lower: 2.0,
                upper: 1.0
            }]
        )
        .is_err());
        assert!(evaluate_intervals(
            &dag,
            &[Interval {
                lower: 0.0,
                upper: f64::INFINITY
            }]
        )
        .is_err());
        assert!(evaluate_intervals(&dag, &[Interval::point(f64::MAX).unwrap()]).is_err());
        let groups = Groups {
            quotient_count: 1,
            groups: vec![(vec![0], 1)],
        };
        assert!(evaluate_h_intervals(
            &dag,
            &[sum, sum],
            &[0, 0],
            &groups,
            &[Interval::point(1.0).unwrap()]
        )
        .is_err());
        assert!(evaluate_h_intervals(
            &dag,
            &[sum],
            &[2],
            &groups,
            &[Interval::point(1.0).unwrap()]
        )
        .is_err());
        assert!(certify_supersolution(&dag, &[a], &[0], &groups, &[f64::NAN], None).is_err());
    }
}
