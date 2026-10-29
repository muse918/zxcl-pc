//! Structurally shared expressions with the original floating-point order intact.
//!
//! Node ids are in topological order. Interning preserves child order, repeated
//! children, explicit zero children, and the full average denominator; it never
//! simplifies, sorts, flattens, or combines expressions by their numeric value.
//! An omitted average child contributes zero without reducing the denominator.
//!
//! `max_nodes` includes the zero node; `max_edges` counts child occurrences in
//! unique nodes, including repeated children. These are hard logical caps, not a
//! byte-memory limit. The prototype keeps each node both in the arena and as a
//! hash-map key, so child storage is duplicated. Capacity slack, hash-table
//! overhead, a temporary candidate node, and allocator overhead require further
//! headroom. Forward evaluation uses one value per node; transpose additionally
//! uses one bool and one f64 per node, a traversal stack, and one f64 per terminal.

use anyhow::{bail, ensure, Result};
use std::collections::HashMap;

pub type NodeId = u32;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Node {
    Zero,
    Terminal(u32),
    Average {
        children: Vec<NodeId>,
        denominator: u8,
    },
    Maximum(Vec<NodeId>),
}

#[derive(Clone, Debug)]
pub struct ExprDag {
    nodes: Vec<Node>,
    interned: HashMap<Node, NodeId>,
    terminal_count: usize,
    edge_count: usize,
    max_nodes: usize,
    max_edges: usize,
}

impl ExprDag {
    /// Create an arena containing only zero. Terminal nodes are added lazily.
    pub fn new(terminal_count: usize, max_nodes: usize, max_edges: usize) -> Result<Self> {
        ensure!(
            max_nodes > 0,
            "expression DAG node cap must include the zero node"
        );
        ensure!(
            u32::try_from(max_nodes - 1).is_ok(),
            "expression DAG node cap exceeds the NodeId address space"
        );
        ensure!(
            u32::try_from(terminal_count.saturating_sub(1)).is_ok(),
            "expression DAG terminal count exceeds the terminal index address space"
        );
        let mut interned = HashMap::new();
        interned.insert(Node::Zero, 0);
        Ok(Self {
            nodes: vec![Node::Zero],
            interned,
            terminal_count,
            edge_count: 0,
            max_nodes,
            max_edges,
        })
    }

    #[inline]
    pub fn zero(&self) -> NodeId {
        0
    }

    pub fn terminal(&mut self, index: usize) -> Result<NodeId> {
        ensure!(
            index < self.terminal_count,
            "terminal index {index} is outside terminal count {}",
            self.terminal_count
        );
        self.intern(Node::Terminal(u32::try_from(index)?), 0)
    }

    /// Preserve the supplied addition order and divide by the full denominator.
    /// Missing children carry zero mass. Explicit zero children are retained.
    pub fn average(&mut self, children: &[NodeId], denominator: usize) -> Result<NodeId> {
        ensure!(
            denominator > 0 && denominator <= u8::MAX as usize,
            "average denominator must be in 1..=255, got {denominator}"
        );
        ensure!(
            children.len() <= denominator,
            "average has {} children but denominator is {denominator}",
            children.len()
        );
        self.validate_children(children)?;
        self.intern(
            Node::Average {
                children: children.to_vec(),
                denominator: denominator as u8,
            },
            children.len(),
        )
    }

    /// Preserve choice order. Evaluation retains the first child on a tie.
    /// Include the zero node explicitly when zero is a legal baseline choice.
    pub fn maximum(&mut self, children: &[NodeId]) -> Result<NodeId> {
        ensure!(!children.is_empty(), "maximum requires at least one child");
        self.validate_children(children)?;
        self.intern(Node::Maximum(children.to_vec()), children.len())
    }

    fn validate_children(&self, children: &[NodeId]) -> Result<()> {
        // A resident expression cannot have more edges than the entire budget.
        // Check this before allocating a candidate key, including on reuse.
        ensure!(
            children.len() <= self.max_edges,
            "expression requires {} edges, exceeding edge cap {}",
            children.len(),
            self.max_edges
        );
        for &child in children {
            ensure!(
                (child as usize) < self.nodes.len(),
                "child node {child} does not exist (node count {})",
                self.nodes.len()
            );
        }
        Ok(())
    }

    fn intern(&mut self, node: Node, edges: usize) -> Result<NodeId> {
        if let Some(&id) = self.interned.get(&node) {
            return Ok(id);
        }
        ensure!(
            self.nodes.len() < self.max_nodes,
            "expression DAG node cap {} exceeded",
            self.max_nodes
        );
        let edge_count = self
            .edge_count
            .checked_add(edges)
            .ok_or_else(|| anyhow::anyhow!("expression DAG edge count overflow"))?;
        ensure!(
            edge_count <= self.max_edges,
            "expression DAG edge cap {} exceeded (would need {edge_count})",
            self.max_edges
        );
        let id = NodeId::try_from(self.nodes.len())?;
        // All validation and cap checks precede changes to either collection.
        self.nodes.push(node.clone());
        self.interned.insert(node, id);
        self.edge_count = edge_count;
        Ok(id)
    }

    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    #[inline]
    pub fn terminal_count(&self) -> usize {
        self.terminal_count
    }

    /// Evaluate every node with ordered f32 operations. Reject non-finite input
    /// and any non-finite intermediate sum, even if later terms would cancel it.
    pub fn eval_f32(&self, terminals: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            terminals.len() == self.terminal_count,
            "expected {} terminal values, got {}",
            self.terminal_count,
            terminals.len()
        );
        for (index, value) in terminals.iter().enumerate() {
            ensure!(value.is_finite(), "terminal {index} is not finite");
        }
        let mut values = Vec::<f32>::with_capacity(self.nodes.len());
        for (id, node) in self.nodes.iter().enumerate() {
            let value = match node {
                Node::Zero => 0.0,
                Node::Terminal(index) => terminals[*index as usize],
                Node::Average {
                    children,
                    denominator,
                } => {
                    let mut sum = 0.0f32;
                    for &child in children {
                        sum += values[child as usize];
                        ensure!(sum.is_finite(), "f32 sum overflow at node {id}");
                    }
                    sum / *denominator as f32
                }
                Node::Maximum(children) => {
                    let mut best = values[children[0] as usize];
                    for &child in &children[1..] {
                        let candidate = values[child as usize];
                        if candidate > best {
                            best = candidate;
                        }
                    }
                    best
                }
            };
            ensure!(value.is_finite(), "non-finite f32 value at node {id}");
            values.push(value);
        }
        Ok(values)
    }

    /// Evaluate every node with ordered f64 operations and finite checks.
    pub fn eval_f64(&self, terminals: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            terminals.len() == self.terminal_count,
            "expected {} terminal values, got {}",
            self.terminal_count,
            terminals.len()
        );
        for (index, value) in terminals.iter().enumerate() {
            ensure!(value.is_finite(), "terminal {index} is not finite");
        }
        let mut values = Vec::<f64>::with_capacity(self.nodes.len());
        for (id, node) in self.nodes.iter().enumerate() {
            let value = match node {
                Node::Zero => 0.0,
                Node::Terminal(index) => terminals[*index as usize],
                Node::Average {
                    children,
                    denominator,
                } => {
                    let mut sum = 0.0f64;
                    for &child in children {
                        sum += values[child as usize];
                        ensure!(sum.is_finite(), "f64 sum overflow at node {id}");
                    }
                    sum / *denominator as f64
                }
                Node::Maximum(children) => {
                    let mut best = values[children[0] as usize];
                    for &child in &children[1..] {
                        let candidate = values[child as usize];
                        if candidate > best {
                            best = candidate;
                        }
                    }
                    best
                }
            };
            ensure!(value.is_finite(), "non-finite f64 value at node {id}");
            values.push(value);
        }
        Ok(values)
    }

    /// Apply the transpose of the linear map from terminals to `roots`.
    ///
    /// Every node reachable from a root must be linear, including roots with
    /// zero weight. Unreachable maximum nodes are permitted. Root weights are
    /// accumulated in their supplied order, then adjoints are propagated in
    /// reverse node order and each average's stored child order. Floating-point
    /// roundoff need not match forward dot products bit for bit.
    pub fn linear_transpose(&self, roots: &[NodeId], weights: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            roots.len() == weights.len(),
            "transpose has {} roots but {} weights",
            roots.len(),
            weights.len()
        );
        for (index, (&root, &weight)) in roots.iter().zip(weights).enumerate() {
            ensure!(
                (root as usize) < self.nodes.len(),
                "transpose root {index} refers to missing node {root}"
            );
            ensure!(weight.is_finite(), "transpose weight {index} is not finite");
        }

        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        // Mark when enqueuing so repeated/shared edges do not grow the stack
        // beyond one entry per node.
        for &root in roots {
            if !reachable[root as usize] {
                reachable[root as usize] = true;
                stack.push(root);
            }
        }
        while let Some(id) = stack.pop() {
            match &self.nodes[id as usize] {
                Node::Maximum(_) => bail!("linear transpose reaches maximum node {id}"),
                Node::Average { children, .. } => {
                    for &child in children {
                        if !reachable[child as usize] {
                            reachable[child as usize] = true;
                            stack.push(child);
                        }
                    }
                }
                Node::Zero | Node::Terminal(_) => {}
            }
        }

        let mut adjoints = vec![0.0f64; self.nodes.len()];
        for (&root, &weight) in roots.iter().zip(weights) {
            let adjoint = &mut adjoints[root as usize];
            *adjoint += weight;
            ensure!(
                adjoint.is_finite(),
                "transpose root accumulation overflow at node {root}"
            );
        }
        let mut terminals = vec![0.0f64; self.terminal_count];
        for id in (0..self.nodes.len()).rev() {
            if !reachable[id] {
                continue;
            }
            match &self.nodes[id] {
                Node::Zero => {}
                Node::Terminal(index) => {
                    let terminal = &mut terminals[*index as usize];
                    *terminal += adjoints[id];
                    ensure!(
                        terminal.is_finite(),
                        "transpose terminal accumulation overflow at node {id}"
                    );
                }
                Node::Average {
                    children,
                    denominator,
                } => {
                    let contribution = adjoints[id] / *denominator as f64;
                    for &child in children {
                        let adjoint = &mut adjoints[child as usize];
                        *adjoint += contribution;
                        ensure!(
                            adjoint.is_finite(),
                            "transpose accumulation overflow at node {child}"
                        );
                    }
                }
                Node::Maximum(_) => {
                    unreachable!("reachable maxima were rejected before propagation")
                }
            }
        }
        Ok(terminals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dag(terminals: usize) -> ExprDag {
        ExprDag::new(terminals, 1024, 4096).unwrap()
    }

    #[test]
    fn shares_only_exact_ordered_structure() {
        let mut dag = dag(2);
        assert_eq!(dag.zero(), 0);
        assert_eq!(dag.nodes(), &[Node::Zero]);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        assert_eq!(a, dag.terminal(0).unwrap());
        let ab = dag.average(&[a, b], 2).unwrap();
        assert_eq!(ab, dag.average(&[a, b], 2).unwrap());
        assert_ne!(ab, dag.average(&[b, a], 2).unwrap());
        assert_ne!(ab, dag.average(&[a, b], 3).unwrap());
        assert_ne!(a, dag.average(&[a], 1).unwrap());
        assert_ne!(a, dag.average(&[a, a], 2).unwrap());
        assert_ne!(ab, dag.average(&[ab], 1).unwrap());
        let empty = dag.average(&[], 2).unwrap();
        assert_ne!(empty, dag.zero());
        assert_ne!(empty, dag.average(&[], 3).unwrap());
        assert_ne!(empty, dag.average(&[dag.zero()], 2).unwrap());
        let max = dag.maximum(&[a, b]).unwrap();
        assert_eq!(max, dag.maximum(&[a, b]).unwrap());
        assert_ne!(max, dag.maximum(&[b, a]).unwrap());
        assert_ne!(a, dag.maximum(&[a]).unwrap());
    }

    #[test]
    fn ordered_addition_preserves_adversarial_f32_and_f64_results() {
        let mut dag = dag(3);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let c = dag.terminal(2).unwrap();
        let abc = dag.average(&[a, b, c], 3).unwrap();
        let acb = dag.average(&[a, c, b], 3).unwrap();
        let f32_values = dag.eval_f32(&[16_777_216.0, 1.0, -16_777_216.0]).unwrap();
        assert_eq!(f32_values[abc as usize].to_bits(), 0.0f32.to_bits());
        assert_eq!(f32_values[acb as usize].to_bits(), (1.0f32 / 3.0).to_bits());
        let f64_values = dag
            .eval_f64(&[9_007_199_254_740_992.0, 1.0, -9_007_199_254_740_992.0])
            .unwrap();
        assert_eq!(f64_values[abc as usize].to_bits(), 0.0f64.to_bits());
        assert_eq!(f64_values[acb as usize].to_bits(), (1.0f64 / 3.0).to_bits());
    }

    #[test]
    fn average_retains_absent_mass_and_duplicate_edges() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let root = dag.average(&[a, a], 3).unwrap();
        assert_eq!(dag.eval_f32(&[-6.0]).unwrap()[root as usize], -4.0);
        assert_eq!(dag.eval_f64(&[-6.0]).unwrap()[root as usize], -4.0);
        assert_eq!(dag.linear_transpose(&[root], &[3.0]).unwrap(), vec![2.0]);
        let empty = dag.average(&[], 7).unwrap();
        assert_eq!(dag.eval_f64(&[100.0]).unwrap()[empty as usize], 0.0);
        assert_eq!(dag.linear_transpose(&[empty], &[7.0]).unwrap(), vec![0.0]);
        assert_eq!(dag.edge_count(), 2);
    }

    #[test]
    fn maximum_supports_signed_inputs_and_retains_first_tie() {
        let mut dag = dag(2);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let max = dag.maximum(&[a, b]).unwrap();
        let reversed = dag.maximum(&[b, a]).unwrap();
        let with_zero = dag.maximum(&[dag.zero(), a, b]).unwrap();
        assert_eq!(dag.eval_f64(&[-4.0, -2.0]).unwrap()[max as usize], -2.0);
        assert_eq!(
            dag.eval_f32(&[-4.0, -2.0]).unwrap()[with_zero as usize],
            0.0
        );
        let values = dag.eval_f64(&[-0.0, 0.0]).unwrap();
        assert_eq!(values[max as usize].to_bits(), (-0.0f64).to_bits());
        assert_eq!(values[reversed as usize].to_bits(), 0.0f64.to_bits());
        let values = dag.eval_f32(&[-0.0, 0.0]).unwrap();
        assert_eq!(values[max as usize].to_bits(), (-0.0f32).to_bits());
        assert_eq!(values[reversed as usize].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn transpose_is_signed_linear_and_satisfies_dot_product_identity() {
        let mut dag = dag(3);
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let c = dag.terminal(2).unwrap();
        let shared = dag.average(&[a, b, a], 4).unwrap();
        let first = dag.average(&[shared, c], 2).unwrap();
        let second = dag.average(&[b, shared], 4).unwrap();
        let roots = [first, second, shared, first, dag.zero()];
        let weights = [-1.0, 2.0, -4.0, 0.5, 10.0];
        let x = [-2.0, 3.0, -5.0];
        let y = [1.0, -4.0, 2.0];
        let combined: Vec<f64> = x.iter().zip(y).map(|(x, y)| 2.0 * x - y).collect();
        let vx = dag.eval_f64(&x).unwrap();
        let vy = dag.eval_f64(&y).unwrap();
        let vc = dag.eval_f64(&combined).unwrap();
        for &root in &roots {
            assert_eq!(
                vc[root as usize],
                2.0 * vx[root as usize] - vy[root as usize]
            );
        }
        let transposed = dag.linear_transpose(&roots, &weights).unwrap();
        let forward_dot: f64 = roots
            .iter()
            .zip(weights)
            .map(|(&root, w)| vx[root as usize] * w)
            .sum();
        let transpose_dot: f64 = x.iter().zip(transposed).map(|(x, w)| x * w).sum();
        assert_eq!(forward_dot, transpose_dot);
        assert_eq!(dag.linear_transpose(&[], &[]).unwrap(), vec![0.0; 3]);
    }

    #[test]
    fn transpose_rejects_only_structurally_reachable_maxima() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let max = dag.maximum(&[a]).unwrap();
        let linear = dag.average(&[a], 2).unwrap();
        let nonlinear = dag.average(&[max], 2).unwrap();
        assert_eq!(
            dag.linear_transpose(&[linear], &[-2.0]).unwrap(),
            vec![-1.0]
        );
        assert!(dag.linear_transpose(&[max], &[0.0]).is_err());
        assert!(dag.linear_transpose(&[max, max], &[1.0, -1.0]).is_err());
        assert!(dag.linear_transpose(&[nonlinear], &[1.0]).is_err());
        assert_eq!(dag.linear_transpose(&[], &[]).unwrap(), vec![0.0]);
    }

    #[test]
    fn cap_failures_do_not_mutate_and_existing_nodes_remain_reusable() {
        let mut dag = ExprDag::new(2, 4, 2).unwrap();
        let a = dag.terminal(0).unwrap();
        let b = dag.terminal(1).unwrap();
        let root = dag.average(&[a, b], 2).unwrap();
        assert_eq!((dag.node_count(), dag.edge_count()), (4, 2));
        assert_eq!(root, dag.average(&[a, b], 2).unwrap());
        assert_eq!(a, dag.terminal(0).unwrap());
        assert!(dag.maximum(&[a]).is_err());
        assert!(dag.average(&[b, a], 2).is_err());
        assert_eq!((dag.node_count(), dag.edge_count()), (4, 2));
        let mut edges = ExprDag::new(1, 8, 1).unwrap();
        let a = edges.terminal(0).unwrap();
        let root = edges.average(&[a], 2).unwrap();
        assert!(edges.average(&[a], 3).is_err());
        assert_eq!((edges.node_count(), edges.edge_count()), (3, 1));
        assert_eq!(root, edges.average(&[a], 2).unwrap());
        // The rejected candidate must not have entered the interner.
        assert!(edges.average(&[a], 3).is_err());
        let mut zero_only = ExprDag::new(0, 1, 0).unwrap();
        assert!(zero_only.average(&[], 1).is_err());
        assert_eq!(zero_only.eval_f64(&[]).unwrap(), vec![0.0]);
    }

    #[test]
    fn malformed_inputs_and_nonfinite_values_are_rejected() {
        assert!(ExprDag::new(1, 0, 0).is_err());
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        assert!(dag.terminal(1).is_err());
        assert!(dag.average(&[], 0).is_err());
        assert!(dag.average(&[], 256).is_err());
        assert!(dag.average(&[a, a], 1).is_err());
        assert!(dag.average(&[99], 1).is_err());
        assert!(dag.maximum(&[]).is_err());
        assert!(dag.maximum(&[99]).is_err());
        assert!(dag.eval_f32(&[]).is_err());
        assert!(dag.eval_f64(&[1.0, 2.0]).is_err());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(dag.eval_f64(&[bad]).is_err());
            assert!(dag.linear_transpose(&[a], &[bad]).is_err());
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(dag.eval_f32(&[bad]).is_err());
        }
        assert!(dag.linear_transpose(&[a], &[]).is_err());
        assert!(dag.linear_transpose(&[99], &[1.0]).is_err());
        assert_eq!((dag.node_count(), dag.edge_count()), (2, 0));
        let all = dag.average(&vec![a; 255], 255).unwrap();
        assert_eq!(dag.eval_f64(&[1.0]).unwrap()[all as usize], 1.0);
        assert!(dag.average(&vec![a; 256], 255).is_err());
    }

    #[test]
    fn finite_inputs_with_overflowing_intermediates_are_rejected() {
        let mut dag = dag(1);
        let a = dag.terminal(0).unwrap();
        let average = dag.average(&[a, a], 2).unwrap();
        assert!(dag.eval_f32(&[f32::MAX]).is_err());
        assert!(dag.eval_f64(&[f64::MAX]).is_err());
        assert!(dag
            .linear_transpose(&[a, a], &[f64::MAX, f64::MAX])
            .is_err());
        assert!(dag
            .linear_transpose(&[average, a], &[f64::MAX, f64::MAX])
            .is_err());
    }
}
