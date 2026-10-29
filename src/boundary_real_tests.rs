//! Explicit, bounded validation of compiled expressions on original pruned data.
//!
//! Run with `--release --ignored --nocapture` and the same explicit
//! ZXCL_VALIDATION_{GRAPH,PRUNED,POLICY,VALUES} paths as the older real-data
//! checks. Source files are opened read-only. Temporary membership contains
//! only sampled roots; descendants always use the original membership.
//! No complete next-value layer, including L7, is allocated or written.

use super::*;
use crate::boundary_dag::{CompileLimits, CompileMode, CompiledDag, Compiler, StateKey};
use crate::expression_dag::Node;
use crate::layer_store::FieldChainLayer;
use crate::pre_reveal::PreRevealIndex;
use std::ops::{AddAssign, Div};

fn required_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("explicit boundary real-data validation requires {name}"))
}

fn bounded_setting(name: &str, default: usize, maximum: usize) -> Result<usize> {
    let value = env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .parse()?;
    anyhow::ensure!(
        (1..=maximum).contains(&value),
        "{name} must be 1..={maximum}"
    );
    Ok(value)
}

fn u24(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | u32::from(bytes[1]) << 8 | u32::from(bytes[2]) << 16
}

#[derive(Clone, Debug)]
struct SampleRoot {
    state: StateKey,
    original_index: u64,
    policy: u16,
}

/// One non-NONE state per spaced field interval. An empty interval is skipped
/// after at most sixteen probes, so sampling cannot turn into a full layer scan.
fn spaced_roots(
    layer: usize,
    original: &FieldChainLayer,
    policy: &File,
    wanted: usize,
) -> Result<Vec<SampleRoot>> {
    let intervals = wanted.min(original.metas.len());
    anyhow::ensure!(intervals > 0, "layer {layer} has no field blocks");
    let membership = File::open(&original.chains_path)?;
    let mut roots = Vec::new();
    let mut raw_policy = Vec::new();
    for slot in 0..intervals {
        let lo = slot * original.metas.len() / intervals;
        let hi = (slot + 1) * original.metas.len() / intervals;
        let mut selected = None;
        for probe in 0..(hi - lo).min(16) {
            let block = lo + ((hi - lo) / 2 + probe) % (hi - lo);
            let meta = &original.metas[block];
            anyhow::ensure!(
                meta.count as usize <= CHECK_QSTATE_COUNT,
                "oversized membership block"
            );
            if meta.count == 0 {
                continue;
            }
            raw_policy.resize(meta.count as usize * 2, 0);
            read_exact_at_file(policy, &mut raw_policy, meta.chain_offset * 2)?;
            let start = (slot * 7919 + probe * 104729) % meta.count as usize;
            for step in 0..meta.count as usize {
                let row = (start + step) % meta.count as usize;
                let code = u16::from_le_bytes([raw_policy[row * 2], raw_policy[row * 2 + 1]]);
                if code == POLICY_NONE_U16 {
                    continue;
                }
                let original_index = meta.chain_offset + row as u64;
                let mut raw_qid = [0; 3];
                read_exact_at_file(&membership, &mut raw_qid, original_index * 3)?;
                selected = Some(SampleRoot {
                    state: (layer, meta.field_id, u24(&raw_qid)),
                    original_index,
                    policy: code,
                });
                break;
            }
            if selected.is_some() {
                break;
            }
        }
        if let Some(root) = selected {
            roots.push(root);
        }
    }
    roots.sort_unstable_by_key(|root| root.state);
    anyhow::ensure!(
        !roots.is_empty(),
        "no non-NONE roots in bounded layer-{layer} sample"
    );
    Ok(roots)
}

struct ScalarBlock {
    layer: usize,
    block: usize,
    qids: Vec<u8>,
    policy: Vec<u8>,
    used: u64,
}

impl ScalarBlock {
    fn bytes(&self) -> usize {
        self.qids.len() + self.policy.len()
    }

    fn lookup(&self, qid: u32, fixed: bool) -> Option<u16> {
        let (mut lo, mut hi) = (0, self.qids.len() / 3);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match u24(&self.qids[mid * 3..mid * 3 + 3]).cmp(&qid) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    return Some(if fixed {
                        u16::from_le_bytes([self.policy[mid * 2], self.policy[mid * 2 + 1]])
                    } else {
                        0
                    })
                }
            }
        }
        None
    }
}

trait Scalar: Copy + PartialOrd + AddAssign + Div<Output = Self> {
    fn zero() -> Self;
    fn denominator(value: usize) -> Self;
    fn finite(self) -> bool;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn denominator(value: usize) -> Self {
        value as Self
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn denominator(value: usize) -> Self {
        value as Self
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}

/// Independent recursive evaluator: it has no expression nodes or state memo.
/// It derives reveals directly from qstates instead of reading QForwardGraph.
/// Only read-only membership/policy blocks are cached under a byte bound.
struct RecursiveScalar<'a> {
    graph: &'a FieldGraph,
    q_index: &'a QStateIndex,
    pre: &'a PreRevealIndex,
    layers: &'a [FieldChainLayer],
    membership: Vec<File>,
    policies: &'a [File],
    fixed: bool,
    two_line: u32,
    cache: Vec<ScalarBlock>,
    cache_bytes: usize,
    cache_limit: usize,
    clock: u64,
    calls: usize,
    call_limit: usize,
}

impl<'a> RecursiveScalar<'a> {
    fn new(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        pre: &'a PreRevealIndex,
        layers: &'a [FieldChainLayer],
        policies: &'a [File],
        mode: CompileMode,
        two_line: u32,
        cache_limit: usize,
        call_limit: usize,
    ) -> Result<Self> {
        let membership = layers
            .iter()
            .map(|layer| File::open(&layer.chains_path))
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Self {
            graph,
            q_index,
            pre,
            layers,
            membership,
            policies,
            fixed: mode == CompileMode::Fixed,
            two_line,
            cache: Vec::new(),
            cache_bytes: 0,
            cache_limit,
            clock: 0,
            calls: 0,
            call_limit,
        })
    }

    fn membership(&mut self, layer: usize, field: u32, qid: u32) -> Result<Option<u16>> {
        let Ok(block) = self.layers[layer]
            .metas
            .binary_search_by_key(&field, |m| m.field_id)
        else {
            return Ok(None);
        };
        self.clock += 1;
        if let Some(entry) = self
            .cache
            .iter_mut()
            .find(|e| e.layer == layer && e.block == block)
        {
            entry.used = self.clock;
            return Ok(entry.lookup(qid, self.fixed));
        }
        let meta = &self.layers[layer].metas[block];
        let qbytes = meta.count as usize * 3;
        let pbytes = if self.fixed {
            meta.count as usize * 2
        } else {
            0
        };
        let bytes = qbytes + pbytes;
        if bytes > self.cache_limit {
            // Positional search keeps even an unusually large field bounded.
            let (mut lo, mut hi) = (0u64, u64::from(meta.count));
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let index = meta.chain_offset + mid;
                let mut raw = [0; 3];
                read_exact_at_file(&self.membership[layer], &mut raw, index * 3)?;
                match u24(&raw).cmp(&qid) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => {
                        let mut code = [0; 2];
                        if self.fixed {
                            read_exact_at_file(&self.policies[layer], &mut code, index * 2)?;
                        }
                        return Ok(Some(u16::from_le_bytes(code)));
                    }
                }
            }
            return Ok(None);
        }
        while self.cache_bytes + bytes > self.cache_limit {
            let oldest = self
                .cache
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .unwrap()
                .0;
            self.cache_bytes -= self.cache.swap_remove(oldest).bytes();
        }
        let mut entry = ScalarBlock {
            layer,
            block,
            qids: vec![0; qbytes],
            policy: vec![0; pbytes],
            used: self.clock,
        };
        read_exact_at_file(
            &self.membership[layer],
            &mut entry.qids,
            meta.chain_offset * 3,
        )?;
        if self.fixed {
            read_exact_at_file(
                &self.policies[layer],
                &mut entry.policy,
                meta.chain_offset * 2,
            )?;
        }
        let value = entry.lookup(qid, self.fixed);
        self.cache_bytes += bytes;
        self.cache.push(entry);
        Ok(value)
    }

    fn apply<T: Scalar>(&mut self, roots: &[StateKey], terminals: &[T]) -> Result<Vec<T>> {
        anyhow::ensure!(terminals.len() == self.pre.len());
        anyhow::ensure!(terminals.iter().all(|value| value.finite()));
        self.calls = 0;
        roots
            .iter()
            .map(|&root| self.state(root, terminals))
            .collect()
    }

    fn state<T: Scalar>(&mut self, key: StateKey, terminals: &[T]) -> Result<T> {
        self.calls += 1;
        anyhow::ensure!(
            self.calls <= self.call_limit,
            "uncached scalar call cap {} exceeded; reduce ZXCL_BOUNDARY_L7_ROOTS/L8_ROOTS",
            self.call_limit
        );
        let (layer, field, qid) = key;
        let Some(code) = self.membership(layer, field, qid)? else {
            return Ok(T::zero());
        };
        if self.fixed {
            if code == POLICY_NONE_U16 {
                return Ok(T::zero());
            }
            let placed = (code >> 8) as u8;
            anyhow::ensure!((placed as usize) < PIECE_COUNT, "invalid fixed placement");
            let target = *self
                .graph
                .edges(field, placed)
                .get((code & 255) as usize)
                .context("fixed policy edge outside graph")?;
            return self.action(key, placed, target, terminals);
        }
        let parent = self.q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(parent.avail);
        let mut best = T::zero();
        for placed in 0..PIECE_COUNT as u8 {
            if placed != a && placed != b {
                continue;
            }
            for edge in 0..self.graph.edges(field, placed).len() {
                let target = self.graph.edges(field, placed)[edge];
                if layer == 9 && target != EMPTY_FIELD_ID {
                    continue;
                }
                let candidate = self.action(key, placed, target, terminals)?;
                if candidate > best {
                    best = candidate;
                }
            }
        }
        Ok(best)
    }

    fn action<T: Scalar>(
        &mut self,
        key: StateKey,
        placed: u8,
        target: u32,
        terminals: &[T],
    ) -> Result<T> {
        let (layer, _, qid) = key;
        let parent = self.q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(parent.avail);
        anyhow::ensure!(placed == a || placed == b, "illegal fixed policy piece");
        if (layer == 4 && target == self.two_line) || (layer == 9 && target == EMPTY_FIELD_ID) {
            let zid = self
                .pre
                .zid_for_placement(qid, placed)
                .context("missing pre-reveal placement")?;
            return Ok(terminals[zid as usize]);
        }
        anyhow::ensure!(layer < 9, "nonterminal fixed L9 action");
        let mut sum = T::zero();
        let mut count = 0;
        // q_forward_after_reveal supplies the original legal reveal support.
        // Missing membership contributes zero and still increments count.
        for reveal in 0..PIECE_COUNT as u8 {
            let Some(child) = q_forward_after_reveal(parent, placed, reveal) else {
                continue;
            };
            let &child_qid = self
                .q_index
                .key_to_id
                .get(&qchain_key(child))
                .context("missing quotient child")?;
            sum += self.state((layer + 1, target, child_qid as u32), terminals)?;
            anyhow::ensure!(sum.finite(), "nonfinite recursive scalar sum");
            count += 1;
        }
        anyhow::ensure!(count > 0, "empty reveal support");
        Ok(sum / T::denominator(count))
    }
}

fn same_f64(expected: &[f64], actual: &[f64], label: &str) -> Result<()> {
    anyhow::ensure!(expected.len() == actual.len());
    for (i, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "{label} root={i}: f64 {a:?} ({:#018x}) != {b:?} ({:#018x})",
            a.to_bits(),
            b.to_bits()
        );
    }
    Ok(())
}

fn same_f32(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    anyhow::ensure!(expected.len() == actual.len());
    for (i, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "{label} root={i}: f32 {a:?} ({:#010x}) != {b:?} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
    Ok(())
}

fn root_f64(compiled: &CompiledDag, terminals: &[f64]) -> Result<Vec<f64>> {
    let values = compiled.dag.eval_f64(terminals)?;
    Ok(compiled
        .roots
        .iter()
        .map(|&node| values[node as usize])
        .collect())
}

fn root_f32(compiled: &CompiledDag, terminals: &[f32]) -> Result<Vec<f32>> {
    let values = compiled.dag.eval_f32(terminals)?;
    Ok(compiled
        .roots
        .iter()
        .map(|&node| values[node as usize])
        .collect())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn timing_line(compiled: &CompiledDag, label: &str, scalar: &mut [f64], eval: &mut [f64]) {
    let s = median(scalar);
    let e = median(eval);
    let b = compiled.stats.construction_seconds;
    if e < s {
        eprintln!("boundary-real {label} B_seconds={b:.9} S_seconds={s:.9} E_seconds={e:.9} break_even_applications={:.3} measured_sample_only=true", b / (s - e));
    } else {
        eprintln!("boundary-real {label} B_seconds={b:.9} S_seconds={s:.9} E_seconds={e:.9} break_even_applications=none measured_sample_only=true");
    }
}

/// ABBA compares applications after warmup. S is uncached recursive state
/// traversal with a bounded warm field cache; E evaluates all compiled nodes.
/// Both include terminal validation and output allocation, exclude setup/I/O
/// sinks, and apply exactly the same sampled root map.
fn compare_and_time(
    compiled: &CompiledDag,
    reference: &mut RecursiveScalar<'_>,
    roots: &[StateKey],
    terminals: &[f64],
    repetitions: usize,
    label: &str,
) -> Result<()> {
    let expected = reference.apply(roots, terminals)?;
    let calls = reference.calls;
    same_f64(&expected, &root_f64(compiled, terminals)?, label)?;
    let terminals32: Vec<f32> = terminals.iter().map(|&value| value as f32).collect();
    let expected32 = reference.apply(roots, &terminals32)?;
    same_f32(&expected32, &root_f32(compiled, &terminals32)?, label)?;
    eprintln!(
        "boundary-real {label} reference_state_calls={calls} reference_field_cache_bytes={}",
        reference.cache_bytes
    );
    let (mut s64, mut e64, mut s32, mut e32) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..repetitions {
        for dag in [false, true, true, false] {
            let began = Instant::now();
            let values = if dag {
                root_f64(compiled, terminals)?
            } else {
                reference.apply(roots, terminals)?
            };
            let seconds = began.elapsed().as_secs_f64();
            std::hint::black_box(&values);
            same_f64(&expected, &values, label)?;
            if dag {
                e64.push(seconds);
            } else {
                s64.push(seconds);
            }
            let began = Instant::now();
            let values = if dag {
                root_f32(compiled, &terminals32)?
            } else {
                reference.apply(roots, &terminals32)?
            };
            let seconds = began.elapsed().as_secs_f64();
            std::hint::black_box(&values);
            same_f32(&expected32, &values, label)?;
            if dag {
                e32.push(seconds);
            } else {
                s32.push(seconds);
            }
        }
    }
    timing_line(
        compiled,
        &format!("{label} precision=f64"),
        &mut s64,
        &mut e64,
    );
    timing_line(
        compiled,
        &format!("{label} precision=f32"),
        &mut s32,
        &mut e32,
    );
    Ok(())
}

fn terminal_cases(count: usize, mode: CompileMode) -> Vec<(&'static str, Vec<f64>)> {
    let varied = |positive: bool| {
        (0..count)
            .map(|i| {
                let mixed = (i as u64).wrapping_mul(6364136223846793005).rotate_left(23);
                if positive {
                    ((mixed % 2047 + 1) as f64) / 1024.0
                } else {
                    ((mixed % 2047) as f64 - 1023.0) / 1024.0
                }
            })
            .collect()
    };
    if mode == CompileMode::Fixed {
        vec![
            ("signed", varied(false)),
            ("all_negative", vec![-1.0; count]),
            (
                "alternating_scale",
                (0..count)
                    .map(|i| if i % 2 == 0 { 1024.0 } else { -1.0 / 1024.0 })
                    .collect(),
            ),
        ]
    } else {
        vec![
            ("positive_varying", varied(true)),
            ("ones_ties", vec![1.0; count]),
            ("zeros_none", vec![0.0; count]),
        ]
    }
}

fn create_output(path: &Path, bytes: u64) -> Result<File> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.set_len(bytes)?;
    Ok(file)
}

/// Invoke the existing scalar kernels with an injected pre-reveal terminal
/// table, rather than the old builder's differently rounded reveal average.
/// L8 uses the existing direct L9 lookup; no full L9 values are materialized.
fn compare_old_kernel(
    graph: &FieldGraph,
    q_index: &QStateIndex,
    qfwd: &QForwardGraph,
    pre: &PreRevealIndex,
    layers: &[FieldChainLayer],
    policies: &[File],
    legal: &[[u8; 2]],
    sample: &[SampleRoot],
    compiled: &CompiledDag,
    mode: CompileMode,
    terminals: &[f64],
    two_line: u32,
    cache_bytes: usize,
    scratch: &Path,
    label: &str,
) -> Result<()> {
    let layer = sample[0].state.0;
    anyhow::ensure!(layer == 8 || layer == 9);
    let prefix = scratch.join("roots");
    let mut writer = layer_store::FieldChainWriter::create(&prefix)?;
    let mut policy_bytes = Vec::new();
    for root in sample {
        writer.push_pair_key((u64::from(root.state.1) << 32) | u64::from(root.state.2))?;
        policy_bytes.extend_from_slice(&root.policy.to_le_bytes());
    }
    writer.finish()?;
    let cur = FieldChainLayer::open_index_only(&prefix)?;
    let policy_path = scratch.join("root-policy.u16");
    fs::write(&policy_path, policy_bytes)?;
    let policy = File::open(&policy_path)?;
    let output_path = scratch.join("old-values.f32");
    let policy_out_path = scratch.join("old-policy.u16");
    let output = create_output(&output_path, cur.total_states * 4)?;
    let policy_out = create_output(&policy_out_path, cur.total_states * 2)?;
    let terminals32: Vec<f32> = terminals.iter().map(|&value| value as f32).collect();
    let mut table = vec![[0.0f32; PIECE_COUNT]; q_index.states.len()];
    for (qid, row) in table.iter_mut().enumerate() {
        for placed in 0..PIECE_COUNT as u8 {
            if let Some(zid) = pre.zid_for_placement(qid as u32, placed) {
                row[placed as usize] = terminals32[zid as usize];
            }
        }
    }
    let source = (layer == 8).then_some(terminal_l9::Source {
        layer: &layers[9],
        graph,
        qfwd,
        terminal_reset_avg: &table,
        mode: if mode == CompileMode::Fixed {
            terminal_l9::Mode::Fixed(&policies[9])
        } else {
            terminal_l9::Mode::Greedy(legal)
        },
    });
    let began = Instant::now();
    if mode == CompileMode::Fixed {
        compute_fixed_policy_chunk_f32_with_l9_source(
            layer,
            0..cur.metas.len(),
            &cur,
            &policy,
            Some(&output),
            None,
            graph,
            qfwd,
            &table,
            (layer == 8).then_some(&layers[9]),
            None,
            None,
            1.0,
            cache_bytes,
            q_index.states.len(),
            65_536,
            two_line,
            source,
        )?;
    } else {
        compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source(
            layer,
            &[0..cur.metas.len()],
            &cur,
            Some(&output),
            None,
            Some(&policy_out),
            graph,
            qfwd,
            legal,
            &table,
            (layer == 8).then_some(&layers[9]),
            None,
            None,
            None,
            1.0,
            cache_bytes,
            q_index.states.len(),
            65_536,
            two_line,
            false,
            false,
            false,
            false,
            source,
        )?;
    }
    let seconds = began.elapsed().as_secs_f64();
    let old_values: Vec<f32> = fs::read(&output_path)?
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    same_f32(&root_f32(compiled, &terminals32)?, &old_values, label)?;
    if mode == CompileMode::Greedy {
        let values = compiled.dag.eval_f32(&terminals32)?;
        let dag_policy: HashMap<_, _> = compiled.greedy_policy_f32(&values)?.into_iter().collect();
        let bytes = fs::read(&policy_out_path)?;
        for (root, raw) in sample.iter().zip(bytes.chunks_exact(2)) {
            let old_code = u16::from_le_bytes(raw.try_into().unwrap());
            anyhow::ensure!(
                dag_policy[&root.state] == old_code,
                "{label} greedy policy mismatch at {:?}",
                root.state
            );
        }
    }
    eprintln!("boundary-real {label} old_kernel_f32_bitwise=true old_kernel_seconds={seconds:.9} full_next_value_bytes=0");
    Ok(())
}

fn transpose_check(compiled: &CompiledDag, terminals: &[f64], label: &str) -> Result<()> {
    anyhow::ensure!(
        compiled
            .dag
            .nodes()
            .iter()
            .all(|node| !matches!(node, Node::Maximum(_))),
        "fixed DAG unexpectedly contains a maximum"
    );
    let weights: Vec<f64> = (0..compiled.roots.len())
        .map(|i| (i % 7) as f64 - 3.0)
        .collect();
    let values = root_f64(compiled, terminals)?;
    let transposed = compiled.dag.linear_transpose(&compiled.roots, &weights)?;
    let forward_dot: f64 = values.iter().zip(&weights).map(|(v, w)| v * w).sum();
    let transpose_dot: f64 = terminals.iter().zip(&transposed).map(|(v, w)| v * w).sum();
    let scale = 1.0 + forward_dot.abs() + transpose_dot.abs();
    anyhow::ensure!(
        (forward_dot - transpose_dot).abs() <= 1e-11 * scale,
        "{label} transpose dot-product mismatch {forward_dot} vs {transpose_dot}"
    );
    eprintln!(
        "boundary-real {label} transpose_dot_error={:.3e}",
        (forward_dot - transpose_dot).abs()
    );
    Ok(())
}

/// This checks pruning only at the supplied positive terminal point. Sampled
/// roots do not define the complete fixed-point operator or certify its optimum.
fn point_box_pruning_check(compiled: &CompiledDag, terminals: &[f64], label: &str) -> Result<()> {
    use crate::certified_bounds::{evaluate_intervals, prune, Interval, PruningValidity};

    let bounds = terminals
        .iter()
        .map(|&value| Interval::point(value))
        .collect::<Result<Vec<_>>>()?;
    let original_intervals = evaluate_intervals(&compiled.dag, &bounds)?;
    let pruned = prune(&compiled.dag, &compiled.roots, &bounds)?;
    anyhow::ensure!(pruned.validity == PruningValidity::SuppliedBox);
    anyhow::ensure!(pruned.removed_branches == pruned.proofs.len());
    anyhow::ensure!(
        pruned
            .proofs
            .iter()
            .all(|proof| proof.removed_upper < proof.witness_lower),
        "{label} pruning removed a child without strict interval dominance"
    );
    let original = compiled.dag.eval_f64(terminals)?;
    let rebuilt = pruned.dag.eval_f64(terminals)?;
    let mut differing_roots = 0;
    let mut largest_difference = 0.0f64;
    for (&original_root, &pruned_root) in compiled.roots.iter().zip(&pruned.roots) {
        let a = original[original_root as usize];
        let b = rebuilt[pruned_root as usize];
        let interval = original_intervals[original_root as usize];
        anyhow::ensure!(
            interval.lower <= a && a <= interval.upper,
            "{label} original f64 root lies outside its outward interval"
        );
        anyhow::ensure!(
            interval.lower <= b && b <= interval.upper,
            "{label} pruned f64 root lies outside the original outward interval"
        );
        differing_roots += usize::from(a.to_bits() != b.to_bits());
        largest_difference = largest_difference.max((a - b).abs());
    }
    eprintln!("boundary-real {label} prune_validity=supplied_point_only original_nodes={} original_edges={} pruned_nodes={} pruned_edges={} removed_branches={} f64_differing_roots={differing_roots} largest_f64_difference={largest_difference:.3e}",
        pruned.original_node_count, pruned.original_edge_count,
        pruned.dag.node_count(), pruned.dag.edge_count(), pruned.removed_branches);
    Ok(())
}

#[test]
#[ignore = "bounded boundary-DAG real-data check; requires explicit ZXCL_VALIDATION_* paths and release mode"]
fn bounded_real_data_boundary_dag_matches_recursive_and_old_kernels() -> Result<()> {
    let _lock = TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run this bounded benchmark in release mode"
    );
    let graph_path = required_path("ZXCL_VALIDATION_GRAPH")?;
    let pruned = required_path("ZXCL_VALIDATION_PRUNED")?;
    let policies_path = required_path("ZXCL_VALIDATION_POLICY")?;
    let values_path = required_path("ZXCL_VALIDATION_VALUES")?;
    let roots7 = bounded_setting("ZXCL_BOUNDARY_L7_ROOTS", 8, 32)?;
    let roots8 = bounded_setting("ZXCL_BOUNDARY_L8_ROOTS", 32, 64)?;
    let repetitions = bounded_setting("ZXCL_VALIDATION_REPETITIONS", 3, 5)?;
    let prune_point = match env::var("ZXCL_BOUNDARY_PRUNE").as_deref() {
        Ok("0") => false,
        Ok("1") | Err(env::VarError::NotPresent) => true,
        _ => anyhow::bail!("ZXCL_BOUNDARY_PRUNE must be 0 or 1"),
    };
    let limits = CompileLimits {
        max_states: bounded_setting("ZXCL_BOUNDARY_MAX_STATES", 1_000_000, 1_000_000)?,
        max_nodes: bounded_setting("ZXCL_BOUNDARY_MAX_NODES", 1_000_000, 1_000_000)?,
        max_edges: bounded_setting("ZXCL_BOUNDARY_MAX_EDGES", 8_000_000, 8_000_000)?,
        field_cache_bytes: bounded_setting("ZXCL_BOUNDARY_CACHE_MIB", 32, 128)? * 1024 * 1024,
    };
    let scalar_calls = bounded_setting("ZXCL_BOUNDARY_SCALAR_CALLS", 5_000_000, 20_000_000)?;
    let setup = Instant::now();
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let q_index = build_q_index_from_old(&enumerate_states());
    anyhow::ensure!(q_index.states.len() == CHECK_QSTATE_COUNT);
    let pre = PreRevealIndex::build(&q_index)?;
    let qfwd = build_qforward_graph(&q_index);
    let legal = build_qfwd_legal_placed_table(&qfwd, q_index.states.len());
    let layers = (0..10)
        .map(|layer| FieldChainLayer::open_index_only(q_pruned_prefix(&pruned, layer)))
        .collect::<Result<Vec<_>>>()?;
    let policies = (0..10)
        .map(|layer| -> Result<File> {
            let file = File::open(mdp_policy_u16_path(&policies_path, layer))?;
            anyhow::ensure!(
                file.metadata()?.len() == layers[layer].total_states * 2,
                "layer {layer} policy/membership size mismatch"
            );
            Ok(file)
        })
        .collect::<Result<Vec<_>>>()?;
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("two-line field missing")?;
    let scratch = GreedyEmitScratch::create(&env::temp_dir())?;
    eprintln!("boundary-real setup_seconds={:.6} qstates={} pre_reveal_states={} limits={limits:?} scalar_call_cap={scalar_calls}",
        setup.elapsed().as_secs_f64(), q_index.states.len(), pre.len());
    for (layer, wanted) in [(9, roots8.min(32)), (8, roots8), (7, roots7)] {
        let sample = spaced_roots(layer, &layers[layer], &policies[layer], wanted)?;
        let roots: Vec<StateKey> = sample.iter().map(|root| root.state).collect();
        // Validate snapshot shape and finite sampled values with positional
        // probes. This does not prove which solver inputs produced a snapshot.
        // Synthetic terminal inputs below intentionally define a different map.
        let snapshot = File::open(mdp_f32_value_path(&values_path, layer))?;
        anyhow::ensure!(
            snapshot.metadata()?.len() == layers[layer].total_states * 4,
            "layer {layer} value/membership size mismatch"
        );
        for root in &sample {
            let mut raw = [0; 4];
            read_exact_at_file(&snapshot, &mut raw, root.original_index * 4)?;
            anyhow::ensure!(
                f32::from_le_bytes(raw).is_finite(),
                "nonfinite sampled snapshot value"
            );
        }
        eprintln!("boundary-real layer={layer} sampled_roots={} original_states={} snapshot_probe_bytes={} first_root={:?} last_root={:?}",
            roots.len(), layers[layer].total_states, roots.len() * 4, roots.first(), roots.last());
        for mode in [CompileMode::Fixed, CompileMode::Greedy] {
            let compiled = Compiler::new(
                &graph,
                &q_index,
                &qfwd,
                &layers,
                &pre,
                Some(&policies_path),
                mode,
                limits,
                two_line,
            )?
            .compile_roots(&roots)?;
            anyhow::ensure!(
                compiled.stats.terminal_actions > 0,
                "sample does not reach a terminal; choose a different dataset/sample size"
            );
            eprintln!(
                "boundary-real layer={layer} mode={mode:?} stats={:?}",
                compiled.stats
            );
            let mut reference = RecursiveScalar::new(
                &graph,
                &q_index,
                &pre,
                &layers,
                &policies,
                mode,
                two_line,
                limits.field_cache_bytes,
                scalar_calls,
            )?;
            for (case, terminals) in terminal_cases(pre.len(), mode) {
                let label = format!("layer={layer} mode={mode:?} case={case}");
                compare_and_time(
                    &compiled,
                    &mut reference,
                    &roots,
                    &terminals,
                    repetitions,
                    &label,
                )?;
                if mode == CompileMode::Fixed {
                    transpose_check(&compiled, &terminals, &label)?;
                }
                if prune_point && mode == CompileMode::Greedy && case == "positive_varying" {
                    point_box_pruning_check(&compiled, &terminals, &label)?;
                }
                if layer >= 8 {
                    let case_path = scratch.path.join(format!("layer-{layer}-{mode:?}-{case}"));
                    fs::create_dir(&case_path)?;
                    compare_old_kernel(
                        &graph,
                        &q_index,
                        &qfwd,
                        &pre,
                        &layers,
                        &policies,
                        &legal,
                        &sample,
                        &compiled,
                        mode,
                        &terminals,
                        two_line,
                        limits.field_cache_bytes,
                        &case_path,
                        &label,
                    )?;
                }
            }
        }
    }
    Ok(())
}
