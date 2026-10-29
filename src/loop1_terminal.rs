//! Shared terminal values with the legacy f32 operation order.
//!
//! A placement's pre-reveal ID identifies its ordered quotient children. The
//! terminal table repeats this same calculation for many `(qid, piece)` pairs.
//! This plan validates that identity once and retains only one list per ID.
//! The reward stays inside the sum: `mean(1 + x)` and `1 + mean(x)` can differ
//! in f32. This is terminal-table reuse, not a reduced boundary operator.

use anyhow::{bail, ensure, Context, Result};

use crate::piece::{FULL_BAG, PIECE_COUNT};
use crate::pre_reveal::{PreRevealIndex, TAIL4_COUNT};
use crate::qgraph::QForwardGraph;
use crate::qstate::{
    after_reveal_forward, decode_avail_pair, QStateIndex, AVAIL_PAIR_COUNT, TAIL5_COUNT,
};

#[derive(Debug)]
pub(crate) struct TerminalPlan {
    qstate_count: usize,
    reveal_offsets: Vec<u32>,
    reveal_children: Vec<u32>,
}

impl TerminalPlan {
    /// Validate complete, ordered reveal support for every qforward placement.
    /// The source index and graph may be dropped after construction; evaluation
    /// uses only these owned, checked lists.
    pub(crate) fn build(
        q_index: &QStateIndex,
        qfwd: &QForwardGraph,
        pre: &PreRevealIndex,
    ) -> Result<Self> {
        let qstate_count = q_index.states.len();
        ensure!(
            qstate_count < u32::MAX as usize && pre.len() < u32::MAX as usize,
            "terminal plan domains exceed u32 IDs"
        );
        ensure!(
            qfwd.offsets.len() == qstate_count,
            "terminal qforward row count {} != quotient count {qstate_count}",
            qfwd.offsets.len()
        );
        ensure!(
            pre.qid_to_zid.len() == qstate_count,
            "terminal pre-reveal quotient count {} != {qstate_count}",
            pre.qid_to_zid.len()
        );
        ensure!(
            qfwd.skipped_invalid_child == 0,
            "terminal qforward skipped {} invalid children",
            qfwd.skipped_invalid_child
        );
        for (qid, state) in q_index.states.iter().enumerate() {
            ensure!(
                (state.avail as usize) < AVAIL_PAIR_COUNT
                    && (state.tail5 as usize) < TAIL5_COUNT
                    && state.bag_mask != 0
                    && state.bag_mask <= FULL_BAG,
                "invalid terminal quotient state at qid {qid}"
            );
        }

        let mut reveal_offsets = Vec::with_capacity(pre.len() + 1);
        let mut reveal_children = Vec::with_capacity(qstate_count);
        reveal_offsets.push(0);
        for (zid, state) in pre.states.iter().enumerate() {
            ensure!(
                (state.avail as usize) < AVAIL_PAIR_COUNT
                    && (state.tail4 as usize) < TAIL4_COUNT
                    && state.bag_mask != 0
                    && state.bag_mask <= FULL_BAG,
                "invalid terminal pre-reveal state at zid {zid}"
            );
            let children = pre.reveal_qids(zid as u32);
            ensure!(
                children.len() == state.bag_mask.count_ones() as usize,
                "terminal pre-reveal support length mismatch at zid {zid}"
            );
            let mut child_index = 0;
            for reveal in 0..PIECE_COUNT as u8 {
                if state.bag_mask & (1 << reveal) == 0 {
                    continue;
                }
                let qid = children[child_index];
                let child = q_index.states.get(qid as usize).with_context(|| {
                    format!("terminal pre-reveal child qid {qid} outside domain at zid {zid}")
                })?;
                ensure!(
                    child.avail == state.avail
                        && child.tail5 == state.tail4 + u32::from(reveal) * TAIL4_COUNT as u32
                        && child.bag_mask == after_reveal_forward(state.bag_mask, reveal)
                        && pre.qid_to_zid[qid as usize] == zid as u32,
                    "terminal pre-reveal child or reveal order mismatch at zid {zid} slot {child_index}"
                );
                child_index += 1;
            }
            ensure!(
                reveal_children.len() + children.len() <= qstate_count,
                "terminal pre-reveal lists exceed quotient count"
            );
            reveal_children.extend_from_slice(children);
            reveal_offsets.push(reveal_children.len() as u32);
        }
        ensure!(
            reveal_children.len() == qstate_count,
            "terminal pre-reveal lists do not partition quotient states"
        );

        let mut cursor = 0usize;
        for (qid, parent) in q_index.states.iter().enumerate() {
            let offsets = &qfwd.offsets[qid];
            ensure!(
                offsets[0] as usize == cursor,
                "terminal qforward row {qid} starts at {}, expected {cursor}",
                offsets[0]
            );
            let (a, b) = decode_avail_pair(parent.avail);
            for placed in 0..PIECE_COUNT as u8 {
                let start = offsets[placed as usize] as usize;
                let end = offsets[placed as usize + 1] as usize;
                let records = qfwd.records.get(start..end).with_context(|| {
                    format!("invalid terminal qforward offsets for qid {qid} placed {placed}")
                })?;
                let zid = pre.zid_for_placement(qid as u32, placed);
                if placed != a && placed != b {
                    ensure!(
                        zid.is_none() && records.is_empty(),
                        "terminal qforward has unavailable placement for qid {qid} placed {placed}"
                    );
                    continue;
                }
                let zid = zid.with_context(|| {
                    format!("missing terminal placement ID for qid {qid} placed {placed}")
                })?;
                let state = pre.states.get(zid as usize).with_context(|| {
                    format!("terminal placement zid {zid} outside domain for qid {qid}")
                })?;
                let next_front = (parent.tail5 % PIECE_COUNT as u32) as u8;
                let remaining = if placed == a { b } else { a };
                let expected_avail = crate::qstate::encode_avail_pair(remaining, next_front);
                ensure!(
                    state.avail == expected_avail
                        && state.tail4 == parent.tail5 / PIECE_COUNT as u32
                        && state.bag_mask == parent.bag_mask,
                    "terminal placement pre-reveal state mismatch for qid {qid} placed {placed}"
                );
                let children = pre.reveal_qids(zid);
                ensure!(
                    records.len() == parent.bag_mask.count_ones() as usize
                        && records.len() == children.len(),
                    "terminal qforward reveal count mismatch for qid {qid} placed {placed}"
                );
                let mut slot = 0;
                for reveal in 0..PIECE_COUNT as u8 {
                    if parent.bag_mask & (1 << reveal) == 0 {
                        continue;
                    }
                    let record = records[slot];
                    ensure!(
                        record.reveal == reveal && record.child_qid == children[slot],
                        "terminal qforward reveal support/order mismatch for qid {qid} placed {placed} slot {slot}"
                    );
                    slot += 1;
                }
            }
            cursor = offsets[PIECE_COUNT] as usize;
        }
        ensure!(
            cursor == qfwd.records.len(),
            "terminal qforward contains unreferenced trailing records"
        );

        Ok(Self {
            qstate_count,
            reveal_offsets,
            reveal_children,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.reveal_offsets.len() - 1
    }

    pub(crate) fn qstate_count(&self) -> usize {
        self.qstate_count
    }

    pub(crate) fn child_count(&self) -> usize {
        self.reveal_children.len()
    }

    /// Retained plan bytes, including the struct and allocated list capacity.
    pub(crate) fn storage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.reveal_offsets.capacity() + self.reveal_children.capacity())
                * std::mem::size_of::<u32>()
    }

    pub(crate) fn evaluate_f32(&self, reset: &[f32], include_reward: bool) -> Result<Vec<f32>> {
        let mut output = vec![0.0; self.len()];
        self.evaluate_into(reset, include_reward, &mut output)?;
        Ok(output)
    }

    /// Return a zid-indexed array, with +0 in unselected slots. Duplicate selected
    /// IDs are allowed. The entire reset input must be finite, even if unused.
    pub(crate) fn evaluate_selected_f32(
        &self,
        reset: &[f32],
        include_reward: bool,
        used_zids: &[u32],
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; self.len()];
        self.evaluate_selected_into(reset, include_reward, used_zids, &mut output)?;
        Ok(output)
    }

    pub(crate) fn evaluate_into(
        &self,
        reset: &[f32],
        include_reward: bool,
        output: &mut [f32],
    ) -> Result<()> {
        self.check_inputs(reset, output)?;
        let reward = if include_reward { 1.0f32 } else { 0.0f32 };
        for (zid, value) in output.iter_mut().enumerate() {
            *value = self.evaluate_one(reset, reward, zid)?;
        }
        Ok(())
    }

    pub(crate) fn evaluate_selected_into(
        &self,
        reset: &[f32],
        include_reward: bool,
        used_zids: &[u32],
        output: &mut [f32],
    ) -> Result<()> {
        self.check_inputs(reset, output)?;
        if let Some(&zid) = used_zids.iter().find(|&&zid| zid as usize >= self.len()) {
            bail!("selected terminal zid {zid} outside domain {}", self.len());
        }
        output.fill(0.0);
        let reward = if include_reward { 1.0f32 } else { 0.0f32 };
        for &zid in used_zids {
            output[zid as usize] = self.evaluate_one(reset, reward, zid as usize)?;
        }
        Ok(())
    }

    fn check_inputs(&self, reset: &[f32], output: &[f32]) -> Result<()> {
        ensure!(
            reset.len() == self.qstate_count,
            "terminal reset length {} != quotient count {}",
            reset.len(),
            self.qstate_count
        );
        ensure!(
            output.len() == self.len(),
            "terminal output length {} != pre-reveal count {}",
            output.len(),
            self.len()
        );
        if let Some(qid) = reset.iter().position(|value| !value.is_finite()) {
            bail!("nonfinite terminal reset at qid {qid}");
        }
        Ok(())
    }

    #[inline]
    fn evaluate_one(&self, reset: &[f32], reward: f32, zid: usize) -> Result<f32> {
        let start = self.reveal_offsets[zid] as usize;
        let end = self.reveal_offsets[zid + 1] as usize;
        let mut sum = 0.0f32;
        for &qid in &self.reveal_children[start..end] {
            // Keep precisely the gamma=1 legacy expression, including adding a
            // zero reward in linear mode (which affects signed zero).
            sum += reward + 1.0f32 * reset[qid as usize];
        }
        let value = sum / (end - start) as f32;
        ensure!(value.is_finite(), "nonfinite terminal average at zid {zid}");
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qgraph::{build_qforward_graph, QForwardRec};
    use crate::qstate::enumerate_qstates;
    use crate::state::enumerate_states;
    use std::sync::OnceLock;

    struct Fixture {
        quotient: QStateIndex,
        pre: PreRevealIndex,
        qfwd: QForwardGraph,
        plan: TerminalPlan,
    }

    fn full_fixture() -> &'static Fixture {
        static FIXTURE: OnceLock<Fixture> = OnceLock::new();
        FIXTURE.get_or_init(|| {
            let old = enumerate_states();
            let quotient = enumerate_qstates(&old);
            drop(old);
            let pre = PreRevealIndex::build(&quotient).unwrap();
            let qfwd = build_qforward_graph(&quotient);
            let plan = TerminalPlan::build(&quotient, &qfwd, &pre).unwrap();
            Fixture {
                quotient,
                pre,
                qfwd,
                plan,
            }
        })
    }

    fn legacy(records: &[QForwardRec], reset: &[f32], include_reward: bool) -> f32 {
        let reward = if include_reward { 1.0f32 } else { 0.0f32 };
        let mut sum = 0.0f32;
        for record in records {
            sum += reward + 1.0f32 * reset[record.child_qid as usize];
        }
        sum / records.len() as f32
    }

    #[test]
    fn all_placements_match_legacy_bits_for_signed_inputs_and_both_modes() {
        let fixture = full_fixture();
        let Fixture {
            quotient,
            pre,
            qfwd,
            plan,
        } = fixture;
        assert_eq!(plan.len(), 205_170);
        assert_eq!(plan.qstate_count(), 817_740);
        assert_eq!(plan.child_count(), 817_740);
        assert!(plan.storage_bytes() < 4_100_000);
        let choices = [
            -0.0f32,
            0.0,
            -16_777_216.0,
            16_777_216.0,
            -1.0,
            f32::from_bits(0x3380_0000),
            -3.25,
            5.5,
            f32::MIN_POSITIVE,
        ];
        let reset: Vec<_> = (0..quotient.states.len())
            .map(|qid| choices[(qid.wrapping_mul(31337) ^ (qid >> 3)) % choices.len()])
            .collect();
        let mut saw_duplicate = false;
        let mut saw_full_bag = false;
        for include_reward in [false, true] {
            let values = plan.evaluate_f32(&reset, include_reward).unwrap();
            for (qid, parent) in quotient.states.iter().enumerate() {
                let (a, b) = decode_avail_pair(parent.avail);
                saw_duplicate |= a == b;
                saw_full_bag |= parent.bag_mask == FULL_BAG;
                for placed in [a, b] {
                    let zid = pre.zid_for_placement(qid as u32, placed).unwrap();
                    let expected =
                        legacy(qfwd.records_for(qid as u32, placed), &reset, include_reward);
                    assert_eq!(
                        values[zid as usize].to_bits(),
                        expected.to_bits(),
                        "qid={qid} placed={placed} reward={include_reward}"
                    );
                }
            }
        }
        assert!(saw_duplicate && saw_full_bag);
    }

    #[test]
    fn affine_reward_remains_inside_ordered_sum_and_zero_is_legacy_zero() {
        let Fixture { pre, plan, .. } = full_fixture();
        let zid = pre
            .states
            .iter()
            .position(|state| state.bag_mask.count_ones() == 2)
            .unwrap() as u32;
        let children = pre.reveal_qids(zid);
        let mut reset = vec![-0.0; plan.qstate_count()];
        reset[children[0] as usize] = 16_777_216.0;
        reset[children[1] as usize] = -16_777_216.0;
        let affine = plan.evaluate_selected_f32(&reset, true, &[zid]).unwrap();
        let average = pre.average_f32(&reset).unwrap();
        assert_ne!(
            affine[zid as usize].to_bits(),
            (1.0 + average[zid as usize]).to_bits()
        );
        assert_eq!(affine[zid as usize], 0.5);
        reset.fill(-0.0);
        let linear = plan.evaluate_f32(&reset, false).unwrap();
        assert!(linear
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits()));
    }

    #[test]
    fn selected_values_match_full_and_reused_output_is_cleared() {
        let Fixture { plan, .. } = full_fixture();
        let reset: Vec<_> = (0..plan.qstate_count())
            .map(|qid| (qid % 37) as f32 - 19.0)
            .collect();
        let selected = [0, 7, 7, (plan.len() - 1) as u32];
        for include_reward in [false, true] {
            let full = plan.evaluate_f32(&reset, include_reward).unwrap();
            let subset = plan
                .evaluate_selected_f32(&reset, include_reward, &selected)
                .unwrap();
            let mut reused = vec![f32::NAN; plan.len()];
            plan.evaluate_selected_into(&reset, include_reward, &selected, &mut reused)
                .unwrap();
            for zid in 0..plan.len() {
                let expected = if selected.contains(&(zid as u32)) {
                    full[zid]
                } else {
                    0.0
                };
                assert_eq!(subset[zid].to_bits(), expected.to_bits());
                assert_eq!(reused[zid].to_bits(), expected.to_bits());
            }
        }
        assert!(plan
            .evaluate_selected_f32(&reset, true, &[])
            .unwrap()
            .iter()
            .all(|&x| x == 0.0));
    }

    #[test]
    fn rejects_nonfinite_overflow_bad_lengths_and_selected_ids() {
        let Fixture { plan, .. } = full_fixture();
        let mut reset = vec![0.0; plan.qstate_count()];
        assert!(plan.evaluate_f32(&[], false).is_err());
        assert!(plan.evaluate_into(&reset, false, &mut []).is_err());
        assert!(plan
            .evaluate_selected_f32(&reset, false, &[plan.len() as u32])
            .is_err());
        reset[0] = f32::NAN;
        assert!(plan.evaluate_selected_f32(&reset, false, &[]).is_err());
        reset[0] = f32::INFINITY;
        assert!(plan.evaluate_f32(&reset, true).is_err());
        reset.fill(f32::MAX);
        assert!(plan.evaluate_f32(&reset, false).is_err());
        reset.fill(-f32::MAX);
        assert!(plan.evaluate_f32(&reset, true).is_err());
    }

    #[test]
    fn rejects_missing_invalid_duplicate_and_reordered_qforward_rows() {
        let Fixture {
            quotient,
            pre,
            qfwd,
            ..
        } = full_fixture();
        let parent = quotient
            .states
            .iter()
            .position(|q| q.bag_mask.count_ones() >= 2)
            .unwrap();
        let (placed, _) = decode_avail_pair(quotient.states[parent].avail);
        let index = qfwd.offsets[parent][placed as usize] as usize;
        let mut damaged = qfwd.clone();
        damaged.offsets.pop();
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.offsets = qfwd.offsets.clone();
        damaged.offsets[parent][placed as usize + 1] -= 1;
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.offsets = qfwd.offsets.clone();
        damaged.records[index].child_qid = quotient.states.len() as u32;
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.records[index] = damaged.records[index + 1];
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.records[index] = qfwd.records[index];
        damaged.records.swap(index, index + 1);
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.records.swap(index, index + 1);
        damaged.records[index].reveal = PIECE_COUNT as u8;
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
        damaged.records[index] = qfwd.records[index];
        damaged.offsets[parent][0] = u32::MAX;
        assert!(TerminalPlan::build(quotient, &damaged, pre).is_err());
    }
}
