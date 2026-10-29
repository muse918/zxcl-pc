//! The boundary after placement and queue advance, before the random reveal.
//!
//! A quotient state has already observed its last queued piece. Removing that
//! reveal gives exactly one pre-reveal state, so the reveal lists partition the
//! quotient states. Values at this boundary average *post-reveal* optimal values;
//! decisions that can use the revealed piece must stay inside that average.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use crate::piece::{FULL_BAG, PIECE_COUNT};
use crate::qstate::{
    decode_avail_pair, encode_avail_pair, reverse_bag, QStateIndex, AVAIL_PAIR_COUNT, TAIL5_COUNT,
};

pub(crate) const TAIL4_COUNT: usize = 7usize.pow(4);

/// The available pair and four queued pieces immediately before a reveal.
/// `bag_mask` is the bag *before* drawing the new final queued piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PreRevealState {
    pub(crate) avail: u8,
    /// Little-endian base-7 encoding of the four known queued pieces.
    pub(crate) tail4: u32,
    pub(crate) bag_mask: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct PreRevealIndex {
    /// Sorted lexicographically by `(avail, tail4, bag_mask)`.
    pub(crate) states: Vec<PreRevealState>,
    /// Each quotient state belongs to exactly one inverse-reveal group.
    pub(crate) qid_to_zid: Vec<u32>,
    reveal_offsets: Vec<u32>,
    /// Each group's children are in ascending revealed-piece order.
    reveal_children: Vec<u32>,
    /// Placement IDs corresponding to the smaller and larger available pieces.
    /// A duplicated available pair has the same ID in both slots.
    placement_zids: Vec<[u32; 2]>,
    qid_avails: Vec<u8>,
}

impl PreRevealIndex {
    pub(crate) fn build(q_index: &QStateIndex) -> Result<Self> {
        if q_index.states.len() >= u32::MAX as usize {
            bail!("too many quotient states for pre-reveal u32 IDs");
        }
        let mut grouped = Vec::with_capacity(q_index.states.len());
        for (qid, q) in q_index.states.iter().enumerate() {
            if q.avail as usize >= AVAIL_PAIR_COUNT
                || q.tail5 as usize >= TAIL5_COUNT
                || q.bag_mask == 0
                || q.bag_mask > FULL_BAG
            {
                bail!("invalid quotient state at qid {qid}: {q:?}");
            }
            let revealed = (q.tail5 / TAIL4_COUNT as u32) as u8;
            let bag_mask = reverse_bag(q.bag_mask, revealed)
                .with_context(|| format!("qid {qid} has no inverse last reveal"))?;
            let state = PreRevealState {
                avail: q.avail,
                tail4: q.tail5 % TAIL4_COUNT as u32,
                bag_mask,
            };
            grouped.push((state, revealed, qid as u32));
        }
        grouped.sort_unstable();

        let mut states = Vec::new();
        let mut qid_to_zid = vec![u32::MAX; q_index.states.len()];
        let mut reveal_offsets = vec![0];
        let mut reveal_children = Vec::with_capacity(grouped.len());
        let mut start = 0;
        while start < grouped.len() {
            let state = grouped[start].0;
            let zid = states.len() as u32;
            let mut seen_reveals = 0u8;
            let mut end = start;
            while end < grouped.len() && grouped[end].0 == state {
                let (_, revealed, qid) = grouped[end];
                let bit = 1u8 << revealed;
                if seen_reveals & bit != 0 {
                    bail!("duplicate reveal {revealed} for pre-reveal state {state:?}");
                }
                seen_reveals |= bit;
                qid_to_zid[qid as usize] = zid;
                reveal_children.push(qid);
                end += 1;
            }
            if seen_reveals != state.bag_mask {
                bail!("incomplete reveal support for {state:?}: got {seen_reveals:#09b}");
            }
            states.push(state);
            reveal_offsets.push(reveal_children.len() as u32);
            start = end;
        }
        drop(grouped);

        let state_to_zid: HashMap<_, _> = states
            .iter()
            .enumerate()
            .map(|(zid, &state)| (state, zid as u32))
            .collect();
        let mut placement_zids = Vec::with_capacity(q_index.states.len());
        let mut qid_avails = Vec::with_capacity(q_index.states.len());
        for (qid, q) in q_index.states.iter().enumerate() {
            let (a, b) = decode_avail_pair(q.avail);
            let next_front = (q.tail5 % PIECE_COUNT as u32) as u8;
            let mut zids = [0u32; 2];
            for (slot, remaining) in [b, a].into_iter().enumerate() {
                let state = PreRevealState {
                    avail: encode_avail_pair(remaining, next_front),
                    tail4: q.tail5 / PIECE_COUNT as u32,
                    bag_mask: q.bag_mask,
                };
                zids[slot] = *state_to_zid.get(&state).with_context(|| {
                    format!("missing placement pre-reveal state from qid {qid}: {state:?}")
                })?;
            }
            placement_zids.push(zids);
            qid_avails.push(q.avail);
        }

        Ok(Self {
            states,
            qid_to_zid,
            reveal_offsets,
            reveal_children,
            placement_zids,
            qid_avails,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }

    /// Retained heap payload, including spare Vec capacity. Callers that retain
    /// this shared source must charge it once to their resident-memory plan.
    pub(crate) fn storage_bytes(&self) -> u64 {
        let parts = [
            (self.states.capacity(), std::mem::size_of::<PreRevealState>()),
            (self.qid_to_zid.capacity(), std::mem::size_of::<u32>()),
            (self.reveal_offsets.capacity(), std::mem::size_of::<u32>()),
            (self.reveal_children.capacity(), std::mem::size_of::<u32>()),
            (self.placement_zids.capacity(), std::mem::size_of::<[u32; 2]>()),
            (self.qid_avails.capacity(), std::mem::size_of::<u8>()),
        ];
        parts.into_iter().fold(0u64, |total, (count, width)| {
            total.saturating_add((count as u64).saturating_mul(width as u64))
        })
    }

    /// The quotient children for all possible reveals, in ascending piece order.
    #[inline]
    pub(crate) fn reveal_qids(&self, zid: u32) -> &[u32] {
        let start = self.reveal_offsets[zid as usize] as usize;
        let end = self.reveal_offsets[zid as usize + 1] as usize;
        &self.reveal_children[start..end]
    }

    /// Advance a quotient state through placement, stopping before the reveal.
    #[inline]
    pub(crate) fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        if placed as usize >= PIECE_COUNT {
            return None;
        }
        let &avail = self.qid_avails.get(qid as usize)?;
        let (a, b) = decode_avail_pair(avail);
        if placed == a {
            Some(self.placement_zids[qid as usize][0])
        } else if placed == b {
            Some(self.placement_zids[qid as usize][1])
        } else {
            None
        }
    }

    /// Average post-reveal values over each bag's equally likely draws.
    /// Signed values are accepted, including Krylov vectors.
    pub(crate) fn average_f64(&self, values_q: &[f64]) -> Result<Vec<f64>> {
        check_f64(values_q, self.qid_to_zid.len(), "quotient")?;
        let mut values_z = Vec::with_capacity(self.len());
        for zid in 0..self.len() {
            let children = self.reveal_qids(zid as u32);
            let mut sum = 0.0;
            for &qid in children {
                sum += values_q[qid as usize];
            }
            let value = sum / children.len() as f64;
            if !value.is_finite() {
                bail!("nonfinite pre-reveal average at zid {zid}");
            }
            values_z.push(value);
        }
        Ok(values_z)
    }

    /// Average using f32 arithmetic in ascending reveal order.
    /// Moving this boundary changes rounding: it need not be bit identical to
    /// averaging `reward + continuation` separately for each reveal in f32.
    pub(crate) fn average_f32(&self, values_q: &[f32]) -> Result<Vec<f32>> {
        check_f32(values_q, self.qid_to_zid.len(), "quotient")?;
        let mut values_z = Vec::with_capacity(self.len());
        for zid in 0..self.len() {
            let children = self.reveal_qids(zid as u32);
            let mut sum = 0.0f32;
            for &qid in children {
                sum += values_q[qid as usize];
            }
            let value = sum / children.len() as f32;
            if !value.is_finite() {
                bail!("nonfinite pre-reveal average at zid {zid}");
            }
            values_z.push(value);
        }
        Ok(values_z)
    }

    /// Assign each qid its inverse-reveal group's value. This is a group-value
    /// lift, not a placement or a max over actions.
    pub(crate) fn lift_f64(&self, values_z: &[f64]) -> Result<Vec<f64>> {
        check_f64(values_z, self.len(), "pre-reveal")?;
        Ok(self
            .qid_to_zid
            .iter()
            .map(|&zid| values_z[zid as usize])
            .collect())
    }

    pub(crate) fn lift_f32(&self, values_z: &[f32]) -> Result<Vec<f32>> {
        check_f32(values_z, self.len(), "pre-reveal")?;
        Ok(self
            .qid_to_zid
            .iter()
            .map(|&zid| values_z[zid as usize])
            .collect())
    }
}

fn check_f64(values: &[f64], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        bail!("{label} value length {}, expected {expected}", values.len());
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        bail!("nonfinite {label} value at index {index}");
    }
    Ok(())
}

fn check_f32(values: &[f32], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        bail!("{label} value length {}, expected {expected}", values.len());
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        bail!("nonfinite {label} value at index {index}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qgraph::build_qforward_graph;
    use crate::qstate::{
        after_reveal_forward, decode_tail5, enumerate_qstates, q_forward_after_reveal, qchain_key,
    };
    use crate::state::enumerate_states;
    use std::sync::OnceLock;

    fn full_index() -> &'static (QStateIndex, PreRevealIndex) {
        static INDEX: OnceLock<(QStateIndex, PreRevealIndex)> = OnceLock::new();
        INDEX.get_or_init(|| {
            let old = enumerate_states();
            assert_eq!(old.states.len(), 1_120_140);
            let quotient = enumerate_qstates(&old);
            drop(old);
            assert_eq!(quotient.states.len(), 817_740);
            let pre_reveal = PreRevealIndex::build(&quotient).unwrap();
            assert_eq!(pre_reveal.len(), 205_170);
            (quotient, pre_reveal)
        })
    }

    #[test]
    fn exact_partition_reveal_order_refill_and_duplicate_available() {
        let (q, z) = full_index();
        assert!(z.states.windows(2).all(|pair| pair[0] < pair[1]));
        let mut visits = vec![0u8; q.states.len()];
        let mut saw_refill = false;
        let mut saw_duplicate = false;
        for (zid, state) in z.states.iter().enumerate() {
            let qids = z.reveal_qids(zid as u32);
            assert_eq!(qids.len(), state.bag_mask.count_ones() as usize);
            let mut previous_reveal = None;
            let mut seen_reveals = 0;
            for &qid in qids {
                visits[qid as usize] += 1;
                assert_eq!(z.qid_to_zid[qid as usize], zid as u32);
                let child = q.states[qid as usize];
                let tail = decode_tail5(child.tail5);
                let reveal = tail[4];
                assert!(previous_reveal.map_or(true, |previous| previous < reveal));
                previous_reveal = Some(reveal);
                seen_reveals |= 1 << reveal;
                assert_eq!(child.avail, state.avail);
                assert_eq!(child.tail5 % TAIL4_COUNT as u32, state.tail4);
                assert_eq!(child.bag_mask, after_reveal_forward(state.bag_mask, reveal));
                assert_eq!(reverse_bag(child.bag_mask, reveal), Some(state.bag_mask));
                if state.bag_mask.count_ones() == 1 {
                    saw_refill = true;
                    assert_eq!(child.bag_mask, FULL_BAG);
                }
            }
            assert_eq!(seen_reveals, state.bag_mask);
            let (a, b) = decode_avail_pair(state.avail);
            saw_duplicate |= a == b;
        }
        assert!(visits.into_iter().all(|count| count == 1));
        assert!(saw_refill && saw_duplicate);
    }

    #[test]
    fn every_placement_matches_direct_forward_reveals() {
        let (q, z) = full_index();
        let mut saw_duplicate = false;
        for (qid, &parent) in q.states.iter().enumerate() {
            let (a, b) = decode_avail_pair(parent.avail);
            for placed in 0..PIECE_COUNT as u8 {
                let Some(zid) = z.zid_for_placement(qid as u32, placed) else {
                    assert!(placed != a && placed != b);
                    continue;
                };
                assert!(placed == a || placed == b);
                let children = z.reveal_qids(zid);
                let mut offset = 0;
                for reveal in 0..PIECE_COUNT as u8 {
                    let Some(child) = q_forward_after_reveal(parent, placed, reveal) else {
                        continue;
                    };
                    assert_eq!(children[offset] as usize, q.key_to_id[&qchain_key(child)]);
                    offset += 1;
                }
                assert_eq!(offset, children.len());
            }
            if a == b {
                saw_duplicate = true;
                let zid = z.zid_for_placement(qid as u32, a).unwrap();
                assert_eq!(z.placement_zids[qid], [zid, zid]);
            }
        }
        assert!(saw_duplicate);
        assert_eq!(z.zid_for_placement(q.states.len() as u32, 0), None);
        assert_eq!(z.zid_for_placement(0, PIECE_COUNT as u8), None);
    }

    #[test]
    fn signed_averages_lifts_and_validation() {
        let (_, z) = full_index();
        let values_z: Vec<f64> = (0..z.len()).map(|i| (i % 29) as f64 - 14.0).collect();
        let lifted = z.lift_f64(&values_z).unwrap();
        assert_eq!(z.average_f64(&lifted).unwrap(), values_z);
        let values_z32: Vec<f32> = values_z.iter().map(|&x| x as f32).collect();
        let lifted32 = z.lift_f32(&values_z32).unwrap();
        assert_eq!(z.average_f32(&lifted32).unwrap(), values_z32);

        assert!(z.average_f64(&[]).is_err());
        assert!(z.average_f32(&[]).is_err());
        assert!(z.lift_f64(&[]).is_err());
        assert!(z.lift_f32(&[]).is_err());
        let mut invalid_q = lifted;
        invalid_q[0] = f64::NAN;
        assert!(z.average_f64(&invalid_q).is_err());
        let mut invalid_q32 = lifted32;
        invalid_q32[0] = f32::INFINITY;
        assert!(z.average_f32(&invalid_q32).is_err());
        let mut invalid_z = values_z;
        invalid_z[0] = f64::NEG_INFINITY;
        assert!(z.lift_f64(&invalid_z).is_err());
        let mut invalid_z32 = values_z32;
        invalid_z32[0] = f32::NAN;
        assert!(z.lift_f32(&invalid_z32).is_err());
    }

    #[test]
    fn maximize_after_observing_the_reveal_before_averaging() {
        let (_, z) = full_index();
        let zid = z
            .states
            .iter()
            .position(|state| state.bag_mask.count_ones() == 2)
            .unwrap() as u32;
        let children = z.reveal_qids(zid);
        let mut action_a = vec![0.0; z.qid_to_zid.len()];
        let mut action_b = action_a.clone();
        action_a[children[0] as usize] = 2.0;
        action_b[children[1] as usize] = 2.0;
        let observed_optimum: Vec<f64> = action_a
            .iter()
            .zip(&action_b)
            .map(|(&a, &b)| f64::max(a, b))
            .collect();
        assert_eq!(z.average_f64(&observed_optimum).unwrap()[zid as usize], 2.0);
        let fixed_a = z.average_f64(&action_a).unwrap()[zid as usize];
        let fixed_b = z.average_f64(&action_b).unwrap()[zid as usize];
        assert_eq!(fixed_a.max(fixed_b), 1.0);
    }

    #[test]
    fn rejects_invalid_inverse_reveal_and_incomplete_groups() {
        let (q, _) = full_index();
        let mut invalid = q.clone();
        let revealed = (invalid.states[0].tail5 / TAIL4_COUNT as u32) as u8;
        invalid.states[0].bag_mask = 1 << revealed;
        assert!(PreRevealIndex::build(&invalid).is_err());
        drop(invalid);
        let mut incomplete = q.clone();
        let remove = incomplete
            .states
            .iter()
            .position(|state| {
                reverse_bag(state.bag_mask, (state.tail5 / TAIL4_COUNT as u32) as u8)
                    .unwrap()
                    .count_ones()
                    > 1
            })
            .unwrap();
        incomplete.states.remove(remove);
        assert!(PreRevealIndex::build(&incomplete).is_err());
    }

    #[test]
    #[ignore = "builds the full redundant quotient forward graph"]
    fn full_qforward_graph_records_equal_placement_reveal_lists() {
        let (q, z) = full_index();
        let graph = build_qforward_graph(q);
        assert_eq!(graph.skipped_invalid_child, 0);
        for qid in 0..q.states.len() {
            for placed in 0..PIECE_COUNT as u8 {
                let records = graph.records_for(qid as u32, placed);
                let Some(zid) = z.zid_for_placement(qid as u32, placed) else {
                    assert!(records.is_empty());
                    continue;
                };
                let children = z.reveal_qids(zid);
                assert_eq!(records.len(), children.len());
                for (record, &child_qid) in records.iter().zip(children) {
                    assert_eq!(record.child_qid, child_qid);
                    assert_eq!(
                        record.reveal,
                        (q.states[child_qid as usize].tail5 / TAIL4_COUNT as u32) as u8
                    );
                }
            }
        }
    }
}
