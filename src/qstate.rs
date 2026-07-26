#![allow(dead_code)]

use std::collections::HashMap;

use crate::piece::{FULL_BAG, PIECE_COUNT};
use crate::state::{decode_queue6, ChainState, StateIndex};

pub const AVAIL_PAIR_COUNT: usize = 28;
pub const TAIL5_COUNT: usize = 7usize.pow(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QChainState {
    /// Encoded unordered pair {hold, queue[0]}, range 0..28.
    pub avail: u8,
    /// Base-7 encoded [queue[1], queue[2], queue[3], queue[4], queue[5]], range 0..7^5.
    pub tail5: u32,
    pub bag_mask: u8,
}

#[derive(Debug, Clone)]
pub struct QStateIndex {
    pub states: Vec<QChainState>,
    pub key_to_id: HashMap<u32, usize>,
    /// old oriented chain id -> quotient chain id
    pub old_to_qid: Vec<u32>,
    pub count_by_popcount: [usize; 8],
}

#[inline]
pub fn encode_tail5(q: [u8; 5]) -> u32 {
    let mut x = 0u32;
    let mut mul = 1u32;
    for &p in &q {
        debug_assert!((p as usize) < PIECE_COUNT);
        x += (p as u32) * mul;
        mul *= 7;
    }
    x
}

#[inline]
pub fn decode_tail5(mut x: u32) -> [u8; 5] {
    let mut q = [0u8; 5];
    for dst in &mut q {
        *dst = (x % 7) as u8;
        x /= 7;
    }
    q
}

/// Encode unordered pair {a,b}, with a,b in 0..7, into 0..28.
#[inline]
pub fn encode_avail_pair(a: u8, b: u8) -> u8 {
    debug_assert!(a < 7 && b < 7);
    let (x, y) = if a <= b {
        (a as u32, b as u32)
    } else {
        (b as u32, a as u32)
    };
    // number of pairs whose first element is < x:
    // sum_{i=0}^{x-1} (7-i) = 7x - x(x-1)/2
    (7 * x - (x * x.saturating_sub(1)) / 2 + (y - x)) as u8
}

#[inline]
pub fn decode_avail_pair(idx: u8) -> (u8, u8) {
    debug_assert!((idx as usize) < AVAIL_PAIR_COUNT);
    let mut n = idx;
    for a in 0u8..7 {
        let cnt = 7 - a;
        if n < cnt {
            return (a, a + n);
        }
        n -= cnt;
    }
    unreachable!()
}

#[inline]
pub fn qchain_key(s: QChainState) -> u32 {
    debug_assert!((s.avail as usize) < AVAIL_PAIR_COUNT);
    debug_assert!((s.tail5 as usize) < TAIL5_COUNT);
    debug_assert!(s.bag_mask > 0 && s.bag_mask < 128);
    (s.avail as u32) | (s.tail5 << 5) | ((s.bag_mask as u32) << 20)
}

#[inline]
pub fn unpack_qchain_key(key: u32) -> QChainState {
    let avail = (key & 0x1F) as u8;
    let tail5 = (key >> 5) & 0x7FFF;
    let bag_mask = ((key >> 20) & 0x7F) as u8;
    QChainState {
        avail,
        tail5,
        bag_mask,
    }
}

#[inline]
pub fn bag_tail_key(s: QChainState) -> u32 {
    debug_assert!((s.tail5 as usize) < TAIL5_COUNT);
    debug_assert!(s.bag_mask > 0 && s.bag_mask < 128);
    // 15 bits tail + 7 bits bag = 22 bits.
    s.tail5 | ((s.bag_mask as u32) << 15)
}

#[inline]
pub fn qstate_from_bag_tail_avail(bag_tail: u32, avail: u8) -> QChainState {
    QChainState {
        avail,
        tail5: bag_tail & 0x7FFF,
        bag_mask: ((bag_tail >> 15) & 0x7F) as u8,
    }
}

#[inline]
pub fn quotient_chain(c: ChainState) -> QChainState {
    let q = decode_queue6(c.queue6);
    QChainState {
        avail: encode_avail_pair(c.hold, q[0]),
        tail5: encode_tail5([q[1], q[2], q[3], q[4], q[5]]),
        bag_mask: c.bag_mask,
    }
}

#[inline]
pub fn after_reveal_forward(old_bag: u8, revealed: u8) -> u8 {
    let bit = 1u8 << revealed;
    debug_assert!((old_bag & bit) != 0);
    let rest = old_bag & !bit;
    if rest == 0 {
        FULL_BAG
    } else {
        rest
    }
}

pub fn q_forward_after_reveal(s: QChainState, placed: u8, revealed: u8) -> Option<QChainState> {
    if (s.bag_mask & (1u8 << revealed)) == 0 {
        return None;
    }
    let (a, b) = decode_avail_pair(s.avail);
    if placed != a && placed != b {
        return None;
    }

    let remaining = if a == b {
        a
    } else if placed == a {
        b
    } else {
        a
    };

    let tail = decode_tail5(s.tail5);
    let next_avail = encode_avail_pair(remaining, tail[0]);
    let next_tail5 = encode_tail5([tail[1], tail[2], tail[3], tail[4], revealed]);
    let next_bag = after_reveal_forward(s.bag_mask, revealed);

    Some(QChainState {
        avail: next_avail,
        tail5: next_tail5,
        bag_mask: next_bag,
    })
}

#[inline]
pub fn reverse_bag(new_bag: u8, revealed: u8) -> Option<u8> {
    let bit = 1u8 << revealed;
    if new_bag == FULL_BAG {
        Some(bit)
    } else if (new_bag & bit) != 0 {
        None
    } else {
        Some(new_bag | bit)
    }
}

/// Return `(placed_piece, previous_qchain_candidate)`.
/// Validity is intentionally checked by QStateIndex lookup after construction.
pub fn reverse_qchain_state(new_s: QChainState, out: &mut Vec<(u8, QChainState)>) {
    out.clear();

    let new_tail = decode_tail5(new_s.tail5);
    let revealed = new_tail[4];
    let Some(old_bag) = reverse_bag(new_s.bag_mask, revealed) else {
        return;
    };

    let (x, y) = decode_avail_pair(new_s.avail);
    let choices = [(x, y), (y, x)]; // (remaining, old_q1)

    for (choice_idx, &(remaining, old_q1)) in choices.iter().enumerate() {
        if x == y && choice_idx == 1 {
            break;
        }

        let old_tail5 = encode_tail5([old_q1, new_tail[0], new_tail[1], new_tail[2], new_tail[3]]);

        for placed in 0u8..PIECE_COUNT as u8 {
            let old = QChainState {
                avail: encode_avail_pair(placed, remaining),
                tail5: old_tail5,
                bag_mask: old_bag,
            };
            out.push((placed, old));
        }
    }
}

pub fn enumerate_qstates(old_index: &StateIndex) -> QStateIndex {
    let mut keys = Vec::<u32>::with_capacity(old_index.states.len());
    for &old in &old_index.states {
        keys.push(qchain_key(quotient_chain(old)));
    }
    keys.sort_unstable();
    keys.dedup();

    let mut states = Vec::<QChainState>::with_capacity(keys.len());
    let mut key_to_id = HashMap::<u32, usize>::with_capacity(keys.len() * 2);
    let mut count_by_popcount = [0usize; 8];

    for (id, &key) in keys.iter().enumerate() {
        let s = unpack_qchain_key(key);
        count_by_popcount[s.bag_mask.count_ones() as usize] += 1;
        states.push(s);
        let old = key_to_id.insert(key, id);
        debug_assert!(old.is_none());
    }

    let mut old_to_qid = Vec::<u32>::with_capacity(old_index.states.len());
    for &old in &old_index.states {
        let key = qchain_key(quotient_chain(old));
        let qid = *key_to_id.get(&key).expect("qchain missing") as u32;
        old_to_qid.push(qid);
    }

    QStateIndex {
        states,
        key_to_id,
        old_to_qid,
        count_by_popcount,
    }
}

pub fn print_qstate_stats(old_index: &StateIndex, q_index: &QStateIndex) {
    println!("old_chain_states={}", old_index.states.len());
    println!("qchain_states={}", q_index.states.len());
    println!(
        "qchain_ratio={:.6}",
        q_index.states.len() as f64 / old_index.states.len() as f64
    );
    for k in 1..=7usize {
        let old = old_index.count_by_popcount[k];
        let q = q_index.count_by_popcount[k];
        println!(
            "qchain popcount={k} old={old} q={q} ratio={:.6}",
            q as f64 / old as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avail_pair_roundtrip() {
        let mut seen = [false; AVAIL_PAIR_COUNT];
        for a in 0u8..7 {
            for b in a..7 {
                let e = encode_avail_pair(a, b);
                assert!((e as usize) < AVAIL_PAIR_COUNT);
                assert!(!seen[e as usize]);
                seen[e as usize] = true;
                assert_eq!(decode_avail_pair(e), (a, b));
            }
        }
        assert!(seen.into_iter().all(|x| x));
    }
}
