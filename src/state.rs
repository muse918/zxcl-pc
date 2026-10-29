use std::collections::HashMap;

use crate::piece::{pieces, FULL_BAG, PIECE_COUNT};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChainState {
    pub hold: u8,
    pub queue6: u32,
    pub bag_mask: u8,
}

#[derive(Debug, Clone)]
pub struct StateIndex {
    pub states: Vec<ChainState>,
    pub key_to_id: HashMap<u32, usize>,
    pub count_by_popcount: [usize; 8],
}

#[inline]
pub fn encode_queue6(q: [u8; 6]) -> u32 {
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
pub fn decode_queue6(mut x: u32) -> [u8; 6] {
    let mut q = [0u8; 6];
    for dst in &mut q {
        *dst = (x % 7) as u8;
        x /= 7;
    }
    q
}

#[inline]
pub fn state_key(s: ChainState) -> u32 {
    ((s.hold as u32) << 24) | ((s.bag_mask as u32) << 17) | s.queue6
}

pub fn is_valid_state(s: ChainState) -> bool {
    if s.hold >= 7 || s.bag_mask == 0 || s.bag_mask >= 128 || s.queue6 >= 7u32.pow(6) {
        return false;
    }

    let q = decode_queue6(s.queue6);
    if q.iter().any(|&p| p >= 7) {
        return false;
    }

    let k = s.bag_mask.count_ones() as usize;
    let prefix_len = k - 1;
    let suffix_len = 7 - k;
    debug_assert_eq!(prefix_len + suffix_len, 6);

    let mut seen_prefix = 0u8;
    for &p in &q[..prefix_len] {
        let bit = 1u8 << p;
        if (seen_prefix & bit) != 0 {
            return false;
        }
        seen_prefix |= bit;
    }

    let mut suffix_mask = 0u8;
    for &p in &q[prefix_len..] {
        let bit = 1u8 << p;
        if (suffix_mask & bit) != 0 {
            return false;
        }
        suffix_mask |= bit;
    }

    suffix_mask == (FULL_BAG ^ s.bag_mask) && suffix_mask.count_ones() as usize == suffix_len
}

fn gen_prefixes(len: usize, cur: &mut Vec<u8>, used: u8, out: &mut Vec<Vec<u8>>) {
    if cur.len() == len {
        out.push(cur.clone());
        return;
    }
    for p in 0u8..7 {
        let bit = 1u8 << p;
        if (used & bit) != 0 {
            continue;
        }
        cur.push(p);
        gen_prefixes(len, cur, used | bit, out);
        cur.pop();
    }
}

fn gen_suffixes_from_mask(mask: u8, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if cur.len() == mask.count_ones() as usize {
        out.push(cur.clone());
        return;
    }
    for p in pieces(mask) {
        if cur.contains(&p) {
            continue;
        }
        cur.push(p);
        gen_suffixes_from_mask(mask, cur, out);
        cur.pop();
    }
}

pub fn enumerate_states() -> StateIndex {
    let mut states = Vec::<ChainState>::with_capacity(1_120_140);
    let mut count_by_popcount = [0usize; 8];

    let mut prefix_cache: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 7];
    for len in 0..=6usize {
        let mut cur = Vec::new();
        gen_prefixes(len, &mut cur, 0, &mut prefix_cache[len]);
    }

    let mut suffix_cache: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 128];
    for mask in 0u8..128u8 {
        let mut cur = Vec::new();
        gen_suffixes_from_mask(mask, &mut cur, &mut suffix_cache[mask as usize]);
    }

    for bag_mask in 1u8..128u8 {
        let k = bag_mask.count_ones() as usize;
        let prefix_len = k - 1;
        let suffix_mask = FULL_BAG ^ bag_mask;

        for prefix in &prefix_cache[prefix_len] {
            for suffix in &suffix_cache[suffix_mask as usize] {
                debug_assert_eq!(prefix.len() + suffix.len(), 6);

                let mut q = [0u8; 6];
                for (i, &p) in prefix.iter().enumerate() {
                    q[i] = p;
                }
                for (j, &p) in suffix.iter().enumerate() {
                    q[prefix.len() + j] = p;
                }

                let queue6 = encode_queue6(q);
                for hold in 0u8..7 {
                    let s = ChainState { hold, queue6, bag_mask };
                    debug_assert!(is_valid_state(s));
                    states.push(s);
                    count_by_popcount[k] += 1;
                }
            }
        }
    }

    let mut key_to_id = HashMap::<u32, usize>::with_capacity(states.len() * 2);
    for (i, &s) in states.iter().enumerate() {
        let old = key_to_id.insert(state_key(s), i);
        debug_assert!(old.is_none());
    }

    debug_assert_eq!(states.len(), 1_120_140);

    StateIndex { states, key_to_id, count_by_popcount }
}
