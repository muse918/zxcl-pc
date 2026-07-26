use crate::graph::{FieldId, FieldGraph, EMPTY_FIELD_ID, NUM_FIELDS};
use crate::piece::{pieces, Piece, FULL_BAG};
use crate::state::{decode_queue6, encode_queue6, ChainState};

pub type SolverKey = u64;

const QUEUE6_BITS: u64 = 17;
const BAG_BITS: u64 = 7;
const HOLD_BITS: u64 = 3;
const BAG_SHIFT: u64 = QUEUE6_BITS;
const HOLD_SHIFT: u64 = BAG_SHIFT + BAG_BITS;
const FIELD_SHIFT: u64 = HOLD_SHIFT + HOLD_BITS;

const QUEUE6_MASK: u64 = (1u64 << QUEUE6_BITS) - 1;
const BAG_MASK: u64 = (1u64 << BAG_BITS) - 1;
const HOLD_MASK: u64 = (1u64 << HOLD_BITS) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SolverState {
    pub field: FieldId,
    pub chain: ChainState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub next: SolverState,
    pub placed: Piece,
    pub revealed: Piece,
}

#[inline]
pub fn solver_key(state: SolverState) -> SolverKey {
    debug_assert!((state.field as usize) < NUM_FIELDS);
    debug_assert!(state.chain.hold < 7);
    debug_assert!(state.chain.bag_mask > 0 && state.chain.bag_mask < 128);
    debug_assert!(state.chain.queue6 < 7u32.pow(6));

    ((state.field as u64) << FIELD_SHIFT)
        | ((state.chain.hold as u64) << HOLD_SHIFT)
        | ((state.chain.bag_mask as u64) << BAG_SHIFT)
        | (state.chain.queue6 as u64)
}

#[inline]
pub fn decode_solver_key(key: SolverKey) -> SolverState {
    let queue6 = (key & QUEUE6_MASK) as u32;
    let bag_mask = ((key >> BAG_SHIFT) & BAG_MASK) as u8;
    let hold = ((key >> HOLD_SHIFT) & HOLD_MASK) as u8;
    let field = (key >> FIELD_SHIFT) as FieldId;

    SolverState {
        field,
        chain: ChainState {
            hold,
            queue6,
            bag_mask,
        },
    }
}

#[inline]
pub fn chain_front(chain: ChainState) -> Piece {
    decode_queue6(chain.queue6)[0]
}

#[inline]
pub fn advance_chain_after_reveal(chain: ChainState, revealed: Piece, next_hold: Piece) -> ChainState {
    debug_assert!(revealed < 7);
    debug_assert!(next_hold < 7);
    debug_assert!(chain.bag_mask != 0);
    debug_assert!((chain.bag_mask & (1 << revealed)) != 0);

    let old_q = decode_queue6(chain.queue6);
    let mut new_q = [0u8; 6];
    new_q[..5].copy_from_slice(&old_q[1..]);
    new_q[5] = revealed;

    let next_bag = chain.bag_mask & !(1 << revealed);
    let next_bag = if next_bag == 0 { FULL_BAG } else { next_bag };

    ChainState {
        hold: next_hold,
        queue6: encode_queue6(new_q),
        bag_mask: next_bag,
    }
}

#[inline]
pub fn has_playable_piece(graph: &FieldGraph, state: SolverState) -> bool {
    if state.field == EMPTY_FIELD_ID {
        return true;
    }

    let front = chain_front(state.chain);
    let hold = state.chain.hold;

    !graph.edges(state.field, front).is_empty()
        || (front != hold && !graph.edges(state.field, hold).is_empty())
}

pub fn transitions(graph: &FieldGraph, state: SolverState, out: &mut Vec<Transition>) {
    out.clear();

    if state.field == EMPTY_FIELD_ID {
        return;
    }

    let chain = state.chain;
    let front = chain_front(chain);
    let hold = chain.hold;

    for &next_field in graph.edges(state.field, front) {
        for revealed in pieces(chain.bag_mask) {
            let next_chain = advance_chain_after_reveal(chain, revealed, hold);
            out.push(Transition {
                next: SolverState {
                    field: next_field,
                    chain: next_chain,
                },
                placed: front,
                revealed,
            });
        }
    }

    if front != hold {
        for &next_field in graph.edges(state.field, hold) {
            for revealed in pieces(chain.bag_mask) {
                let next_chain = advance_chain_after_reveal(chain, revealed, front);
                out.push(Transition {
                    next: SolverState {
                        field: next_field,
                        chain: next_chain,
                    },
                    placed: hold,
                    revealed,
                });
            }
        }
    }
}
