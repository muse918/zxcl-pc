use crate::graph::{FieldGraph, EMPTY_FIELD_ID};
use crate::piece::{pieces, Piece};
use crate::product::{
    advance_chain_after_reveal, chain_front, has_playable_piece, solver_key, SolverKey, SolverState,
};
use hashbrown::HashMap;

#[derive(Default)]
pub struct PcPruner {
    /// `true` means at least one legal future sequence can reach PC.
    exists_memo: HashMap<SolverKey, bool>,
    /// `true` means there is a placement choice such that every possible next reveal remains PC-able.
    forced_memo: HashMap<SolverKey, bool>,
}

impl PcPruner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.exists_memo.clear();
        self.forced_memo.clear();
    }

    #[inline]
    pub fn is_immediately_dead(&self, graph: &FieldGraph, state: SolverState) -> bool {
        state.field != EMPTY_FIELD_ID && !has_playable_piece(graph, state)
    }

    /// Weak reachability prune.
    ///
    /// Returns true if some sequence of placements and future reveals can reach `EMPTY_FIELD_ID`.
    /// This is safe but weaker than boolean-mode reachability pruning.
    pub fn can_pc_exists(&mut self, graph: &FieldGraph, state: SolverState) -> bool {
        let key = solver_key(state);
        if let Some(&v) = self.exists_memo.get(&key) {
            return v;
        }

        let result = self.can_pc_exists_inner(graph, state);
        self.exists_memo.insert(key, result);
        result
    }

    fn can_pc_exists_inner(&mut self, graph: &FieldGraph, state: SolverState) -> bool {
        if state.field == EMPTY_FIELD_ID {
            return true;
        }
        if !has_playable_piece(graph, state) {
            return false;
        }

        let chain = state.chain;
        let front = chain_front(chain);
        let hold = chain.hold;

        for &next_field in graph.edges(state.field, front) {
            for revealed in pieces(chain.bag_mask) {
                let next_chain = advance_chain_after_reveal(chain, revealed, hold);
                if self.can_pc_exists(graph, SolverState { field: next_field, chain: next_chain }) {
                    return true;
                }
            }
        }

        if front != hold {
            for &next_field in graph.edges(state.field, hold) {
                for revealed in pieces(chain.bag_mask) {
                    let next_chain = advance_chain_after_reveal(chain, revealed, front);
                    if self.can_pc_exists(graph, SolverState { field: next_field, chain: next_chain }) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Boolean-mode reachability prune.
    ///
    /// Returns true if there exists a current placement such that every possible next revealed piece
    /// from the current bag leads to a state that is again forced-PC-able.
    pub fn can_pc_forced(&mut self, graph: &FieldGraph, state: SolverState) -> bool {
        let key = solver_key(state);
        if let Some(&v) = self.forced_memo.get(&key) {
            return v;
        }

        let result = self.can_pc_forced_inner(graph, state);
        self.forced_memo.insert(key, result);
        result
    }

    fn can_pc_forced_inner(&mut self, graph: &FieldGraph, state: SolverState) -> bool {
        if state.field == EMPTY_FIELD_ID {
            return true;
        }
        if !has_playable_piece(graph, state) {
            return false;
        }

        let chain = state.chain;
        let front = chain_front(chain);
        let hold = chain.hold;

        for &next_field in graph.edges(state.field, front) {
            if self.all_reveals_forced_ok(graph, next_field, chain, hold) {
                return true;
            }
        }

        if front != hold {
            for &next_field in graph.edges(state.field, hold) {
                if self.all_reveals_forced_ok(graph, next_field, chain, front) {
                    return true;
                }
            }
        }

        false
    }

    fn all_reveals_forced_ok(
        &mut self,
        graph: &FieldGraph,
        next_field: crate::graph::FieldId,
        chain: crate::state::ChainState,
        next_hold: Piece,
    ) -> bool {
        for revealed in pieces(chain.bag_mask) {
            let next_chain = advance_chain_after_reveal(chain, revealed, next_hold);
            let next_state = SolverState {
                field: next_field,
                chain: next_chain,
            };
            if !self.can_pc_forced(graph, next_state) {
                return false;
            }
        }
        true
    }
}
