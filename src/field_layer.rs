use rayon::prelude::*;

use crate::graph::{FieldId, FieldGraph};
use crate::piece::{pieces, FULL_BAG};
use crate::state::{
    decode_queue6, encode_queue6, is_valid_state, state_key, ChainState, StateIndex,
};

pub type ChainId = u32;

pub const CHAIN_COUNT: usize = 1_120_140;
pub const DENSE_THRESHOLD: usize = 46_673;
pub const RANK_BLOCK_WORDS: usize = 8; // 8 * u64 = 512 bits

#[derive(Clone, Copy, Debug)]
pub struct MoveAction {
    pub placed: u8,
    pub use_hold: bool,
    pub target_field: FieldId,
}

#[derive(Debug, Clone)]
pub struct SparseBlock {
    pub field_id: FieldId,
    pub chains: Vec<ChainId>, // sorted
    pub values: Vec<f32>,     // chains[i]의 V
}

#[derive(Debug, Clone)]
pub struct DenseBlock {
    pub field_id: FieldId,
    pub bits: Vec<u64>,     // CHAIN_COUNT bits
    pub rank_lut: Vec<u32>, // prefix rank every 512 bits
    pub values: Vec<f32>,   // active chains only, bitset order
}

#[derive(Debug, Clone)]
pub enum FieldBlock {
    Sparse(SparseBlock),
    Dense(DenseBlock),
}

#[derive(Debug, Clone)]
pub struct FieldLayer {
    pub blocks: Vec<FieldBlock>, // sorted by field_id
    pub total_states: usize,
}

pub trait LayerLookup: Sync {
    fn get(&self, field_id: FieldId, chain_id: ChainId) -> f32;
}

pub struct TerminalLookup<'a> {
    pub terminal_field: FieldId,
    pub u: &'a [f32],
}

impl<'a> LayerLookup for TerminalLookup<'a> {
    #[inline]
    fn get(&self, field_id: FieldId, chain_id: ChainId) -> f32 {
        if field_id != self.terminal_field {
            return 0.0;
        }

        1.0 + self.u[chain_id as usize]
    }
}

impl FieldBlock {
    #[inline]
    pub fn field_id(&self) -> FieldId {
        match self {
            FieldBlock::Sparse(b) => b.field_id,
            FieldBlock::Dense(b) => b.field_id,
        }
    }

    #[inline]
    pub fn active_count(&self) -> usize {
        match self {
            FieldBlock::Sparse(b) => b.chains.len(),
            FieldBlock::Dense(b) => b.values.len(),
        }
    }

    #[inline]
    pub fn value_slice(&self) -> &[f32] {
        match self {
            FieldBlock::Sparse(b) => &b.values,
            FieldBlock::Dense(b) => &b.values,
        }
    }
}

impl FieldLayer {
    #[inline]
    pub fn find_block(&self, field_id: FieldId) -> Option<&FieldBlock> {
        self.blocks
            .binary_search_by_key(&field_id, |b| b.field_id())
            .ok()
            .map(|i| &self.blocks[i])
    }

    pub fn print_summary(&self, label: &str) {
        let mut counts = self
            .blocks
            .iter()
            .map(|b| b.active_count())
            .collect::<Vec<_>>();

        counts.sort_unstable();

        let sparse_count = self
            .blocks
            .iter()
            .filter(|b| matches!(b, FieldBlock::Sparse(_)))
            .count();

        let dense_count = self.blocks.len() - sparse_count;

        let q = |p: f64| -> usize {
            if counts.is_empty() {
                return 0;
            }
            let idx = ((counts.len() - 1) as f64 * p).round() as usize;
            counts[idx]
        };

        println!("{label} field_layer blocks={}", self.blocks.len());
        println!("{label} field_layer total_states={}", self.total_states);
        println!("{label} field_layer sparse_blocks={sparse_count}");
        println!("{label} field_layer dense_blocks={dense_count}");

        if !counts.is_empty() {
            println!("{label} chains_per_field min={}", counts[0]);
            println!("{label} chains_per_field p50={}", q(0.50));
            println!("{label} chains_per_field p90={}", q(0.90));
            println!("{label} chains_per_field p99={}", q(0.99));
            println!("{label} chains_per_field max={}", counts[counts.len() - 1]);
        }

        let sparse_bytes_u32_fp32 = self
            .blocks
            .iter()
            .map(|b| match b {
                FieldBlock::Sparse(s) => s.chains.len() * (4 + 4),
                FieldBlock::Dense(d) => d.bits.len() * 8 + d.rank_lut.len() * 4 + d.values.len() * 4,
            })
            .sum::<usize>();

        let sparse_bytes_u24_fp16 = self
            .blocks
            .iter()
            .map(|b| match b {
                FieldBlock::Sparse(s) => s.chains.len() * (3 + 2),
                FieldBlock::Dense(d) => d.bits.len() * 8 + d.rank_lut.len() * 4 + d.values.len() * 2,
            })
            .sum::<usize>();

        println!(
            "{label} estimated adaptive u32/f32 bytes={} ({:.3} GiB)",
            sparse_bytes_u32_fp32,
            sparse_bytes_u32_fp32 as f64 / 1024.0 / 1024.0 / 1024.0
        );

        println!(
            "{label} estimated adaptive u24/fp16 bytes={} ({:.3} GiB)",
            sparse_bytes_u24_fp16,
            sparse_bytes_u24_fp16 as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }
}

impl LayerLookup for FieldLayer {
    #[inline]
    fn get(&self, field_id: FieldId, chain_id: ChainId) -> f32 {
        let Some(block) = self.find_block(field_id) else {
            return 0.0;
        };

        match block {
            FieldBlock::Sparse(b) => match b.chains.binary_search(&chain_id) {
                Ok(i) => b.values[i],
                Err(_) => 0.0,
            },
            FieldBlock::Dense(b) => b.get(chain_id as usize),
        }
    }
}

impl DenseBlock {
    #[inline]
    pub fn contains(&self, chain_id: usize) -> bool {
        let word = chain_id >> 6;
        let bit = chain_id & 63;

        if word >= self.bits.len() {
            return false;
        }

        ((self.bits[word] >> bit) & 1) != 0
    }

    #[inline]
    pub fn rank1(&self, chain_id: usize) -> usize {
        let word = chain_id >> 6;
        let bit = chain_id & 63;

        let block = word / RANK_BLOCK_WORDS;
        let block_start_word = block * RANK_BLOCK_WORDS;

        let mut rank = self.rank_lut[block] as usize;

        for w in block_start_word..word {
            rank += self.bits[w].count_ones() as usize;
        }

        let mask = if bit == 0 {
            0
        } else {
            (1u64 << bit) - 1
        };

        rank += (self.bits[word] & mask).count_ones() as usize;
        rank
    }

    #[inline]
    pub fn get(&self, chain_id: usize) -> f32 {
        if !self.contains(chain_id) {
            return 0.0;
        }

        let idx = self.rank1(chain_id);
        self.values[idx]
    }
}

pub fn build_rank_lut(bits: &[u64]) -> Vec<u32> {
    let block_count = (bits.len() + RANK_BLOCK_WORDS - 1) / RANK_BLOCK_WORDS;
    let mut lut = Vec::<u32>::with_capacity(block_count + 1);

    let mut prefix = 0u32;

    for block in 0..block_count {
        lut.push(prefix);

        let start = block * RANK_BLOCK_WORDS;
        let end = ((block + 1) * RANK_BLOCK_WORDS).min(bits.len());

        for &w in &bits[start..end] {
            prefix += w.count_ones();
        }
    }

    lut.push(prefix);
    lut
}

pub fn make_dense_block(field_id: FieldId, chains: &[ChainId], init_value: f32) -> DenseBlock {
    let word_count = (CHAIN_COUNT + 63) / 64;
    let mut bits = vec![0u64; word_count];

    for &cid in chains {
        let c = cid as usize;
        debug_assert!(c < CHAIN_COUNT);

        let word = c >> 6;
        let bit = c & 63;
        bits[word] |= 1u64 << bit;
    }

    let rank_lut = build_rank_lut(&bits);
    let values = vec![init_value; chains.len()];

    DenseBlock {
        field_id,
        bits,
        rank_lut,
        values,
    }
}

pub fn iter_dense_chain_ids(block: &DenseBlock) -> Vec<ChainId> {
    let mut out = Vec::<ChainId>::with_capacity(block.values.len());

    for (word_idx, &word_value) in block.bits.iter().enumerate() {
        let mut w = word_value;

        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            let cid = word_idx * 64 + bit;

            if cid < CHAIN_COUNT {
                out.push(cid as ChainId);
            }

            w &= w - 1;
        }
    }

    debug_assert_eq!(out.len(), block.values.len());
    out
}

pub fn make_sparse_block(field_id: FieldId, chains: Vec<ChainId>, init_value: f32) -> SparseBlock {
    let count = chains.len();

    SparseBlock {
        field_id,
        chains,
        values: vec![init_value; count],
    }
}

pub fn make_adaptive_block(field_id: FieldId, chains: Vec<ChainId>, init_value: f32) -> FieldBlock {
    if chains.len() >= DENSE_THRESHOLD {
        FieldBlock::Dense(make_dense_block(field_id, &chains, init_value))
    } else {
        FieldBlock::Sparse(make_sparse_block(field_id, chains, init_value))
    }
}

/// `keys`는 field-major sorted SolverKey라고 가정한다.
/// 기존 `solver_key` packing에서는 field가 상위 bit라 전체 sort면 field-major가 된다.
pub fn solver_keys_to_field_layer(
    keys: &[u64],
    state_index: &StateIndex,
    unpack_solver_key: impl Fn(u64) -> (FieldId, ChainState),
    init_value: f32,
) -> FieldLayer {
    let mut blocks = Vec::<FieldBlock>::new();
    let mut total_states = 0usize;

    let mut i = 0usize;

    while i < keys.len() {
        let (field_id, _) = unpack_solver_key(keys[i]);

        let start = i;

        while i < keys.len() {
            let (f, _) = unpack_solver_key(keys[i]);
            if f != field_id {
                break;
            }
            i += 1;
        }

        let end = i;
        let mut chains = Vec::<ChainId>::with_capacity(end - start);

        for &key in &keys[start..end] {
            let (_f, chain) = unpack_solver_key(key);

            let cid = *state_index
                .key_to_id
                .get(&state_key(chain))
                .expect("chain missing in StateIndex");

            chains.push(cid as ChainId);
        }

        chains.sort_unstable();
        chains.dedup();

        total_states += chains.len();

        blocks.push(make_adaptive_block(field_id, chains, init_value));
    }

    blocks.sort_by_key(|b| b.field_id());

    FieldLayer {
        blocks,
        total_states,
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

#[inline]
pub fn shift_reveal(old_q: [u8; 6], revealed: u8) -> u32 {
    encode_queue6([
        old_q[1],
        old_q[2],
        old_q[3],
        old_q[4],
        old_q[5],
        revealed,
    ])
}

#[inline]
pub fn forward_chain_after_reveal(
    old: ChainState,
    use_hold: bool,
    revealed: u8,
) -> Option<ChainState> {
    let bit = 1u8 << revealed;

    if (old.bag_mask & bit) == 0 {
        return None;
    }

    let old_q = decode_queue6(old.queue6);
    let old_front = old_q[0];

    let new_hold = if use_hold {
        old_front
    } else {
        old.hold
    };

    let new_queue6 = shift_reveal(old_q, revealed);
    let new_bag = after_reveal_forward(old.bag_mask, revealed);

    let new_chain = ChainState {
        hold: new_hold,
        queue6: new_queue6,
        bag_mask: new_bag,
    };

    if is_valid_state(new_chain) {
        Some(new_chain)
    } else {
        None
    }
}

pub fn enumerate_actions(
    graph: &FieldGraph,
    field: FieldId,
    chain: ChainState,
    out: &mut Vec<MoveAction>,
) {
    out.clear();

    let q = decode_queue6(chain.queue6);
    let front = q[0];

    // Place queue front.
    for &target in graph.edges(field, front) {
        out.push(MoveAction {
            placed: front,
            use_hold: false,
            target_field: target,
        });
    }

    // Place hold.
    let hold = chain.hold;

    for &target in graph.edges(field, hold) {
        out.push(MoveAction {
            placed: hold,
            use_hold: true,
            target_field: target,
        });
    }
}

pub fn backup_state_by_parts(
    graph: &FieldGraph,
    state_index: &StateIndex,
    field: FieldId,
    chain: ChainState,
    next: &dyn LayerLookup,
    action_buf: &mut Vec<MoveAction>,
) -> f32 {
    enumerate_actions(graph, field, chain, action_buf);

    if action_buf.is_empty() {
        return 0.0;
    }

    let reveal_count = chain.bag_mask.count_ones() as f32;
    debug_assert!(reveal_count >= 1.0);

    let mut best = 0.0f32;

    for &action in action_buf.iter() {
        let mut sum = 0.0f32;

        for revealed in pieces(chain.bag_mask) {
            let Some(next_chain) =
                forward_chain_after_reveal(chain, action.use_hold, revealed)
            else {
                continue;
            };

            let chain_key = state_key(next_chain);

            let Some(&next_cid) = state_index.key_to_id.get(&chain_key) else {
                continue;
            };

            let v = next.get(action.target_field, next_cid as ChainId);
            sum += v;
        }

        let q = sum / reveal_count;

        if q > best {
            best = q;
        }
    }

    best
}

pub fn backup_field_layer(
    graph: &FieldGraph,
    state_index: &StateIndex,
    current: &FieldLayer,
    next: &dyn LayerLookup,
) -> FieldLayer {
    let blocks = current
        .blocks
        .par_iter()
        .map(|block| backup_block(graph, state_index, block, next))
        .collect::<Vec<_>>();

    let total_states = blocks.iter().map(|b| b.active_count()).sum();

    FieldLayer {
        blocks,
        total_states,
    }
}

fn backup_block(
    graph: &FieldGraph,
    state_index: &StateIndex,
    block: &FieldBlock,
    next: &dyn LayerLookup,
) -> FieldBlock {
    match block {
        FieldBlock::Sparse(b) => {
            let mut values = Vec::<f32>::with_capacity(b.chains.len());
            let mut action_buf = Vec::<MoveAction>::new();

            for &cid in &b.chains {
                let chain = state_index.states[cid as usize];

                let v = backup_state_by_parts(
                    graph,
                    state_index,
                    b.field_id,
                    chain,
                    next,
                    &mut action_buf,
                );

                values.push(v);
            }

            FieldBlock::Sparse(SparseBlock {
                field_id: b.field_id,
                chains: b.chains.clone(),
                values,
            })
        }

        FieldBlock::Dense(b) => {
            let chain_ids = iter_dense_chain_ids(b);
            let mut values = Vec::<f32>::with_capacity(chain_ids.len());
            let mut action_buf = Vec::<MoveAction>::new();

            for cid in chain_ids {
                let chain = state_index.states[cid as usize];

                let v = backup_state_by_parts(
                    graph,
                    state_index,
                    b.field_id,
                    chain,
                    next,
                    &mut action_buf,
                );

                values.push(v);
            }

            FieldBlock::Dense(DenseBlock {
                field_id: b.field_id,
                bits: b.bits.clone(),
                rank_lut: b.rank_lut.clone(),
                values,
            })
        }
    }
}

pub fn extract_start_values(
    layer0_values: &FieldLayer,
    start_field: FieldId,
    out_u: &mut [f32],
) {
    out_u.fill(0.0);

    let Some(block) = layer0_values.find_block(start_field) else {
        return;
    };

    match block {
        FieldBlock::Sparse(b) => {
            for (&cid, &v) in b.chains.iter().zip(&b.values) {
                out_u[cid as usize] = out_u[cid as usize].max(v);
            }
        }

        FieldBlock::Dense(b) => {
            let chain_ids = iter_dense_chain_ids(b);

            for (i, cid) in chain_ids.into_iter().enumerate() {
                out_u[cid as usize] = out_u[cid as usize].max(b.values[i]);
            }
        }
    }
}

pub fn print_u_stats(label: &str, u: &[f32]) {
    let mut nonzero = 0usize;
    let mut sum = 0.0f64;
    let mut max_v = 0.0f32;

    for &v in u {
        if v > 0.0 {
            nonzero += 1;
        }

        sum += v as f64;

        if v > max_v {
            max_v = v;
        }
    }

    println!("{label} u_nonzero={nonzero}");
    println!("{label} u_mean={:.9}", sum / u.len() as f64);
    println!("{label} u_max={:.9}", max_v);
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}