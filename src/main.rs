#![allow(dead_code)]

mod action_avg_cache;
mod action_cache_probe;
mod compact_field;
mod compact_field_probe;
mod greedy_cache_probe;
#[cfg(test)]
mod greedy_action_cache_tests;
mod external_layer;
mod f32_ram;
mod graph;
mod layer_store;
mod piece;
mod qexternal;
mod qstate;
mod qgraph;
mod rgraph;
mod state;
mod sweep_schedule;
mod qid_rank;
mod pre_reveal;
mod expression_dag;
mod terminal_l9;
mod certified_bounds;
mod boundary_dag;
mod reduced_solver;
mod failure_deflation;
mod boundary_commands;
mod boundary_probe;
mod loop1_terminal;
mod loop1_prepared;
mod loop1_prepared_l8;
#[cfg(test)]
mod loop1_prepared_tests;
#[cfg(test)]
mod full_l8_tests;
mod loop1_stream;
mod loop1_probe;
mod loop1_sparse;
mod loop1_whole_operator;
mod full_l8_probe;
#[cfg(unix)]
mod index_io_probe;

#[cfg(test)]
mod loop1_stream_tests;
#[cfg(test)]
mod loop1_real_tests;

#[cfg(test)]
mod boundary_real_tests;

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(test)]
mod fixed_policy_tests;
#[cfg(test)]
mod realdata_validation_tests;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use external_layer::{external_step, sample_external, ExternalConfig};
use qexternal::{q_pair_key, build_q_index_from_old, convert_layer_to_q, external_step_q_class, external_step_q_class_fieldreduce, external_step_q_class_fieldreduce_preq, external_step_q_preq_frontier_direct, external_step_q_preq_frontier_spill, external_step_q_preq_frontier_spill_target_shard, QPreqTargetShardConfig, external_step_q_live, external_step_q_live_frontier_spill, external_step_q_live_frontier_spill_deferred, external_step_q_live_frontier_spill_strong, qlive_frontier_spill_sim, qlive_greedy_order_stats, QLiveFrontierSpillConfig, gen_prev_q_layer_masked, make_q_terminal_layer, print_q_product_layer_stats, sample_external_q_masked, save_q_pair_layer, QExternalClassConfig};
use qstate::{decode_avail_pair, decode_tail5, print_qstate_stats, QStateIndex, enumerate_qstates, encode_avail_pair, encode_tail5, qchain_key, q_forward_after_reveal, QChainState};
use qgraph::{
    build_qforward_graph, build_qrev_class_graph, print_qforward_graph_stats,
    load_qrev_class_graph, print_qrev_class_graph_stats, save_qforward_graph, save_qrev_class_graph, QForwardGraph,
};
use graph::{FieldId, FieldGraph, EMPTY_FIELD_ID, NUM_FIELDS, PIECE_COUNT};
use piece::{pieces, FULL_BAG, mask_to_string, piece_char, piece_from_char};
use rgraph::{build_reverse_graph, load_reverse_graph, save_reverse_graph, ReverseGraph};
use state::{
    decode_queue6, encode_queue6, enumerate_states, is_valid_state, state_key, ChainState,
    StateIndex,
};

pub type SolverKey = u64;

const QUEUE_BITS: u64 = 17;
const BAG_SHIFT: u64 = QUEUE_BITS;
const CHECK_QSTATE_COUNT: usize = 817_740;
const HOLD_SHIFT: u64 = BAG_SHIFT + 7;
const FIELD_SHIFT: u64 = HOLD_SHIFT + 3;

const FIELD_ID_BITS: u64 = 24;
const SHARD_BITS: u64 = 12;
const SHARDS: usize = 1usize << SHARD_BITS;

const PAR_CHUNK_KEYS: usize = 250_000;
const SHARD_TAIL_FLUSH_KEYS: usize = 100_000;
const STOP_AFTER_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy)]
struct LayerGenStats {
    raw: u64,
    deduped: usize,
    removed_dupes: u64,
}

#[inline]
pub fn solver_key(field: FieldId, chain: ChainState) -> SolverKey {
    debug_assert!((field as usize) < NUM_FIELDS);
    debug_assert!(chain.hold < 7);
    debug_assert!(chain.bag_mask > 0 && chain.bag_mask < 128);
    debug_assert!(chain.queue6 < 7u32.pow(6));

    ((field as u64) << FIELD_SHIFT)
        | ((chain.hold as u64) << HOLD_SHIFT)
        | ((chain.bag_mask as u64) << BAG_SHIFT)
        | chain.queue6 as u64
}

#[inline]
pub fn unpack_solver_key(key: SolverKey) -> (FieldId, ChainState) {
    let queue6 = (key & ((1u64 << QUEUE_BITS) - 1)) as u32;
    let bag_mask = ((key >> BAG_SHIFT) & 0x7F) as u8;
    let hold = ((key >> HOLD_SHIFT) & 0x07) as u8;
    let field = (key >> FIELD_SHIFT) as FieldId;

    (field, ChainState { hold, queue6, bag_mask })
}

#[inline]
fn ordered_shard_of_key(key: SolverKey) -> usize {
    let shift = FIELD_SHIFT + (FIELD_ID_BITS - SHARD_BITS);
    ((key >> shift) as usize) & (SHARDS - 1)
}

#[inline]
fn reverse_bag(new_bag: u8, revealed: u8) -> Option<u8> {
    let bit = 1u8 << revealed;

    if new_bag == FULL_BAG {
        Some(bit)
    } else if (new_bag & bit) != 0 {
        None
    } else {
        Some(new_bag | bit)
    }
}

#[inline]
fn prepend_front(front: u8, new_q: [u8; 6]) -> u32 {
    encode_queue6([front, new_q[0], new_q[1], new_q[2], new_q[3], new_q[4]])
}

/// Return `(placed_piece, previous_chain)`.
fn reverse_chain_state(new_chain: ChainState, out: &mut Vec<(u8, ChainState)>) {
    out.clear();

    let new_q = decode_queue6(new_chain.queue6);
    let revealed = new_q[5];

    let Some(old_bag) = reverse_bag(new_chain.bag_mask, revealed) else {
        return;
    };

    // Case A: previous front was placed.
    for placed in 0u8..PIECE_COUNT as u8 {
        let old_queue6 = prepend_front(placed, new_q);

        let old = ChainState {
            hold: new_chain.hold,
            queue6: old_queue6,
            bag_mask: old_bag,
        };

        if is_valid_state(old) {
            out.push((placed, old));
        }
    }

    // Case B: previous hold was placed.
    let old_front = new_chain.hold;
    let old_queue6 = prepend_front(old_front, new_q);

    for placed in 0u8..PIECE_COUNT as u8 {
        if placed == old_front {
            continue;
        }

        let old = ChainState {
            hold: placed,
            queue6: old_queue6,
            bag_mask: old_bag,
        };

        if is_valid_state(old) {
            out.push((placed, old));
        }
    }
}

fn make_terminal_layer_from_state_index(
    terminal_field: FieldId,
    state_index: &StateIndex,
) -> Vec<SolverKey> {
    let mut keys = Vec::<SolverKey>::with_capacity(state_index.states.len());

    for &chain in &state_index.states {
        keys.push(solver_key(terminal_field, chain));
    }

    keys.sort_unstable();
    keys.dedup();

    keys
}

fn merge_sorted_unique_vecs(a: Vec<SolverKey>, b: Vec<SolverKey>) -> Vec<SolverKey> {
    if a.is_empty() {
        return b;
    }

    if b.is_empty() {
        return a;
    }

    let mut out = Vec::<SolverKey>::with_capacity(a.len() + b.len());

    let mut i = 0usize;
    let mut j = 0usize;
    let mut last: Option<SolverKey> = None;

    while i < a.len() && j < b.len() {
        let x = if a[i] <= b[j] {
            let v = a[i];
            i += 1;
            v
        } else {
            let v = b[j];
            j += 1;
            v
        };

        if last != Some(x) {
            out.push(x);
            last = Some(x);
        }
    }

    while i < a.len() {
        let x = a[i];
        i += 1;

        if last != Some(x) {
            out.push(x);
            last = Some(x);
        }
    }

    while j < b.len() {
        let x = b[j];
        j += 1;

        if last != Some(x) {
            out.push(x);
            last = Some(x);
        }
    }

    out
}

#[derive(Debug)]
struct DedupShard {
    sorted: Vec<SolverKey>,
    tail: Vec<SolverKey>,
}

impl DedupShard {
    fn new() -> Self {
        Self { sorted: Vec::new(), tail: Vec::new() }
    }

    #[inline]
    fn push(&mut self, key: SolverKey) {
        self.tail.push(key);
    }

    #[inline]
    fn tail_len(&self) -> usize {
        self.tail.len()
    }

    #[inline]
    fn len(&self) -> usize {
        self.sorted.len() + self.tail.len()
    }

    fn flush(&mut self) {
        if self.tail.is_empty() {
            return;
        }

        self.tail.sort_unstable();
        self.tail.dedup();

        let tail = std::mem::take(&mut self.tail);
        let sorted = std::mem::take(&mut self.sorted);
        self.sorted = merge_sorted_unique_vecs(sorted, tail);
    }

    fn merge_sorted_vec(&mut self, other_sorted: Vec<SolverKey>) {
        if other_sorted.is_empty() {
            return;
        }

        self.flush();

        let sorted = std::mem::take(&mut self.sorted);
        self.sorted = merge_sorted_unique_vecs(sorted, other_sorted);
    }

    fn into_sorted(mut self) -> Vec<SolverKey> {
        self.flush();
        self.sorted
    }
}

#[derive(Debug)]
struct LayerPartial {
    raw: u64,
    shards: Vec<DedupShard>,
}

impl LayerPartial {
    fn empty() -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(DedupShard::new());
        }
        Self { raw: 0, shards }
    }

    fn stored_len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

fn merge_layer_partials(mut a: LayerPartial, mut b: LayerPartial) -> LayerPartial {
    let start = Instant::now();

    a.raw += b.raw;

    for i in 0..SHARDS {
        let other_sorted = std::mem::replace(&mut b.shards[i], DedupShard::new()).into_sorted();
        a.shards[i].merge_sorted_vec(other_sorted);
    }

    eprintln!(
        "  reduce merge raw={} stored={} elapsed={:.3}s",
        a.raw,
        a.stored_len(),
        start.elapsed().as_secs_f64()
    );

    a
}

fn process_chunk_to_partial(
    rgraph: &ReverseGraph,
    chunk_idx: usize,
    total_chunks: usize,
    chunk: &[SolverKey],
    done_chunks: &AtomicUsize,
    global_start: Instant,
) -> LayerPartial {
    let chunk_start = Instant::now();

    let mut partial = LayerPartial::empty();
    let mut rev_buf = Vec::<(u8, ChainState)>::new();

    for &next_key in chunk {
        let (next_field, next_chain) = unpack_solver_key(next_key);

        reverse_chain_state(next_chain, &mut rev_buf);

        for &(placed, old_chain) in &rev_buf {
            for &old_field in rgraph.edges(next_field, placed) {
                let key = solver_key(old_field, old_chain);
                let shard = ordered_shard_of_key(key);

                partial.shards[shard].push(key);
                partial.raw += 1;

                if partial.shards[shard].tail_len() >= SHARD_TAIL_FLUSH_KEYS {
                    partial.shards[shard].flush();
                }
            }
        }
    }

    let flush_start = Instant::now();

    for shard in &mut partial.shards {
        shard.flush();
    }

    let flush_elapsed = flush_start.elapsed().as_secs_f64();

    let done = done_chunks.fetch_add(1, Ordering::Relaxed) + 1;
    let stored = partial.stored_len();
    let ratio = if partial.raw == 0 { 0.0 } else { stored as f64 / partial.raw as f64 };

    eprintln!(
        "  chunk {}/{} done_global={}/{} raw={} stored={} ratio={:.3} chunk_elapsed={:.1}s flush={:.3}s total_elapsed={:.1}s",
        chunk_idx + 1,
        total_chunks,
        done,
        total_chunks,
        partial.raw,
        stored,
        ratio,
        chunk_start.elapsed().as_secs_f64(),
        flush_elapsed,
        global_start.elapsed().as_secs_f64()
    );

    partial
}

fn gen_prev_good_layer_parallel_dedup(
    rgraph: &ReverseGraph,
    next_good: &[SolverKey],
    prev_good: &mut Vec<SolverKey>,
) -> LayerGenStats {
    let start = Instant::now();
    let total_chunks = (next_good.len() + PAR_CHUNK_KEYS - 1) / PAR_CHUNK_KEYS;

    eprintln!(
        "parallel layer gen chunks={} chunk_size={} threads={} shards={} tail_flush={}",
        total_chunks,
        PAR_CHUNK_KEYS,
        rayon::current_num_threads(),
        SHARDS,
        SHARD_TAIL_FLUSH_KEYS
    );

    let done_chunks = AtomicUsize::new(0);

    let partial = next_good
        .par_chunks(PAR_CHUNK_KEYS)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            process_chunk_to_partial(
                rgraph,
                chunk_idx,
                total_chunks,
                chunk,
                &done_chunks,
                start,
            )
        })
        .reduce(LayerPartial::empty, merge_layer_partials);

    let mut shards = partial.shards;
    let raw = partial.raw;

    let flush_start = Instant::now();
    for shard in &mut shards {
        shard.flush();
    }
    eprintln!("final shard flush done in {:.3}s", flush_start.elapsed().as_secs_f64());

    let deduped = shards.iter().map(|s| s.sorted.len()).sum::<usize>();
    eprintln!("final stored={deduped}");

    let collect_start = Instant::now();

    prev_good.clear();
    prev_good.reserve(deduped);

    for shard in shards {
        prev_good.extend(shard.into_sorted());
    }

    debug_assert_eq!(prev_good.len(), deduped);
    debug_assert!(prev_good.windows(2).all(|w| w[0] < w[1]));

    eprintln!(
        "collected ordered shards into prev_good in {:.3}s",
        collect_start.elapsed().as_secs_f64()
    );

    eprintln!(
        "parallel layer gen total elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );

    LayerGenStats { raw, deduped, removed_dupes: raw - deduped as u64 }
}

fn print_product_layer_field_stats(keys: &[SolverKey], depth: usize) {
    if keys.is_empty() {
        println!("product layer {depth} graph_nodes=0");
        println!("product layer {depth} success_field_keys=0");
        println!("product layer {depth} field_range=None");
        return;
    }

    let mut graph_nodes = 0usize;
    let mut last_field: Option<FieldId> = None;

    let mut success_field_keys = 0usize;
    let mut min_field = FieldId::MAX;
    let mut max_field = 0u32;

    for &key in keys {
        let (field, _) = unpack_solver_key(key);

        if last_field != Some(field) {
            graph_nodes += 1;
            last_field = Some(field);
        }

        if field == EMPTY_FIELD_ID {
            success_field_keys += 1;
        }

        min_field = min_field.min(field);
        max_field = max_field.max(field);
    }

    println!("product layer {depth} graph_nodes={graph_nodes}");
    println!("product layer {depth} success_field_keys={success_field_keys}");
    println!("product layer {depth} field_range={min_field}..={max_field}");
}

fn write_u24_le<W: Write>(w: &mut W, x: u32) -> Result<()> {
    debug_assert!(x < (1u32 << 24));
    let b = [(x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8, ((x >> 16) & 0xFF) as u8];
    w.write_all(&b)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct FieldChainBlockMeta {
    field_id: FieldId,
    count: u32,
    chain_offset: u64,
}

fn estimate_field_chain_u24_size(keys: &[SolverKey]) -> (usize, usize, u64) {
    let mut blocks = 0usize;
    let mut total_states = 0usize;

    let mut i = 0usize;

    while i < keys.len() {
        let (field, _) = unpack_solver_key(keys[i]);

        blocks += 1;

        while i < keys.len() {
            let (f, _) = unpack_solver_key(keys[i]);
            if f != field {
                break;
            }
            total_states += 1;
            i += 1;
        }
    }

    let chain_bytes = total_states as u64 * 3;
    let index_bytes = 24u64 + blocks as u64 * 16u64;

    (blocks, total_states, index_bytes + chain_bytes)
}

fn print_field_chain_save_estimate(label: &str, keys: &[SolverKey]) {
    let (blocks, states, total_bytes) = estimate_field_chain_u24_size(keys);

    let chain_bytes = states as u64 * 3;
    let fp16_value_bytes = states as u64 * 2;
    let u24_plus_fp16 = chain_bytes + fp16_value_bytes;

    println!("{label} field_chain blocks={blocks}");
    println!("{label} field_chain states={states}");

    println!(
        "{label} u24 membership bytes={} ({:.3} GiB)",
        chain_bytes,
        chain_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

    println!(
        "{label} index+u24 membership bytes={} ({:.3} GiB)",
        total_bytes,
        total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

    println!(
        "{label} future fp16 values bytes={} ({:.3} GiB)",
        fp16_value_bytes,
        fp16_value_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

    println!(
        "{label} future u24+fp16 bytes={} ({:.3} GiB)",
        u24_plus_fp16,
        u24_plus_fp16 as f64 / 1024.0 / 1024.0 / 1024.0
    );
}

/// Save files:
///   {prefix}.field_index.bin
///   {prefix}.chains.u24.bin
///
/// index format:
///   magic:        8 bytes = b"FCHNIDX1"
///   version:      u32 LE = 1
///   block_count:  u32 LE
///   total_states: u64 LE
///   blocks:
///     field_id:     u32 LE
///     count:        u32 LE
///     chain_offset: u64 LE, element offset in chains.u24.bin
fn save_field_chain_layer_u24(
    prefix: impl AsRef<Path>,
    keys: &[SolverKey],
    state_index: &StateIndex,
) -> Result<()> {
    let prefix = prefix.as_ref();

    let index_path = prefix.with_extension("field_index.bin");
    let chains_path = prefix.with_extension("chains.u24.bin");

    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = chains_path.parent() {
        fs::create_dir_all(parent)?;
    }

    eprintln!("saving field-chain index: {}", index_path.display());
    eprintln!("saving field-chain chains: {}", chains_path.display());

    let index_file = File::create(&index_path)
        .with_context(|| format!("failed to create {}", index_path.display()))?;
    let chains_file = File::create(&chains_path)
        .with_context(|| format!("failed to create {}", chains_path.display()))?;

    let mut index_writer = BufWriter::new(index_file);
    let mut chains_writer = BufWriter::new(chains_file);

    let mut metas = Vec::<FieldChainBlockMeta>::new();
    let mut chain_offset = 0u64;

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
        let mut chain_ids = Vec::<u32>::with_capacity(end - start);

        for &key in &keys[start..end] {
            let (_field, chain) = unpack_solver_key(key);

            let cid = *state_index
                .key_to_id
                .get(&state_key(chain))
                .expect("chain missing in StateIndex") as u32;

            debug_assert!(cid < (1u32 << 24));
            chain_ids.push(cid);
        }

        chain_ids.sort_unstable();
        chain_ids.dedup();

        let count = chain_ids.len();

        for cid in chain_ids {
            write_u24_le(&mut chains_writer, cid)?;
        }

        metas.push(FieldChainBlockMeta { field_id, count: count as u32, chain_offset });
        chain_offset += count as u64;
    }

    chains_writer.flush()?;

    index_writer.write_all(b"FCHNIDX1")?;
    index_writer.write_all(&1u32.to_le_bytes())?;
    index_writer.write_all(&(metas.len() as u32).to_le_bytes())?;
    index_writer.write_all(&chain_offset.to_le_bytes())?;

    for meta in &metas {
        index_writer.write_all(&meta.field_id.to_le_bytes())?;
        index_writer.write_all(&meta.count.to_le_bytes())?;
        index_writer.write_all(&meta.chain_offset.to_le_bytes())?;
    }

    index_writer.flush()?;

    eprintln!("saved {} blocks, {} states", metas.len(), chain_offset);

    Ok(())
}

fn count_pc_field_layers_from_success(rgraph: &ReverseGraph) {
    let mut current = vec![EMPTY_FIELD_ID];

    current.sort_unstable();
    current.dedup();

    println!("pc layer 10 / reverse step 0: raw=1 nodes={}", current.len());

    let mut total_nodes = current.len();

    for step in 1..=10usize {
        let mut prev = Vec::<FieldId>::new();

        for &next_field in &current {
            for placed in 0u8..PIECE_COUNT as u8 {
                for &old_field in rgraph.edges(next_field, placed) {
                    prev.push(old_field);
                }
            }
        }

        let raw = prev.len();
        prev.sort_unstable();
        prev.dedup();

        let pc_layer = 10usize - step;

        println!(
            "pc layer {pc_layer} / reverse step {step}: raw={raw} nodes={}",
            prev.len()
        );

        total_nodes += prev.len();
        current = prev;
    }

    println!("pc field layers total_nodes={total_nodes}");
}

fn load_or_build_rgraph(graph: &FieldGraph, rgraph_path: &str) -> Result<ReverseGraph> {
    if Path::new(rgraph_path).exists() {
        let s = Instant::now();
        eprintln!("loading reverse graph from {rgraph_path}...");

        match load_reverse_graph(rgraph_path) {
            Ok(rg) => {
                eprintln!("loaded reverse graph in {:.3}s", s.elapsed().as_secs_f64());
                return Ok(rg);
            }
            Err(err) => {
                eprintln!("warning: failed to load reverse graph: {err:#}");
                eprintln!("rebuilding reverse graph and overwriting {rgraph_path}...");
            }
        }
    }

    let s = Instant::now();
    let rg = build_reverse_graph(graph);
    eprintln!("built reverse graph in {:.3}s", s.elapsed().as_secs_f64());

    eprintln!("saving reverse graph to {rgraph_path}...");
    save_reverse_graph(rgraph_path, &rg)
        .with_context(|| format!("failed to save reverse graph: {rgraph_path}"))?;

    Ok(rg)
}


fn run_build_prefix(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        eprintln!("usage: {} build-prefix <graph.bin> <rgraph.bin> <out_dir>", args[0]);
        eprintln!("   or: {} <graph.bin> <rgraph.bin> [out_dir]", args[0]);
        std::process::exit(2);
    }

    let graph_path = &args[1];
    let rgraph_path = &args[2];
    let out_dir = args.get(3).map(String::as_str).unwrap_or(".");
    let out_dir = PathBuf::from(out_dir);
    fs::create_dir_all(&out_dir)?;

    let start = Instant::now();
    eprintln!("loading graph from {graph_path}...");
    let graph = FieldGraph::load_graph_bin(graph_path)
        .with_context(|| format!("failed to load graph: {graph_path}"))?;
    eprintln!("loaded graph in {:.3}s", start.elapsed().as_secs_f64());

    let summary = graph.summary();
    println!("num_fields={}", summary.num_fields);
    println!("total_edges={}", summary.total_edges);
    println!("empty_field_id={}", EMPTY_FIELD_ID);
    println!("empty_hash={}", graph.hash(EMPTY_FIELD_ID));
    println!("edge_count_by_piece={:?}", summary.edge_count_by_piece);
    println!("max_degree_by_piece={:?}", summary.max_degree_by_piece);

    let rgraph = load_or_build_rgraph(&graph, rgraph_path)?;
    let rsummary = rgraph.summary();
    println!("reverse total_edges={}", rsummary.total_edges);
    println!("reverse edge_count_by_piece={:?}", rsummary.edge_count_by_piece);
    println!("reverse max_degree_by_piece={:?}", rsummary.max_degree_by_piece);
    assert_eq!(summary.total_edges, rsummary.total_edges);
    assert_eq!(summary.edge_count_by_piece, rsummary.edge_count_by_piece);
    eprintln!("reverse graph check: ok");

    count_pc_field_layers_from_success(&rgraph);

    let state_index = enumerate_states();
    println!("chain_states={}", state_index.states.len());

    let mut good = make_terminal_layer_from_state_index(EMPTY_FIELD_ID, &state_index);
    println!("product layer 10 good={}", good.len());
    print_product_layer_field_stats(&good, 10);

    for depth in [9usize, 8usize] {
        eprintln!("generating product layer {depth} good with parallel ordered-shard dedupe...");
        let layer_start = Instant::now();
        let mut prev_good = Vec::<SolverKey>::new();
        let stats = gen_prev_good_layer_parallel_dedup(&rgraph, &good, &mut prev_good);
        eprintln!("generated product layer {depth} good in {:.3}s", layer_start.elapsed().as_secs_f64());
        println!("product layer {depth} raw={}", stats.raw);
        println!("product layer {depth} good={}", stats.deduped);
        println!("product layer {depth} removed_dupes={}", stats.removed_dupes);
        print_product_layer_field_stats(&prev_good, depth);

        let label = format!("layer{depth}");
        print_field_chain_save_estimate(&label, &prev_good);
        let save_start = Instant::now();
        let prefix = out_dir.join(&label);
        save_field_chain_layer_u24(&prefix, &prev_good, &state_index)
            .with_context(|| format!("failed to save {label} field-chain layer"))?;
        eprintln!("saved {label} field-chain layer in {:.3}s", save_start.elapsed().as_secs_f64());
        good = prev_good;
    }

    eprintln!("stopping after saving layer 8");
    Ok(())
}

fn run_build_rgraph(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        eprintln!("usage: {} build-rgraph <graph.bin> <rgraph.bin>", args[0]);
        std::process::exit(2);
    }

    let graph = FieldGraph::load_graph_bin(&args[2])
        .with_context(|| format!("failed to load graph: {}", args[2]))?;
    let summary = graph.summary();
    let rgraph = load_or_build_rgraph(&graph, &args[3])?;
    let rsummary = rgraph.summary();
    anyhow::ensure!(summary.total_edges == rsummary.total_edges, "reverse graph edge count mismatch");
    anyhow::ensure!(summary.edge_count_by_piece == rsummary.edge_count_by_piece, "reverse graph piece counts mismatch");
    eprintln!("build-rgraph ok: fields={} edges={} output={}", summary.num_fields, summary.total_edges, args[3]);
    Ok(())
}

fn run_stats_layer(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        eprintln!("usage: {} stats-layer <layer_prefix>", args[0]);
        std::process::exit(2);
    }
    let layer = layer_store::FieldChainLayer::open(&args[2])?;
    layer.print_stats(&args[2]);
    Ok(())
}


#[inline]
fn validate_sorted_unique_qids(layer: usize, label: &str, field: FieldId, qids: &[u32]) -> Result<()> {
    let mut prev: Option<u32> = None;
    for &qid in qids {
        if qid as usize >= CHECK_QSTATE_COUNT {
            anyhow::bail!(
                "L{} {} field={} has out-of-range qid={} >= {}",
                layer,
                label,
                field,
                qid,
                CHECK_QSTATE_COUNT
            );
        }
        if let Some(p) = prev {
            if qid <= p {
                anyhow::bail!(
                    "L{} {} field={} qids are not strictly sorted/unique: prev={} cur={}",
                    layer,
                    label,
                    field,
                    p,
                    qid
                );
            }
        }
        prev = Some(qid);
    }
    Ok(())
}

fn diff_sample_sorted(a: &[u32], b: &[u32], out: &mut Vec<u32>, limit: usize) {
    let mut j = 0usize;
    for &av in a {
        while j < b.len() && b[j] < av {
            j += 1;
        }
        if j == b.len() || b[j] != av {
            out.push(av);
            if out.len() >= limit {
                return;
            }
        }
    }
}

fn validate_subset_sorted(layer: usize, field: FieldId, sub: &[u32], sup: &[u32]) -> Result<()> {
    let mut j = 0usize;
    for &q in sub {
        while j < sup.len() && sup[j] < q {
            j += 1;
        }
        if j == sup.len() || sup[j] != q {
            anyhow::bail!(
                "L{} subset violation: forward-pruned contains state not in backward: field={} qid={}",
                layer,
                field,
                q
            );
        }
    }
    Ok(())
}


fn load_field_hashes_only(path: &Path) -> Result<Vec<u64>> {
    let mut r = BufReader::new(File::open(path).with_context(|| format!("failed to open graph bin {}", path.display()))?);
    let mut hashes = Vec::<u64>::with_capacity(NUM_FIELDS);
    let mut hbuf = [0u8; 5];
    let mut degree = [0u8; 1];

    for field in 0..NUM_FIELDS {
        r.read_exact(&mut hbuf)
            .with_context(|| format!("failed to read hash for field {} from {}", field, path.display()))?;
        let hash = ((hbuf[0] as u64) << 32)
            | ((hbuf[1] as u64) << 24)
            | ((hbuf[2] as u64) << 16)
            | ((hbuf[3] as u64) << 8)
            | (hbuf[4] as u64);
        hashes.push(hash);

        for _ in 0..PIECE_COUNT {
            r.read_exact(&mut degree)
                .with_context(|| format!("failed to read degree for field {} from {}", field, path.display()))?;
            let skip = degree[0] as i64 * 3;
            if skip != 0 {
                r.seek(SeekFrom::Current(skip))
                    .with_context(|| format!("failed to skip targets for field {} from {}", field, path.display()))?;
            }
        }
    }

    Ok(hashes)
}

fn print_field_board_from_hash(hash: u64) {
    for y in (0..4).rev() {
        print!("    ");
        for x in 0..10 {
            let bit = y * 10 + x;
            let c = if ((hash >> bit) & 1) != 0 { '1' } else { '0' };
            print!("{}", c);
        }
        println!();
    }
}

fn qchain_text(qid: u32, q_index: Option<&QStateIndex>) -> Option<String> {
    let q_index = q_index?;
    let s = *q_index.states.get(qid as usize)?;
    let (a, b) = decode_avail_pair(s.avail);
    let tail = decode_tail5(s.tail5);

    let mut avail = String::new();
    avail.push(piece_char(a));
    if b != a {
        avail.push('/');
        avail.push(piece_char(b));
    }

    let mut tail_s = String::new();
    for &p in &tail {
        tail_s.push(piece_char(p));
    }

    Some(format!(
        "avail={{{}}} tail5={} bag_mask=0b{:07b} bag={}",
        avail,
        tail_s,
        s.bag_mask,
        mask_to_string(s.bag_mask),
    ))
}

fn print_removed_state_sample(
    field: FieldId,
    qid: u32,
    field_hashes: Option<&[u64]>,
    q_index: Option<&QStateIndex>,
) {
    if let Some(hashes) = field_hashes {
        let hash = hashes[field as usize];
        println!("  field={} qid={} hash=0x{:010X}", field, qid, hash);
        print_field_board_from_hash(hash);
    } else {
        println!("  field={} qid={}", field, qid);
    }

    if let Some(text) = qchain_text(qid, q_index) {
        println!("    qchain: {}", text);
    }
}

fn check_removed_states_for_layer(
    layer: usize,
    backward_dir: &Path,
    pruned_dir: &Path,
    sample_n: usize,
    field_hashes: Option<&[u64]>,
    q_index: Option<&QStateIndex>,
) -> Result<()> {
    let b_prefix = backward_dir.join(format!("layer{}q", layer));
    let f_prefix = pruned_dir.join(format!("layer{}q_pruned", layer));

    let b = layer_store::FieldChainLayer::open_index_only(&b_prefix)
        .with_context(|| format!("failed to open backward layer prefix {}", b_prefix.display()))?;
    let f = layer_store::FieldChainLayer::open_index_only(&f_prefix)
        .with_context(|| format!("failed to open forward-pruned layer prefix {}", f_prefix.display()))?;

    let mut removed_samples: Vec<(FieldId, u32)> = Vec::new();
    let mut removed_seen: u64 = 0;
    let mut f_i = 0usize;
    let mut max_removed_in_field: u32 = 0;
    let mut fields_with_removed: u64 = 0;
    let mut fields_only_in_backward: u64 = 0;

    for (b_i, bm) in b.metas.iter().enumerate() {
        if f_i < f.metas.len() && f.metas[f_i].field_id < bm.field_id {
            anyhow::bail!(
                "L{} subset violation: forward-pruned has field not present in backward: field={}",
                layer,
                f.metas[f_i].field_id
            );
        }

        if f_i < f.metas.len() && f.metas[f_i].field_id == bm.field_id {
            let fm = &f.metas[f_i];
            if fm.count > bm.count {
                anyhow::bail!(
                    "L{} impossible count: field={} forward_count={} backward_count={}",
                    layer,
                    bm.field_id,
                    fm.count,
                    bm.count
                );
            }

            let bq = b.read_block_chains(b_i);
            let fq = f.read_block_chains(f_i);
            validate_sorted_unique_qids(layer, "backward", bm.field_id, &bq)?;
            validate_sorted_unique_qids(layer, "forward", bm.field_id, &fq)?;
            validate_subset_sorted(layer, bm.field_id, &fq, &bq)?;

            let removed_here = bm.count - fm.count;
            if removed_here > 0 {
                fields_with_removed += 1;
                max_removed_in_field = max_removed_in_field.max(removed_here);
                if removed_samples.len() < sample_n {
                    let mut qs = Vec::new();
                    diff_sample_sorted(&bq, &fq, &mut qs, sample_n - removed_samples.len());
                    for q in qs {
                        removed_samples.push((bm.field_id, q));
                    }
                }
            }
            removed_seen += removed_here as u64;
            f_i += 1;
        } else {
            fields_with_removed += 1;
            fields_only_in_backward += 1;
            max_removed_in_field = max_removed_in_field.max(bm.count);
            removed_seen += bm.count as u64;
            if removed_samples.len() < sample_n {
                let bq = b.read_block_chains(b_i);
                validate_sorted_unique_qids(layer, "backward", bm.field_id, &bq)?;
                for &q in bq.iter().take(sample_n - removed_samples.len()) {
                    removed_samples.push((bm.field_id, q));
                }
            }
        }
    }

    if f_i != f.metas.len() {
        anyhow::bail!(
            "L{} subset violation: trailing forward-pruned field not present in backward: field={}",
            layer,
            f.metas[f_i].field_id
        );
    }

    let expected_removed = b.total_states.checked_sub(f.total_states).with_context(|| {
        format!(
            "L{} invalid totals: backward_states={} < forward_states={}",
            layer,
            b.total_states,
            f.total_states
        )
    })?;
    if removed_seen != expected_removed {
        anyhow::bail!(
            "L{} removed count mismatch: scanned={} expected={}",
            layer,
            removed_seen,
            expected_removed
        );
    }

    println!("===== L{} =====", layer);
    println!("backward_fields={} backward_states={}", b.metas.len(), b.total_states);
    println!("forward_fields={} forward_states={}", f.metas.len(), f.total_states);
    println!("removed_states={}", expected_removed);
    println!("fields_with_removed={}", fields_with_removed);
    println!("fields_only_in_backward={}", fields_only_in_backward);
    println!("max_removed_in_one_field={}", max_removed_in_field);
    println!("subset_check=OK");
    println!("removed_samples limit={}", sample_n);
    if removed_samples.is_empty() {
        println!("  <none>");
    } else {
        for (field, qid) in removed_samples {
            print_removed_state_sample(field, qid, field_hashes, q_index);
        }
    }
    Ok(())
}

fn run_check_forward_pruned_removed(args: &[String]) -> Result<()> {
    if args.len() < 4 || args.len() > 8 {
        eprintln!("usage: {} check-forward-pruned-removed <backward_layer_dir> <pruned_layer_dir> [sample_n=10] [start_layer=0] [end_layer=9] [graph_bin=graph.bin]", args[0]);
        std::process::exit(2);
    }
    let backward_dir = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let sample_n = args.get(4).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(10);
    let start_layer = args.get(5).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(0);
    let end_layer = args.get(6).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(9);
    let graph_path = PathBuf::from(args.get(7).map(|s| s.as_str()).unwrap_or("graph.bin"));
    eprintln!("loading field hashes from {}", graph_path.display());
    let field_hashes = load_field_hashes_only(&graph_path)?;
    eprintln!("building qchain text index");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);

    if start_layer > end_layer || end_layer > 9 {
        anyhow::bail!("invalid layer range: {}..={}", start_layer, end_layer);
    }

    for layer in start_layer..=end_layer {
        check_removed_states_for_layer(layer, &backward_dir, &pruned_dir, sample_n, Some(&field_hashes), Some(&q_index))
            .with_context(|| format!("failed while checking layer {}", layer))?;
    }
    println!("ALL_CHECKS_PASSED layers={}..={} sample_n={}", start_layer, end_layer, sample_n);
    Ok(())
}


#[derive(Clone)]
struct XorShift64 {
    x: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let x = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { x }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.x;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.x = x;
        x
    }

    fn gen_range(&mut self, n: u64) -> u64 {
        if n == 0 { return 0; }
        self.next_u64() % n
    }
}

fn find_meta_by_field(metas: &[layer_store::FieldChainBlockMeta], field: FieldId) -> Option<usize> {
    metas.binary_search_by_key(&field, |m| m.field_id).ok()
}

fn find_meta_by_global_offset(metas: &[layer_store::FieldChainBlockMeta], global_item: u64) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = metas.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let m = &metas[mid];
        let start = m.chain_offset;
        let end = start + m.count as u64;
        if global_item < start {
            hi = mid;
        } else if global_item >= end {
            lo = mid + 1;
        } else {
            return Some(mid);
        }
    }
    None
}

fn check_subset_only_and_random_removed_for_layer(
    layer: usize,
    backward_dir: &Path,
    pruned_dir: &Path,
    sample_n: usize,
    seed: u64,
    verify_subset: bool,
    field_hashes: Option<&[u64]>,
    q_index: Option<&QStateIndex>,
) -> Result<()> {
    let b_prefix = backward_dir.join(format!("layer{}q", layer));
    let f_prefix = pruned_dir.join(format!("layer{}q_pruned", layer));

    let b = layer_store::FieldChainLayer::open_index_only(&b_prefix)
        .with_context(|| format!("failed to open backward layer prefix {}", b_prefix.display()))?;
    let f = layer_store::FieldChainLayer::open_index_only(&f_prefix)
        .with_context(|| format!("failed to open forward-pruned layer prefix {}", f_prefix.display()))?;

    let expected_removed = b.total_states.checked_sub(f.total_states).with_context(|| {
        format!(
            "L{} invalid totals: backward_states={} < forward_states={}",
            layer,
            b.total_states,
            f.total_states
        )
    })?;

    // Optional correctness check needed for MDP safety: F[layer] must be subset of B[layer].
    // This scans only forward fields and the matching backward blocks, not all backward-only fields.
    // Use --no-verify when you only want quick random removed examples.
    if verify_subset {
        let mut b_i = 0usize;
        for (f_i, fm) in f.metas.iter().enumerate() {
            while b_i < b.metas.len() && b.metas[b_i].field_id < fm.field_id {
                b_i += 1;
            }
            if b_i == b.metas.len() || b.metas[b_i].field_id != fm.field_id {
                anyhow::bail!(
                    "L{} subset violation: forward-pruned has field not present in backward: field={}",
                    layer,
                    fm.field_id
                );
            }
            let bm = &b.metas[b_i];
            if fm.count > bm.count {
                anyhow::bail!(
                    "L{} impossible count: field={} forward_count={} backward_count={}",
                    layer,
                    fm.field_id,
                    fm.count,
                    bm.count
                );
            }

            let bq = b.read_block_chains(b_i);
            let fq = f.read_block_chains(f_i);
            validate_sorted_unique_qids(layer, "backward", bm.field_id, &bq)?;
            validate_sorted_unique_qids(layer, "forward", fm.field_id, &fq)?;
            validate_subset_sorted(layer, fm.field_id, &fq, &bq)?;
        }
    }

    // Random examples from B-F. This is for inspection only, not for proving counts.
    let mut removed_samples: Vec<(FieldId, u32)> = Vec::new();
    let mut rng = XorShift64::new(seed ^ ((layer as u64 + 1) * 0x9E37_79B9_7F4A_7C15));
    let max_attempts = (sample_n.max(1) as u64 * 20_000).max(20_000);
    let mut attempts = 0u64;

    while removed_samples.len() < sample_n && attempts < max_attempts && expected_removed > 0 {
        attempts += 1;
        let pos = rng.gen_range(b.total_states);
        let Some(bm_i) = find_meta_by_global_offset(&b.metas, pos) else { continue; };
        let bm = &b.metas[bm_i];
        if bm.count == 0 { continue; }
        let local = (pos - bm.chain_offset) as usize;
        let bq = b.read_block_chains(bm_i);
        if verify_subset {
            validate_sorted_unique_qids(layer, "backward", bm.field_id, &bq)?;
        }
        let qid = bq[local];

        let in_forward = if let Some(fm_i) = find_meta_by_field(&f.metas, bm.field_id) {
            let fq = f.read_block_chains(fm_i);
            if verify_subset {
                validate_sorted_unique_qids(layer, "forward", bm.field_id, &fq)?;
            }
            fq.binary_search(&qid).is_ok()
        } else {
            false
        };

        if !in_forward && !removed_samples.iter().any(|&(f0, q0)| f0 == bm.field_id && q0 == qid) {
            removed_samples.push((bm.field_id, qid));
        }
    }

    println!("===== L{} =====", layer);
    println!("backward_fields={} backward_states={}", b.metas.len(), b.total_states);
    println!("forward_fields={} forward_states={}", f.metas.len(), f.total_states);
    println!("removed_states_by_total={}", expected_removed);
    if verify_subset {
        println!("subset_check=OK");
    } else {
        println!("subset_check=SKIPPED");
    }
    println!("random_removed_samples limit={} seed={} attempts={}", sample_n, seed, attempts);
    if removed_samples.is_empty() {
        println!("  <none>");
    } else {
        for (field, qid) in removed_samples {
            print_removed_state_sample(field, qid, field_hashes, q_index);
        }
    }
    Ok(())
}

fn run_check_forward_subset_random_removed(args: &[String]) -> Result<()> {
    if args.len() < 4 || args.len() > 10 {
        eprintln!("usage: {} check-forward-subset-random-removed <backward_layer_dir> <pruned_layer_dir> [sample_n=10] [start_layer=0] [end_layer=9] [seed=1] [graph_bin=graph.bin] [--no-verify]", args[0]);
        std::process::exit(2);
    }

    let mut positional: Vec<String> = Vec::new();
    let mut verify_subset = true;
    for a in args.iter().skip(2) {
        match a.as_str() {
            "--no-verify" => verify_subset = false,
            "--verify" => verify_subset = true,
            _ => positional.push(a.clone()),
        }
    }
    if positional.len() < 2 || positional.len() > 7 {
        eprintln!("usage: {} check-forward-subset-random-removed <backward_layer_dir> <pruned_layer_dir> [sample_n=10] [start_layer=0] [end_layer=9] [seed=1] [graph_bin=graph.bin] [--no-verify]", args[0]);
        std::process::exit(2);
    }

    let backward_dir = PathBuf::from(&positional[0]);
    let pruned_dir = PathBuf::from(&positional[1]);
    let sample_n = positional.get(2).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(10);
    let start_layer = positional.get(3).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(0);
    let end_layer = positional.get(4).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(9);
    let seed = positional.get(5).map(|s| s.parse::<u64>()).transpose()?.unwrap_or(1);
    let graph_path = PathBuf::from(positional.get(6).map(|s| s.as_str()).unwrap_or("graph.bin"));

    eprintln!("verify_subset={}", verify_subset);
    eprintln!("loading field hashes from {}", graph_path.display());
    let field_hashes = load_field_hashes_only(&graph_path)?;
    eprintln!("building qchain text index");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);

    if start_layer > end_layer || end_layer > 9 {
        anyhow::bail!("invalid layer range: {}..={}", start_layer, end_layer);
    }

    for layer in start_layer..=end_layer {
        check_subset_only_and_random_removed_for_layer(layer, &backward_dir, &pruned_dir, sample_n, seed, verify_subset, Some(&field_hashes), Some(&q_index))
            .with_context(|| format!("failed while checking layer {}", layer))?;
    }
    if verify_subset {
        println!("ALL_SUBSET_CHECKS_PASSED_RANDOM_REMOVED_SAMPLED layers={}..={} sample_n={} seed={}", start_layer, end_layer, sample_n, seed);
    } else {
        println!("RANDOM_REMOVED_SAMPLED_WITHOUT_SUBSET_VERIFY layers={}..={} sample_n={} seed={}", start_layer, end_layer, sample_n, seed);
    }
    Ok(())
}

#[inline]
fn field_layer_from_hash(hash: u64) -> u32 {
    hash.count_ones() / 4
}

fn format_4line_board(hash: u64) -> String {
    let mut s = String::new();
    for y in (0..4).rev() {
        for x in 0..10 {
            let bit = y * 10 + x;
            s.push(if ((hash >> bit) & 1) != 0 { '1' } else { '0' });
        }
        if y != 0 {
            s.push('\n');
        }
    }
    s
}

fn run_dump_missing_layer_fields(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 8 {
        eprintln!("usage: {} dump-missing-layer-fields <graph.bin> <layer_prefix> <layer_num> <output_txt> [--boards N]", args[0]);
        std::process::exit(2);
    }

    let graph_path = &args[2];
    let layer_prefix = &args[3];
    let layer_num: u32 = args[4].parse().context("invalid layer_num")?;
    let output_path = PathBuf::from(&args[5]);

    let mut boards_limit = 0usize;
    if args.len() == 8 {
        if args[6] != "--boards" {
            eprintln!("usage: {} dump-missing-layer-fields <graph.bin> <layer_prefix> <layer_num> <output_txt> [--boards N]", args[0]);
            std::process::exit(2);
        }
        boards_limit = args[7].parse().context("invalid --boards N")?;
    } else if args.len() == 7 {
        eprintln!("usage: {} dump-missing-layer-fields <graph.bin> <layer_prefix> <layer_num> <output_txt> [--boards N]", args[0]);
        std::process::exit(2);
    }

    eprintln!("loading graph from {graph_path}...");
    let graph = FieldGraph::load_graph_bin(graph_path)?;
    eprintln!("loading layer index from {layer_prefix}...");
    let layer = layer_store::FieldChainLayer::open_index_only(layer_prefix)?;

    let mut present = vec![false; NUM_FIELDS];
    let mut present_wrong_layer = 0usize;
    for m in &layer.metas {
        let fid = m.field_id as usize;
        if fid >= NUM_FIELDS {
            anyhow::bail!("layer contains invalid field_id {}", m.field_id);
        }
        present[fid] = true;
        if field_layer_from_hash(graph.hash(m.field_id)) != layer_num {
            present_wrong_layer += 1;
        }
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut w = BufWriter::new(File::create(&output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?);

    writeln!(w, "# missing fields for layer {layer_num}")?;
    writeln!(w, "# columns: field_id\thash_hex\tpopcnt\tlayer")?;

    let mut all_layer = 0usize;
    let mut missing = 0usize;
    let mut boards_written = 0usize;

    for fid in 0..NUM_FIELDS {
        let hash = graph.hashes[fid];
        let pop = hash.count_ones();
        let lyr = pop / 4;
        if lyr != layer_num {
            continue;
        }
        all_layer += 1;
        if !present[fid] {
            missing += 1;
            writeln!(w, "{}\t0x{:010X}\t{}\t{}", fid, hash, pop, lyr)?;
            if boards_written < boards_limit {
                writeln!(w, "board:")?;
                for line in format_4line_board(hash).lines() {
                    writeln!(w, "  {line}")?;
                }
                writeln!(w)?;
                boards_written += 1;
            }
        }
    }
    w.flush()?;

    eprintln!("all_layer_fields={all_layer}");
    eprintln!("present_fields={}", layer.metas.len());
    eprintln!("present_wrong_layer={present_wrong_layer}");
    eprintln!("missing_fields={missing}");
    eprintln!("wrote {}", output_path.display());
    Ok(())
}

fn run_sample_external(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        eprintln!("usage: {} sample-external <rgraph.bin> <input_layer_prefix> <sample_states>", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = &args[3];
    let sample_states: u64 = args[4].parse().context("invalid sample_states")?;

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;
    eprintln!("loading layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("input");
    let state_index = enumerate_states();
    sample_external(&rgraph, &layer, &state_index, sample_states)?;
    Ok(())
}

fn run_external_step(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} external-step <rgraph.bin> <input_layer_prefix> <output_layer_prefix> <work_dir> [flush_pairs] [merge_fanin]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = PathBuf::from(&args[3]);
    let output_prefix = PathBuf::from(&args[4]);
    let work_dir = PathBuf::from(&args[5]);
    let flush_pairs = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(10_000_000usize);
    let merge_fanin = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(128usize);

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;
    eprintln!("loading input layer from {}...", input_prefix.display());
    let layer = layer_store::FieldChainLayer::open(&input_prefix)?;
    layer.print_stats("input");
    let state_index = enumerate_states();
    let config = ExternalConfig { flush_pairs, merge_fanin };
    let stats = external_step(&rgraph, &layer, &state_index, &output_prefix, &work_dir, config)?;
    println!("external input_states={}", stats.input_states);
    println!("external raw_pairs={}", stats.raw_pairs);
    println!("external run_unique_pairs={}", stats.run_unique_pairs);
    println!("external run_files={}", stats.run_files);
    println!("external final_unique_pairs={}", stats.final_unique_pairs);
    Ok(())
}

fn run_qstate_stats(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        eprintln!("usage: {} qstate-stats", args[0]);
        std::process::exit(2);
    }
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);
    Ok(())
}

fn run_qgraph_stats(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        eprintln!("usage: {} qgraph-stats", args[0]);
        std::process::exit(2);
    }
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("building qchain forward graph...");
    let start = Instant::now();
    let qfwd = build_qforward_graph(&q_index);
    eprintln!("built qchain forward graph in {:.3}s", start.elapsed().as_secs_f64());
    print_qforward_graph_stats(&q_index, &qfwd);

    eprintln!("building qchain reverse-class graph...");
    let start = Instant::now();
    let qrev = build_qrev_class_graph(&q_index);
    eprintln!("built qchain reverse-class graph in {:.3}s", start.elapsed().as_secs_f64());
    print_qrev_class_graph_stats(&q_index, &qrev);
    Ok(())
}

fn run_build_qgraph(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        eprintln!("usage: {} build-qgraph <out_prefix>", args[0]);
        std::process::exit(2);
    }
    let out_prefix = PathBuf::from(&args[2]);

    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("building qchain forward graph...");
    let start = Instant::now();
    let qfwd = build_qforward_graph(&q_index);
    eprintln!("built qchain forward graph in {:.3}s", start.elapsed().as_secs_f64());
    print_qforward_graph_stats(&q_index, &qfwd);

    let qfwd_path = out_prefix.with_extension("qfwd.bin");
    eprintln!("saving qchain forward graph to {}...", qfwd_path.display());
    save_qforward_graph(&qfwd_path, &qfwd)?;

    eprintln!("building qchain reverse-class graph...");
    let start = Instant::now();
    let qrev = build_qrev_class_graph(&q_index);
    eprintln!("built qchain reverse-class graph in {:.3}s", start.elapsed().as_secs_f64());
    print_qrev_class_graph_stats(&q_index, &qrev);

    let qrev_path = out_prefix.with_extension("qrevclass.bin");
    eprintln!("saving qchain reverse-class graph to {}...", qrev_path.display());
    save_qrev_class_graph(&qrev_path, &qrev)?;

    Ok(())
}

fn run_convert_layer_q(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        eprintln!("usage: {} convert-layer-q <old_input_layer_prefix> <q_output_layer_prefix>", args[0]);
        std::process::exit(2);
    }
    let input_prefix = &args[2];
    let output_prefix = PathBuf::from(&args[3]);

    eprintln!("loading input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("converting layer to qchain prefix {}...", output_prefix.display());
    convert_layer_to_q(&layer, &output_prefix, &q_index)
        .with_context(|| format!("failed to convert layer to qchain: {}", output_prefix.display()))?;
    Ok(())
}

fn run_sample_external_q(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        eprintln!("usage: {} sample-external-q <rgraph.bin> <q_input_layer_prefix> <sample_states>", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = &args[3];
    let sample_states: u64 = args[4].parse().context("invalid sample_states")?;

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;
    eprintln!("loading q layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("input");

    eprintln!("enumerating qchain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("building qchain reverse-class graph...");
    let qgraph = build_qrev_class_graph(&q_index);
    print_qrev_class_graph_stats(&q_index, &qgraph);
    sample_external_q_masked(&rgraph, &layer, &q_index, &qgraph, sample_states)?;
    Ok(())
}


fn run_external_step_q_class(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 10 {
        eprintln!("usage: {} external-step-q-class <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records] [merge_sub_bits]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let bucket_bits = if args.len() >= 8 { args[7].parse().context("invalid bucket_bits")? } else { 12usize };
    let flush_records = if args.len() >= 9 { args[8].parse().context("invalid flush_records")? } else { 4_000_000usize };
    let merge_sub_bits = if args.len() >= 10 { args[9].parse().context("invalid merge_sub_bits")? } else { 0usize };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QExternalClassConfig { bucket_bits, flush_records, merge_sub_bits };
    eprintln!("running external q class step output={} work_dir={} bucket_bits={} flush_records={} merge_sub_bits={}...", output_prefix.display(), work_dir.display(), bucket_bits, flush_records, merge_sub_bits);
    external_step_q_class(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}


fn run_external_step_q_class_fieldreduce(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 9 {
        eprintln!("usage: {} external-step-q-class-fieldreduce <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let bucket_bits = if args.len() >= 8 { args[7].parse().context("invalid bucket_bits")? } else { 12usize };
    let flush_records = if args.len() >= 9 { args[8].parse().context("invalid flush_records")? } else { 4_000_000usize };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QExternalClassConfig { bucket_bits, flush_records, merge_sub_bits: 0 };
    eprintln!("running external q class fieldreduce output={} work_dir={} bucket_bits={} flush_records={}...", output_prefix.display(), work_dir.display(), bucket_bits, flush_records);
    external_step_q_class_fieldreduce(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}

fn run_external_step_q_class_fieldreduce_preq(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 9 {
        eprintln!("usage: {} external-step-q-class-fieldreduce-preq <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let bucket_bits = if args.len() >= 8 { args[7].parse().context("invalid bucket_bits")? } else { 12usize };
    let flush_records = if args.len() >= 9 { args[8].parse().context("invalid flush_records")? } else { 4_000_000usize };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QExternalClassConfig { bucket_bits, flush_records, merge_sub_bits: 0 };
    eprintln!("running external q class fieldreduce-preq output={} work_dir={} bucket_bits={} flush_records={}...", output_prefix.display(), work_dir.display(), bucket_bits, flush_records);
    external_step_q_class_fieldreduce_preq(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}



fn run_external_step_q_preq_frontier_direct(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 8 {
        eprintln!("usage: {} external-step-q-preq-frontier-direct <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = &args[3];
    let output_prefix = PathBuf::from(&args[4]);
    let work_dir = PathBuf::from(&args[5]);
    let vec_to_bitset_threshold = if args.len() >= 7 { args[6].parse().context("invalid vec_to_bitset_threshold")? } else { 65536usize };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("running q preq frontier-direct output={} work_dir={} vec_to_bitset_threshold={}...", output_prefix.display(), work_dir.display(), vec_to_bitset_threshold);
    external_step_q_preq_frontier_direct(&rgraph, &layer, &q_index, &output_prefix, &work_dir, vec_to_bitset_threshold)?;
    Ok(())
}


fn run_external_step_q_preq_frontier_spill(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 9 {
        eprintln!("usage: {} external-step-q-preq-frontier-spill <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold] [hot_cap_gib] [inject_full_q_field_id]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = &args[3];
    let output_prefix = PathBuf::from(&args[4]);
    let work_dir = PathBuf::from(&args[5]);
    let vec_to_bitset_threshold = if args.len() >= 7 { args[6].parse().context("invalid vec_to_bitset_threshold")? } else { 65536usize };
    let hot_cap_gib = if args.len() >= 8 { args[7].parse().context("invalid hot_cap_gib")? } else { 24usize };
    let hot_cap_bytes = hot_cap_gib.saturating_mul(1024usize).saturating_mul(1024usize).saturating_mul(1024usize);
    let inject_full_q_field = if args.len() >= 9 { Some(args[8].parse::<FieldId>().context("invalid inject_full_q_field_id")?) } else { None };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer index from {input_prefix} without resident chains...");
    let layer = layer_store::FieldChainLayer::open_index_only(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("running q preq frontier-spill output={} work_dir={} vec_to_bitset_threshold={} hot_cap_gib={} inject_full_q_field={:?}...", output_prefix.display(), work_dir.display(), vec_to_bitset_threshold, hot_cap_gib, inject_full_q_field);
    external_step_q_preq_frontier_spill(&rgraph, &layer, &q_index, &output_prefix, &work_dir, vec_to_bitset_threshold, hot_cap_bytes, inject_full_q_field)?;
    Ok(())
}


fn run_external_step_q_preq_frontier_spill_target_shard(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 12 {
        eprintln!("usage: {} external-step-q-preq-frontier-spill-target-shard <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold] [total_hot_cap_gib] [inject_full_q_field_id|-1] [num_shards] [target_bucket_size] [refine_passes]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let input_prefix = &args[3];
    let output_prefix = PathBuf::from(&args[4]);
    let work_dir = PathBuf::from(&args[5]);
    let vec_to_bitset_threshold = if args.len() >= 7 { args[6].parse().context("invalid vec_to_bitset_threshold")? } else { 65536usize };
    let hot_cap_gib = if args.len() >= 8 { args[7].parse().context("invalid total_hot_cap_gib")? } else { 96usize };
    let total_hot_cap_bytes = hot_cap_gib.saturating_mul(1024usize).saturating_mul(1024usize).saturating_mul(1024usize);
    let inject_full_q_field = if args.len() >= 9 {
        let s = args[8].as_str();
        if s == "-1" || s.eq_ignore_ascii_case("none") { None }
        else { Some(args[8].parse::<FieldId>().context("invalid inject_full_q_field_id")?) }
    } else { None };
    let num_shards = if args.len() >= 10 { args[9].parse().context("invalid num_shards")? } else { 16usize };
    let target_bucket_size = if args.len() >= 11 { args[10].parse().context("invalid target_bucket_size")? } else { 256u32 };
    let refine_passes = if args.len() >= 12 { args[11].parse().context("invalid refine_passes")? } else { 4usize };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer index from {input_prefix} without resident chains...");
    let layer = layer_store::FieldChainLayer::open_index_only(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    let cfg = QPreqTargetShardConfig {
        vec_to_bitset_threshold,
        total_hot_cap_bytes,
        num_shards,
        target_bucket_size,
        refine_passes,
        inject_full_q_field,
    };
    eprintln!("running q preq frontier-spill-target-shard output={} work_dir={} vec_to_bitset_threshold={} hot_cap_gib={} inject_full_q_field={:?} num_shards={} target_bucket_size={} refine_passes={}...", output_prefix.display(), work_dir.display(), vec_to_bitset_threshold, hot_cap_gib, inject_full_q_field, num_shards, target_bucket_size, refine_passes);
    external_step_q_preq_frontier_spill_target_shard(&rgraph, &layer, &q_index, &output_prefix, &work_dir, cfg)?;
    Ok(())
}


fn run_q_live_frontier_spill_sim(args: &[String]) -> Result<()> {
    if args.len() < 5 || args.len() > 8 {
        eprintln!("usage: {} q-live-frontier-spill-sim <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let global_live_class_limit = if args.len() >= 6 { args[5].parse().context("invalid global_live_class_limit")? } else { 300_000_000u64 };
    let per_field_class_limit = if args.len() >= 7 { args[6].parse().context("invalid per_field_class_limit")? } else { 1_000_000usize };
    let max_spills_per_parent = if args.len() >= 8 { args[7].parse().context("invalid max_spills_per_parent")? } else { 2u32 };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;

    let cfg = QLiveFrontierSpillConfig { global_live_class_limit, per_field_class_limit, max_spills_per_parent };
    eprintln!("running q live frontier-spill scheduler simulation only; no layer output will be written...");
    qlive_frontier_spill_sim(&rgraph, &layer, &qgraph, cfg)?;
    Ok(())
}

fn run_external_step_q_live_greedy_stats(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        eprintln!("usage: {} q-live-greedy-stats <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix>", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;

    eprintln!("running q live greedy order stats only; no layer output will be written...");
    qlive_greedy_order_stats(&rgraph, &layer, &qgraph)?;
    Ok(())
}

fn run_external_step_q_live(args: &[String]) -> Result<()> {
    if args.len() != 7 {
        eprintln!("usage: {} external-step-q-live <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir>", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    eprintln!("running external q live step output={} work_dir={}...", output_prefix.display(), work_dir.display());
    external_step_q_live(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir)?;
    Ok(())
}

fn run_external_step_q_live_frontier_spill(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 10 {
        eprintln!("usage: {} external-step-q-live-frontier-spill <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let global_live_class_limit = if args.len() >= 8 { args[7].parse().context("invalid global_live_class_limit")? } else { 300_000_000u64 };
    let per_field_class_limit = if args.len() >= 9 { args[8].parse().context("invalid per_field_class_limit")? } else { 1_000_000usize };
    let max_spills_per_parent = if args.len() >= 10 { args[9].parse().context("invalid max_spills_per_parent")? } else { 2u32 };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer index from {input_prefix} without resident chains...");
    // Frontier prepass and generation both use FieldChainLayer's bounded block
    // readers. A large predecessor input must not be loaded as one allocation.
    let layer = layer_store::FieldChainLayer::open_index_only(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QLiveFrontierSpillConfig { global_live_class_limit, per_field_class_limit, max_spills_per_parent };
    eprintln!("running external q live frontier-spill output={} work_dir={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}...", output_prefix.display(), work_dir.display(), global_live_class_limit, per_field_class_limit, max_spills_per_parent);
    external_step_q_live_frontier_spill(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}


fn run_external_step_q_live_frontier_spill_deferred(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 10 {
        eprintln!("usage: {} external-step-q-live-frontier-spill-deferred <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let global_live_class_limit = if args.len() >= 8 { args[7].parse().context("invalid global_live_class_limit")? } else { 600_000_000u64 };
    let per_field_class_limit = if args.len() >= 9 { args[8].parse().context("invalid per_field_class_limit")? } else { 3_000_000usize };
    let max_spills_per_parent = if args.len() >= 10 { args[9].parse().context("invalid max_spills_per_parent")? } else { 2u32 };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QLiveFrontierSpillConfig { global_live_class_limit, per_field_class_limit, max_spills_per_parent };
    eprintln!("running external q live frontier-spill-deferred output={} work_dir={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}...", output_prefix.display(), work_dir.display(), global_live_class_limit, per_field_class_limit, max_spills_per_parent);
    external_step_q_live_frontier_spill_deferred(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}


fn run_external_step_q_live_frontier_spill_strong(args: &[String]) -> Result<()> {
    if args.len() < 7 || args.len() > 10 {
        eprintln!("usage: {} external-step-q-live-frontier-spill-strong <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let qrev_path = PathBuf::from(&args[3]);
    let input_prefix = &args[4];
    let output_prefix = PathBuf::from(&args[5]);
    let work_dir = PathBuf::from(&args[6]);
    let global_live_class_limit = if args.len() >= 8 { args[7].parse().context("invalid global_live_class_limit")? } else { 600_000_000u64 };
    let per_field_class_limit = if args.len() >= 9 { args[8].parse().context("invalid per_field_class_limit")? } else { 3_000_000usize };
    let max_spills_per_parent = if args.len() >= 10 { args[9].parse().context("invalid max_spills_per_parent")? } else { 2u32 };

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;

    eprintln!("loading q input layer from {input_prefix}...");
    let layer = layer_store::FieldChainLayer::open(input_prefix)?;
    layer.print_stats("q_input");

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("loading qchain reverse-class graph from {}...", qrev_path.display());
    let qgraph = load_qrev_class_graph(&qrev_path)?;
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let cfg = QLiveFrontierSpillConfig { global_live_class_limit, per_field_class_limit, max_spills_per_parent };
    eprintln!("running external q live frontier-spill-strong output={} work_dir={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}...", output_prefix.display(), work_dir.display(), global_live_class_limit, per_field_class_limit, max_spills_per_parent);
    external_step_q_live_frontier_spill_strong(&rgraph, &layer, &q_index, &qgraph, &output_prefix, &work_dir, cfg)?;
    Ok(())
}

fn run_build_prefix_q(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        eprintln!("usage: {} build-prefix-q <rgraph.bin> <out_dir>", args[0]);
        std::process::exit(2);
    }
    let rgraph_path = &args[2];
    let out_dir = PathBuf::from(&args[3]);
    fs::create_dir_all(&out_dir)?;

    eprintln!("loading reverse graph from {rgraph_path}...");
    let rgraph = load_reverse_graph(rgraph_path)?;
    let rsummary = rgraph.summary();
    println!("reverse total_edges={}", rsummary.total_edges);
    println!("reverse edge_count_by_piece={:?}", rsummary.edge_count_by_piece);
    println!("reverse max_degree_by_piece={:?}", rsummary.max_degree_by_piece);

    eprintln!("enumerating old/q chain states...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    print_qstate_stats(&old_index, &q_index);

    eprintln!("building qchain reverse-class graph...");
    let qgraph_start = Instant::now();
    let qgraph = build_qrev_class_graph(&q_index);
    eprintln!("built qchain reverse-class graph in {:.3}s", qgraph_start.elapsed().as_secs_f64());
    print_qrev_class_graph_stats(&q_index, &qgraph);

    let mut good = make_q_terminal_layer(EMPTY_FIELD_ID, &q_index);
    println!("q product layer 10 good={}", good.len());
    print_q_product_layer_stats(&good, 10);

    for depth in [9usize, 8usize] {
        eprintln!("generating q product layer {depth} with masked q dedupe...");
        let layer_start = Instant::now();
        let mut prev_good = Vec::<u64>::new();
        let stats = gen_prev_q_layer_masked(&rgraph, &q_index, &qgraph, &good, &mut prev_good);
        eprintln!("generated q product layer {depth} in {:.3}s", layer_start.elapsed().as_secs_f64());
        println!("q product layer {depth} raw={}", stats.raw);
        println!("q product layer {depth} local_masked_unique={}", stats.local_masked_unique);
        println!("q product layer {depth} good={}", stats.deduped);
        println!("q product layer {depth} removed_dupes={}", stats.removed_dupes);
        print_q_product_layer_stats(&prev_good, depth);

        let label = format!("layer{depth}q_direct");
        let chain_bytes = prev_good.len() as u64 * 3;
        let fp16_bytes = prev_good.len() as u64 * 2;
        println!("{label} field_chain states={}", prev_good.len());
        println!("{label} u24 membership bytes={} ({:.3} GiB)", chain_bytes, chain_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("{label} future fp16 values bytes={} ({:.3} GiB)", fp16_bytes, fp16_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("{label} future u24+fp16 bytes={} ({:.3} GiB)", chain_bytes + fp16_bytes, (chain_bytes + fp16_bytes) as f64 / 1024.0 / 1024.0 / 1024.0);

        let prefix = out_dir.join(&label);
        eprintln!("saving q layer {depth} to prefix {}...", prefix.display());
        let written = save_q_pair_layer(&prefix, &prev_good)
            .with_context(|| format!("failed to save q layer {depth}: {}", prefix.display()))?;
        println!("saved {label} states={written}");
        good = prev_good;
    }

    eprintln!("stopping after saving q layer 8");
    Ok(())
}


fn q_layer_prefix(base: &Path, layer: usize) -> PathBuf {
    if layer <= 10 {
        base.join(format!("layer{layer}q"))
    } else {
        panic!("unsupported q layer {layer}")
    }
}

fn q_pruned_prefix(out_dir: &Path, layer: usize) -> PathBuf {
    out_dir.join(format!("layer{layer}q_pruned"))
}

#[inline]
fn field_range_shard(field: FieldId, num_shards: usize) -> usize {
    ((field as u64 * num_shards as u64) / NUM_FIELDS as u64) as usize
}


#[derive(Clone)]
struct QidBitSet {
    words: Vec<u64>,
    count: u32,
}

impl QidBitSet {
    fn new(qwords: usize) -> Self {
        Self { words: vec![0u64; qwords], count: 0 }
    }

    #[inline]
    fn set(&mut self, qid: u32) -> bool {
        let wi = (qid >> 6) as usize;
        let bit = 1u64 << (qid & 63);
        let old = self.words[wi];
        if (old & bit) == 0 {
            self.words[wi] = old | bit;
            self.count += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    fn test(&self, qid: u32) -> bool {
        let wi = (qid >> 6) as usize;
        let bit = 1u64 << (qid & 63);
        (self.words[wi] & bit) != 0
    }

    #[inline]
    fn or_intersection_count_new(&mut self, a: &QidBitSet, b: &QidBitSet) -> u32 {
        debug_assert_eq!(self.words.len(), a.words.len());
        debug_assert_eq!(self.words.len(), b.words.len());
        let mut added = 0u32;
        for i in 0..self.words.len() {
            let old = self.words[i];
            let add = a.words[i] & b.words[i];
            let neww = old | add;
            if neww != old {
                added += (neww ^ old).count_ones();
                self.words[i] = neww;
            }
        }
        self.count += added;
        added
    }

    #[inline]
    fn intersection_count(a: &QidBitSet, b: &QidBitSet) -> u32 {
        debug_assert_eq!(a.words.len(), b.words.len());
        let mut n = 0u32;
        for i in 0..a.words.len() {
            n += (a.words[i] & b.words[i]).count_ones();
        }
        n
    }

    fn iter_set_bits(&self, mut f: impl FnMut(u32) -> Result<()>) -> Result<()> {
        for (wi, &word0) in self.words.iter().enumerate() {
            let mut w = word0;
            while w != 0 {
                let tz = w.trailing_zeros();
                let qid = ((wi as u32) << 6) + tz;
                f(qid)?;
                w &= w - 1;
            }
        }
        Ok(())
    }
}

struct BlockPlacedQBits {
    bits: [QidBitSet; PIECE_COUNT],
    rec_counts: [u64; PIECE_COUNT],
    qid_counts: [u64; PIECE_COUNT],
}


fn add_intersection_to_acc(
    slot: &mut Option<QidBitSet>,
    qwords: usize,
    a: &QidBitSet,
    b: &QidBitSet,
) -> (u32, u32) {
    debug_assert_eq!(a.words.len(), b.words.len());
    const SPARSE_SET_BITS_THRESHOLD: u32 = 4096;

    if a.count <= SPARSE_SET_BITS_THRESHOLD {
        let mut hits = 0u32;
        let mut added = 0u32;
        for (wi, &word0) in a.words.iter().enumerate() {
            let mut w = word0;
            while w != 0 {
                let tz = w.trailing_zeros();
                let bit = 1u64 << tz;
                if (b.words[wi] & bit) != 0 {
                    hits += 1;
                    let bs = slot.get_or_insert_with(|| QidBitSet::new(qwords));
                    let old = bs.words[wi];
                    if (old & bit) == 0 {
                        bs.words[wi] = old | bit;
                        bs.count += 1;
                        added += 1;
                    }
                }
                w &= w - 1;
            }
        }
        return (hits, added);
    }

    if let Some(bs) = slot.as_mut() {
        let mut hits = 0u32;
        let mut added = 0u32;
        for i in 0..a.words.len() {
            let inter = a.words[i] & b.words[i];
            if inter == 0 { continue; }
            hits += inter.count_ones();
            let old = bs.words[i];
            let neww = old | inter;
            if neww != old {
                added += (neww ^ old).count_ones();
                bs.words[i] = neww;
            }
        }
        bs.count += added;
        (hits, added)
    } else {
        let mut words = vec![0u64; qwords];
        let mut hits = 0u32;
        for i in 0..a.words.len() {
            let inter = a.words[i] & b.words[i];
            if inter != 0 {
                hits += inter.count_ones();
                words[i] = inter;
            }
        }
        if hits != 0 {
            *slot = Some(QidBitSet { words, count: hits });
        }
        (hits, hits)
    }
}

fn build_block_placed_qbits(qids: &[u32], qfwd: &QForwardGraph, qwords: usize) -> BlockPlacedQBits {
    let mut bits: [QidBitSet; PIECE_COUNT] = std::array::from_fn(|_| QidBitSet::new(qwords));
    let mut rec_counts = [0u64; PIECE_COUNT];
    let mut qid_counts = [0u64; PIECE_COUNT];
    for &qid in qids {
        for placed in 0u8..PIECE_COUNT as u8 {
            let recs = qfwd.records_for(qid, placed);
            if recs.is_empty() { continue; }
            let p = placed as usize;
            qid_counts[p] += 1;
            rec_counts[p] += recs.len() as u64;
            for rec in recs {
                bits[p].set(rec.child_qid);
            }
        }
    }
    BlockPlacedQBits { bits, rec_counts, qid_counts }
}

struct BackwardShardBits {
    fields: Vec<FieldId>,
    bits: Vec<QidBitSet>,
    pos: HashMap<FieldId, usize>,
    states: u64,
}

fn load_backward_shard_bits(
    next_layer: &layer_store::FieldChainLayer,
    shard: usize,
    num_shards: usize,
    qwords: usize,
) -> Result<BackwardShardBits> {
    let mut fields = Vec::<FieldId>::new();
    let mut bits = Vec::<QidBitSet>::new();
    let mut pos = HashMap::<FieldId, usize>::new();
    let mut states = 0u64;

    for (bi, meta) in next_layer.metas.iter().enumerate() {
        if field_range_shard(meta.field_id, num_shards) != shard { continue; }
        let qids = next_layer.read_block_chains(bi);
        let mut bs = QidBitSet::new(qwords);
        for qid in qids {
            bs.set(qid);
        }
        states += bs.count as u64;
        let idx = fields.len();
        pos.insert(meta.field_id, idx);
        fields.push(meta.field_id);
        bits.push(bs);
    }

    Ok(BackwardShardBits { fields, bits, pos, states })
}


fn build_layer_range_block_indices(
    layer: &layer_store::FieldChainLayer,
    num_shards: usize,
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::<usize>::new(); num_shards.max(1)];
    for (bi, meta) in layer.metas.iter().enumerate() {
        let shard = field_range_shard(meta.field_id, num_shards);
        out[shard].push(bi);
    }
    out
}

fn load_backward_indexed_shard_bits(
    next_layer: &layer_store::FieldChainLayer,
    block_indices: &[usize],
    qwords: usize,
) -> Result<BackwardShardBits> {
    let mut fields = Vec::<FieldId>::with_capacity(block_indices.len());
    let mut bits = Vec::<QidBitSet>::with_capacity(block_indices.len());
    let mut pos = HashMap::<FieldId, usize>::with_capacity(block_indices.len() * 2);
    let mut states = 0u64;

    for &bi in block_indices {
        let meta = &next_layer.metas[bi];
        let qids = next_layer.read_block_chains(bi);
        let mut bs = QidBitSet::new(qwords);
        for qid in qids { bs.set(qid); }
        states += bs.count as u64;
        let idx = fields.len();
        pos.insert(meta.field_id, idx);
        fields.push(meta.field_id);
        bits.push(bs);
    }

    Ok(BackwardShardBits { fields, bits, pos, states })
}

fn collect_forward_target_range_shards_for_block(
    layer: usize,
    field: FieldId,
    graph: &FieldGraph,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_shards: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    if layer >= 9 { return; }
    for placed in 0u8..PIECE_COUNT as u8 {
        for &field2 in graph.edges(field, placed) {
            if layer + 1 == 5 && field2 == two_line_field { continue; }
            if layer + 1 == 10 && field2 == EMPTY_FIELD_ID { continue; }
            if !next_map.contains_key(&field2) { continue; }
            out.push(field_range_shard(field2, num_shards));
        }
    }
    out.sort_unstable();
    out.dedup();
}

fn build_forward_target_owned_source_shards(
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_shards: usize,
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::<usize>::new(); num_shards.max(1)];
    let mut tmp = Vec::<usize>::new();
    for (bi, meta) in cur.metas.iter().enumerate() {
        collect_forward_target_range_shards_for_block(layer, meta.field_id, graph, next_map, two_line_field, num_shards, &mut tmp);
        for &shard in &tmp { out[shard].push(bi); }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetOwnedShardFormat {
    /// Retained for comparisons with the original temporary representation.
    PairsU64,
    /// One sorted u24 membership stream; field IDs/counts stay in metadata.
    CompactU24,
}

struct TargetOwnedShardOutput {
    shard: usize,
    stats: ForwardPrefilterChunkStats,
    output_fields: usize,
    output_pairs: u64,
    path: PathBuf,
    format: TargetOwnedShardFormat,
    metas: Vec<layer_store::FieldChainBlockMeta>,
}

fn process_forward_target_owned_range_shard(
    layer: usize,
    range_shard: usize,
    source_blocks: &[usize],
    next_block_indices: &[usize],
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: &layer_store::FieldChainLayer,
    two_line_field: FieldId,
    num_range_shards: usize,
    qwords: usize,
    out_path: &Path,
) -> Result<TargetOwnedShardOutput> {
    process_forward_target_owned_range_shard_with_format(
        layer, range_shard, source_blocks, next_block_indices, cur, graph,
        qfwd, next_layer, two_line_field, num_range_shards, qwords, out_path,
        TargetOwnedShardFormat::CompactU24,
    )
}

fn process_forward_target_owned_range_shard_with_format(
    layer: usize,
    range_shard: usize,
    source_blocks: &[usize],
    next_block_indices: &[usize],
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: &layer_store::FieldChainLayer,
    two_line_field: FieldId,
    num_range_shards: usize,
    qwords: usize,
    out_path: &Path,
    format: TargetOwnedShardFormat,
) -> Result<TargetOwnedShardOutput> {
    let back = load_backward_indexed_shard_bits(next_layer, next_block_indices, qwords)?;
    let mut acc: Vec<Option<QidBitSet>> = (0..back.fields.len()).map(|_| None).collect();
    let mut st = ForwardPrefilterChunkStats {
        current_states: 0,
        placed_options: 0,
        graph_edges_seen: 0,
        terminal_2lpc_hits: 0,
        terminal_4lpc_hits: 0,
        raw_qforward_candidates: 0,
        membership_hits: 0,
        unique_output: 0,
        emitted_pairs: 0,
        touched_shards: source_blocks.len(),
        loaded_backward_states: back.states,
    };

    for &bi in source_blocks {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        if layer == 5 && field == two_line_field { continue; }
        let qids = cur.read_block_chains(bi);
        st.current_states += qids.len() as u64;
        let placed_bits = build_block_placed_qbits(&qids, qfwd, qwords);
        for placed in 0u8..PIECE_COUNT as u8 {
            let p = placed as usize;
            if placed_bits.qid_counts[p] == 0 { continue; }
            st.placed_options += placed_bits.qid_counts[p];
            let edges = graph.edges(field, placed);
            if edges.is_empty() { continue; }
            for &field2 in edges {
                if layer + 1 == 5 && field2 == two_line_field {
                    st.terminal_2lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer + 1 == 10 && field2 == EMPTY_FIELD_ID {
                    st.terminal_4lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer == 9 { continue; }
                if field_range_shard(field2, num_range_shards) != range_shard { continue; }
                st.graph_edges_seen += placed_bits.qid_counts[p];
                let Some(&idx) = back.pos.get(&field2) else { continue; };
                st.raw_qforward_candidates += placed_bits.rec_counts[p];
                let (hits, added) = add_intersection_to_acc(&mut acc[idx], qwords, &placed_bits.bits[p], &back.bits[idx]);
                st.membership_hits += hits as u64;
                st.unique_output += added as u64;
            }
        }
    }

    // A worker publishes its scratch shard only after the complete stream was
    // flushed. An error removes only this invocation's pending file and leaves
    // a previous completed shard at out_path untouched.
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let pending = out_path.with_extension(format!("pending.{}.{}", std::process::id(), stamp));
    let mut pending_owned = false;
    let result = (|| -> Result<TargetOwnedShardOutput> {
        let f = OpenOptions::new().write(true).create_new(true).open(&pending)
            .with_context(|| format!("create pending target-owned shard {}", pending.display()))?;
        pending_owned = true;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f);
        let mut output_fields = 0usize;
        let mut output_pairs = 0u64;
        let mut metas = Vec::with_capacity(back.fields.len());
        let mut encoded = Vec::<u8>::new();
        let mut previous_field = None;
        for (idx, maybe_bs) in acc.into_iter().enumerate() {
            let Some(bs) = maybe_bs else { continue; };
            if bs.count == 0 { continue; }
            let field = back.fields[idx];
            anyhow::ensure!(previous_field.is_none_or(|old| old < field),
                "target-owned output fields are not strictly sorted");
            let count = bs.count;
            let count_usize = usize::try_from(count).context("target-owned field count exceeds address space")?;
            anyhow::ensure!(count_usize <= qfwd.offsets.len(), "target-owned bit count exceeds quotient domain");
            let offset = output_pairs;
            encoded.clear();
            if format == TargetOwnedShardFormat::CompactU24 {
                encoded.reserve(count_usize.checked_mul(3).context("target-owned field byte size overflow")?);
            }
            let mut previous_qid = None;
            bs.iter_set_bits(|qid| {
                anyhow::ensure!((qid as usize) < qfwd.offsets.len() && qid < (1 << 24)
                    && previous_qid.is_none_or(|old| old < qid),
                    "target-owned output qids are not sorted, unique and in-domain");
                if format == TargetOwnedShardFormat::PairsU64 {
                    w.write_all(&q_pair_key(field, qid).to_le_bytes())?;
                } else {
                    encoded.extend_from_slice(&[(qid & 255) as u8, ((qid >> 8) & 255) as u8, ((qid >> 16) & 255) as u8]);
                }
                previous_qid = Some(qid);
                output_pairs = output_pairs.checked_add(1).context("target-owned output count overflow")?;
                Ok(())
            })?;
            anyhow::ensure!(output_pairs - offset == u64::from(count), "target-owned bit count differs from iteration");
            if format == TargetOwnedShardFormat::CompactU24 { w.write_all(&encoded)?; }
            metas.push(layer_store::FieldChainBlockMeta { field_id: field, count, chain_offset: offset });
            output_fields += 1;
            previous_field = Some(field);
        }
        w.flush()?;
        drop(w);
        fs::rename(&pending, out_path).with_context(|| format!("publish target-owned shard {}", out_path.display()))?;
        st.emitted_pairs = output_pairs;
        Ok(TargetOwnedShardOutput {
            shard: range_shard, stats: st, output_fields, output_pairs,
            path: out_path.to_path_buf(), format, metas,
        })
    })();
    if result.is_err() && pending_owned { let _ = fs::remove_file(&pending); }
    result
}

fn append_target_owned_output_to_writer(
    output: &TargetOwnedShardOutput,
    writer: &mut layer_store::FieldChainWriter,
) -> Result<u64> {
    let result = (|| -> Result<u64> {
    if output.format == TargetOwnedShardFormat::PairsU64 {
        return append_target_owned_shard_to_writer(&output.path, writer);
    }
    let file = File::open(&output.path).with_context(|| format!("open compact target-owned output {}", output.path.display()))?;
    let mut count = 0u64;
    let mut previous = None;
    for meta in &output.metas {
        anyhow::ensure!(meta.count > 0 && meta.chain_offset == count
            && previous.is_none_or(|old| old < meta.field_id),
            "compact target-owned shard metadata is not a sorted contiguous tiling");
        count = count.checked_add(u64::from(meta.count)).context("compact target-owned count overflow")?;
        previous = Some(meta.field_id);
    }
    anyhow::ensure!(count == output.output_pairs && output.metas.len() == output.output_fields,
        "compact target-owned shard metadata/count mismatch");
    let expected = count.checked_mul(3).context("compact target-owned byte size overflow")?;
    anyhow::ensure!(file.metadata()?.len() == expected, "compact target-owned shard size mismatch");
    let mut scratch = Vec::new();
    let mut appended = 0u64;
    for meta in &output.metas {
        let emitted = writer.append_sorted_u24_file_with_buf(meta.field_id, &file,
            meta.chain_offset.checked_mul(3).context("compact target-owned offset overflow")?,
            u64::from(meta.count), &mut scratch)?;
        anyhow::ensure!(emitted == u64::from(meta.count), "compact target-owned shard contains duplicate qids");
        appended = appended.checked_add(emitted).context("compact target-owned append overflow")?;
    }
    Ok(appended)
    })();
    if result.is_err() { writer.invalidate(); }
    result
}

#[cfg(test)]
#[path = "forward_index_io_tests.rs"]
mod forward_index_io_tests;

/// Only this newly created directory is removed on scope exit. Old scratch
/// directories and any unrelated files in step_work remain untouched.
struct TargetOwnedShardScratch {
    path: PathBuf,
}

impl TargetOwnedShardScratch {
    fn create(parent: &Path) -> Result<Self> {
        fs::create_dir_all(parent)?;
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        for attempt in 0..100u32 {
            let path = parent.join(format!(".target-owned-{}-{stamp}-{attempt}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error).context("create invocation-owned target shard directory"),
            }
        }
        anyhow::bail!("cannot create a unique target-owned shard directory")
    }
}

impl Drop for TargetOwnedShardScratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.path); }
}

fn append_target_owned_shard_to_writer(
    path: &Path,
    writer: &mut layer_store::FieldChainWriter,
) -> Result<u64> {
    let file = File::open(path).with_context(|| format!("open target-owned output {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut carry = Vec::<u8>::new();
    let mut n_pairs = 0u64;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 { break; }
        let mut chunk: Vec<u8>;
        let bytes: &[u8] = if carry.is_empty() {
            &buf[..n]
        } else {
            chunk = Vec::with_capacity(carry.len() + n);
            chunk.extend_from_slice(&carry);
            chunk.extend_from_slice(&buf[..n]);
            &chunk
        };
        let usable = bytes.len() / 8 * 8;
        let mut off = 0usize;
        while off < usable {
            let pair = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            off += 8;
            writer.push_pair_key(pair)?;
            n_pairs += 1;
        }
        carry.clear();
        if usable < bytes.len() { carry.extend_from_slice(&bytes[usable..]); }
    }
    if !carry.is_empty() { anyhow::bail!("target-owned shard {} has trailing partial u64", path.display()); }
    Ok(n_pairs)
}

fn process_candidate_shard_bitset(
    path: &Path,
    back: &BackwardShardBits,
    qwords: usize,
    writer: &mut layer_store::FieldChainWriter,
) -> Result<(u64, u64, u64, usize)> {
    let mut raw_pairs = 0u64;
    let mut membership_hits = 0u64;
    let mut unique_pairs = 0u64;
    let mut output_fields = 0usize;

    let mut acc: Vec<Option<QidBitSet>> = (0..back.fields.len()).map(|_| None).collect();

    let file = File::open(path).with_context(|| format!("open candidate shard {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut carry = Vec::<u8>::new();

    loop {
        let n = reader.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 { break; }
        let mut chunk: Vec<u8>;
        let bytes: &[u8] = if carry.is_empty() {
            &buf[..n]
        } else {
            chunk = Vec::with_capacity(carry.len() + n);
            chunk.extend_from_slice(&carry);
            chunk.extend_from_slice(&buf[..n]);
            &chunk
        };
        let usable = bytes.len() / 8 * 8;
        let mut off = 0usize;
        while off < usable {
            let pair = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            off += 8;
            raw_pairs += 1;
            let field = (pair >> 32) as FieldId;
            let qid = pair as u32;
            let Some(&idx) = back.pos.get(&field) else { continue; };
            if !back.bits[idx].test(qid) { continue; }
            membership_hits += 1;
            let bs = acc[idx].get_or_insert_with(|| QidBitSet::new(qwords));
            if bs.set(qid) {
                unique_pairs += 1;
            }
        }
        carry.clear();
        if usable < bytes.len() {
            carry.extend_from_slice(&bytes[usable..]);
        }
    }
    if !carry.is_empty() {
        anyhow::bail!("candidate shard {} has trailing partial u64 bytes", path.display());
    }

    for (idx, maybe_bs) in acc.into_iter().enumerate() {
        let Some(bs) = maybe_bs else { continue; };
        if bs.count == 0 { continue; }
        output_fields += 1;
        let field = back.fields[idx];
        bs.iter_set_bits(|qid| writer.push_pair_key(q_pair_key(field, qid)))?;
    }

    Ok((raw_pairs, membership_hits, unique_pairs, output_fields))
}


fn collect_forward_target_shards_for_field(
    layer: usize,
    field: FieldId,
    graph: &FieldGraph,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_shards: usize,
    out: &mut HashSet<usize>,
) {
    if layer >= 9 { return; }
    for placed in 0u8..PIECE_COUNT as u8 {
        for &field2 in graph.edges(field, placed) {
            if layer + 1 == 5 && field2 == two_line_field { continue; }
            if layer + 1 == 10 && field2 == EMPTY_FIELD_ID { continue; }
            if !next_map.contains_key(&field2) { continue; }
            out.insert(field_range_shard(field2, num_shards));
        }
    }
}

#[derive(Default)]
struct ForwardPrefilterChunkStats {
    current_states: u64,
    placed_options: u64,
    graph_edges_seen: u64,
    terminal_2lpc_hits: u64,
    terminal_4lpc_hits: u64,
    raw_qforward_candidates: u64,
    membership_hits: u64,
    unique_output: u64,
    emitted_pairs: u64,
    touched_shards: usize,
    loaded_backward_states: u64,
}



struct TargetLocalShardOutput {
    stats: ForwardPrefilterChunkStats,
    pairs_by_range: Vec<(usize, Vec<u64>)>,
}

fn add_forward_stats(dst: &mut ForwardPrefilterChunkStats, src: &ForwardPrefilterChunkStats) {
    dst.current_states += src.current_states;
    dst.placed_options += src.placed_options;
    dst.graph_edges_seen += src.graph_edges_seen;
    dst.terminal_2lpc_hits += src.terminal_2lpc_hits;
    dst.terminal_4lpc_hits += src.terminal_4lpc_hits;
    dst.raw_qforward_candidates += src.raw_qforward_candidates;
    dst.membership_hits += src.membership_hits;
    dst.unique_output += src.unique_output;
    dst.emitted_pairs += src.emitted_pairs;
    dst.touched_shards = dst.touched_shards.max(src.touched_shards);
    dst.loaded_backward_states = dst.loaded_backward_states.max(src.loaded_backward_states);
}

struct ForwardTouch {
    block_idx: usize,
    targets: Vec<FieldId>,
    sig: u64,
    weight: u64,
}

fn collect_forward_target_fields_for_block(
    layer: usize,
    field: FieldId,
    graph: &FieldGraph,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    out: &mut Vec<FieldId>,
) {
    out.clear();
    if layer >= 9 { return; }
    for placed in 0u8..PIECE_COUNT as u8 {
        for &field2 in graph.edges(field, placed) {
            if layer + 1 == 5 && field2 == two_line_field { continue; }
            if layer + 1 == 10 && field2 == EMPTY_FIELD_ID { continue; }
            if !next_map.contains_key(&field2) { continue; }
            out.push(field2);
        }
    }
    out.sort_unstable();
    out.dedup();
}

#[inline]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

fn forward_touch_signature(targets: &[FieldId], target_bucket_size: u32) -> u64 {
    if targets.is_empty() { return u64::MAX; }
    // Min-hash over coarse target buckets. This is the same locality idea as qpreq
    // target-shard: blocks touching similar target buckets sort close together.
    let mut best = u64::MAX;
    for &f in targets {
        let bucket = (f / target_bucket_size.max(1)) as u64;
        best = best.min(mix_u64(bucket));
    }
    best
}

fn build_forward_target_local_shards(
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_shards: usize,
    target_bucket_size: u32,
) -> Vec<Vec<usize>> {
    let mut touches = Vec::<ForwardTouch>::with_capacity(cur.metas.len());
    let mut tmp = Vec::<FieldId>::new();
    for (bi, meta) in cur.metas.iter().enumerate() {
        collect_forward_target_fields_for_block(layer, meta.field_id, graph, next_map, two_line_field, &mut tmp);
        if tmp.is_empty() { continue; }
        let sig = forward_touch_signature(&tmp, target_bucket_size);
        let weight = (meta.count as u64).saturating_mul(tmp.len().max(1) as u64).max(1);
        touches.push(ForwardTouch { block_idx: bi, targets: tmp.clone(), sig, weight });
    }
    touches.sort_unstable_by(|a, b| a.sig.cmp(&b.sig).then(a.targets.first().cmp(&b.targets.first())).then(a.block_idx.cmp(&b.block_idx)));

    let work_shards = num_shards.max(1).min(touches.len().max(1));
    let total_weight: u128 = touches.iter().map(|t| t.weight as u128).sum();
    let target_per_shard = ((total_weight + work_shards as u128 - 1) / work_shards as u128).max(1);
    let mut shards = vec![Vec::<usize>::new(); work_shards];
    let mut sid = 0usize;
    let mut acc: u128 = 0;
    for t in touches {
        if sid + 1 < work_shards && !shards[sid].is_empty() && acc >= target_per_shard {
            sid += 1;
            acc = 0;
        }
        acc += t.weight as u128;
        shards[sid].push(t.block_idx);
    }
    shards
}

struct BackwardFieldBits {
    fields: Vec<FieldId>,
    bits: Vec<QidBitSet>,
    pos: HashMap<FieldId, usize>,
    states: u64,
}

fn load_backward_field_bits(
    next_layer: &layer_store::FieldChainLayer,
    next_map: &HashMap<FieldId, usize>,
    fields: &[FieldId],
    qwords: usize,
) -> Result<BackwardFieldBits> {
    let mut uniq = fields.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let mut out_fields = Vec::<FieldId>::with_capacity(uniq.len());
    let mut bits = Vec::<QidBitSet>::with_capacity(uniq.len());
    let mut pos = HashMap::<FieldId, usize>::with_capacity(uniq.len() * 2);
    let mut states = 0u64;
    for field in uniq {
        let Some(&bi) = next_map.get(&field) else { continue; };
        let qids = next_layer.read_block_chains(bi);
        let mut bs = QidBitSet::new(qwords);
        for qid in qids { bs.set(qid); }
        states += bs.count as u64;
        let idx = out_fields.len();
        pos.insert(field, idx);
        out_fields.push(field);
        bits.push(bs);
    }
    Ok(BackwardFieldBits { fields: out_fields, bits, pos, states })
}

fn process_forward_target_local_shard(
    layer: usize,
    block_indices: &[usize],
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: &layer_store::FieldChainLayer,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_range_shards: usize,
    target_bucket_size: u32,
    qwords: usize,
) -> Result<TargetLocalShardOutput> {
    let mut target_fields = Vec::<FieldId>::new();
    let mut tmp = Vec::<FieldId>::new();
    for &bi in block_indices {
        let field = cur.metas[bi].field_id;
        collect_forward_target_fields_for_block(layer, field, graph, next_map, two_line_field, &mut tmp);
        target_fields.extend_from_slice(&tmp);
    }
    target_fields.sort_unstable();
    target_fields.dedup();

    let back = load_backward_field_bits(next_layer, next_map, &target_fields, qwords)?;
    let mut acc: Vec<Option<QidBitSet>> = (0..back.fields.len()).map(|_| None).collect();
    let mut st = ForwardPrefilterChunkStats {
        current_states: 0,
        placed_options: 0,
        graph_edges_seen: 0,
        terminal_2lpc_hits: 0,
        terminal_4lpc_hits: 0,
        raw_qforward_candidates: 0,
        membership_hits: 0,
        unique_output: 0,
        emitted_pairs: 0,
        touched_shards: block_indices.len(),
        loaded_backward_states: back.states,
    };

    for &bi in block_indices {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        // If a terminal state is ever materialized in the frontier, it is a stop state.
        if layer == 5 && field == two_line_field { continue; }
        let qids = cur.read_block_chains(bi);
        st.current_states += qids.len() as u64;

        // Q transition is independent of the target field.  Build placed -> qid2 bitsets once
        // per source field block, then intersect those bitsets with each target field's
        // backward membership bitset.  This removes the field-edge multiplier from qforward.
        let placed_bits = build_block_placed_qbits(&qids, qfwd, qwords);
        for placed in 0u8..PIECE_COUNT as u8 {
            let p = placed as usize;
            if placed_bits.qid_counts[p] == 0 { continue; }
            st.placed_options += placed_bits.qid_counts[p];
            let edges = graph.edges(field, placed);
            if edges.is_empty() { continue; }
            for &field2 in edges {
                st.graph_edges_seen += placed_bits.qid_counts[p];
                if layer + 1 == 5 && field2 == two_line_field {
                    st.terminal_2lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer + 1 == 10 && field2 == EMPTY_FIELD_ID {
                    st.terminal_4lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer == 9 { continue; }
                let Some(&idx) = back.pos.get(&field2) else { continue; };
                let bbits = &back.bits[idx];
                st.raw_qforward_candidates += placed_bits.rec_counts[p];
                let (hits, added) = add_intersection_to_acc(&mut acc[idx], qwords, &placed_bits.bits[p], bbits);
                st.membership_hits += hits as u64;
                st.unique_output += added as u64;
            }
        }
    }

    let mut pairs_map = HashMap::<usize, Vec<u64>>::new();
    for (idx, maybe_bs) in acc.into_iter().enumerate() {
        let Some(bs) = maybe_bs else { continue; };
        if bs.count == 0 { continue; }
        let field = back.fields[idx];
        let range_shard = field_range_shard(field, num_range_shards);
        let pairs = pairs_map.entry(range_shard).or_insert_with(Vec::new);
        bs.iter_set_bits(|qid| {
            pairs.push(q_pair_key(field, qid));
            st.emitted_pairs += 1;
            Ok(())
        })?;
    }
    st.touched_shards = pairs_map.len();
    // bucket size is part of the signature/partitioning contract; touch it to keep warning-free in old compilers.
    let _ = target_bucket_size;
    Ok(TargetLocalShardOutput { stats: st, pairs_by_range: pairs_map.into_iter().collect() })
}


fn process_forward_direct_full_bitset_layer(
    layer: usize,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: &layer_store::FieldChainLayer,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    qwords: usize,
    out_prefix: &Path,
) -> Result<(ForwardPrefilterChunkStats, u64, usize)> {
    let mut all_fields = Vec::<FieldId>::with_capacity(next_layer.metas.len());
    for meta in &next_layer.metas { all_fields.push(meta.field_id); }
    let load_start = Instant::now();
    let back = load_backward_field_bits(next_layer, next_map, &all_fields, qwords)?;
    eprintln!(
        "forward-prune-q direct-full-bitset block-qbits layer={} next_fields={} next_states={} load_elapsed={:.3}s",
        layer, back.fields.len(), back.states, load_start.elapsed().as_secs_f64()
    );

    let mut acc: Vec<Option<QidBitSet>> = (0..back.fields.len()).map(|_| None).collect();
    let mut st = ForwardPrefilterChunkStats {
        current_states: 0,
        placed_options: 0,
        graph_edges_seen: 0,
        terminal_2lpc_hits: 0,
        terminal_4lpc_hits: 0,
        raw_qforward_candidates: 0,
        membership_hits: 0,
        unique_output: 0,
        emitted_pairs: 0,
        touched_shards: 0,
        loaded_backward_states: back.states,
    };

    let start = Instant::now();
    for (bi, meta) in cur.metas.iter().enumerate() {
        let field = meta.field_id;
        // Materialized terminal states are stop states.  The current one-loop/forward prune
        // semantics treats terminal success as an edge event, not a layer state to expand.
        if layer == 5 && field == two_line_field { continue; }
        let qids = cur.read_block_chains(bi);
        st.current_states += qids.len() as u64;

        // Q transition is independent of the target field.  Build placed -> qid2 bitsets once
        // per source field block, then use word-wise AND with each next-layer field bitset.
        let placed_bits = build_block_placed_qbits(&qids, qfwd, qwords);
        for placed in 0u8..PIECE_COUNT as u8 {
            let p = placed as usize;
            if placed_bits.qid_counts[p] == 0 { continue; }
            st.placed_options += placed_bits.qid_counts[p];
            let edges = graph.edges(field, placed);
            if edges.is_empty() { continue; }
            for &field2 in edges {
                st.graph_edges_seen += placed_bits.qid_counts[p];
                if layer + 1 == 5 && field2 == two_line_field {
                    st.terminal_2lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer + 1 == 10 && field2 == EMPTY_FIELD_ID {
                    st.terminal_4lpc_hits += placed_bits.qid_counts[p];
                    continue;
                }
                if layer == 9 { continue; }
                let Some(&idx) = back.pos.get(&field2) else { continue; };
                let bbits = &back.bits[idx];
                st.raw_qforward_candidates += placed_bits.rec_counts[p];
                let (hits, added) = add_intersection_to_acc(&mut acc[idx], qwords, &placed_bits.bits[p], bbits);
                st.membership_hits += hits as u64;
                st.unique_output += added as u64;
            }
        }
        if (bi + 1) % 1024 == 0 || bi + 1 == cur.metas.len() {
            eprintln!(
                "  forward-prune-q gen-direct-full-bitset layer={} blocks_done={}/{} current_states={} raw_qfwd={} membership_hits={} unique_output={} term2={} term4={} elapsed={:.1}s",
                layer, bi + 1, cur.metas.len(), st.current_states, st.raw_qforward_candidates,
                st.membership_hits, st.unique_output, st.terminal_2lpc_hits, st.terminal_4lpc_hits,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let mut writer = layer_store::FieldChainWriter::create(out_prefix)?;
    let mut output_fields = 0usize;
    for (idx, maybe_bs) in acc.into_iter().enumerate() {
        let Some(bs) = maybe_bs else { continue; };
        if bs.count == 0 { continue; }
        output_fields += 1;
        let field = back.fields[idx];
        bs.iter_set_bits(|qid| {
            writer.push_pair_key(q_pair_key(field, qid))?;
            st.emitted_pairs += 1;
            Ok(())
        })?;
    }
    let written = writer.finish()?;
    if written != st.unique_output {
        anyhow::bail!("direct full-bitset writer mismatch: unique={} written={}", st.unique_output, written);
    }
    Ok((st, written, output_fields))
}

fn process_forward_prefilter_chunk(
    layer: usize,
    block_indices: &[usize],
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: &layer_store::FieldChainLayer,
    next_map: &HashMap<FieldId, usize>,
    two_line_field: FieldId,
    num_shards: usize,
    qwords: usize,
    shard_writers: &mut LazyShardSpoolWriters,
) -> Result<ForwardPrefilterChunkStats> {
    let mut touched = HashSet::<usize>::new();
    for &bi in block_indices {
        let field = cur.metas[bi].field_id;
        collect_forward_target_shards_for_field(layer, field, graph, next_map, two_line_field, num_shards, &mut touched);
    }

    let mut backs = HashMap::<usize, BackwardShardBits>::with_capacity(touched.len() * 2);
    let mut loaded_backward_states = 0u64;
    for &shard in &touched {
        let back = load_backward_shard_bits(next_layer, shard, num_shards, qwords)?;
        loaded_backward_states += back.states;
        backs.insert(shard, back);
    }

    let mut acc = HashMap::<usize, Vec<Option<QidBitSet>>>::with_capacity(backs.len() * 2);
    let mut st = ForwardPrefilterChunkStats {
        current_states: 0,
        placed_options: 0,
        graph_edges_seen: 0,
        terminal_2lpc_hits: 0,
        terminal_4lpc_hits: 0,
        raw_qforward_candidates: 0,
        membership_hits: 0,
        unique_output: 0,
        emitted_pairs: 0,
        touched_shards: touched.len(),
        loaded_backward_states,
    };

    for &bi in block_indices {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        let qids = cur.read_block_chains(bi);
        st.current_states += qids.len() as u64;
        for qid in qids {
            for placed in 0u8..PIECE_COUNT as u8 {
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() { continue; }
                st.placed_options += 1;
                let edges = graph.edges(field, placed);
                if edges.is_empty() { continue; }
                for &field2 in edges {
                    st.graph_edges_seen += 1;
                    if layer + 1 == 5 && field2 == two_line_field {
                        st.terminal_2lpc_hits += 1;
                        continue;
                    }
                    if layer + 1 == 10 && field2 == EMPTY_FIELD_ID {
                        st.terminal_4lpc_hits += 1;
                        continue;
                    }
                    if layer == 9 { continue; }
                    if !next_map.contains_key(&field2) { continue; }
                    let shard = field_range_shard(field2, num_shards);
                    let Some(back) = backs.get(&shard) else { continue; };
                    let Some(&idx) = back.pos.get(&field2) else { continue; };
                    for rec in recs {
                        st.raw_qforward_candidates += 1;
                        if !back.bits[idx].test(rec.child_qid) { continue; }
                        st.membership_hits += 1;
                        let avec = acc.entry(shard).or_insert_with(|| (0..back.fields.len()).map(|_| None).collect());
                        let bs = avec[idx].get_or_insert_with(|| QidBitSet::new(qwords));
                        if bs.set(rec.child_qid) {
                            st.unique_output += 1;
                        }
                    }
                }
            }
        }
    }

    for (shard, avec) in acc.into_iter() {
        let back = backs.get(&shard).expect("acc shard must be loaded");
        for (idx, maybe_bs) in avec.into_iter().enumerate() {
            let Some(bs) = maybe_bs else { continue; };
            if bs.count == 0 { continue; }
            let field = back.fields[idx];
            bs.iter_set_bits(|qid| {
                shard_writers.write_pair(shard, q_pair_key(field, qid))?;
                st.emitted_pairs += 1;
                Ok(())
            })?;
        }
    }

    Ok(st)
}


struct LazyShardSpoolWriters {
    paths: Vec<PathBuf>,
    open: HashMap<usize, (BufWriter<File>, u64)>,
    tick: u64,
    max_open: usize,
}

impl LazyShardSpoolWriters {
    fn create(dir: &Path, num_shards: usize, max_open: usize) -> Result<Self> {
        let mut paths = Vec::<PathBuf>::with_capacity(num_shards);
        for sidx in 0..num_shards {
            let p = dir.join(format!("cand_{sidx:05}.u64.bin"));
            // Truncate stale files now; later opens append.
            File::create(&p).with_context(|| format!("create {}", p.display()))?;
            paths.push(p);
        }
        Ok(Self { paths, open: HashMap::new(), tick: 0, max_open: max_open.max(1) })
    }

    fn paths(&self) -> &[PathBuf] { &self.paths }

    fn write_pair(&mut self, shard: usize, pair: u64) -> Result<()> {
        self.tick = self.tick.wrapping_add(1);
        if !self.open.contains_key(&shard) {
            if self.open.len() >= self.max_open {
                let (&victim, _) = self.open.iter()
                    .min_by_key(|(_, (_, last))| *last)
                    .expect("open map nonempty");
                if let Some((mut w, _)) = self.open.remove(&victim) {
                    w.flush()?;
                }
            }
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.paths[shard])
                .with_context(|| format!("open append {}", self.paths[shard].display()))?;
            self.open.insert(shard, (BufWriter::with_capacity(1 << 20, f), self.tick));
        }
        let (w, last) = self.open.get_mut(&shard).unwrap();
        *last = self.tick;
        w.write_all(&pair.to_le_bytes())?;
        Ok(())
    }

    fn flush_all(mut self) -> Result<()> {
        for (_, (mut w, _)) in self.open.drain() {
            w.flush()?;
        }
        Ok(())
    }
}

fn copy_layer_as_pruned(input_prefix: &Path, output_prefix: &Path) -> Result<u64> {
    let layer = layer_store::FieldChainLayer::open_index_only(input_prefix)?;
    let mut writer = layer_store::FieldChainWriter::create(output_prefix)?;
    for i in 0..layer.metas.len() {
        let meta = &layer.metas[i];
        let qids = layer.read_block_chains(i);
        for qid in qids {
            writer.push_pair_key(q_pair_key(meta.field_id, qid))?;
        }
    }
    writer.finish()
}

fn read_u64_pairs_file(path: &Path) -> Result<Vec<u64>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 8 != 0 {
        anyhow::bail!("candidate shard size is not multiple of 8: {}", path.display());
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    let mut pos = 0usize;
    while pos < bytes.len() {
        out.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
        pos += 8;
    }
    Ok(out)
}

fn intersect_sorted_candidate_qids_with_backward(
    cand_pairs: &[u64],
    next_layer: &layer_store::FieldChainLayer,
    next_map: &HashMap<FieldId, usize>,
    writer: &mut layer_store::FieldChainWriter,
) -> Result<(u64, u64, u64)> {
    let mut input_pairs = 0u64;
    let mut unique_pairs = 0u64;
    let mut accepted = 0u64;
    let mut i = 0usize;
    while i < cand_pairs.len() {
        let field = (cand_pairs[i] >> 32) as FieldId;
        let start = i;
        while i < cand_pairs.len() && ((cand_pairs[i] >> 32) as FieldId) == field {
            i += 1;
        }
        let end = i;
        input_pairs += (end - start) as u64;

        let Some(&block_idx) = next_map.get(&field) else { continue; };
        let bq = next_layer.read_block_chains(block_idx);
        let mut bi = 0usize;
        let mut last_qid = None::<u32>;
        for &pair in &cand_pairs[start..end] {
            let qid = pair as u32;
            if last_qid == Some(qid) { continue; }
            last_qid = Some(qid);
            unique_pairs += 1;
            while bi < bq.len() && bq[bi] < qid { bi += 1; }
            if bi < bq.len() && bq[bi] == qid {
                writer.push_pair_key(q_pair_key(field, qid))?;
                accepted += 1;
            }
        }
    }
    Ok((input_pairs, unique_pairs, accepted))
}

fn run_forward_prune_q(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 11 {
        eprintln!("usage: {} forward-prune-q-frontier-target-shard <graph.bin> <backward_layer_dir> <output_dir> <work_dir> [num_shards=4096] [vec_to_bitset_threshold=8192] [hot_cap_gib=64] [target_bucket_size=65536] [refine_passes=4]", args[0]);
        eprintln!("       {} forward-prune-q <graph.bin> <backward_layer_dir> <output_dir> <work_dir> [num_shards]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let backward_dir = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
    let work_dir = PathBuf::from(&args[5]);
    let num_shards: usize = if args.len() >= 7 { args[6].parse().context("invalid num_shards")? } else { 4096 };
    let vec_to_bitset_threshold: usize = if args.len() >= 8 { args[7].parse().context("invalid vec_to_bitset_threshold")? } else { 8192 };
    let hot_cap_gib: usize = if args.len() >= 9 { args[8].parse().context("invalid hot_cap_gib")? } else { 64 };
    let target_bucket_size: u32 = if args.len() >= 10 { args[9].parse().context("invalid target_bucket_size")? } else { 65536 };
    let refine_passes: usize = if args.len() >= 11 { args[10].parse().context("invalid refine_passes")? } else { 4 };
    if num_shards == 0 { anyhow::bail!("num_shards must be > 0"); }

    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(&work_dir)?;

    eprintln!("forward-prune-q-frontier-target-shard loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    eprintln!("forward-prune-q building q index and qforward graph...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    eprintln!("forward-prune-q qstates={} qfwd_records={}", q_index.states.len(), qfwd.records.len());
    let qwords = (q_index.states.len() + 63) / 64;
    eprintln!("forward-prune-q-frontier-target-shard config shards={} vec_to_bitset_threshold={} hot_cap_gib={} target_bucket_size={} refine_passes={} qwords={} bytes_per_field={}",
        num_shards, vec_to_bitset_threshold, hot_cap_gib, target_bucket_size, refine_passes, qwords, qwords * 8);

    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH)
        .unwrap_or(11335);
    eprintln!("forward-prune-q two_line_field={} four_line_terminal_field={}", two_line_field, EMPTY_FIELD_ID);

    let b0 = q_layer_prefix(&backward_dir, 0);
    let f0 = q_pruned_prefix(&output_dir, 0);
    eprintln!("forward-prune-q initializing F0 from {} -> {}", b0.display(), f0.display());
    let f0_states = copy_layer_as_pruned(&b0, &f0)?;
    eprintln!("forward-prune-q layer=0 initialized states={f0_states}");

    for layer in 0usize..=9usize {
        let cur_prefix = q_pruned_prefix(&output_dir, layer);
        let step_work = work_dir.join(format!("step{layer}_to{}", layer + 1));
        if step_work.exists() { fs::remove_dir_all(&step_work)?; }
        fs::create_dir_all(&step_work)?;

        eprintln!("forward-prune-q step layer={layer} loading current {}", cur_prefix.display());
        let cur = layer_store::FieldChainLayer::open_index_only(&cur_prefix)?;

        let (next_layer, next_map, out_prefix) = if layer < 9 {
            let nb = q_layer_prefix(&backward_dir, layer + 1);
            eprintln!("forward-prune-q step layer={layer} loading next backward {}", nb.display());
            let nl = layer_store::FieldChainLayer::open_index_only(&nb)?;
            let mut map = HashMap::<FieldId, usize>::with_capacity(nl.metas.len() * 2);
            for (i, m) in nl.metas.iter().enumerate() {
                map.insert(m.field_id, i);
            }
            (Some(nl), Some(map), Some(q_pruned_prefix(&output_dir, layer + 1)))
        } else {
            (None, None, None)
        };

        let max_open_shard_files = std::env::var("FWD_PRUNE_MAX_OPEN_SHARDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256);
        let chunk_blocks = std::env::var("FWD_PRUNE_CHUNK_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1);
        let direct_max_next_layer = std::env::var("FWD_PRUNE_DIRECT_FULL_BITSET_MAX_NEXT_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);

        if layer < 9 && ((layer + 1) <= direct_max_next_layer || (layer + 1) >= 8) {
            let start = Instant::now();
            let next_layer_ref = next_layer.as_ref().unwrap();
            let next_map_ref = next_map.as_ref().unwrap();
            let out_prefix_ref = out_prefix.as_ref().unwrap();
            eprintln!(
                "forward-prune-q layer={} direct-full-bitset path next_layer={} max_next_layer={} current_fields={} next_fields={}",
                layer, layer + 1, direct_max_next_layer, cur.metas.len(), next_layer_ref.metas.len()
            );
            let (st, accepted_states, output_fields) = process_forward_direct_full_bitset_layer(
                layer,
                &cur,
                &graph,
                &qfwd,
                next_layer_ref,
                next_map_ref,
                two_line_field,
                qwords,
                out_prefix_ref,
            )?;
            eprintln!(
                "forward-prune-q generation layer={} direct-full-bitset current_states={} placed_options={} graph_edges={} raw_qforward_candidates={} generation_membership_hits={} generation_unique_output={} terminal_2lpc_hits={} terminal_4lpc_hits={} loaded_backward_states={} elapsed={:.3}s",
                layer, st.current_states, st.placed_options, st.graph_edges_seen, st.raw_qforward_candidates,
                st.membership_hits, st.unique_output, st.terminal_2lpc_hits, st.terminal_4lpc_hits,
                st.loaded_backward_states, start.elapsed().as_secs_f64()
            );
            eprintln!(
                "forward-prune-q layer={} done output_layer={} output_states={} output_fields={} direct_full_bitset=1 step_elapsed={:.3}s",
                layer, layer + 1, accepted_states, output_fields, start.elapsed().as_secs_f64()
            );
            continue;
        }

        let mut shard_writers = LazyShardSpoolWriters::create(&step_work, num_shards, max_open_shard_files)?;
        let shard_paths: Vec<PathBuf> = shard_writers.paths().to_vec();
        eprintln!("forward-prune-q layer={} prefiltered candidate spools shards={} max_open_files={} chunk_blocks={}", layer, num_shards, max_open_shard_files, chunk_blocks);

        let start = Instant::now();
        let mut current_states = 0u64;
        let mut placed_options = 0u64;
        let mut graph_edges_seen = 0u64;
        let mut terminal_2lpc_hits = 0u64;
        let mut terminal_4lpc_hits = 0u64;
        let mut raw_qforward_candidates = 0u64;
        let mut generation_membership_hits = 0u64;
        let mut generation_unique_output = 0u64;
        let mut candidate_pairs = 0u64;
        let mut nonterminal_after_layer9 = 0u64;
        let mut max_chunk_touched_shards = 0usize;
        let mut max_chunk_backward_states = 0u64;

        if layer < 9 {
            let next_layer_ref = next_layer.as_ref().unwrap();
            let next_map_ref = next_map.as_ref().unwrap();
            let use_target_owned = std::env::var("FWD_PRUNE_TARGET_OWNED")
                .ok()
                .map(|s| s != "0")
                .unwrap_or(true);
            if use_target_owned {
                let owned_start = Instant::now();
                let source_by_range = build_forward_target_owned_source_shards(
                    layer, &cur, &graph, next_map_ref, two_line_field, num_shards,
                );
                let next_by_range = build_layer_range_block_indices(next_layer_ref, num_shards);
                let active = source_by_range.iter().enumerate()
                    .filter(|(s, v)| !v.is_empty() && !next_by_range[*s].is_empty())
                    .count();
                let max_source_blocks = source_by_range.iter().map(|v| v.len()).max().unwrap_or(0);
                let max_next_blocks = next_by_range.iter().map(|v| v.len()).max().unwrap_or(0);
                eprintln!(
                    "forward-prune-q target-owned block-qbits partition layer={} range_shards={} active={} max_source_blocks={} max_next_blocks={} elapsed={:.3}s",
                    layer, num_shards, active, max_source_blocks, max_next_blocks, owned_start.elapsed().as_secs_f64()
                );
                eprintln!(
                    "forward-prune-q target-owned block-qbits parallel layer={} rayon_threads={}",
                    layer, rayon::current_num_threads()
                );

                let owned_scratch = TargetOwnedShardScratch::create(&step_work)?;
                let owned_dir = &owned_scratch.path;
                let done = AtomicUsize::new(0);
                let total_active = active.max(1);
                let results: Vec<Result<TargetOwnedShardOutput>> = (0..num_shards).into_par_iter()
                    .filter(|&sidx| !source_by_range[sidx].is_empty() && !next_by_range[sidx].is_empty())
                    .map(|sidx| {
                        let path = owned_dir.join(format!("owned_{sidx:05}.u24.bin"));
                        let res = process_forward_target_owned_range_shard(
                            layer,
                            sidx,
                            &source_by_range[sidx],
                            &next_by_range[sidx],
                            &cur,
                            &graph,
                            &qfwd,
                            next_layer_ref,
                            two_line_field,
                            num_shards,
                            qwords,
                            &path,
                        );
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 32 == 0 || n == total_active {
                            match &res {
                                Ok(out) => eprintln!(
                                    "  forward-prune-q gen-target-owned block-qbits layer={} range_shard={}/{} active_done={}/{} source_blocks={} current_states={} raw_qfwd={} membership_hits={} emitted={} output_fields={} back_states={} elapsed={:.1}s",
                                    layer, sidx + 1, num_shards, n, total_active, source_by_range[sidx].len(),
                                    out.stats.current_states, out.stats.raw_qforward_candidates,
                                    out.stats.membership_hits, out.output_pairs, out.output_fields,
                                    out.stats.loaded_backward_states, owned_start.elapsed().as_secs_f64()
                                ),
                                Err(e) => eprintln!(
                                    "  forward-prune-q gen-target-owned block-qbits layer={} range_shard={}/{} ERROR: {:#}",
                                    layer, sidx + 1, num_shards, e
                                ),
                            }
                        }
                        res
                    })
                    .collect();

                let mut owned_outputs = Vec::<TargetOwnedShardOutput>::with_capacity(results.len());
                for r in results { owned_outputs.push(r?); }
                owned_outputs.sort_unstable_by_key(|o| o.shard);

                let out_prefix_ref = out_prefix.as_ref().unwrap();
                let mut writer = layer_store::FieldChainWriter::create(out_prefix_ref)?;
                let mut accepted_states = 0u64;
                let mut output_fields = 0usize;
                let mut nonempty_shards = 0usize;
                let mut max_shard_bytes = 0u64;
                let mut max_backward_shard_states = 0u64;
                for out in &owned_outputs {
                    let bytes = fs::metadata(&out.path)?.len();
                    let appended = append_target_owned_output_to_writer(out, &mut writer)?;
                    if appended != out.output_pairs {
                        writer.invalidate();
                        anyhow::bail!("target-owned append mismatch shard={} appended={} expected={}", out.shard, appended, out.output_pairs);
                    }
                    if bytes == 0 { continue; }
                    nonempty_shards += 1;
                    max_shard_bytes = max_shard_bytes.max(bytes);
                    max_backward_shard_states = max_backward_shard_states.max(out.stats.loaded_backward_states);
                    accepted_states += appended;
                    output_fields += out.output_fields;
                    current_states += out.stats.current_states;
                    placed_options += out.stats.placed_options;
                    graph_edges_seen += out.stats.graph_edges_seen;
                    terminal_2lpc_hits += out.stats.terminal_2lpc_hits;
                    terminal_4lpc_hits += out.stats.terminal_4lpc_hits;
                    raw_qforward_candidates += out.stats.raw_qforward_candidates;
                    generation_membership_hits += out.stats.membership_hits;
                    generation_unique_output += out.stats.unique_output;
                    candidate_pairs += out.stats.emitted_pairs;
                    max_chunk_touched_shards = max_chunk_touched_shards.max(out.stats.touched_shards);
                    max_chunk_backward_states = max_chunk_backward_states.max(out.stats.loaded_backward_states);
                }
                let written = writer.finish()?;
                if written != accepted_states {
                    anyhow::bail!("target-owned writer state mismatch: accepted={accepted_states} written={written}");
                }
                eprintln!(
                    "forward-prune-q generation layer={} target-owned block-qbits current_states={} placed_options={} graph_edges={} raw_qforward_candidates={} generation_membership_hits={} generation_unique_output={} candidate_pairs={} terminal_2lpc_hits={} terminal_4lpc_hits={} nonterminal_after_layer9={} max_chunk_touched_shards={} max_chunk_backward_states={} elapsed={:.3}s",
                    layer, current_states, placed_options, graph_edges_seen, raw_qforward_candidates,
                    generation_membership_hits, generation_unique_output, candidate_pairs,
                    terminal_2lpc_hits, terminal_4lpc_hits, nonterminal_after_layer9,
                    max_chunk_touched_shards, max_chunk_backward_states, owned_start.elapsed().as_secs_f64()
                );
                eprintln!(
                    "forward-prune-q layer={} done output_layer={} output_states={} output_fields={} target_owned=1 nonempty_shards={} max_shard_bytes={} max_backward_shard_states={} step_elapsed={:.3}s",
                    layer, layer + 1, accepted_states, output_fields, nonempty_shards,
                    max_shard_bytes, max_backward_shard_states, owned_start.elapsed().as_secs_f64()
                );
                continue;
            }

            let partition_start = Instant::now();
            let work_shards = build_forward_target_local_shards(
                layer,
                &cur,
                &graph,
                next_map_ref,
                two_line_field,
                num_shards,
                target_bucket_size,
            );
            let active_work_shards = work_shards.iter().filter(|v| !v.is_empty()).count();
            let max_work_blocks = work_shards.iter().map(|v| v.len()).max().unwrap_or(0);
            eprintln!(
                "forward-prune-q target-local block-qbits partition layer={} work_shards={} active={} max_blocks={} elapsed={:.3}s",
                layer, work_shards.len(), active_work_shards, max_work_blocks, partition_start.elapsed().as_secs_f64()
            );
            let par_batch = std::env::var("FWD_PRUNE_TARGET_LOCAL_PAR_BATCH")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(20)
                .max(1);
            eprintln!(
                "forward-prune-q target-local block-qbits parallel layer={} rayon_threads={} par_batch={}",
                layer, rayon::current_num_threads(), par_batch
            );

            let mut blocks_done = 0usize;
            let mut shards_done = 0usize;
            for batch in work_shards.chunks(par_batch) {
                let batch_results: Vec<(usize, usize, Result<TargetLocalShardOutput>)> = batch.par_iter()
                    .enumerate()
                    .filter(|(_, blocks)| !blocks.is_empty())
                    .map(|(local_idx, blocks)| {
                        let sid = shards_done + local_idx;
                        let out = process_forward_target_local_shard(
                            layer,
                            blocks,
                            &cur,
                            &graph,
                            &qfwd,
                            next_layer_ref,
                            next_map_ref,
                            two_line_field,
                            num_shards,
                            target_bucket_size,
                            qwords,
                        );
                        (sid, blocks.len(), out)
                    })
                    .collect();

                for (sid, block_count, out_res) in batch_results {
                    let out = out_res?;
                    let st = out.stats;
                    for (range_shard, pairs) in out.pairs_by_range {
                        for pair in pairs {
                            shard_writers.write_pair(range_shard, pair)?;
                        }
                    }
                    blocks_done += block_count;
                    current_states += st.current_states;
                    placed_options += st.placed_options;
                    graph_edges_seen += st.graph_edges_seen;
                    terminal_2lpc_hits += st.terminal_2lpc_hits;
                    terminal_4lpc_hits += st.terminal_4lpc_hits;
                    raw_qforward_candidates += st.raw_qforward_candidates;
                    generation_membership_hits += st.membership_hits;
                    generation_unique_output += st.unique_output;
                    candidate_pairs += st.emitted_pairs;
                    max_chunk_touched_shards = max_chunk_touched_shards.max(st.touched_shards);
                    max_chunk_backward_states = max_chunk_backward_states.max(st.loaded_backward_states);
                    if blocks_done % 1024 <= block_count || blocks_done == cur.metas.len() {
                        eprintln!(
                            "  forward-prune-q gen-target-local block-qbits parallel layer={} work_shard={}/{} blocks_done={}/{} current_states={} raw_qfwd={} membership_hits={} emitted={} term2={} term4={} out_range_shards={} shard_back_states={} elapsed={:.1}s",
                            layer, sid + 1, work_shards.len(), blocks_done, cur.metas.len(), current_states,
                            raw_qforward_candidates, generation_membership_hits, candidate_pairs,
                            terminal_2lpc_hits, terminal_4lpc_hits, st.touched_shards,
                            st.loaded_backward_states, start.elapsed().as_secs_f64()
                        );
                    }
                }
                shards_done += batch.len();
            }
        } else {
            // Layer 9 has only terminal success edges to layer10.  We do not write layer10q_pruned
            // for pruning/one-loop MDP; terminal edges are handled directly by the MDP solver.
            for (bi, meta) in cur.metas.iter().enumerate() {
                let field = meta.field_id;
                let qids = cur.read_block_chains(bi);
                current_states += qids.len() as u64;
                for qid in qids {
                    for placed in 0u8..PIECE_COUNT as u8 {
                        let recs = qfwd.records_for(qid, placed);
                        if recs.is_empty() { continue; }
                        placed_options += 1;
                        for &field2 in graph.edges(field, placed) {
                            graph_edges_seen += 1;
                            if field2 == EMPTY_FIELD_ID {
                                terminal_4lpc_hits += 1;
                            } else {
                                nonterminal_after_layer9 += 1;
                            }
                        }
                    }
                }
                if (bi + 1) % 1024 == 0 || bi + 1 == cur.metas.len() {
                    eprintln!(
                        "  forward-prune-q gen-terminal layer={} blocks_done={}/{} current_states={} term4={} nonterminal_after_layer9={} elapsed={:.1}s",
                        layer, bi + 1, cur.metas.len(), current_states, terminal_4lpc_hits,
                        nonterminal_after_layer9, start.elapsed().as_secs_f64()
                    );
                }
            }
        }
        shard_writers.flush_all()?;

        eprintln!(
            "forward-prune-q generation layer={} current_states={} placed_options={} graph_edges={} raw_qforward_candidates={} generation_membership_hits={} generation_unique_output={} candidate_pairs={} terminal_2lpc_hits={} terminal_4lpc_hits={} nonterminal_after_layer9={} max_chunk_touched_shards={} max_chunk_backward_states={} elapsed={:.3}s",
            layer, current_states, placed_options, graph_edges_seen, raw_qforward_candidates,
            generation_membership_hits, generation_unique_output, candidate_pairs,
            terminal_2lpc_hits, terminal_4lpc_hits, nonterminal_after_layer9,
            max_chunk_touched_shards, max_chunk_backward_states, start.elapsed().as_secs_f64()
        );

        if layer == 9 {
            eprintln!("forward-prune-q layer=9 terminal step done; no layer10 output is written");
            continue;
        }

        let next_layer = next_layer.as_ref().unwrap();
        let out_prefix = out_prefix.as_ref().unwrap();
        let mut writer = layer_store::FieldChainWriter::create(out_prefix)?;
        let merge_start = Instant::now();
        let mut merge_input_pairs = 0u64;
        let mut merge_membership_hits = 0u64;
        let mut accepted_states = 0u64;
        let mut output_fields = 0usize;
        let mut nonempty_shards = 0usize;
        let mut max_shard_bytes = 0u64;
        let mut max_backward_shard_states = 0u64;

        for (sidx, path) in shard_paths.iter().enumerate() {
            let bytes = fs::metadata(path)?.len();
            if bytes == 0 { continue; }
            nonempty_shards += 1;
            max_shard_bytes = max_shard_bytes.max(bytes);
            let load_start = Instant::now();
            let back = load_backward_shard_bits(next_layer, sidx, num_shards, qwords)?;
            max_backward_shard_states = max_backward_shard_states.max(back.states);
            let load_elapsed = load_start.elapsed().as_secs_f64();
            let (raw, hits, uniq, fields) = process_candidate_shard_bitset(
                path, &back, qwords, &mut writer,
            )?;
            merge_input_pairs += raw;
            merge_membership_hits += hits;
            accepted_states += uniq;
            output_fields += fields;
            eprintln!(
                "  forward-prune-q bitset-merge layer={} shard={}/{} raw_pairs={} membership_hits={} unique_output={} output_fields={} back_fields={} back_states={} load_elapsed={:.3}s elapsed={:.1}s",
                layer, sidx + 1, num_shards, raw, hits, uniq, fields, back.fields.len(), back.states, load_elapsed, merge_start.elapsed().as_secs_f64()
            );
        }
        let written = writer.finish()?;
        if written != accepted_states {
            anyhow::bail!("writer state mismatch: accepted={accepted_states} written={written}");
        }
        eprintln!(
            "forward-prune-q layer={} done output_layer={} output_states={} output_fields={} merge_input_pairs={} merge_membership_hits={} nonempty_shards={} max_shard_bytes={} max_backward_shard_states={} merge_elapsed={:.3}s step_elapsed={:.3}s",
            layer, layer + 1, accepted_states, output_fields, merge_input_pairs, merge_membership_hits,
            nonempty_shards, max_shard_bytes, max_backward_shard_states, merge_start.elapsed().as_secs_f64(), start.elapsed().as_secs_f64()
        );
    }

    eprintln!("forward-prune-q all steps done output_dir={}", output_dir.display());
    Ok(())
}


#[cfg(unix)]
fn read_exact_at_file(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let at = offset + done as u64;
        let n = file.read_at(&mut buf[done..], at)
            .with_context(|| format!("failed to read at byte offset {}", at))?;
        if n == 0 {
            anyhow::bail!("unexpected EOF reading at byte offset {}", at);
        }
        done += n;
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at_file(file: &File, buf: &[u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let at = offset + done as u64;
        let n = file.write_at(&buf[done..], at)
            .with_context(|| format!("failed to write at byte offset {}", at))?;
        if n == 0 {
            anyhow::bail!("short write at byte offset {}", at);
        }
        done += n;
    }
    Ok(())
}

#[cfg(unix)]
fn write_u16_le_buf(out: &mut Vec<u8>, x: u16) {
    out.extend_from_slice(&x.to_le_bytes());
}

#[derive(Debug)]
struct OneLoopNextBlock {
    qids: Vec<u32>,
    vals: Vec<u16>,
}

#[derive(Debug)]
struct OneLoopValueLookup<'a> {
    layer: &'a layer_store::FieldChainLayer,
    values_file: File,
    cache: HashMap<FieldId, OneLoopNextBlock>,
    cache_bytes: usize,
    cache_cap_bytes: usize,
}

impl<'a> OneLoopValueLookup<'a> {
    fn open(layer: &'a layer_store::FieldChainLayer, values_path: &Path, cache_cap_bytes: usize) -> Result<Self> {
        let actual = fs::metadata(values_path)
            .with_context(|| format!("failed to stat {}", values_path.display()))?
            .len();
        let expected = layer.total_states.checked_mul(2).context("value file size overflow")?;
        if actual != expected {
            anyhow::bail!("value file size mismatch {}: got {}, expected {}", values_path.display(), actual, expected);
        }
        let values_file = File::open(values_path)
            .with_context(|| format!("failed to open {}", values_path.display()))?;
        // metas is strictly sorted by field_id (enforced in layer_store), so a binary
        // search replaces the per-task field->block HashMap (large alloc + hash probe).
        Ok(Self { layer, values_file, cache: HashMap::new(), cache_bytes: 0, cache_cap_bytes })
    }

    fn load_block(&mut self, field: FieldId) -> Result<Option<&OneLoopNextBlock>> {
        if self.cache.contains_key(&field) {
            return Ok(self.cache.get(&field));
        }
        let Ok(bi) = self.layer.metas.binary_search_by_key(&field, |m| m.field_id) else { return Ok(None); };
        let meta = &self.layer.metas[bi];
        let qids = self.layer.read_block_chains(bi);
        let mut val_bytes = vec![0u8; meta.count as usize * 2];
        read_exact_at_file(&self.values_file, &mut val_bytes, meta.chain_offset * 2)?;
        let mut vals = Vec::<u16>::with_capacity(meta.count as usize);
        for c in val_bytes.chunks_exact(2) {
            vals.push(u16::from_le_bytes([c[0], c[1]]));
        }
        let block_bytes = qids.len() * 4 + vals.len() * 2;
        // Crude cache limiter: if this block would exceed cap, drop old cache first.
        // This intentionally avoids complex LRU overhead. Source field processing usually reuses
        // the same small target set heavily inside one block.
        if self.cache_bytes + block_bytes > self.cache_cap_bytes && !self.cache.is_empty() {
            self.cache.clear();
            self.cache_bytes = 0;
        }
        self.cache.insert(field, OneLoopNextBlock { qids, vals });
        self.cache_bytes += block_bytes;
        Ok(self.cache.get(&field))
    }

    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<u16> {
        let Some(block) = self.load_block(field)? else { return Ok(0); };
        match block.qids.binary_search(&qid) {
            Ok(pos) => Ok(block.vals[pos]),
            Err(_) => Ok(0),
        }
    }

}

#[derive(Debug)]
struct OneLoopFullBlock {
    qids: Vec<u32>,
}

#[derive(Debug)]
struct OneLoopFullLookup<'a> {
    layer: &'a layer_store::FieldChainLayer,
    cache: HashMap<FieldId, OneLoopFullBlock>,
    cache_bytes: usize,
    cache_cap_bytes: usize,
    scale: u16,
}

impl<'a> OneLoopFullLookup<'a> {
    fn open(layer: &'a layer_store::FieldChainLayer, cache_cap_bytes: usize, scale: u16) -> Result<Self> {
        // metas is strictly sorted by field_id; use binary search instead of a HashMap.
        Ok(Self { layer, cache: HashMap::new(), cache_bytes: 0, cache_cap_bytes, scale })
    }

    fn load_block(&mut self, field: FieldId) -> Result<Option<&OneLoopFullBlock>> {
        if self.cache.contains_key(&field) {
            return Ok(self.cache.get(&field));
        }
        let Ok(bi) = self.layer.metas.binary_search_by_key(&field, |m| m.field_id) else { return Ok(None); };
        let qids = self.layer.read_block_chains(bi);
        let block_bytes = qids.len() * 4;
        if self.cache_bytes + block_bytes > self.cache_cap_bytes && !self.cache.is_empty() {
            self.cache.clear();
            self.cache_bytes = 0;
        }
        self.cache.insert(field, OneLoopFullBlock { qids });
        self.cache_bytes += block_bytes;
        Ok(self.cache.get(&field))
    }

    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<u16> {
        let Some(block) = self.load_block(field)? else { return Ok(0); };
        match block.qids.binary_search(&qid) {
            Ok(_) => Ok(self.scale),
            Err(_) => Ok(0),
        }
    }
}

#[derive(Debug)]
enum OneLoopLookup<'a> {
    Values(OneLoopValueLookup<'a>),
    Full(OneLoopFullLookup<'a>),
}

impl<'a> OneLoopLookup<'a> {
    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<u16> {
        match self {
            OneLoopLookup::Values(x) => x.get(field, qid),
            OneLoopLookup::Full(x) => x.get(field, qid),
        }
    }
}

fn mdp_value_path(out_dir: &Path, layer: usize) -> PathBuf {
    out_dir.join(format!("layer{layer}.values.u16.bin"))
}

fn write_one_loop_stats(out_dir: &Path, layer: usize, total_states: u64, nonzero: u64, full: u64, max_v: u16, sum_v: u128, elapsed: f64) -> Result<()> {
    let stats_path = out_dir.join(format!("layer{layer}.values.u16.stats.txt"));
    let mut w = BufWriter::new(File::create(&stats_path)
        .with_context(|| format!("failed to create {}", stats_path.display()))?);
    let avg = if total_states == 0 { 0.0 } else { sum_v as f64 / total_states as f64 };
    writeln!(w, "layer={layer}")?;
    writeln!(w, "total_states={total_states}")?;
    writeln!(w, "nonzero={nonzero}")?;
    writeln!(w, "full={full}")?;
    writeln!(w, "max_raw={max_v}")?;
    writeln!(w, "avg_raw={avg:.6}")?;
    writeln!(w, "elapsed_sec={elapsed:.6}")?;
    Ok(())
}


#[derive(Debug, Default, Clone)]
struct OneLoopLayerStats {
    total_states: u64,
    nonzero: u64,
    full: u64,
    max_v: u16,
    sum_v: u128,
    terminal_2_edges: u64,
    terminal_4_edges: u64,
    action_count: u64,
    missing_next_terms: u64,
}

impl OneLoopLayerStats {
    fn add(&mut self, other: &OneLoopLayerStats) {
        self.total_states += other.total_states;
        self.nonzero += other.nonzero;
        self.full += other.full;
        self.max_v = self.max_v.max(other.max_v);
        self.sum_v += other.sum_v;
        self.terminal_2_edges += other.terminal_2_edges;
        self.terminal_4_edges += other.terminal_4_edges;
        self.action_count += other.action_count;
        self.missing_next_terms += other.missing_next_terms;
    }
}

#[derive(Debug)]
struct OneLoopChunkResult {
    stats: OneLoopLayerStats,
    bytes: Vec<u8>,
    blocks: usize,
}

#[allow(clippy::too_many_arguments)]
fn compute_one_loop_chunk_u16(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_full_value: bool,
    cache_cap_bytes: usize,
    scale: u16,
    two_line_field: FieldId,
) -> Result<OneLoopChunkResult> {
    let mut lookup_opt = if layer < 9 {
        let nl = next_layer.expect("missing next layer");
        if next_full_value {
            Some(OneLoopLookup::Full(OneLoopFullLookup::open(nl, cache_cap_bytes, scale)?))
        } else {
            Some(OneLoopLookup::Values(OneLoopValueLookup::open(
                nl,
                next_values_path.expect("missing next values path"),
                cache_cap_bytes,
            )?))
        }
    } else {
        None
    };

    let mut stats = OneLoopLayerStats::default();
    let mut bytes = Vec::<u8>::new();

    for bi in block_range.clone() {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        let qids = cur.read_block_chains(bi);
        bytes.reserve(qids.len() * 2);

        for qid in qids {
            stats.total_states += 1;
            let mut best = 0u16;
            for placed in 0u8..PIECE_COUNT as u8 {
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() { continue; }
                let edges = graph.edges(field, placed);
                if edges.is_empty() { continue; }
                for &field2 in edges {
                    // action_count disabled in opt5 fast path
                    let cand = if layer + 1 == 5 && field2 == two_line_field {
                        stats.terminal_2_edges += 1;
                        scale
                    } else if layer + 1 == 10 && field2 == EMPTY_FIELD_ID {
                        stats.terminal_4_edges += 1;
                        scale
                    } else if layer == 9 {
                        0
                    } else {
                        let mut s = 0u64;
                        let lookup = lookup_opt.as_mut().expect("missing next lookup");
                        for rec in recs {
                            let v = lookup.get(field2, rec.child_qid)?;
                            if v == 0 { stats.missing_next_terms += 1; }
                            s += v as u64;
                        }
                        let denom = recs.len() as u64;
                        if s % denom != 0 {
                            anyhow::bail!(
                                "one-loop value is not integral under scale: layer={} field={} qid={} placed={} field2={} sum={} denom={} scale={}",
                                layer, field, qid, placed, field2, s, denom, scale
                            );
                        }
                        (s / denom) as u16
                    };
                    if cand > best { best = cand; }
                }
            }
            if best != 0 { stats.nonzero += 1; }
            if best == scale { stats.full += 1; }
            stats.max_v = stats.max_v.max(best);
            stats.sum_v += best as u128;
            bytes.extend_from_slice(&best.to_le_bytes());
        }
    }

    Ok(OneLoopChunkResult { stats, bytes, blocks: block_range.end - block_range.start })
}

fn run_solve_one_loop_mdp_u16(args: &[String]) -> Result<()> {
    if args.len() < 5 || args.len() > 9 {
        eprintln!("usage: {} solve-one-loop-mdp-u16 <graph.bin> <pruned_layer_dir> <out_dir> [scale=5040] [lookup_cache_gib=8] [start_layer=9] [base_full_next_layer=none]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    let scale: u16 = if args.len() >= 6 { args[5].parse().context("invalid scale")? } else { 5040 };
    let cache_gib: usize = if args.len() >= 7 { args[6].parse().context("invalid lookup_cache_gib")? } else { 8 };
    let start_layer: usize = if args.len() >= 8 { args[7].parse().context("invalid start_layer")? } else { 9 };
    if start_layer > 9 { anyhow::bail!("start_layer must be 0..=9, got {}", start_layer); }
    let base_full_next_layer: Option<usize> = if args.len() >= 9 {
        let s = args[8].as_str();
        if s == "none" || s == "-" { None } else { Some(s.parse().context("invalid base_full_next_layer")?) }
    } else {
        None
    };
    if let Some(b) = base_full_next_layer {
        if b > 9 { anyhow::bail!("base_full_next_layer must be 0..=9, got {}", b); }
        if start_layer + 1 != b {
            anyhow::bail!("base_full_next_layer={} requires start_layer={}, got start_layer={}", b, b.saturating_sub(1), start_layer);
        }
    }
    let cache_cap_bytes = cache_gib.saturating_mul(1024usize).saturating_mul(1024usize).saturating_mul(1024usize);
    fs::create_dir_all(&out_dir)?;

    eprintln!("solve-one-loop-mdp-u16 loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    eprintln!("solve-one-loop-mdp-u16 building q index and qforward graph...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    eprintln!("solve-one-loop-mdp-u16 qstates={} qfwd_records={} scale={} lookup_cache_gib={} start_layer={} base_full_next_layer={:?}", q_index.states.len(), qfwd.records.len(), scale, cache_gib, start_layer, base_full_next_layer);

    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).unwrap_or(11335);
    eprintln!("solve-one-loop-mdp-u16 two_line_field={} four_line_terminal_field={}", two_line_field, EMPTY_FIELD_ID);

    for layer in (0usize..=start_layer).rev() {
        let start = Instant::now();
        let cur_prefix = q_pruned_prefix(&pruned_dir, layer);
        eprintln!("solve-one-loop-mdp-u16 layer={} loading current {}", layer, cur_prefix.display());
        let cur = layer_store::FieldChainLayer::open_index_only(&cur_prefix)?;
        let out_path = mdp_value_path(&out_dir, layer);
        let mut out = BufWriter::new(File::create(&out_path)
            .with_context(|| format!("failed to create {}", out_path.display()))?);

        let next_layer_holder;
        let next_values_path;
        if layer < 9 {
            let next_prefix = q_pruned_prefix(&pruned_dir, layer + 1);
            let next_full_value = base_full_next_layer == Some(layer + 1);
            eprintln!("solve-one-loop-mdp-u16 layer={} loading next {} full_base={}", layer, next_prefix.display(), next_full_value);
            next_layer_holder = Some(layer_store::FieldChainLayer::open_index_only(&next_prefix)?);
            next_values_path = if next_full_value { None } else { Some(mdp_value_path(&out_dir, layer + 1)) };
        } else {
            next_layer_holder = None;
            next_values_path = None;
            let _ = (&next_layer_holder, &next_values_path);
        }

        let rayon_threads = rayon::current_num_threads();
        let block_chunk: usize = env::var("ONE_LOOP_MDP_BLOCK_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64)
            .max(1);
        let batch_chunks: usize = env::var("ONE_LOOP_MDP_BATCH_CHUNKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| rayon_threads.saturating_mul(4).max(1))
            .max(1);
        let per_task_cache_cap = if layer < 9 {
            (cache_cap_bytes / rayon_threads.max(1)).max(64usize * 1024 * 1024)
        } else {
            0
        };
        eprintln!(
            "solve-one-loop-mdp-u16 layer={} parallel rayon_threads={} block_chunk={} batch_chunks={} per_task_cache_mib={}",
            layer, rayon_threads, block_chunk, batch_chunks, per_task_cache_cap / (1024 * 1024)
        );

        let next_layer_ref = next_layer_holder.as_ref();
        let next_values_ref = next_values_path.as_deref();
        let next_full_value = base_full_next_layer == Some(layer + 1);
        let mut stats = OneLoopLayerStats::default();
        let mut blocks_done = 0usize;

        let ranges: Vec<std::ops::Range<usize>> = (0..cur.metas.len())
            .step_by(block_chunk)
            .map(|s| s..(s + block_chunk).min(cur.metas.len()))
            .collect();

        for batch in ranges.chunks(batch_chunks) {
            let results: Vec<Result<OneLoopChunkResult>> = batch
                .par_iter()
                .map(|r| {
                    compute_one_loop_chunk_u16(
                        layer,
                        r.clone(),
                        &cur,
                        &graph,
                        &qfwd,
                        next_layer_ref,
                        next_values_ref,
                        next_full_value,
                        per_task_cache_cap,
                        scale,
                        two_line_field,
                    )
                })
                .collect();

            for res in results {
                let chunk = res?;
                out.write_all(&chunk.bytes)?;
                stats.add(&chunk.stats);
                blocks_done += chunk.blocks;
            }

            eprintln!(
                "  solve-one-loop-mdp-u16 layer={} blocks_done={}/{} states={} nonzero={} full={} max_raw={} term2={} term4={} actions={} elapsed={:.1}s",
                layer, blocks_done, cur.metas.len(), stats.total_states, stats.nonzero, stats.full,
                stats.max_v, stats.terminal_2_edges, stats.terminal_4_edges,
                stats.action_count, start.elapsed().as_secs_f64()
            );
        }
        out.flush()?;
        let actual = fs::metadata(&out_path)?.len();
        let expected = cur.total_states * 2;
        if actual != expected {
            anyhow::bail!("value output size mismatch layer={}: got {}, expected {}", layer, actual, expected);
        }
        write_one_loop_stats(&out_dir, layer, stats.total_states, stats.nonzero, stats.full, stats.max_v, stats.sum_v, start.elapsed().as_secs_f64())?;
        eprintln!(
            "solve-one-loop-mdp-u16 layer={} done states={} nonzero={} full={} max_raw={} avg_raw={:.6} terminal_2lpc_edges={} terminal_4lpc_edges={} actions={} missing_zero_terms={} bytes={} elapsed={:.3}s",
            layer, stats.total_states, stats.nonzero, stats.full, stats.max_v,
            if stats.total_states == 0 { 0.0 } else { stats.sum_v as f64 / stats.total_states as f64 },
            stats.terminal_2_edges, stats.terminal_4_edges, stats.action_count, stats.missing_next_terms, actual, start.elapsed().as_secs_f64()
        );
    }

    let l0 = mdp_value_path(&out_dir, 0);
    let special = out_dir.join("layer0_one_loop_pc_count_5040.u16.bin");
    fs::copy(&l0, &special).with_context(|| format!("failed to copy {} -> {}", l0.display(), special.display()))?;
    eprintln!("solve-one-loop-mdp-u16 wrote special layer0 one-loop counts {}", special.display());
    eprintln!("solve-one-loop-mdp-u16 all done out_dir={}", out_dir.display());
    Ok(())
}



fn mdp_f32_value_path(base_dir: &Path, layer: usize) -> PathBuf {
    base_dir.join(format!("layer{layer}.values.f32.bin"))
}

fn mdp_policy_u16_path(base_dir: &Path, layer: usize) -> PathBuf {
    base_dir.join(format!("layer{layer}.policy.u16.bin"))
}

// Per-state greedy action code: high byte = placed piece (0..6), low byte = index into
// graph.edges(field, placed). 0xFFFF = no action (dead state, V = 0). placed <= 6 keeps
// every real code's high byte below 0xFF, so it can never collide with POLICY_NONE_U16.
const POLICY_NONE_U16: u16 = 0xFFFF;

#[inline]
fn encode_policy_u16(placed: u8, edge_idx: usize) -> u16 {
    debug_assert!((placed as usize) < PIECE_COUNT);
    debug_assert!(edge_idx < 256);
    ((placed as u16) << 8) | (edge_idx as u16)
}

fn mdp_l0_input_value_path(base_dir: &Path) -> PathBuf {
    base_dir.join("layer0.input.values.f32.bin")
}

fn mdp_l0_next_value_path(base_dir: &Path) -> PathBuf {
    base_dir.join("layer0.next.values.f32.bin")
}

fn mdp_l0_last_f_value_path(base_dir: &Path) -> PathBuf {
    base_dir.join("layer0.last_F.values.f32.bin")
}

fn mdp_l0_work_value_path(work_dir: &Path, layer: usize) -> PathBuf {
    work_dir.join(format!("layer{layer}.values.f32.bin"))
}

fn write_f32_le_buf(out: &mut Vec<u8>, x: f32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn create_zero_f32_file(path: &Path, states: u64) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?);
    let chunk_vals = 1usize << 20;
    let zeros = vec![0u8; chunk_vals * 4];
    let mut remain = states;
    while remain > 0 {
        let n = (remain as usize).min(chunk_vals);
        out.write_all(&zeros[..n * 4])?;
        remain -= n as u64;
    }
    out.flush()?;
    Ok(())
}

fn create_full_f32_file(path: &Path, states: u64, value: f32) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?);
    let chunk_vals = 1usize << 20;
    let mut buf = Vec::<u8>::with_capacity(chunk_vals * 4);
    for _ in 0..chunk_vals { buf.extend_from_slice(&value.to_le_bytes()); }
    let mut remain = states;
    while remain > 0 {
        let n = (remain as usize).min(chunk_vals);
        out.write_all(&buf[..n * 4])?;
        remain -= n as u64;
    }
    out.flush()?;
    Ok(())
}

fn convert_one_loop_u16_to_f32(src: &Path, dst: &Path, states: u64, scale: f32) -> Result<()> {
    let actual = fs::metadata(src)
        .with_context(|| format!("failed to stat {}", src.display()))?
        .len();
    let expected = states.checked_mul(2).context("u16 expected size overflow")?;
    if actual != expected {
        anyhow::bail!("one-loop init size mismatch {}: got {}, expected {}", src.display(), actual, expected);
    }
    let mut inp = BufReader::new(File::open(src)
        .with_context(|| format!("failed to open {}", src.display()))?);
    let mut out = BufWriter::new(File::create(dst)
        .with_context(|| format!("failed to create {}", dst.display()))?);
    let mut buf = vec![0u8; 1usize << 22]; // even byte count
    let mut carry = Vec::<u8>::new();
    loop {
        let n = inp.read(&mut buf)?;
        if n == 0 { break; }
        let mut bytes = Vec::with_capacity(carry.len() + n);
        bytes.extend_from_slice(&carry);
        bytes.extend_from_slice(&buf[..n]);
        let even = bytes.len() & !1usize;
        let mut ob = Vec::<u8>::with_capacity((even / 2) * 4);
        for c in bytes[..even].chunks_exact(2) {
            let raw = u16::from_le_bytes([c[0], c[1]]);
            ob.extend_from_slice(&((raw as f32) / scale).to_le_bytes());
        }
        out.write_all(&ob)?;
        carry.clear();
        if even != bytes.len() { carry.extend_from_slice(&bytes[even..]); }
    }
    if !carry.is_empty() { anyhow::bail!("odd trailing byte in {}", src.display()); }
    out.flush()?;
    let got = fs::metadata(dst)?.len();
    let exp = states.checked_mul(4).context("f32 expected size overflow")?;
    if got != exp { anyhow::bail!("converted f32 size mismatch {}: got {}, expected {}", dst.display(), got, exp); }
    Ok(())
}

#[derive(Debug)]
enum DenseF32Block {
    /// Full qid -> value table. Fast O(1) lookup, expensive to materialize.
    Dense { vals: Vec<f32> },
    /// Sorted qids for a small target field. Avoids allocating/zeroing qstate_count floats.
    Sparse { qids: Vec<u32>, vals: Vec<f32> },
}

impl DenseF32Block {
    #[inline]
    fn get(&self, qid: u32) -> f32 {
        match self {
            DenseF32Block::Dense { vals } => vals[qid as usize],
            DenseF32Block::Sparse { qids, vals } => {
                // Most target field blocks in the huge middle layers are sparse.
                // For tiny blocks, a linear scan beats binary_search's branchy setup.
                if qids.len() <= 32 {
                    for (i, &x) in qids.iter().enumerate() {
                        if x == qid { return vals[i]; }
                    }
                    0.0
                } else {
                    match qids.binary_search(&qid) {
                        Ok(i) => vals[i],
                        Err(_) => 0.0,
                    }
                }
            },
        }
    }

    #[inline]
    fn cache_bytes(&self) -> usize {
        match self {
            DenseF32Block::Dense { vals } => vals.capacity() * std::mem::size_of::<f32>() + 256,
            DenseF32Block::Sparse { qids, vals } => {
                qids.capacity() * std::mem::size_of::<u32>() + vals.capacity() * std::mem::size_of::<f32>() + 256
            }
        }
    }
}

#[derive(Default, Debug, Clone)]
struct DenseLookupStats {
    cache_hits: u64,
    cache_misses: u64,
    missing_fields: u64,
    evictions: u64,
    materialized_states: u64,
    dense_blocks: u64,
    sparse_blocks: u64,
    bitmap_blocks: u64,
    const_checks: u64,
    const_hits: u64,
    const_misses: u64,
}

#[derive(Debug)]
struct DenseCacheEntry {
    block: DenseF32Block,
    epoch: u64,
}

#[derive(Debug)]
struct DenseF32ValueLookup<'a> {
    layer: &'a layer_store::FieldChainLayer,
    values_file: File,
    cache: HashMap<FieldId, DenseCacheEntry>,
    // Load-order FIFO cache. Hits do not mutate the queue, keeping memory bounded and
    // avoiding the previous O(cache_size) touch_lru scan in the hot lookup path.
    cache_order: VecDeque<(FieldId, u64)>,
    cache_epoch: u64,
    cache_bytes: usize,
    cache_cap_bytes: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    collect_stats: bool,
    stats: DenseLookupStats,
    scratch_qids: Vec<u32>,
    scratch_qid_bytes: Vec<u8>,
    scratch_val_bytes: Vec<u8>,
}

impl<'a> DenseF32ValueLookup<'a> {
    fn open(
        layer: &'a layer_store::FieldChainLayer,
        values_path: &Path,
        cache_cap_bytes: usize,
        qstate_count: usize,
        sparse_lookup_max_count: usize,
        collect_stats: bool,
    ) -> Result<Self> {
        let actual = fs::metadata(values_path)
            .with_context(|| format!("failed to stat {}", values_path.display()))?
            .len();
        let expected = layer.total_states.checked_mul(4).context("f32 value file size overflow")?;
        if actual != expected {
            anyhow::bail!("f32 value file size mismatch {}: got {}, expected {}", values_path.display(), actual, expected);
        }
        let values_file = File::open(values_path)
            .with_context(|| format!("failed to open {}", values_path.display()))?;
        Ok(Self {
            layer,
            values_file,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_epoch: 0,
            cache_bytes: 0,
            cache_cap_bytes,
            qstate_count,
            sparse_lookup_max_count,
            collect_stats,
            stats: DenseLookupStats::default(),
            scratch_qids: Vec::new(),
            scratch_qid_bytes: Vec::new(),
            scratch_val_bytes: Vec::new(),
        })
    }

    #[inline]
    fn bump_lru(&mut self, field: FieldId) -> u64 {
        self.cache_epoch = self.cache_epoch.wrapping_add(1);
        let epoch = self.cache_epoch;
        self.cache_order.push_back((field, epoch));
        epoch
    }

    fn load_block(&mut self, field: FieldId) -> Result<Option<&DenseF32Block>> {
        // Use contains_key + a second get on the return path. Returning a reference from
        // HashMap::get() in the same if-let that precedes later cache mutation extends
        // the immutable borrow too far for the borrow checker.
        if self.cache.contains_key(&field) {
            if self.collect_stats { self.stats.cache_hits += 1; }
            return Ok(self.cache.get(&field).map(|entry| &entry.block));
        }
        let Ok(bi) = self.layer.metas.binary_search_by_key(&field, |m| m.field_id) else {
            if self.collect_stats { self.stats.missing_fields += 1; }
            return Ok(None);
        };
        if self.collect_stats { self.stats.cache_misses += 1; }
        let meta = &self.layer.metas[bi];
        let count = meta.count as usize;
        if count == 0 { return Ok(None); }

        // Reuse the qid and byte buffers across cache misses. Sparse blocks take ownership
        // of qids; dense blocks return the scratch qid vector for the next miss.
        let mut qids = std::mem::take(&mut self.scratch_qids);
        self.layer.read_block_chains_into_with_buf(bi, &mut qids, &mut self.scratch_qid_bytes);
        self.scratch_val_bytes.resize(count * 4, 0);
        read_exact_at_file(&self.values_file, self.scratch_val_bytes.as_mut_slice(), meta.chain_offset * 4)?;

        let block = if count <= self.sparse_lookup_max_count {
            let mut vals = Vec::<f32>::with_capacity(count);
            for c in self.scratch_val_bytes.chunks_exact(4) {
                vals.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            if self.collect_stats { self.stats.sparse_blocks += 1; }
            DenseF32Block::Sparse { qids, vals }
        } else {
            let mut dense = vec![0.0f32; self.qstate_count];
            for (i, c) in self.scratch_val_bytes.chunks_exact(4).enumerate() {
                dense[qids[i] as usize] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            qids.clear();
            self.scratch_qids = qids;
            if self.collect_stats { self.stats.dense_blocks += 1; }
            DenseF32Block::Dense { vals: dense }
        };

        if self.collect_stats { self.stats.materialized_states += meta.count as u64; }
        let block_bytes = block.cache_bytes();
        if block_bytes <= self.cache_cap_bytes {
            while self.cache_bytes + block_bytes > self.cache_cap_bytes {
                let Some((old_field, old_epoch)) = self.cache_order.pop_front() else { break; };
                let remove = self.cache.get(&old_field)
                    .map(|entry| entry.epoch == old_epoch)
                    .unwrap_or(false);
                if remove {
                    if let Some(old_entry) = self.cache.remove(&old_field) {
                        self.cache_bytes = self.cache_bytes.saturating_sub(old_entry.block.cache_bytes());
                        if self.collect_stats { self.stats.evictions += 1; }
                    }
                }
            }
        } else {
            // Budget smaller than one block. Keep only this block to avoid immediate re-read inside this action.
            let evicted = self.cache.len() as u64;
            self.cache.clear();
            self.cache_order.clear();
            self.cache_bytes = 0;
            if self.collect_stats { self.stats.evictions += evicted; }
        }
        let epoch = self.bump_lru(field);
        self.cache.insert(field, DenseCacheEntry { block, epoch });
        self.cache_bytes += block_bytes;
        Ok(self.cache.get(&field).map(|e| &e.block))
    }

    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<f32> {
        let Some(block) = self.load_block(field)? else { return Ok(0.0); };
        Ok(block.get(qid))
    }

    /// Resolve the target field block ONCE, then sum V over every reveal child. All recs
    /// of one action share `field`, so this replaces up to 7 per-reveal load_block()
    /// probes with a single one. Returns (sum, missing_children) where an absent child
    /// contributes 0. Hot path.
    #[inline]
    fn sum_recs(&mut self, field: FieldId, recs: &[qgraph::QForwardRec]) -> Result<(f32, u64)> {
        let Some(block) = self.load_block(field)? else {
            return Ok((0.0, recs.len() as u64));
        };
        let mut s = 0.0f32;
        let mut miss = 0u64;
        for r in recs {
            let v = block.get(r.child_qid);
            s += v;
            if v == 0.0 { miss += 1; }
        }
        Ok((s, miss))
    }

    fn prewarm(&mut self, field: FieldId) -> Result<()> {
        let _ = self.load_block(field)?;
        Ok(())
    }

    fn stats_snapshot(&self) -> DenseLookupStats { self.stats.clone() }
}

#[derive(Debug)]
enum RamQidIndex {
    Sparse(Box<[u32]>),
    Bitmap(qid_rank::QidRankIndex),
    Dense(Box<[f32]>),
}

#[derive(Debug)]
struct RamQidBlock {
    index: RamQidIndex,
    qid_count: usize,
    value_offset: usize,
}

impl RamQidBlock {
    fn new(qids: Vec<u32>, value_offset: usize, bitmap_min_count: usize) -> Result<Self> {
        let qid_count = qids.len();
        // A bitmap only replaces a sufficiently large sparse index when its allocated
        // words + rank directory use no more bytes than the original qid list. Boxing
        // all retained arrays makes cache accounting independent of Vec growth slack.
        let use_bitmap = qid_count >= bitmap_min_count
            && qid_rank::QidRankIndex::estimated_heap_bytes(&qids) <= qid_count * 4;
        let index = if use_bitmap {
            let bitmap = qid_rank::QidRankIndex::from_sorted(&qids, CHECK_QSTATE_COUNT)
                .map_err(|message| anyhow::anyhow!(message))?;
            // The qid allocation is dropped here; only membership/rank is retained.
            RamQidIndex::Bitmap(bitmap)
        } else {
            RamQidIndex::Sparse(qids.into_boxed_slice())
        };
        Ok(Self { index, qid_count, value_offset })
    }

    #[inline]
    fn value_index(&self, qid: u32) -> Option<usize> {
        let local = match &self.index {
            RamQidIndex::Sparse(qids) if qids.len() <= 32 => {
                let mut found = None;
                for (i, &x) in qids.iter().enumerate() {
                    if x == qid { found = Some(i); break; }
                }
                found?
            }
            RamQidIndex::Sparse(qids) => qids.binary_search(&qid).ok()?,
            RamQidIndex::Bitmap(bitmap) => bitmap.rank(qid)?,
            RamQidIndex::Dense(_) => return None,
        };
        Some(self.value_offset + local)
    }

    #[inline]
    fn dense_values(&self) -> Option<&[f32]> {
        match &self.index { RamQidIndex::Dense(values) => Some(values), _ => None }
    }

    #[inline]
    fn cache_bytes(&self) -> usize {
        // Also reserve space for the cache entry, map bucket and FIFO pair. Arrays
        // are boxed so their allocated payload equals the charged slice lengths.
        256 + match &self.index {
            RamQidIndex::Sparse(qids) => std::mem::size_of_val(qids.as_ref()),
            RamQidIndex::Bitmap(bitmap) => bitmap.heap_bytes(),
            RamQidIndex::Dense(values) => std::mem::size_of_val(values.as_ref()),
        }
    }

    fn promote(&mut self, values: &[f32]) {
        let mut dense = vec![0.0f32; CHECK_QSTATE_COUNT].into_boxed_slice();
        let offset = self.value_offset;
        match &self.index {
            RamQidIndex::Sparse(qids) => {
                for (i, &qid) in qids.iter().enumerate() { dense[qid as usize] = values[offset + i]; }
            }
            RamQidIndex::Bitmap(bitmap) => {
                // Set bits enumerate the original sorted qid order, so the pinned
                // values retain exactly the same local index after conversion.
                for (i, qid) in bitmap.iter_qids().enumerate() { dense[qid as usize] = values[offset + i]; }
            }
            RamQidIndex::Dense(_) => return,
        }
        self.index = RamQidIndex::Dense(dense);
    }
}

#[derive(Debug)]
struct RamQidCacheEntry {
    block: RamQidBlock,
    epoch: u64,
    // Cache hits since insertion. Densification is deferred until a block proves hot
    // (hits >= dense_heat_min): cold fields keep the cheap sparse qid list and never pay
    // the 3.27 MB materialization, which otherwise thrashes when the next layer has many
    // distinct fields (measured: L6 sweep 612s -> 1622s with eager densification).
    hits: u32,
}

/// RAM-backed next-layer value lookup.
///
/// The next layer's values are fully resident in `values`, while this lookup caches
/// qid indexes for target fields. Sparse and bitmap indexes read the pinned values
/// directly; the existing heat gate may materialize a dense value copy for hot fields.
/// Every lookup is tied to this immutable `values` borrow for one layer/sweep.
#[derive(Debug)]
struct RamNextValueLookup<'a> {
    layer: &'a layer_store::FieldChainLayer,
    values: &'a [f32],
    const_fastpath: bool,
    const_cache: HashMap<FieldId, Option<f32>>,
    cache: HashMap<FieldId, RamQidCacheEntry>,
    cache_order: VecDeque<(FieldId, u64)>,
    cache_epoch: u64,
    cache_bytes: usize,
    cache_cap_bytes: usize,
    collect_stats: bool,
    stats: DenseLookupStats,
    scratch_qids: Vec<u32>,
    scratch_qid_bytes: Vec<u8>,
    // 0 in the environment disables bitmap indexes. The memory comparison in
    // RamQidBlock::new additionally prevents bitmap expansion for sparse fields.
    bitmap_min_count: usize,
    // Blocks with at least this many qids get a dense qid->value copy on materialization.
    // usize::MAX disables densification entirely (FULL_MDP_RAM_DENSE_MIN_COUNT=0 to disable).
    dense_min_count: usize,
    // A block must additionally be HIT this many times before it is densified (heat gate).
    dense_heat_min: u32,
    // Sticky cap: at most this many dense blocks live per task. Once full, promotion stops
    // (slots free up only when FIFO eviction removes a dense entry). Without this cap a
    // next layer with thousands of hot fields cycles promote->evict->reheat forever
    // (measured L6: 612s baseline -> 1466s with heat gate alone).
    dense_max_blocks: usize,
    dense_live: usize,
}

fn ram_dense_min_count_from_env() -> usize {
    match env::var("FULL_MDP_RAM_DENSE_MIN_COUNT").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(0) => usize::MAX, // explicit opt-out
        Some(n) => n,
        None => 512,
    }
}

fn ram_bitmap_min_count_from_env() -> usize {
    match env::var("FULL_MDP_RAM_BITMAP_MIN_COUNT").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(0) => usize::MAX,
        Some(n) => n,
        // Index construction loses on cold fields even when point lookups are
        // faster. Keep this measured experiment opt-in until workload reuse is known.
        None => usize::MAX,
    }
}

fn ram_dense_heat_min_from_env() -> u32 {
    env::var("FULL_MDP_RAM_DENSE_HEAT_MIN").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(64)
}

fn ram_dense_max_blocks_from_env() -> usize {
    env::var("FULL_MDP_RAM_DENSE_MAX_BLOCKS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(160)
}

impl<'a> RamNextValueLookup<'a> {
    fn new(
        layer: &'a layer_store::FieldChainLayer,
        values: &'a [f32],
        cache_cap_bytes: usize,
        collect_stats: bool,
        const_fastpath: bool,
    ) -> Result<Self> {
        if values.len() != layer.total_states as usize {
            anyhow::bail!(
                "RAM next values length mismatch: got {}, expected {}",
                values.len(), layer.total_states
            );
        }
        Ok(Self {
            layer,
            values,
            const_fastpath,
            const_cache: HashMap::new(),
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_epoch: 0,
            cache_bytes: 0,
            cache_cap_bytes,
            collect_stats,
            stats: DenseLookupStats::default(),
            scratch_qids: Vec::new(),
            scratch_qid_bytes: Vec::new(),
            bitmap_min_count: ram_bitmap_min_count_from_env(),
            dense_min_count: ram_dense_min_count_from_env(),
            dense_heat_min: ram_dense_heat_min_from_env(),
            dense_max_blocks: ram_dense_max_blocks_from_env(),
            dense_live: 0,
        })
    }

    #[inline]
    fn bump_fifo_epoch(&mut self, field: FieldId) -> u64 {
        self.cache_epoch = self.cache_epoch.wrapping_add(1);
        let epoch = self.cache_epoch;
        self.cache_order.push_back((field, epoch));
        epoch
    }

    fn load_block(&mut self, field: FieldId) -> Result<Option<&RamQidBlock>> {
        // Same pattern as DenseF32ValueLookup::load_block(): avoid keeping the
        // immutable get() borrow alive across the miss/eviction mutation path.
        // Heat-gated + sticky-capped densification: only a block that keeps getting hit
        // pays the 3.27 MB materialization (eager densify-on-miss thrashed: L6 612->1622s),
        // and at most dense_max_blocks live at once (uncapped promotion cycles
        // promote->evict->reheat: L6 still 1466s with the heat gate alone).
        let hit_promote = match self.cache.get_mut(&field) {
            Some(e) => {
                e.hits = e.hits.saturating_add(1);
                let p = e.block.dense_values().is_none()
                    && e.block.qid_count >= self.dense_min_count
                    && e.hits >= self.dense_heat_min;
                Some(p)
            }
            None => None,
        };
        if let Some(promote) = hit_promote {
            if self.collect_stats { self.stats.cache_hits += 1; }
            if promote && self.dense_live < self.dense_max_blocks {
                self.promote_to_dense(field);
            }
            return Ok(self.cache.get(&field).map(|entry| &entry.block));
        }
        let Ok(bi) = self.layer.metas.binary_search_by_key(&field, |m| m.field_id) else {
            if self.collect_stats { self.stats.missing_fields += 1; }
            return Ok(None);
        };
        if self.collect_stats { self.stats.cache_misses += 1; }
        let meta = &self.layer.metas[bi];
        if meta.count == 0 { return Ok(None); }
        let mut qids = std::mem::take(&mut self.scratch_qids);
        self.layer.read_block_chains_into_with_buf(bi, &mut qids, &mut self.scratch_qid_bytes);
        let value_offset = meta.chain_offset as usize;
        // Membership indexes may use a bitmap immediately; copying values into a
        // dense array still waits for the established heat gate and sticky cap.
        let block = RamQidBlock::new(qids, value_offset, self.bitmap_min_count)?;
        if self.collect_stats {
            self.stats.materialized_states += meta.count as u64;
            match &block.index {
                RamQidIndex::Bitmap(_) => self.stats.bitmap_blocks += 1,
                _ => self.stats.sparse_blocks += 1,
            }
        }
        let block_bytes = block.cache_bytes();
        if block_bytes <= self.cache_cap_bytes {
            while self.cache_bytes + block_bytes > self.cache_cap_bytes {
                let Some((old_field, old_epoch)) = self.cache_order.pop_front() else { break; };
                let remove = self.cache.get(&old_field)
                    .map(|entry| entry.epoch == old_epoch)
                    .unwrap_or(false);
                if remove {
                    if let Some(old_entry) = self.cache.remove(&old_field) {
                        self.cache_bytes = self.cache_bytes.saturating_sub(old_entry.block.cache_bytes());
                        if old_entry.block.dense_values().is_some() { self.dense_live = self.dense_live.saturating_sub(1); }
                        if self.collect_stats { self.stats.evictions += 1; }
                    }
                }
            }
        } else {
            let evicted = self.cache.len() as u64;
            self.cache.clear();
            self.cache_order.clear();
            self.cache_bytes = 0;
            self.dense_live = 0;
            if self.collect_stats { self.stats.evictions += evicted; }
        }
        let epoch = self.bump_fifo_epoch(field);
        self.cache.insert(field, RamQidCacheEntry { block, epoch, hits: 0 });
        self.cache_bytes += block_bytes;
        Ok(self.cache.get(&field).map(|e| &e.block))
    }

    // Promote a proven-hot sparse block to a dense qid-indexed array. The entry is removed,
    // densified, and re-inserted through the normal eviction path; the stale FIFO queue pair
    // for the old epoch is skipped by the existing epoch check. Values are copied verbatim
    // from the pinned layer (absent qids stay 0.0), so lookups remain bit-identical.
    fn promote_to_dense(&mut self, field: FieldId) {
        let Some(mut entry) = self.cache.remove(&field) else { return; };
        self.cache_bytes = self.cache_bytes.saturating_sub(entry.block.cache_bytes());
        entry.block.promote(self.values);
        if self.collect_stats { self.stats.dense_blocks += 1; }
        let block_bytes = entry.block.cache_bytes();
        if block_bytes <= self.cache_cap_bytes {
            while self.cache_bytes + block_bytes > self.cache_cap_bytes {
                let Some((old_field, old_epoch)) = self.cache_order.pop_front() else { break; };
                let remove = self.cache.get(&old_field)
                    .map(|e| e.epoch == old_epoch)
                    .unwrap_or(false);
                if remove {
                    if let Some(old_entry) = self.cache.remove(&old_field) {
                        self.cache_bytes = self.cache_bytes.saturating_sub(old_entry.block.cache_bytes());
                        if old_entry.block.dense_values().is_some() { self.dense_live = self.dense_live.saturating_sub(1); }
                        if self.collect_stats { self.stats.evictions += 1; }
                    }
                }
            }
        } else {
            let evicted = self.cache.len() as u64;
            self.cache.clear();
            self.cache_order.clear();
            self.cache_bytes = 0;
            self.dense_live = 0;
            if self.collect_stats { self.stats.evictions += evicted; }
        }
        let epoch = self.bump_fifo_epoch(field);
        entry.epoch = epoch;
        self.cache.insert(field, entry);
        self.cache_bytes += block_bytes;
        self.dense_live += 1;
    }

    fn const_value(&mut self, _field: FieldId) -> Result<Option<f32>> {
        // PERMANENTLY DISABLED (unsound). The const-field fast path used to collapse a
        // target field to `c * recs.len()` when every *present* qid in that field block
        // equalled `c`, WITHOUT verifying that each reveal's individual `child_qid` was
        // actually present. A pruned (absent) child must contribute 0, not `c`, so the
        // fold over-counted actions and silently biased MDP values upward whenever the
        // RAM next-layer backend was used (FULL_MDP_RAM_NEXT_LAYER=1). Returning None
        // unconditionally forces the correct per-reveal getv() path for every field,
        // regardless of self.const_fastpath or the FULL_MDP_CONST_FIELD_FASTPATH flag.
        //
        // To re-enable safely, restore the block-scan fold BUT first confirm each
        // `child_qid` is present in the block (falling back to 0 for absent children),
        // or restrict the fold to fields proven closed under qforward reveals.
        Ok(None)
    }

    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<f32> {
        let value_index = {
            let Some(block) = self.load_block(field)? else { return Ok(0.0); };
            if let Some(d) = block.dense_values() {
                return Ok(d.get(qid as usize).copied().unwrap_or(0.0));
            }
            block.value_index(qid)
        };
        Ok(value_index.and_then(|idx| self.values.get(idx).copied()).unwrap_or(0.0))
    }

    /// Resolve the target field block ONCE, then sum V over every reveal child. Reveals
    /// come from a 7-bag so `recs.len() <= 7`; the value indexes are buffered on the stack
    /// while the block borrow is live, then `self.values` is read after it is dropped
    /// (load_block borrows self mutably). Returns (sum, missing_children). Hot path.
    /// Dense blocks skip the binary search entirely: prefetch all reveal children, then
    /// sum in recs order (identical FP order to the sparse path -> bit-identical result;
    /// absent children read 0.0 in both paths and count as misses in both paths).
    #[inline]
    fn sum_recs(&mut self, field: FieldId, recs: &[qgraph::QForwardRec]) -> Result<(f32, u64)> {
        let n = recs.len();
        if n <= 8 {
            let mut idxs: [Option<usize>; 8] = [None; 8];
            {
                let Some(block) = self.load_block(field)? else {
                    return Ok((0.0, n as u64));
                };
                if let Some(d) = block.dense_values() {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        for r in recs {
                            if (r.child_qid as usize) < d.len() {
                                std::arch::x86_64::_mm_prefetch(
                                    d.as_ptr().add(r.child_qid as usize) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                    }
                    let mut s = 0.0f32;
                    let mut miss = 0u64;
                    for r in recs {
                        let v = d.get(r.child_qid as usize).copied().unwrap_or(0.0);
                        s += v;
                        if v == 0.0 { miss += 1; }
                    }
                    return Ok((s, miss));
                }
                for (k, r) in recs.iter().enumerate() {
                    idxs[k] = block.value_index(r.child_qid);
                }
            }
            let mut s = 0.0f32;
            let mut miss = 0u64;
            for slot in idxs.iter().take(n) {
                let v = slot.and_then(|i| self.values.get(i).copied()).unwrap_or(0.0);
                s += v;
                if v == 0.0 { miss += 1; }
            }
            Ok((s, miss))
        } else {
            // Defensive fallback (never taken for a 7-bag): per-reveal get().
            let mut s = 0.0f32;
            let mut miss = 0u64;
            for r in recs {
                let v = self.get(field, r.child_qid)?;
                s += v;
                if v == 0.0 { miss += 1; }
            }
            Ok((s, miss))
        }
    }

    fn prewarm(&mut self, field: FieldId) -> Result<()> {
        let _ = self.load_block(field)?;
        Ok(())
    }

    fn stats_snapshot(&self) -> DenseLookupStats { self.stats.clone() }
}

#[cfg(test)]
#[path = "ram_qid_tests.rs"]
mod ram_qid_tests;

#[derive(Debug)]
enum NextValueLookup<'a> {
    File(DenseF32ValueLookup<'a>),
    Ram(RamNextValueLookup<'a>),
    TerminalL9(terminal_l9::Lookup<'a>),
    CompiledL9(loop1_stream::CompiledL9Lookup<'a>),
}

impl<'a> NextValueLookup<'a> {
    #[inline]
    fn get(&mut self, field: FieldId, qid: u32) -> Result<f32> {
        match self {
            NextValueLookup::File(x) => x.get(field, qid),
            NextValueLookup::Ram(x) => x.get(field, qid),
            NextValueLookup::TerminalL9(x) => x.get(field, qid),
            NextValueLookup::CompiledL9(x) => x.get(field, qid),
        }
    }

    /// Sum V over all reveal children of one action, resolving the shared target field
    /// block only once. Returns (sum, missing_children).
    #[inline]
    fn sum_recs(&mut self, field: FieldId, recs: &[qgraph::QForwardRec]) -> Result<(f32, u64)> {
        match self {
            NextValueLookup::File(x) => x.sum_recs(field, recs),
            NextValueLookup::Ram(x) => x.sum_recs(field, recs),
            NextValueLookup::TerminalL9(x) => x.sum_recs(field, recs),
            NextValueLookup::CompiledL9(x) => x.sum_recs(field, recs),
        }
    }

    fn const_value(&mut self, field: FieldId) -> Result<Option<f32>> {
        match self {
            NextValueLookup::File(_) => Ok(None),
            NextValueLookup::Ram(x) => x.const_value(field),
            NextValueLookup::TerminalL9(_) => Ok(None),
            NextValueLookup::CompiledL9(_) => Ok(None),
        }
    }

    fn prewarm(&mut self, field: FieldId) -> Result<()> {
        match self {
            NextValueLookup::File(x) => x.prewarm(field),
            NextValueLookup::Ram(x) => x.prewarm(field),
            NextValueLookup::TerminalL9(x) => x.prewarm(field),
            NextValueLookup::CompiledL9(x) => x.prewarm(field),
        }
    }

    fn stats_snapshot(&self) -> DenseLookupStats {
        match self {
            NextValueLookup::File(x) => x.stats_snapshot(),
            NextValueLookup::Ram(x) => x.stats_snapshot(),
            NextValueLookup::TerminalL9(x) => x.stats_snapshot(),
            NextValueLookup::CompiledL9(x) => x.stats_snapshot(),
        }
    }
}

fn load_f32_values_vec(path: &Path, states: u64) -> Result<Vec<f32>> {
    f32_ram::load(path, states)
}

fn load_layer0_reset_dense(layer0: &layer_store::FieldChainLayer, values_path: &Path, qstate_count: usize) -> Result<Vec<f32>> {
    let actual = fs::metadata(values_path)
        .with_context(|| format!("failed to stat {}", values_path.display()))?
        .len();
    let expected = layer0.total_states.checked_mul(4).context("layer0 f32 size overflow")?;
    if actual != expected { anyhow::bail!("layer0 f32 size mismatch: got {}, expected {}", actual, expected); }
    if layer0.metas.is_empty() { anyhow::bail!("empty layer0 membership"); }
    let bi = if layer0.metas.len() == 1 {
        0usize
    } else if let Some(pos) = layer0.metas.iter().position(|m| m.field_id == EMPTY_FIELD_ID) {
        pos
    } else if let Some(pos) = layer0.metas.iter().position(|m| m.field_id == 0) {
        pos
    } else {
        anyhow::bail!("could not find layer0 empty/reset field block");
    };
    let meta = &layer0.metas[bi];
    let qids = layer0.read_block_chains(bi);
    let file = File::open(values_path).with_context(|| format!("failed to open {}", values_path.display()))?;
    let mut val_bytes = vec![0u8; meta.count as usize * 4];
    read_exact_at_file(&file, &mut val_bytes, meta.chain_offset * 4)?;
    let mut dense = vec![0.0f32; qstate_count];
    for (i, c) in val_bytes.chunks_exact(4).enumerate() {
        dense[qids[i] as usize] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(dense)
}

#[derive(Default, Debug, Clone)]
struct FullMdpLayerStats {
    total_states: u64,
    nonzero: u64,
    terminal_2_edges: u64,
    terminal_4_edges: u64,
    action_count: u64,
    missing_next_terms: u64,
    sum_v: f64,
    max_v: f32,
    max_delta: f32,
    sum_abs_delta: f64,
    dense_cache_hits: u64,
    dense_cache_misses: u64,
    dense_missing_fields: u64,
    dense_evictions: u64,
    dense_materialized_states: u64,
    dense_blocks: u64,
    sparse_blocks: u64,
    bitmap_blocks: u64,
    const_checks: u64,
    const_hits: u64,
    const_misses: u64,
    action_avg_hits: u64,
    action_avg_misses: u64,
    action_avg_collisions: u64,
    action_avg_saved_reveals: u64,
    action_avg_cache_bytes: usize,
}

impl FullMdpLayerStats {
    fn add(&mut self, other: &FullMdpLayerStats) {
        self.total_states += other.total_states;
        self.nonzero += other.nonzero;
        self.terminal_2_edges += other.terminal_2_edges;
        self.terminal_4_edges += other.terminal_4_edges;
        self.action_count += other.action_count;
        self.missing_next_terms += other.missing_next_terms;
        self.sum_v += other.sum_v;
        self.max_v = self.max_v.max(other.max_v);
        self.max_delta = self.max_delta.max(other.max_delta);
        self.sum_abs_delta += other.sum_abs_delta;
        self.dense_cache_hits += other.dense_cache_hits;
        self.dense_cache_misses += other.dense_cache_misses;
        self.dense_missing_fields += other.dense_missing_fields;
        self.dense_evictions += other.dense_evictions;
        self.dense_materialized_states += other.dense_materialized_states;
        self.dense_blocks += other.dense_blocks;
        self.sparse_blocks += other.sparse_blocks;
        self.bitmap_blocks += other.bitmap_blocks;
        self.const_checks += other.const_checks;
        self.const_hits += other.const_hits;
        self.const_misses += other.const_misses;
        self.action_avg_hits += other.action_avg_hits;
        self.action_avg_misses += other.action_avg_misses;
        self.action_avg_collisions += other.action_avg_collisions;
        self.action_avg_saved_reveals += other.action_avg_saved_reveals;
        self.action_avg_cache_bytes = self.action_avg_cache_bytes.max(other.action_avg_cache_bytes);
    }
}

struct FullMdpChunkResult {
    stats: FullMdpLayerStats,
    blocks: usize,
}


#[inline]
fn sum_dense_recs_scaled(
    lookup: &mut NextValueLookup<'_>,
    field2: FieldId,
    recs: &[qgraph::QForwardRec],
    gamma: f32,
    stats: &mut FullMdpLayerStats,
    collect_action_stats: bool,
) -> Result<f32> {
    // All reveals of one action land on the same target field `field2`, so resolve its
    // block exactly once (sum_recs) instead of re-probing the block cache per reveal.
    // gamma is constant across reveals, so it is applied once to the summed value.
    let (s, miss) = lookup.sum_recs(field2, recs)?;
    if collect_action_stats { stats.missing_next_terms += miss; }
    if gamma == 1.0 { Ok(s) } else { Ok(gamma * s) }
}


#[inline]
fn decode_block_qids_from_range(
    cur: &layer_store::FieldChainLayer,
    bi: usize,
    range_chain_base: u64,
    range_qid_bytes: &[u8],
    out: &mut Vec<u32>,
) {
    let meta = &cur.metas[bi];
    out.clear();
    out.reserve(meta.count as usize);
    let local_base = meta.chain_offset - range_chain_base;
    for i in 0..meta.count as u64 {
        out.push(layer_store::read_u24_le_at(range_qid_bytes, local_base + i));
    }
}

#[inline]
fn append_block_output_and_maybe_flush(
    out_file: &File,
    meta: &layer_store::FieldChainBlockMeta,
    out_bytes: &[u8],
    chunk_start_off: &mut Option<u64>,
    chunk_next_off: &mut u64,
    chunk_out_bytes: &mut Vec<u8>,
) -> Result<()> {
    let byte_off = meta.chain_offset * 4;
    if chunk_start_off.is_none() {
        *chunk_start_off = Some(byte_off);
        *chunk_next_off = byte_off;
    }
    if byte_off != *chunk_next_off {
        if !chunk_out_bytes.is_empty() {
            write_all_at_file(out_file, chunk_out_bytes.as_slice(), chunk_start_off.unwrap())?;
            chunk_out_bytes.clear();
        }
        *chunk_start_off = Some(byte_off);
        *chunk_next_off = byte_off;
    }
    chunk_out_bytes.extend_from_slice(out_bytes);
    *chunk_next_off += out_bytes.len() as u64;
    if chunk_out_bytes.len() >= 8 * 1024 * 1024 {
        write_all_at_file(out_file, chunk_out_bytes.as_slice(), chunk_start_off.unwrap())?;
        chunk_out_bytes.clear();
        *chunk_start_off = None;
    }
    Ok(())
}

#[inline]
fn flush_chunk_output(
    out_file: &File,
    chunk_start_off: Option<u64>,
    chunk_out_bytes: &mut Vec<u8>,
) -> Result<()> {
    if let Some(off) = chunk_start_off {
        if !chunk_out_bytes.is_empty() {
            write_all_at_file(out_file, chunk_out_bytes.as_slice(), off)?;
            chunk_out_bytes.clear();
        }
    }
    Ok(())
}

// Same coalescing append as append_block_output_and_maybe_flush, but for a file whose
// per-state element is `elem_bytes` wide (u16 policy files use 2 instead of f32's 4).
#[inline]
fn append_block_output_and_maybe_flush_stride(
    out_file: &File,
    meta: &layer_store::FieldChainBlockMeta,
    out_bytes: &[u8],
    elem_bytes: u64,
    chunk_start_off: &mut Option<u64>,
    chunk_next_off: &mut u64,
    chunk_out_bytes: &mut Vec<u8>,
) -> Result<()> {
    let byte_off = meta.chain_offset * elem_bytes;
    if chunk_start_off.is_none() {
        *chunk_start_off = Some(byte_off);
        *chunk_next_off = byte_off;
    }
    if byte_off != *chunk_next_off {
        if !chunk_out_bytes.is_empty() {
            write_all_at_file(out_file, chunk_out_bytes.as_slice(), chunk_start_off.unwrap())?;
            chunk_out_bytes.clear();
        }
        *chunk_start_off = Some(byte_off);
        *chunk_next_off = byte_off;
    }
    chunk_out_bytes.extend_from_slice(out_bytes);
    *chunk_next_off += out_bytes.len() as u64;
    Ok(())
}

fn compute_full_mdp_chunk_f32(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    prev_cur_values_path: Option<&Path>,
    out_file: &File,
    policy_out_file: Option<&File>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
) -> Result<FullMdpChunkResult> {
    let _ = const_field_fastpath; // const-field fastpath permanently disabled (see const_value)
    let mut lookup_opt = if layer < 9 {
        let nl = next_layer.expect("missing next layer");
        // When the whole next layer is resident in RAM (rolling single-layer buffer), read
        // it lock-free through RamNextValueLookup: no per-task dense-block materialization
        // and no next-layer disk reads. Otherwise fall back to the File backend + LRU cache.
        Some(if let Some(values) = next_values_ram {
            NextValueLookup::Ram(RamNextValueLookup::new(
                nl,
                values,
                per_task_cache_cap,
                collect_cache_stats,
                false,
            )?)
        } else {
            NextValueLookup::File(DenseF32ValueLookup::open(
                nl,
                next_values_path.expect("missing next values path"),
                per_task_cache_cap,
                qstate_count,
                sparse_lookup_max_count,
                collect_cache_stats,
            )?)
        })
    } else { None };

    // prev values are only needed for the per-iteration convergence delta. The single-pass
    // emit path (emit-full-mdp-layers-from-l0) has no previous iterate, so prev is optional;
    // when None we skip the read and report delta=0.
    let prev_cur_file = match prev_cur_values_path {
        Some(p) => Some(File::open(p).with_context(|| format!("failed to open {}", p.display()))?),
        None => None,
    };
    let mut stats = FullMdpLayerStats::default();
    let mut qids_buf = Vec::<u32>::new();
    let mut range_qid_bytes = Vec::<u8>::new();
    let range_chain_base = cur.read_block_range_chains_bytes_into_with_buf(block_range.start, block_range.end, &mut range_qid_bytes);
    let mut prev_bytes = Vec::<u8>::new();
    let mut out_bytes = Vec::<u8>::new();
    let mut chunk_out_bytes = Vec::<u8>::new();
    let mut chunk_start_off: Option<u64> = None;
    let mut chunk_next_off: u64 = 0;
    let record_policy = policy_out_file.is_some();
    let mut policy_bytes = Vec::<u8>::new();
    let mut policy_chunk_bytes = Vec::<u8>::new();
    let mut policy_chunk_start_off: Option<u64> = None;
    let mut policy_chunk_next_off: u64 = 0;

    for bi in block_range.clone() {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        decode_block_qids_from_range(cur, bi, range_chain_base, range_qid_bytes.as_slice(), &mut qids_buf);
        let qids = qids_buf.as_slice();
        let byte_count = meta.count as usize * 4;
        if let Some(pf) = prev_cur_file.as_ref() {
            prev_bytes.resize(byte_count, 0);
            read_exact_at_file(pf, &mut prev_bytes, meta.chain_offset * 4)?;
        }
        out_bytes.clear();
        out_bytes.reserve(byte_count);
        let mut field_edges: [Option<&[FieldId]>; PIECE_COUNT] = [None; PIECE_COUNT];
        let mut active_piece_mask: u8 = 0;
        for p in 0..PIECE_COUNT {
            let e = graph.edges(field, p as u8);
            if !e.is_empty() {
                field_edges[p] = Some(e);
                active_piece_mask |= 1u8 << p;
            }
        }
        if prewarm_field_targets {
            if let Some(lookup) = lookup_opt.as_mut() {
                let mut targets = Vec::<FieldId>::new();
                for p in 0..PIECE_COUNT {
                    if let Some(edges) = field_edges[p] {
                        for &field2 in edges {
                            let terminal = (layer == 4 && field2 == two_line_field) || (layer == 9 && field2 == EMPTY_FIELD_ID);
                            if !terminal { targets.push(field2); }
                        }
                    }
                }
                targets.sort_unstable();
                targets.dedup();
                for field2 in targets { lookup.prewarm(field2)?; }
            }
        }

        if record_policy {
            policy_bytes.clear();
            policy_bytes.reserve(meta.count as usize * 2);
        }
        for (i, &qid) in qids.iter().enumerate() {
            stats.total_states += 1;
            let mut best = 0.0f32;
            // best_code tracks the argmax action alongside best. It is written only when
            // record_policy is set; otherwise the assignment in the rare improve branch is noise.
            let mut best_code = POLICY_NONE_U16;
            for placed in legal_placed[qid as usize] {
                if placed == 255 { continue; }
                if (active_piece_mask & (1u8 << placed)) == 0 { continue; }
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() { continue; }
                let Some(edges) = field_edges[placed as usize] else { continue; };
                let denom = recs.len() as f32;

                if layer == 4 {
                    // 2LPC terminal split: terminal edge is a table lookup; all other targets use next-layer dense values.
                    let lookup = lookup_opt.as_mut().expect("missing next lookup");
                    for (ei, &field2) in edges.iter().enumerate() {
                        if collect_action_stats { stats.action_count += 1; }
                        let cand = if field2 == two_line_field {
                            stats.terminal_2_edges += 1;
                            terminal_reset_avg[qid as usize][placed as usize]
                        } else {
                            sum_dense_recs_scaled(lookup, field2, recs, gamma, &mut stats, collect_action_stats)? / denom
                        };
                        if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                    }
                } else if layer == 9 {
                    // 4LPC terminal split. There is no nonterminal next layer after L9.
                    for (ei, &field2) in edges.iter().enumerate() {
                        if field2 == EMPTY_FIELD_ID {
                            if collect_action_stats { stats.action_count += 1; }
                            stats.terminal_4_edges += 1;
                            let cand = terminal_reset_avg[qid as usize][placed as usize];
                            if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                        }
                    }
                } else {
                    // Terminal-free fast path for all other layers: no terminal branch in the hot field2 loop.
                    let lookup = lookup_opt.as_mut().expect("missing next lookup");
                    for (ei, &field2) in edges.iter().enumerate() {
                        if collect_action_stats { stats.action_count += 1; }
                        let cand = sum_dense_recs_scaled(lookup, field2, recs, gamma, &mut stats, collect_action_stats)? / denom;
                        if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                    }
                }
            }
            let delta = if prev_cur_file.is_some() {
                let po = i * 4;
                let old = f32::from_le_bytes([prev_bytes[po], prev_bytes[po + 1], prev_bytes[po + 2], prev_bytes[po + 3]]);
                (best - old).abs()
            } else { 0.0 };
            if best != 0.0 { stats.nonzero += 1; }
            stats.max_v = stats.max_v.max(best);
            stats.sum_v += best as f64;
            stats.max_delta = stats.max_delta.max(delta);
            stats.sum_abs_delta += delta as f64;
            write_f32_le_buf(&mut out_bytes, best);
            if record_policy { policy_bytes.extend_from_slice(&best_code.to_le_bytes()); }
        }
        append_block_output_and_maybe_flush(out_file, meta, out_bytes.as_slice(), &mut chunk_start_off, &mut chunk_next_off, &mut chunk_out_bytes)?;
        if let Some(pf) = policy_out_file {
            append_block_output_and_maybe_flush_stride(pf, meta, policy_bytes.as_slice(), 2, &mut policy_chunk_start_off, &mut policy_chunk_next_off, &mut policy_chunk_bytes)?;
        }
    }
    flush_chunk_output(out_file, chunk_start_off, &mut chunk_out_bytes)?;
    if let Some(pf) = policy_out_file {
        flush_chunk_output(pf, policy_chunk_start_off, &mut policy_chunk_bytes)?;
    }
    if collect_cache_stats {
        if let Some(lookup) = lookup_opt.as_ref() {
            let ds = lookup.stats_snapshot();
            stats.dense_cache_hits += ds.cache_hits;
            stats.dense_cache_misses += ds.cache_misses;
            stats.dense_missing_fields += ds.missing_fields;
            stats.dense_evictions += ds.evictions;
            stats.dense_materialized_states += ds.materialized_states;
            stats.dense_blocks += ds.dense_blocks;
            stats.sparse_blocks += ds.sparse_blocks;
            stats.bitmap_blocks += ds.bitmap_blocks;
            stats.const_checks += ds.const_checks;
            stats.const_hits += ds.const_hits;
            stats.const_misses += ds.const_misses;
        }
    }
    Ok(FullMdpChunkResult { stats, blocks: block_range.end - block_range.start })
}

// ---------------------------------------------------------------------------
fn compute_full_mdp_range_group_f32_rolling(
    layer: usize,
    block_ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    out_file: &File,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    old_l0_dense: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
) -> Result<FullMdpChunkResult> {
    compute_full_mdp_range_group_f32_rolling_sinks(
        layer, block_ranges, cur, Some(out_file), None, None,
        graph, qfwd, legal_placed, terminal_reset_avg, next_layer,
        next_values_path, next_values_ram, old_l0_dense, gamma,
        per_task_cache_cap, qstate_count, sparse_lookup_max_count,
        two_line_field, prewarm_field_targets, const_field_fastpath,
        collect_cache_stats, collect_action_stats,
    )
}

// One lookup cache survives all adjacent ranges in a scheduled group. RAM output
// avoids a temporary file write/read cycle; the policy sink records the same strict
// argmax and action order as the full materialization kernel.
fn compute_full_mdp_range_group_f32_rolling_sinks(
    layer: usize,
    block_ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    policy_out_file: Option<&File>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    old_l0_dense: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
) -> Result<FullMdpChunkResult> {
    compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source(
        layer, block_ranges, cur, out_file, ram_out, policy_out_file, graph, qfwd, legal_placed, terminal_reset_avg, next_layer, next_values_path, next_values_ram, old_l0_dense, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, prewarm_field_targets, const_field_fastpath, collect_cache_stats, collect_action_stats, None,
    )
}

fn compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source(
    layer: usize,
    block_ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    policy_out_file: Option<&File>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    old_l0_dense: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
) -> Result<FullMdpChunkResult> {
    compute_full_mdp_range_group_f32_rolling_sinks_with_action_cache(
        layer, block_ranges, cur, out_file, ram_out, policy_out_file, graph, qfwd, legal_placed, terminal_reset_avg, next_layer, next_values_path, next_values_ram, old_l0_dense, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, prewarm_field_targets, const_field_fastpath, collect_cache_stats, collect_action_stats, direct_l9_source, None,
    )
}

fn compute_full_mdp_range_group_f32_rolling_sinks_with_action_cache(
    layer: usize,
    block_ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    policy_out_file: Option<&File>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    old_l0_dense: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
    action_source: Option<action_avg_cache::ActionSource<'_>>,
) -> Result<FullMdpChunkResult> {
    if action_source.is_some() {
        compute_greedy_group_with_action_cache_impl::<true>(
            layer, block_ranges, cur, out_file, ram_out, policy_out_file, graph, qfwd, legal_placed, terminal_reset_avg, next_layer, next_values_path, next_values_ram, old_l0_dense, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, prewarm_field_targets, const_field_fastpath, collect_cache_stats, collect_action_stats, direct_l9_source, action_source,
        )
    } else {
        compute_greedy_group_with_action_cache_impl::<false>(
            layer, block_ranges, cur, out_file, ram_out, policy_out_file, graph, qfwd, legal_placed, terminal_reset_avg, next_layer, next_values_path, next_values_ram, old_l0_dense, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, prewarm_field_targets, const_field_fastpath, collect_cache_stats, collect_action_stats, direct_l9_source, None,
        )
    }
}

// The const-generic false specialization keeps the default greedy hot path
// free of action-cache branches. A cache belongs to one group and one immutable
// next-value source, so gamma and denominator are part of its stored final score.
#[inline]
fn greedy_action_average<const USE_ACTION_CACHE: bool>(
    action_cache: &mut Option<action_avg_cache::ActionAverageCache>,
    zid: u32,
    lookup: &mut NextValueLookup<'_>,
    field2: FieldId,
    recs: &[qgraph::QForwardRec],
    gamma: f32,
    denom: f32,
    stats: &mut FullMdpLayerStats,
    collect_action_stats: bool,
) -> Result<f32> {
    if USE_ACTION_CACHE {
        if let Some(cache) = action_cache.as_mut() {
            if let Some(cached) = cache.get(field2, zid, recs.len()) {
                if collect_action_stats { stats.missing_next_terms += u64::from(cached.missing_terms); }
                return Ok(cached.value);
            }
            let (sum, missing) = lookup.sum_recs(field2, recs)?;
            if collect_action_stats { stats.missing_next_terms += missing; }
            let scaled = if gamma == 1.0 { sum } else { gamma * sum };
            let value = scaled / denom;
            cache.insert(field2, zid, action_avg_cache::CachedAverage {
                value,
                missing_terms: u8::try_from(missing).context("greedy action missing-term count exceeds u8")?,
            });
            return Ok(value);
        }
    }
    Ok(sum_dense_recs_scaled(lookup, field2, recs, gamma, stats, collect_action_stats)? / denom)
}

fn compute_greedy_group_with_action_cache_impl<const USE_ACTION_CACHE: bool>(
    layer: usize,
    block_ranges: &[std::ops::Range<usize>],
    cur: &layer_store::FieldChainLayer,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    policy_out_file: Option<&File>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    old_l0_dense: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    prewarm_field_targets: bool,
    const_field_fastpath: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
    action_source: Option<action_avg_cache::ActionSource<'_>>,
) -> Result<FullMdpChunkResult> {
    anyhow::ensure!(out_file.is_some() || ram_out.is_some() || (layer == 9 && policy_out_file.is_some()), "greedy sweep needs a value sink or an L9 policy sink");
    if USE_ACTION_CACHE {
        if let Some(source) = action_source { source.placements.check_graph(qfwd, qstate_count)?; }
    }
    let (mut action_cache, lookup_cache_cap) = if USE_ACTION_CACHE && layer < 9 {
        match action_source {
            Some(source) => action_avg_cache::ActionAverageCache::partition(
                source.requested_bytes, per_task_cache_cap, qstate_count,
            ),
            None => (None, per_task_cache_cap),
        }
    } else { (None, per_task_cache_cap) };
    // The legacy const-field shortcut is unsound for pruned reveal children.
    let _ = const_field_fastpath;
    let mut lookup_opt = if layer < 9 {
        let nl = next_layer.expect("missing next layer");
        Some(if let Some(source) = direct_l9_source.filter(|_| layer == 8) {
            // This bounded membership/policy cache consumes the planner's
            // per-task cache allowance; no L9 value layer is pinned.
            NextValueLookup::TerminalL9(terminal_l9::Lookup::new(source, lookup_cache_cap, collect_cache_stats)?)
        } else if let Some(values) = next_values_ram {
            NextValueLookup::Ram(RamNextValueLookup::new(
                nl,
                values,
                lookup_cache_cap,
                collect_cache_stats,
                false,
            )?)
        } else {
            NextValueLookup::File(DenseF32ValueLookup::open(
                nl,
                next_values_path.expect("missing next values path"),
                lookup_cache_cap,
                qstate_count,
                sparse_lookup_max_count,
                collect_cache_stats,
            )?)
        })
    } else { None };

    let mut stats = FullMdpLayerStats::default();
    let mut total_blocks = 0usize;
    let mut qids_buf = Vec::<u32>::new();
    let mut range_qid_bytes = Vec::<u8>::new();
    let mut out_bytes = Vec::<u8>::new();
    let mut chunk_out_bytes = Vec::<u8>::new();
    let mut chunk_start_off: Option<u64> = None;
    let mut chunk_next_off: u64 = 0;
    let mut policy_bytes = Vec::<u8>::new();
    let mut policy_chunk_bytes = Vec::<u8>::new();
    let mut policy_chunk_start_off: Option<u64> = None;
    let mut policy_chunk_next_off: u64 = 0;
    const OUTPUT_BUFFER_BYTES: usize = 8 * 1024 * 1024;

    // Read the current-layer qid chains once for the whole lookup group, not once
    // per block_chunk. Ranges are created from adjacent chunks, so the group is
    // contiguous. This is the MDP analogue of the backward generator's large
    // sequential shard read and removes another factor of lookup_group_chunks
    // from current-chain read syscalls.
    let group_start = block_ranges.first().map(|r| r.start).unwrap_or(0);
    let group_end = block_ranges.last().map(|r| r.end).unwrap_or(group_start);
    let group_chain_base = if group_start < group_end {
        cur.read_block_range_chains_bytes_into_with_buf(group_start, group_end, &mut range_qid_bytes)
    } else {
        0
    };

    for block_range in block_ranges.iter().cloned() {
        total_blocks += block_range.end.saturating_sub(block_range.start);
        for bi in block_range.clone() {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        decode_block_qids_from_range(cur, bi, group_chain_base, range_qid_bytes.as_slice(), &mut qids_buf);
        let qids = qids_buf.as_slice();
        out_bytes.clear();
        if out_file.is_some() { out_bytes.reserve(meta.count as usize * 4); }
        policy_bytes.clear();
        if policy_out_file.is_some() { policy_bytes.reserve(meta.count as usize * 2); }
        if let Some(ro) = ram_out {
            let end = meta.chain_offset.checked_add(meta.count as u64).context("RAM output offset overflow")?;
            anyhow::ensure!(end <= ro.len as u64, "greedy RAM output overflow: block end {} > len {}", end, ro.len);
        }
        let mut field_edges: [Option<&[FieldId]>; PIECE_COUNT] = [None; PIECE_COUNT];
        let mut active_piece_mask: u8 = 0;
        for p in 0..PIECE_COUNT {
            let e = graph.edges(field, p as u8);
            if !e.is_empty() {
                field_edges[p] = Some(e);
                active_piece_mask |= 1u8 << p;
            }
        }
        if prewarm_field_targets {
            if let Some(lookup) = lookup_opt.as_mut() {
                let mut targets = Vec::<FieldId>::new();
                for p in 0..PIECE_COUNT {
                    if let Some(edges) = field_edges[p] {
                        for &field2 in edges {
                            let terminal = (layer == 4 && field2 == two_line_field) || (layer == 9 && field2 == EMPTY_FIELD_ID);
                            if !terminal { targets.push(field2); }
                        }
                    }
                }
                targets.sort_unstable();
                targets.dedup();
                for field2 in targets { lookup.prewarm(field2)?; }
            }
        }

        for (i, &qid) in qids.iter().enumerate() {
            stats.total_states += 1;
            let mut best = 0.0f32;
            let mut best_code = POLICY_NONE_U16;
            for placed in legal_placed[qid as usize] {
                if placed == 255 { continue; }
                if (active_piece_mask & (1u8 << placed)) == 0 { continue; }
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() { continue; }
                let Some(edges) = field_edges[placed as usize] else { continue; };
                let denom = recs.len() as f32;
                let action_zid = if USE_ACTION_CACHE && action_cache.is_some() {
                    action_source.expect("greedy action cache needs a placement plan")
                        .placements.zid_for_placement(qid, placed)
                        .context("greedy action has no validated pre-reveal ID")?
                } else { 0 };

                if layer == 4 {
                    let lookup = lookup_opt.as_mut().expect("missing next lookup");
                    for (ei, &field2) in edges.iter().enumerate() {
                        if collect_action_stats { stats.action_count += 1; }
                        let cand = if field2 == two_line_field {
                            stats.terminal_2_edges += 1;
                            terminal_reset_avg[qid as usize][placed as usize]
                        } else {
                            greedy_action_average::<USE_ACTION_CACHE>(&mut action_cache, action_zid, lookup, field2, recs, gamma, denom, &mut stats, collect_action_stats)?
                        };
                        if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                    }
                } else if layer == 9 {
                    for (ei, &field2) in edges.iter().enumerate() {
                        if field2 == EMPTY_FIELD_ID {
                            if collect_action_stats { stats.action_count += 1; }
                            stats.terminal_4_edges += 1;
                            let cand = terminal_reset_avg[qid as usize][placed as usize];
                            if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                        }
                    }
                } else {
                    let lookup = lookup_opt.as_mut().expect("missing next lookup");
                    for (ei, &field2) in edges.iter().enumerate() {
                        if collect_action_stats { stats.action_count += 1; }
                        let cand = greedy_action_average::<USE_ACTION_CACHE>(&mut action_cache, action_zid, lookup, field2, recs, gamma, denom, &mut stats, collect_action_stats)?;
                        if cand > best { best = cand; best_code = encode_policy_u16(placed, ei); }
                    }
                }
            }
            if best != 0.0 { stats.nonzero += 1; }
            stats.max_v = stats.max_v.max(best);
            stats.sum_v += best as f64;
            if let Some(old_l0) = old_l0_dense {
                let old = old_l0[qid as usize];
                let delta = (best - old).abs();
                stats.max_delta = stats.max_delta.max(delta);
                stats.sum_abs_delta += delta as f64;
            }
            if let Some(ro) = ram_out {
                // SAFETY: the block bounds were checked above, and scheduled block
                // ranges are disjoint for the lifetime of all Rayon tasks.
                unsafe { *ro.ptr.add(meta.chain_offset as usize + i) = best; }
            }
            if out_file.is_some() { write_f32_le_buf(&mut out_bytes, best); }
            if policy_out_file.is_some() { policy_bytes.extend_from_slice(&best_code.to_le_bytes()); }
        }
            if let Some(of) = out_file {
                append_block_output_and_maybe_flush(of, meta, out_bytes.as_slice(), &mut chunk_start_off, &mut chunk_next_off, &mut chunk_out_bytes)?;
                if chunk_out_bytes.len() >= OUTPUT_BUFFER_BYTES {
                    flush_chunk_output(of, chunk_start_off, &mut chunk_out_bytes)?;
                    chunk_start_off = None;
                }
            }
            if let Some(pf) = policy_out_file {
                append_block_output_and_maybe_flush_stride(pf, meta, policy_bytes.as_slice(), 2, &mut policy_chunk_start_off, &mut policy_chunk_next_off, &mut policy_chunk_bytes)?;
                if policy_chunk_bytes.len() >= OUTPUT_BUFFER_BYTES {
                    flush_chunk_output(pf, policy_chunk_start_off, &mut policy_chunk_bytes)?;
                    policy_chunk_start_off = None;
                }
            }
        }
    }
    if let Some(of) = out_file { flush_chunk_output(of, chunk_start_off, &mut chunk_out_bytes)?; }
    if let Some(pf) = policy_out_file { flush_chunk_output(pf, policy_chunk_start_off, &mut policy_chunk_bytes)?; }
    if let Some(cache) = action_cache {
        let counters = cache.stats();
        stats.action_avg_hits = counters.hits;
        stats.action_avg_misses = counters.misses;
        stats.action_avg_collisions = counters.collisions;
        stats.action_avg_saved_reveals = counters.saved_reveals;
        stats.action_avg_cache_bytes = counters.payload_bytes;
    }
    if collect_cache_stats {
        if let Some(lookup) = lookup_opt.as_ref() {
            let ds = lookup.stats_snapshot();
            stats.dense_cache_hits += ds.cache_hits;
            stats.dense_cache_misses += ds.cache_misses;
            stats.dense_missing_fields += ds.missing_fields;
            stats.dense_evictions += ds.evictions;
            stats.dense_materialized_states += ds.materialized_states;
            stats.dense_blocks += ds.dense_blocks;
            stats.sparse_blocks += ds.sparse_blocks;
            stats.bitmap_blocks += ds.bitmap_blocks;
            stats.const_checks += ds.const_checks;
            stats.const_hits += ds.const_hits;
            stats.const_misses += ds.const_misses;
        }
    }
    Ok(FullMdpChunkResult { stats, blocks: total_blocks })
}


fn build_qfwd_legal_placed_table(qfwd: &QForwardGraph, qstate_count: usize) -> Vec<[u8; 2]> {
    (0..qstate_count as u32)
        .map(|qid| qfwd_legal_placed_pieces(qid, qfwd))
        .collect()
}


fn print_mdp_target_reuse_stats(layers: &[layer_store::FieldChainLayer], graph: &FieldGraph, two_line_field: FieldId) -> Result<()> {
    let mut counts = vec![0u32; NUM_FIELDS];
    let mut touched = Vec::<FieldId>::new();
    for layer in 0usize..=9 {
        let mut total_edges = 0u64;
        let mut terminal_edges = 0u64;
        for meta in &layers[layer].metas {
            let field = meta.field_id;
            for p in 0..PIECE_COUNT {
                for &field2 in graph.edges(field, p as u8) {
                    total_edges += 1;
                    let terminal = (layer == 4 && field2 == two_line_field) || (layer == 9 && field2 == EMPTY_FIELD_ID);
                    if terminal {
                        terminal_edges += 1;
                        continue;
                    }
                    let idx = field2 as usize;
                    if counts[idx] == 0 { touched.push(field2); }
                    counts[idx] = counts[idx].saturating_add(1);
                }
            }
        }
        let mut reuse = Vec::<u32>::with_capacity(touched.len());
        for &f in &touched { reuse.push(counts[f as usize]); counts[f as usize] = 0; }
        touched.clear();
        reuse.sort_unstable();
        let target_fields = reuse.len();
        let nonterminal_edges: u64 = reuse.iter().map(|&x| x as u64).sum();
        let q = |pct: usize| -> u32 {
            if reuse.is_empty() { return 0; }
            let idx = ((reuse.len() - 1) * pct) / 100;
            reuse[idx]
        };
        let max_reuse = reuse.last().copied().unwrap_or(0);
        let avg = if target_fields == 0 { 0.0 } else { nonterminal_edges as f64 / target_fields as f64 };
        eprintln!(
            "mdp-target-reuse layer={} current_fields={} total_field_edges={} nonterminal_field_edges={} terminal_field_edges={} target_fields={} avg_parents_per_target={:.3} p50={} p90={} p99={} max={}",
            layer,
            layers[layer].metas.len(),
            total_edges,
            nonterminal_edges,
            terminal_edges,
            target_fields,
            avg,
            q(50),
            q(90),
            q(99),
            max_reuse
        );
    }
    Ok(())
}

fn build_terminal_reset_avg_table(qfwd: &QForwardGraph, reset_values: &[f32], gamma: f32, qstate_count: usize) -> Vec<[f32; 7]> {
    build_terminal_reset_avg_table_opt(qfwd, reset_values, gamma, qstate_count, true)
}

// include_reward=true  -> E[1 + gamma*x(child)]  (the affine map T_pi's terminal contribution)
// include_reward=false -> E[gamma*x(child)]       (the LINEAR map M_pi's terminal contribution;
//   the +1 reward is the constant part c and is dropped, so a fixed-policy sweep with this table
//   computes exactly gamma*M_pi*x). Used by the matrix-free GMRES matvec to avoid catastrophic
//   cancellation: gamma*M*q has magnitude ~||q|| (tiny for a normalized Krylov vector) instead of
//   being extracted as the difference of two ~3-magnitude affine sweeps.
fn build_terminal_reset_avg_table_opt(qfwd: &QForwardGraph, reset_values: &[f32], gamma: f32, qstate_count: usize, include_reward: bool) -> Vec<[f32; 7]> {
    let mut out = vec![[0.0f32; 7]; qstate_count];
    let gamma_is_one = gamma == 1.0;
    let reward: f32 = if include_reward { 1.0 } else { 0.0 };
    for qid in 0..qstate_count as u32 {
        for p in 0u8..PIECE_COUNT as u8 {
            let recs = qfwd.records_for(qid, p);
            if recs.is_empty() { continue; }
            let mut s = 0.0f32;
            if gamma_is_one {
                for rec in recs { s += reward + reset_values[rec.child_qid as usize]; }
            } else {
                for rec in recs { s += reward + gamma * reset_values[rec.child_qid as usize]; }
            }
            out[qid as usize][p as usize] = s / (recs.len() as f32);
        }
    }
    out
}

#[inline]
fn qfwd_legal_placed_pieces(qid: u32, qfwd: &QForwardGraph) -> [u8; 2] {
    // Avoid carrying QStateIndex into the hot loop. A placed piece is legal iff records_for(qid,p) is nonempty.
    // The caller still iterates at most two pieces by scanning all seven and returning first two nonempty entries.
    let mut out = [255u8, 255u8];
    let mut n = 0usize;
    for p in 0u8..PIECE_COUNT as u8 {
        if !qfwd.records_for(qid, p).is_empty() {
            out[n] = p;
            n += 1;
            if n == 2 { break; }
        }
    }
    out
}

fn initialize_full_mdp_values(pruned_dir: &Path, out_a: &Path, init_spec: &str) -> Result<()> {
    fs::create_dir_all(out_a)?;
    let one_loop_prefix = init_spec.strip_prefix("one_loop:");
    for layer in 0usize..=9 {
        let pr = q_pruned_prefix(pruned_dir, layer);
        let layer_meta = layer_store::FieldChainLayer::open_index_only(&pr)?;
        let dst = mdp_f32_value_path(out_a, layer);
        if let Some(dir) = one_loop_prefix {
            let src = mdp_value_path(Path::new(dir), layer);
            if src.exists() {
                eprintln!("full-mdp init layer={} from one-loop {}", layer, src.display());
                convert_one_loop_u16_to_f32(&src, &dst, layer_meta.total_states, 5040.0)?;
            } else if layer >= 4 {
                eprintln!("full-mdp init layer={} one-loop missing; using full 1.0 base", layer);
                create_full_f32_file(&dst, layer_meta.total_states, 1.0)?;
            } else {
                eprintln!("full-mdp init layer={} one-loop missing; using zero", layer);
                create_zero_f32_file(&dst, layer_meta.total_states)?;
            }
        } else {
            create_zero_f32_file(&dst, layer_meta.total_states)?;
        }
    }
    Ok(())
}


fn write_layer0_dense_values(layer0: &layer_store::FieldChainLayer, path: &Path, dense: &[f32]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?);
    let mut buf = Vec::<u8>::new();
    for bi in 0..layer0.metas.len() {
        let meta = &layer0.metas[bi];
        let qids = layer0.read_block_chains(bi);
        if qids.len() != meta.count as usize {
            anyhow::bail!("layer0 block qid/count mismatch block={} qids={} count={}", bi, qids.len(), meta.count);
        }
        buf.clear();
        buf.reserve(qids.len() * 4);
        for qid in qids {
            let idx = qid as usize;
            if idx >= dense.len() { anyhow::bail!("qid {} out of dense len {}", qid, dense.len()); }
            write_f32_le_buf(&mut buf, dense[idx]);
        }
        out.write_all(&buf)?;
    }
    out.flush()?;
    let actual = fs::metadata(path)?.len();
    let expected = layer0.total_states.checked_mul(4).context("layer0 dense write size overflow")?;
    if actual != expected {
        anyhow::bail!("layer0 dense write size mismatch {}: got {}, expected {}", path.display(), actual, expected);
    }
    Ok(())
}

fn copy_checked_file(src: &Path, dst: &Path, expected_bytes: u64) -> Result<()> {
    let actual = fs::metadata(src)
        .with_context(|| format!("failed to stat {}", src.display()))?
        .len();
    if actual != expected_bytes {
        anyhow::bail!("copy source size mismatch {}: got {}, expected {}", src.display(), actual, expected_bytes);
    }
    fs::copy(src, dst)
        .with_context(|| format!("failed to copy {} -> {}", src.display(), dst.display()))?;
    let got = fs::metadata(dst)?.len();
    if got != expected_bytes {
        anyhow::bail!("copy destination size mismatch {}: got {}, expected {}", dst.display(), got, expected_bytes);
    }
    Ok(())
}

fn initialize_l0_mdp_values(pruned_dir: &Path, out_dir: &Path, init_spec: &str) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let layer0 = layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(pruned_dir, 0))?;
    let expected = layer0.total_states.checked_mul(4).context("layer0 init expected size overflow")?;
    let exact_path = mdp_f32_value_path(out_dir, 0);
    let input_path = mdp_l0_input_value_path(out_dir);

    if input_path.exists() {
        let actual = fs::metadata(&input_path)?.len();
        if actual != expected {
            anyhow::bail!("existing layer0 input size mismatch {}: got {}, expected {}", input_path.display(), actual, expected);
        }
        eprintln!("solve-full-mdp-l0-f32 reusing existing input {}", input_path.display());
        if !exact_path.exists() { fs::copy(&input_path, &exact_path)?; }
        return Ok(());
    }
    if exact_path.exists() {
        let actual = fs::metadata(&exact_path)?.len();
        if actual != expected {
            anyhow::bail!("existing layer0 value size mismatch {}: got {}, expected {}", exact_path.display(), actual, expected);
        }
        eprintln!("solve-full-mdp-l0-f32 initializing input from existing {}", exact_path.display());
        fs::copy(&exact_path, &input_path)?;
        return Ok(());
    }

    if let Some(dir) = init_spec.strip_prefix("one_loop:") {
        let src = mdp_value_path(Path::new(dir), 0);
        eprintln!("solve-full-mdp-l0-f32 init layer0 from one-loop {}", src.display());
        convert_one_loop_u16_to_f32(&src, &exact_path, layer0.total_states, 5040.0)?;
    } else if let Some(file) = init_spec.strip_prefix("l0:") {
        let src = Path::new(file);
        eprintln!("solve-full-mdp-l0-f32 init layer0 from l0 file {}", src.display());
        copy_checked_file(src, &exact_path, expected)?;
    } else if let Some(dir) = init_spec.strip_prefix("full:") {
        let src = mdp_f32_value_path(Path::new(dir), 0);
        eprintln!("solve-full-mdp-l0-f32 init layer0 from full dir {}", src.display());
        copy_checked_file(&src, &exact_path, expected)?;
    } else if init_spec == "zero" {
        eprintln!("solve-full-mdp-l0-f32 init layer0 zero");
        create_zero_f32_file(&exact_path, layer0.total_states)?;
    } else {
        anyhow::bail!("unsupported l0 init spec: {}", init_spec);
    }
    copy_checked_file(&exact_path, &input_path, expected)?;
    Ok(())
}

#[derive(Clone)]
struct AndersonEntry {
    f: Vec<f32>,
    r: Vec<f32>,
}

fn dot_f32_as_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum()
}

fn solve_dense_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best { best = v; pivot = row; }
        }
        if best < 1.0e-30 || !best.is_finite() { return None; }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let diag = a[col][col];
        for j in col..n { a[col][j] /= diag; }
        b[col] /= diag;
        for row in 0..n {
            if row == col { continue; }
            let factor = a[row][col];
            if factor == 0.0 { continue; }
            for j in col..n { a[row][j] -= factor * a[col][j]; }
            b[row] -= factor * b[col];
        }
    }
    if b.iter().all(|x| x.is_finite()) { Some(b) } else { None }
}

// Type-II Anderson candidate with mixing `beta` and Tikhonov `reg_scale`.
// The fixed-point map is g(x)=T(x) (plain VI does x<-T(x)), residual r=g(x)-x. AA solves
// min_{sum a=1} || sum a_i r_i ||_2 then sets x_new = sum a_i ( g(x_i) - (1-beta) r_i ).
// beta=1 -> x_new = sum a_i g(x_i) (undamped, the original behavior). beta<1 damps the
// step toward the relaxed iterate, which stabilizes AA when gamma=1 makes the local
// linearization near-singular. reg_scale grows the diagonal load to keep the normal-
// equations well conditioned as the residual history becomes collinear.
fn anderson_l0_candidate(hist: &VecDeque<AndersonEntry>, beta: f32, reg_scale: f64) -> Option<Vec<f32>> {
    let k = hist.len();
    if k < 2 { return None; }
    let n = k + 1;
    let mut mat = vec![vec![0.0f64; n]; n];
    let mut rhs = vec![0.0f64; n];
    let mut trace = 0.0f64;
    for i in 0..k {
        for j in 0..k {
            let v = dot_f32_as_f64(&hist[i].r, &hist[j].r);
            mat[i][j] = v;
            if i == j { trace += v.abs(); }
        }
        mat[i][k] = 1.0;
        mat[k][i] = 1.0;
    }
    let reg = (trace / (k as f64)).max(1.0) * reg_scale;
    for i in 0..k { mat[i][i] += reg; }
    rhs[k] = 1.0;
    let sol = solve_dense_linear(mat, rhs)?;
    let alpha = &sol[..k];
    let one_minus_beta = 1.0f32 - beta;
    let len = hist[0].f.len();
    let mut out = vec![0.0f32; len];
    for (entry, &a) in hist.iter().zip(alpha.iter()) {
        let af = a as f32;
        if one_minus_beta == 0.0 {
            for i in 0..len { out[i] += af * entry.f[i]; }
        } else {
            for i in 0..len { out[i] += af * (entry.f[i] - one_minus_beta * entry.r[i]); }
        }
    }
    for v in &mut out {
        if !v.is_finite() { return None; }
        if *v < 0.0 { *v = 0.0; }
    }
    Some(out)
}

#[cfg(test)]
mod seed_sweep_tests;

fn run_solve_full_mdp_l0_f32(args: &[String]) -> Result<()> {
    if args.len() < 5 || args.len() > 11 {
        eprintln!("usage: {} solve-full-mdp-l0-f32 <graph.bin> <pruned_layer_dir> <out_dir> [gamma=1.0] [max_iters=50] [dense_cache_gib=32] [init=zero|one_loop:<dir>|l0:<file>|full:<dir>] [tol_abs=1e-5] [tol_mean=0]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    let gamma: f32 = if args.len() >= 6 { args[5].parse().context("invalid gamma")? } else { 1.0 };
    let max_iters: usize = if args.len() >= 7 { args[6].parse().context("invalid max_iters")? } else { 50 };
    let cache_gib: usize = if args.len() >= 8 { args[7].parse().context("invalid dense_cache_gib")? } else { 32 };
    let init_spec = if args.len() >= 9 { args[8].as_str() } else { "zero" };
    let tol_abs: f32 = if args.len() >= 10 { args[9].parse().context("invalid tol_abs")? } else { 1.0e-5 };
    let tol_mean: f64 = if args.len() >= 11 { args[10].parse().context("invalid tol_mean")? } else { 0.0 };

    let cache_cap_bytes = cache_gib.saturating_mul(1024usize).saturating_mul(1024usize).saturating_mul(1024usize);
    let tuning = sweep_schedule::SweepTuning::from_env(cache_cap_bytes)?;

    fs::create_dir_all(&out_dir)?;
    eprintln!("solve-full-mdp-l0-f32 loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    eprintln!("solve-full-mdp-l0-f32 building q index and qforward graph...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {} != {}", qstate_count, CHECK_QSTATE_COUNT); }
    let qfwd = build_qforward_graph(&q_index);
    let legal_placed = build_qfwd_legal_placed_table(&qfwd, qstate_count);
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).unwrap_or(11335);
    initialize_l0_mdp_values(&pruned_dir, &out_dir, init_spec)?;

    let sync_layer_outputs: bool = env::var("FULL_MDP_SYNC_LAYER")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let prewarm_field_targets: bool = env::var("FULL_MDP_PREWARM_FIELD_TARGETS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let collect_cache_stats: bool = env::var("FULL_MDP_CACHE_STATS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let collect_action_stats: bool = env::var("FULL_MDP_ACTION_STATS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    // Anderson depth ~ GMRES(m) on the 817,740-dim boundary map. Cap raised so deeper
    // history (more Krylov-like) can capture the slow modes when spectral radius -> 1.
    // Each stored iterate is 817,740 f32 (~3.2 MiB), so even depth 64 is ~200 MiB.
    let aa_depth: usize = env::var("FULL_MDP_ANDERSON_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(0usize).min(64);
    let aa_start: usize = env::var("FULL_MDP_ANDERSON_START").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize);
    // Safeguarding (do the acceleration robustly): AA on the nonlinear Bellman max-operator
    // can overshoot at gamma=1. If the fixed-point residual ||T(x)-x||_2 grows after an AA
    // step, the extrapolation was harmful -> clear the history and fall back to plain VI for
    // this step (restart). beta damps every AA step; reg_scale loads the LS diagonal.
    let aa_restart: bool = env::var("FULL_MDP_ANDERSON_RESTART").ok().map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no").unwrap_or(true);
    let aa_restart_tol: f64 = env::var("FULL_MDP_ANDERSON_RESTART_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0e-3);
    let aa_beta: f32 = env::var("FULL_MDP_ANDERSON_BETA").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0f32).clamp(0.05, 1.0);
    let aa_reg_scale: f64 = env::var("FULL_MDP_ANDERSON_REG").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0e-12);
    let batch_chunks_env: Option<usize> = env::var("FULL_MDP_BATCH_CHUNKS").ok().and_then(|s| s.parse().ok());
    let sparse_lookup_max_count: usize = env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536);
    let ram_next_layer: bool = env::var("FULL_MDP_RAM_NEXT_LAYER")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let ram_next_layer_max_bytes: u64 = env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(96)
        .saturating_mul(1024 * 1024 * 1024);

    let progress_every_batches: usize = env::var("FULL_MDP_PROGRESS_EVERY_BATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        // Massive external-memory sweeps can emit tens of thousands of progress
        // lines per layer. Throttle by default; set to 1 for the old behavior.
        .unwrap_or(20usize)
        .max(1);

    // FULL_MDP_CONST_FIELD_FASTPATH is DISABLED: the const-field fast path is unsound
    // (over-counts pruned/absent reveal children; see RamNextValueLookup::const_value).
    // We still read the env var so an explicit opt-in produces a loud warning instead of
    // silently-wrong values, but the effective value is forced to false.
    let const_field_fastpath_requested: bool = env::var("FULL_MDP_CONST_FIELD_FASTPATH")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    if const_field_fastpath_requested {
        eprintln!("solve-full-mdp-l0-f32 WARNING: FULL_MDP_CONST_FIELD_FASTPATH was requested but is permanently disabled (known correctness bug: pruned reveal children over-counted). Ignoring; using the correct per-reveal path.");
    }
    let const_field_fastpath: bool = false;
    let target_reuse_stats: bool = env::var("FULL_MDP_TARGET_REUSE_STATS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    eprintln!("solve-full-mdp-l0-f32 mode=layer0_fixed_point rolling_layers=true terminal_reset_source=input_layer0 output=layer0_only gamma={} max_iters={} lookup_cache_gib={} sparse_lookup_max_count={} ram_next_layer={} max_pin_gib={} progress_every_batches={} const_field_fastpath={} target_reuse_stats={} tol_abs={} tol_mean={} anderson_depth={} anderson_start={} sync_layer_outputs={} prewarm={} cache_stats={} action_stats={}",
        gamma, max_iters, cache_gib, sparse_lookup_max_count, ram_next_layer, ram_next_layer_max_bytes / (1024 * 1024 * 1024), progress_every_batches, const_field_fastpath, target_reuse_stats, tol_abs, tol_mean, aa_depth, aa_start, sync_layer_outputs, prewarm_field_targets, collect_cache_stats, collect_action_stats);

    let input_path = mdp_l0_input_value_path(&out_dir);
    let exact_path = mdp_f32_value_path(&out_dir, 0);
    let last_f_path = mdp_l0_last_f_value_path(&out_dir);
    let next_l0_path = mdp_l0_next_value_path(&out_dir);
    let work_dir = env::var("FULL_MDP_WORK_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join("rolling_work"));
    if work_dir.as_os_str().is_empty() || work_dir == Path::new("/") {
        anyhow::bail!("refusing unsafe FULL_MDP_WORK_DIR={}", work_dir.display());
    }
    let _ = fs::remove_dir_all(&work_dir);
    fs::create_dir_all(&work_dir)?;

    let l0_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    if target_reuse_stats {
        print_mdp_target_reuse_stats(&l0_layers, &graph, two_line_field)?;
    }
    let layer0_for_l0 = &l0_layers[0];
    let mut aa_hist = VecDeque::<AndersonEntry>::new();
    let mut prev_resid_l2: Option<f64> = None; // safeguarding: last iterate's ||T(x)-x||_2
    let mut last_accel_was_anderson = false;

    // layer0 checkpoints. layer0 is tiny (817,740 f32 ~= 3.2 MiB) so we archive it often:
    // iterations 1..=DENSE_UNTIL every iter, then every EVERY iterations. Snapshots go to
    // out_dir/l0_checkpoints/layer0_iter{N:05}.f32.bin (N is 1-indexed) plus the always-current
    // exact_path. Set FULL_MDP_L0_CKPT_EVERY=0 to disable the periodic tail.
    let ckpt_dense_until: usize = env::var("FULL_MDP_L0_CKPT_DENSE_UNTIL").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let ckpt_every: usize = env::var("FULL_MDP_L0_CKPT_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let ckpt_dir = out_dir.join("l0_checkpoints");
    fs::create_dir_all(&ckpt_dir)?;
    eprintln!("solve-full-mdp-l0-f32 layer0 checkpoints dir={} dense_until={} every={}", ckpt_dir.display(), ckpt_dense_until, ckpt_every);

    for iter in 0usize..max_iters {
        let iter_start = Instant::now();
        let reset_values = load_layer0_reset_dense(layer0_for_l0, &input_path, qstate_count)?;
        let terminal_reset_avg = build_terminal_reset_avg_table(&qfwd, &reset_values, gamma, qstate_count);
        let mut iter_stats = FullMdpLayerStats::default();
        let mut layer0_stats = FullMdpLayerStats::default();
        let mut next_values_path_holder: Option<PathBuf> = None;
        let mut next_values_ram_holder: Option<Vec<f32>> = None;

        eprintln!("solve-full-mdp-l0-f32 iter={} input={} work={}", iter, input_path.display(), work_dir.display());
        for layer in (0usize..=9).rev() {
            let layer_start = Instant::now();
            let cur = &l0_layers[layer];
            let out_path = if layer == 0 { next_l0_path.clone() } else { mdp_l0_work_value_path(&work_dir, layer) };
            let expected_layer_bytes = cur.total_states.checked_mul(4).context("l0-mdp f32 output size overflow")?;
            let next_layer_ref = if layer < 9 { Some(&l0_layers[layer + 1]) } else { None };
            let next_values_ref = next_values_path_holder.as_deref();
            let next_values_ram_ref = next_values_ram_holder.as_deref();
            let old_l0_for_delta = if layer == 0 { Some(reset_values.as_slice()) } else { None };

            let next_resident_bytes = next_values_ram_ref.map_or(0, |v| std::mem::size_of_val(v) as u64);
            let plan = tuning.plan(&cur.metas, 0, next_resident_bytes,
                rayon::current_num_threads(), qstate_count)?;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
            let batch_groups = batch_chunks_env.unwrap_or_else(|| plan.workers.saturating_mul(4)).max(1);
            let next_backend = if layer >= 9 {
                "none".to_string()
            } else if next_values_ram_ref.is_some() {
                format!("ram_values states={}", next_values_ram_ref.map_or(0, |v| v.len()))
            } else {
                next_values_ref.map(|p| p.display().to_string()).unwrap_or_else(|| "none".to_string())
            };
            eprintln!("solve-full-mdp-l0-f32 iter={} layer={} parallel workers={} groups={} batch_groups={} per_task_lookup_cache_mib={} max_working_mib={} next_backend={}",
                iter, layer, plan.workers, plan.ranges.len(), batch_groups,
                plan.per_task_cache_bytes / (1024 * 1024), plan.max_working_bytes / (1024 * 1024), next_backend);

            let out_file = Arc::new(File::create(&out_path)
                .with_context(|| format!("failed to create {}", out_path.display()))?);
            out_file.set_len(expected_layer_bytes)?;
            let mut layer_stats = FullMdpLayerStats::default();
            let mut blocks_done = 0usize;
            let total_batches = plan.ranges.len().div_ceil(batch_groups);
            for (batch_idx, batch) in plan.ranges.chunks(batch_groups).enumerate() {
                let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|range| {
                    compute_full_mdp_range_group_f32_rolling(
                        layer,
                        std::slice::from_ref(range),
                        cur,
                        &out_file,
                        &graph,
                        &qfwd,
                        &legal_placed,
                        &terminal_reset_avg,
                        next_layer_ref,
                        next_values_ref,
                        next_values_ram_ref,
                        old_l0_for_delta,
                        gamma,
                        plan.per_task_cache_bytes,
                        qstate_count,
                        sparse_lookup_max_count,
                        two_line_field,
                        prewarm_field_targets,
                        const_field_fastpath,
                        collect_cache_stats,
                        collect_action_stats,
                    )
                }).collect());
                for res in results {
                    let chunk = res?;
                    layer_stats.add(&chunk.stats);
                    blocks_done += chunk.blocks;
                }
                let is_last_batch = batch_idx + 1 == total_batches;
                if batch_idx % progress_every_batches == 0 || is_last_batch {
                    eprintln!("  solve-full-mdp-l0-f32 iter={} layer={} batch={}/{} blocks_done={}/{} states={} max_v={:.6} l0_max_resid={:.8} l0_mean_resid={:.8} cache_hit={} cache_miss={} evict={} const_hit={} const_miss={} elapsed={:.1}s",
                        iter, layer, batch_idx + 1, total_batches, blocks_done, cur.metas.len(), layer_stats.total_states, layer_stats.max_v,
                        layer_stats.max_delta,
                        if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_abs_delta / layer_stats.total_states as f64 },
                        layer_stats.dense_cache_hits, layer_stats.dense_cache_misses, layer_stats.dense_evictions, layer_stats.const_hits, layer_stats.const_misses, layer_start.elapsed().as_secs_f64());
                }
            }
            if sync_layer_outputs { out_file.sync_data()?; }
            drop(out_file);
            let actual = fs::metadata(&out_path)?.len();
            if actual != expected_layer_bytes { anyhow::bail!("l0-mdp f32 output size mismatch iter={} layer={}: got {}, expected {}", iter, layer, actual, expected_layer_bytes); }
            eprintln!("solve-full-mdp-l0-f32 iter={} layer={} done states={} nonzero={} max_v={:.8} avg_v={:.8} l0_max_resid={:.8} l0_mean_resid={:.8} term2={} term4={} cache_hit={} cache_miss={} missing_fields={} evict={} mat_states={} dense_blocks={} sparse_blocks={} bitmap_blocks={} const_checks={} const_hits={} const_misses={} bytes={} elapsed={:.3}s",
                iter, layer, layer_stats.total_states, layer_stats.nonzero, layer_stats.max_v,
                if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_v / layer_stats.total_states as f64 },
                layer_stats.max_delta,
                if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_abs_delta / layer_stats.total_states as f64 },
                layer_stats.terminal_2_edges, layer_stats.terminal_4_edges,
                layer_stats.dense_cache_hits, layer_stats.dense_cache_misses, layer_stats.dense_missing_fields, layer_stats.dense_evictions, layer_stats.dense_materialized_states, layer_stats.dense_blocks, layer_stats.sparse_blocks, layer_stats.bitmap_blocks, layer_stats.const_checks, layer_stats.const_hits, layer_stats.const_misses,
                actual, layer_start.elapsed().as_secs_f64());
            iter_stats.add(&layer_stats);
            // Only retain one backing source. A RAM pin is optional per layer;
            // when the shared budget cannot fit it the next layer reads this file.
            if let Some(old_next) = next_values_path_holder.take() {
                let _ = fs::remove_file(&old_next);
            }
            next_values_ram_holder.take();
            if layer == 0 {
                if last_f_path.exists() { let _ = fs::remove_file(&last_f_path); }
                fs::rename(&out_path, &last_f_path)
                    .with_context(|| format!("failed to rename {} -> {}", out_path.display(), last_f_path.display()))?;
                layer0_stats = layer_stats;
            } else {
                // Check the actual following layer as well: a single oversized
                // membership block may need more than the usual task allowance.
                let can_pin = ram_next_layer && expected_layer_bytes <= ram_next_layer_max_bytes
                    && tuning.can_hold_layers(0, expected_layer_bytes, qstate_count)
                    && tuning.plan(&l0_layers[layer - 1].metas, 0, expected_layer_bytes,
                        rayon::current_num_threads(), qstate_count).is_ok();
                if can_pin {
                    let load_start = Instant::now();
                    let loaded = load_f32_values_vec(&out_path, cur.total_states)
                        .with_context(|| format!("failed to load layer {} values into RAM from {}", layer, out_path.display()))?;
                    let loaded_states = loaded.len();
                    let _ = fs::remove_file(&out_path);
                    eprintln!("solve-full-mdp-l0-f32 iter={} layer={} ram_next_loaded states={} bytes={} elapsed={:.3}s removed_temp={}",
                        iter, layer, loaded_states, expected_layer_bytes, load_start.elapsed().as_secs_f64(), out_path.display());
                    next_values_ram_holder = Some(loaded);
                } else {
                    if ram_next_layer {
                        eprintln!("solve-full-mdp-l0-f32 iter={} layer={} ram_pin_skipped bytes={} max_pin_bytes={} (layer {} will read from disk)",
                            iter, layer, expected_layer_bytes, ram_next_layer_max_bytes, layer - 1);
                    }
                    next_values_path_holder = Some(out_path);
                }
            }
        }
        if let Some(leftover) = next_values_path_holder.take() { let _ = fs::remove_file(leftover); }
        next_values_ram_holder.take();
        copy_checked_file(&last_f_path, &exact_path, layer0_for_l0.total_states * 4)?;

        let l0_mean_resid = if layer0_stats.total_states == 0 { 0.0 } else { layer0_stats.sum_abs_delta / layer0_stats.total_states as f64 };
        // Convergence criteria. For each tolerance: < 0 disables it, == 0 requires the
        // residual to be EXACTLY zero (bit-exact f32 fixed point), > 0 is a threshold.
        // Defaults (tol_abs=1e-5, tol_mean=0) => converge on max_delta<=1e-5 as before.
        // To run until max_delta==0, pass tol_abs=0 tol_mean=-1 (or any tol_mean<0).
        let abs_enabled = tol_abs >= 0.0;
        let mean_enabled = tol_mean > 0.0;
        let abs_ok = !abs_enabled || layer0_stats.max_delta <= tol_abs;
        let mean_ok = !mean_enabled || l0_mean_resid <= tol_mean;
        let converged = abs_ok && mean_ok && (abs_enabled || mean_enabled);

        // Archive this iteration's layer0 (in exact_path) per the checkpoint schedule
        // (1-indexed). Always archive the converged/final iteration so the fixed point is
        // captured even if it lands off-schedule.
        let n = iter + 1;
        if n <= ckpt_dense_until || (ckpt_every > 0 && n % ckpt_every == 0) || converged || n == max_iters {
            let ckpt_path = ckpt_dir.join(format!("layer0_iter{:05}.f32.bin", n));
            copy_checked_file(&exact_path, &ckpt_path, layer0_for_l0.total_states * 4)?;
            eprintln!("solve-full-mdp-l0-f32 iter={} checkpoint saved {} (converged={})", iter, ckpt_path.display(), converged);
        }

        let f_values = load_layer0_reset_dense(layer0_for_l0, &last_f_path, qstate_count)?;
        let mut accel_used = "plain".to_string();
        if converged {
            copy_checked_file(&last_f_path, &input_path, layer0_for_l0.total_states * 4)?;
        } else {
            let mut r = Vec::<f32>::with_capacity(qstate_count);
            for i in 0..qstate_count { r.push(f_values[i] - reset_values[i]); }
            let resid_l2 = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            // Safeguard: if the last step used AA and the residual grew beyond tolerance, the
            // extrapolation was harmful. Discard the AA history and take a plain VI step now.
            let grew = prev_resid_l2.map_or(false, |p| resid_l2 > p * (1.0 + aa_restart_tol));
            let do_restart = aa_restart && last_accel_was_anderson && grew;
            if do_restart {
                aa_hist.clear();
            }
            aa_hist.push_back(AndersonEntry { f: f_values.clone(), r });
            while aa_hist.len() > aa_depth.max(1) { aa_hist.pop_front(); }
            let mut used_anderson = false;
            if !do_restart && aa_depth >= 2 && iter + 1 >= aa_start {
                if let Some(candidate) = anderson_l0_candidate(&aa_hist, aa_beta, aa_reg_scale) {
                    write_layer0_dense_values(layer0_for_l0, &input_path, &candidate)?;
                    accel_used = format!("anderson(k={},beta={:.2})", aa_hist.len(), aa_beta);
                    used_anderson = true;
                } else {
                    copy_checked_file(&last_f_path, &input_path, layer0_for_l0.total_states * 4)?;
                    accel_used = "plain_anderson_rejected".to_string();
                }
            } else {
                copy_checked_file(&last_f_path, &input_path, layer0_for_l0.total_states * 4)?;
                if do_restart { accel_used = format!("plain_restart(resid_grew {:.6e}->{:.6e})", prev_resid_l2.unwrap_or(0.0), resid_l2); }
            }
            last_accel_was_anderson = used_anderson;
            prev_resid_l2 = Some(resid_l2);
            eprintln!("solve-full-mdp-l0-f32 iter={} resid_l2={:.8e} accel={} aa_hist={}", iter, resid_l2, accel_used, aa_hist.len());
        }

        let latest = out_dir.join("LATEST.txt");
        fs::write(&latest, format!("iter={} converged={} exact_l0={} next_input={} gamma={} l0_max_resid={} l0_mean_resid={} accel={} elapsed_sec={:.3}\n",
            iter, converged, exact_path.display(), input_path.display(), gamma, layer0_stats.max_delta, l0_mean_resid, accel_used, iter_start.elapsed().as_secs_f64()))?;
        eprintln!("solve-full-mdp-l0-f32 iter={} done converged={} l0_max_resid={:.8} l0_mean_resid={:.8} total_states={} max_v={:.8} avg_v={:.8} accel={} elapsed={:.3}s latest={}",
            iter, converged, layer0_stats.max_delta, l0_mean_resid, iter_stats.total_states, iter_stats.max_v,
            if iter_stats.total_states == 0 { 0.0 } else { iter_stats.sum_v / iter_stats.total_states as f64 },
            accel_used, iter_start.elapsed().as_secs_f64(), latest.display());
        if converged { break; }
    }
    let _ = fs::remove_dir_all(&work_dir);
    eprintln!("solve-full-mdp-l0-f32 all done exact_l0={} input_l0={} marker={}", exact_path.display(), input_path.display(), out_dir.join("LATEST.txt").display());
    Ok(())
}

fn run_solve_full_mdp_discount_f32(args: &[String]) -> Result<()> {
    if args.len() < 5 || args.len() > 9 {
        eprintln!("usage: {} solve-full-mdp-discount-f32 <graph.bin> <pruned_layer_dir> <out_dir> [gamma=1.0] [iters=20] [dense_cache_gib=32] [init=zero|one_loop:<dir>]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    let gamma: f32 = if args.len() >= 6 { args[5].parse().context("invalid gamma")? } else { 1.0 };
    let iters: usize = if args.len() >= 7 { args[6].parse().context("invalid iters")? } else { 20 };
    let cache_gib: usize = if args.len() >= 8 { args[7].parse().context("invalid dense_cache_gib")? } else { 32 };
    let init_spec = if args.len() >= 9 { args[8].as_str() } else { "zero" };
    if !(0.0..=1.0).contains(&gamma) { anyhow::bail!("gamma must be in [0,1], got {}", gamma); }

    let cache_cap_bytes = cache_gib.saturating_mul(1024usize).saturating_mul(1024usize).saturating_mul(1024usize);
    let tuning = sweep_schedule::SweepTuning::from_env(cache_cap_bytes)?;

    fs::create_dir_all(&out_dir)?;
    let values_a = out_dir.join("values_a");
    let values_b = out_dir.join("values_b");
    fs::create_dir_all(&values_b)?;

    eprintln!("solve-full-mdp-discount-f32 loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    eprintln!("solve-full-mdp-discount-f32 building q index and qforward graph...");
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    let legal_placed = build_qfwd_legal_placed_table(&qfwd, qstate_count);
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).unwrap_or(11335);
    eprintln!("solve-full-mdp-discount-f32 qstates={} qfwd_records={} gamma={} iters={} dense_cache_gib={} init={} two_line_field={} four_line_terminal_field={}",
        qstate_count, qfwd.records.len(), gamma, iters, cache_gib, init_spec, two_line_field, EMPTY_FIELD_ID);

    if !mdp_f32_value_path(&values_a, 0).exists() {
        eprintln!("solve-full-mdp-discount-f32 initializing values_a from {}", init_spec);
        initialize_full_mdp_values(&pruned_dir, &values_a, init_spec)?;
    } else {
        eprintln!("solve-full-mdp-discount-f32 reusing existing {}", values_a.display());
    }

    let sync_layer_outputs: bool = env::var("FULL_MDP_SYNC_LAYER")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let prewarm_field_targets: bool = env::var("FULL_MDP_PREWARM_FIELD_TARGETS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let collect_cache_stats: bool = env::var("FULL_MDP_CACHE_STATS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let collect_action_stats: bool = env::var("FULL_MDP_ACTION_STATS")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let sparse_lookup_max_count: usize = env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536);
    let batch_chunks_env: Option<usize> = env::var("FULL_MDP_BATCH_CHUNKS").ok().and_then(|s| s.parse().ok());
    // Optional convergence early-stop (opt-in; 0 = disabled = old fixed-iters behavior).
    // Residuals are accumulated in f64 (sum_abs_delta), so tol_mean is reliable even when
    // f32-stored values are large under gamma=1; tol_max reads the (f32) max per-iter delta.
    let tol_mean: f64 = env::var("FULL_MDP_TOL_MEAN").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let tol_max: f64 = env::var("FULL_MDP_TOL_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    if tol_mean > 0.0 || tol_max > 0.0 {
        eprintln!("solve-full-mdp-discount-f32 early_stop enabled tol_mean={} tol_max={}", tol_mean, tol_max);
    }
    // Rolling in-RAM next-layer (opt-in). The Gauss-Seidel sweep reads layer L+1 while
    // computing layer L; instead of re-reading it from disk per lookup, keep just the ONE
    // next layer resident (single rolling Vec<f32>, output still streamed to disk for the
    // next iteration's prev/reset). Peak RAM ~= largest single layer + working set. A layer
    // is only pinned if it fits FULL_MDP_RAM_NEXT_LAYER_MAX_GIB; larger layers transparently
    // fall back to the File backend, so this is safe under a fixed RAM cap.
    let ram_next_layer: bool = env::var("FULL_MDP_RAM_NEXT_LAYER")
        .ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
        .unwrap_or(false);
    let ram_next_layer_max_bytes: u64 = env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(96)
        .saturating_mul(1024 * 1024 * 1024);
    if ram_next_layer {
        eprintln!("solve-full-mdp-discount-f32 ram_next_layer enabled max_pin_gib={}", ram_next_layer_max_bytes / (1024 * 1024 * 1024));
    }
    eprintln!("solve-full-mdp-discount-f32 sweep=backward-gauss-seidel nonterminal_source=current_iter_next_layer terminal_reset_source=prev_iter_layer0 sync_layer_outputs={} sparse_lookup_max_count={} opt5_terminal_free_fastpath=true opt5_unroll_reveal_sum=true opt5_active_piece_mask=true opt5_cache_stats={} opt5_action_stats={} opt5_prewarm_field_targets={}", sync_layer_outputs, sparse_lookup_max_count, collect_cache_stats, collect_action_stats, prewarm_field_targets);
    let mut prev_dir = values_a;
    let mut next_dir = values_b;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;

    for iter in 0usize..iters {
        let iter_start = Instant::now();
        eprintln!("solve-full-mdp-discount-f32 iter={} prev={} next={}", iter, prev_dir.display(), next_dir.display());
        fs::create_dir_all(&next_dir)?;
        let layer0 = &full_layers[0];
        let reset_values = load_layer0_reset_dense(layer0, &mdp_f32_value_path(&prev_dir, 0), qstate_count)?;
        let terminal_reset_avg = build_terminal_reset_avg_table(&qfwd, &reset_values, gamma, qstate_count);

        let mut iter_stats = FullMdpLayerStats::default();
        // Holds the ONE next layer (layer+1) resident in RAM for the rolling-buffer path.
        // Loaded from each just-written layer file; reset at the start of every iteration.
        let mut next_values_ram_holder: Option<Vec<f32>> = None;
        for layer in (0usize..=9).rev() {
            let layer_start = Instant::now();
            let cur = &full_layers[layer];
            let prev_cur_path = mdp_f32_value_path(&prev_dir, layer);
            let out_path = mdp_f32_value_path(&next_dir, layer);
            let expected_layer_bytes = cur.total_states.checked_mul(4).context("full-mdp f32 output size overflow")?;
            let next_layer_ref = if layer < 9 { Some(&full_layers[layer + 1]) } else { None };
            // Backward Gauss-Seidel sweep: nonterminal transitions read the layer+1 values
            // already computed in this same iteration.  The only cross-iteration dependency is
            // terminal reset, which intentionally uses prev_dir/layer0 via terminal_reset_avg.
            let next_values_path_holder = if layer < 9 { Some(mdp_f32_value_path(&next_dir, layer + 1)) } else { None };
            let next_values_ref = next_values_path_holder.as_deref();
            // Use the RAM-resident next layer when available (layer<9 and it was pinned);
            // otherwise next_values_ram_ref is None and the File backend + path is used.
            let next_values_ram_ref = if layer < 9 { next_values_ram_holder.as_deref() } else { None };
            if layer < 9 {
                eprintln!("solve-full-mdp-discount-f32 iter={} layer={} next_source={}", iter, layer,
                    if next_values_ram_ref.is_some() { "ram" } else { "file" });
            }

            let next_resident_bytes = next_values_ram_ref.map_or(0, |v| std::mem::size_of_val(v) as u64);
            let plan = tuning.plan(&cur.metas, 0, next_resident_bytes,
                rayon::current_num_threads(), qstate_count)?;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
            let batch_groups = batch_chunks_env.unwrap_or_else(|| plan.workers.saturating_mul(4)).max(1);
            eprintln!("solve-full-mdp-discount-f32 iter={} layer={} parallel workers={} groups={} batch_groups={} per_task_dense_cache_mib={} max_working_mib={}",
                iter, layer, plan.workers, plan.ranges.len(), batch_groups,
                plan.per_task_cache_bytes / (1024 * 1024), plan.max_working_bytes / (1024 * 1024));

            let out_file = Arc::new(File::create(&out_path)
                .with_context(|| format!("failed to create {}", out_path.display()))?);
            out_file.set_len(expected_layer_bytes)?;
            let mut layer_stats = FullMdpLayerStats::default();
            let mut blocks_done = 0usize;
            for batch in plan.ranges.chunks(batch_groups) {
                let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|r| {
                    compute_full_mdp_chunk_f32(
                        layer,
                        r.clone(),
                        cur,
                        Some(prev_cur_path.as_path()),
                        &out_file,
                        None,
                        &graph,
                        &qfwd,
                        &legal_placed,
                        &terminal_reset_avg,
                        next_layer_ref,
                        next_values_ref,
                        next_values_ram_ref,
                        gamma,
                        plan.per_task_cache_bytes,
                        qstate_count,
                        sparse_lookup_max_count,
                        two_line_field,
                        prewarm_field_targets,
                        false,
                        collect_cache_stats,
                        collect_action_stats,
                    )
                }).collect());
                for res in results {
                    let chunk = res?;
                    layer_stats.add(&chunk.stats);
                    blocks_done += chunk.blocks;
                }
                eprintln!("  solve-full-mdp-discount-f32 iter={} layer={} blocks_done={}/{} states={} max_v={:.6} max_delta={:.8} mean_delta={:.8} actions={} cache_hit={} cache_miss={} evict={} mat_states={} elapsed={:.1}s",
                    iter, layer, blocks_done, cur.metas.len(), layer_stats.total_states, layer_stats.max_v, layer_stats.max_delta,
                    if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_abs_delta / layer_stats.total_states as f64 },
                    layer_stats.action_count, layer_stats.dense_cache_hits, layer_stats.dense_cache_misses, layer_stats.dense_evictions, layer_stats.dense_materialized_states, layer_start.elapsed().as_secs_f64());
            }
            if sync_layer_outputs { out_file.sync_data()?; }
            let actual = fs::metadata(&out_path)?.len();
            let expected = cur.total_states * 4;
            if actual != expected { anyhow::bail!("full-mdp f32 output size mismatch iter={} layer={}: got {}, expected {}", iter, layer, actual, expected); }
            eprintln!("solve-full-mdp-discount-f32 iter={} layer={} done states={} nonzero={} max_v={:.8} avg_v={:.8} max_delta={:.8} mean_delta={:.8} term2={} term4={} actions={} missing_zero_terms={} cache_hit={} cache_miss={} missing_fields={} evict={} mat_states={} dense_blocks={} sparse_blocks={} bitmap_blocks={} const_checks={} const_hits={} const_misses={} bytes={} elapsed={:.3}s",
                iter, layer, layer_stats.total_states, layer_stats.nonzero, layer_stats.max_v,
                if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_v / layer_stats.total_states as f64 },
                layer_stats.max_delta,
                if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_abs_delta / layer_stats.total_states as f64 },
                layer_stats.terminal_2_edges, layer_stats.terminal_4_edges, layer_stats.action_count, layer_stats.missing_next_terms,
                layer_stats.dense_cache_hits, layer_stats.dense_cache_misses, layer_stats.dense_missing_fields, layer_stats.dense_evictions, layer_stats.dense_materialized_states, layer_stats.dense_blocks, layer_stats.sparse_blocks, layer_stats.bitmap_blocks, layer_stats.const_checks, layer_stats.const_hits, layer_stats.const_misses,
                actual, layer_start.elapsed().as_secs_f64());
            iter_stats.add(&layer_stats);

            // Rolling RAM next-layer: after writing layer L, pin it in RAM for the layer
            // below (L-1) to read, instead of a disk re-read. Drop the previous resident
            // (layer L+1) FIRST so peak stays near one layer. Unlike the l0 solver we KEEP
            // the on-disk file: discount-f32 needs it as prev_dir next iteration.
            if ram_next_layer && layer >= 1 {
                next_values_ram_holder = None; // free layer L+1 before allocating layer L
                if expected_layer_bytes <= ram_next_layer_max_bytes
                    && tuning.can_hold_layers(0, expected_layer_bytes, qstate_count)
                    && tuning.plan(&full_layers[layer - 1].metas, 0, expected_layer_bytes,
                        rayon::current_num_threads(), qstate_count).is_ok() {
                    let load_start = Instant::now();
                    let loaded = load_f32_values_vec(&out_path, cur.total_states)
                        .with_context(|| format!("failed to pin layer {} values into RAM from {}", layer, out_path.display()))?;
                    eprintln!("solve-full-mdp-discount-f32 iter={} layer={} ram_pinned states={} bytes={} elapsed={:.3}s",
                        iter, layer, loaded.len(), expected_layer_bytes, load_start.elapsed().as_secs_f64());
                    next_values_ram_holder = Some(loaded);
                } else {
                    eprintln!("solve-full-mdp-discount-f32 iter={} layer={} ram_pin_skipped bytes={} max_pin_bytes={} (layer {} will read from disk)",
                        iter, layer, expected_layer_bytes, ram_next_layer_max_bytes, layer - 1);
                }
            }
        }
        next_values_ram_holder = None;
        let latest = out_dir.join("LATEST.txt");
        fs::write(&latest, format!("iter={} dir={} gamma={} max_delta={} mean_delta={} elapsed_sec={:.3}\n", iter, next_dir.display(), gamma, iter_stats.max_delta,
            if iter_stats.total_states == 0 { 0.0 } else { iter_stats.sum_abs_delta / iter_stats.total_states as f64 },
            iter_start.elapsed().as_secs_f64()))?;
        eprintln!("solve-full-mdp-discount-f32 iter={} done total_states={} max_v={:.8} avg_v={:.8} max_delta={:.8} mean_delta={:.8} elapsed={:.3}s latest={}",
            iter, iter_stats.total_states, iter_stats.max_v,
            if iter_stats.total_states == 0 { 0.0 } else { iter_stats.sum_v / iter_stats.total_states as f64 },
            iter_stats.max_delta,
            if iter_stats.total_states == 0 { 0.0 } else { iter_stats.sum_abs_delta / iter_stats.total_states as f64 },
            iter_start.elapsed().as_secs_f64(), next_dir.display());
        std::mem::swap(&mut prev_dir, &mut next_dir);
        // Early-stop once the sweep has stabilized. prev_dir now holds this iter's output
        // (post-swap), so breaking here leaves the converged values in prev_dir.
        let iter_mean_delta = if iter_stats.total_states == 0 { 0.0 } else { iter_stats.sum_abs_delta / iter_stats.total_states as f64 };
        let iter_max_delta = iter_stats.max_delta as f64;
        let mean_ok = tol_mean > 0.0 && iter_mean_delta <= tol_mean;
        let max_ok = tol_max > 0.0 && iter_max_delta <= tol_max;
        if mean_ok || max_ok {
            eprintln!("solve-full-mdp-discount-f32 converged early at iter={} (max_delta={:.8} mean_delta={:.8}; tol_max={} tol_mean={})",
                iter, iter_max_delta, iter_mean_delta, tol_max, tol_mean);
            break;
        }
    }
    eprintln!("solve-full-mdp-discount-f32 all done latest_values_dir={} marker={}", prev_dir.display(), out_dir.join("LATEST.txt").display());
    Ok(())
}

// Emit all 10 layer value files in ONE backward Gauss-Seidel pass, using an already-converged
// layer0 (from solve-full-mdp-l0-f32) as the fixed terminal-reset source. This is step 2 of the
// "converge layer0 cheaply, then materialize every state once" plan: instead of persisting all
// layers on every value-iteration sweep (321 GiB written per iter), the l0 solver converges the
// fixed point keeping only layer0, and this pass writes the full per-state value table exactly once.
// Only intermediates created in this fresh directory belong to this invocation.
// In particular, policy-only sweeps never remove pre-existing value snapshots.
struct GreedyEmitScratch {
    path: PathBuf,
}

impl GreedyEmitScratch {
    fn create(parent: &Path) -> Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        for attempt in 0..100u32 {
            let path = parent.join(format!(".policy-rolling-{}-{}-{}", std::process::id(), stamp, attempt));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).with_context(|| format!("failed to create {}", path.display())),
            }
        }
        anyhow::bail!("could not create an owned policy rolling directory in {}", parent.display())
    }
}

impl Drop for GreedyEmitScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct GreedyEmitConfig {
    policy_only: bool,
    ram_next_layer: bool,
    ram_next_layer_max_bytes: u64,
    ram_pair_max_bytes: u64,
    mmap_spill: bool,
    prewarm_field_targets: bool,
    collect_cache_stats: bool,
    collect_action_stats: bool,
    sync_layer_outputs: bool,
    sparse_lookup_max_count: usize,
    tuning: sweep_schedule::SweepTuning,
}

fn write_greedy_ram_snapshot(path: &Path, values: &[f32]) -> Result<()> {
    if cfg!(target_endian = "little") {
        // SAFETY: f32 has no padding, every value is initialized, and the shared
        // byte view lasts only for this synchronous write.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) };
        let mut file = File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(bytes)?;
        Ok(())
    } else {
        write_f32_vec_file(path, values)
    }
}

fn emit_greedy_mdp_sweep(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    full_layers: &[layer_store::FieldChainLayer],
    out_dir: &Path,
    policy_out_dir: Option<&Path>,
    qstate_count: usize,
    two_line_field: FieldId,
    gamma: f32,
    config: &GreedyEmitConfig,
) -> Result<()> {
    emit_greedy_mdp_sweep_with_action_cache(
        graph, qfwd, legal_placed, terminal_reset_avg, full_layers, out_dir, policy_out_dir, qstate_count, two_line_field, gamma, config, None,
    )
}

fn emit_greedy_mdp_sweep_with_action_cache(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal_placed: &[[u8; 2]],
    terminal_reset_avg: &[[f32; 7]],
    full_layers: &[layer_store::FieldChainLayer],
    out_dir: &Path,
    policy_out_dir: Option<&Path>,
    qstate_count: usize,
    two_line_field: FieldId,
    gamma: f32,
    config: &GreedyEmitConfig,
    action_source: Option<action_avg_cache::ActionSource<'_>>,
) -> Result<()> {
    anyhow::ensure!(full_layers.len() == 10, "greedy sweep needs ten membership layers");
    anyhow::ensure!(!config.policy_only || policy_out_dir.is_some(), "FULL_MDP_POLICY_ONLY=1 requires policy_out_dir");
    if let Some(source) = action_source { source.placements.check_graph(qfwd, qstate_count)?; }
    let action_resident_bytes = action_source.map_or(0, |source| source.storage_bytes());
    let direct_l9 = config.policy_only && terminal_l9::enabled_from_env();
    let scratch = if config.policy_only { Some(GreedyEmitScratch::create(out_dir)?) } else { None };
    let mut next_values_holder: Option<PinnedLayer> = None;
    let mut next_values_path: Option<PathBuf> = None;
    let mut next_path_is_owned = false;
    if let Some(source) = action_source {
        eprintln!("greedy-action-cache enabled requested_bytes_per_task={} placement_resident_bytes={} boundary_reduced=0 scope=group_next_layer",
            source.requested_bytes, action_resident_bytes);
    }
    for layer in (0usize..=9).rev() {
        let layer_start = Instant::now();
        let cur = &full_layers[layer];
        if layer == 9 && direct_l9 {
            // L9 values are evaluated at the L8 lookup. Stream the terminal
            // policy using the same strict argmax without creating an f32 sink.
            terminal_l9::validate_membership(terminal_l9::Source {
                layer: cur, graph, qfwd, terminal_reset_avg,
                mode: terminal_l9::Mode::Greedy(legal_placed),
            })?;
            let policy_path = mdp_policy_u16_path(policy_out_dir.expect("policy-only directory"), 9);
            let plan = config.tuning.plan(&cur.metas, 0, action_resident_bytes, rayon::current_num_threads(), qstate_count)?;
            let policy_file = File::create(&policy_path)
                .with_context(|| format!("failed to create {}", policy_path.display()))?;
            policy_file.set_len(cur.total_states.checked_mul(2).context("L9 policy size overflow")?)?;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
            let mut states = 0u64;
            for batch in plan.ranges.chunks(plan.workers.saturating_mul(4).max(1)) {
                let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|range| {
                    compute_full_mdp_range_group_f32_rolling_sinks_with_l9_source(
                        9, std::slice::from_ref(range), cur, None, None, Some(&policy_file),
                        graph, qfwd, legal_placed, terminal_reset_avg, None, None, None, None,
                        gamma, plan.per_task_cache_bytes, qstate_count, config.sparse_lookup_max_count,
                        two_line_field, false, false, config.collect_cache_stats, config.collect_action_stats, None,
                    )
                }).collect());
                for result in results { states += result?.stats.total_states; }
            }
            anyhow::ensure!(states == cur.total_states, "direct L9 policy state count mismatch");
            if config.sync_layer_outputs { policy_file.sync_data()?; }
            eprintln!("emit-full-mdp-layers-from-l0 layer=9 states={} sink=policy next_source=terminal_l9 value_bytes=0 elapsed={:.3}s", states, layer_start.elapsed().as_secs_f64());
            continue;
        }
        let out_path = mdp_f32_value_path(
            if layer != 0 { scratch.as_ref().map_or(out_dir, |s| s.path.as_path()) } else { out_dir },
            layer,
        );
        let persist_values = !config.policy_only || layer == 0;
        let expected_layer_bytes = cur.total_states.checked_mul(4).context("emit f32 output size overflow")?;
        let next_anonymous_bytes = match next_values_holder.as_ref() {
            Some(PinnedLayer::Ram(v)) => std::mem::size_of_val(v.as_slice()) as u64,
            _ => 0,
        };
        let next_resident_bytes = next_anonymous_bytes.checked_add(action_resident_bytes)
            .context("greedy placement resident memory overflow")?;
        let mut use_ram_out = expected_layer_bytes == 0 || (!config.mmap_spill && config.ram_next_layer
            && expected_layer_bytes <= config.ram_next_layer_max_bytes
            && expected_layer_bytes.saturating_add(next_anonymous_bytes) <= config.ram_pair_max_bytes
            && config.tuning.can_hold_layers(expected_layer_bytes, next_resident_bytes, qstate_count));
        // Validate the membership and memory plan before allocating or truncating
        // an output. A single large block can require more task scratch than the
        // approximate RAM fit check; spill that layer instead of exceeding budget.
        let plan = match config.tuning.plan(
            &cur.metas, if use_ram_out { expected_layer_bytes } else { 0 },
            next_resident_bytes, rayon::current_num_threads(), qstate_count,
        ) {
            Ok(plan) => plan,
            Err(_) if use_ram_out && expected_layer_bytes != 0 => {
                use_ram_out = false;
                config.tuning.plan(&cur.metas, 0, next_resident_bytes,
                    rayon::current_num_threads(), qstate_count)?
            }
            Err(err) => return Err(err),
        };
        let (out_file, mut ram_buf, mut mmap_buf): (Option<File>, Option<Vec<f32>>, Option<memmap2::MmapMut>) = if use_ram_out {
            (None, Some(vec![0.0f32; usize::try_from(cur.total_states).context("RAM layer length overflow")?]), None)
        } else if config.mmap_spill {
            let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&out_path)
                .with_context(|| format!("failed to open {}", out_path.display()))?;
            f.set_len(expected_layer_bytes)?;
            // SAFETY: this invocation exclusively owns the output file and mapping.
            let mm = unsafe { memmap2::MmapMut::map_mut(&f) }
                .with_context(|| format!("failed to mmap {}", out_path.display()))?;
            (None, None, Some(mm))
        } else {
            let f = File::create(&out_path).with_context(|| format!("failed to create {}", out_path.display()))?;
            f.set_len(expected_layer_bytes)?;
            (Some(f), None, None)
        };
        let ram_out = ram_buf.as_mut().map(|b| SharedF32Out { ptr: b.as_mut_ptr(), len: b.len() })
            .or_else(|| mmap_buf.as_mut().map(|m| SharedF32Out { ptr: m.as_mut_ptr() as *mut f32, len: m.len() / 4 }));
        let policy_out_file = if let Some(pd) = policy_out_dir {
            let pp = mdp_policy_u16_path(pd, layer);
            let f = File::create(&pp).with_context(|| format!("failed to create {}", pp.display()))?;
            f.set_len(cur.total_states.checked_mul(2).context("emit policy output size overflow")?)?;
            Some(f)
        } else { None };
        let next_layer_ref = if layer < 9 { Some(&full_layers[layer + 1]) } else { None };
        let next_values_ram = next_values_holder.as_ref().map(|p| p.as_f32());
        let direct_l9_source = if direct_l9 && layer == 8 {
            Some(terminal_l9::Source {
                layer: &full_layers[9], graph, qfwd, terminal_reset_avg,
                mode: terminal_l9::Mode::Greedy(legal_placed),
            })
        } else { None };
        eprintln!("emit-full-mdp-layers-from-l0 layer={} sink={} next_source={} groups={} workers={} per_task_cache_mib={} max_working_mib={} policy_only={}",
            layer, if mmap_buf.is_some() { "mmap" } else if use_ram_out { "ram" } else { "file" },
            if direct_l9_source.is_some() { "terminal_l9" } else if next_values_ram.is_some() { "pinned" } else { "file" }, plan.ranges.len(), plan.workers,
            plan.per_task_cache_bytes / (1024 * 1024), plan.max_working_bytes / (1024 * 1024), config.policy_only);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
        let mut layer_stats = FullMdpLayerStats::default();
        // Each group owns one bounded sequential chain read and one cache. Process
        // bounded batches so completed results do not accumulate for an entire layer.
        for batch in plan.ranges.chunks(plan.workers.saturating_mul(4).max(1)) {
            let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|range| {
                compute_full_mdp_range_group_f32_rolling_sinks_with_action_cache(
                    layer, std::slice::from_ref(range), cur, out_file.as_ref(), ram_out, policy_out_file.as_ref(),
                    graph, qfwd, legal_placed, terminal_reset_avg,
                    next_layer_ref, next_values_path.as_deref(), next_values_ram, None,
                    gamma, plan.per_task_cache_bytes, qstate_count, config.sparse_lookup_max_count,
                    two_line_field, config.prewarm_field_targets, false,
                    config.collect_cache_stats, config.collect_action_stats,
                    direct_l9_source, action_source,
                )
            }).collect());
            for res in results { layer_stats.add(&res?.stats); }
        }
        anyhow::ensure!(layer_stats.total_states == cur.total_states,
            "emit state count mismatch layer={}: computed {}, expected {}", layer, layer_stats.total_states, cur.total_states);
        if persist_values {
            if let Some(values) = ram_buf.as_ref() { write_greedy_ram_snapshot(&out_path, values)?; }
        }
        if config.sync_layer_outputs {
            if let Some(mm) = mmap_buf.as_ref() { mm.flush()?; }
            if persist_values || out_file.is_some() { File::open(&out_path)?.sync_data()?; }
            if let Some(pf) = policy_out_file.as_ref() { pf.sync_data()?; }
        }
        let values_on_disk = !use_ram_out || persist_values;
        if values_on_disk {
            let actual = fs::metadata(&out_path)?.len();
            anyhow::ensure!(actual == expected_layer_bytes, "emit f32 output size mismatch layer={}: got {}, expected {}", layer, actual, expected_layer_bytes);
        }
        if let Some(pf) = policy_out_file.as_ref() {
            anyhow::ensure!(pf.metadata()?.len() == cur.total_states * 2, "emit policy output size mismatch layer={}", layer);
        }
        eprintln!("emit-full-mdp-layers-from-l0 layer={} done states={} nonzero={} max_v={:.8} avg_v={:.8} actions={} missing_terms={} cache_hit={} cache_miss={} evict={} mat_states={} dense_blocks={} sparse_blocks={} bitmap_blocks={} action_avg_hits={} action_avg_misses={} action_avg_collisions={} action_avg_saved_reveals={} action_avg_cache_bytes={} bytes={} elapsed={:.3}s",
            layer, layer_stats.total_states, layer_stats.nonzero, layer_stats.max_v,
            if layer_stats.total_states == 0 { 0.0 } else { layer_stats.sum_v / layer_stats.total_states as f64 },
            layer_stats.action_count, layer_stats.missing_next_terms, layer_stats.dense_cache_hits,
            layer_stats.dense_cache_misses, layer_stats.dense_evictions, layer_stats.dense_materialized_states,
            layer_stats.dense_blocks, layer_stats.sparse_blocks, layer_stats.bitmap_blocks,
            layer_stats.action_avg_hits, layer_stats.action_avg_misses, layer_stats.action_avg_collisions,
            layer_stats.action_avg_saved_reveals, layer_stats.action_avg_cache_bytes,
            if values_on_disk { expected_layer_bytes } else { 0 }, layer_start.elapsed().as_secs_f64());

        // Readers of the old next layer have completed. Release its mapping before
        // deleting the disposable file created by this sweep, never a user snapshot.
        next_values_holder = None;
        if next_path_is_owned {
            if let Some(path) = next_values_path.take() { fs::remove_file(&path).with_context(|| format!("failed to remove owned intermediate {}", path.display()))?; }
        }
        next_values_path = None;
        next_path_is_owned = false;
        drop(out_file);
        if layer != 0 {
            if let Some(mm) = mmap_buf.take() {
                let _ = mm.flush_async();
                next_values_holder = Some(PinnedLayer::Mapped(mm.make_read_only().context("make greedy layer mapping read-only")?));
            } else if let Some(values) = ram_buf.take() {
                next_values_holder = Some(PinnedLayer::Ram(values));
            } else if config.ram_next_layer {
                // Large file sinks are mapped for the next layer rather than read
                // back into an anonymous allocation. Clean pages remain evictable.
                let file = File::open(&out_path)?;
                let mm = unsafe { memmap2::Mmap::map(&file) }
                    .with_context(|| format!("failed to map completed layer {}", out_path.display()))?;
                next_values_holder = Some(PinnedLayer::Mapped(mm));
            }
            if values_on_disk {
                next_values_path = Some(out_path);
                next_path_is_owned = config.policy_only;
            }
        }
    }
    eprintln!("emit-full-mdp-layers-from-l0 all done out_dir={} ({})", out_dir.display(),
        if config.policy_only { "policies and layer0 retained; rolling intermediates removed" } else { "10 value layers written" });
    Ok(())
}

#[cfg(test)]
#[path = "greedy_emit_tests.rs"]
mod greedy_emit_tests;

// This allowance is per scheduled task and is carved out of the task lookup
// cap; zero preserves the default kernel and avoids constructing placement IDs.
fn greedy_action_cache_bytes_from_env() -> Result<usize> {
    let mib = match env::var("FULL_MDP_GREEDY_ACTION_CACHE_MIB") {
        Ok(value) => value.parse::<usize>().context("invalid FULL_MDP_GREEDY_ACTION_CACHE_MIB")?,
        Err(env::VarError::NotPresent) => 0,
        Err(error) => return Err(error).context("invalid FULL_MDP_GREEDY_ACTION_CACHE_MIB"),
    };
    mib.checked_mul(1024 * 1024).context("FULL_MDP_GREEDY_ACTION_CACHE_MIB byte overflow")
}

fn run_emit_full_mdp_layers_from_l0(args: &[String]) -> Result<()> {
    if args.len() < 6 || args.len() > 9 {
        eprintln!("usage: {} emit-full-mdp-layers-from-l0 <graph.bin> <pruned_layer_dir> <l0_values.f32> <out_dir> [gamma=1.0] [dense_cache_gib=32] [policy_out_dir]", args[0]);
        eprintln!("  l0_values.f32 = converged layer0 file, e.g. <l0_out_dir>/layer0.values.f32.bin");
        eprintln!("  policy_out_dir: also record the greedy argmax action per state as layer{{L}}.policy.u16.bin (for fixed-policy Policy Iteration)");
        eprintln!("  FULL_MDP_POLICY_ONLY=1: retain policies and layer0, rolling intermediate values only");
        eprintln!("  FULL_MDP_GREEDY_ACTION_CACHE_MIB=0: per-task greedy average reuse, carved out of the lookup cache (off by default)");
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let l0_values_path = PathBuf::from(&args[4]);
    let out_dir = PathBuf::from(&args[5]);
    let gamma: f32 = if args.len() >= 7 { args[6].parse().context("invalid gamma")? } else { 1.0 };
    let cache_gib: usize = if args.len() >= 8 { args[7].parse().context("invalid dense_cache_gib")? } else { 32 };
    let policy_out_dir: Option<PathBuf> = if args.len() >= 9 { Some(PathBuf::from(&args[8])) } else { None };
    if !(0.0..=1.0).contains(&gamma) { anyhow::bail!("gamma must be in [0,1], got {}", gamma); }
    let flag = |name: &str, default| env::var(name).ok()
        .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no").unwrap_or(default);
    let policy_only = flag("FULL_MDP_POLICY_ONLY", false);
    anyhow::ensure!(!policy_only || policy_out_dir.is_some(), "FULL_MDP_POLICY_ONLY=1 requires policy_out_dir");
    let greedy_action_cache_bytes = greedy_action_cache_bytes_from_env()?;
    let cache_cap_bytes = cache_gib.saturating_mul(1024).saturating_mul(1024).saturating_mul(1024);
    let config = GreedyEmitConfig {
        policy_only,
        ram_next_layer: flag("FULL_MDP_RAM_NEXT_LAYER", true),
        ram_next_layer_max_bytes: env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(96).saturating_mul(1024 * 1024 * 1024),
        ram_pair_max_bytes: env::var("FULL_MDP_RAM_PAIR_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(88).saturating_mul(1024 * 1024 * 1024),
        mmap_spill: flag("FULL_MDP_MMAP_SPILL", false),
        prewarm_field_targets: flag("FULL_MDP_PREWARM_FIELD_TARGETS", false),
        collect_cache_stats: flag("FULL_MDP_CACHE_STATS", false),
        collect_action_stats: flag("FULL_MDP_ACTION_STATS", false),
        sync_layer_outputs: flag("FULL_MDP_SYNC_LAYER", false),
        sparse_lookup_max_count: env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536),
        tuning: sweep_schedule::SweepTuning::from_env(cache_cap_bytes)?,
    };
    fs::create_dir_all(&out_dir)?;
    if let Some(pd) = policy_out_dir.as_ref() { fs::create_dir_all(pd)?; }
    eprintln!("emit-full-mdp-layers-from-l0 loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {} != {}", qstate_count, CHECK_QSTATE_COUNT); }
    let legal_placed = build_qfwd_legal_placed_table(&qfwd, qstate_count);
    let action_plan_started = Instant::now();
    let action_pre_reveal = if greedy_action_cache_bytes > 0 {
        Some(pre_reveal::PreRevealIndex::build(&q_index)?)
    } else { None };
    let action_placements = if let Some(pre) = action_pre_reveal.as_ref() {
        Some(action_avg_cache::ValidatedPlacements::build(&q_index, &qfwd, pre)?)
    } else { None };
    if let Some(placements) = action_placements.as_ref() {
        eprintln!("greedy-action-cache plan ready requested_bytes_per_task={} placement_resident_bytes={} build_seconds={:.9} boundary_reduced=0",
            greedy_action_cache_bytes, placements.storage_bytes(), action_plan_started.elapsed().as_secs_f64());
    }
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    let reset_values = load_layer0_reset_dense(&full_layers[0], &l0_values_path, qstate_count)?;
    let terminal_reset_avg = build_terminal_reset_avg_table(&qfwd, &reset_values, gamma, qstate_count);
    eprintln!("emit-full-mdp-layers-from-l0 qstates={} gamma={} dense_cache_gib={} ram_next_layer={} policy_only={} l0={} out={} policy_out={}",
        qstate_count, gamma, cache_gib, config.ram_next_layer, config.policy_only, l0_values_path.display(), out_dir.display(),
        policy_out_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "-".to_string()));
    emit_greedy_mdp_sweep_with_action_cache(&graph, &qfwd, &legal_placed, &terminal_reset_avg, &full_layers,
        &out_dir, policy_out_dir.as_deref(), qstate_count, two_line_field, gamma, &config,
        action_placements.as_ref().map(|placements| action_avg_cache::ActionSource {
            placements, requested_bytes: greedy_action_cache_bytes,
        }))
}

// Matrix-free Policy Iteration on the boundary map.
//
// Materializing M (extract-boundary-m) explodes: origins converge onto shared mid-layer
// (field, chain) states, so the per-origin wavefront costs ~|origins| x |per-origin reach|
// tuples (TBs) or forces origin batches that each re-run the argmax sweep. Instead we keep
// M implicit: one full argmax sweep records the greedy policy pi as a u16 action code per
// state (emit-full-mdp-layers-from-l0 with policy_out_dir), and then ONE application of
// M_pi (plus the constant c_pi) is a fixed-policy backward sweep that evaluates only the
// single chosen action per state -- ~an order of magnitude cheaper than the argmax sweep.
//
// T_pi(x) = c_pi + gamma * M_pi x  is affine in the boundary vector x (terminal resets are
// E[1 + gamma*x(child)], missing/pruned children contribute exactly 0, dead states are 0),
// so (I - gamma*M_pi) v = c_pi is solved with unrestarted GMRES whose matvec is
// A x = x - (T_pi(x) - T_pi(0)).

// Shared output buffer for RAM-accumulated layers (I/O diet): rayon chunk tasks write
// disjoint [chain_offset, chain_offset+count) ranges (block metas tile the layer without
// overlap), so raw concurrent writes never alias. This skips both the scratch-file write
// and the pin read-back for layers whose (current + pinned-next) sizes fit in RAM.
#[derive(Clone, Copy)]
struct SharedF32Out {
    ptr: *mut f32,
    len: usize,
}
unsafe impl Send for SharedF32Out {}
unsafe impl Sync for SharedF32Out {}

// Fixed-policy analogue of compute_full_mdp_chunk_f32: no argmax, evaluates the one action
// recorded in the policy file. Value expressions are IDENTICAL to the argmax path (same
// sum_dense_recs_scaled / denom, same terminal table), so for the greedy policy of x0 and
// input x0 the output is bit-identical to the argmax sweep's.
// Output goes to `ram_out` when Some (RAM-accumulated layer), else to `out_file`.
fn compute_fixed_policy_chunk_f32(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    policy_file: &File,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
) -> Result<FullMdpChunkResult> {
    compute_fixed_policy_chunk_f32_with_l9_source(
        layer, block_range, cur, policy_file, out_file, ram_out, graph, qfwd, terminal_reset_avg, next_layer, next_values_path, next_values_ram, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, None,
    )
}

fn compute_fixed_policy_chunk_f32_with_l9_source(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    policy_file: &File,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
) -> Result<FullMdpChunkResult> {
    compute_fixed_policy_chunk_f32_with_sources(
        layer, block_range, cur, policy_file, out_file, ram_out, graph, qfwd, terminal_reset_avg, next_layer, next_values_path, next_values_ram, gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count, two_line_field, direct_l9_source, None,
    )
}

fn compute_fixed_policy_chunk_f32_with_sources(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    policy_file: &File,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
    compiled_l9_source: Option<loop1_stream::PreparedL9Values<'_>>,
) -> Result<FullMdpChunkResult> {
    compute_fixed_policy_chunk_f32_with_action_cache(
        layer, block_range, cur, policy_file, out_file, ram_out, graph, qfwd,
        terminal_reset_avg, next_layer, next_values_path, next_values_ram,
        gamma, per_task_cache_cap, qstate_count, sparse_lookup_max_count,
        two_line_field, direct_l9_source, compiled_l9_source, None,
    )
}

fn compute_fixed_policy_chunk_f32_with_action_cache(
    layer: usize,
    block_range: std::ops::Range<usize>,
    cur: &layer_store::FieldChainLayer,
    policy_file: &File,
    out_file: Option<&File>,
    ram_out: Option<SharedF32Out>,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal_reset_avg: &[[f32; 7]],
    next_layer: Option<&layer_store::FieldChainLayer>,
    next_values_path: Option<&Path>,
    next_values_ram: Option<&[f32]>,
    gamma: f32,
    per_task_cache_cap: usize,
    qstate_count: usize,
    sparse_lookup_max_count: usize,
    two_line_field: FieldId,
    direct_l9_source: Option<terminal_l9::Source<'_>>,
    compiled_l9_source: Option<loop1_stream::PreparedL9Values<'_>>,
    action_source: Option<action_avg_cache::ActionSource<'_>>,
) -> Result<FullMdpChunkResult> {
    let collect_cache_stats: bool = env::var("FULL_MDP_CACHE_STATS").ok().map(|s| s != "0").unwrap_or(false);
    let collect_action_stats: bool = env::var("FULL_MDP_ACTION_STATS").ok().map(|s| s != "0").unwrap_or(false);
    if let Some(source) = action_source {
        source.placements.check_graph(qfwd, qstate_count)?;
    }
    let (mut action_cache, lookup_cache_cap) = match action_source.filter(|_| layer < 9) {
        Some(source) => action_avg_cache::ActionAverageCache::partition(
            source.requested_bytes, per_task_cache_cap, qstate_count,
        ),
        None => (None, per_task_cache_cap),
    };
    let mut lookup_opt = if layer < 9 {
        let nl = next_layer.expect("missing next layer");
        Some(if let Some(source) = compiled_l9_source.filter(|_| layer == 8) {
            NextValueLookup::CompiledL9(source.lookup(lookup_cache_cap, collect_cache_stats)?)
        } else if let Some(source) = direct_l9_source.filter(|_| layer == 8) {
            // This bounded membership/policy cache consumes the planner's
            // per-task cache allowance; no L9 value layer is pinned.
            NextValueLookup::TerminalL9(terminal_l9::Lookup::new(source, lookup_cache_cap, collect_cache_stats)?)
        } else if let Some(values) = next_values_ram {
            NextValueLookup::Ram(RamNextValueLookup::new(nl, values, lookup_cache_cap, collect_cache_stats, false)?)
        } else {
            NextValueLookup::File(DenseF32ValueLookup::open(
                nl,
                next_values_path.expect("missing next values path"),
                lookup_cache_cap,
                qstate_count,
                sparse_lookup_max_count,
                collect_cache_stats,
            )?)
        })
    } else { None };

    let mut stats = FullMdpLayerStats::default();
    let mut qids_buf = Vec::<u32>::new();
    let mut range_qid_bytes = Vec::<u8>::new();
    let range_chain_base = cur.read_block_range_chains_bytes_into_with_buf(block_range.start, block_range.end, &mut range_qid_bytes);
    let mut policy_bytes = Vec::<u8>::new();
    // Layer metadata is validated as a continuous, disjoint tiling on load.
    // Read the policy once for this whole group instead of once per field.
    let policy_chain_base = if block_range.is_empty() { 0 } else { cur.metas[block_range.start].chain_offset };
    if !block_range.is_empty() {
        let last = &cur.metas[block_range.end - 1];
        let end = last.chain_offset.checked_add(last.count as u64).context("policy range overflow")?;
        let bytes = end.checked_sub(policy_chain_base).and_then(|n| n.checked_mul(2)).context("policy range size overflow")?;
        policy_bytes.resize(usize::try_from(bytes)?, 0);
        read_exact_at_file(policy_file, &mut policy_bytes, policy_chain_base.checked_mul(2).context("policy offset overflow")?)?;
    }
    let mut out_bytes = Vec::<u8>::new();
    let mut chunk_out_bytes = Vec::<u8>::new();
    let mut chunk_start_off: Option<u64> = None;
    let mut chunk_next_off: u64 = 0;

    if ram_out.is_none() && out_file.is_none() && layer != 9 {
        anyhow::bail!("compute_fixed_policy_chunk_f32 needs an output sink (file or ram)");
    }
    for bi in block_range.clone() {
        let meta = &cur.metas[bi];
        let field = meta.field_id;
        decode_block_qids_from_range(cur, bi, range_chain_base, range_qid_bytes.as_slice(), &mut qids_buf);
        let qids = qids_buf.as_slice();
        let policy_start = usize::try_from((meta.chain_offset - policy_chain_base) * 2)?;
        if let Some(ro) = ram_out {
            let base = meta.chain_offset as usize;
            anyhow::ensure!(base + meta.count as usize <= ro.len, "ram_out overflow: block base {} + count {} > len {}", base, meta.count, ro.len);
        }
        out_bytes.clear();
        if ram_out.is_none() && out_file.is_some() { out_bytes.reserve(meta.count as usize * 4); }
        for (i, &qid) in qids.iter().enumerate() {
            stats.total_states += 1;
            let code = u16::from_le_bytes([policy_bytes[policy_start + i * 2], policy_bytes[policy_start + i * 2 + 1]]);
            let v = if code == POLICY_NONE_U16 {
                0.0f32
            } else {
                let placed = (code >> 8) as u8;
                let ei = (code & 0xFF) as usize;
                if placed as usize >= PIECE_COUNT {
                    anyhow::bail!("corrupt policy code {:#06x} layer={} field={} qid={}", code, layer, field, qid);
                }
                let edges = graph.edges(field, placed);
                if ei >= edges.len() {
                    anyhow::bail!("policy edge index {} out of range {} layer={} field={} placed={} qid={}", ei, edges.len(), layer, field, placed, qid);
                }
                let field2 = edges[ei];
                let recs = qfwd.records_for(qid, placed);
                if recs.is_empty() {
                    anyhow::bail!("policy places piece {} with no reveal records layer={} field={} qid={}", placed, layer, field, qid);
                }
                if (layer == 4 && field2 == two_line_field) || (layer == 9 && field2 == EMPTY_FIELD_ID) {
                    if layer == 4 { stats.terminal_2_edges += 1; } else { stats.terminal_4_edges += 1; }
                    terminal_reset_avg[qid as usize][placed as usize]
                } else if layer == 9 {
                    anyhow::bail!("layer 9 policy targets non-terminal field {} (field={} qid={})", field2, field, qid);
                } else {
                    let lookup = lookup_opt.as_mut().expect("missing next lookup");
                    let denom = recs.len() as f32;
                    if let Some(cache) = action_cache.as_mut() {
                        let source = action_source.expect("action cache needs a placement plan");
                        let zid = source.placements.zid_for_placement(qid, placed)
                            .context("fixed policy action has no validated pre-reveal ID")?;
                        if let Some(cached) = cache.get(field2, zid, recs.len()) {
                            if collect_action_stats { stats.missing_next_terms += u64::from(cached.missing_terms); }
                            cached.value
                        } else {
                            let (sum, missing) = lookup.sum_recs(field2, recs)?;
                            if collect_action_stats { stats.missing_next_terms += missing; }
                            let scaled = if gamma == 1.0 { sum } else { gamma * sum };
                            let value = scaled / denom;
                            cache.insert(field2, zid, action_avg_cache::CachedAverage {
                                value,
                                missing_terms: u8::try_from(missing).context("action average missing-term count exceeds u8")?,
                            });
                            value
                        }
                    } else {
                        sum_dense_recs_scaled(lookup, field2, recs, gamma, &mut stats, collect_action_stats)? / denom
                    }
                }
            };
            if v != 0.0 { stats.nonzero += 1; }
            stats.max_v = stats.max_v.max(v);
            stats.sum_v += v as f64;
            if let Some(ro) = ram_out {
                // SAFETY: bounds checked above; this task exclusively owns this block's range.
                unsafe { *ro.ptr.add(meta.chain_offset as usize + i) = v; }
            } else if out_file.is_some() {
                write_f32_le_buf(&mut out_bytes, v);
            }
        }
        if let Some(of) = out_file {
            append_block_output_and_maybe_flush(of, meta, out_bytes.as_slice(), &mut chunk_start_off, &mut chunk_next_off, &mut chunk_out_bytes)?;
        }
    }
    if let Some(of) = out_file {
        flush_chunk_output(of, chunk_start_off, &mut chunk_out_bytes)?;
    }
    if let Some(cache) = action_cache {
        let counters = cache.stats();
        stats.action_avg_hits = counters.hits;
        stats.action_avg_misses = counters.misses;
        stats.action_avg_collisions = counters.collisions;
        stats.action_avg_saved_reveals = counters.saved_reveals;
        stats.action_avg_cache_bytes = counters.payload_bytes;
    }
    if collect_cache_stats {
        if let Some(lookup) = lookup_opt.as_ref() {
            let ds = lookup.stats_snapshot();
            stats.dense_cache_hits += ds.cache_hits;
            stats.dense_cache_misses += ds.cache_misses;
            stats.dense_missing_fields += ds.missing_fields;
            stats.dense_evictions += ds.evictions;
            stats.dense_materialized_states += ds.materialized_states;
            stats.dense_blocks += ds.dense_blocks;
            stats.sparse_blocks += ds.sparse_blocks;
            stats.bitmap_blocks += ds.bitmap_blocks;
        }
    }
    Ok(FullMdpChunkResult { stats, blocks: block_range.end - block_range.start })
}

// One boundary-map application y = T_pi(x) as a full backward sweep 9..0 under the stored
// policy, writing per-layer scratch values (rolling RAM pin like emit) and returning the
// layer0 vector in file order. Reused by apply-fixed-policy-boundary and the GMRES driver.
struct FixedPolicySweep<'a> {
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    full_layers: &'a [layer_store::FieldChainLayer],
    policy_dir: &'a Path,
    scratch_dir: &'a Path,
    two_line_field: FieldId,
    qstate_count: usize,
    gamma: f32,
    cache_cap_bytes: usize,
    sparse_lookup_max_count: usize,
    block_chunk: usize,
    batch_chunks_env: Option<usize>,
    ram_next_layer: bool,
    ram_next_layer_max_bytes: u64,
    l0_reset_qids: &'a [u32],
    verbose_layers: bool,
    prepared_l9: Option<&'a loop1_prepared::PreparedL9<'a>>,
    prepared_l8: Option<&'a loop1_stream::CompiledL8<'a>>,
    action_source: Option<action_avg_cache::ActionSource<'a>>,
    operator_bindings: Option<&'a loop1_prepared::SourceBindings>,
    action_resident_bytes: u64,
}

// Pinned next-layer values: either an anonymous RAM buffer (classic path) or a
// file-backed read-only mapping (FULL_MDP_MMAP_SPILL=1 "own swap": the kernel keeps hot
// pages resident within the cgroup and reclaims cold ones back to the scratch file, so a
// file pages are reclaimable under memory pressure. Anonymous task caches and the
// GMRES basis still need their own allowance; an OS limit is separate from planning.
enum PinnedLayer {
    Ram(Vec<f32>),
    Mapped(memmap2::Mmap),
}

impl PinnedLayer {
    fn as_f32(&self) -> &[f32] {
        match self {
            PinnedLayer::Ram(v) => v.as_slice(),
            // Safety: mapping length is total_states*4 (set_len below), page-aligned base
            // satisfies f32 alignment, and the mapping is read-only for its lifetime.
            PinnedLayer::Mapped(m) => unsafe {
                std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len() / 4)
            },
        }
    }
    fn byte_len(&self) -> u64 {
        (self.as_f32().len() as u64) * 4
    }
    fn resident_bytes(&self) -> u64 {
        match self { Self::Ram(v) => v.len() as u64 * 4, Self::Mapped(_) => 0 }
    }
}

impl FixedPolicySweep<'_> {
    fn check_prepared_inputs(&self) -> Result<()> {
        if let Some(bindings) = self.operator_bindings { bindings.check()?; }
        if let Some(prepared) = self.prepared_l9 { prepared.check_inputs_unchanged()?; }
        if let Some(prepared) = self.prepared_l8 { prepared.check_inputs_unchanged()?; }
        Ok(())
    }

    // linear=false: y = T_pi(x)          (affine: terminal reward +1 included)
    // linear=true : y = gamma * M_pi * x (homogeneous: reward dropped, avoids GMRES cancellation)
    fn apply(&self, x: &[f32], linear: bool) -> Result<Vec<f32>> {
        self.apply_impl(x, linear, true)
    }

    // Independent acceptance path: force the original materialized sweep and
    // bypass compiled L9/L8 and the action-average memo.
    fn apply_original(&self, x: &[f32], linear: bool) -> Result<Vec<f32>> {
        self.apply_impl(x, linear, false)
    }

    fn apply_impl(&self, x: &[f32], linear: bool, use_prepared: bool) -> Result<Vec<f32>> {
        self.check_prepared_inputs()?;
        let layer0 = &self.full_layers[0];
        anyhow::ensure!(x.len() as u64 == layer0.total_states, "boundary vector length {} != layer0 states {}", x.len(), layer0.total_states);
        anyhow::ensure!(self.l0_reset_qids.len() <= x.len(), "reset block larger than boundary vector");
        let mut dense = vec![0.0f32; self.qstate_count];
        for (i, &q) in self.l0_reset_qids.iter().enumerate() {
            dense[q as usize] = x[i];
        }
        if let Some(prepared) = self.prepared_l9.filter(|_| use_prepared) {
            anyhow::ensure!(self.gamma == 1.0, "compiled L9 requires gamma=1");
            let terminals = prepared.terminal_values(&dense, !linear)?;
            drop(dense);
            let terminal_reset_avg = prepared.terminal_table(&terminals)?;
            let source = prepared.compiled.prepare_f32(&terminals)?;
            return self.apply_terminal_impl(&terminal_reset_avg, Some(source), Some(&terminals), false);
        }
        let terminal_reset_avg = build_terminal_reset_avg_table_opt(self.qfwd, &dense, self.gamma, self.qstate_count, !linear);
        drop(dense);
        self.apply_terminal_impl(&terminal_reset_avg, None, None, !use_prepared)
    }

    // A reduced boundary map supplies terminal expectations directly. The
    // backward kernels still apply gamma to each nonterminal placement.
    pub(crate) fn apply_terminal(&self, terminal_reset_avg: &[[f32; 7]]) -> Result<Vec<f32>> {
        self.apply_terminal_impl(terminal_reset_avg, None, None, false)
    }

    fn apply_terminal_impl(&self, terminal_reset_avg: &[[f32; 7]], compiled_l9_source: Option<loop1_stream::PreparedL9Values<'_>>, compiled_terminals: Option<&[f32]>, force_materialized_l9: bool) -> Result<Vec<f32>> {
        // Check before creating, truncating, or writing any scratch layer.
        self.check_prepared_inputs()?;
        let prepared_resident_bytes = self.prepared_l9.map_or(0, |prepared| prepared.storage_bytes())
            .checked_add(self.prepared_l8.map_or(0, |prepared| prepared.storage_bytes()))
            .and_then(|bytes| bytes.checked_add(self.action_resident_bytes))
            .context("prepared operator resident memory overflow")?;
        let active_l8 = self.prepared_l8.filter(|_| compiled_terminals.is_some() && !force_materialized_l9);
        anyhow::ensure!(terminal_reset_avg.len() == self.qstate_count,
            "terminal table length {} != qstate count {}", terminal_reset_avg.len(), self.qstate_count);
        anyhow::ensure!(terminal_reset_avg.iter().flatten().all(|value| value.is_finite()),
            "terminal table contains a non-finite value");
        let apply_start = Instant::now();
        let tuning = sweep_schedule::SweepTuning::from_env(self.cache_cap_bytes)?;
        let layer0 = &self.full_layers[0];
        let direct_l9 = compiled_l9_source.is_none() && !force_materialized_l9 && terminal_l9::enabled_from_env();
        let direct_l9_policy = if direct_l9 {
            let path = mdp_policy_u16_path(self.policy_dir, 9);
            let file = File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
            let expected = self.full_layers[9].total_states.checked_mul(2).context("L9 policy size overflow")?;
            anyhow::ensure!(file.metadata()?.len() == expected, "policy size mismatch {}: expected {} bytes", path.display(), expected);
            Some(file)
        } else { None };

        // I/O diet: a layer whose size plus the currently pinned next layer fits the pair
        // budget is accumulated directly in RAM (no scratch write, no pin read-back). With
        // the default 88 GiB budget that is L9/L8/L7/L2/L1/L0; the big middle layers
        // (L6/L5/L4/L3, adjacent pairs 106-154 GiB) still spill to scratch files.
        let ram_pair_max_bytes: u64 = env::var("FULL_MDP_RAM_PAIR_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(88).saturating_mul(1024 * 1024 * 1024);
        // FULL_MDP_MMAP_SPILL=1: every layer buffer (output + next-layer pin) is a mapping
        // of its scratch file instead of anonymous RAM. Residency is then windowed by the
        // kernel inside the cgroup limit — our own swap, no kernel swap needed.
        let mmap_spill: bool = env::var("FULL_MDP_MMAP_SPILL").ok()
            .map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no")
            .unwrap_or(false);
        let mut next_values_ram_holder: Option<PinnedLayer> = None;
        let mut result0: Option<Vec<f32>> = None;
        for layer in (0usize..=9).rev() {
            let layer_start = Instant::now();
            let cur = &self.full_layers[layer];
            if layer == 9 && active_l8.is_some() {
                if self.verbose_layers {
                    eprintln!("  fixed-policy layer=9 states={} sink=compiled-l8-leaves value_bytes=0 elapsed={:.3}s", cur.total_states, layer_start.elapsed().as_secs_f64());
                }
                continue;
            }
            if layer == 8 {
                if let Some(compiled) = active_l8 {
                    next_values_ram_holder = loop1_prepared_l8::materialize(
                        compiled, compiled_terminals.expect("compiled L8 terminal values"), self,
                        &tuning, prepared_resident_bytes, mmap_spill, ram_pair_max_bytes,
                    )?;
                    continue;
                }
            }
            if layer == 9 && compiled_l9_source.is_some() {
                let source = compiled_l9_source.expect("compiled L9 source");
                let value_bytes = cur.total_states.checked_mul(4)
                    .context("compiled L9 f32 size overflow")?;
                let gather_to_ram = self.ram_next_layer && !mmap_spill
                    && value_bytes <= self.ram_next_layer_max_bytes
                    && value_bytes <= ram_pair_max_bytes
                    && tuning.can_hold_layers(value_bytes, prepared_resident_bytes, self.qstate_count);
                if gather_to_ram {
                    // Preserve the existing L8 RAM lookup: gather each shared
                    // formula through its compact ID once, without replaying
                    // L9 policy decoding or rebuilding membership caches.
                    next_values_ram_holder = Some(PinnedLayer::Ram(source.materialize_f32()?));
                }
                eprintln!("  fixed-policy layer=9 states={} sink={} value_bytes={} elapsed={:.3}s", cur.total_states,
                    if gather_to_ram { "compiled-ram-gather" } else { "compiled-rank-direct" },
                    if gather_to_ram { value_bytes } else { 0 }, layer_start.elapsed().as_secs_f64());
                continue;
            }
            if layer == 9 && direct_l9 {
                // Validate every stored L9 action, including states never reached
                // by an L8 policy. No L9 value buffer or scratch file is needed.
                terminal_l9::validate_membership(terminal_l9::Source {
                    layer: cur, graph: self.graph, qfwd: self.qfwd, terminal_reset_avg,
                    mode: terminal_l9::Mode::Fixed(direct_l9_policy.as_ref().expect("direct L9 policy")),
                })?;
                let plan = tuning.plan(&cur.metas, 0, prepared_resident_bytes, rayon::current_num_threads(), self.qstate_count)?;
                let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
                let mut states = 0u64;
                for batch in plan.ranges.chunks(plan.workers.saturating_mul(4).max(1)) {
                    let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|range| {
                        compute_fixed_policy_chunk_f32_with_l9_source(
                            9, range.clone(), cur, direct_l9_policy.as_ref().expect("direct L9 policy"),
                            None, None, self.graph, self.qfwd, terminal_reset_avg, None, None, None,
                            self.gamma, plan.per_task_cache_bytes, self.qstate_count,
                            self.sparse_lookup_max_count, self.two_line_field, None,
                        )
                    }).collect());
                    for result in results { states += result?.stats.total_states; }
                }
                anyhow::ensure!(states == cur.total_states, "direct L9 fixed-policy state count mismatch");
                if self.verbose_layers {
                    eprintln!("  fixed-policy layer=9 states={} sink=validation value_bytes=0 elapsed={:.3}s", states, layer_start.elapsed().as_secs_f64());
                }
                continue;
            }
            let out_path = mdp_f32_value_path(self.scratch_dir, layer);
            let expected_layer_bytes = cur.total_states.checked_mul(4).context("fixed-policy f32 output size overflow")?;
            let next_pinned_bytes = next_values_ram_holder.as_ref().map_or(0u64, |p| p.byte_len());
            let next_resident_bytes = next_values_ram_holder.as_ref().map_or(0u64, |p| p.resident_bytes())
                .checked_add(prepared_resident_bytes).context("prepared L9 resident memory overflow")?;
            let mut use_ram_out = !mmap_spill && self.ram_next_layer
                && expected_layer_bytes.saturating_add(next_pinned_bytes) <= ram_pair_max_bytes
                && tuning.can_hold_layers(expected_layer_bytes, next_resident_bytes, self.qstate_count);
            // Plan before allocating/truncating a sink. An indivisible large block
            // can require more scratch than the initial RAM fit estimate permits.
            let rayon_threads = rayon::current_num_threads();
            let plan = match tuning.plan(&cur.metas, if use_ram_out { expected_layer_bytes } else { 0 },
                next_resident_bytes, rayon_threads, self.qstate_count) {
                Ok(plan) => plan,
                Err(_) if use_ram_out => {
                    use_ram_out = false;
                    tuning.plan(&cur.metas, 0, next_resident_bytes, rayon_threads, self.qstate_count)?
                }
                Err(err) => return Err(err),
            };
            let (out_file_holder, mut ram_buf, mut mmap_buf): (Option<Arc<File>>, Option<Vec<f32>>, Option<memmap2::MmapMut>) = if expected_layer_bytes == 0 {
                // Empty files cannot be mapped. Keep a placeholder for file-only
                // predecessors, and an empty pin for RAM/mmap predecessors.
                File::create(&out_path)?;
                (None, Some(Vec::new()), None)
            } else if mmap_spill {
                let f = OpenOptions::new().read(true).write(true).create(true).open(&out_path)
                    .with_context(|| format!("failed to open {}", out_path.display()))?;
                // Truncate-then-extend guarantees hole-backed zero pages (same semantics as
                // the vec![0.0] path) even when a stale scratch file exists.
                f.set_len(0)?;
                f.set_len(expected_layer_bytes)?;
                let mm = unsafe { memmap2::MmapMut::map_mut(&f) }
                    .with_context(|| format!("failed to mmap {}", out_path.display()))?;
                (None, None, Some(mm))
            } else if use_ram_out {
                (None, Some(vec![0.0f32; cur.total_states as usize]), None)
            } else {
                let f = File::create(&out_path).with_context(|| format!("failed to create {}", out_path.display()))?;
                f.set_len(expected_layer_bytes)?;
                (Some(Arc::new(f)), None, None)
            };
            let ram_out: Option<SharedF32Out> = ram_buf.as_mut()
                .map(|b| SharedF32Out { ptr: b.as_mut_ptr(), len: b.len() })
                .or_else(|| mmap_buf.as_mut().map(|m| SharedF32Out { ptr: m.as_mut_ptr() as *mut f32, len: (expected_layer_bytes / 4) as usize }));
            let out_file_ref: Option<&File> = out_file_holder.as_deref();
            let policy_path = mdp_policy_u16_path(self.policy_dir, layer);
            let policy_file = File::open(&policy_path).with_context(|| format!("failed to open {}", policy_path.display()))?;
            let expected_policy_bytes = cur.total_states.checked_mul(2).context("policy size overflow")?;
            let policy_actual = policy_file.metadata()?.len();
            anyhow::ensure!(policy_actual == expected_policy_bytes, "policy size mismatch {}: got {}, expected {}", policy_path.display(), policy_actual, expected_policy_bytes);

            let next_layer_ref = if layer < 9 { Some(&self.full_layers[layer + 1]) } else { None };
            let next_values_path_holder = if layer < 9 { Some(mdp_f32_value_path(self.scratch_dir, layer + 1)) } else { None };
            let next_values_ref = next_values_path_holder.as_deref();
            let next_values_ram_ref: Option<&[f32]> = if layer < 9 { next_values_ram_holder.as_ref().map(|p| p.as_f32()) } else { None };
            let direct_l9_source = if direct_l9 && layer == 8 {
                Some(terminal_l9::Source {
                    layer: &self.full_layers[9], graph: self.graph, qfwd: self.qfwd,
                    terminal_reset_avg,
                    mode: terminal_l9::Mode::Fixed(direct_l9_policy.as_ref().expect("direct L9 policy")),
                })
            } else { None };

            let batch_chunks: usize = self.batch_chunks_env.unwrap_or_else(|| rayon_threads.saturating_mul(4).max(1)).max(1);
            let pool = rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()?;
            let ranges = &plan.ranges;
            let mut layer_stats = FullMdpLayerStats::default();
            for batch in ranges.chunks(batch_chunks) {
                let results: Vec<Result<FullMdpChunkResult>> = pool.install(|| batch.par_iter().map(|r| {
                    compute_fixed_policy_chunk_f32_with_action_cache(
                        layer,
                        r.clone(),
                        cur,
                        &policy_file,
                        out_file_ref,
                        ram_out,
                        self.graph,
                        self.qfwd,
                        terminal_reset_avg,
                        next_layer_ref,
                        next_values_ref,
                        next_values_ram_ref,
                        self.gamma,
                        plan.per_task_cache_bytes,
                        self.qstate_count,
                        self.sparse_lookup_max_count,
                        self.two_line_field,
                        direct_l9_source,
                        compiled_l9_source.filter(|_| layer == 8 && next_values_ram_holder.is_none()),
                        self.action_source.filter(|_| !force_materialized_l9),
                    )
                }).collect());
                for res in results { layer_stats.add(&res?.stats); }
            }
            anyhow::ensure!(layer_stats.total_states == cur.total_states, "fixed-policy state count mismatch layer={}: computed {}, expected {}", layer, layer_stats.total_states, cur.total_states);
            if !use_ram_out {
                let actual = fs::metadata(&out_path)?.len();
                anyhow::ensure!(actual == expected_layer_bytes, "fixed-policy output size mismatch layer={}: got {}, expected {}", layer, actual, expected_layer_bytes);
            }
            if self.verbose_layers || env::var("FULL_MDP_CACHE_STATS").is_ok_and(|s| s != "0") || env::var("FULL_MDP_ACTION_STATS").is_ok_and(|s| s != "0") {
                eprintln!("  fixed-policy layer={} states={} nonzero={} sink={} hits={} misses={} promotions={} sparse_inserts={} bitmap_inserts={} evictions={} materialized_states={} actions={} missing_terms={} workers={} task_cache_bytes={} task_buffer_bytes={} groups={} action_avg_hits={} action_avg_misses={} action_avg_collisions={} action_avg_saved_reveals={} action_avg_cache_bytes={} elapsed={:.3}s",
                    layer, layer_stats.total_states, layer_stats.nonzero, if mmap_buf.is_some() { "mmap" } else if use_ram_out { "ram" } else { "file" },
                    layer_stats.dense_cache_hits, layer_stats.dense_cache_misses, layer_stats.dense_blocks, layer_stats.sparse_blocks, layer_stats.bitmap_blocks, layer_stats.dense_evictions,
                    layer_stats.dense_materialized_states, layer_stats.action_count, layer_stats.missing_next_terms,
                    plan.workers, plan.per_task_cache_bytes, plan.max_working_bytes, ranges.len(),
                    layer_stats.action_avg_hits, layer_stats.action_avg_misses, layer_stats.action_avg_collisions,
                    layer_stats.action_avg_saved_reveals, layer_stats.action_avg_cache_bytes,
                    layer_start.elapsed().as_secs_f64());
            }

            if layer == 0 {
                result0 = if let Some(mm) = mmap_buf.take() {
                    // Layer 0 is boundary-sized (few MB) — copy out of the mapping.
                    Some(unsafe { std::slice::from_raw_parts(mm.as_ptr() as *const f32, mm.len() / 4) }.to_vec())
                } else {
                    ram_buf.take()
                };
            } else if expected_layer_bytes == 0 {
                next_values_ram_holder = Some(PinnedLayer::Ram(Vec::new()));
            } else if mmap_spill {
                // Own-swap pin: reuse the freshly written mapping read-only. Kick off async
                // writeback so dirty pages become cheaply-evictable clean pages early.
                next_values_ram_holder = None;
                let mm = mmap_buf.take().expect("mmap_spill layer buffer missing");
                let _ = mm.flush_async();
                let ro = mm.make_read_only().context("make_read_only failed on layer mapping")?;
                next_values_ram_holder = Some(PinnedLayer::Mapped(ro));
            } else if self.ram_next_layer {
                // Drop the old pin first, then install the new layer (RAM buffer moves in
                // free of charge; file layers are read back once).
                next_values_ram_holder = None;
                if let Some(buf) = ram_buf.take() {
                    next_values_ram_holder = Some(PinnedLayer::Ram(buf));
                } else if expected_layer_bytes <= self.ram_next_layer_max_bytes && tuning.can_hold_layers(prepared_resident_bytes, expected_layer_bytes, self.qstate_count) {
                    let loaded = load_f32_values_vec(&out_path, cur.total_states)
                        .with_context(|| format!("failed to pin layer {} into RAM from {}", layer, out_path.display()))?;
                    next_values_ram_holder = Some(PinnedLayer::Ram(loaded));
                } else if self.verbose_layers {
                    eprintln!("  fixed-policy layer={} ram_pin_skipped bytes={}", layer, expected_layer_bytes);
                }
            } else {
                // A previous empty layer may have installed an empty pin even
                // in file-only mode. It must not masquerade as this layer.
                next_values_ram_holder = None;
            }
        }
        let y = match result0 {
            Some(v) => v,
            None => load_f32_values_vec(&mdp_f32_value_path(self.scratch_dir, 0), layer0.total_states)?,
        };
        self.check_prepared_inputs()?;
        eprintln!("  fixed-policy apply done elapsed={:.1}s", apply_start.elapsed().as_secs_f64());
        Ok(y)
    }
}

// Shared loader for the fixed-policy subcommands. Returns the reset-block qids of layer0
// (the block whose entries feed terminal resets), chosen like load_layer0_reset_dense.
fn layer0_reset_block_qids(layer0: &layer_store::FieldChainLayer) -> Result<Vec<u32>> {
    anyhow::ensure!(!layer0.metas.is_empty(), "empty layer0 membership");
    let bi = if layer0.metas.len() == 1 {
        0usize
    } else if let Some(pos) = layer0.metas.iter().position(|m| m.field_id == EMPTY_FIELD_ID) {
        pos
    } else if let Some(pos) = layer0.metas.iter().position(|m| m.field_id == 0) {
        pos
    } else {
        anyhow::bail!("could not find layer0 empty/reset field block");
    };
    anyhow::ensure!(bi == 0, "layer0 reset block is not the first block; boundary vector order would not match file order");
    Ok(layer0.read_block_chains(bi))
}

fn load_boundary_vector(arg: &str, states: u64) -> Result<Vec<f32>> {
    if arg == "zero" {
        Ok(vec![0.0f32; states as usize])
    } else {
        load_f32_values_vec(Path::new(arg), states)
    }
}

fn write_f32_vec_file(path: &Path, v: &[f32]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path).with_context(|| format!("failed to create {}", path.display()))?);
    for &x in v { out.write_all(&x.to_le_bytes())?; }
    out.flush()?;
    Ok(())
}

fn loop1_operator_settings(gamma: f32) -> Result<(bool, bool, usize)> {
    let compiled_l8 = loop1_prepared_l8::enabled_from_env();
    let compiled_l9 = loop1_prepared::enabled_from_env() || compiled_l8;
    anyhow::ensure!(!compiled_l9 || gamma == 1.0,
        "PI_LOOP1_COMPILED_L9 requires gamma=1 (also enabled by PI_LOOP1_COMPILED_L8)");
    let action_mib = match env::var("PI_LOOP1_ACTION_CACHE_MIB") {
        Ok(value) => value.parse::<usize>().context("invalid PI_LOOP1_ACTION_CACHE_MIB")?,
        Err(env::VarError::NotPresent) => 0,
        Err(error) => return Err(error).context("invalid PI_LOOP1_ACTION_CACHE_MIB"),
    };
    let action_bytes = action_mib.checked_mul(1024 * 1024)
        .context("PI_LOOP1_ACTION_CACHE_MIB byte overflow")?;
    Ok((compiled_l9, compiled_l8, action_bytes))
}

fn publish_finite_f32_atomic(path: &Path, values: &[f32]) -> Result<()> {
    publish_finite_f32_atomic_checked(path, values, || Ok(()))
}

fn publish_finite_f32_atomic_checked(path: &Path, values: &[f32], check: impl FnOnce() -> Result<()>) -> Result<()> {
    anyhow::ensure!(values.iter().all(|value| value.is_finite()), "nonfinite fixed-policy boundary output");
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let pending = path.with_extension(format!("pending.{}.{}", std::process::id(), stamp));
    // Only clean up a temporary file that this invocation created.
    let file = OpenOptions::new().write(true).create_new(true).open(&pending)?;
    let result = (|| -> Result<()> {
        let mut writer = BufWriter::new(file);
        for &value in values { writer.write_all(&value.to_le_bytes())?; }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        check()?;
        fs::rename(&pending, path).with_context(|| format!("publish fixed-policy boundary {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&pending); }
    result
}

// y = T_pi(x) once. usage:
//   apply-fixed-policy-boundary <graph.bin> <pruned_layer_dir> <policy_dir> <x.f32|zero> <y_out.f32> <scratch_dir> [gamma=1.0] [dense_cache_gib=16] [compare.f32]
// compare.f32: require finite, bit-identical values before atomic publication
// (e.g. the argmax emit's layer0 output for the same boundary input).
fn run_apply_fixed_policy_boundary(args: &[String]) -> Result<()> {
    if args.len() < 8 || args.len() > 11 {
        eprintln!("usage: {} apply-fixed-policy-boundary <graph.bin> <pruned_layer_dir> <policy_dir> <x.f32|zero> <y_out.f32> <scratch_dir> [gamma=1.0] [dense_cache_gib=16] [compare.f32]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let x_arg = args[5].clone();
    let y_out_path = PathBuf::from(&args[6]);
    let scratch_dir = PathBuf::from(&args[7]);
    let gamma: f32 = if args.len() >= 9 { args[8].parse().context("invalid gamma")? } else { 1.0 };
    let cache_gib: usize = if args.len() >= 10 { args[9].parse().context("invalid dense_cache_gib")? } else { 16 };
    let compare_path: Option<PathBuf> = if args.len() >= 11 { Some(PathBuf::from(&args[10])) } else { None };
    if !(0.0..=1.0).contains(&gamma) { anyhow::bail!("gamma must be in [0,1], got {}", gamma); }
    let (compiled_l9_enabled, compiled_l8_enabled, action_cache_bytes) = loop1_operator_settings(gamma)?;
    let whole_settings = loop1_whole_operator::Settings::from_env(gamma, compiled_l8_enabled, action_cache_bytes)?;
    let compiled_l9_enabled = compiled_l9_enabled || whole_settings.is_some_and(|settings| settings.l9_tail);
    let mut input_paths = vec![graph_path.clone()];
    for layer in 0usize..=9 {
        let prefix = q_pruned_prefix(&pruned_dir, layer);
        input_paths.extend([prefix.with_extension("field_index.bin"), prefix.with_extension("chains.u24.bin"), mdp_policy_u16_path(&policy_dir, layer)]);
    }
    input_paths.extend(compare_path.iter().cloned());
    let initial_path = if x_arg == "zero" { None } else { Some(Path::new(&x_arg)) };
    loop1_prepared::reject_solver_aliases(&input_paths, initial_path, &y_out_path, &scratch_dir)?;
    let whole_guard = if whole_settings.is_some() {
        let mut bound_paths = input_paths.clone();
        bound_paths.extend(initial_path.map(Path::to_path_buf));
        Some(loop1_whole_operator::InputGuard::capture(bound_paths)?)
    } else { None };
    let mut source_bindings = if compiled_l9_enabled || action_cache_bytes > 0 {
        let mut bound_paths = input_paths.clone();
        bound_paths.extend(initial_path.map(Path::to_path_buf));
        Some(loop1_prepared::SourceBindings::capture_paths(bound_paths)?)
    } else { None };
    fs::create_dir_all(&scratch_dir)?;

    eprintln!("apply-fixed-policy-boundary loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    drop(old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {} != {}", qstate_count, CHECK_QSTATE_COUNT); }
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    let l0_reset_qids = layer0_reset_block_qids(&full_layers[0])?;
    if whole_settings.is_some() { loop1_whole_operator::validate_root_layout(&graph, &full_layers, &l0_reset_qids)?; }
    let x = load_boundary_vector(&x_arg, full_layers[0].total_states)?;
    anyhow::ensure!(x.iter().all(|value| value.is_finite()), "nonfinite fixed-policy boundary input");
    let whole_start = Instant::now();
    let pre_reveal = if compiled_l9_enabled || action_cache_bytes > 0 || whole_settings.is_some() {
        Some(pre_reveal::PreRevealIndex::build(&q_index)?)
    } else { None };
    let action_placements = if action_cache_bytes > 0 {
        Some(action_avg_cache::ValidatedPlacements::build(&q_index, &qfwd,
            pre_reveal.as_ref().expect("action cache pre-reveal index"))?)
    } else { None };
    let prepared_l9 = if compiled_l9_enabled {
        Some(loop1_prepared::PreparedL9::build(&graph, &q_index, &qfwd,
            pre_reveal.as_ref().expect("compiled L9 pre-reveal index"), &full_layers,
            &policy_dir, source_bindings.take().expect("compiled L9 source bindings"))?)
    } else { None };
    let prepared_l8 = if compiled_l8_enabled {
        Some(loop1_prepared_l8::build(&graph, &q_index, &qfwd,
            pre_reveal.as_ref().expect("compiled L8 pre-reveal index"), &full_layers[8],
            &mdp_policy_u16_path(&policy_dir, 8),
            &prepared_l9.as_ref().expect("compiled L8 requires L9").compiled)?)
    } else { None };
    let action_resident_bytes = if !compiled_l9_enabled && action_cache_bytes > 0 {
        action_placements.as_ref().expect("action placement plan").storage_bytes()
    } else { 0 };
    let whole_policy = if let Some(guard) = whole_guard.as_ref() {
        Some(loop1_whole_operator::validated_policy(&graph, &qfwd,
            pre_reveal.as_ref().expect("whole DAG formula index"), &full_layers,
            &policy_dir, two_line_field, guard)?)
    } else { None };
    let whole_terminal = if whole_settings.is_some() {
        Some(loop1_terminal::TerminalPlan::build(&q_index, &qfwd,
            pre_reveal.as_ref().expect("whole DAG formula index"))?)
    } else { None };
    let whole_plan = if let Some(settings) = whole_settings {
        Some(loop1_whole_operator::compile(whole_policy.as_ref().expect("whole DAG policy"),
            settings, prepared_l9.as_ref().map(|prepared| &prepared.compiled),
            whole_guard.as_ref().expect("whole DAG input guard"))?)
    } else { None };
    let whole_operator = if let Some(plan) = whole_plan.as_ref() {
        Some(loop1_whole_operator::FullBoundaryDag::new(plan,
            whole_policy.as_ref().expect("whole DAG policy"),
            whole_terminal.as_ref().expect("whole DAG terminal plan"), &qfwd, &l0_reset_qids,
            whole_guard.as_ref().expect("whole DAG input guard"),
            whole_settings.expect("whole DAG settings").reference)?)
    } else { None };
    if whole_operator.is_some() {
        eprintln!("whole-dag startup seconds={:.6} includes_formulas_tail_fullvalidation_plan=1 excludes_graph_qstate_load=1", whole_start.elapsed().as_secs_f64());
    }

    let sweep = FixedPolicySweep {
        graph: &graph,
        qfwd: &qfwd,
        full_layers: &full_layers,
        policy_dir: &policy_dir,
        scratch_dir: &scratch_dir,
        two_line_field,
        qstate_count,
        gamma,
        cache_cap_bytes: cache_gib.saturating_mul(1024).saturating_mul(1024).saturating_mul(1024),
        sparse_lookup_max_count: env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536),
        block_chunk: env::var("FULL_MDP_BLOCK_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512).max(1),
        batch_chunks_env: env::var("FULL_MDP_BATCH_CHUNKS").ok().and_then(|s| s.parse().ok()),
        ram_next_layer: env::var("FULL_MDP_RAM_NEXT_LAYER").ok().map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no").unwrap_or(true),
        ram_next_layer_max_bytes: env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(96).saturating_mul(1024 * 1024 * 1024),
        l0_reset_qids: &l0_reset_qids,
        verbose_layers: true,
        prepared_l9: prepared_l9.as_ref(),
        prepared_l8: prepared_l8.as_ref(),
        action_source: action_placements.as_ref().map(|placements| action_avg_cache::ActionSource {
            placements, requested_bytes: action_cache_bytes,
        }),
        operator_bindings: source_bindings.as_ref(),
        action_resident_bytes,
    };
    // Affine sweep: must reproduce the argmax sweep bit-for-bit (gate B).
    let y = if let Some(operator) = whole_operator.as_ref() {
        operator.apply_verified(&x, false)?
    } else { sweep.apply(&x, false)? };
    anyhow::ensure!(y.iter().all(|value| value.is_finite()), "nonfinite fixed-policy boundary output");
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in &y { mn = mn.min(v); mx = mx.max(v); sum += v as f64; }
    if let Some(cp) = compare_path {
        let refv = load_f32_values_vec(&cp, full_layers[0].total_states)?;
        anyhow::ensure!(refv.iter().all(|value| value.is_finite()), "nonfinite fixed-policy comparison value");
        let mut max_abs = 0.0f32;
        let mut n_diff = 0u64;
        for i in 0..y.len() {
            let d = (y[i] - refv[i]).abs();
            if y[i].to_bits() != refv[i].to_bits() { n_diff += 1; }
            max_abs = max_abs.max(d);
        }
        eprintln!("apply-fixed-policy-boundary compare vs {}: max_abs_diff={:.10} differing={}({:.4}%)", cp.display(), max_abs, n_diff, 100.0 * n_diff as f64 / y.len().max(1) as f64);
        anyhow::ensure!(n_diff == 0, "fixed-policy boundary comparison failed: {n_diff} bitwise differences; output was not published");
    }
    sweep.check_prepared_inputs()?;
    if let Some(operator) = whole_operator.as_ref() { operator.check_inputs_unchanged()?; }
    if let Some(operator) = whole_operator.as_ref() {
        publish_finite_f32_atomic_checked(&y_out_path, &y, || operator.check_inputs_unchanged())?;
    } else { publish_finite_f32_atomic(&y_out_path, &y)?; }
    eprintln!("apply-fixed-policy-boundary done y={} min={:.8} max={:.8} avg={:.8}", y_out_path.display(), mn, mx, sum / y.len().max(1) as f64);
    Ok(())
}

// Policy-evaluation step of PI: solve (I - gamma*M_pi) v = c_pi with unrestarted GMRES,
// matvec = one fixed-policy sweep. Krylov basis is f64 in RAM (n = 817,740 -> ~6.5 MB per
// vector). x0 warm-starts from the current boundary V. usage:
//   pi-policy-eval-gmres <graph.bin> <pruned_layer_dir> <policy_dir> <x0.f32|zero> <out_v.f32> <scratch_dir> [gamma=1.0] [max_iters=200] [rel_tol=1e-6] [dense_cache_gib=16]
// Reconstruct the current GMRES iterate v = x0 + V_m y from the (already Givens-triangularized)
// Hessenberg h_cols, the RHS-image g, and the Krylov basis. No matvec/sweep — pure back-sub +
// AXPY, so it is cheap enough to call every iteration for a snapshot. h_cols[k][i] holds R[i][k].
fn gmres_reconstruct_solution(
    x0: &[f64],
    basis: &[Vec<f64>],
    h_cols: &[Vec<f64>],
    g: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let m = h_cols.len();
    let mut y = vec![0.0f64; m];
    for i in (0..m).rev() {
        let mut s = g[i];
        for k in i + 1..m {
            s -= h_cols[k][i] * y[k];
        }
        anyhow::ensure!(h_cols[i][i] != 0.0, "singular Hessenberg diagonal at {}", i);
        y[i] = s / h_cols[i][i];
    }
    let mut v = x0.to_vec();
    for (i, &yi) in y.iter().enumerate() {
        let qi = &basis[i];
        for k in 0..n {
            v[k] += yi * qi[k];
        }
    }
    Ok(v)
}

// Publish only the finite, accepted f32 vector consumed by the next PI round.
// A failed verification or interrupted write leaves an existing accepted output intact.
fn gmres_accept_and_publish(path: &Path, values: &[f32], applied: &[f32], tolerance: f64) -> Result<f64> {
    gmres_accept_and_publish_checked(path, values, applied, tolerance, || Ok(()))
}

fn gmres_bellman_inf(values: &[f32], applied: &[f32]) -> Result<f64> {
    anyhow::ensure!(values.len() == applied.len() && !values.is_empty(), "policy residual length mismatch");
    let mut residual = 0.0f64;
    for (&value, &next) in values.iter().zip(applied) {
        anyhow::ensure!(value.is_finite() && next.is_finite(), "nonfinite policy value or residual");
        residual = residual.max((next as f64 - value as f64).abs());
    }
    Ok(residual)
}

fn gmres_accept_and_publish_checked(path: &Path, values: &[f32], applied: &[f32], tolerance: f64, check: impl FnOnce() -> Result<()>) -> Result<f64> {
    anyhow::ensure!(tolerance.is_finite() && tolerance >= 0.0, "invalid policy residual tolerance");
    let residual = gmres_bellman_inf(values, applied)?;
    anyhow::ensure!(residual <= tolerance,
        "pi-policy-eval-gmres NOT ACCEPTED: final |T(v)-v|_inf={residual:.6e} > accept_tol={tolerance:.1e}; output was not published");
    publish_finite_f32_atomic_checked(path, values, check)?;
    Ok(residual)
}

#[cfg(test)]
mod gmres_acceptance_tests {
    use super::gmres_accept_and_publish;

    #[test]
    fn rejected_or_nonfinite_candidate_never_replaces_accepted_output() {
        let path = std::env::temp_dir().join(format!("zxcl-gmres-accept-{}", std::process::id()));
        std::fs::write(&path, b"old accepted value").unwrap();
        for (v, tv) in [(1.0, 2.0), (f32::NAN, 1.0), (1.0, f32::INFINITY)] {
            assert!(gmres_accept_and_publish(&path, &[v], &[tv], 1e-3).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"old accepted value");
        }
        assert_eq!(gmres_accept_and_publish(&path, &[2.0], &[2.0], 1e-3).unwrap(), 0.0);
        assert_eq!(std::fs::read(&path).unwrap(), 2.0f32.to_le_bytes());
        std::fs::remove_file(path).unwrap();
    }
}

fn run_pi_policy_eval_gmres(args: &[String]) -> Result<()> {
    if args.len() < 8 || args.len() > 12 {
        eprintln!("usage: {} pi-policy-eval-gmres <graph.bin> <pruned_layer_dir> <policy_dir> <x0.f32|zero> <out_v.f32> <scratch_dir> [gamma=1.0] [max_iters=200] [rel_tol=1e-6] [dense_cache_gib=16]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let x0_arg = args[5].clone();
    let out_v_path = PathBuf::from(&args[6]);
    let scratch_dir = PathBuf::from(&args[7]);
    let gamma: f32 = if args.len() >= 9 { args[8].parse().context("invalid gamma")? } else { 1.0 };
    let max_iters: usize = if args.len() >= 10 { args[9].parse().context("invalid max_iters")? } else { 200 };
    let rel_tol: f64 = if args.len() >= 11 { args[10].parse().context("invalid rel_tol")? } else { 1e-6 };
    let cache_gib: usize = if args.len() >= 12 { args[11].parse().context("invalid dense_cache_gib")? } else { 16 };
    if !(0.0..=1.0).contains(&gamma) { anyhow::bail!("gamma must be in [0,1], got {}", gamma); }
    let (compiled_l9_enabled, compiled_l8_enabled, action_cache_bytes) = loop1_operator_settings(gamma)?;
    let whole_settings = loop1_whole_operator::Settings::from_env(gamma, compiled_l8_enabled, action_cache_bytes)?;
    let raw_warm_start = loop1_whole_operator::raw_warm_start_enabled(whole_settings.is_some())?;
    let compiled_l9_enabled = compiled_l9_enabled || whole_settings.is_some_and(|settings| settings.l9_tail);
    anyhow::ensure!(rel_tol.is_finite() && rel_tol >= 0.0, "rel_tol must be finite and nonnegative");
    let accept_tol: f64 = match env::var("PI_ACCEPT_BELLMAN_INF") {
        Ok(value) => value.parse().context("invalid PI_ACCEPT_BELLMAN_INF")?,
        Err(env::VarError::NotPresent) => 1e-3,
        Err(error) => return Err(error).context("invalid PI_ACCEPT_BELLMAN_INF"),
    };
    anyhow::ensure!(accept_tol.is_finite() && accept_tol >= 0.0, "PI_ACCEPT_BELLMAN_INF must be finite and nonnegative");
    // Bind inputs before loading graph/index data. A prepared policy must not
    // survive a source replacement or in-place policy edit between applies.
    let mut input_paths = vec![graph_path.clone()];
    for layer in 0usize..=9 {
        let prefix = q_pruned_prefix(&pruned_dir, layer);
        input_paths.extend([prefix.with_extension("field_index.bin"), prefix.with_extension("chains.u24.bin"), mdp_policy_u16_path(&policy_dir, layer)]);
    }
    let initial_path = if x0_arg == "zero" { None } else { Some(Path::new(&x0_arg)) };
    loop1_prepared::reject_solver_aliases(&input_paths, initial_path, &out_v_path, &scratch_dir)?;
    let whole_guard = if whole_settings.is_some() {
        let mut bound_paths = input_paths.clone();
        bound_paths.extend(initial_path.map(Path::to_path_buf));
        Some(loop1_whole_operator::InputGuard::capture(bound_paths)?)
    } else { None };
    let mut source_bindings = if compiled_l9_enabled || action_cache_bytes > 0 {
        let mut bound_paths = input_paths.clone();
        bound_paths.extend(initial_path.map(Path::to_path_buf));
        Some(loop1_prepared::SourceBindings::capture_paths(bound_paths)?)
    } else { None };
    fs::create_dir_all(&scratch_dir)?;

    eprintln!("pi-policy-eval-gmres loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    drop(old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {} != {}", qstate_count, CHECK_QSTATE_COUNT); }
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    let l0_reset_qids = layer0_reset_block_qids(&full_layers[0])?;
    let n = usize::try_from(full_layers[0].total_states).context("boundary state count exceeds usize")?;
    anyhow::ensure!(n > 0, "empty boundary state set");
    if whole_settings.is_some() { loop1_whole_operator::validate_root_layout(&graph, &full_layers, &l0_reset_qids)?; }

    let whole_start = Instant::now();
    let pre_reveal = if compiled_l9_enabled || action_cache_bytes > 0 || whole_settings.is_some() {
        Some(pre_reveal::PreRevealIndex::build(&q_index)?)
    } else { None };
    let action_placements = if action_cache_bytes > 0 {
        Some(action_avg_cache::ValidatedPlacements::build(&q_index, &qfwd,
            pre_reveal.as_ref().expect("action cache pre-reveal index"))?)
    } else { None };
    // The raw test precedes all compiled-tail and DAG construction. A rejected
    // preflight keeps its fully validated policy and the exact loaded x0 for
    // the normal solver, while its temporary numeric caches are already gone.
    let mut whole_policy = None;
    let mut preflight_x0f = None;
    if raw_warm_start {
        let preflight_start = Instant::now();
        let initial = load_boundary_vector(&x0_arg, full_layers[0].total_states)?;
        anyhow::ensure!(initial.iter().all(|value| value.is_finite()), "nonfinite GMRES initial value");
        let guard = whole_guard.as_ref().expect("raw warm start requires whole input guard");
        let settings = whole_settings.expect("raw warm start requires whole settings");
        whole_policy = Some(loop1_whole_operator::validated_policy(&graph, &qfwd,
            pre_reveal.as_ref().expect("raw warm start formula index"), &full_layers,
            &policy_dir, two_line_field, guard)?);
        let policy = whole_policy.as_ref().expect("raw warm start validated policy");
        eprintln!("pi-policy-eval-gmres raw-warm-start roots={n} gamma=1 compiled_l9=0 dag=0 full_policy=1 workers=4 original_boundary=1");
        let reference = loop1_whole_operator::raw_boundary_apply(policy, &qfwd, &l0_reset_qids,
            guard, &initial, false, settings.reference)?;
        reference.log_timing("pi-policy-eval-gmres raw-warm-start");
        let residual = gmres_bellman_inf(&initial, &reference.values)?;
        if residual <= accept_tol {
            let applied_path = out_v_path.with_extension("applied.f32.bin");
            // Publish the companion first. An authorized x0==out replacement
            // must be the last operation that checks the original input name.
            publish_finite_f32_atomic_checked(&applied_path, &reference.values, || {
                guard.check()?;
                policy.check_inputs_unchanged()
            })?;
            let bellman = gmres_accept_and_publish_checked(&out_v_path, &initial, &reference.values,
                accept_tol, || {
                    guard.check()?;
                    policy.check_inputs_unchanged()
                })?;
            let (mut mn, mut mx, mut sum, mut negs) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0u64);
            for &value in &initial {
                mn = mn.min(value);
                mx = mx.max(value);
                sum += value as f64;
                if value < 0.0 { negs += 1; }
            }
            eprintln!("pi-policy-eval-gmres raw-warm-start accepted roots={n} finite=1 raw_all_roots=1 elapsed_seconds={:.9} dag_built=0 relative_residual=not_computed", preflight_start.elapsed().as_secs_f64());
            eprintln!("pi-policy-eval-gmres done v={} status=ACCEPTED_WARM_START krylov_iters=0 |T(v)-v|_inf={:.6e} accept_tol={:.1e} relative_residual=not_computed dag_built=0 v_min={:.8} v_max={:.8} v_avg={:.8} negatives={} (T(v) also written to {})",
                out_v_path.display(), bellman, accept_tol, mn, mx, sum / n.max(1) as f64, negs, applied_path.display());
            return Ok(());
        }
        eprintln!("pi-policy-eval-gmres raw-warm-start rejected |T(x0)-x0|_inf={residual:.6e} accept_tol={accept_tol:.1e} fallback=whole-dag elapsed_seconds={:.9}", preflight_start.elapsed().as_secs_f64());
        preflight_x0f = Some(initial);
    }
    let prepared_l9 = if compiled_l9_enabled {
        Some(loop1_prepared::PreparedL9::build(&graph, &q_index, &qfwd,
            pre_reveal.as_ref().expect("compiled L9 pre-reveal index"), &full_layers,
            &policy_dir, source_bindings.take().expect("compiled L9 source bindings"))?)
    } else { None };
    let prepared_l8 = if compiled_l8_enabled {
        Some(loop1_prepared_l8::build(&graph, &q_index, &qfwd,
            pre_reveal.as_ref().expect("compiled L8 pre-reveal index"), &full_layers[8],
            &mdp_policy_u16_path(&policy_dir, 8),
            &prepared_l9.as_ref().expect("compiled L8 requires L9").compiled)?)
    } else { None };
    let action_resident_bytes = if !compiled_l9_enabled && action_cache_bytes > 0 {
        action_placements.as_ref().expect("action placement plan").storage_bytes()
    } else { 0 };
    if whole_policy.is_none() {
        if let Some(guard) = whole_guard.as_ref() {
            whole_policy = Some(loop1_whole_operator::validated_policy(&graph, &qfwd,
                pre_reveal.as_ref().expect("whole DAG formula index"), &full_layers,
                &policy_dir, two_line_field, guard)?);
        }
    }
    let whole_terminal = if whole_settings.is_some() {
        Some(loop1_terminal::TerminalPlan::build(&q_index, &qfwd,
            pre_reveal.as_ref().expect("whole DAG formula index"))?)
    } else { None };
    let whole_plan = if let Some(settings) = whole_settings {
        Some(loop1_whole_operator::compile(whole_policy.as_ref().expect("whole DAG policy"),
            settings, prepared_l9.as_ref().map(|prepared| &prepared.compiled),
            whole_guard.as_ref().expect("whole DAG input guard"))?)
    } else { None };
    let whole_operator = if let Some(plan) = whole_plan.as_ref() {
        Some(loop1_whole_operator::FullBoundaryDag::new(plan,
            whole_policy.as_ref().expect("whole DAG policy"),
            whole_terminal.as_ref().expect("whole DAG terminal plan"), &qfwd, &l0_reset_qids,
            whole_guard.as_ref().expect("whole DAG input guard"),
            whole_settings.expect("whole DAG settings").reference)?)
    } else { None };
    if whole_operator.is_some() {
        eprintln!("whole-dag startup seconds={:.6} includes_formulas_tail_fullvalidation_plan=1 excludes_graph_qstate_load=1", whole_start.elapsed().as_secs_f64());
    }
    let sweep = FixedPolicySweep {
        graph: &graph,
        qfwd: &qfwd,
        full_layers: &full_layers,
        policy_dir: &policy_dir,
        scratch_dir: &scratch_dir,
        two_line_field,
        qstate_count,
        gamma,
        cache_cap_bytes: cache_gib.saturating_mul(1024).saturating_mul(1024).saturating_mul(1024),
        sparse_lookup_max_count: env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536),
        block_chunk: env::var("FULL_MDP_BLOCK_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512).max(1),
        batch_chunks_env: env::var("FULL_MDP_BATCH_CHUNKS").ok().and_then(|s| s.parse().ok()),
        ram_next_layer: env::var("FULL_MDP_RAM_NEXT_LAYER").ok().map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no").unwrap_or(true),
        ram_next_layer_max_bytes: env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(96).saturating_mul(1024 * 1024 * 1024),
        l0_reset_qids: &l0_reset_qids,
        verbose_layers: false,
        prepared_l9: prepared_l9.as_ref(),
        prepared_l8: prepared_l8.as_ref(),
        action_source: action_placements.as_ref().map(|placements| action_avg_cache::ActionSource {
            placements, requested_bytes: action_cache_bytes,
        }),
        operator_bindings: source_bindings.as_ref(),
        action_resident_bytes,
    };
    let to_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let l2 = |v: &[f64]| -> f64 { v.iter().map(|&x| x * x).sum::<f64>().sqrt() };
    let apply = |values: &[f32], linear: bool| -> Result<Vec<f32>> {
        if let Some(operator) = whole_operator.as_ref() { operator.apply(values, linear) }
        else { sweep.apply(values, linear) }
    };

    eprintln!("pi-policy-eval-gmres n={} computing c_pi = T_pi(0) (affine sweep 1)", n);
    let b32 = apply(&vec![0.0f32; n], false)?;
    anyhow::ensure!(b32.iter().all(|v| v.is_finite()), "nonfinite affine policy sweep");
    let b: Vec<f64> = b32.iter().map(|&v| v as f64).collect();
    let b_norm = l2(&b).max(1e-300);

    let x0f = match preflight_x0f.take() {
        Some(initial) => initial,
        None => load_boundary_vector(&x0_arg, full_layers[0].total_states)?,
    };
    anyhow::ensure!(x0f.iter().all(|v| v.is_finite()), "nonfinite GMRES initial value");
    let mut x0: Vec<f64> = x0f.iter().map(|&v| v as f64).collect();
    // r0 = b - A*x0 = b - (x0 - gamma*M*x0) = b + gamma*M*x0 - x0. gamma*M*x0 via the LINEAR
    // sweep (no cancellation against the affine constant).
    eprintln!("pi-policy-eval-gmres computing r0 = b + gamma*M*x0 - x0 (linear sweep 2)");
    let mx0 = apply(&x0f, true)?;
    drop(x0f);
    let mut r0: Vec<f64> = (0..n).map(|i| b[i] + mx0[i] as f64 - x0[i]).collect();
    anyhow::ensure!(r0.iter().all(|v| v.is_finite()), "nonfinite initial policy residual");
    let bellman0 = r0.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut beta = l2(&r0);
    eprintln!("pi-policy-eval-gmres |r0|_2={:.6e} |r0|_inf={:.6e} |b|_2={:.6e} rel={:.6e}", beta, bellman0, b_norm, beta / b_norm);

    // Aitken/rank-1 deflation jump (opt-out with PI_GMRES_AITKEN=0). The VI transient is
    // dominated by one near-Perron mode: r0 is nearly an eigenvector of gamma*M with
    // eigenvalue lam ~ rho(M) ~ 1 - 1/V*. Jump the geometric series along r0 in closed
    // form: x' = x0 + r0/(1-lam). Because r(x) = b + gamma*M*x - x is AFFINE, the new
    // residual is exactly r' = r0 + (M*r0 - r0)/(1-lam) — one extra linear sweep total,
    // no re-evaluation, and GMRES then only has to clean up the deflated remainder.
    // Worst case (lam estimate poor): ||r'|| >= ||r0|| and we keep the original x0.
    let aitken_on: bool = env::var("PI_GMRES_AITKEN").ok().map(|s| s != "0").unwrap_or(true);
    if aitken_on && beta > rel_tol * b_norm && bellman0 > accept_tol {
        let r0f: Vec<f32> = r0.iter().map(|&v| v as f32).collect();
        eprintln!("pi-policy-eval-gmres aitken: computing gamma*M*r0 (linear sweep 3)");
        let mr0 = apply(&r0f, true)?;
        drop(r0f);
        let rr: f64 = r0.iter().map(|&v| v * v).sum();
        let mrr: f64 = (0..n).map(|i| mr0[i] as f64 * r0[i]).sum();
        let lam = mrr / rr.max(1e-300);
        if lam > 0.0 && lam < 0.999_999 {
            let a = 1.0 / (1.0 - lam);
            let x_new: Vec<f64> = (0..n).map(|i| x0[i] + a * r0[i]).collect();
            let r_new: Vec<f64> = (0..n).map(|i| r0[i] + a * (mr0[i] as f64 - r0[i])).collect();
            let beta_new = l2(&r_new);
            if beta_new < beta {
                eprintln!("pi-policy-eval-gmres aitken ACCEPTED lam={:.9} jump={:.3e} |r|_2 {:.6e} -> {:.6e} ({:.2e}x)", lam, a, beta, beta_new, beta / beta_new.max(1e-300));
                x0 = x_new;
                r0 = r_new;
                beta = beta_new;
            } else {
                eprintln!("pi-policy-eval-gmres aitken REJECTED lam={:.9} jump={:.3e} |r|_2 would go {:.6e} -> {:.6e}", lam, a, beta, beta_new);
            }
        } else {
            eprintln!("pi-policy-eval-gmres aitken SKIPPED lam={:.9} out of (0,1) range", lam);
        }
    }

    // Periodic solution snapshots (PI_GMRES_SNAPSHOT_EVERY=N, 0=off). Every N iters, reconstruct
    // the current iterate (no sweep) and write out_v.iter{j}.f32.bin so mid-solve avg V / greedy
    // policy can be inspected before full convergence.
    let snap_every: usize = env::var("PI_GMRES_SNAPSHOT_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut h_cols: Vec<Vec<f64>> = Vec::new();
    let mut cs = Vec::<f64>::new();
    let mut sn = Vec::<f64>::new();
    let mut g = vec![beta];
    let initial_inf = r0.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut converged = beta <= rel_tol * b_norm || initial_inf <= accept_tol;
    // A warm start at tolerance still goes through the true affine residual gate below.
    if !converged && beta > 0.0 {
        // The guard applies only when a Krylov basis is actually needed. A
        // converged warm start needs no basis, regardless of max_iters/the cap.
        let basis_bytes = (max_iters as u64).saturating_mul(n as u64).saturating_mul(8);
        let mut basis_cap: u64 = env::var("PI_GMRES_BASIS_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(24).saturating_mul(1024 * 1024 * 1024);
        if whole_operator.is_some() { basis_cap = basis_cap.min(loop1_whole_operator::BASIS_CAP_BYTES); }
        anyhow::ensure!(basis_bytes <= basis_cap, "GMRES basis would need {} bytes (max_iters={} n={}) > cap {} bytes; lower max_iters or raise PI_GMRES_BASIS_MAX_GIB", basis_bytes, max_iters, n, basis_cap);
        basis.push(r0.iter().map(|&v| v / beta).collect());
        for j in 0..max_iters {
            let iter_start = Instant::now();
            // w = A q_j = q_j - gamma*M*q_j, with gamma*M*q_j from the LINEAR sweep directly.
            let mqj = apply(&to_f32(&basis[j]), true)?;
            let mut w: Vec<f64> = (0..n).map(|i| basis[j][i] - mqj[i] as f64).collect();
            let mut hcol = vec![0.0f64; j + 2];
            for i in 0..=j {
                let mut s = 0.0f64;
                let qi = &basis[i];
                for k in 0..n { s += qi[k] * w[k]; }
                hcol[i] = s;
                for k in 0..n { w[k] -= s * qi[k]; }
            }
            let hlast = l2(&w);
            hcol[j + 1] = hlast;
            for i in 0..j {
                let t1 = cs[i] * hcol[i] + sn[i] * hcol[i + 1];
                let t2 = -sn[i] * hcol[i] + cs[i] * hcol[i + 1];
                hcol[i] = t1;
                hcol[i + 1] = t2;
            }
            let denom = (hcol[j] * hcol[j] + hcol[j + 1] * hcol[j + 1]).sqrt();
            let (c, s) = if denom == 0.0 { (1.0, 0.0) } else { (hcol[j] / denom, hcol[j + 1] / denom) };
            cs.push(c);
            sn.push(s);
            hcol[j] = c * hcol[j] + s * hcol[j + 1];
            hcol[j + 1] = 0.0;
            g.push(-s * g[j]);
            g[j] = c * g[j];
            h_cols.push(hcol);
            let resid = g[j + 1].abs();
            eprintln!("pi-policy-eval-gmres iter={} resid={:.6e} rel={:.6e} sweep_elapsed={:.1}s", j + 1, resid, resid / b_norm, iter_start.elapsed().as_secs_f64());
            if snap_every > 0 && (j + 1) % snap_every == 0 {
                let vj = gmres_reconstruct_solution(&x0, &basis, &h_cols, &g, n)?;
                let vjf = to_f32(&vj);
                let snap_path = out_v_path.with_extension(format!("iter{}.f32.bin", j + 1));
                loop1_prepared::reject_aux_output_aliases(&snap_path, &input_paths, initial_path, &out_v_path)?;
                write_f32_vec_file(&snap_path, &vjf)?;
                let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
                for &x in &vjf { mn = mn.min(x); mx = mx.max(x); sum += x as f64; }
                eprintln!("pi-policy-eval-gmres snapshot iter={} -> {} v_min={:.6} v_max={:.6} v_avg={:.6}", j + 1, snap_path.display(), mn, mx, sum / n.max(1) as f64);
            }
            if resid <= rel_tol * b_norm {
                eprintln!("pi-policy-eval-gmres converged at iter {}", j + 1);
                converged = true;
                break;
            }
            if hlast <= 1e-300 {
                eprintln!("pi-policy-eval-gmres happy breakdown at iter {} (Krylov space exhausted)", j + 1);
                converged = true;
                break;
            }
            if j + 1 < max_iters {
                basis.push(w.iter().map(|&v| v / hlast).collect());
            }
        }
    }
    // back substitution on the triangularized Hessenberg (R[i][k] = h_cols[k][i])
    let m = h_cols.len();
    let v = gmres_reconstruct_solution(&x0, &basis, &h_cols, &g, n)?;
    drop(basis);
    let vf = to_f32(&v);
    anyhow::ensure!(vf.iter().all(|v| v.is_finite()), "nonfinite GMRES candidate value");

    // Ground-truth check: one more AFFINE true sweep at the solution. bellman = |T_pi(v)-v|_inf
    // is the real fixed-point residual (independent of the Krylov residual estimate), and is what
    // policy improvement will consume, so it is the acceptance gate.
    let tv = if let Some(operator) = whole_operator.as_ref() {
        eprintln!("pi-policy-eval-gmres final verification affine T_pi(v) (raw numeric all-boundary, DAG bit comparison)");
        operator.apply_verified(&vf, false)?
    } else {
        eprintln!("pi-policy-eval-gmres final verification affine sweep T_pi(v) (original uncompiled)");
        sweep.apply_original(&vf, false)?
    };
    if let Some(operator) = whole_operator.as_ref() { operator.check_inputs_unchanged()?; }
    let bellman = if let Some(operator) = whole_operator.as_ref() {
        gmres_accept_and_publish_checked(&out_v_path, &vf, &tv, accept_tol,
            || operator.check_inputs_unchanged())?
    } else { gmres_accept_and_publish(&out_v_path, &vf, &tv, accept_tol)? };
    let (mut mn, mut mx, mut sum, mut negs) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0u64);
    for i in 0..n {
        mn = mn.min(vf[i]);
        mx = mx.max(vf[i]);
        sum += vf[i] as f64;
        if vf[i] < 0.0 { negs += 1; }
    }
    let applied_path = out_v_path.with_extension("applied.f32.bin");
    write_f32_vec_file(&applied_path, &tv)?;
    let status = if converged { "CONVERGED" } else { "MAX_ITERS" };
    eprintln!("pi-policy-eval-gmres done v={} status={} krylov_iters={} |T(v)-v|_inf={:.6e} accept_tol={:.1e} v_min={:.8} v_max={:.8} v_avg={:.8} negatives={} (T(v) also written to {})",
        out_v_path.display(), status, m, bellman, accept_tol, mn, mx, sum / n.max(1) as f64, negs, applied_path.display());
    Ok(())
}

// ============================================================================
// survival-krylov: matrix-free Arnoldi on the boundary->boundary survival map
// M_pi (fixed optimal policy pi*), to build P(die within n PCs) for any n.
//
//   d_pi(b) = 1 - sum_b' M_pi[b,b'] = per-loop death prob from boundary b.
//   g_n = sum_{j=0}^{n-1} M_pi^j d_pi ; A(n) = p0^T g_n.
//   Arnoldi K_k(M_pi, d_pi): M_pi V_k = V_{k+1} Hbar, so
//     p0^T M_pi^j d_pi ~= ||d_pi|| (p0^T V_k) H_k^j e_1
//   -> every n reconstructs from the tiny k x k H_k (no extra sweeps).
//   The only expensive op is the matvec M_pi x = one linear fixed-policy sweep
//   (FixedPolicySweep::apply(x, linear=true), gamma=1).
// ============================================================================

const SURVIVAL_STATE_MAGIC: u64 = 0x5355_5256_4B52_5931; // "SURVKRY1"

// Partial-pivot LU solve of A x = b (A consumed). None if singular.
fn dense_lu_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let m = b.len();
    for col in 0..m {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..m {
            let v = a[r][col].abs();
            if v > best { best = v; piv = r; }
        }
        if best < 1e-300 { return None; }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in col + 1..m {
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..m { a[r][c] -= f * a[col][c]; }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0f64; m];
    for r in (0..m).rev() {
        let mut s = b[r];
        for c in r + 1..m { s -= a[r][c] * x[c]; }
        x[r] = s / a[r][r];
    }
    Some(x)
}

// H*z for a square (row-major) m x m matrix.
fn hsq_matvec(hsq: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let m = z.len();
    let mut out = vec![0.0f64; m];
    for r in 0..m {
        let row = &hsq[r];
        let mut s = 0.0f64;
        for c in 0..m { s += row[c] * z[c]; }
        out[r] = s;
    }
    out
}

// Build the square top-left sz x sz Hessenberg from ragged columns (h[c] has c+2 entries,
// h[c][r] = H[r][c] for r in 0..=c+1).
fn build_square_h(h: &[Vec<f64>], sz: usize) -> Vec<Vec<f64>> {
    let mut hsq = vec![vec![0.0f64; sz]; sz];
    for c in 0..sz {
        let col = &h[c];
        let rmax = (c + 1).min(sz - 1);
        for r in 0..=rmax { hsq[r][c] = col[r]; }
    }
    hsq
}

// A_uniform(n) for n=1..=max_n using the size-sz projection:
//   value_j = beta * (u . H^j e_1);  A(n)=sum_{j<n} value_j.
// u has length sz (u[i] = p0 . v_i). Returns vec indexed A[n-1].
fn survival_a_curve(hsq: &[Vec<f64>], u: &[f64], beta: f64, max_n: usize) -> Vec<f64> {
    let sz = u.len();
    let mut z = vec![0.0f64; sz];
    if sz > 0 { z[0] = 1.0; } // e_1
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(max_n);
    for _n in 1..=max_n {
        let mut val = 0.0f64;
        for i in 0..sz { val += u[i] * z[i]; }
        acc += beta * val;
        out.push(acc);
        z = hsq_matvec(hsq, &z);
    }
    out
}

// A_uniform at an arbitrary single horizon n (n may be huge); iterative, cheap.
fn survival_a_at(hsq: &[Vec<f64>], u: &[f64], beta: f64, n: u64) -> f64 {
    let sz = u.len();
    let mut z = vec![0.0f64; sz];
    if sz > 0 { z[0] = 1.0; }
    let mut acc = 0.0f64;
    for _ in 0..n {
        let mut val = 0.0f64;
        for i in 0..sz { val += u[i] * z[i]; }
        acc += beta * val;
        z = hsq_matvec(hsq, &z);
    }
    acc
}

// A_uniform(inf) = beta * u^T (I-H)^{-1} e_1 (exact for the projected model; ->1 as k grows
// since (I-M)^{-1} d_pi = 1).
fn survival_a_inf(hsq: &[Vec<f64>], u: &[f64], beta: f64) -> Option<f64> {
    let sz = u.len();
    let mut a = vec![vec![0.0f64; sz]; sz];
    for r in 0..sz {
        for c in 0..sz { a[r][c] = -hsq[r][c]; }
        a[r][r] += 1.0;
    }
    let mut e1 = vec![0.0f64; sz];
    if sz > 0 { e1[0] = 1.0; }
    let x = dense_lu_solve(a, e1)?;
    let mut val = 0.0f64;
    for i in 0..sz { val += u[i] * x[i]; }
    Some(beta * val)
}

// Dominant (Perron) eigenvalue of H via shifted inverse iteration. Returns
// (lambda, |last eigvec component| / ||eigvec||) so the caller can form the Arnoldi
// eigenresidual bound |h_{k+1,k}| * |e_k^T s|.
fn dominant_eig(hsq: &[Vec<f64>]) -> (f64, f64) {
    let m = hsq.len();
    if m == 0 { return (0.0, 0.0); }
    if m == 1 { return (hsq[0][0], 0.0); }
    let mut mu = 0.999_9f64;
    let mut y = vec![1.0f64 / (m as f64).sqrt(); m];
    for _it in 0..300 {
        // A = H - mu I
        let mut a = vec![vec![0.0f64; m]; m];
        for r in 0..m {
            for c in 0..m { a[r][c] = hsq[r][c]; }
            a[r][r] -= mu;
        }
        let z = match dense_lu_solve(a, y.clone()) {
            Some(z) => z,
            None => { mu += 1e-6; continue; }
        };
        let nrm = z.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if nrm < 1e-300 { break; }
        let ynew: Vec<f64> = z.iter().map(|&v| v / nrm).collect();
        // Rayleigh quotient with current H: lambda = y^T H y (y unit).
        let hy = hsq_matvec(hsq, &ynew);
        let lam = ynew.iter().zip(hy.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let delta = (ynew.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f64>()).sqrt();
        y = ynew;
        // Re-center the shift on the current Rayleigh estimate to accelerate.
        mu = lam;
        if delta < 1e-14 { break; }
    }
    let hy = hsq_matvec(hsq, &y);
    let lam = y.iter().zip(hy.iter()).map(|(&a, &b)| a * b).sum::<f64>();
    let last = y[m - 1].abs();
    (lam, last)
}

// Binary resume/eval state. Layout (LE):
//   magic u64, n u64, m u64, beta f64, lambda f64,
//   colsums: m f64 (colsum[i] = sum over components of basis vector v_i, i=0..m-1),
//   h columns: for c in 0..m: (c+2) f64  (h[c][0..=c+1]).
fn write_survival_state(
    path: &Path, n: usize, beta: f64, lambda: f64, colsums: &[f64], h: &[Vec<f64>],
) -> Result<()> {
    let m = h.len();
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    w.write_all(&SURVIVAL_STATE_MAGIC.to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(m as u64).to_le_bytes())?;
    w.write_all(&beta.to_le_bytes())?;
    w.write_all(&lambda.to_le_bytes())?;
    for i in 0..m { w.write_all(&colsums[i].to_le_bytes())?; }
    for c in 0..m {
        anyhow::ensure!(h[c].len() == c + 2, "h column {} has len {} != {}", c, h[c].len(), c + 2);
        for &x in &h[c] { w.write_all(&x.to_le_bytes())?; }
    }
    w.flush()?;
    Ok(())
}

struct SurvivalState { n: usize, beta: f64, lambda: f64, colsums: Vec<f64>, h: Vec<Vec<f64>> }

fn read_survival_state(path: &Path) -> Result<SurvivalState> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut p = 0usize;
    let mut rd_u64 = |p: &mut usize| -> u64 { let v = u64::from_le_bytes(bytes[*p..*p + 8].try_into().unwrap()); *p += 8; v };
    anyhow::ensure!(rd_u64(&mut p) == SURVIVAL_STATE_MAGIC, "bad survival state magic");
    let n = rd_u64(&mut p) as usize;
    let m = rd_u64(&mut p) as usize;
    let mut rd_f64 = |p: &mut usize| -> f64 { let v = f64::from_le_bytes(bytes[*p..*p + 8].try_into().unwrap()); *p += 8; v };
    let beta = rd_f64(&mut p);
    let lambda = rd_f64(&mut p);
    let mut colsums = Vec::with_capacity(m);
    for _ in 0..m { colsums.push(rd_f64(&mut p)); }
    let mut h = Vec::with_capacity(m);
    for c in 0..m {
        let mut col = Vec::with_capacity(c + 2);
        for _ in 0..c + 2 { col.push(rd_f64(&mut p)); }
        h.push(col);
    }
    Ok(SurvivalState { n, beta, lambda, colsums, h })
}

// One full-reorthogonalization Arnoldi step given a matvec closure. `basis` holds v_0..v_{m}
// (m+1 vectors, m = h.len()); appends h column m and, unless breakdown, v_{m+1}.
fn arnoldi_step(
    basis: &mut Vec<Vec<f64>>,
    h: &mut Vec<Vec<f64>>,
    colsums: &mut Vec<f64>,
    w: Vec<f64>, // = M * v_j already computed
) {
    let j = h.len();
    debug_assert_eq!(basis.len(), j + 1);
    let n = basis[0].len();
    let mut w = w;
    let mut hcol = vec![0.0f64; j + 2];
    // Classical Gram-Schmidt twice (full re-orthogonalization) for stability.
    for _pass in 0..2 {
        for i in 0..=j {
            let mut s = 0.0f64;
            let vi = &basis[i];
            for k in 0..n { s += vi[k] * w[k]; }
            hcol[i] += s;
            for k in 0..n { w[k] -= s * vi[k]; }
        }
    }
    let hlast = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
    hcol[j + 1] = hlast;
    h.push(hcol);
    if hlast > 1e-30 {
        let vnext: Vec<f64> = w.iter().map(|&v| v / hlast).collect();
        colsums.push(vnext.iter().sum());
        basis.push(vnext);
    }
}

// boundary-cond-avg <pruned_layer_dir> <values.f32>
// Per-popcount V averages over the layer-0 boundary block, plus the fresh-game subset
// (bag popcnt=7 AND all 7 preview pieces distinct = first round of a new game).
fn run_boundary_cond_avg(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} boundary-cond-avg <pruned_layer_dir> <values.f32>", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let layer0 = layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, 0))?;
    let l0_reset_qids = layer0_reset_block_qids(&layer0)?;
    let v = load_f32_values_vec(&values_path, layer0.total_states)?;
    anyhow::ensure!(l0_reset_qids.len() == v.len(), "boundary count {} != values {}", l0_reset_qids.len(), v.len());

    let mut sum_by_pop = [0f64; 8];
    let mut cnt_by_pop = [0usize; 8];
    let (mut fsum, mut fcnt) = (0f64, 0usize);
    let (mut fmin, mut fmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for (i, &qid) in l0_reset_qids.iter().enumerate() {
        let s = &q_index.states[qid as usize];
        let k = s.bag_mask.count_ones() as usize;
        sum_by_pop[k] += v[i] as f64;
        cnt_by_pop[k] += 1;
        if s.bag_mask == FULL_BAG {
            let (a, b) = decode_avail_pair(s.avail);
            let tail = decode_tail5(s.tail5);
            let mut mask = (1u8 << a) | (1u8 << b);
            let mut dup = a == b;
            for &p in &tail {
                if mask & (1 << p) != 0 { dup = true; }
                mask |= 1 << p;
            }
            if !dup {
                debug_assert_eq!(mask, FULL_BAG);
                fsum += v[i] as f64;
                fcnt += 1;
                fmin = fmin.min(v[i]);
                fmax = fmax.max(v[i]);
            }
        }
    }
    let total: usize = cnt_by_pop.iter().sum();
    let tsum: f64 = sum_by_pop.iter().sum();
    println!("boundary total n={} avg={:.4}", total, tsum / total as f64);
    for k in 1..8 {
        if cnt_by_pop[k] > 0 {
            println!("popcnt={} n={} avg={:.4}", k, cnt_by_pop[k], sum_by_pop[k] / cnt_by_pop[k] as f64);
        }
    }
    println!("fresh-start (popcnt=7, no-dupe) n={} avg={:.4} min={:.2} max={:.2}", fcnt, fsum / fcnt as f64, fmin, fmax);
    Ok(())
}

// boundary-survival-rank <pruned_layer_dir> <survival_artifacts_dir> <n> [topk=10] [values.f32]
// Rank boundary states by p_die(s, n) = basis_rows(s) · w_n. Prints the topk lowest
// and highest states, decoded.
fn run_boundary_survival_rank(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        eprintln!("usage: {} boundary-survival-rank <pruned_layer_dir> <survival_artifacts_dir> <n> [topk=10] [values.f32]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let art_dir = PathBuf::from(&args[3]);
    let horizon: usize = args[4].parse().context("invalid n")?;
    let topk: usize = if args.len() >= 6 { args[5].parse().context("invalid topk")? } else { 10 };

    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let layer0 = layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, 0))?;
    let qids = layer0_reset_block_qids(&layer0)?;
    let nb = qids.len();
    let values: Option<Vec<f32>> = if args.len() >= 7 {
        Some(load_f32_values_vec(Path::new(&args[6]), nb as u64)?)
    } else { None };

    let st = read_survival_state(&art_dir.join("state.bin"))?;
    anyhow::ensure!(st.n == nb, "survival state n={} != boundary count {}", st.n, nb);
    let m = st.h.len();
    let hsq = build_square_h(&st.h, m);
    // w = beta * sum_{j<horizon} H^j e_1  (identical to the oracle's w_dense recursion)
    let mut z = vec![0.0f64; m];
    z[0] = 1.0;
    let mut w = vec![0.0f64; m];
    for _ in 0..horizon {
        for i in 0..m { w[i] += st.beta * z[i]; }
        z = hsq_matvec(&hsq, &z);
    }

    let bf = fs::read(art_dir.join("basis.f64.bin")).with_context(|| format!("read {}/basis.f64.bin", art_dir.display()))?;
    anyhow::ensure!(bf.len() >= m * nb * 8, "basis file too small for m={}", m);
    let mut ps = vec![0.0f64; nb];
    for i in 0..m {
        let base = i * nb * 8;
        let wi = w[i];
        if wi == 0.0 { continue; }
        for s in 0..nb {
            ps[s] += wi * f64::from_le_bytes(bf[base + s * 8..base + s * 8 + 8].try_into().unwrap());
        }
    }

    let mut order: Vec<usize> = (0..nb).collect();
    order.sort_by(|&a, &b| ps[a].partial_cmp(&ps[b]).unwrap());
    let render = |idx: usize| -> String {
        let s = &q_index.states[qids[idx] as usize];
        let (a, b) = decode_avail_pair(s.avail);
        let tail = decode_tail5(s.tail5);
        let tail_s: String = tail.iter().map(|&p| piece_char(p)).collect();
        let vtxt = values.as_ref().map(|v| format!(" V={:.2}", v[idx])).unwrap_or_default();
        format!("p_die({})={:+.3e} clamp={:.6} avail={{{},{}}} tail={} bag={}{}",
            horizon, ps[idx], ps[idx].clamp(0.0, 1.0), piece_char(a), piece_char(b), tail_s, mask_to_string(s.bag_mask), vtxt)
    };
    println!("boundary-survival-rank n={} m={} nb={} lambda={:.8}", horizon, m, nb, st.lambda);
    println!("-- lowest p_die (safest) --");
    for r in 0..topk.min(nb) { println!("#{:<2} {}", r + 1, render(order[r])); }
    println!("-- highest p_die (most dangerous) --");
    for r in 0..topk.min(nb) { println!("#{:<2} {}", r + 1, render(order[nb - 1 - r])); }
    Ok(())
}

fn run_survival_krylov(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} survival-krylov <graph.bin> <pruned_layer_dir> <policy_dir> <out_dir> [max_k=60] [tol=1e-3] [dense_cache_gib=16]", args[0]);
        eprintln!("  Resumable/growable: re-run with a larger max_k to add matvecs (prior sweeps in <out_dir> are reused).");
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let out_dir = PathBuf::from(&args[5]);
    let max_k: usize = if args.len() >= 7 { args[6].parse().context("invalid max_k")? } else { 60 };
    let tol: f64 = if args.len() >= 8 { args[7].parse().context("invalid tol")? } else { 1e-3 };
    let cache_gib: usize = if args.len() >= 9 { args[8].parse().context("invalid dense_cache_gib")? } else { 16 };
    fs::create_dir_all(&out_dir)?;
    let scratch_dir = out_dir.join("scratch");
    fs::create_dir_all(&scratch_dir)?;

    eprintln!("survival-krylov loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {} != {}", qstate_count, CHECK_QSTATE_COUNT); }
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    let l0_reset_qids = layer0_reset_block_qids(&full_layers[0])?;
    let n = full_layers[0].total_states as usize;

    let sweep = FixedPolicySweep {
        graph: &graph,
        qfwd: &qfwd,
        full_layers: &full_layers,
        policy_dir: &policy_dir,
        scratch_dir: &scratch_dir,
        two_line_field,
        qstate_count,
        gamma: 1.0,
        cache_cap_bytes: cache_gib.saturating_mul(1024).saturating_mul(1024).saturating_mul(1024),
        sparse_lookup_max_count: env::var("FULL_MDP_SPARSE_LOOKUP_MAX_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(65_536),
        block_chunk: env::var("FULL_MDP_BLOCK_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512).max(1),
        batch_chunks_env: env::var("FULL_MDP_BATCH_CHUNKS").ok().and_then(|s| s.parse().ok()),
        ram_next_layer: env::var("FULL_MDP_RAM_NEXT_LAYER").ok().map(|s| s != "0" && s.to_ascii_lowercase() != "false" && s.to_ascii_lowercase() != "no").unwrap_or(true),
        ram_next_layer_max_bytes: env::var("FULL_MDP_RAM_NEXT_LAYER_MAX_GIB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(96).saturating_mul(1024 * 1024 * 1024),
        l0_reset_qids: &l0_reset_qids,
        verbose_layers: false,
        prepared_l9: None,
        prepared_l8: None,
        action_source: None,
        operator_bindings: None,
        action_resident_bytes: 0,
    };
    let matvec = |v: &[f64]| -> Result<Vec<f64>> {
        let vf: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let mv = sweep.apply(&vf, true)?;
        Ok(mv.iter().map(|&x| x as f64).collect())
    };

    let dpi_path = out_dir.join("dpi.f64.bin");
    let basis_path = out_dir.join("basis.f64.bin");
    let state_path = out_dir.join("state.bin");

    // ---- d_pi = 1 - M_pi*1 (one linear sweep on the all-ones vector), persisted. ----
    let dpi: Vec<f64> = if dpi_path.exists() {
        eprintln!("survival-krylov loading cached d_pi from {}", dpi_path.display());
        let bytes = fs::read(&dpi_path)?;
        anyhow::ensure!(bytes.len() == n * 8, "dpi cache size mismatch");
        (0..n).map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())).collect()
    } else {
        eprintln!("survival-krylov computing d_pi = 1 - M_pi*1 (linear sweep on ones)");
        let m1 = matvec(&vec![1.0f64; n])?;
        let d: Vec<f64> = m1.iter().map(|&s| 1.0 - s).collect();
        let mut w = BufWriter::new(File::create(&dpi_path)?);
        for &x in &d { w.write_all(&x.to_le_bytes())?; }
        w.flush()?;
        d
    };
    let (mut dmin, mut dmax, mut dsum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    let mut dneg = 0u64;
    for &d in &dpi { dmin = dmin.min(d); dmax = dmax.max(d); dsum += d; if d < 0.0 { dneg += 1; } }
    let dmean = dsum / n as f64;
    eprintln!("survival-krylov d_pi stats: min={:.6e} mean={:.6e} max={:.6e} negatives={} (1/mean={:.1})", dmin, dmean, dmax, dneg, 1.0 / dmean);
    let beta = dpi.iter().map(|&v| v * v).sum::<f64>().sqrt();

    // ---- Load or initialize the Arnoldi basis + Hessenberg (resume path). ----
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut h: Vec<Vec<f64>> = Vec::new();
    let mut colsums: Vec<f64> = Vec::new();
    if basis_path.exists() && state_path.exists() {
        let bytes_len = fs::metadata(&basis_path)?.len() as usize;
        anyhow::ensure!(bytes_len % (n * 8) == 0, "basis file not a multiple of vector size");
        let num_basis = bytes_len / (n * 8);
        eprintln!("survival-krylov RESUME: loading {} basis vectors + state from {}", num_basis, out_dir.display());
        let st = read_survival_state(&state_path)?;
        anyhow::ensure!(st.n == n, "state n mismatch");
        anyhow::ensure!(st.h.len() + 1 == num_basis, "state m={} inconsistent with basis count {}", st.h.len(), num_basis);
        let bf = fs::read(&basis_path)?;
        for vi in 0..num_basis {
            let base = vi * n * 8;
            let v: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(bf[base + i * 8..base + i * 8 + 8].try_into().unwrap())).collect();
            basis.push(v);
        }
        h = st.h;
        colsums = st.colsums;
    } else {
        // v_0 = d_pi / ||d_pi||
        let v0: Vec<f64> = dpi.iter().map(|&v| v / beta).collect();
        colsums.push(v0.iter().sum());
        let mut w = BufWriter::new(File::create(&basis_path)?);
        for &x in &v0 { w.write_all(&x.to_le_bytes())?; }
        w.flush()?;
        basis.push(v0);
    }

    let report_ns: Vec<usize> = (1..=100).collect();
    let big_ns: [u64; 4] = [1000, 10000, 100000, 1000000];
    let mut prev_curve: Option<Vec<f64>> = None;

    // ---- Arnoldi iterations (each = one M_pi matvec = one fixed-policy sweep). ----
    while h.len() < max_k {
        let j = h.len();
        let iter_start = Instant::now();
        eprintln!("survival-krylov matvec {} (Arnoldi step for column {}) ...", j + 1, j);
        let w = matvec(&basis[j])?;
        arnoldi_step(&mut basis, &mut h, &mut colsums, w);
        let hlast = h[j][j + 1];
        // Persist the new basis vector (if produced) by appending, and rewrite state.bin.
        if basis.len() == j + 2 {
            let mut bf = OpenOptions::new().append(true).open(&basis_path)?;
            for &x in &basis[j + 1] { bf.write_all(&x.to_le_bytes())?; }
            bf.flush()?;
        }
        let m = h.len();
        let hsq = build_square_h(&h, m);
        let u_uniform: Vec<f64> = colsums[0..m].iter().map(|&c| c / n as f64).collect();
        let (lambda, eig_last) = dominant_eig(&hsq);
        write_survival_state(&state_path, n, beta, lambda, &colsums, &h)?;

        // A(n) table for n=1..100, both start distributions.
        let curve_uniform = survival_a_curve(&hsq, &u_uniform, beta, *report_ns.last().unwrap());
        // Successive-difference relative error vs the size-(m-1) projection.
        let mut max_relerr = f64::NAN;
        let mut relerrs = vec![f64::NAN; report_ns.len()];
        if m >= 2 {
            let hsq_prev = build_square_h(&h, m - 1);
            let u_prev: Vec<f64> = colsums[0..m - 1].iter().map(|&c| c / n as f64).collect();
            let curve_prev = survival_a_curve(&hsq_prev, &u_prev, beta, *report_ns.last().unwrap());
            let mut mx = 0.0f64;
            for (idx, _n) in report_ns.iter().enumerate() {
                let a = curve_uniform[idx];
                let e = if a.abs() > 1e-300 { (a - curve_prev[idx]).abs() / a.abs() } else { 0.0 };
                relerrs[idx] = e;
                mx = mx.max(e);
            }
            max_relerr = mx;
        }
        let eig_resid = hlast * eig_last;
        let a_inf = survival_a_inf(&hsq, &u_uniform, beta).unwrap_or(f64::NAN);
        eprintln!(
            "survival-krylov k={} lambda={:.10} 1/(1-lambda)={:.2} eig_resid={:.3e} hlast={:.3e} A_unif(1)={:.6e}(dmean={:.6e}) A_unif(100)={:.6e} A_inf={:.6} max_relerr(n<=100)={:.3e} sweep={:.1}s",
            m, lambda, 1.0 / (1.0 - lambda), eig_resid, hlast,
            curve_uniform[0], dmean, curve_uniform[99], a_inf, max_relerr, iter_start.elapsed().as_secs_f64()
        );

        // Write the human/machine outputs each iteration (so a background run is always queryable).
        write_survival_outputs(&out_dir, &report_ns, &curve_uniform, &relerrs, lambda, &big_ns, &hsq, &u_uniform, beta, dmin, dmean, dmax, m, eig_resid, hlast)?;

        prev_curve = Some(curve_uniform);
        if m >= 2 && max_relerr < tol {
            eprintln!("survival-krylov CONVERGED at k={} (max_relerr={:.3e} < tol={:.3e})", m, max_relerr, tol);
            break;
        }
    }
    let _ = prev_curve;
    eprintln!("survival-krylov done. artifacts in {} (state.bin, basis.f64.bin, survival_table.csv, model.json)", out_dir.display());
    eprintln!("  resume/grow: re-run `survival-krylov {} {} {} {} <max_k_bigger> <tol>`", graph_path.display(), pruned_dir.display(), policy_dir.display(), out_dir.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_survival_outputs(
    out_dir: &Path, report_ns: &[usize], curve_uniform: &[f64], relerrs: &[f64], lambda: f64,
    big_ns: &[u64; 4], hsq: &[Vec<f64>], u_uniform: &[f64], beta: f64,
    dmin: f64, dmean: f64, dmax: f64, k: usize, eig_resid: f64, hlast: f64,
) -> Result<()> {
    // CSV: n, p_die_uniform, p_die_stationary(=1-lambda^n), relerr_uniform
    let csv_path = out_dir.join("survival_table.csv");
    let mut w = BufWriter::new(File::create(&csv_path)?);
    writeln!(w, "n,p_die_uniform_start,p_die_stationary_start,relerr_uniform")?;
    for (idx, &nn) in report_ns.iter().enumerate() {
        let stat = 1.0 - lambda.powi(nn as i32);
        writeln!(w, "{},{:.10e},{:.10e},{:.3e}", nn, curve_uniform[idx], stat, relerrs[idx])?;
    }
    w.flush()?;

    // JSON model + headline numbers (large-n included). Hand-rolled (no serde).
    let json_path = out_dir.join("model.json");
    let mut j = BufWriter::new(File::create(&json_path)?);
    writeln!(j, "{{")?;
    writeln!(j, "  \"k\": {}, \"n_boundaries\": 817740,", k)?;
    writeln!(j, "  \"lambda\": {:.15}, \"one_over_1_minus_lambda\": {:.6},", lambda, 1.0 / (1.0 - lambda))?;
    writeln!(j, "  \"eig_arnoldi_residual\": {:.6e}, \"hlast\": {:.6e},", eig_resid, hlast)?;
    writeln!(j, "  \"dpi_min\": {:.9e}, \"dpi_mean\": {:.9e}, \"dpi_max\": {:.9e},", dmin, dmean, dmax)?;
    writeln!(j, "  \"beta_norm_dpi\": {:.9e},", beta)?;
    let a_inf = survival_a_inf(hsq, u_uniform, beta).unwrap_or(f64::NAN);
    writeln!(j, "  \"A_uniform_inf\": {:.9},", a_inf)?;
    // headline horizons
    write!(j, "  \"p_die_uniform\": {{")?;
    let heads = [1usize, 2, 5, 10, 20, 50, 100];
    for (i, &nn) in heads.iter().enumerate() {
        write!(j, "{}\"{}\": {:.9e}", if i > 0 { ", " } else { "" }, nn, curve_uniform[nn - 1])?;
    }
    for &nn in big_ns.iter() {
        let v = survival_a_at(hsq, u_uniform, beta, nn);
        write!(j, ", \"{}\": {:.9e}", nn, v)?;
    }
    writeln!(j, "}},")?;
    write!(j, "  \"p_die_stationary\": {{")?;
    for (i, &nn) in heads.iter().enumerate() {
        write!(j, "{}\"{}\": {:.9e}", if i > 0 { ", " } else { "" }, nn, 1.0 - lambda.powi(nn as i32))?;
    }
    for &nn in big_ns.iter() {
        let ln1m = (nn as f64) * (lambda.ln());
        let v = 1.0 - ln1m.exp();
        write!(j, ", \"{}\": {:.9e}", nn, v)?;
    }
    writeln!(j, "}}")?;
    writeln!(j, "}}")?;
    j.flush()?;
    Ok(())
}

// survival-eval-n: reconstruct P(die within n PCs) for ANY n from saved artifacts (no sweeps).
fn run_survival_eval_n(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} survival-eval-n <artifacts_dir> <n1> [n2 ...]", args[0]);
        eprintln!("  reads <artifacts_dir>/state.bin (H_k, u projections, beta, lambda) and prints P(die within n).");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[2]);
    let st = read_survival_state(&dir.join("state.bin"))?;
    let m = st.h.len();
    let hsq = build_square_h(&st.h, m);
    let u_uniform: Vec<f64> = st.colsums[0..m].iter().map(|&c| c / st.n as f64).collect();
    println!("# artifacts={} k={} lambda={:.12} 1/(1-lambda)={:.4} beta={:.6e}", dir.display(), m, st.lambda, 1.0 / (1.0 - st.lambda), st.beta);
    println!("# n\tp_die_uniform_start\tp_die_stationary_start");
    for a in &args[3..] {
        let nn: u64 = a.parse().with_context(|| format!("invalid n '{}'", a))?;
        let uni = survival_a_at(&hsq, &u_uniform, st.beta, nn);
        let stat = 1.0 - (nn as f64 * st.lambda.ln()).exp();
        println!("{}\t{:.10e}\t{:.10e}", nn, uni, stat);
    }
    Ok(())
}

// Self-test: Arnoldi(full reorth) on a small dense matrix, with a disk round-trip of state.bin
// mid-way, must reproduce the SAME Hessenberg as an uninterrupted run. Validates the resume path.
fn run_survival_krylov_selftest(_args: &[String]) -> Result<()> {
    let nn = 12usize;
    // deterministic pseudo-random substochastic-ish matrix
    let mut a = vec![vec![0.0f64; nn]; nn];
    let mut s: u64 = 0x1234_5678;
    let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
    for r in 0..nn { for c in 0..nn { a[r][c] = rnd(); } }
    let matvec = |v: &[f64]| -> Vec<f64> {
        let mut o = vec![0.0f64; nn];
        for r in 0..nn { let mut t = 0.0; for c in 0..nn { t += a[r][c] * v[c]; } o[r] = t; }
        o
    };
    let d: Vec<f64> = (0..nn).map(|i| 1.0 + i as f64).collect();
    let beta = d.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let run = |total: usize, break_at: Option<usize>, dir: Option<&Path>| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut basis: Vec<Vec<f64>> = vec![d.iter().map(|&x| x / beta).collect()];
        let mut h: Vec<Vec<f64>> = Vec::new();
        let mut colsums: Vec<f64> = vec![basis[0].iter().sum()];
        for step in 0..total {
            let w = matvec(&basis[h.len()]);
            arnoldi_step(&mut basis, &mut h, &mut colsums, w);
            if let (Some(ba), Some(dd)) = (break_at, dir) {
                if step + 1 == ba {
                    // round-trip state.bin, reload
                    let (lam, _) = dominant_eig(&build_square_h(&h, h.len()));
                    write_survival_state(&dd.join("state.bin"), nn, beta, lam, &colsums, &h).unwrap();
                    let stt = read_survival_state(&dd.join("state.bin")).unwrap();
                    h = stt.h; colsums = stt.colsums;
                    // basis is kept in-RAM here (in the real tool it is reloaded from basis.f64.bin)
                }
            }
        }
        (h, colsums)
    };
    let (h_full, _) = run(6, None, None);
    let td = std::env::temp_dir().join("survkry_selftest");
    fs::create_dir_all(&td)?;
    let (h_resumed, _) = run(6, Some(3), Some(&td));
    let mut maxd = 0.0f64;
    for c in 0..h_full.len() {
        for r in 0..h_full[c].len() {
            maxd = maxd.max((h_full[c][r] - h_resumed[c][r]).abs());
        }
    }
    let hsq = build_square_h(&h_full, h_full.len());
    let (lam, _) = dominant_eig(&hsq);
    println!("survival-krylov-selftest: max |H_full - H_resumed| = {:.3e}  (dominant eig of H6 = {:.6})", maxd, lam);
    anyhow::ensure!(maxd < 1e-12, "resume mismatch {:.3e}", maxd);
    println!("survival-krylov-selftest: PASS (resume reproduces H bit-for-bit within {:.1e})", 1e-12);
    Ok(())
}

// within-loop-hazard: exact forward propagation of ONE PC loop from a single boundary state,
// under the stored pi*, tracking the probability mass that dies at each placement ply.
// Cheap + local (one boundary, <=10 layers). total death == d_pi(b); survival == 1 - d_pi(b).
fn run_within_loop_hazard(args: &[String]) -> Result<()> {
    if args.len() < 7 {
        eprintln!("usage: {} within-loop-hazard <graph.bin> <pruned_layer_dir> <policy_dir> <queue7> <bag|size>", args[0]);
        eprintln!("  queue7 = 7 visible pieces (avail {{hold,q0}} + tail5), same convention as eval-one-loop-query.");
        eprintln!("  example: {} within-loop-hazard graph.bin forward_pruned pi_policy_v3 IJLOSTZ IJLOSTZ", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let queue = args[5].clone();
    let bag_arg = args[6].clone();

    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {}", qstate_count); }
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found")?;
    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;
    let policy_files: Vec<File> = (0usize..=9)
        .map(|layer| File::open(mdp_policy_u16_path(&policy_dir, layer))
            .with_context(|| format!("open policy layer {}", layer)))
        .collect::<Result<Vec<_>>>()?;

    let (qs, bag_text) = parse_queue7_to_qstate(&queue, &bag_arg)?;
    let qid = *q_index.key_to_id.get(&qchain_key(qs))
        .with_context(|| format!("qstate not found for queue={} bag={}", queue, bag_arg))? as u32;
    let start_block = find_layer0_start_block(&full_layers[0])?;
    let start_field = full_layers[0].metas[start_block].field_id;

    // block-chains cache per (layer, field) -> sorted qids.
    let mut chains_cache: HashMap<(usize, FieldId), Option<(usize, Vec<u32>)>> = HashMap::new();
    // Returns (block_index, sorted qids) for a field in a layer, or None if the field is absent.
    let mut get_block = |layer: usize, field: FieldId, layers: &[layer_store::FieldChainLayer], cache: &mut HashMap<(usize, FieldId), Option<(usize, Vec<u32>)>>| -> Option<(usize, Vec<u32>)> {
        cache.entry((layer, field)).or_insert_with(|| {
            match layers[layer].metas.binary_search_by_key(&field, |m| m.field_id) {
                Ok(bi) => Some((bi, layers[layer].read_block_chains(bi))),
                Err(_) => None,
            }
        }).clone()
    };
    // policy code for (layer, field, qid); None if state not a member (dead / not PC-able).
    let policy_code = |layer: usize, field: FieldId, qid: u32,
                       block: &(usize, Vec<u32>)| -> Result<Option<u16>> {
        let (bi, chains) = block;
        let pos = match chains.binary_search(&qid) { Ok(p) => p, Err(_) => return Ok(None) };
        let goff = full_layers[layer].metas[*bi].chain_offset + pos as u64;
        let mut buf = [0u8; 2];
        read_exact_at_file(&policy_files[layer], &mut buf, goff * 2)?;
        Ok(Some(u16::from_le_bytes(buf)))
    };

    let mut frontier: HashMap<(FieldId, u32), f64> = HashMap::new();
    frontier.insert((start_field, qid), 1.0);
    let mut hazard = [0.0f64; 11]; // hazard[L] = mass that fails to place the L-th piece
    let mut completed = 0.0f64;
    let mut arrived = [0.0f64; 11]; // mass entering ply L (diagnostics)
    arrived[0] = 1.0;

    for layer in 0usize..=9 {
        if frontier.is_empty() { break; }
        let mut next: HashMap<(FieldId, u32), f64> = HashMap::new();
        let cur: Vec<((FieldId, u32), f64)> = frontier.drain().collect();
        for ((field, qs_id), mass) in cur {
            let blk = match get_block(layer, field, &full_layers, &mut chains_cache) {
                Some(b) => b, None => { hazard[layer] += mass; continue; }
            };
            let code = policy_code(layer, field, qs_id, &blk)?;
            let code = match code { Some(c) => c, None => { hazard[layer] += mass; continue; } };
            if code == POLICY_NONE_U16 { hazard[layer] += mass; continue; }
            let placed = (code >> 8) as u8;
            let ei = (code & 0xFF) as usize;
            let edges = graph.edges(field, placed);
            anyhow::ensure!(ei < edges.len(), "policy edge oob layer={} field={} placed={} ei={}", layer, field, placed, ei);
            let field2 = edges[ei];
            let recs = qfwd.records_for(qs_id, placed);
            anyhow::ensure!(!recs.is_empty(), "policy places piece with no reveal records layer={} qid={}", layer, qs_id);
            let terminal = (layer == 4 && field2 == two_line_field) || (layer == 9 && field2 == EMPTY_FIELD_ID);
            if terminal { completed += mass; continue; }
            anyhow::ensure!(layer < 9, "layer 9 non-terminal action (field2={})", field2);
            let share = mass / recs.len() as f64;
            // Look up next-layer block once for field2.
            let nb = get_block(layer + 1, field2, &full_layers, &mut chains_cache);
            for rec in recs {
                let child = (field2, rec.child_qid);
                let alive = match &nb {
                    Some((_, chains)) => chains.binary_search(&rec.child_qid).is_ok(),
                    None => false,
                };
                if alive { *next.entry(child).or_insert(0.0) += share; }
                else { hazard[layer + 1] += share; } // reveal leads to a dead state at ply L+1
            }
        }
        arrived[layer + 1] = next.values().sum();
        frontier = next;
    }

    let total_death: f64 = hazard.iter().sum();
    let (a, b) = decode_avail_pair(qs.avail);
    println!("within-loop-hazard queue={} bag={} (avail={{{},{}}} tail5={}..)", queue, bag_text, piece_char(a), piece_char(b), qs.tail5);
    println!("start (field_id={}, qid={})  d_pi(b) = {:.6e}   survival = {:.9}", start_field, qid, total_death, 1.0 - total_death);
    println!("ply  arriving_mass     death_hazard     cum_death");
    let mut cum = 0.0f64;
    for l in 0..=9 {
        cum += hazard[l];
        println!("{:>3}  {:>13.6e}  {:>13.6e}  {:>13.6e}", l, arrived[l], hazard[l], cum);
    }
    // any hazard[10] = death discovered arriving at the (non-existent) 11th ply from a bad L9 reveal
    if hazard[10] > 0.0 { println!(" 10  {:>13.6e}  {:>13.6e}  {:>13.6e}", arrived[10], hazard[10], cum + hazard[10]); }
    println!("completed(PC survived) = {:.9}   check death+survive = {:.9}", completed, total_death + completed);
    Ok(())
}

// Forward-enumerate ONE optimal-loop's worth of boundary transitions from a single boundary
// state (empty field + chain qid), over ALL valid placements and ALL 7-bag reveals, and return
// the SET of distinct reset boundary q-states reached at a 2LPC (layer 4) or 4LPC (layer 9)
// terminal. This upper-bounds the per-row fanout of the boundary transition matrix M (the true
// fanout under any fixed policy is <= this). Used to size M for the Policy-Iteration approach.
// Chain-space (board-independent) reachability from a boundary chain: how many DISTINCT chain
// q-states are reachable after exactly `depth` placements (place either element of the {hold,q0}
// pair, reveal any 7-bag piece). A 2LPC loop = 5 placements, a 4LPC loop = 10. The union over
// depths 5 and 10 UPPER-BOUNDS the per-row fanout of the boundary matrix M (board constraints
// only reduce it). Bounded by the 817,740 chain states, so it never explodes.
fn boundary_chain_reach(q_index: &QStateIndex, start_qid: u32) -> (usize, usize) {
    use std::collections::HashSet;
    let mut cur: HashSet<u32> = HashSet::new();
    cur.insert(start_qid);
    let mut reach5 = 0usize;
    let mut union: HashSet<u32> = HashSet::new();
    for step in 1usize..=10 {
        let mut next: HashSet<u32> = HashSet::new();
        for &qid in cur.iter() {
            let s = q_index.states[qid as usize];
            let (pa, pb) = decode_avail_pair(s.avail);
            let placements: [u8; 2] = [pa, pb];
            let placement_count = if pa == pb { 1 } else { 2 };
            for &placed in placements.iter().take(placement_count) {
                for revealed in pieces(s.bag_mask) {
                    if let Some(ns) = q_forward_after_reveal(s, placed, revealed) {
                        if let Some(&id) = q_index.key_to_id.get(&qchain_key(ns)) {
                            next.insert(id as u32);
                        }
                    }
                }
            }
        }
        cur = next;
        if step == 5 { reach5 = cur.len(); for &q in cur.iter() { union.insert(q); } }
        if step == 10 { for &q in cur.iter() { union.insert(q); } }
    }
    (reach5, union.len())
}

// ---------------------------------------------------------------------------
// Boundary transition matrix (M) extraction for Policy Iteration.
//
// For each boundary chain b (empty board = field 0 + chain qid), follow the GREEDY policy
// w.r.t. a full-layer value table V (produced by emit-full-mdp-layers-from-l0) forward through
// one PC loop, branching only on reveals. Mass that completes a PC (2LPC at layer 4 into
// two_line_field, or 4LPC at layer 9 into EMPTY_FIELD_ID) is recorded against the reset chain
// b' it lands on: M(b, b') += prob. Mass with no chosen action dies (failure).
//
// Layer-synchronous wavefront: all origins advance together layer by layer, so V(layer+1) is
// RAM-pinned once per layer (no random access into the 321 GiB V table). Frontier states with
// V == 0 are dropped at push time: under the emit-produced V (one-loop-seeded, monotone), a
// pruned-layer member that can still PC has V > 0, so V == 0 mass can never reach a terminal.
//
// Policy argmax replicates the sweep exactly (same candidate order, strict `cand > best` from
// best = 0.0), so V^pi from solving (I - gamma*M)V = c matches the sweep's greedy semantics.
fn run_extract_boundary_m(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} extract-boundary-m <graph.bin> <pruned_layer_dir> <full_v_dir> <out_dir> [origin_start=0] [origin_count=all] [gamma=1.0] [pin_max_gib=96]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let full_v_dir = PathBuf::from(&args[4]);
    let out_dir = PathBuf::from(&args[5]);
    let origin_start: usize = if args.len() >= 7 { args[6].parse().context("invalid origin_start")? } else { 0 };
    let origin_count_arg: Option<usize> = if args.len() >= 8 { Some(args[7].parse().context("invalid origin_count")?) } else { None };
    let gamma: f32 = if args.len() >= 9 { args[8].parse().context("invalid gamma")? } else { 1.0 };
    let pin_max_gib: u64 = if args.len() >= 10 { args[9].parse().context("invalid pin_max_gib")? } else { 96 };
    fs::create_dir_all(&out_dir)?;

    let total_start = Instant::now();
    eprintln!("extract-boundary-m loading graph from {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qfwd = build_qforward_graph(&q_index);
    let qstate_count = q_index.states.len();
    if qstate_count != CHECK_QSTATE_COUNT { anyhow::bail!("unexpected qstate count {}", qstate_count); }
    let legal_placed = build_qfwd_legal_placed_table(&qfwd, qstate_count);
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;

    let full_layers: Vec<layer_store::FieldChainLayer> = (0usize..=9)
        .map(|layer| layer_store::FieldChainLayer::open_index_only(q_pruned_prefix(&pruned_dir, layer)))
        .collect::<Result<Vec<_>>>()?;

    // Terminal reset table from the SAME layer0 V the emit used as its reset source.
    let l0_path = mdp_f32_value_path(&full_v_dir, 0);
    let reset_values = load_layer0_reset_dense(&full_layers[0], &l0_path, qstate_count)?;
    let terminal_reset_avg = build_terminal_reset_avg_table(&qfwd, &reset_values, gamma, qstate_count);

    let origin_end = origin_count_arg.map(|c| (origin_start + c).min(qstate_count)).unwrap_or(qstate_count);
    let n_origins = origin_end.saturating_sub(origin_start);
    if n_origins == 0 { anyhow::bail!("empty origin range"); }
    eprintln!("extract-boundary-m origins=[{},{}) n={} gamma={} two_line_field={} out={}",
        origin_start, origin_end, n_origins, gamma, two_line_field, out_dir.display());

    // Frontier: (field, qid) -> list of (origin_local, prob). origin_local = origin - origin_start.
    let mut frontier: HashMap<(u32, u32), Vec<(u32, f32)>> = HashMap::new();
    for o in origin_start..origin_end {
        frontier.insert((0u32, o as u32), vec![((o - origin_start) as u32, 1.0f32)]);
    }
    // Terminal triples (origin_local, reset_child_qid, prob), accumulated across layers 4 and 9.
    let mut triples: Vec<(u32, u32, f32)> = Vec::new();
    let pin_max_bytes = pin_max_gib.saturating_mul(1024 * 1024 * 1024);

    for layer in 0usize..=9 {
        if frontier.is_empty() { break; }
        let layer_start = Instant::now();

        // Pin V(layer+1) when the policy needs continuation values.
        let next_values_holder: Option<Vec<f32>> = if layer < 9 {
            let nl = &full_layers[layer + 1];
            let bytes = nl.total_states.checked_mul(4).context("v size overflow")?;
            if bytes > pin_max_bytes { anyhow::bail!("layer {} V ({} bytes) exceeds pin_max_gib={}; raise it or batch origins", layer + 1, bytes, pin_max_gib); }
            let p = mdp_f32_value_path(&full_v_dir, layer + 1);
            Some(load_f32_values_vec(&p, nl.total_states).with_context(|| format!("failed to load {}", p.display()))?)
        } else { None };

        // Drain frontier, merge duplicate (state, origin) mass, and process in parallel.
        let mut states: Vec<((u32, u32), Vec<(u32, f32)>)> = frontier.drain().collect();
        states.par_iter_mut().for_each(|(_, origins)| {
            origins.sort_unstable_by_key(|&(o, _)| o);
            let mut w = 0usize;
            for r in 0..origins.len() {
                if w > 0 && origins[w - 1].0 == origins[r].0 { origins[w - 1].1 += origins[r].1; }
                else { origins[w] = origins[r]; w += 1; }
            }
            origins.truncate(w);
        });
        let tuple_count: usize = states.iter().map(|(_, v)| v.len()).sum();

        let next_layer_ref = if layer < 9 { Some(&full_layers[layer + 1]) } else { None };
        let next_values_ref = next_values_holder.as_deref();

        // Per-thread outputs: (next frontier tuples, terminal triples, dead mass, terminal mass).
        type ThreadOut = (Vec<((u32, u32), (u32, f32))>, Vec<(u32, u32, f32)>, f64, f64);
        let outs: Vec<ThreadOut> = states.par_chunks(1024).map(|chunk| {
            let mut lookup_opt = next_values_ref.map(|values| {
                RamNextValueLookup::new(next_layer_ref.unwrap(), values, 256 * 1024 * 1024, false, false)
                    .expect("ram lookup")
            });
            let mut next_out: Vec<((u32, u32), (u32, f32))> = Vec::new();
            let mut term_out: Vec<(u32, u32, f32)> = Vec::new();
            let mut dead = 0.0f64;
            let mut term_mass = 0.0f64;
            for ((field, qid), origins) in chunk {
                let (field, qid) = (*field, *qid);
                // --- policy argmax: EXACT replica of the sweep's candidate scan ---
                let mut best = 0.0f32;
                let mut chosen: Option<(u8, u32, bool)> = None; // (placed, field2, terminal)
                for placed in legal_placed[qid as usize] {
                    if placed == 255 { continue; }
                    let recs = qfwd.records_for(qid, placed);
                    if recs.is_empty() { continue; }
                    let edges = graph.edges(field, placed);
                    if edges.is_empty() { continue; }
                    let denom = recs.len() as f32;
                    for &field2 in edges {
                        let (cand, terminal) = if layer == 4 && field2 == two_line_field {
                            (terminal_reset_avg[qid as usize][placed as usize], true)
                        } else if layer == 9 {
                            if field2 == EMPTY_FIELD_ID {
                                (terminal_reset_avg[qid as usize][placed as usize], true)
                            } else { continue; }
                        } else {
                            let lookup = lookup_opt.as_mut().expect("lookup");
                            let (s, _miss) = lookup.sum_recs(field2, recs).expect("sum_recs");
                            (if gamma == 1.0 { s / denom } else { gamma * s / denom }, false)
                        };
                        if cand > best { best = cand; chosen = Some((placed, field2, terminal)); }
                    }
                }
                let total_p: f64 = origins.iter().map(|&(_, p)| p as f64).sum();
                let Some((placed, field2, terminal)) = chosen else { dead += total_p; continue; };
                let recs = qfwd.records_for(qid, placed);
                let share = 1.0f32 / recs.len() as f32;
                if terminal {
                    term_mass += total_p;
                    for rec in recs {
                        for &(o, p) in origins.iter() {
                            term_out.push((o, rec.child_qid, p * share));
                        }
                    }
                } else {
                    let lookup = lookup_opt.as_mut().expect("lookup");
                    for rec in recs {
                        // V==0 ⇒ this reveal-child can never PC; its mass is dead. Drop early.
                        let v = lookup.get(field2, rec.child_qid).expect("get");
                        if v == 0.0 {
                            dead += total_p * share as f64;
                            continue;
                        }
                        for &(o, p) in origins.iter() {
                            next_out.push(((field2, rec.child_qid), (o, p * share)));
                        }
                    }
                }
            }
            (next_out, term_out, dead, term_mass)
        }).collect();

        let mut next_frontier: HashMap<(u32, u32), Vec<(u32, f32)>> = HashMap::new();
        let mut layer_dead = 0.0f64;
        let mut layer_term = 0.0f64;
        let mut layer_next_tuples = 0usize;
        for (next_out, term_out, dead, term) in outs {
            layer_dead += dead; layer_term += term;
            layer_next_tuples += next_out.len();
            for (k, v) in next_out { next_frontier.entry(k).or_default().push(v); }
            triples.extend(term_out);
        }
        eprintln!("extract-boundary-m layer={} states={} tuples_in={} next_states={} next_tuples={} term_mass={:.6} dead_mass={:.6} triples_total={} elapsed={:.1}s",
            layer, states.len(), tuple_count, next_frontier.len(), layer_next_tuples, layer_term, layer_dead, triples.len(), layer_start.elapsed().as_secs_f64());
        frontier = next_frontier;
    }

    // Merge-sum triples per (origin, child) and write CSR.
    let merge_start = Instant::now();
    triples.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut merged: Vec<(u32, u32, f64)> = Vec::with_capacity(triples.len() / 2);
    for &(o, c, p) in &triples {
        if let Some(last) = merged.last_mut() {
            if last.0 == o && last.1 == c { last.2 += p as f64; continue; }
        }
        merged.push((o, c, p as f64));
    }
    drop(triples);

    let idx_path = out_dir.join(format!("m_{}_{}.idx.bin", origin_start, origin_end));
    let dat_path = out_dir.join(format!("m_{}_{}.rows.bin", origin_start, origin_end));
    let mut idx = BufWriter::new(File::create(&idx_path)?);
    let mut dat = BufWriter::new(File::create(&dat_path)?);
    let mut row_off = 0u64;
    let mut mi = 0usize;
    let mut c_min = f64::INFINITY; let mut c_max = 0.0f64; let mut c_sum = 0.0f64;
    let mut fanout_max = 0u32; let mut fanout_sum = 0u64;
    for local in 0..n_origins as u32 {
        let start_i = mi;
        let mut csum = 0.0f64;
        while mi < merged.len() && merged[mi].0 == local {
            dat.write_all(&merged[mi].1.to_le_bytes())?;
            dat.write_all(&(merged[mi].2 as f32).to_le_bytes())?;
            csum += merged[mi].2;
            mi += 1;
        }
        let count = (mi - start_i) as u32;
        idx.write_all(&row_off.to_le_bytes())?;
        idx.write_all(&count.to_le_bytes())?;
        idx.write_all(&csum.to_le_bytes())?;
        row_off += count as u64;
        c_min = c_min.min(csum); c_max = c_max.max(csum); c_sum += csum;
        fanout_max = fanout_max.max(count); fanout_sum += count as u64;
    }
    idx.flush()?; dat.flush()?;
    eprintln!("extract-boundary-m merged nnz={} write done (merge+write {:.1}s)", merged.len(), merge_start.elapsed().as_secs_f64());
    println!("extract-boundary-m origins=[{},{}) nnz={} avg_fanout={:.1} max_fanout={} c_avg={:.6} c_min={:.6} c_max={:.6} bytes_rows={} elapsed={:.1}s",
        origin_start, origin_end, merged.len(), fanout_sum as f64 / n_origins as f64, fanout_max,
        c_sum / n_origins as f64, c_min, c_max, row_off * 8, total_start.elapsed().as_secs_f64());
    println!("  idx={} rows={}", idx_path.display(), dat_path.display());
    Ok(())
}

fn run_measure_boundary_fanout(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} measure-boundary-fanout <graph.bin> <sample_n> [seed=1] [max_states_per_boundary=500000]", args[0]);
        std::process::exit(2);
    }
    let graph_path = PathBuf::from(&args[2]);
    let sample_n: usize = args[3].parse().context("invalid sample_n")?;
    let seed: u64 = if args.len() >= 5 { args[4].parse().context("invalid seed")? } else { 1 };
    let max_states: usize = if args.len() >= 6 { args[5].parse().context("invalid max_states")? } else { 500_000 };

    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old_index = enumerate_states();
    let q_index = build_q_index_from_old(&old_index);
    let qstate_count = q_index.states.len();
    let two_line_field = graph.hash_lookup(crate::graph::TWO_LINE_HASH).context("TWO_LINE_HASH not found in graph")?;
    eprintln!("measure-boundary-fanout qstates={} two_line_field={} sample_n={} seed={} max_states_per_boundary={}",
        qstate_count, two_line_field, sample_n, seed, max_states);
    // Geometry diagnostic: which field is the empty START (has forward edges)?
    let deg = |f: FieldId| -> usize { (0..7u8).map(|p| graph.edges(f, p).len()).sum() };
    eprintln!("geom: EMPTY_FIELD_ID={} hash={:#x} outdeg={}", EMPTY_FIELD_ID, graph.hash(EMPTY_FIELD_ID), deg(EMPTY_FIELD_ID));
    eprintln!("geom: field0 hash={:#x} outdeg={}", graph.hash(0), deg(0));
    eprintln!("geom: two_line_field={} hash={:#x} outdeg={}", two_line_field, graph.hash(two_line_field), deg(two_line_field));
    let hash0_field = (0..NUM_FIELDS as FieldId).find(|&f| graph.hash(f) == 0);
    if let Some(f) = hash0_field { eprintln!("geom: field-with-hash0 = {} outdeg={}", f, deg(f)); }

    // Sample distinct boundary qids via a xorshift64 PRNG (no rand dependency).
    let mut x = seed | 1;
    let mut sample: Vec<u32> = Vec::with_capacity(sample_n);
    let mut seen = std::collections::HashSet::<u32>::new();
    let want = sample_n.min(qstate_count);
    while sample.len() < want {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let qid = (x % qstate_count as u64) as u32;
        if seen.insert(qid) { sample.push(qid); }
    }

    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let results: Vec<(usize, usize)> = sample.par_iter().map(|&qid| {
        let r = boundary_chain_reach(&q_index, qid);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 200 == 0 { eprintln!("  measured {}/{} elapsed={:.1}s", d, want, start.elapsed().as_secs_f64()); }
        r
    }).collect();

    // r.0 = reach at 5 placements (2LPC reset chains), r.1 = |reach5 ∪ reach10| = fanout upper bound.
    let mut fanouts: Vec<usize> = results.iter().map(|r| r.1).collect();
    fanouts.sort_unstable();
    let n = fanouts.len();
    let sum: u128 = fanouts.iter().map(|&v| v as u128).sum();
    let avg = sum as f64 / n as f64;
    let median = fanouts[n / 2];
    let p90 = fanouts[(n * 9 / 10).min(n - 1)];
    let max = *fanouts.last().unwrap();
    let avg5 = results.iter().map(|r| r.0 as u128).sum::<u128>() as f64 / n as f64;

    println!("boundary-fanout-UB (chain-reach, board-independent) samples={} avg_reach5={:.0} avg_fanout_ub={:.0} median={} p90={} max={} elapsed={:.1}s",
        n, avg5, avg, median, p90, max, start.elapsed().as_secs_f64());
    let total_nnz = avg * qstate_count as f64;
    for bpn in [6.0f64, 8.0] {
        println!("extrapolated M (upper bound): boundaries={} avg_fanout_ub={:.0} total_nnz~={:.3e} size~={:.1} GB (at {:.0} B/nnz)",
            qstate_count, avg, total_nnz, total_nnz * bpn / 1e9, bpn);
    }
    let gb = total_nnz * 8.0 / 1e9;
    println!("interpretation (UB): fits in {}", if gb < 100.0 { "RAM (node0 204GB) -> in-core PI/GMRES ideal" } else if gb < 800.0 { "NVMe out-of-core (streamed) -> still viable" } else { "large; true (board-constrained) fanout is smaller -> measure exactly before deciding" });
    Ok(())
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn parse_piece_string_mask(s: &str) -> Result<u8> {
    let mut mask = 0u8;
    for ch in s.chars() {
        let p = piece_from_char(ch).with_context(|| format!("invalid piece character '{ch}' in '{s}'"))?;
        mask |= 1u8 << p;
    }
    if mask == 0 {
        anyhow::bail!("empty bag mask from '{s}'");
    }
    Ok(mask)
}

fn parse_queue7_to_qstate(queue: &str, bag_arg: &str) -> Result<(QChainState, String)> {
    let qchars = queue.chars().collect::<Vec<_>>();
    if qchars.len() != 7 {
        anyhow::bail!("queue must have length 7 for see=7, got len={} queue={}", qchars.len(), queue);
    }

    let mut q = [0u8; 7];
    let mut seen_mask = 0u8;
    for (i, &ch) in qchars.iter().enumerate() {
        let p = piece_from_char(ch).with_context(|| format!("invalid queue piece '{ch}' in '{queue}'"))?;
        q[i] = p;
        seen_mask |= 1u8 << p;
    }

    let (bag_mask, bag_text) = if bag_arg.len() == 1 && bag_arg.as_bytes()[0].is_ascii_digit() {
        let n = bag_arg.parse::<u32>().with_context(|| format!("invalid bag size '{bag_arg}'"))?;
        if !(1..=7).contains(&n) {
            anyhow::bail!("bag size digit must be in 1..=7, got {n}");
        }
        let inferred = FULL_BAG & !seen_mask;
        let pop = inferred.count_ones();
        if pop != n {
            anyhow::bail!(
                "cannot infer bag of size {n} from queue {queue}: complement mask={} has popcount={pop}",
                mask_to_string(inferred)
            );
        }
        (inferred, mask_to_string(inferred))
    } else {
        let mask = parse_piece_string_mask(bag_arg)?;
        (mask, mask_to_string(mask))
    };

    let s = QChainState {
        avail: encode_avail_pair(q[0], q[1]),
        tail5: encode_tail5([q[2], q[3], q[4], q[5], q[6]]),
        bag_mask,
    };
    Ok((s, bag_text))
}


fn find_layer0_start_block(layer0: &layer_store::FieldChainLayer) -> Result<usize> {
    if layer0.metas.len() == 1 {
        return Ok(0);
    }
    if let Ok(i) = layer0.metas.binary_search_by_key(&EMPTY_FIELD_ID, |m| m.field_id) {
        return Ok(i);
    }
    if let Ok(i) = layer0.metas.binary_search_by_key(&0u32, |m| m.field_id) {
        return Ok(i);
    }
    anyhow::bail!(
        "could not identify layer0 start field block: metas_len={} EMPTY_FIELD_ID={} field0_absent",
        layer0.metas.len(),
        EMPTY_FIELD_ID
    );
}

// Duplicated-mino statistics on layer0. The visible 7 (avail 2 + tail5) split at the
// queue-to-bag boundary: the LAST (7-popcnt) pieces are the already-revealed head of the
// current reveal bag (distinct, = complement of bag_mask), and the FIRST popcnt pieces
// (hold + queue0 + tail[0..popcnt-2]) are the old-bag segment. A type appearing in both
// segments is natural cross-bag repetition, NOT a duplication. A true duplication is
// multiplicity >= 2 WITHIN the first popcnt pieces: the old-bag queue tail is distinct by
// construction, so this happens only when hold retained a piece from an even older bag
// (displacing a bag piece consumed early). popcnt=1 (3회차) has a 1-piece front segment,
// so duplication is structurally impossible there. Reports, per bag popcount (loop-cycle
// position: popcnt 7=1회차, 4=2, 1=3, 5=4, 2=5, 6=6, 3=7) and per duplicated mino, the
// one-loop PC probability stats.
// usage: dump-l0-oneloop-dup-stats <pruned_layer_dir> <l0_values.u16> [scale=5040]
// Average of a layer0 f32 value file over states whose 7 VISIBLE pieces (hold + queue0 from the
// avail pair, plus tail5 = queue1..5) are ALL DISTINCT (no duplicate piece type). With 7 pieces
// and 7 types, "all distinct" == the 7 pieces are a full permutation of IJLOSTZ.
fn run_layer0_nodupe7_avg_f32(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} layer0-nodupe7-avg-f32 <pruned_layer_dir> <l0_values.f32>", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 4, "f32 size mismatch: got {} expected {}", bytes.len(), layer0.total_states * 4);

    let (mut nd_count, mut nd_sum) = (0u64, 0.0f64);
    let (mut all_count, mut all_sum) = (0u64, 0.0f64);
    let (mut nd_min, mut nd_max) = (f32::INFINITY, f32::NEG_INFINITY);
    // bag_mask == 127 (all 7 bag bits set, popcnt 7 = fresh-bag / round 1) breakdowns
    let (mut bm_count, mut bm_sum) = (0u64, 0.0f64);
    let (mut bmnd_count, mut bmnd_sum) = (0u64, 0.0f64);
    let (mut bmnd_min, mut bmnd_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for (i, &qid) in chains.iter().enumerate() {
        let off = (meta.chain_offset as usize + i) * 4;
        let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let qs = q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut seen: u8 = (1 << a) | (1 << b);
        for &p in &tail { seen |= 1 << p; }
        let nodupe = seen.count_ones() == 7;
        let bm127 = qs.bag_mask == 127;
        all_count += 1;
        all_sum += v as f64;
        if nodupe {
            nd_count += 1;
            nd_sum += v as f64;
            nd_min = nd_min.min(v);
            nd_max = nd_max.max(v);
        }
        if bm127 {
            bm_count += 1;
            bm_sum += v as f64;
            if nodupe {
                bmnd_count += 1;
                bmnd_sum += v as f64;
                bmnd_min = bmnd_min.min(v);
                bmnd_max = bmnd_max.max(v);
            }
        }
    }
    println!("layer0-nodupe7-avg-f32 file={}", values_path.display());
    println!("  ALL states:              count={} avg={:.6}", all_count, all_sum / all_count.max(1) as f64);
    println!("  NO-DUPE-7:               count={} ({:.4}%) avg={:.6} min={:.6} max={:.6}",
        nd_count, 100.0 * nd_count as f64 / all_count.max(1) as f64,
        nd_sum / nd_count.max(1) as f64,
        if nd_count == 0 { 0.0 } else { nd_min }, if nd_count == 0 { 0.0 } else { nd_max });
    println!("  DUPE:                   count={} avg={:.6}",
        all_count - nd_count, (all_sum - nd_sum) / (all_count - nd_count).max(1) as f64);
    println!("  bag_mask==127 (round1): count={} ({:.4}%) avg={:.6}",
        bm_count, 100.0 * bm_count as f64 / all_count.max(1) as f64, bm_sum / bm_count.max(1) as f64);
    println!("  NO-DUPE-7 & bag==127:   count={} ({:.4}%) avg={:.6} min={:.6} max={:.6}",
        bmnd_count, 100.0 * bmnd_count as f64 / all_count.max(1) as f64,
        bmnd_sum / bmnd_count.max(1) as f64,
        if bmnd_count == 0 { 0.0 } else { bmnd_min }, if bmnd_count == 0 { 0.0 } else { bmnd_max });
    Ok(())
}

fn run_dump_l0_oneloop_dup_stats(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-oneloop-dup-stats <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    impl Agg { fn new() -> Self { Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX } } }
    let mut agg = [[Agg::new(); 7]; 8];      // [popcnt][piece]
    let mut nodup = [Agg::new(); 8];         // [popcnt] states with no duplicate
    let mut multi_dup_states = [0u64; 8];    // states with >=2 distinct duplicated minos

    for (i, &qid) in chains.iter().enumerate() {
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones() as usize;
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        // Only the FIRST popcnt visible pieces (old-bag segment: hold + queue0 +
        // tail[0..popcnt-2]) can carry a true duplication. popcnt=1 -> front segment is
        // the hold alone -> never duplicated.
        let mut mult = [0u8; 7];
        if pc >= 2 {
            mult[a as usize] += 1;
            mult[b as usize] += 1;
            for &p in tail.iter().take(pc - 2) { mult[p as usize] += 1; }
        }
        let mut dup_types = 0usize;
        for piece in 0..7 {
            if mult[piece] >= 2 {
                dup_types += 1;
                let e = &mut agg[pc][piece];
                e.count += 1;
                e.sum += raw as f64;
                if raw >= scale { e.full += 1; }
                e.min_raw = e.min_raw.min(raw);
            }
        }
        if dup_types == 0 {
            let e = &mut nodup[pc];
            e.count += 1;
            e.sum += raw as f64;
            if raw >= scale { e.full += 1; }
            e.min_raw = e.min_raw.min(raw);
        }
        if dup_types >= 2 { multi_dup_states[pc] += 1; }
    }

    // popcnt -> loop-cycle position (회차): 10 pieces/loop, popcnt_next = (popcnt+4) mod 7
    let round_of = |pc: usize| -> usize { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };
    let mut order: Vec<usize> = (1..8).collect();
    order.sort_by_key(|&pc| round_of(pc));
    for pc in order {
        println!("회차={} (popcnt={}) no_dup: count={} avg_p={:.9} full={} ({:.4}%) min_raw={} | multi_dup_states={}",
            round_of(pc), pc, nodup[pc].count,
            if nodup[pc].count == 0 { 0.0 } else { nodup[pc].sum / nodup[pc].count as f64 / scale as f64 },
            nodup[pc].full,
            if nodup[pc].count == 0 { 0.0 } else { 100.0 * nodup[pc].full as f64 / nodup[pc].count as f64 },
            if nodup[pc].count == 0 { 0 } else { nodup[pc].min_raw },
            multi_dup_states[pc]);
        for piece in 0..7 {
            let e = &agg[pc][piece];
            if e.count == 0 { continue; }
            println!("  dup={} count={:>7} avg_p={:.9} full={:>7} ({:.4}%) min_raw={}",
                piece_char(piece as u8), e.count, e.sum / e.count as f64 / scale as f64,
                e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
        }
    }
    Ok(())
}

// Same-queue O>T-8회차 vs 1회차 comparison. Each 1회차 (popcnt=7 no-dup, full bag) state with T
// in avail maps to an O>T 8회차 state by swapping that T -> O (avail {T,X} -> {O,X}, tail
// unchanged): OX|rest5 vs TX|rest5. Report pairs where the O>T version has HIGHER one-loop prob.
// usage: dump-l0-ot8-vs-round1 <pruned_layer_dir> <l0_values.u16> [scale=5040] [top_n=all]
fn run_dump_l0_ot8_vs_round1(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-ot8-vs-round1 <pruned_layer_dir> <l0_values.u16> [scale=5040] [top_n=all]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let top_n: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
    let find = |c: char| (0u8..7).find(|&p| piece_char(p) == c).unwrap();
    let (po, pt) = (find('O'), find('T'));

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");
    let raw_of_qid = |qid: u32| -> Option<u32> {
        let pos = chains.binary_search(&qid).ok()?;
        let off = (meta.chain_offset as usize + pos) * 2;
        Some(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32)
    };

    let (mut pairs, mut better, mut worse, mut equal, mut missing) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut hits: Vec<(i64, u32, u32, u32)> = Vec::new(); // (diff=ot8-r1, r1_raw, ot8_raw, qid_r1)
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 7 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        // 1회차 (no-dup, full bag) with T in avail
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1; mult[b as usize] += 1;
        for &p in tail.iter() { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; } // no-dup only (1회차)
        let x = if a == pt { b } else if b == pt { a } else { continue }; // T in avail; other = X
        // map T->O: avail {O,X}
        let mqs = QChainState { avail: encode_avail_pair(po, x), tail5: qs.tail5, bag_mask: qs.bag_mask };
        let Some(&mid) = q_index.key_to_id.get(&qchain_key(mqs)) else { missing += 1; continue; };
        let off = (meta.chain_offset as usize + i) * 2;
        let r1 = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let Some(ot8) = raw_of_qid(mid as u32) else { missing += 1; continue; };
        pairs += 1;
        if ot8 > r1 { better += 1; hits.push((ot8 as i64 - r1 as i64, r1, ot8, qid)); }
        else if ot8 < r1 { worse += 1; }
        else { equal += 1; }
    }
    hits.sort_unstable_by(|x, y| y.0.cmp(&x.0));
    println!("O>T 8회차 vs 1회차 (same queue): pairs={} better(O>T higher)={} worse={} equal={} mapped_missing={}",
        pairs, better, worse, equal, missing);
    for &(diff, r1, ot8, qid) in hits.iter().take(top_n) {
        let qs = q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
        let x = if a == pt { b } else { a };
        println!("  1회차 {{{},{}}}|{}  T:{}/{}  ->  O>T 8회차 {{O,{}}}|{}  O:{}/{}  diff=+{}",
            piece_char(a), piece_char(b), tail_s, r1, scale, piece_char(x), tail_s, ot8, scale, diff);
    }
    Ok(())
}

// Per-회차 / per-종류 aggregate sheet: for every layer0 boundary state compute V* and
// death-in-n (n=1,2,7) via the survival Krylov projection, classify into (round, type) where
// type = the front old-bag multiset (visible 7 minus the fresh new-bag reveals = complement of
// bag_mask, one each), and emit mean V / mean d1 / mean d2 / mean d7 per type. Label follows the
// user notation: pc7 full|A>B, pc5/6 "no.."|A>.., others plain letters. dup flag marks duplicates.
// usage: dump-round-sheet <pruned_layer_dir> <vstar.f32> <survival_art_dir>
fn run_dump_round_sheet(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        eprintln!("usage: {} dump-round-sheet <pruned_layer_dir> <vstar.f32> <survival_art_dir>", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let vstar_path = PathBuf::from(&args[3]);
    let art_dir = PathBuf::from(&args[4]);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0 = layer_store::FieldChainLayer::open_index_only(&pruned_dir.join("layer0q_pruned"))?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let nb = layer0.total_states as usize;
    anyhow::ensure!(chains.len() == nb && meta.chain_offset == 0,
        "expected single reset block spanning all states (chains={} nb={} off={})", chains.len(), nb, meta.chain_offset);
    let vstar = load_f32_values_vec(&vstar_path, nb as u64)?;

    // survival projection: p_die(s,n) = sum_i w_n[i] * basis[i][s], w_n = beta*sum_{j<n} H^j e1
    let st = read_survival_state(&art_dir.join("state.bin"))?;
    anyhow::ensure!(st.n == nb, "survival state n={} != nb={}", st.n, nb);
    let m = st.h.len();
    let hsq = build_square_h(&st.h, m);
    let horizons = [1usize, 2, 7];
    let mut ws: Vec<Vec<f64>> = Vec::new();
    {
        let mut z = vec![0.0f64; m]; z[0] = 1.0;
        let mut w = vec![0.0f64; m];
        let maxh = *horizons.iter().max().unwrap();
        let mut wn: Vec<Vec<f64>> = vec![vec![]; horizons.len()];
        for step in 0..maxh {
            for i in 0..m { w[i] += st.beta * z[i]; }
            for (hi, &h) in horizons.iter().enumerate() { if step + 1 == h { wn[hi] = w.clone(); } }
            z = hsq_matvec(&hsq, &z);
        }
        ws = wn;
    }
    let bf = fs::read(art_dir.join("basis.f64.bin"))?;
    anyhow::ensure!(bf.len() >= m * nb * 8, "basis too small");
    let mut d = [vec![0.0f64; nb], vec![0.0f64; nb], vec![0.0f64; nb]];
    for i in 0..m {
        let base = i * nb * 8;
        let (w0, w1, w2) = (ws[0][i], ws[1][i], ws[2][i]);
        for s in 0..nb {
            let bv = f64::from_le_bytes(bf[base + s * 8..base + s * 8 + 8].try_into().unwrap());
            d[0][s] += w0 * bv; d[1][s] += w1 * bv; d[2][s] += w2 * bv;
        }
    }

    // aggregate by (round, type_label). round = f(popcnt); dup separated by label content.
    let round_of = |pc: u32| -> u32 { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };
    struct Agg { round: u32, dup: bool, count: u64, sv: f64, sd: [f64; 3] }
    let mut map: HashMap<(u32, String), Agg> = HashMap::new();
    for (s, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones();
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        // front old-bag multiset = visible - fresh new-bag reveals (complement of bag_mask, one each)
        let mut cnt = [0u8; 7];
        cnt[a as usize] += 1; cnt[b as usize] += 1;
        for &p in &tail { cnt[p as usize] += 1; }
        let fresh = FULL_BAG & !qs.bag_mask;
        for p in 0u8..7 { if fresh & (1 << p) != 0 { cnt[p as usize] = cnt[p as usize].saturating_sub(1); } }
        debug_assert_eq!(cnt.iter().map(|&c| c as u32).sum::<u32>(), pc);
        let dup = cnt.iter().any(|&c| c >= 2);
        let letters: String = (0u8..7).flat_map(|p| std::iter::repeat(piece_char(p)).take(cnt[p as usize] as usize)).collect();
        let dup_piece = (0u8..7).find(|&p| cnt[p as usize] >= 2).map(piece_char);
        let missing: String = (0u8..7).filter(|&p| cnt[p as usize] == 0).map(piece_char).collect();
        let label = match pc {
            7 => if dup { format!("{}>{}", dup_piece.unwrap(), missing) } else { "full".to_string() },
            5 | 6 => if dup { format!("{}>{}", dup_piece.unwrap(), missing) } else { format!("no {}", missing) },
            _ => letters,
        };
        let round = if pc == 7 && dup { 8 } else { round_of(pc) };
        let e = map.entry((round, label)).or_insert(Agg { round, dup, count: 0, sv: 0.0, sd: [0.0; 3] });
        e.count += 1; e.sv += vstar[s] as f64;
        for k in 0..3 { e.sd[k] += d[k][s].clamp(0.0, 1.0); }
    }

    println!("round,type,dup,count,meanV,death1,death2,death7");
    let mut rows: Vec<(&(u32, String), &Agg)> = map.iter().collect();
    rows.sort_by(|x, y| x.1.round.cmp(&y.1.round).then(x.0 .1.cmp(&y.0 .1)));
    let mut total = 0u64;
    for ((_r, label), e) in rows {
        total += 1;
        let c = e.count as f64;
        println!("{},{},{},{},{:.4},{:.6e},{:.6e},{:.6e}",
            e.round, label, if e.dup { 1 } else { 0 }, e.count,
            e.sv / c, e.sd[0] / c, e.sd[1] / c, e.sd[2] / c);
    }
    eprintln!("total distinct (round,type) = {}", total);
    Ok(())
}

// Emit CSV of every no-dup full-bag 1회차/O>T same-queue pair: outcome,X,tail,r1,ot8.
// outcome: B=O>T higher, W=1회차 higher, E=equal. For downstream (python) analysis of all 720.
// usage: dump-l0-ot8-csv <pruned_layer_dir> <l0_values.u16>
fn run_dump_l0_ot8_csv(args: &[String]) -> Result<()> {
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    // Piece that replaces the missing piece in avail (the duplicated piece D). Default O.
    let dup_char = args.get(4).map(|s| s.to_uppercase()).unwrap_or_else(|| "O".to_string());
    // The missing piece M (must be present in avail of the 1회차; gets replaced by D). Default T.
    let miss_char = args.get(5).map(|s| s.to_uppercase()).unwrap_or_else(|| "T".to_string());
    let pfind = |c: &str| (0u8..7).find(|&p| piece_char(p).to_string() == c)
        .with_context(|| format!("unknown piece {}", c));
    let pdup = pfind(&dup_char)?;
    let pt = pfind(&miss_char)?;  // pt now = missing piece M (avail-required slot)
    anyhow::ensure!(pdup != pt, "dup piece D must differ from missing piece M");
    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0 = layer_store::FieldChainLayer::open_index_only(&pruned_dir.join("layer0q_pruned"))?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path)?;
    // Auto-detect u16 (one-loop, *2) or f32 (V*/policy value, *4) by file length.
    let n = layer0.total_states;
    let is_f32 = bytes.len() as u64 == n * 4;
    anyhow::ensure!(is_f32 || bytes.len() as u64 == n * 2, "value file size matches neither u16 nor f32");
    let val_at = |pos: usize| -> f64 {
        if is_f32 {
            let o = pos * 4;
            f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64
        } else {
            let o = pos * 2;
            u16::from_le_bytes([bytes[o], bytes[o + 1]]) as f64
        }
    };
    let raw_of_qid = |qid: u32| -> Option<f64> {
        let pos = chains.binary_search(&qid).ok()?;
        Some(val_at(meta.chain_offset as usize + pos))
    };
    eprintln!("# 8회차 {}>{} (dup={} missing={}) vs 1회차  fmt={}",
        dup_char, miss_char, dup_char, miss_char, if is_f32 { "f32" } else { "u16" });
    println!("outcome,X,tail,r1,ot8");
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 7 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1; mult[b as usize] += 1;
        for &p in tail.iter() { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; }
        let x = if a == pt { b } else if b == pt { a } else { continue };
        // X must differ from the dup piece? No -- X==dup means dup appears in avail twice, still valid.
        let mqs = QChainState { avail: encode_avail_pair(pdup, x), tail5: qs.tail5, bag_mask: qs.bag_mask };
        let Some(&mid) = q_index.key_to_id.get(&qchain_key(mqs)) else { continue; };
        let r1 = val_at(meta.chain_offset as usize + i);
        let Some(ot8) = raw_of_qid(mid as u32) else { continue; };
        let oc = if ot8 > r1 { 'B' } else if ot8 < r1 { 'W' } else { 'E' };
        let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
        println!("{},{},{},{:.6},{:.6}", oc, piece_char(x), tail_s, r1, ot8);
    }
    Ok(())
}

// Structural profile of the O>T 8회차 vs 1회차 same-queue comparison. For every no-dup full-bag
// 1회차 state with T in avail, we map T->O in avail (queue identical) and bucket the pair as
// better/worse/equal (O>T raw vs 1회차 raw). Then per bucket we tabulate: the avail partner X,
// per-tail-position piece frequency, and where O sits in the 1회차 queue (avail if X=O, else tail idx).
// usage: dump-l0-ot8-structure <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_ot8_structure(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-ot8-structure <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let find = |c: char| (0u8..7).find(|&p| piece_char(p) == c).unwrap();
    let (po, pt) = (find('O'), find('T'));

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");
    let raw_of_qid = |qid: u32| -> Option<u32> {
        let pos = chains.binary_search(&qid).ok()?;
        let off = (meta.chain_offset as usize + pos) * 2;
        Some(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32)
    };

    // bucket 0=better(O>T higher) 1=worse 2=equal
    #[derive(Clone)]
    struct Bucket {
        count: u64,
        sum_r1: f64,   // 1회차 raw sum
        sum_ot8: f64,  // O>T raw sum
        x_hist: [u64; 7],          // avail partner X
        tail_hist: [[u64; 7]; 5],  // piece freq at tail position 0..4
        o_depth: [u64; 6],         // 0 = X=O (O in avail), 1..5 = O at tail idx 0..4
        sz_tail0: u64,             // count where tail[0] in {S,Z}
        o_in_first3: u64,          // O in {avail-partner X, tail[0]} (leading 3 slots incl. T/O swap)
    }
    let empty = Bucket { count: 0, sum_r1: 0.0, sum_ot8: 0.0, x_hist: [0; 7],
        tail_hist: [[0; 7]; 5], o_depth: [0; 6], sz_tail0: 0, o_in_first3: 0 };
    let mut bk = [empty.clone(), empty.clone(), empty];
    let (ps, pz) = (find('S'), find('Z'));

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 7 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1; mult[b as usize] += 1;
        for &p in tail.iter() { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; } // no-dup 1회차
        let x = if a == pt { b } else if b == pt { a } else { continue };
        let mqs = QChainState { avail: encode_avail_pair(po, x), tail5: qs.tail5, bag_mask: qs.bag_mask };
        let Some(&mid) = q_index.key_to_id.get(&qchain_key(mqs)) else { continue; };
        let off = (meta.chain_offset as usize + i) * 2;
        let r1 = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let Some(ot8) = raw_of_qid(mid as u32) else { continue; };
        let bi = if ot8 > r1 { 0 } else if ot8 < r1 { 1 } else { 2 };
        let e = &mut bk[bi];
        e.count += 1; e.sum_r1 += r1 as f64; e.sum_ot8 += ot8 as f64;
        e.x_hist[x as usize] += 1;
        for (ti, &p) in tail.iter().enumerate() { e.tail_hist[ti][p as usize] += 1; }
        // O depth in the 1회차 queue: avail (X=O) else its tail index
        if x == po { e.o_depth[0] += 1; }
        else if let Some(oi) = tail.iter().position(|&p| p == po) { e.o_depth[1 + oi] += 1; }
        if tail[0] == ps || tail[0] == pz { e.sz_tail0 += 1; }
        if x == po || tail[0] == po { e.o_in_first3 += 1; }
    }

    let names = ["better(O>T higher)", "worse(1회차 higher)", "equal"];
    let pc = |p: u8| piece_char(p);
    for bi in 0..3 {
        let e = &bk[bi];
        if e.count == 0 { continue; }
        let c = e.count as f64;
        println!("\n=== {} : n={} ===", names[bi], e.count);
        println!("  mean 1회차 raw={:.1}/{}  mean O>T raw={:.1}/{}  (mean diff={:+.1})",
            e.sum_r1 / c, scale, e.sum_ot8 / c, scale, (e.sum_ot8 - e.sum_r1) / c);
        print!("  avail partner X:");
        for p in 0..7 { if e.x_hist[p] > 0 { print!("  {}={} ({:.0}%)", pc(p as u8), e.x_hist[p], 100.0 * e.x_hist[p] as f64 / c); } }
        println!();
        for ti in 0..5 {
            print!("  tail[{}] :", ti);
            let mut v: Vec<(u8, u64)> = (0u8..7).map(|p| (p, e.tail_hist[ti][p as usize])).filter(|&(_, n)| n > 0).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            for (p, n) in v { print!("  {}={} ({:.0}%)", pc(p), n, 100.0 * n as f64 / c); }
            println!();
        }
        print!("  O depth (queue slot of O):  avail(X=O)={} ({:.0}%)", e.o_depth[0], 100.0 * e.o_depth[0] as f64 / c);
        for d in 0..5 { print!("  tail[{}]={} ({:.0}%)", d, e.o_depth[1 + d], 100.0 * e.o_depth[1 + d] as f64 / c); }
        println!();
        println!("  tail[0]∈{{S,Z}}: {} ({:.0}%)   O in leading-3 (avail-X or tail0): {} ({:.0}%)",
            e.sz_tail0, 100.0 * e.sz_tail0 as f64 / c, e.o_in_first3, 100.0 * e.o_in_first3 as f64 / c);
    }
    Ok(())
}

// Compare two f32 layer0 value files: report per-popcnt (loop-cycle) mean of each and the diff,
// split by no-dup vs dup on the front old-bag segment. Aligns each value to its qid via the
// layer0 chain order (file order).
// usage: dump-l0-f32-round-diff <pruned_layer_dir> <vA.f32> <vB.f32>
fn run_dump_l0_f32_round_diff(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        eprintln!("usage: {} dump-l0-f32-round-diff <pruned_layer_dir> <vA.f32> <vB.f32>", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let a_path = PathBuf::from(&args[3]);
    let b_path = PathBuf::from(&args[4]);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let va = load_f32_values_vec(&a_path, layer0.total_states)?;
    let vb = load_f32_values_vec(&b_path, layer0.total_states)?;

    // [popcnt][dup? 0=nodup 1=dup] -> (count, sumA, sumB)
    let mut acc = [[(0u64, 0.0f64, 0.0f64); 2]; 8];
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones() as usize;
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        if pc >= 2 {
            mult[a as usize] += 1; mult[b as usize] += 1;
            for &p in tail.iter().take(pc - 2) { mult[p as usize] += 1; }
        }
        let dup = if mult.iter().any(|&m| m >= 2) { 1 } else { 0 };
        let pos = meta.chain_offset as usize + i;
        let e = &mut acc[pc][dup];
        e.0 += 1; e.1 += va[pos] as f64; e.2 += vb[pos] as f64;
    }

    let round_of = |pc: usize| -> usize { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };
    println!("A={}", a_path.display());
    println!("B={}", b_path.display());
    println!("{:<22} {:>8} {:>18} {:>18} {:>16}", "round(popcnt,dup)", "count", "meanA", "meanB", "diff(B-A)");
    let mut order: Vec<usize> = (1..8).collect();
    order.sort_by_key(|&pc| round_of(pc));
    for pc in order {
        for dup in 0..2 {
            let (c, sa, sb) = acc[pc][dup];
            if c == 0 { continue; }
            let (ma, mb) = (sa / c as f64, sb / c as f64);
            println!("회차{}(pc={},{}) {:>8} {:>18.9} {:>18.9} {:>16.9}",
                round_of(pc), pc, if dup == 1 { "dup " } else { "nodup" }, c, ma, mb, mb - ma);
        }
    }
    Ok(())
}

// popcnt=2 (5회차, front old-bag = the avail pair): condition on the next bag's first mino
// (tail[0] = a given piece) and report avg one-loop prob per avail pair, ranked. Lets you
// compare e.g. avail {T,L} vs {I,L} given tail[0]=S.
// usage: dump-l0-round-avail-cond <pruned_layer_dir> <l0_values.u16> <popcnt> <tail0_piece> [scale=5040]
fn run_dump_l0_round_avail_cond(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} dump-l0-round-avail-cond <pruned_layer_dir> <l0_values.u16> <popcnt> <tail0_piece> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let popcnt: u32 = args[4].parse().context("invalid popcnt")?;
    // tail0 filter; "*" or "any" = no condition (rank avail pairs over all tail).
    let t0: Option<u8> = if args[5] == "*" || args[5].eq_ignore_ascii_case("any") { None } else {
        Some((0u8..7).find(|&p| piece_char(p).to_string() == args[5].to_uppercase())
            .with_context(|| format!("unknown piece {}", args[5]))?)
    };
    let scale: u32 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let mut per: HashMap<(u8, u8), Agg> = HashMap::new(); // sorted avail pair
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != popcnt { continue; }
        let tail = decode_tail5(qs.tail5);
        if let Some(t0) = t0 { if tail[0] != t0 { continue; } }
        let (a, b) = decode_avail_pair(qs.avail);
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let e = per.entry((a, b)).or_insert(Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX });
        e.count += 1; e.sum += raw as f64; if raw >= scale { e.full += 1; } e.min_raw = e.min_raw.min(raw);
    }
    let mut v: Vec<((u8, u8), Agg)> = per.into_iter().collect();
    v.sort_by(|x, y| (y.1.sum / y.1.count as f64).partial_cmp(&(x.1.sum / x.1.count as f64)).unwrap());
    println!("popcnt={} avail-pair ranking given tail[0]={} ({} pairs, scale={}):", popcnt, t0.map(piece_char).unwrap_or('*'), v.len(), scale);
    for (rank, ((a, b), e)) in v.iter().enumerate() {
        println!("  {:>2}. {{{},{}}} avg_p={:.9} full={:.4}% min_raw={} count={}",
            rank + 1, piece_char(*a), piece_char(*b),
            e.sum / e.count as f64 / scale as f64, 100.0 * e.full as f64 / e.count as f64, e.min_raw, e.count);
    }
    Ok(())
}

// 3회차 (popcnt=1): rank ALL ordered first-3 fresh-bag triples (q0, tail[0], tail[1]) -- 7P3 =
// 210 permutations -- by avg one-loop prob. carry excluded (it is the old-bag "extra"; q0 = the
// avail piece that is not the carry).
// usage: dump-l0-round3-first3perm <pruned_layer_dir> <l0_values.u16> [scale=5040] [top_n=all]
fn run_dump_l0_round3_first3perm(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-round3-first3perm <pruned_layer_dir> <l0_values.u16> [scale=5040] [top_n=all]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let top_n: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let mut per: HashMap<u16, Agg> = HashMap::new();
    let mut anomaly = 0u64;

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 1 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut vis = [0i32; 7];
        vis[a as usize] += 1; vis[b as usize] += 1;
        for &p in tail.iter() { vis[p as usize] += 1; }
        let mut carry: Option<u8> = None;
        for t in 0u8..7 {
            let revealed = (qs.bag_mask & (1 << t)) == 0;
            if vis[t as usize] - if revealed { 1 } else { 0 } >= 1 {
                if carry.is_some() { carry = None; break; }
                carry = Some(t);
            }
        }
        let Some(carry) = carry else { anomaly += 1; continue; };
        if a != carry && b != carry { anomaly += 1; continue; }
        let q0 = if a == carry { b } else { a };
        let key = (q0 as u16) * 49 + (tail[0] as u16) * 7 + tail[1] as u16;
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let e = per.entry(key).or_insert(Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX });
        e.count += 1; e.sum += raw as f64; if raw >= scale { e.full += 1; } e.min_raw = e.min_raw.min(raw);
    }

    let mut v: Vec<(u16, Agg)> = per.into_iter().collect();
    v.sort_by(|x, y| (y.1.sum / y.1.count as f64).partial_cmp(&(x.1.sum / x.1.count as f64)).unwrap());
    println!("3회차 first-3 fresh ordered triples (7P3), {} distinct, anomaly={}, scale={}:", v.len(), anomaly, scale);
    for (rank, (key, e)) in v.iter().enumerate().take(top_n) {
        let (p0, p1, p2) = ((key / 49) as u8, ((key / 7) % 7) as u8, (key % 7) as u8);
        println!("  {:>3}. {}{}{} avg_p={:.9} full={:.4}% min_raw={} count={}",
            rank + 1, piece_char(p0), piece_char(p1), piece_char(p2),
            e.sum / e.count as f64 / scale as f64, 100.0 * e.full as f64 / e.count as f64, e.min_raw, e.count);
    }
    Ok(())
}

// 3회차 (popcnt=1): each state = 1 old-bag carry piece + a fresh new bag. Excluding the carry
// (the "extra"), rank the 7 pieces by avg one-loop prob over states where the piece is among
// the FIRST 3 fresh-bag pieces to appear = {q0 (the avail piece that is NOT the carry),
// tail[0], tail[1]}. Carry is identified by type (visible minus fresh-revealed). Also prints a
// per-carry breakdown ("각 carry에 대해 각각").
// usage: dump-l0-round3-first3fresh <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_round3_first3fresh(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-round3-first3fresh <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let z = Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX };
    let mut overall = [z; 7];
    let mut by_carry = [[z; 7]; 7]; // [carry][piece]
    let mut anomaly = 0u64;

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 1 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        // carry (old-bag piece) = visible type left over after removing 6 fresh-revealed
        let mut vis = [0i32; 7];
        vis[a as usize] += 1; vis[b as usize] += 1;
        for &p in tail.iter() { vis[p as usize] += 1; }
        let mut carry: Option<u8> = None;
        for t in 0u8..7 {
            let revealed = (qs.bag_mask & (1 << t)) == 0;
            if vis[t as usize] - if revealed { 1 } else { 0 } >= 1 {
                if carry.is_some() { carry = None; break; }
                carry = Some(t);
            }
        }
        let Some(carry) = carry else { anomaly += 1; continue; };
        // q0 = the avail piece that is not the carry; first-3 fresh = {q0, tail[0], tail[1]}
        if a != carry && b != carry { anomaly += 1; continue; }
        let q0 = if a == carry { b } else { a };
        let first3 = [q0, tail[0], tail[1]];
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        for &p in &first3 {
            for arr in [&mut overall[p as usize], &mut by_carry[carry as usize][p as usize]] {
                arr.count += 1; arr.sum += raw as f64; if raw >= scale { arr.full += 1; } arr.min_raw = arr.min_raw.min(raw);
            }
        }
    }

    let show = |title: &str, arr: &[Agg; 7]| {
        let mut v: Vec<(u8, &Agg)> = (0u8..7).map(|p| (p, &arr[p as usize])).filter(|(_, e)| e.count > 0).collect();
        if v.is_empty() { return; }
        v.sort_by(|x, y| (y.1.sum / y.1.count as f64).partial_cmp(&(x.1.sum / x.1.count as f64)).unwrap());
        println!("{title}");
        for (rank, (p, e)) in v.iter().enumerate() {
            println!("  {}. {} avg_p={:.9} full={:.4}% min_raw={} count={}",
                rank + 1, piece_char(*p), e.sum / e.count as f64 / scale as f64,
                100.0 * e.full as f64 / e.count as f64, e.min_raw, e.count);
        }
    };
    println!("3회차 (popcnt=1) first-3 FRESH-bag mino ranking (carry excluded, scale={scale}); anomaly={anomaly}");
    show("[overall]", &overall);
    for c in 0u8..7 {
        println!();
        show(&format!("[carry={}]", piece_char(c)), &by_carry[c as usize]);
    }
    Ok(())
}

// For each loop-cycle (회차) with front old-bag segment >= 3, restrict to NO-DUPLICATE states
// and rank the 7 pieces by avg one-loop prob over states where the piece is among the FIRST 3
// old-bag pieces (= avail pair {a,b} + tail[0]). One ranking per round.
// usage: dump-l0-first3-rank <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_first3_rank(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-first3-rank <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let z = Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX };
    let mut per = [[z; 7]; 8]; // [popcnt][piece]

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones() as usize;
        if pc < 3 { continue; } // need >=3 old-bag pieces
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        // no-duplicate check over the full front (old-bag) segment
        let mut mult = [0u8; 7];
        mult[a as usize] += 1; mult[b as usize] += 1;
        for &p in tail.iter().take(pc - 2) { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; }
        let first3 = [a, b, tail[0]]; // first 3 old-bag pieces
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        for &p in &first3 {
            let e = &mut per[pc][p as usize];
            e.count += 1; e.sum += raw as f64; if raw >= scale { e.full += 1; } e.min_raw = e.min_raw.min(raw);
        }
    }

    let round_of = |pc: usize| -> usize { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };
    let mut order: Vec<usize> = (3..8).collect();
    order.sort_by_key(|&pc| round_of(pc));
    println!("first-3 old-bag mino ranking per round (no-dup states, scale={scale}):");
    for pc in order {
        let mut v: Vec<(u8, &Agg)> = (0u8..7).map(|p| (p, &per[pc][p as usize])).filter(|(_, e)| e.count > 0).collect();
        if v.is_empty() { continue; }
        v.sort_by(|x, y| (y.1.sum / y.1.count as f64).partial_cmp(&(x.1.sum / x.1.count as f64)).unwrap());
        println!("회차={} (popcnt={}, front={} pieces):", round_of(pc), pc, pc);
        for (rank, (p, e)) in v.iter().enumerate() {
            println!("  {}. {} avg_p={:.9} full={:.4}% min_raw={} count={}",
                rank + 1, piece_char(*p), e.sum / e.count as f64 / scale as f64,
                100.0 * e.full as f64 / e.count as f64, e.min_raw, e.count);
        }
    }
    Ok(())
}

// List states of a given popcnt (loop-cycle) where <piece> is in avail ("first mino") and the
// one-loop value is NOT confirmed (raw < scale). Used to check whether a piece that is always
// confirmed in one round has exceptions in another.
// usage: dump-l0-round-piece-notfull <pruned_layer_dir> <l0_values.u16> <popcnt> <piece> [scale=5040] [top_n=all]
fn run_dump_l0_round_piece_notfull(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} dump-l0-round-piece-notfull <pruned_layer_dir> <l0_values.u16> <popcnt> <piece> [scale=5040] [top_n=all]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let popcnt: u32 = args[4].parse().context("invalid popcnt")?;
    let piece: u8 = (0u8..7).find(|&p| piece_char(p).to_string() == args[5].to_uppercase())
        .with_context(|| format!("unknown piece {}", args[5]))?;
    let scale: u32 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let top_n: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    let mut in_avail = 0u64;
    let mut fails: Vec<(u32, u32)> = Vec::new();
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != popcnt { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        if a != piece && b != piece { continue; }
        in_avail += 1;
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        if raw < scale { fails.push((raw, qid)); }
    }
    fails.sort_unstable_by(|x, y| x.0.cmp(&y.0));
    println!("popcnt={} piece={} in avail: {} states, not-confirmed(raw<{}): {}",
        popcnt, piece_char(piece), in_avail, scale, fails.len());
    for &(raw, qid) in fails.iter().take(top_n) {
        let qs = q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
        println!("  p={}/{} ({:.6}) avail={{{},{}}} tail={} bag={} qid={}",
            raw, scale, raw as f64 / scale as f64, piece_char(a), piece_char(b), tail_s, mask_to_string(qs.bag_mask), qid);
    }
    Ok(())
}

// 회차1 (popcnt=7, no-dup = fresh full bag): rank by "first mino". The quotient forgets
// hold/queue0 order (avail is an unordered sorted pair), so a single true first piece isn't
// recoverable. Two definable proxies: (A) avail-membership — each state contributes to BOTH
// of its two free pieces; (B) tail[0] — the first FORCED piece after the two free ones.
// usage: dump-l0-round1-firstpiece <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_round1_firstpiece(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-round1-firstpiece <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let z = Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX };
    let mut avail = [z; 7];   // (A) membership in the free pair
    let mut first = [z; 7];   // (B) tail[0]

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 7 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1; mult[b as usize] += 1;
        for &p in tail.iter() { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; } // no-dup (fresh bag) only
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        for &p in &[a, b] {
            let e = &mut avail[p as usize];
            e.count += 1; e.sum += raw as f64; if raw >= scale { e.full += 1; } e.min_raw = e.min_raw.min(raw);
        }
        let e = &mut first[tail[0] as usize];
        e.count += 1; e.sum += raw as f64; if raw >= scale { e.full += 1; } e.min_raw = e.min_raw.min(raw);
    }

    let print_ranked = |title: &str, arr: &[Agg; 7]| {
        let mut v: Vec<(u8, &Agg)> = (0u8..7).map(|p| (p, &arr[p as usize])).filter(|(_, e)| e.count > 0).collect();
        v.sort_by(|x, y| (y.1.sum / y.1.count as f64).partial_cmp(&(x.1.sum / x.1.count as f64)).unwrap());
        println!("{title}");
        for (rank, (p, e)) in v.iter().enumerate() {
            println!("  {}. piece={} count={:>5} avg_p={:.9} full={:>5} ({:.4}%) min_raw={}",
                rank + 1, piece_char(*p), e.count, e.sum / e.count as f64 / scale as f64,
                e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
        }
    };
    println!("회차1 (popcnt=7, no-dup) first-mino ranking (scale={scale}):");
    print_ranked("[A] by free-pair membership (each state counts to BOTH avail pieces):", &avail);
    println!();
    print_ranked("[B] by tail[0] (first forced piece after the two free ones):", &first);
    Ok(())
}

// 회차3 (popcnt=1): the loop-cycle is defined by a SINGLE old-bag carry piece. Identify it as
// the visible-7 type left over after removing the 6 freshly-revealed current-bag pieces
// (= complement of bag_mask), and report one-loop PC prob stats per that piece.
// usage: dump-l0-round3-piece <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_round3_piece(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-round3-piece <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let mut per = [Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX }; 7];
    let mut anomaly = 0u64;

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 1 { continue; } // 회차3
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut vis = [0i32; 7];
        vis[a as usize] += 1; vis[b as usize] += 1;
        for &p in tail.iter() { vis[p as usize] += 1; }
        // fresh-bag revealed = complement of bag_mask (6 types, one each). leftover = old-bag piece.
        let mut oldbag: Option<u8> = None;
        let mut multi = false;
        for t in 0u8..7 {
            let revealed = (qs.bag_mask & (1 << t)) == 0;
            let left = vis[t as usize] - if revealed { 1 } else { 0 };
            if left >= 1 {
                if oldbag.is_some() { multi = true; }
                oldbag = Some(t);
            }
        }
        let Some(p) = oldbag else { anomaly += 1; continue; };
        if multi { anomaly += 1; continue; }
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let e = &mut per[p as usize];
        e.count += 1;
        e.sum += raw as f64;
        if raw >= scale { e.full += 1; }
        e.min_raw = e.min_raw.min(raw);
    }

    println!("회차3 (popcnt=1) one-loop by old-bag carry piece (scale={scale}):");
    let tot: u64 = per.iter().map(|e| e.count).sum();
    for p in 0u8..7 {
        let e = per[p as usize];
        if e.count == 0 { continue; }
        println!("  piece={} count={:>6} ({:>6.3}%) avg_p={:.9} full={:>6} ({:.4}%) min_raw={}",
            piece_char(p), e.count, 100.0 * e.count as f64 / tot as f64,
            e.sum / e.count as f64 / scale as f64, e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
    }
    println!("  total={} anomaly={}", tot, anomaly);
    Ok(())
}

// Mirror-asymmetry check on layer0: reflect each state left-right (relabel L<->J, S<->Z;
// I,O,T fixed; queue order unchanged) and compare its one-loop value to the mirror's. Perfect
// L/R symmetry would make them equal; differences isolate the I-piece (SRS) asymmetry. Can
// filter to popcnt=7 dup states of an exact "D>M" class (e.g. T>J). Reports states where
// raw != mirror_raw.
// usage: dump-l0-mirror-check <pruned_layer_dir> <l0_values.u16> [scale=5040] [filter e.g. T>J] [only_fail=1]
fn run_dump_l0_mirror_check(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-mirror-check <pruned_layer_dir> <l0_values.u16> [scale=5040] [filter=T>J] [only_fail=1]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let filter = args.get(5).cloned();
    let only_fail: bool = args.get(6).map(|s| s != "0").unwrap_or(true);
    let find = |c: char| (0u8..7).find(|&p| piece_char(p) == c).unwrap();
    let (pi, pj, pl, po, ps, pt, pz) = (find('I'), find('J'), find('L'), find('O'), find('S'), find('T'), find('Z'));
    let mir = |p: u8| -> u8 { if p == pj { pl } else if p == pl { pj } else if p == ps { pz } else if p == pz { ps } else { p } };
    let _ = (pi, po, pt);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");
    let raw_of_qid = |qid: u32| -> Option<u32> {
        let pos = chains.binary_search(&qid).ok()?;
        let off = (meta.chain_offset as usize + pos) * 2;
        Some(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32)
    };

    let mut checked = 0u64;
    let mut mismatch: Vec<(u32, u32, u32)> = Vec::new(); // (raw, mraw, qid)
    let mut mirror_missing = 0u64;
    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        // optional filter: popcnt=7 dup state of exact D>M
        if let Some(f) = filter.as_ref() {
            if qs.bag_mask.count_ones() != 7 { continue; }
            let (a, b) = decode_avail_pair(qs.avail);
            let tail = decode_tail5(qs.tail5);
            let mut mult = [0u8; 7];
            mult[a as usize] += 1; mult[b as usize] += 1;
            for &p in tail.iter() { mult[p as usize] += 1; }
            let dups: Vec<u8> = (0u8..7).filter(|&p| mult[p as usize] >= 2).collect();
            let miss: Vec<u8> = (0u8..7).filter(|&p| mult[p as usize] == 0).collect();
            if dups.len() != 1 || miss.len() != 1 || mult[dups[0] as usize] != 2 { continue; }
            let name = format!("{}>{}", piece_char(dups[0]), piece_char(miss[0]));
            if &name != f { continue; }
        }
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        if only_fail && filter.is_some() && raw >= scale { continue; }
        // build mirror state
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mtail: [u8; 5] = [mir(tail[0]), mir(tail[1]), mir(tail[2]), mir(tail[3]), mir(tail[4])];
        let mut mbag: u8 = 0;
        for p in 0u8..7 { if qs.bag_mask & (1 << p) != 0 { mbag |= 1 << mir(p); } }
        let mqs = QChainState { avail: encode_avail_pair(mir(a), mir(b)), tail5: encode_tail5(mtail), bag_mask: mbag };
        let Some(&mid) = q_index.key_to_id.get(&qchain_key(mqs)) else { mirror_missing += 1; continue; };
        let Some(mraw) = raw_of_qid(mid as u32) else { mirror_missing += 1; continue; };
        checked += 1;
        if raw != mraw { mismatch.push((raw, mraw, qid)); }
    }

    println!("mirror-check (L<->J, S<->Z){}: checked={} mismatches={} mirror_missing={}",
        filter.as_ref().map(|f| format!(" filter={f} only_fail={only_fail}")).unwrap_or_default(), checked, mismatch.len(), mirror_missing);
    mismatch.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for &(raw, mraw, qid) in mismatch.iter() {
        let qs = q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
        let mtail: [u8; 5] = [mir(tail[0]), mir(tail[1]), mir(tail[2]), mir(tail[3]), mir(tail[4])];
        let mtail_s = mtail.iter().map(|&p| piece_char(p)).collect::<String>();
        println!("  raw={}/{} ({:.6}) vs mirror={}/{} ({:.6}) diff={:+}  avail={{{},{}}} tail={} -> mirror avail={{{},{}}} tail={}",
            raw, scale, raw as f64 / scale as f64, mraw, scale, mraw as f64 / scale as f64, mraw as i64 - raw as i64,
            piece_char(a), piece_char(b), tail_s, piece_char(mir(a)), piece_char(mir(b)), mtail_s);
    }
    Ok(())
}

// 8회차 = 1회차 (popcnt=7) but WITH a duplicated mino. The front 7 pieces (= all visible)
// then contain exactly one duplicated type D (mult 2) and one missing type M (mult 0): the
// hold carried an extra D across the bag boundary, displacing the M that got consumed early.
// Classify by "D replaced M" (D>M), each of D,M in {T, L|J (=L or J), X (=I,O,S,Z)}. T>T is
// impossible (only one T), so 8 categories. Reports one-loop PC prob stats.
// usage: dump-l0-8round-replace <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_8round_replace(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-8round-replace <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    // Optional: list the FAILING (raw < scale) states of one category, e.g. "T>X".
    let detail_cat = args.get(5).cloned();
    let detail_top: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
    let find = |c: char| (0u8..7).find(|&p| piece_char(p) == c).unwrap();
    let (pt, pl, pj) = (find('T'), find('L'), find('J'));
    let class_of = |p: u8| -> usize { if p == pt { 0 } else if p == pl || p == pj { 1 } else { 2 } }; // 0=T 1=L|J 2=X
    let cat_name = |d: usize, m: usize| -> String {
        let n = |c: usize| match c { 0 => "T", 1 => "L|J", _ => "X" };
        format!("{}>{}", n(d), n(m))
    };
    let mut fails: Vec<(u32, u32, u8, u8)> = Vec::new(); // (raw, qid, dup, miss) for detail_cat

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let mut per = [[Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX }; 3]; 3]; // [dclass][mclass]
    let mut exact = [[Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX }; 7]; 7]; // [dup][miss]
    let mut anomaly = 0u64;

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        if qs.bag_mask.count_ones() != 7 { continue; } // 8회차: popcnt=7
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1;
        mult[b as usize] += 1;
        for &p in tail.iter() { mult[p as usize] += 1; }
        // need exactly one duplicated (mult>=2) and one missing (mult==0)
        let dups: Vec<u8> = (0u8..7).filter(|&p| mult[p as usize] >= 2).collect();
        let miss: Vec<u8> = (0u8..7).filter(|&p| mult[p as usize] == 0).collect();
        if dups.len() != 1 || miss.len() != 1 || mult[dups[0] as usize] != 2 {
            if dups.is_empty() { continue; } // no-dup state (handled elsewhere)
            anomaly += 1; continue;
        }
        let (dc, mc) = (class_of(dups[0]), class_of(miss[0]));
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let e = &mut per[dc][mc];
        e.count += 1;
        e.sum += raw as f64;
        if raw >= scale { e.full += 1; }
        e.min_raw = e.min_raw.min(raw);
        let ex = &mut exact[dups[0] as usize][miss[0] as usize];
        ex.count += 1;
        ex.sum += raw as f64;
        if raw >= scale { ex.full += 1; }
        ex.min_raw = ex.min_raw.min(raw);
        if raw < scale {
            if let Some(dc_name) = detail_cat.as_ref() {
                let exact_name = format!("{}>{}", piece_char(dups[0]), piece_char(miss[0]));
                if &cat_name(dc, mc) == dc_name || &exact_name == dc_name {
                    fails.push((raw, qid, dups[0], miss[0]));
                }
            }
        }
    }

    // user's order: T>X, T>L|J, L|J>X, L|J>J|L, L|J>T, X>X, X>L|J, X>T
    let cats: [((usize, usize), &str); 8] = [
        ((0, 2), "T>X"), ((0, 1), "T>L|J"),
        ((1, 2), "L|J>X"), ((1, 1), "L|J>J|L"), ((1, 0), "L|J>T"),
        ((2, 2), "X>X"), ((2, 1), "X>L|J"), ((2, 0), "X>T"),
    ];
    println!("8회차 (popcnt=7, 중복) D>M replacement classes (scale={scale}):");
    let total: u64 = cats.iter().map(|&((d, m), _)| per[d][m].count).sum();
    for ((d, m), name) in cats {
        let e = per[d][m];
        if e.count == 0 { println!("  {:<9} count=0", name); continue; }
        println!("  {:<9} count={:>6} ({:>6.3}%) avg_p={:.9} full={:>6} ({:.4}%) min_raw={}",
            name, e.count, 100.0 * e.count as f64 / total as f64,
            e.sum / e.count as f64 / scale as f64, e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
    }
    println!("  total_dup={} anomaly(multi-dup/triple)={}", total, anomaly);

    println!("\nexact D>M (dup piece > missing piece), full% (nonzero only):");
    for d in 0u8..7 {
        for m in 0u8..7 {
            let e = exact[d as usize][m as usize];
            if e.count == 0 { continue; }
            println!("  {}>{} count={:>5} avg_p={:.9} full={:>5} ({:.4}%) min_raw={}",
                piece_char(d), piece_char(m), e.count, e.sum / e.count as f64 / scale as f64,
                e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
        }
    }

    if let Some(dc_name) = detail_cat.as_ref() {
        fails.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        println!("\n[{dc_name}] failing states (raw < {scale}): {} total", fails.len());
        for &(raw, qid, dup, miss) in fails.iter().take(detail_top) {
            let qs = q_index.states[qid as usize];
            let (a, b) = decode_avail_pair(qs.avail);
            let tail = decode_tail5(qs.tail5);
            let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
            println!("  p={}/{} ({:.6}) avail={{{},{}}} tail={} bag={} | dup={} miss={} qid={}",
                raw, scale, raw as f64 / scale as f64,
                piece_char(a), piece_char(b), tail_s, mask_to_string(qs.bag_mask),
                piece_char(dup), piece_char(miss), qid);
        }
    }
    Ok(())
}

// Classify NO-DUPLICATE layer0 states by the T / {L,J} content of the cycle-defining FRONT
// popcnt pieces (hold + queue0 + tail[0..popcnt-2], the old-bag remnant, all distinct when
// no dup). Six classes by (T present?) x (#of {L,J} present in {0,1,2}):
//   TLJ    = T and both L,J     TL|TJ = T and exactly one of L,J    T   = T, no L, no J
//   LJ     = no T, both L,J      L|J   = no T, exactly one of L,J    none= no T,L,J
// Reports one-loop PC prob stats per loop-cycle (회차) and class. Skips popcnt=1 (회차3).
// usage: dump-l0-oneloop-tlj-class <pruned_layer_dir> <l0_values.u16> [scale=5040]
fn run_dump_l0_oneloop_tlj_class(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-oneloop-tlj-class <pruned_layer_dir> <l0_values.u16> [scale=5040]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let find = |c: char| (0u8..7).find(|&p| piece_char(p) == c).unwrap();
    let (pt, pl, pj) = (find('T'), find('L'), find('J'));

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");

    #[derive(Clone, Copy)]
    struct Agg { count: u64, sum: f64, full: u64, min_raw: u32 }
    let names = ["TLJ", "TL|TJ", "T", "LJ", "L|J", "none"];
    let mut per = [[Agg { count: 0, sum: 0.0, full: 0, min_raw: u32::MAX }; 6]; 8]; // [popcnt][class]

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones() as usize;
        if pc < 2 { continue; } // skip popcnt=1 (회차3)
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut mult = [0u8; 7];
        mult[a as usize] += 1;
        mult[b as usize] += 1;
        for &p in tail.iter().take(pc - 2) { mult[p as usize] += 1; }
        if mult.iter().any(|&m| m >= 2) { continue; } // no-duplicate only
        let has_t = mult[pt as usize] >= 1;
        let lj = (mult[pl as usize] >= 1) as u8 + (mult[pj as usize] >= 1) as u8;
        let class = match (has_t, lj) {
            (true, 2) => 0, (true, 1) => 1, (true, 0) => 2,
            (false, 2) => 3, (false, 1) => 4, _ => 5,
        };
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let e = &mut per[pc][class];
        e.count += 1;
        e.sum += raw as f64;
        if raw >= scale { e.full += 1; }
        e.min_raw = e.min_raw.min(raw);
    }

    let round_of = |pc: usize| -> usize { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };
    let mut order: Vec<usize> = (2..8).collect();
    order.sort_by_key(|&pc| round_of(pc));
    println!("layer0 no-dup states by front-popcnt T/{{L,J}} class (scale={scale}):");
    for pc in order {
        let tot: u64 = per[pc].iter().map(|e| e.count).sum();
        if tot == 0 { continue; }
        println!("회차={} (popcnt={}, front={} pieces) total_nodup={}", round_of(pc), pc, pc, tot);
        for (ci, e) in per[pc].iter().enumerate() {
            if e.count == 0 { continue; }
            println!("  {:<6} count={:>7} avg_p={:.9} full={:>7} ({:.4}%) min_raw={}",
                names[ci], e.count, e.sum / e.count as f64 / scale as f64,
                e.full, 100.0 * e.full as f64 / e.count as f64, e.min_raw);
        }
    }
    Ok(())
}

// Pairwise dup-vs-original comparison on layer0: for every state where <piece> is duplicated
// (multiplicity >= 2 in the FIRST popcnt visible pieces, i.e. hold = <piece> retained across a
// bag boundary), compare its one-loop value against every "original" counterpart obtained by
// replacing the held duplicate with a piece X that keeps the front segment duplicate-free
// (X not among the old-bag queue pieces). Reports, per loop-cycle class, how often the dup is
// strictly worse than SOME original and than ALL originals, plus the worst examples.
// usage: dump-l0-dup-vs-original <pruned_layer_dir> <l0_values.u16> [piece=T] [scale=5040] [top_n=20]
fn run_dump_l0_dup_vs_original(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-dup-vs-original <pruned_layer_dir> <l0_values.u16> [piece=T] [scale=5040] [top_n=20]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let piece_arg = args.get(4).map(|s| s.as_str()).unwrap_or("T");
    let scale: u32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let top_n: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(20);
    let target: u8 = (0u8..7).find(|&p| piece_char(p).to_string() == piece_arg.to_uppercase())
        .with_context(|| format!("unknown piece {piece_arg}"))?;

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch");
    let raw_of = |qid: u32| -> Option<u32> {
        let pos = chains.binary_search(&qid).ok()?;
        let off = (meta.chain_offset as usize + pos) * 2;
        Some(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32)
    };
    let round_of = |pc: usize| -> usize { match pc { 7 => 1, 4 => 2, 1 => 3, 5 => 4, 2 => 5, 6 => 6, 3 => 7, _ => 0 } };

    #[derive(Clone, Copy, Default)]
    struct Cmp { dup_states: u64, pairs: u64, worse_pairs: u64, worse_any: u64, worse_all: u64, ge_all: u64, missing: u64 }
    let mut per = [Cmp::default(); 8];
    // (gap_to_best_original, dup_raw, best_x, best_raw, qid)
    let mut worst: Vec<(i64, u32, u8, u32, u32)> = Vec::new();

    for (i, &qid) in chains.iter().enumerate() {
        let qs = q_index.states[qid as usize];
        let pc = qs.bag_mask.count_ones() as usize;
        if pc < 2 { continue; }
        let (a, b) = decode_avail_pair(qs.avail);
        if a != target && b != target { continue; }
        let tail = decode_tail5(qs.tail5);
        // front-queue part (old-bag pieces in queue): queue0 + tail[0..pc-2]
        let queue0 = if a == target { b } else { a }; // hold = duplicated target; for {T,T} queue0 = T
        let mut front_queue_mask: u8 = 1u8 << queue0;
        for &p in tail.iter().take(pc - 2) { front_queue_mask |= 1u8 << p; }
        // duplicated <=> hold(target) also appears among front queue pieces
        if front_queue_mask & (1u8 << target) == 0 { continue; }
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;

        let st = &mut per[pc];
        st.dup_states += 1;
        let mut n_worse = 0u64;
        let mut n_orig = 0u64;
        let mut best_raw = 0u32;
        let mut best_x = 255u8;
        for x in 0u8..7 {
            if front_queue_mask & (1u8 << x) != 0 { continue; } // would not be duplicate-free
            let cand = QChainState { avail: encode_avail_pair(x, queue0), tail5: qs.tail5, bag_mask: qs.bag_mask };
            let Some(&cid) = q_index.key_to_id.get(&qchain_key(cand)) else { st.missing += 1; continue; };
            let Some(craw) = raw_of(cid as u32) else { st.missing += 1; continue; };
            n_orig += 1;
            st.pairs += 1;
            if raw < craw { n_worse += 1; st.worse_pairs += 1; }
            if craw > best_raw { best_raw = craw; best_x = x; }
        }
        if n_orig == 0 { continue; }
        if n_worse > 0 { st.worse_any += 1; }
        if n_worse == n_orig { st.worse_all += 1; }
        if n_worse == 0 { st.ge_all += 1; }
        if best_raw > raw {
            worst.push((best_raw as i64 - raw as i64, raw, best_x, best_raw, qid));
        }
    }

    println!("dup-vs-original piece={} (scale={scale}):", piece_char(target));
    let mut order: Vec<usize> = (1..8).collect();
    order.sort_by_key(|&pc| round_of(pc));
    for pc in order {
        let s = per[pc];
        if s.dup_states == 0 { continue; }
        println!("회차={} (popcnt={}) dup_states={} pairs={} | dup이_어떤_원본보다_나쁨={} ({:.4}%) 모든_원본보다_나쁨={} ({:.4}%) 모든_원본_이상={} ({:.4}%) missing={}",
            round_of(pc), pc, s.dup_states, s.pairs,
            s.worse_any, 100.0 * s.worse_any as f64 / s.dup_states as f64,
            s.worse_all, 100.0 * s.worse_all as f64 / s.dup_states as f64,
            s.ge_all, 100.0 * s.ge_all as f64 / s.dup_states as f64,
            s.missing);
    }
    worst.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    println!("\nworst {} dup states vs best original (gap desc):", top_n.min(worst.len()));
    for &(gap, raw, bx, braw, qid) in worst.iter().take(top_n) {
        let qs = q_index.states[qid as usize];
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
        println!("회차={} gap={} dup: avail={{{},{}}} tail={} bag={} p={}/{} ({:.6}) | best_orig hold={} p={}/{} ({:.6})",
            round_of(qs.bag_mask.count_ones() as usize), gap, piece_char(a), piece_char(b), tail_s, mask_to_string(qs.bag_mask),
            raw, scale, raw as f64 / scale as f64,
            piece_char(bx), braw, scale, braw as f64 / scale as f64);
    }
    Ok(())
}

// Distribution of layer0 one-loop values grouped by bag popcount.
// usage: dump-l0-oneloop-bag-hist <pruned_layer_dir> <l0_values.u16> [scale=5040] [detail_popcnt=3]
fn run_dump_l0_oneloop_bag_hist(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        eprintln!("usage: {} dump-l0-oneloop-bag-hist <pruned_layer_dir> <l0_values.u16> [scale=5040] [detail_popcnt=3]", args[0]);
        std::process::exit(2);
    }
    let pruned_dir = PathBuf::from(&args[2]);
    let values_path = PathBuf::from(&args[3]);
    let scale: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let detail_popcnt: u32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(3);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);

    let bytes = fs::read(&values_path).with_context(|| format!("failed to read {}", values_path.display()))?;
    anyhow::ensure!(bytes.len() as u64 == layer0.total_states * 2, "u16 size mismatch: got {}, expected {}", bytes.len(), layer0.total_states * 2);

    // Per-popcount aggregates + exact raw-value histogram for the detail popcount.
    let mut count = [0u64; 8];
    let mut sum = [0f64; 8];
    let mut full = [0u64; 8];
    let mut zero = [0u64; 8];
    let mut min_raw = [u32::MAX; 8];
    let mut detail: HashMap<u32, u64> = HashMap::new();
    for (i, &qid) in chains.iter().enumerate() {
        let off = (meta.chain_offset as usize + i) * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
        let pc = q_index.states[qid as usize].bag_mask.count_ones() as usize;
        count[pc] += 1;
        sum[pc] += raw as f64;
        if raw >= scale { full[pc] += 1; }
        if raw == 0 { zero[pc] += 1; }
        min_raw[pc] = min_raw[pc].min(raw);
        if pc as u32 == detail_popcnt { *detail.entry(raw).or_insert(0) += 1; }
    }
    println!("layer0 one-loop by bag popcount (scale={scale}):");
    for pc in 1..8 {
        if count[pc] == 0 { continue; }
        println!("popcnt={} count={} avg_prob={:.9} full(p=1)={} ({:.4}%) zero={} min_raw={}",
            pc, count[pc], sum[pc] / count[pc] as f64 / scale as f64,
            full[pc], 100.0 * full[pc] as f64 / count[pc] as f64, zero[pc], min_raw[pc]);
    }
    let mut items: Vec<(u32, u64)> = detail.into_iter().collect();
    items.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    println!("\ndetail popcnt={} distinct_raw_values={}", detail_popcnt, items.len());
    let total: u64 = items.iter().map(|&(_, c)| c).sum();
    println!("{:>6} {:>12} {:>10} {:>12}", "raw", "prob", "count", "share%");
    for &(raw, c) in items.iter() {
        println!("{:>6} {:>12.9} {:>10} {:>11.5}%", raw, raw as f64 / scale as f64, c, 100.0 * c as f64 / total as f64);
    }

    // Optional: long-run V (f32 layer0, e.g. full_v_iter0) — per-popcount stats plus the
    // bottom-N detail_popcnt states with full state decode (the one-loop metric can be
    // degenerate, e.g. popcnt=3 is all-1.0, while V still varies by reset landing).
    if let Some(f32_path) = args.get(6) {
        let bottom_n: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(20);
        let v = load_f32_values_vec(Path::new(f32_path), layer0.total_states)?;
        let mut vmin = [f32::INFINITY; 8];
        let mut vmax = [f32::NEG_INFINITY; 8];
        let mut vsum = [0f64; 8];
        let mut detail_v: Vec<(f32, u32, usize)> = Vec::new(); // (V, qid, pos)
        for (i, &qid) in chains.iter().enumerate() {
            let pos = meta.chain_offset as usize + i;
            let val = v[pos];
            let pc = q_index.states[qid as usize].bag_mask.count_ones() as usize;
            vmin[pc] = vmin[pc].min(val);
            vmax[pc] = vmax[pc].max(val);
            vsum[pc] += val as f64;
            if pc as u32 == detail_popcnt { detail_v.push((val, qid, pos)); }
        }
        println!("\nlong-run V ({f32_path}) by bag popcount:");
        for pc in 1..8 {
            if count[pc] == 0 { continue; }
            println!("popcnt={} V_min={:.9} V_avg={:.9} V_max={:.9} (max_bits=0x{:08x}, exact3={})", pc, vmin[pc], vsum[pc] / count[pc] as f64, vmax[pc], vmax[pc].to_bits(), vmax[pc] == 3.0);
        }
        detail_v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("\nlowest {} states of popcnt={} by V:", bottom_n.min(detail_v.len()), detail_popcnt);
        for &(val, qid, pos) in detail_v.iter().take(bottom_n) {
            let qs = q_index.states[qid as usize];
            let (a, b) = decode_avail_pair(qs.avail);
            let tail = decode_tail5(qs.tail5);
            let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
            let off = pos * 2;
            let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
            println!("V={:.6} one_loop={}/{} avail={{{},{}}} tail={} bag={} (0x{:02x}) qid={}",
                val, raw, scale, piece_char(a), piece_char(b), tail_s, mask_to_string(qs.bag_mask), qs.bag_mask, qid);
        }
    }
    Ok(())
}

fn run_eval_one_loop_query(args: &[String]) -> Result<()> {
    if args.len() < 6 {
        eprintln!("usage: {} eval-one-loop-query <pruned_layer_dir> <mdp_out_dir> <queue7> <bag|string-size> [scale=5040]", args[0]);
        eprintln!("example: {} eval-one-loop-query ~/.../forward_pruned ~/.../mdp_one_loop_u16_from_l3 IJLOSTZ IJLOSTZ", args[0]);
        std::process::exit(2);
    }

    let pruned_dir = PathBuf::from(&args[2]);
    let out_dir = PathBuf::from(&args[3]);
    let queue = &args[4];
    let bag_arg = &args[5];
    let scale: u32 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);
    let (qs, bag_text) = parse_queue7_to_qstate(queue, bag_arg)?;
    let key = qchain_key(qs);
    let qid = *q_index.key_to_id.get(&key)
        .with_context(|| format!("qstate not found for queue={queue} bag={bag_arg}: {:?}", qs))? as u32;

    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)
        .with_context(|| format!("failed to open layer0 membership prefix {}", layer0_prefix.display()))?;

    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let start_field_id = meta.field_id;
    let chains = layer0.read_block_chains(block_idx);
    let pos = chains.binary_search(&qid)
        .map_err(|_| anyhow::anyhow!("qid={} not found in layer0 empty-field block; queue={} bag={}", qid, queue, bag_arg))?;
    let value_offset = meta.chain_offset + pos as u64;

    let values_path = out_dir.join("layer0.values.u16.bin");
    let mut vf = File::open(&values_path)
        .with_context(|| format!("failed to open {}", values_path.display()))?;
    vf.seek(SeekFrom::Start(value_offset * 2))?;
    let mut buf = [0u8; 2];
    vf.read_exact(&mut buf)?;
    let raw = u16::from_le_bytes(buf) as u32;
    let g = gcd_u32(raw, scale);
    let num = raw / g;
    let den = scale / g;

    let (a, b) = decode_avail_pair(qs.avail);
    let tail = decode_tail5(qs.tail5);
    let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();

    println!("queue={queue} bag_arg={bag_arg} bag={bag_text}");
    println!("field_id={} field_hash=0 layer=0", start_field_id);
    println!("qid={qid} qkey={key} value_offset={} raw={raw}/{scale} reduced={num}/{den} prob={:.9}", value_offset, raw as f64 / scale as f64);
    println!("qstate avail={{{},{}}} tail5={} bag_mask=0x{:02x} bag={}", piece_char(a), piece_char(b), tail_s, qs.bag_mask, mask_to_string(qs.bag_mask));
    Ok(())
}


fn permute_tail5_rec(xs: &mut [u8], i: usize, out: &mut Vec<[u8; 5]>) {
    if i == xs.len() {
        out.push([xs[0], xs[1], xs[2], xs[3], xs[4]]);
        return;
    }
    for j in i..xs.len() {
        xs.swap(i, j);
        permute_tail5_rec(xs, i + 1, out);
        xs.swap(i, j);
    }
}


fn permute_all_rec(xs: &mut [u8], i: usize, out: &mut Vec<[u8; 7]>) {
    if i == xs.len() {
        out.push([xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6]]);
        return;
    }
    for j in i..xs.len() {
        xs.swap(i, j);
        permute_all_rec(xs, i + 1, out);
        xs.swap(i, j);
    }
}

fn run_avg_one_loop_fullbag(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        eprintln!("usage: {} avg-one-loop-fullbag <pruned_layer_dir> <mdp_out_dir> <pieces7> [scale=5040]", args[0]);
        eprintln!("example: {} avg-one-loop-fullbag ~/.../forward_pruned ~/.../mdp_one_loop_u16_from_l3 IJLTSOZ", args[0]);
        std::process::exit(2);
    }

    let pruned_dir = PathBuf::from(&args[2]);
    let out_dir = PathBuf::from(&args[3]);
    let pieces_arg = &args[4];
    let scale: u32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let mut pieces = Vec::<u8>::new();
    let mut mask = 0u8;
    for ch in pieces_arg.chars() {
        let p = piece_from_char(ch).with_context(|| format!("invalid piece character '{ch}' in pieces='{pieces_arg}'"))?;
        if (mask & (1u8 << p)) != 0 {
            anyhow::bail!("pieces must be unique, duplicate '{}' in '{}'", ch, pieces_arg);
        }
        mask |= 1u8 << p;
        pieces.push(p);
    }
    if pieces.len() != 7 || mask != FULL_BAG {
        anyhow::bail!("pieces7 must contain exactly one of each IJLOSTZ; got '{}' mask={}", pieces_arg, mask_to_string(mask));
    }

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);

    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)
        .with_context(|| format!("failed to open layer0 membership prefix {}", layer0_prefix.display()))?;
    let block_idx = find_layer0_start_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let start_field_id = meta.field_id;
    let chains = layer0.read_block_chains(block_idx);

    let values_path = out_dir.join("layer0.values.u16.bin");
    let values_bytes = fs::read(&values_path)
        .with_context(|| format!("failed to read {}", values_path.display()))?;
    if values_bytes.len() % 2 != 0 {
        anyhow::bail!("{} has odd byte length {}", values_path.display(), values_bytes.len());
    }

    let mut total_raw: u64 = 0;
    let mut count: u64 = 0;
    let mut full_count: u64 = 0;
    let mut hist = std::collections::BTreeMap::<u16, u64>::new();
    let mut qids = std::collections::HashSet::<u32>::new();

    // Enumerate all 7! ordered see strings. Because avail={hold,active} is unordered,
    // these collapse to 2520 unique qids, each with multiplicity exactly 2.
    let mut all = pieces.clone();
    let mut sees = Vec::<[u8; 7]>::new();
    permute_all_rec(&mut all, 0, &mut sees);
    debug_assert_eq!(sees.len(), 5040);

    for see in sees {
        let qs = QChainState {
            avail: encode_avail_pair(see[0], see[1]),
            tail5: encode_tail5([see[2], see[3], see[4], see[5], see[6]]),
            bag_mask: FULL_BAG,
        };
        let key = qchain_key(qs);
        let qid = *q_index.key_to_id.get(&key)
            .with_context(|| format!("qstate not found for key={} state={:?}", key, qs))? as u32;
        qids.insert(qid);

        let pos = chains.binary_search(&qid)
            .map_err(|_| anyhow::anyhow!("qid={} not found in layer0 start-field block; state={:?}", qid, qs))?;
        let value_offset = meta.chain_offset + pos as u64;
        let byte_off = (value_offset as usize) * 2;
        if byte_off + 2 > values_bytes.len() {
            anyhow::bail!("value offset {} out of range for {} bytes", value_offset, values_bytes.len());
        }
        let raw = u16::from_le_bytes([values_bytes[byte_off], values_bytes[byte_off + 1]]);
        total_raw += raw as u64;
        count += 1;
        if raw as u32 == scale {
            full_count += 1;
        }
        *hist.entry(raw).or_insert(0) += 1;
    }

    let expected = 5040u64;
    let expected_unique = 2520u64;
    let den = count * (scale as u64);
    let g = gcd_u64(total_raw, den);
    println!("pieces={} bag={} bag_mask=0x{:02x}", pieces_arg, mask_to_string(FULL_BAG), FULL_BAG);
    println!("field_id={} field_hash=0 layer=0", meta.field_id);
    println!("count={} expected={} unique_qids={} expected_unique={}", count, expected, qids.len(), expected_unique);
    println!("sum_raw={}", total_raw);
    println!("avg_raw={:.12}", total_raw as f64 / count as f64);
    println!("avg_prob={:.12}", total_raw as f64 / den as f64);
    println!("fraction={}/{}", total_raw / g, den / g);
    println!("full_count={}", full_count);
    println!("full_ratio={}/{} = {:.12}", full_count, count, full_count as f64 / count as f64);
    println!("hist raw,count descending:");
    for (&raw, &n) in hist.iter().rev() {
        println!("{} {} {:.12}", raw, n, raw as f64 / scale as f64);
    }
    if count != expected || qids.len() as u64 != expected_unique {
        anyhow::bail!("count/unique_qids mismatch: count={} unique_qids={} expected_count={} expected_unique={}", count, qids.len(), expected, expected_unique);
    }
    Ok(())
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}


fn parse_u64_auto(s: &str) -> Result<u64> {
    let t = s.replace('_', "");
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Ok(u64::from_str_radix(hex, 16).with_context(|| format!("invalid hex u64 '{s}'"))?)
    } else if let Some(bin) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        Ok(u64::from_str_radix(bin, 2).with_context(|| format!("invalid binary u64 '{s}'"))?)
    } else {
        Ok(t.parse::<u64>().with_context(|| format!("invalid u64 '{s}'"))?)
    }
}

fn qstate_short_string(s: QChainState) -> String {
    let (a, b) = decode_avail_pair(s.avail);
    let tail = decode_tail5(s.tail5);
    let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
    format!(
        "avail={{{},{}}} tail5={} bag_mask=0x{:02x} bag={}",
        piece_char(a), piece_char(b), tail_s, s.bag_mask, mask_to_string(s.bag_mask)
    )
}

fn run_avg_one_loop_field_first4(args: &[String]) -> Result<()> {
    if args.len() < 9 {
        eprintln!("usage: {} avg-one-loop-field-first4 <graph.bin> <pruned_layer_dir> <mdp_out_dir> <layer> <field_hash> <first4_pieces> <bag_popcnt> [scale=5040]", args[0]);
        eprintln!("example: {} avg-one-loop-field-first4 graph.bin ~/.../forward_pruned ~/.../mdp_one_loop_u16_from_l3 3 481439260033 IOSZ 4", args[0]);
        std::process::exit(2);
    }

    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    let layer: usize = args[5].parse().with_context(|| format!("invalid layer '{}'", args[5]))?;
    if layer > 9 {
        anyhow::bail!("layer must be in 0..=9, got {layer}");
    }
    let field_hash = parse_u64_auto(&args[6])?;
    let first4_arg = &args[7];
    let bag_popcnt: u32 = args[8].parse().with_context(|| format!("invalid bag_popcnt '{}'", args[8]))?;
    if !(1..=7).contains(&bag_popcnt) {
        anyhow::bail!("bag_popcnt must be in 1..=7, got {bag_popcnt}");
    }
    let scale: u32 = args.get(9).map(|s| s.parse()).transpose()?.unwrap_or(5040);

    let mut first4_mask = 0u8;
    let mut first4_len = 0usize;
    for ch in first4_arg.chars() {
        let p = piece_from_char(ch).with_context(|| format!("invalid first4 piece '{ch}' in '{first4_arg}'"))?;
        if (first4_mask & (1u8 << p)) != 0 {
            anyhow::bail!("first4_pieces must be unique, duplicate '{}' in '{}'", ch, first4_arg);
        }
        first4_mask |= 1u8 << p;
        first4_len += 1;
    }
    if first4_len != 4 || first4_mask.count_ones() != 4 {
        anyhow::bail!("first4_pieces must contain exactly 4 unique pieces, got '{}'", first4_arg);
    }

    eprintln!("loading graph {}", graph_path.display());
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let field_id = graph.hash_lookup(field_hash)
        .with_context(|| format!("field hash 0x{field_hash:010X} ({field_hash}) not found in graph"))?;
    let actual_layer = field_layer_from_hash(field_hash) as usize;
    if actual_layer != layer {
        anyhow::bail!("field hash layer mismatch: arg layer={} but popcount(hash)/4={}", layer, actual_layer);
    }

    let layer_prefix = pruned_dir.join(format!("layer{}q_pruned", layer));
    let layer_store = layer_store::FieldChainLayer::open_index_only(&layer_prefix)
        .with_context(|| format!("failed to open layer membership prefix {}", layer_prefix.display()))?;
    let block_idx = layer_store.metas.binary_search_by_key(&field_id, |m| m.field_id)
        .map_err(|_| anyhow::anyhow!("field_id={} hash=0x{:010X} not found in layer{} membership", field_id, field_hash, layer))?;
    let meta = &layer_store.metas[block_idx];
    let chains = layer_store.read_block_chains(block_idx);
    if chains.len() != meta.count as usize {
        anyhow::bail!("block chain count mismatch: read {} meta.count {}", chains.len(), meta.count);
    }

    let values_path = out_dir.join(format!("layer{}.values.u16.bin", layer));
    let mut vf = File::open(&values_path)
        .with_context(|| format!("failed to open {}", values_path.display()))?;
    let byte_off = meta.chain_offset.checked_mul(2).context("value byte offset overflow")?;
    let byte_len = (meta.count as usize).checked_mul(2).context("value byte len overflow")?;
    let mut value_bytes = vec![0u8; byte_len];
    vf.seek(SeekFrom::Start(byte_off))?;
    vf.read_exact(&mut value_bytes)
        .with_context(|| format!("failed to read {} bytes from {} at byte offset {}", byte_len, values_path.display(), byte_off))?;

    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);

    let mut count: u64 = 0;
    let mut total_raw: u64 = 0;
    let mut full_count: u64 = 0;
    let mut hist = std::collections::BTreeMap::<u16, u64>::new();
    let mut min_raw: u16 = u16::MAX;
    let mut max_raw: u16 = 0;
    let mut worst = Vec::<(u32, u64, u16, QChainState)>::new();
    let mut best = Vec::<(u32, u64, u16, QChainState)>::new();

    for (i, &qid) in chains.iter().enumerate() {
        let Some(&qs) = q_index.states.get(qid as usize) else {
            anyhow::bail!("qid {} out of q_index range {}", qid, q_index.states.len());
        };
        if qs.bag_mask.count_ones() != bag_popcnt {
            continue;
        }
        let (a, b) = decode_avail_pair(qs.avail);
        let tail = decode_tail5(qs.tail5);
        let mut m = 0u8;
        m |= 1u8 << a;
        m |= 1u8 << b;
        m |= 1u8 << tail[0];
        m |= 1u8 << tail[1];
        if m != first4_mask || m.count_ones() != 4 {
            continue;
        }

        let raw = u16::from_le_bytes([value_bytes[i * 2], value_bytes[i * 2 + 1]]);
        let value_offset = meta.chain_offset + i as u64;
        count += 1;
        total_raw += raw as u64;
        if raw as u32 == scale {
            full_count += 1;
        }
        *hist.entry(raw).or_insert(0) += 1;
        if raw < min_raw {
            min_raw = raw;
            worst.clear();
            worst.push((qid, value_offset, raw, qs));
        } else if raw == min_raw {
            worst.push((qid, value_offset, raw, qs));
        }
        if raw > max_raw {
            max_raw = raw;
            best.clear();
            best.push((qid, value_offset, raw, qs));
        } else if raw == max_raw {
            best.push((qid, value_offset, raw, qs));
        }
    }

    if count == 0 {
        anyhow::bail!("no qstates matched filter: layer={} field_id={} hash=0x{:010X} first4={} bag_popcnt={}", layer, field_id, field_hash, first4_arg, bag_popcnt);
    }

    let den = count * (scale as u64);
    let g = gcd_u64(total_raw, den);
    println!("layer={} field_id={} field_hash=0x{:010X} field_hash_dec={} popcnt={} block_count={}", layer, field_id, field_hash, field_hash, field_hash.count_ones(), meta.count);
    println!("first4_pieces={} first4_mask=0x{:02x} bag_popcnt={} scale={}", first4_arg, first4_mask, bag_popcnt, scale);
    println!("count={}", count);
    println!("sum_raw={}", total_raw);
    println!("avg_raw={:.12}", total_raw as f64 / count as f64);
    println!("avg_prob={:.12}", total_raw as f64 / den as f64);
    println!("fraction={}/{}", total_raw / g, den / g);
    println!("min_raw={} min_prob={:.12} worst_count={}", min_raw, min_raw as f64 / scale as f64, worst.len());
    println!("max_raw={} max_prob={:.12} best_count={}", max_raw, max_raw as f64 / scale as f64, best.len());
    println!("full_count={}", full_count);
    println!("full_ratio={}/{} = {:.12}", full_count, count, full_count as f64 / count as f64);

    println!("hist raw,count ascending:");
    for (&raw, &n) in hist.iter() {
        println!("{} {} {:.12}", raw, n, raw as f64 / scale as f64);
    }

    let print_limit = 200usize;
    println!("worst qchains{}:", if worst.len() > print_limit { " first_200" } else { "" });
    for (qid, value_offset, raw, qs) in worst.iter().take(print_limit) {
        println!("raw={} qid={} value_offset={} {}", raw, qid, value_offset, qstate_short_string(*qs));
    }
    if worst.len() > print_limit {
        println!("... omitted {} more worst qchains", worst.len() - print_limit);
    }

    println!("best qchains{}:", if best.len() > print_limit { " first_200" } else { "" });
    for (qid, value_offset, raw, qs) in best.iter().take(print_limit) {
        println!("raw={} qid={} value_offset={} {}", raw, qid, value_offset, qstate_short_string(*qs));
    }
    if best.len() > print_limit {
        println!("... omitted {} more best qchains", best.len() - print_limit);
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!("usage:");
    eprintln!("  {program} <graph.bin> <rgraph.bin> [out_dir]");
    eprintln!("  {program} build-prefix <graph.bin> <rgraph.bin> <out_dir>");
    eprintln!("  {program} build-rgraph <graph.bin> <rgraph.bin>");
    eprintln!("  {program} build-prefix-q <rgraph.bin> <out_dir>");
    eprintln!("  {program} stats-layer <layer_prefix>");
    eprintln!("  {program} check-forward-pruned-removed <backward_layer_dir> <pruned_layer_dir> [sample_n=10] [start_layer=0] [end_layer=9] [graph_bin=graph.bin]
  {program} check-forward-subset-random-removed <backward_layer_dir> <pruned_layer_dir> [sample_n=10] [start_layer=0] [end_layer=9] [seed=1] [graph_bin=graph.bin] [--no-verify]");
    eprintln!("  {program} dump-missing-layer-fields <graph.bin> <layer_prefix> <layer_num> <output_txt> [--boards N]");
    eprintln!("  {program} forward-prune-q-frontier-target-shard <graph.bin> <backward_layer_dir> <output_dir> <work_dir> [num_shards=4096] [vec_to_bitset_threshold=8192] [hot_cap_gib=64] [target_bucket_size=65536] [refine_passes=4]");
    eprintln!("  {program} solve-one-loop-mdp-u16 <graph.bin> <pruned_layer_dir> <out_dir> [scale=5040] [lookup_cache_gib=8] [start_layer=9] [base_full_next_layer=none]");
    eprintln!("  {program} solve-full-mdp-discount-f32 <graph.bin> <pruned_layer_dir> <out_dir> [gamma=1.0] [iters=20] [dense_cache_gib=32] [init=zero|one_loop:<dir>]");
    eprintln!("  {program} solve-full-mdp-l0-f32 <graph.bin> <pruned_layer_dir> <out_dir> [gamma=1.0] [max_iters=50] [dense_cache_gib=32] [init=zero|one_loop:<dir>|l0:<file>|full:<dir>] [tol_abs=1e-5] [tol_mean=0]");
    eprintln!("  {program} eval-one-loop-query <pruned_layer_dir> <mdp_out_dir> <queue7> <bag|string-size> [scale=5040]");
    eprintln!("  {program} avg-one-loop-fullbag <pruned_layer_dir> <mdp_out_dir> <pieces7> [scale=5040]");
    eprintln!("  {program} avg-one-loop-field-first4 <graph.bin> <pruned_layer_dir> <mdp_out_dir> <layer> <field_hash> <first4_pieces> <bag_popcnt> [scale=5040]");
    eprintln!("  {program} forward-prune-q <graph.bin> <backward_layer_dir> <output_dir> <work_dir> [num_shards]");
    eprintln!("  {program} qstate-stats");
    eprintln!("  {program} qgraph-stats");
    eprintln!("  {program} build-qgraph <out_prefix>");
    eprintln!("  {program} convert-layer-q <old_input_layer_prefix> <q_output_layer_prefix>");
    eprintln!("  {program} sample-external-q <rgraph.bin> <q_input_layer_prefix> <sample_states>");
    eprintln!("  {program} external-step-q-class <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records] [merge_sub_bits]");
    eprintln!("  {program} external-step-q-class-fieldreduce <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records]");
    eprintln!("  {program} external-step-q-class-fieldreduce-preq <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [bucket_bits] [flush_records]");
    eprintln!("  {program} external-step-q-preq-frontier-direct <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold]");
    eprintln!("  {program} external-step-q-preq-frontier-spill <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold] [hot_cap_gib] [inject_full_q_field_id]");
    eprintln!("  {program} external-step-q-preq-frontier-spill-target-shard <rgraph.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [vec_to_bitset_threshold] [total_hot_cap_gib] [inject_full_q_field_id|-1] [num_shards] [target_bucket_size] [refine_passes]");
    eprintln!("  {program} external-step-q-live-frontier-spill <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> <q_output_layer_prefix> <work_dir> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]");
    eprintln!("  {program} q-live-frontier-spill-sim <rgraph.bin> <qrevclass.bin> <q_input_layer_prefix> [global_live_class_limit] [per_field_class_limit] [max_spills_per_parent]");
    eprintln!("  {program} emit-full-mdp-layers-from-l0 <graph.bin> <pruned_layer_dir> <l0_values.f32> <out_dir> [gamma=1.0] [dense_cache_gib=32] [policy_out_dir]");
    eprintln!("  {program} apply-fixed-policy-boundary <graph.bin> <pruned_layer_dir> <policy_dir> <x.f32|zero> <y_out.f32> <scratch_dir> [gamma=1.0] [dense_cache_gib=16] [compare.f32]");
    eprintln!("  {program} pi-policy-eval-gmres <graph.bin> <pruned_layer_dir> <policy_dir> <x0.f32|zero> <out_v.f32> <scratch_dir> [gamma=1.0] [max_iters=200] [rel_tol=1e-6] [dense_cache_gib=16]");
    eprintln!("  {program} pi-policy-eval-reduced <graph.bin> <pruned_layer_dir> <policy_dir> <x0.f32|zero> <out_v.f32> <scratch_dir> [gamma=1.0 only] [max_iters=300] [rel_tol=1e-6] [dense_cache_gib=16]");
    eprintln!("  {program} pre-reveal-check");
    eprintln!("  {program} boundary-dag-probe <graph.bin> <pruned_layer_dir> <policy_dir|greedy> <layer=7|8|9> <max_roots> <l0.f32> [repetitions=10]");
    eprintln!("  {program} loop1-probe <graph.bin> <pruned_layer_dir> <policy_dir> <l0.f32|zero> <max_l8_states> [repetitions=5]");
    eprintln!("  {program} greedy-cache-probe <graph.bin> <pruned> <values> <work_parent> [layer=5] [max_current_states=2000000] [ABBA_cycles=2] [cache_mib=768] [memo_mib=1] [start_field_index]");
    eprintln!("  {program} compact-field-probe <pruned> <values> <policies> <work_parent> [max_span_states=1000000] [max_span_fields=512] [ABBA_cycles=2] [minimum_sample_ms=25]");
    eprintln!("  {program} loop1-sparse-probe <graph.bin> <pruned_layer_dir> <policy_dir> <l0.f32|zero> <new-y.f32> [roots=32;0=all] [memo_mib=4096] [validation=reachable|full] [mode=affine|linear] [gamma=1] [workers=4]");
    eprintln!("  {program} full-l8-probe <graph.bin> <pruned_layer_dir> <policy_dir> <l0.f32|zero> [repetitions=3] [Vstar_dir]");
    eprintln!("  {program} loop1-dag-probe <graph.bin> <pruned_layer_dir> <policy_dir> <l0.f32|zero> <new-y.f32> [roots=32;0=all] [memo_mib=256] [validation=reachable|full] [repetitions=5] [tail=l9|none] [baseline=full|1|0]");
    #[cfg(unix)]
    eprintln!("  {program} index-io-probe <graph.bin> <pruned_layer_dir> <work_parent> [source_layer=3] [first_target_field=0] [target_fields=64] [repetitions=2] [workers=4]");
    eprintln!("  {program} sample-external <rgraph.bin> <input_layer_prefix> <sample_states>");
    eprintln!("  {program} external-step <rgraph.bin> <input_layer_prefix> <output_layer_prefix> <work_dir> [flush_pairs] [merge_fanin]");
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        print_usage(args.get(0).map(String::as_str).unwrap_or("zxcl"));
        std::process::exit(2);
    }

    match args[1].as_str() {
        "build-rgraph" => run_build_rgraph(&args),
        "build-prefix-q" => run_build_prefix_q(&args),
        "build-prefix" => {
            let mut legacy = vec![args[0].clone()];
            legacy.extend_from_slice(&args[2..]);
            run_build_prefix(&legacy)
        }
        "stats-layer" => run_stats_layer(&args),
        "check-forward-pruned-removed" => run_check_forward_pruned_removed(&args),
        "check-forward-subset-random-removed" => run_check_forward_subset_random_removed(&args),
        "dump-missing-layer-fields" => run_dump_missing_layer_fields(&args),
        "forward-prune-q" => run_forward_prune_q(&args),
        "forward-prune-q-frontier-target-shard" => run_forward_prune_q(&args),
        "solve-one-loop-mdp-u16" => run_solve_one_loop_mdp_u16(&args),
        "solve-full-mdp-discount-f32" => run_solve_full_mdp_discount_f32(&args),
        "dump-pcable-hashes" => {
            let graph = FieldGraph::load_graph_bin(&args[2])?;
            let mut out = BufWriter::new(File::create(&args[3])?);
            for &h in &graph.hashes { out.write_all(&h.to_le_bytes())?; }
            out.flush()?;
            eprintln!("wrote {} pcable hashes to {}", graph.hashes.len(), args[3]);
            Ok(())
        }
        "solve-full-mdp-l0-f32" => run_solve_full_mdp_l0_f32(&args),
        "emit-full-mdp-layers-from-l0" => run_emit_full_mdp_layers_from_l0(&args),
        "measure-boundary-fanout" => run_measure_boundary_fanout(&args),
        "extract-boundary-m" => run_extract_boundary_m(&args),
        "apply-fixed-policy-boundary" => run_apply_fixed_policy_boundary(&args),
        "pi-policy-eval-gmres" => run_pi_policy_eval_gmres(&args),
        "pi-policy-eval-reduced" => boundary_commands::run_pi_policy_eval_reduced(&args),
        "pre-reveal-check" => boundary_commands::run_pre_reveal_check(&args),
        "boundary-dag-probe" => boundary_probe::run(&args),
        "loop1-probe" => loop1_probe::run(&args),
        "loop1-sparse-probe" => loop1_sparse::run(&args),
        "full-l8-probe" => full_l8_probe::run(&args),
        "loop1-dag-probe" => loop1_sparse::dag::run(&args),
        #[cfg(unix)]
        "index-io-probe" => index_io_probe::run(&args),
        "action-cache-probe" => action_cache_probe::run(&args),
        "greedy-cache-probe" => greedy_cache_probe::run(&args),
        "compact-field-probe" => compact_field_probe::run(&args),
        "survival-krylov" => run_survival_krylov(&args),
        "boundary-cond-avg" => run_boundary_cond_avg(&args),
        "boundary-survival-rank" => run_boundary_survival_rank(&args),
        "survival-eval-n" => run_survival_eval_n(&args),
        "survival-krylov-selftest" => run_survival_krylov_selftest(&args),
        "within-loop-hazard" => run_within_loop_hazard(&args),
        "eval-one-loop-query" => run_eval_one_loop_query(&args),
        "dump-l0-oneloop-bag-hist" => run_dump_l0_oneloop_bag_hist(&args),
        "dump-l0-oneloop-dup-stats" => run_dump_l0_oneloop_dup_stats(&args),
        "layer0-nodupe7-avg-f32" => run_layer0_nodupe7_avg_f32(&args),
        "dump-l0-dup-vs-original" => run_dump_l0_dup_vs_original(&args),
        "dump-l0-oneloop-tlj-class" => run_dump_l0_oneloop_tlj_class(&args),
        "dump-l0-8round-replace" => run_dump_l0_8round_replace(&args),
        "dump-l0-mirror-check" => run_dump_l0_mirror_check(&args),
        "dump-l0-round3-piece" => run_dump_l0_round3_piece(&args),
        "dump-l0-round1-firstpiece" => run_dump_l0_round1_firstpiece(&args),
        "dump-l0-round-piece-notfull" => run_dump_l0_round_piece_notfull(&args),
        "dump-l0-first3-rank" => run_dump_l0_first3_rank(&args),
        "dump-l0-round3-first3fresh" => run_dump_l0_round3_first3fresh(&args),
        "dump-l0-round3-first3perm" => run_dump_l0_round3_first3perm(&args),
        "dump-l0-round-avail-cond" => run_dump_l0_round_avail_cond(&args),
        "dump-l0-f32-round-diff" => run_dump_l0_f32_round_diff(&args),
        "dump-l0-ot8-vs-round1" => run_dump_l0_ot8_vs_round1(&args),
        "dump-l0-ot8-structure" => run_dump_l0_ot8_structure(&args),
        "dump-l0-ot8-csv" => run_dump_l0_ot8_csv(&args),
        "dump-round-sheet" => run_dump_round_sheet(&args),
        "avg-one-loop-fullbag" => run_avg_one_loop_fullbag(&args),
        "avg-one-loop-field-first4" => run_avg_one_loop_field_first4(&args),
        "qstate-stats" => run_qstate_stats(&args),
        "qgraph-stats" => run_qgraph_stats(&args),
        "build-qgraph" => run_build_qgraph(&args),
        "convert-layer-q" => run_convert_layer_q(&args),
        "sample-external-q" => run_sample_external_q(&args),
        "external-step-q-class" => run_external_step_q_class(&args),
        "external-step-q-class-fieldreduce" => run_external_step_q_class_fieldreduce(&args),
        "external-step-q-class-fieldreduce-preq" => run_external_step_q_class_fieldreduce_preq(&args),
        "external-step-q-preq-frontier-direct" => run_external_step_q_preq_frontier_direct(&args),
        "external-step-q-preq-frontier-spill" => run_external_step_q_preq_frontier_spill(&args),
        "external-step-q-preq-frontier-spill-target-shard" => run_external_step_q_preq_frontier_spill_target_shard(&args),
        "external-step-q-live" => run_external_step_q_live(&args),
        "external-step-q-live-frontier-spill" => run_external_step_q_live_frontier_spill(&args),
        "external-step-q-live-frontier-spill-deferred" => run_external_step_q_live_frontier_spill_deferred(&args),
        "external-step-q-live-frontier-spill-strong" => run_external_step_q_live_frontier_spill_strong(&args),
        "q-live-frontier-spill-sim" => run_q_live_frontier_spill_sim(&args),
        "q-live-greedy-stats" => run_external_step_q_live_greedy_stats(&args),
        "sample-external" => run_sample_external(&args),
        "external-step" => run_external_step(&args),
        "help" | "--help" | "-h" => {
            print_usage(&args[0]);
            Ok(())
        }
        _ => run_build_prefix(&args),
    }
}
