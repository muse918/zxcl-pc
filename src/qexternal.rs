#![allow(dead_code)]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::graph::FieldId;
use crate::layer_store::{FieldChainLayer, FieldChainWriter};
use crate::qgraph::QRevClassGraph;
use crate::qstate::{enumerate_qstates, QStateIndex};
use crate::rgraph::ReverseGraph;
use crate::state::StateIndex;

#[inline]
fn pair_key(field: FieldId, chain_id: u32) -> u64 {
    ((field as u64) << 32) | chain_id as u64
}

#[inline]
fn class_key(field: FieldId, bag_tail: u32) -> u64 {
    ((field as u64) << 32) | bag_tail as u64
}

pub fn convert_layer_to_q(
    input: &FieldChainLayer,
    output_prefix: &Path,
    q_index: &QStateIndex,
) -> Result<u64> {
    let start = Instant::now();
    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut total_out = 0u64;

    for (bi, meta) in input.metas.iter().enumerate() {
        let chains = input.read_block_chains(bi);
        let mut qids = Vec::<u32>::with_capacity(chains.len());
        for cid in chains {
            qids.push(q_index.old_to_qid[cid as usize]);
        }
        qids.sort_unstable();
        qids.dedup();
        total_out += qids.len() as u64;

        for qid in qids {
            writer.push_pair_key(pair_key(meta.field_id, qid))?;
        }

        if (bi + 1) % 1000 == 0 || bi + 1 == input.metas.len() {
            eprintln!(
                "  convert-q blocks_done={}/{} out_states={} elapsed={:.1}s",
                bi + 1,
                input.metas.len(),
                total_out,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let written = writer.finish()?;
    debug_assert_eq!(written, total_out);
    let in_states = input.total_states;
    println!("convert_q input_states={in_states}");
    println!("convert_q output_states={written}");
    println!("convert_q ratio={:.6}", written as f64 / in_states as f64);
    println!(
        "convert_q output_u24_membership_bytes={} ({:.3} GiB)",
        written * 3,
        written as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!("convert-q elapsed={:.3}s", start.elapsed().as_secs_f64());
    Ok(written)
}

pub fn sample_external_q_masked(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    _q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    sample_states: u64,
) -> Result<()> {
    let start = Instant::now();
    let mut processed = 0u64;
    let mut raw = 0u64;
    // Exact qchain dedupe for the sample, but using the structure
    // qchain = (bag_tail, avail). For each (field, bag_tail), a 28-bit mask is enough.
    let mut masks = HashMap::<u64, u32>::with_capacity(16_000_000);

    'outer: for (bi, meta) in input.metas.iter().enumerate() {
        let chains = input.read_block_chains(bi);
        for qid in chains {
            if processed >= sample_states {
                break 'outer;
            }

            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                for &old_field in rgraph.edges(meta.field_id, placed) {
                    for rec in recs {
                        let ck = class_key(old_field, rec.bag_tail);
                        let entry = masks.entry(ck).or_insert(0);
                        *entry |= rec.avail_mask;
                        raw += rec.avail_mask.count_ones() as u64;
                    }
                }
            }
            processed += 1;
        }

        if (bi + 1) % 100 == 0 {
            let unique_so_far: u64 = masks.values().map(|m| m.count_ones() as u64).sum();
            eprintln!(
                "  qsample blocks_done={} processed={} raw={} classes={} unique={} rss_struct_elapsed={:.1}s",
                bi + 1,
                processed,
                raw,
                masks.len(),
                unique_so_far,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let unique: u64 = masks.values().map(|m| m.count_ones() as u64).sum();
    let class_count = masks.len() as u64;
    let ratio = if raw == 0 {
        0.0
    } else {
        unique as f64 / raw as f64
    };
    let raw_per_input = if processed == 0 {
        0.0
    } else {
        raw as f64 / processed as f64
    };
    let est_raw = raw_per_input * input.total_states as f64;
    let est_unique = est_raw * ratio;

    println!("qsample input_states={processed}");
    println!("qsample raw_pairs={raw}");
    println!("qsample class_count={class_count}");
    println!("qsample unique_pairs={unique}");
    println!("qsample unique_ratio={ratio:.6}");
    println!("qsample raw_per_input={raw_per_input:.3}");
    println!("qsample estimated_full_raw_pairs={:.0}", est_raw);
    println!("qsample estimated_full_unique_pairs={:.0}", est_unique);
    println!(
        "qsample estimated_full_u24_membership_bytes={:.0} ({:.3} GiB)",
        est_unique * 3.0,
        est_unique * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "qsample estimated_full_u24_plus_fp16_bytes={:.0} ({:.3} GiB)",
        est_unique * 5.0,
        est_unique * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!("qsample elapsed={:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

pub fn build_q_index_from_old(old_index: &StateIndex) -> QStateIndex {
    enumerate_qstates(old_index)
}

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::qstate::{AVAIL_PAIR_COUNT, TAIL5_COUNT};

const Q_FIELD_ID_BITS: u64 = 24;
const Q_SHARD_BITS: u64 = 12;
const Q_SHARDS: usize = 1usize << Q_SHARD_BITS;
const Q_PAR_CHUNK_KEYS: usize = 50_000;
const QID_MISSING: u32 = u32::MAX;

#[inline]
pub fn q_pair_key(field: FieldId, qid: u32) -> u64 {
    debug_assert!(qid < (1u32 << 24));
    ((field as u64) << 32) | qid as u64
}

#[inline]
pub fn unpack_q_pair_key(key: u64) -> (FieldId, u32) {
    ((key >> 32) as FieldId, key as u32)
}

#[inline]
fn class_shard_of_key(key: u64) -> usize {
    let field = (key >> 32) as u64;
    ((field >> (Q_FIELD_ID_BITS - Q_SHARD_BITS)) as usize) & (Q_SHARDS - 1)
}

#[inline]
fn compact_qid_index(bag_tail: u32, avail: u8) -> usize {
    let tail5 = (bag_tail & 0x7FFF) as usize;
    let bag = ((bag_tail >> 15) & 0x7F) as usize;
    ((bag * TAIL5_COUNT) + tail5) * AVAIL_PAIR_COUNT + avail as usize
}

fn build_qid_lookup(q_index: &QStateIndex) -> Vec<u32> {
    let mut table = vec![QID_MISSING; 128 * TAIL5_COUNT * AVAIL_PAIR_COUNT];
    for (qid, &qs) in q_index.states.iter().enumerate() {
        let bt = crate::qstate::bag_tail_key(qs);
        let idx = compact_qid_index(bt, qs.avail);
        table[idx] = qid as u32;
    }
    table
}

#[inline]
fn qid_from_lookup(table: &[u32], bag_tail: u32, avail: u8) -> u32 {
    let qid = table[compact_qid_index(bag_tail, avail)];
    debug_assert_ne!(qid, QID_MISSING);
    qid
}

#[derive(Debug)]
struct QClassShard {
    sorted: Vec<(u64, u32)>,
    tail: Vec<(u64, u32)>,
}

impl QClassShard {
    fn new() -> Self {
        Self {
            sorted: Vec::new(),
            tail: Vec::new(),
        }
    }
    fn push_or(&mut self, key: u64, mask: u32) {
        self.tail.push((key, mask));
    }
    fn len(&self) -> usize {
        self.sorted.len() + self.tail.len()
    }

    fn flush(&mut self) {
        if self.tail.is_empty() {
            return;
        }
        self.tail.sort_unstable_by_key(|&(k, _)| k);
        let mut compact = Vec::<(u64, u32)>::with_capacity(self.tail.len());
        for (k, m) in self.tail.drain(..) {
            if let Some(last) = compact.last_mut() {
                if last.0 == k {
                    last.1 |= m;
                    continue;
                }
            }
            compact.push((k, m));
        }
        let sorted = std::mem::take(&mut self.sorted);
        self.sorted = merge_sorted_class_vecs(sorted, compact);
    }

    fn merge_sorted_vec(&mut self, other_sorted: Vec<(u64, u32)>) {
        if other_sorted.is_empty() {
            return;
        }
        self.flush();
        let sorted = std::mem::take(&mut self.sorted);
        self.sorted = merge_sorted_class_vecs(sorted, other_sorted);
    }

    fn into_sorted(mut self) -> Vec<(u64, u32)> {
        self.flush();
        self.sorted
    }
}

fn merge_sorted_class_vecs(a: Vec<(u64, u32)>, b: Vec<(u64, u32)>) -> Vec<(u64, u32)> {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let mut out = Vec::<(u64, u32)>::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (k, m) = if a[i].0 <= b[j].0 {
            let v = a[i];
            i += 1;
            v
        } else {
            let v = b[j];
            j += 1;
            v
        };
        if let Some(last) = out.last_mut() {
            if last.0 == k {
                last.1 |= m;
                continue;
            }
        }
        out.push((k, m));
    }
    while i < a.len() {
        let (k, m) = a[i];
        i += 1;
        if let Some(last) = out.last_mut() {
            if last.0 == k {
                last.1 |= m;
                continue;
            }
        }
        out.push((k, m));
    }
    while j < b.len() {
        let (k, m) = b[j];
        j += 1;
        if let Some(last) = out.last_mut() {
            if last.0 == k {
                last.1 |= m;
                continue;
            }
        }
        out.push((k, m));
    }
    out
}

#[derive(Debug)]
struct QLayerPartial {
    raw: u64,
    local_chain_masked: u64,
    class_count: u64,
    shards: Vec<QClassShard>,
}

impl QLayerPartial {
    fn empty() -> Self {
        let mut shards = Vec::with_capacity(Q_SHARDS);
        for _ in 0..Q_SHARDS {
            shards.push(QClassShard::new());
        }
        Self {
            raw: 0,
            local_chain_masked: 0,
            class_count: 0,
            shards,
        }
    }
    fn stored_len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

fn merge_q_partials(mut a: QLayerPartial, mut b: QLayerPartial) -> QLayerPartial {
    let start = Instant::now();
    a.raw += b.raw;
    a.local_chain_masked += b.local_chain_masked;
    a.class_count += b.class_count;
    for i in 0..Q_SHARDS {
        let other = std::mem::replace(&mut b.shards[i], QClassShard::new()).into_sorted();
        a.shards[i].merge_sorted_vec(other);
    }
    eprintln!(
        "  qclass reduce merge raw={} local_chain_masked={} classes={} stored_classes={} elapsed={:.3}s",
        a.raw,
        a.local_chain_masked,
        a.class_count,
        a.stored_len(),
        start.elapsed().as_secs_f64()
    );
    a
}

fn emit_grouped_field(
    rgraph: &ReverseGraph,
    child_field: FieldId,
    by_piece: &mut [HashMap<u32, u32>; crate::piece::PIECE_COUNT],
    partial: &mut QLayerPartial,
) {
    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
        let map = &mut by_piece[placed as usize];
        if map.is_empty() {
            continue;
        }
        let parents = rgraph.edges(child_field, placed);
        if parents.is_empty() {
            map.clear();
            continue;
        }
        for &old_field in parents {
            for (&bag_tail, &mask) in map.iter() {
                let ck = class_key(old_field, bag_tail);
                let shard = class_shard_of_key(ck);
                partial.shards[shard].push_or(ck, mask);
                partial.local_chain_masked += mask.count_ones() as u64;
            }
        }
        partial.class_count += map.len() as u64 * parents.len() as u64;
        map.clear();
    }
}

fn process_q_chunk_class_grouped(
    rgraph: &ReverseGraph,
    qgraph: &QRevClassGraph,
    chunk_idx: usize,
    total_chunks: usize,
    chunk: &[u64],
    done_chunks: &AtomicUsize,
    global_start: Instant,
) -> QLayerPartial {
    let chunk_start = Instant::now();
    let mut partial = QLayerPartial::empty();
    let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
        std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

    let mut i = 0usize;
    while i < chunk.len() {
        let (child_field, _) = unpack_q_pair_key(chunk[i]);
        while i < chunk.len() {
            let (f, child_qid) = unpack_q_pair_key(chunk[i]);
            if f != child_field {
                break;
            }
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(child_qid, placed);
                if recs.is_empty() {
                    continue;
                }
                let parent_count = rgraph.edges(child_field, placed).len() as u64;
                if parent_count == 0 {
                    continue;
                }
                let map = &mut by_piece[placed as usize];
                for rec in recs {
                    partial.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                    *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                }
            }
            i += 1;
        }
        emit_grouped_field(rgraph, child_field, &mut by_piece, &mut partial);
    }

    for shard in &mut partial.shards {
        shard.flush();
    }

    let done = done_chunks.fetch_add(1, Ordering::Relaxed) + 1;
    let stored_classes = partial.stored_len();
    let ratio = if partial.raw == 0 {
        0.0
    } else {
        partial.local_chain_masked as f64 / partial.raw as f64
    };
    eprintln!(
        "  qclass chunk {}/{} done_global={}/{} raw={} local_chain_masked={} classes={} stored_classes={} ratio={:.3} elapsed={:.1}s total={:.1}s",
        chunk_idx + 1,
        total_chunks,
        done,
        total_chunks,
        partial.raw,
        partial.local_chain_masked,
        partial.class_count,
        stored_classes,
        ratio,
        chunk_start.elapsed().as_secs_f64(),
        global_start.elapsed().as_secs_f64()
    );

    partial
}

fn materialize_class_shards(
    q_index: &QStateIndex,
    shards: Vec<QClassShard>,
    out: &mut Vec<u64>,
) -> usize {
    let start = Instant::now();
    eprintln!("building compact qid lookup table...");
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup table entries={} bytes={} elapsed={:.3}s",
        qid_lookup.len(),
        qid_lookup.len() * std::mem::size_of::<u32>(),
        start.elapsed().as_secs_f64()
    );

    out.clear();
    let mut qids = Vec::<u32>::new();
    let mut current_field = None::<FieldId>;
    let mut total = 0usize;

    let flush_field =
        |field: FieldId, qids: &mut Vec<u32>, out: &mut Vec<u64>, total: &mut usize| {
            if qids.is_empty() {
                return;
            }
            qids.sort_unstable();
            qids.dedup();
            *total += qids.len();
            out.reserve(qids.len());
            for &qid in qids.iter() {
                out.push(q_pair_key(field, qid));
            }
            qids.clear();
        };

    for shard in shards {
        let classes = shard.into_sorted();
        for (ck, mask) in classes {
            let field = (ck >> 32) as FieldId;
            let bag_tail = ck as u32;
            match current_field {
                None => current_field = Some(field),
                Some(f) if f != field => {
                    flush_field(f, &mut qids, out, &mut total);
                    current_field = Some(field);
                }
                Some(_) => {}
            }
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros() as u8;
                m &= m - 1;
                let qid = qid_from_lookup(&qid_lookup, bag_tail, bit);
                if qid != QID_MISSING {
                    qids.push(qid);
                }
            }
        }
    }
    if let Some(f) = current_field {
        flush_field(f, &mut qids, out, &mut total);
    }
    debug_assert_eq!(out.len(), total);
    debug_assert!(out.windows(2).all(|w| w[0] < w[1]));
    eprintln!(
        "materialized q pairs={} elapsed={:.3}s",
        total,
        start.elapsed().as_secs_f64()
    );
    total
}

#[derive(Debug, Clone, Copy)]
pub struct QLayerGenStats {
    pub raw: u64,
    pub local_masked_unique: u64,
    pub deduped: usize,
    pub removed_dupes: u64,
}

pub fn gen_prev_q_layer_masked(
    rgraph: &ReverseGraph,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    next_good: &[u64],
    prev_good: &mut Vec<u64>,
) -> QLayerGenStats {
    let start = Instant::now();
    let total_chunks = (next_good.len() + Q_PAR_CHUNK_KEYS - 1) / Q_PAR_CHUNK_KEYS;
    eprintln!(
        "q class-masked grouped layer gen chunks={} chunk_size={} threads={} shards={}",
        total_chunks,
        Q_PAR_CHUNK_KEYS,
        rayon::current_num_threads(),
        Q_SHARDS
    );
    let done_chunks = AtomicUsize::new(0);
    let partial = next_good
        .par_chunks(Q_PAR_CHUNK_KEYS)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            process_q_chunk_class_grouped(
                rgraph,
                qgraph,
                chunk_idx,
                total_chunks,
                chunk,
                &done_chunks,
                start,
            )
        })
        .reduce(QLayerPartial::empty, merge_q_partials);

    let raw = partial.raw;
    let local_masked_unique = partial.local_chain_masked;
    let mut shards = partial.shards;
    for shard in &mut shards {
        shard.flush();
    }
    let stored_classes = shards.iter().map(|s| s.sorted.len()).sum::<usize>();
    eprintln!("q final stored_classes={stored_classes}");

    let deduped = materialize_class_shards(q_index, shards, prev_good);
    eprintln!(
        "q class-masked grouped layer gen total elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );

    QLayerGenStats {
        raw,
        local_masked_unique,
        deduped,
        removed_dupes: raw - deduped as u64,
    }
}

pub fn make_q_terminal_layer(terminal_field: FieldId, q_index: &QStateIndex) -> Vec<u64> {
    let mut out = Vec::<u64>::with_capacity(q_index.states.len());
    for qid in 0..q_index.states.len() as u32 {
        out.push(q_pair_key(terminal_field, qid));
    }
    out.sort_unstable();
    out
}

pub fn print_q_product_layer_stats(keys: &[u64], depth: usize) {
    if keys.is_empty() {
        println!("q product layer {depth} graph_nodes=0");
        println!("q product layer {depth} success_field_keys=0");
        println!("q product layer {depth} field_range=None");
        return;
    }
    let mut graph_nodes = 0usize;
    let mut last_field = None::<FieldId>;
    let mut success_field_keys = 0usize;
    let mut min_field = FieldId::MAX;
    let mut max_field = 0u32;
    for &key in keys {
        let (field, _) = unpack_q_pair_key(key);
        if last_field != Some(field) {
            graph_nodes += 1;
            last_field = Some(field);
        }
        if field == crate::graph::EMPTY_FIELD_ID {
            success_field_keys += 1;
        }
        min_field = min_field.min(field);
        max_field = max_field.max(field);
    }
    println!("q product layer {depth} graph_nodes={graph_nodes}");
    println!("q product layer {depth} success_field_keys={success_field_keys}");
    println!("q product layer {depth} field_range={min_field}..={max_field}");
}

pub fn save_q_pair_layer(prefix: &Path, keys: &[u64]) -> Result<u64> {
    let mut writer = FieldChainWriter::create(prefix)?;
    for &key in keys {
        writer.push_pair_key(key)?;
    }
    writer.finish()
}

#[derive(Debug, Clone, Copy)]
pub struct QExternalClassConfig {
    pub bucket_bits: usize,
    pub flush_records: usize,
    /// Optional second-stage field shard bits used during bucket merge.
    /// 0 keeps the old one-pass bucket merge. 4 means each coarse bucket is
    /// split into 16 field-ordered subshards before sort/merge/materialize.
    pub merge_sub_bits: usize,
}

impl Default for QExternalClassConfig {
    fn default() -> Self {
        Self {
            bucket_bits: 12,
            flush_records: 4_000_000,
            merge_sub_bits: 0,
        }
    }
}

#[inline]
fn q_bucket_of_class_key(key: u64, bucket_bits: usize) -> usize {
    let field = (key >> 32) as u64;
    ((field >> (Q_FIELD_ID_BITS - bucket_bits as u64)) as usize) & ((1usize << bucket_bits) - 1)
}

#[inline]
fn q_subshard_of_class_key(key: u64, bucket_bits: usize, sub_bits: usize) -> usize {
    if sub_bits == 0 {
        return 0;
    }
    let field = (key >> 32) as u64;
    let shift = Q_FIELD_ID_BITS - bucket_bits as u64 - sub_bits as u64;
    ((field >> shift) as usize) & ((1usize << sub_bits) - 1)
}

fn materialize_class_records(
    mut recs: Vec<(u64, u32)>,
    qid_lookup: &[u32],
    writer: &mut FieldChainWriter,
    qids: &mut Vec<u32>,
    written: &mut u64,
) -> Result<u64> {
    recs.sort_unstable_by_key(|&(k, _)| k);

    let mut current_field = None::<FieldId>;
    let mut i = 0usize;
    let mut classes = 0u64;
    while i < recs.len() {
        let key = recs[i].0;
        let mut mask = 0u32;
        while i < recs.len() && recs[i].0 == key {
            mask |= recs[i].1;
            i += 1;
        }
        classes += 1;
        let field = (key >> 32) as FieldId;
        let bag_tail = key as u32;
        match current_field {
            None => current_field = Some(field),
            Some(f) if f != field => {
                qids.sort_unstable();
                qids.dedup();
                for &qid in qids.iter() {
                    writer.push_pair_key(q_pair_key(f, qid))?;
                }
                *written += qids.len() as u64;
                qids.clear();
                current_field = Some(field);
            }
            Some(_) => {}
        }
        let mut m = mask;
        while m != 0 {
            let avail = m.trailing_zeros() as u8;
            m &= m - 1;
            let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
            if qid != QID_MISSING {
                qids.push(qid);
            }
        }
    }
    if let Some(f) = current_field {
        qids.sort_unstable();
        qids.dedup();
        for &qid in qids.iter() {
            writer.push_pair_key(q_pair_key(f, qid))?;
        }
        *written += qids.len() as u64;
        qids.clear();
    }
    Ok(classes)
}

fn append_class_records(path: &Path, records: &[(u64, u32)]) -> Result<()> {
    use std::io::Write;
    if records.is_empty() {
        return Ok(());
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut buf = Vec::<u8>::with_capacity(records.len() * 12);
    for &(k, m) in records {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&m.to_le_bytes());
    }
    f.write_all(&buf)?;
    Ok(())
}

fn flush_class_buffers(
    work_dir: &Path,
    buffers: &mut [Vec<(u64, u32)>],
    total_buffered: &mut usize,
) -> Result<()> {
    let start = Instant::now();
    let mut flushed = 0usize;
    for (b, buf) in buffers.iter_mut().enumerate() {
        if buf.is_empty() {
            continue;
        }
        let path = work_dir.join(format!("qclass_bucket_{b:04}.bin"));
        append_class_records(&path, buf)?;
        flushed += buf.len();
        buf.clear();
    }
    *total_buffered = 0;
    eprintln!(
        "  flushed qclass records={} elapsed={:.3}s",
        flushed,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn external_step_q_class(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    config: QExternalClassConfig,
) -> Result<u64> {
    use anyhow::{bail, Context};
    use std::io::{BufWriter, Read, Write};
    use std::sync::{Arc, Mutex};

    if config.bucket_bits == 0
        || config.bucket_bits > 16
        || config.bucket_bits as u64 > Q_FIELD_ID_BITS
    {
        bail!("bucket_bits must be in 1..=16 and <= {Q_FIELD_ID_BITS}");
    }
    if config.merge_sub_bits > 12
        || (config.bucket_bits + config.merge_sub_bits) as u64 > Q_FIELD_ID_BITS
    {
        bail!("merge_sub_bits must be <=12 and bucket_bits + merge_sub_bits <= {Q_FIELD_ID_BITS}");
    }
    std::fs::create_dir_all(work_dir)?;
    for ent in std::fs::read_dir(work_dir)? {
        let ent = ent?;
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("qclass_bucket_") && name.ends_with(".bin") {
            std::fs::remove_file(ent.path())?;
        }
    }

    let start = Instant::now();
    let bucket_count = 1usize << config.bucket_bits;
    let block_chunk = 16usize;
    eprintln!(
        "q external class step PARALLEL blocks={} input_states={} buckets={} flush_records={} merge_sub_bits={} block_chunk={} threads={}",
        input.metas.len(),
        input.total_states,
        bucket_count,
        config.flush_records,
        config.merge_sub_bits,
        block_chunk,
        rayon::current_num_threads()
    );

    let mut writer_vec = Vec::with_capacity(bucket_count);
    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_bucket_{b:04}.bin"));
        let f =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        writer_vec.push(Mutex::new(BufWriter::with_capacity(8 * 1024 * 1024, f)));
    }
    let writers = Arc::new(writer_vec);

    #[derive(Default, Clone, Copy)]
    struct Stats {
        raw: u64,
        local_chain_masked: u64,
        emitted_records: u64,
        flushed_records: u64,
    }

    fn flush_parallel_buffers(
        writers: &[Mutex<BufWriter<std::fs::File>>],
        buffers: &mut [Vec<(u64, u32)>],
        total_buffered: &mut usize,
    ) -> Result<usize> {
        let mut flushed = 0usize;
        for (b, buf) in buffers.iter_mut().enumerate() {
            if buf.is_empty() {
                continue;
            }
            let mut w = writers[b].lock().expect("bucket writer mutex poisoned");
            let mut bytes = Vec::<u8>::with_capacity(buf.len() * 12);
            for &(k, m) in buf.iter() {
                bytes.extend_from_slice(&k.to_le_bytes());
                bytes.extend_from_slice(&m.to_le_bytes());
            }
            w.write_all(&bytes)?;
            flushed += buf.len();
            buf.clear();
        }
        *total_buffered = 0;
        Ok(flushed)
    }

    let block_indices = (0..input.metas.len()).collect::<Vec<_>>();
    let done_blocks = AtomicUsize::new(0);

    let stats_vec: Vec<Stats> = block_indices
        .par_chunks(block_chunk)
        .map(|chunk| -> Result<Stats> {
            let mut stats = Stats::default();
            let mut buffers: Vec<Vec<(u64, u32)>> = (0..bucket_count).map(|_| Vec::new()).collect();
            let mut total_buffered = 0usize;

            for &bi in chunk {
                let meta = &input.metas[bi];
                let qids = input.read_block_chains(bi);
                let child_field = meta.field_id;
                let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
                    std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

                for qid in qids {
                    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                        let recs = qgraph.records_for(qid, placed);
                        if recs.is_empty() {
                            continue;
                        }
                        let parent_count = rgraph.edges(child_field, placed).len() as u64;
                        if parent_count == 0 {
                            continue;
                        }
                        let map = &mut by_piece[placed as usize];
                        for rec in recs {
                            stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                            *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                        }
                    }
                }

                for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                    let map = &mut by_piece[placed as usize];
                    if map.is_empty() {
                        continue;
                    }
                    let parents = rgraph.edges(child_field, placed);
                    for &old_field in parents {
                        for (&bag_tail, &mask) in map.iter() {
                            let ck = class_key(old_field, bag_tail);
                            let b = q_bucket_of_class_key(ck, config.bucket_bits);
                            buffers[b].push((ck, mask));
                            total_buffered += 1;
                            stats.emitted_records += 1;
                            stats.local_chain_masked += mask.count_ones() as u64;
                        }
                    }
                    map.clear();
                }

                if total_buffered >= config.flush_records {
                    stats.flushed_records +=
                        flush_parallel_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
                }
            }

            stats.flushed_records +=
                flush_parallel_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
            let done = done_blocks.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
            if done % 1024 < block_chunk || done == input.metas.len() {
                eprintln!(
                    "  q external parallel blocks_done={}/{} elapsed={:.1}s",
                    done,
                    input.metas.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            Ok(stats)
        })
        .collect::<Result<Vec<_>>>()?;

    let raw: u64 = stats_vec.iter().map(|s| s.raw).sum();
    let local_chain_masked: u64 = stats_vec.iter().map(|s| s.local_chain_masked).sum();
    let emitted_records: u64 = stats_vec.iter().map(|s| s.emitted_records).sum();
    let flushed_records: u64 = stats_vec.iter().map(|s| s.flushed_records).sum();

    for w in writers.iter() {
        w.lock().expect("bucket writer mutex poisoned").flush()?;
    }
    drop(writers);

    eprintln!(
        "q external generation done raw={} emitted_class_records={} local_chain_masked={} flushed_records={} elapsed={:.1}s",
        raw,
        emitted_records,
        local_chain_masked,
        flushed_records,
        start.elapsed().as_secs_f64()
    );

    eprintln!("building compact qid lookup table for materialization...");
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );

    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut written = 0u64;
    let mut global_classes = 0u64;
    let mut qids = Vec::<u32>::new();

    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_bucket_{b:04}.bin"));
        if !path.exists() {
            continue;
        }
        let file_len = std::fs::metadata(&path)?.len();
        if file_len == 0 {
            let _ = std::fs::remove_file(&path);
            continue;
        }

        let mut bucket_classes = 0u64;
        if config.merge_sub_bits == 0 {
            let mut bytes = Vec::<u8>::new();
            std::fs::File::open(&path)
                .with_context(|| format!("open {}", path.display()))?
                .read_to_end(&mut bytes)?;
            if bytes.len() % 12 != 0 {
                bail!(
                    "bucket {} size {} is not divisible by 12",
                    path.display(),
                    bytes.len()
                );
            }
            let rec_count = bytes.len() / 12;
            let mut recs = Vec::<(u64, u32)>::with_capacity(rec_count);
            let mut pos = 0usize;
            while pos < bytes.len() {
                let k = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                let m = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
                pos += 12;
                recs.push((k, m));
            }
            drop(bytes);
            bucket_classes +=
                materialize_class_records(recs, &qid_lookup, &mut writer, &mut qids, &mut written)?;
        } else {
            let sub_count = 1usize << config.merge_sub_bits;
            let sub_dir = work_dir.join(format!("qclass_bucket_{b:04}_sub"));
            std::fs::create_dir_all(&sub_dir)?;

            let mut sub_writers = Vec::with_capacity(sub_count);
            for sidx in 0..sub_count {
                let sp = sub_dir.join(format!("sub_{sidx:04}.bin"));
                let f = std::fs::File::create(&sp)
                    .with_context(|| format!("create {}", sp.display()))?;
                sub_writers.push(BufWriter::with_capacity(8 * 1024 * 1024, f));
            }

            let mut f =
                std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let mut read_buf = vec![0u8; 64 * 1024 * 1024];
            let mut carry = Vec::<u8>::new();
            loop {
                let n = f.read(&mut read_buf)?;
                if n == 0 {
                    break;
                }
                carry.extend_from_slice(&read_buf[..n]);
                let usable = (carry.len() / 12) * 12;
                let mut pos = 0usize;
                while pos < usable {
                    let k = u64::from_le_bytes(carry[pos..pos + 8].try_into().unwrap());
                    let m = u32::from_le_bytes(carry[pos + 8..pos + 12].try_into().unwrap());
                    let sidx =
                        q_subshard_of_class_key(k, config.bucket_bits, config.merge_sub_bits);
                    sub_writers[sidx].write_all(&k.to_le_bytes())?;
                    sub_writers[sidx].write_all(&m.to_le_bytes())?;
                    pos += 12;
                }
                if usable > 0 {
                    carry.drain(..usable);
                }
            }
            if !carry.is_empty() {
                bail!("bucket {} size is not divisible by 12", path.display());
            }
            for w in sub_writers.iter_mut() {
                w.flush()?;
            }
            drop(sub_writers);

            for sidx in 0..sub_count {
                let sp = sub_dir.join(format!("sub_{sidx:04}.bin"));
                let slen = std::fs::metadata(&sp)?.len();
                if slen == 0 {
                    let _ = std::fs::remove_file(&sp);
                    continue;
                }
                let mut bytes = Vec::<u8>::new();
                std::fs::File::open(&sp)
                    .with_context(|| format!("open {}", sp.display()))?
                    .read_to_end(&mut bytes)?;
                if bytes.len() % 12 != 0 {
                    bail!(
                        "subshard {} size {} is not divisible by 12",
                        sp.display(),
                        bytes.len()
                    );
                }
                let rec_count = bytes.len() / 12;
                let mut recs = Vec::<(u64, u32)>::with_capacity(rec_count);
                let mut pos = 0usize;
                while pos < bytes.len() {
                    let k = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                    let m = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
                    pos += 12;
                    recs.push((k, m));
                }
                drop(bytes);
                bucket_classes += materialize_class_records(
                    recs,
                    &qid_lookup,
                    &mut writer,
                    &mut qids,
                    &mut written,
                )?;
                let _ = std::fs::remove_file(&sp);
            }
            let _ = std::fs::remove_dir(&sub_dir);
        }

        global_classes += bucket_classes;
        if (b + 1) % 64 == 0 || b + 1 == bucket_count {
            eprintln!(
                "  merged bucket {}/{} classes={} written_so_far={} elapsed={:.1}s",
                b + 1,
                bucket_count,
                bucket_classes,
                written,
                start.elapsed().as_secs_f64()
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    let finished = writer.finish()?;
    debug_assert_eq!(finished, written);
    println!("q_external_class raw_pairs={raw}");
    println!("q_external_class emitted_class_records={emitted_records}");
    println!("q_external_class global_classes={global_classes}");
    println!("q_external_class output_states={finished}");
    println!(
        "q_external_class removed_dupes={}",
        raw.saturating_sub(finished)
    );
    println!(
        "q_external_class output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "q_external_class output_u24_plus_fp16_bytes={} ({:.3} GiB)",
        finished * 5,
        finished as f64 * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external class step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

fn materialize_field_partitioned_bucket(
    bucket_index: usize,
    bucket_bits: usize,
    bytes: Vec<u8>,
    qid_lookup: &[u32],
    writer: &mut FieldChainWriter,
    qids: &mut Vec<u32>,
    written: &mut u64,
) -> Result<u64> {
    use anyhow::bail;
    if bytes.len() % 12 != 0 {
        bail!(
            "bucket {} size {} is not divisible by 12",
            bucket_index,
            bytes.len()
        );
    }
    let rec_count = bytes.len() / 12;
    if rec_count == 0 {
        return Ok(0);
    }

    let field_low_bits = (Q_FIELD_ID_BITS as usize).saturating_sub(bucket_bits);
    let fields_in_bucket = 1usize << field_low_bits;
    let first_field = (bucket_index as u32) << field_low_bits;
    let field_mask = fields_in_bucket - 1;

    // Count records per field. Field is the outer independent unit; different fields
    // never need comparison, sorting, or merging against each other.
    let mut counts = vec![0usize; fields_in_bucket];
    let mut pos = 0usize;
    while pos < bytes.len() {
        let k = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let field = (k >> 32) as usize;
        let local = field & field_mask;
        counts[local] += 1;
        pos += 12;
    }

    let mut offsets = vec![0usize; fields_in_bucket + 1];
    for i in 0..fields_in_bucket {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    debug_assert_eq!(offsets[fields_in_bucket], rec_count);

    let mut cursor = offsets[..fields_in_bucket].to_vec();
    let mut field_recs = vec![(0u32, 0u32); rec_count]; // (bag_tail, avail_mask)
    pos = 0usize;
    while pos < bytes.len() {
        let k = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let m = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
        let field = (k >> 32) as usize;
        let local = field & field_mask;
        let dst = cursor[local];
        cursor[local] += 1;
        field_recs[dst] = (k as u32, m);
        pos += 12;
    }
    drop(cursor);
    drop(bytes);

    let mut classes = 0u64;
    for local in 0..fields_in_bucket {
        let begin = offsets[local];
        let end = offsets[local + 1];
        if begin == end {
            continue;
        }
        let field = first_field + local as u32;
        let slice = &mut field_recs[begin..end];
        slice.sort_unstable_by_key(|&(bag_tail, _)| bag_tail);

        qids.clear();
        let mut i = 0usize;
        while i < slice.len() {
            let bag_tail = slice[i].0;
            let mut mask = 0u32;
            while i < slice.len() && slice[i].0 == bag_tail {
                mask |= slice[i].1;
                i += 1;
            }
            classes += 1;
            let mut m = mask;
            while m != 0 {
                let avail = m.trailing_zeros() as u8;
                m &= m - 1;
                let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
                if qid != QID_MISSING {
                    qids.push(qid);
                }
            }
        }
        // Writer requires qids sorted within each field. After (bag_tail,avail)
        // expansion qid order is not guaranteed, so keep the final per-field sort.
        qids.sort_unstable();
        qids.dedup();
        for &qid in qids.iter() {
            writer.push_pair_key(q_pair_key(field, qid))?;
        }
        *written += qids.len() as u64;
    }
    qids.clear();
    Ok(classes)
}

/// Build an equivalence class for each concrete child qid based on the exact
/// reverse-class output it can generate for all placed pieces.  Two qids with
/// identical signatures are interchangeable for layer (d+1)->d generation:
/// they produce the same `(parent_bag_tail, avail_mask)` records for every
/// placed piece, so within a fixed child field we only need to process one
/// representative.
fn build_qrev_signature_classes(qgraph: &QRevClassGraph) -> (Vec<u32>, Vec<u32>) {
    let qn = qgraph.offsets.len();
    let mut norm_id_by_qid = vec![0u32; qn];
    let mut rep_qid_by_norm = Vec::<u32>::new();
    let mut table = HashMap::<Vec<u64>, u32>::with_capacity(qn * 2);

    for qid in 0..qn {
        let mut sig = Vec::<u64>::new();
        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            sig.push(u64::MAX - placed as u64); // piece delimiter
            for rec in qgraph.records_for(qid as u32, placed) {
                sig.push(((rec.bag_tail as u64) << 32) | rec.avail_mask as u64);
            }
        }
        let nid = if let Some(&id) = table.get(&sig) {
            id
        } else {
            let id = rep_qid_by_norm.len() as u32;
            table.insert(sig, id);
            rep_qid_by_norm.push(qid as u32);
            id
        };
        norm_id_by_qid[qid] = nid;
    }
    (norm_id_by_qid, rep_qid_by_norm)
}

pub fn external_step_q_class_fieldreduce(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    config: QExternalClassConfig,
) -> Result<u64> {
    use anyhow::{bail, Context};
    use std::io::{BufWriter, Read, Write};
    use std::sync::{Arc, Mutex};

    if config.bucket_bits == 0
        || config.bucket_bits > 16
        || config.bucket_bits as u64 > Q_FIELD_ID_BITS
    {
        bail!("bucket_bits must be in 1..=16 and <= {Q_FIELD_ID_BITS}");
    }
    // This path is already field-range bucketed and then field-local partitioned.
    // merge_sub_bits is deliberately ignored; pass 0 for clean logs.
    std::fs::create_dir_all(work_dir)?;
    for ent in std::fs::read_dir(work_dir)? {
        let ent = ent?;
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("qclass_field_bucket_") && name.ends_with(".bin") {
            std::fs::remove_file(ent.path())?;
        }
    }

    let start = Instant::now();
    let bucket_count = 1usize << config.bucket_bits;
    let block_chunk = 16usize;
    eprintln!(
        "q external class fieldreduce step blocks={} input_states={} buckets={} fields_per_bucket={} flush_records={} block_chunk={} threads={}",
        input.metas.len(),
        input.total_states,
        bucket_count,
        1usize << ((Q_FIELD_ID_BITS as usize) - config.bucket_bits),
        config.flush_records,
        block_chunk,
        rayon::current_num_threads()
    );

    let mut writer_vec = Vec::with_capacity(bucket_count);
    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_field_bucket_{b:04}.bin"));
        let f =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        writer_vec.push(Mutex::new(BufWriter::with_capacity(8 * 1024 * 1024, f)));
    }
    let writers = Arc::new(writer_vec);

    #[derive(Default, Clone, Copy)]
    struct Stats {
        raw: u64,
        local_chain_masked: u64,
        emitted_records: u64,
        flushed_records: u64,
    }

    fn flush_field_buffers(
        writers: &[Mutex<BufWriter<std::fs::File>>],
        buffers: &mut [Vec<(u64, u32)>],
        total_buffered: &mut usize,
    ) -> Result<usize> {
        let mut flushed = 0usize;
        for (b, buf) in buffers.iter_mut().enumerate() {
            if buf.is_empty() {
                continue;
            }
            let mut w = writers[b]
                .lock()
                .expect("field bucket writer mutex poisoned");
            let mut bytes = Vec::<u8>::with_capacity(buf.len() * 12);
            for &(k, m) in buf.iter() {
                bytes.extend_from_slice(&k.to_le_bytes());
                bytes.extend_from_slice(&m.to_le_bytes());
            }
            w.write_all(&bytes)?;
            flushed += buf.len();
            buf.clear();
        }
        *total_buffered = 0;
        Ok(flushed)
    }

    let block_indices = (0..input.metas.len()).collect::<Vec<_>>();
    let done_blocks = AtomicUsize::new(0);
    let stats_vec: Vec<Stats> = block_indices
        .par_chunks(block_chunk)
        .map(|chunk| -> Result<Stats> {
            let mut stats = Stats::default();
            let mut buffers: Vec<Vec<(u64, u32)>> = (0..bucket_count).map(|_| Vec::new()).collect();
            let mut total_buffered = 0usize;

            for &bi in chunk {
                let meta = &input.metas[bi];
                let qids_in = input.read_block_chains(bi);
                let child_field = meta.field_id;
                let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
                    std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

                for qid in qids_in {
                    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                        let recs = qgraph.records_for(qid, placed);
                        if recs.is_empty() {
                            continue;
                        }
                        let parent_count = rgraph.edges(child_field, placed).len() as u64;
                        if parent_count == 0 {
                            continue;
                        }
                        let map = &mut by_piece[placed as usize];
                        for rec in recs {
                            stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                            *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                        }
                    }
                }

                for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                    let map = &mut by_piece[placed as usize];
                    if map.is_empty() {
                        continue;
                    }
                    let parents = rgraph.edges(child_field, placed);
                    for &old_field in parents {
                        for (&bag_tail, &mask) in map.iter() {
                            let ck = class_key(old_field, bag_tail);
                            let b = q_bucket_of_class_key(ck, config.bucket_bits);
                            buffers[b].push((ck, mask));
                            total_buffered += 1;
                            stats.emitted_records += 1;
                            stats.local_chain_masked += mask.count_ones() as u64;
                        }
                    }
                    map.clear();
                }

                if total_buffered >= config.flush_records {
                    stats.flushed_records +=
                        flush_field_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
                }
            }
            stats.flushed_records +=
                flush_field_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
            let done = done_blocks.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
            if done % 1024 < block_chunk || done == input.metas.len() {
                eprintln!(
                    "  q external fieldreduce generation blocks_done={}/{} elapsed={:.1}s",
                    done,
                    input.metas.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            Ok(stats)
        })
        .collect::<Result<Vec<_>>>()?;

    let raw: u64 = stats_vec.iter().map(|s| s.raw).sum();
    let local_chain_masked: u64 = stats_vec.iter().map(|s| s.local_chain_masked).sum();
    let emitted_records: u64 = stats_vec.iter().map(|s| s.emitted_records).sum();
    let flushed_records: u64 = stats_vec.iter().map(|s| s.flushed_records).sum();
    for w in writers.iter() {
        w.lock()
            .expect("field bucket writer mutex poisoned")
            .flush()?;
    }
    drop(writers);

    eprintln!(
        "q external fieldreduce generation done raw={} emitted_class_records={} local_chain_masked={} flushed_records={} elapsed={:.1}s",
        raw, emitted_records, local_chain_masked, flushed_records, start.elapsed().as_secs_f64()
    );

    eprintln!("building compact qid lookup table for fieldreduce materialization...");
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );

    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut written = 0u64;
    let mut global_classes = 0u64;
    let mut qids = Vec::<u32>::new();

    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_field_bucket_{b:04}.bin"));
        if !path.exists() {
            continue;
        }
        let file_len = std::fs::metadata(&path)?.len();
        if file_len == 0 {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        let mut bytes = Vec::<u8>::new();
        std::fs::File::open(&path)
            .with_context(|| format!("open {}", path.display()))?
            .read_to_end(&mut bytes)?;
        let bucket_classes = materialize_field_partitioned_bucket(
            b,
            config.bucket_bits,
            bytes,
            &qid_lookup,
            &mut writer,
            &mut qids,
            &mut written,
        )?;
        global_classes += bucket_classes;
        if (b + 1) % 64 == 0 || b + 1 == bucket_count {
            eprintln!(
                "  fieldreduce bucket {}/{} classes={} written_so_far={} elapsed={:.1}s",
                b + 1,
                bucket_count,
                bucket_classes,
                written,
                start.elapsed().as_secs_f64()
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    let finished = writer.finish()?;
    debug_assert_eq!(finished, written);
    println!("q_external_class_fieldreduce raw_pairs={raw}");
    println!("q_external_class_fieldreduce emitted_class_records={emitted_records}");
    println!("q_external_class_fieldreduce local_chain_masked={local_chain_masked}");
    println!("q_external_class_fieldreduce global_classes={global_classes}");
    println!("q_external_class_fieldreduce output_states={finished}");
    println!(
        "q_external_class_fieldreduce removed_dupes={}",
        raw.saturating_sub(finished)
    );
    eprintln!(
        "q external class fieldreduce step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}
pub fn external_step_q_class_fieldreduce_preq(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    config: QExternalClassConfig,
) -> Result<u64> {
    use anyhow::{bail, Context};
    use std::io::{BufWriter, Read, Write};
    use std::sync::{Arc, Mutex};

    if config.bucket_bits == 0
        || config.bucket_bits > 16
        || config.bucket_bits as u64 > Q_FIELD_ID_BITS
    {
        bail!("bucket_bits must be in 1..=16 and <= {Q_FIELD_ID_BITS}");
    }
    // This path is already field-range bucketed and then field-local partitioned.
    // merge_sub_bits is deliberately ignored; pass 0 for clean logs.
    std::fs::create_dir_all(work_dir)?;
    for ent in std::fs::read_dir(work_dir)? {
        let ent = ent?;
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("qclass_field_bucket_") && name.ends_with(".bin") {
            std::fs::remove_file(ent.path())?;
        }
    }

    let start = Instant::now();
    let bucket_count = 1usize << config.bucket_bits;
    let block_chunk = 16usize;
    eprintln!(
        "q external class fieldreduce-preq step blocks={} input_states={} buckets={} fields_per_bucket={} flush_records={} block_chunk={} threads={}",
        input.metas.len(),
        input.total_states,
        bucket_count,
        1usize << ((Q_FIELD_ID_BITS as usize) - config.bucket_bits),
        config.flush_records,
        block_chunk,
        rayon::current_num_threads()
    );

    eprintln!("building qrev reverse-signature equivalence classes for preq dedupe...");
    let preq_start = Instant::now();
    let (norm_id_by_qid, rep_qid_by_norm) = build_qrev_signature_classes(qgraph);
    let norm_count = rep_qid_by_norm.len();
    eprintln!(
        "built qrev preq classes concrete_qids={} norm_qids={} ratio={:.6} elapsed={:.3}s",
        norm_id_by_qid.len(),
        norm_count,
        norm_count as f64 / norm_id_by_qid.len().max(1) as f64,
        preq_start.elapsed().as_secs_f64()
    );

    let mut writer_vec = Vec::with_capacity(bucket_count);
    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_field_bucket_{b:04}.bin"));
        let f =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        writer_vec.push(Mutex::new(BufWriter::with_capacity(8 * 1024 * 1024, f)));
    }
    let writers = Arc::new(writer_vec);

    #[derive(Default, Clone, Copy)]
    struct Stats {
        raw: u64,
        local_chain_masked: u64,
        emitted_records: u64,
        flushed_records: u64,
    }

    fn flush_field_buffers(
        writers: &[Mutex<BufWriter<std::fs::File>>],
        buffers: &mut [Vec<(u64, u32)>],
        total_buffered: &mut usize,
    ) -> Result<usize> {
        let mut flushed = 0usize;
        for (b, buf) in buffers.iter_mut().enumerate() {
            if buf.is_empty() {
                continue;
            }
            let mut w = writers[b]
                .lock()
                .expect("field bucket writer mutex poisoned");
            let mut bytes = Vec::<u8>::with_capacity(buf.len() * 12);
            for &(k, m) in buf.iter() {
                bytes.extend_from_slice(&k.to_le_bytes());
                bytes.extend_from_slice(&m.to_le_bytes());
            }
            w.write_all(&bytes)?;
            flushed += buf.len();
            buf.clear();
        }
        *total_buffered = 0;
        Ok(flushed)
    }

    let block_indices = (0..input.metas.len()).collect::<Vec<_>>();
    let done_blocks = AtomicUsize::new(0);
    let stats_vec: Vec<Stats> = block_indices
        .par_chunks(block_chunk)
        .map(|chunk| -> Result<Stats> {
            let mut stats = Stats::default();
            let mut buffers: Vec<Vec<(u64, u32)>> = (0..bucket_count).map(|_| Vec::new()).collect();
            let mut total_buffered = 0usize;
            let mut seen_norm = vec![0u32; norm_count];
            let mut epoch = 1u32;
            let mut unique_rep_qids = Vec::<u32>::new();

            for &bi in chunk {
                let meta = &input.metas[bi];
                let qids_in = input.read_block_chains(bi);
                let child_field = meta.field_id;
                let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
                    std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

                unique_rep_qids.clear();
                if epoch == u32::MAX {
                    seen_norm.fill(0);
                    epoch = 1;
                }
                for qid in qids_in {
                    let nid = norm_id_by_qid[qid as usize] as usize;
                    if seen_norm[nid] != epoch {
                        seen_norm[nid] = epoch;
                        unique_rep_qids.push(rep_qid_by_norm[nid]);
                    }
                }
                epoch = epoch.wrapping_add(1);

                for qid in unique_rep_qids.iter().copied() {
                    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                        let recs = qgraph.records_for(qid, placed);
                        if recs.is_empty() {
                            continue;
                        }
                        let parent_count = rgraph.edges(child_field, placed).len() as u64;
                        if parent_count == 0 {
                            continue;
                        }
                        let map = &mut by_piece[placed as usize];
                        for rec in recs {
                            stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                            *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                        }
                    }
                }

                for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                    let map = &mut by_piece[placed as usize];
                    if map.is_empty() {
                        continue;
                    }
                    let parents = rgraph.edges(child_field, placed);
                    for &old_field in parents {
                        for (&bag_tail, &mask) in map.iter() {
                            let ck = class_key(old_field, bag_tail);
                            let b = q_bucket_of_class_key(ck, config.bucket_bits);
                            buffers[b].push((ck, mask));
                            total_buffered += 1;
                            stats.emitted_records += 1;
                            stats.local_chain_masked += mask.count_ones() as u64;
                        }
                    }
                    map.clear();
                }

                if total_buffered >= config.flush_records {
                    stats.flushed_records +=
                        flush_field_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
                }
            }
            stats.flushed_records +=
                flush_field_buffers(&writers, &mut buffers, &mut total_buffered)? as u64;
            let done = done_blocks.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
            if done % 1024 < block_chunk || done == input.metas.len() {
                eprintln!(
                    "  q external fieldreduce-preq generation blocks_done={}/{} elapsed={:.1}s",
                    done,
                    input.metas.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            Ok(stats)
        })
        .collect::<Result<Vec<_>>>()?;

    let raw: u64 = stats_vec.iter().map(|s| s.raw).sum();
    let local_chain_masked: u64 = stats_vec.iter().map(|s| s.local_chain_masked).sum();
    let emitted_records: u64 = stats_vec.iter().map(|s| s.emitted_records).sum();
    let flushed_records: u64 = stats_vec.iter().map(|s| s.flushed_records).sum();
    for w in writers.iter() {
        w.lock()
            .expect("field bucket writer mutex poisoned")
            .flush()?;
    }
    drop(writers);

    eprintln!(
        "q external fieldreduce-preq generation done raw={} emitted_class_records={} local_chain_masked={} flushed_records={} elapsed={:.1}s",
        raw, emitted_records, local_chain_masked, flushed_records, start.elapsed().as_secs_f64()
    );

    eprintln!("building compact qid lookup table for fieldreduce materialization...");
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );

    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut written = 0u64;
    let mut global_classes = 0u64;
    let mut qids = Vec::<u32>::new();

    for b in 0..bucket_count {
        let path = work_dir.join(format!("qclass_field_bucket_{b:04}.bin"));
        if !path.exists() {
            continue;
        }
        let file_len = std::fs::metadata(&path)?.len();
        if file_len == 0 {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        let mut bytes = Vec::<u8>::new();
        std::fs::File::open(&path)
            .with_context(|| format!("open {}", path.display()))?
            .read_to_end(&mut bytes)?;
        let bucket_classes = materialize_field_partitioned_bucket(
            b,
            config.bucket_bits,
            bytes,
            &qid_lookup,
            &mut writer,
            &mut qids,
            &mut written,
        )?;
        global_classes += bucket_classes;
        if (b + 1) % 64 == 0 || b + 1 == bucket_count {
            eprintln!(
                "  fieldreduce-preq bucket {}/{} classes={} written_so_far={} elapsed={:.1}s",
                b + 1,
                bucket_count,
                bucket_classes,
                written,
                start.elapsed().as_secs_f64()
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    let finished = writer.finish()?;
    debug_assert_eq!(finished, written);
    println!("q_external_class_fieldreduce_preq raw_pairs={raw}");
    println!("q_external_class_fieldreduce_preq emitted_class_records={emitted_records}");
    println!("q_external_class_fieldreduce_preq local_chain_masked={local_chain_masked}");
    println!("q_external_class_fieldreduce_preq global_classes={global_classes}");
    println!("q_external_class_fieldreduce_preq output_states={finished}");
    println!(
        "q_external_class_fieldreduce_preq removed_dupes={}",
        raw.saturating_sub(finished)
    );
    eprintln!(
        "q external class fieldreduce-preq step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

#[derive(Debug, Clone)]
struct QLiveChildTouch {
    block_idx: usize,
    child_field: FieldId,
    parents: Vec<FieldId>,
    min_parent: FieldId,
    max_parent: FieldId,
    median_parent: FieldId,
}

#[derive(Debug, Clone, Copy, Default)]
struct QLiveStats {
    raw: u64,
    local_chain_masked: u64,
    emitted_records: u64,
}

fn qlive_compute_child_touch(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qgraph: &QRevClassGraph,
    bi: usize,
) -> QLiveChildTouch {
    let meta = &input.metas[bi];
    let child_field = meta.field_id;
    let qids = input.read_block_chains(bi);
    let mut piece_active = [false; crate::piece::PIECE_COUNT];

    for qid in qids {
        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            if piece_active[placed as usize] {
                continue;
            }
            if !rgraph.edges(child_field, placed).is_empty()
                && !qgraph.records_for(qid, placed).is_empty()
            {
                piece_active[placed as usize] = true;
            }
        }
        if piece_active.iter().all(|&x| x) {
            break;
        }
    }

    let mut parents = Vec::<FieldId>::new();
    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
        if piece_active[placed as usize] {
            parents.extend_from_slice(rgraph.edges(child_field, placed));
        }
    }
    parents.sort_unstable();
    parents.dedup();

    let (min_parent, max_parent, median_parent) = if parents.is_empty() {
        (FieldId::MAX, FieldId::MAX, FieldId::MAX)
    } else {
        (
            parents[0],
            *parents.last().unwrap(),
            parents[parents.len() / 2],
        )
    };

    QLiveChildTouch {
        block_idx: bi,
        child_field,
        parents,
        min_parent,
        max_parent,
        median_parent,
    }
}

fn qlive_build_child_order_and_counts(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qgraph: &QRevClassGraph,
) -> (Vec<QLiveChildTouch>, Vec<u32>) {
    let start = Instant::now();
    eprintln!(
        "q live prepass start blocks={} input_states={} threads={}",
        input.metas.len(),
        input.total_states,
        rayon::current_num_threads()
    );

    let done = AtomicUsize::new(0);
    let mut touches: Vec<QLiveChildTouch> = (0..input.metas.len())
        .into_par_iter()
        .map(|bi| {
            let t = qlive_compute_child_touch(rgraph, input, qgraph, bi);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 1024 == 0 || d == input.metas.len() {
                eprintln!(
                    "  q live prepass blocks_done={}/{} elapsed={:.1}s",
                    d,
                    input.metas.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            t
        })
        .collect();

    let mut remaining = vec![0u32; crate::graph::NUM_FIELDS];
    let mut touched_edges = 0u64;
    let mut max_parents = 0usize;
    for t in &touches {
        touched_edges += t.parents.len() as u64;
        max_parents = max_parents.max(t.parents.len());
        for &p in &t.parents {
            remaining[p as usize] = remaining[p as usize].saturating_add(1);
        }
    }

    // Keep parent fields with similar neighborhoods close together.  This is a
    // cheap interval-style ordering; the exact optimum is a graph layout problem.
    touches.sort_unstable_by(|a, b| {
        (
            a.median_parent,
            a.min_parent,
            a.max_parent,
            std::cmp::Reverse(a.parents.len()),
            a.child_field,
        )
            .cmp(&(
                b.median_parent,
                b.min_parent,
                b.max_parent,
                std::cmp::Reverse(b.parents.len()),
                b.child_field,
            ))
    });

    let active_parent_fields = remaining.iter().filter(|&&c| c != 0).count();
    eprintln!(
        "q live prepass done parent_fields={} child_parent_edges={} avg_parents_per_child={:.3} max_parents_per_child={} elapsed={:.3}s",
        active_parent_fields,
        touched_edges,
        touched_edges as f64 / input.metas.len().max(1) as f64,
        max_parents,
        start.elapsed().as_secs_f64()
    );

    (touches, remaining)
}

struct QLiveSpoolWriter {
    file: std::io::BufWriter<std::fs::File>,
    metas: Vec<(FieldId, u32, u64)>,
    total_qids: u64,
    encode_buf: Vec<u8>,
}

impl QLiveSpoolWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        Ok(Self {
            file,
            metas: Vec::new(),
            total_qids: 0,
            encode_buf: Vec::with_capacity(1 << 20),
        })
    }

    fn write_field(&mut self, field: FieldId, qids: &[u32]) -> Result<()> {
        use std::io::Write;
        if qids.is_empty() {
            return Ok(());
        }
        let offset = self.total_qids;
        let bytes_len = qids.len().saturating_mul(3);
        self.encode_buf.resize(bytes_len, 0);
        for (i, &qid) in qids.iter().enumerate() {
            let j = i * 3;
            self.encode_buf[j] = (qid & 0xff) as u8;
            self.encode_buf[j + 1] = ((qid >> 8) & 0xff) as u8;
            self.encode_buf[j + 2] = ((qid >> 16) & 0xff) as u8;
        }
        self.file.write_all(&self.encode_buf)?;
        self.encode_buf.clear();
        self.metas.push((field, qids.len() as u32, offset));
        self.total_qids += qids.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        use std::io::Write;
        self.file.flush()?;
        Ok(())
    }
}

fn qlive_finalize_field(
    field: FieldId,
    acc: HashMap<u32, u32>,
    qid_lookup: &[u32],
    qids: &mut Vec<u32>,
    spool: &mut QLiveSpoolWriter,
) -> Result<u64> {
    qids.clear();
    for (bag_tail, mask) in acc {
        let mut m = mask;
        while m != 0 {
            let avail = m.trailing_zeros() as u8;
            m &= m - 1;
            let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
            if qid != QID_MISSING {
                qids.push(qid);
            }
        }
    }
    qids.sort_unstable();
    qids.dedup();
    let n = qids.len() as u64;
    spool.write_field(field, qids)?;
    qids.clear();
    Ok(n)
}

struct QLiveClassSpoolWriter {
    file: std::io::BufWriter<std::fs::File>,
    metas: Vec<(FieldId, u32, u64)>,
    total_records: u64,
}

impl QLiveClassSpoolWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        Ok(Self {
            file,
            metas: Vec::new(),
            total_records: 0,
        })
    }

    fn write_field_classes(&mut self, field: FieldId, classes: &[(u32, u32)]) -> Result<u64> {
        use std::io::Write;
        if classes.is_empty() {
            return Ok(0);
        }
        let offset = self.total_records;
        for &(bag_tail, mask) in classes {
            self.file.write_all(&bag_tail.to_le_bytes())?;
            self.file.write_all(&mask.to_le_bytes())?;
        }
        self.metas.push((field, classes.len() as u32, offset));
        self.total_records += classes.len() as u64;
        Ok(classes.len() as u64)
    }

    fn flush(&mut self) -> Result<()> {
        use std::io::Write;
        self.file.flush()?;
        Ok(())
    }
}

fn qlive_write_closed_class_run_with_spills(
    field: FieldId,
    mut acc: HashMap<u32, u32>,
    spill: &mut QLiveSpillWriter,
    class_tmp: &mut Vec<(u32, u32)>,
    class_spool: &mut QLiveClassSpoolWriter,
) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    if let Some(runs) = spill.runs.remove(&field) {
        spill.flush()?;
        let mut file = std::fs::File::open(&spill.path)?;
        let mut buf = Vec::<u8>::new();
        for run in runs {
            let bytes_len = run.count as usize * 8;
            buf.resize(bytes_len, 0);
            file.seek(SeekFrom::Start(run.offset_records * 8))?;
            file.read_exact(&mut buf)?;
            let mut pos = 0usize;
            while pos < buf.len() {
                let bag_tail = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
                let mask = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
                pos += 8;
                *acc.entry(bag_tail).or_insert(0) |= mask;
            }
        }
    }
    class_tmp.clear();
    class_tmp.extend(acc.into_iter().filter(|(_, mask)| *mask != 0));
    class_tmp.sort_unstable_by_key(|&(bag_tail, _)| bag_tail);
    let n = class_spool.write_field_classes(field, class_tmp)?;
    class_tmp.clear();
    Ok(n)
}

fn qlive_materialize_class_spool(
    class_spool_path: &Path,
    mut metas: Vec<(FieldId, u32, u64)>,
    qid_lookup: &[u32],
    output_prefix: &Path,
) -> Result<u64> {
    use anyhow::bail;
    use std::io::{Read, Seek, SeekFrom};
    metas.sort_unstable_by_key(|&(field, _, _)| field);
    for w in metas.windows(2) {
        if w[0].0 >= w[1].0 {
            bail!(
                "live class spool has duplicate or unsorted field after sort: {} then {}",
                w[0].0,
                w[1].0
            );
        }
    }

    let mut class_file = std::fs::File::open(class_spool_path)?;
    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut buf = Vec::<u8>::new();
    let mut qids = Vec::<u32>::new();
    let mut total = 0u64;
    let st = Instant::now();
    let total_fields = metas.len();
    for (i, (field, count, offset)) in metas.into_iter().enumerate() {
        let bytes_len = count as usize * 8;
        buf.resize(bytes_len, 0);
        class_file.seek(SeekFrom::Start(offset * 8))?;
        class_file.read_exact(&mut buf)?;
        qids.clear();
        let mut pos = 0usize;
        while pos < buf.len() {
            let bag_tail = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let mut mask = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;
            while mask != 0 {
                let avail = mask.trailing_zeros() as u8;
                mask &= mask - 1;
                let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
                if qid != QID_MISSING {
                    qids.push(qid);
                }
            }
        }
        qids.sort_unstable();
        qids.dedup();
        for &qid in &qids {
            writer.push_pair_key(q_pair_key(field, qid))?;
        }
        total += qids.len() as u64;
        let done = i + 1;
        if done % 65536 == 0 || done == total_fields {
            eprintln!(
                "  q live deferred materialize fields_done={}/{} output_states={} elapsed={:.1}s",
                done,
                total_fields,
                total,
                st.elapsed().as_secs_f64()
            );
        }
    }
    writer.finish()
}

fn qlive_assemble_spool(
    spool_path: &Path,
    mut metas: Vec<(FieldId, u32, u64)>,
    output_prefix: &Path,
) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    metas.sort_unstable_by_key(|&(field, _, _)| field);

    let mut spool_file = std::fs::File::open(spool_path)?;
    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut buf = Vec::<u8>::new();
    let mut qids = Vec::<u32>::new();
    let mut i = 0usize;

    while i < metas.len() {
        let field = metas[i].0;
        qids.clear();
        let mut duplicate_blocks = 0usize;

        while i < metas.len() && metas[i].0 == field {
            let (_, count, offset) = metas[i];
            let bytes_len = count as usize * 3;
            buf.resize(bytes_len, 0);
            spool_file.seek(SeekFrom::Start(offset * 3))?;
            spool_file.read_exact(&mut buf)?;
            for j in 0..count as u64 {
                qids.push(crate::layer_store::read_u24_le_at(&buf, j));
            }
            duplicate_blocks += 1;
            i += 1;
        }

        if duplicate_blocks > 1 {
            qids.sort_unstable();
            qids.dedup();
            eprintln!(
                "    q live assemble merged duplicate field={} blocks={} qstates={}",
                field,
                duplicate_blocks,
                qids.len()
            );
        }

        for &qid in &qids {
            writer.push_pair_key(q_pair_key(field, qid))?;
        }
    }
    writer.finish()
}

// -----------------------------------------------------------------------------
// PreQ frontier-direct path.
//
// This is the simplified algorithm:
//   layer8 block qids -> block-local preq dedupe -> direct q7 insert into
//   live parent-field accumulators -> close finalized fields to a compact spool
//   -> assemble sorted output layer.
//
// A preq is the layer8 qchain with the last/revealed tail piece removed and
// returned to the bag mask through the reverse bag-boundary rule.  It is not a
// persisted layer state; it is only a generation view local to each child field.
// -----------------------------------------------------------------------------

const PREQ_MISSING: u32 = u32::MAX;

#[inline]
fn encode_tail4(q: [u8; 4]) -> u32 {
    let mut x = 0u32;
    let mut mul = 1u32;
    for &p in &q {
        debug_assert!(p < crate::piece::PIECE_COUNT as u8);
        x += (p as u32) * mul;
        mul *= 7;
    }
    x
}

#[inline]
fn decode_tail4(mut x: u32) -> [u8; 4] {
    let mut q = [0u8; 4];
    for dst in &mut q {
        *dst = (x % 7) as u8;
        x /= 7;
    }
    q
}

#[inline]
fn preq_key(avail: u8, tail4: u32, bag_mask: u8) -> u32 {
    debug_assert!((avail as usize) < crate::qstate::AVAIL_PAIR_COUNT);
    debug_assert!(tail4 < 7u32.pow(4));
    debug_assert!(bag_mask > 0 && bag_mask < 128);
    (avail as u32) | (tail4 << 5) | ((bag_mask as u32) << 17)
}

#[inline]
fn unpack_preq_key(key: u32) -> (u8, u32, u8) {
    let avail = (key & 0x1F) as u8;
    let tail4 = (key >> 5) & 0x0FFF;
    let bag = ((key >> 17) & 0x7F) as u8;
    (avail, tail4, bag)
}

#[inline]
fn qstate_to_preq_key(s: crate::qstate::QChainState) -> Option<u32> {
    let tail = crate::qstate::decode_tail5(s.tail5);
    let removed = tail[4];
    let old_bag = crate::qstate::reverse_bag(s.bag_mask, removed)?;
    let tail4 = encode_tail4([tail[0], tail[1], tail[2], tail[3]]);
    Some(preq_key(s.avail, tail4, old_bag))
}

#[derive(Debug, Clone)]
struct PreqDirectEntry {
    key: u32,
    q7_by_piece: [Vec<u32>; crate::piece::PIECE_COUNT],
    total_q7: usize,
}

fn build_preq_direct_table(q_index: &QStateIndex) -> (Vec<u32>, Vec<PreqDirectEntry>) {
    let start = Instant::now();
    eprintln!("building structural preq direct table...");

    let mut key_to_id = HashMap::<u32, u32>::with_capacity(q_index.states.len() / 2);
    let mut qid_to_preq = vec![PREQ_MISSING; q_index.states.len()];
    let mut keys = Vec::<u32>::new();

    for (qid, &s) in q_index.states.iter().enumerate() {
        let Some(k) = qstate_to_preq_key(s) else {
            continue;
        };
        let id = if let Some(&id) = key_to_id.get(&k) {
            id
        } else {
            let id = keys.len() as u32;
            key_to_id.insert(k, id);
            keys.push(k);
            id
        };
        qid_to_preq[qid] = id;
    }

    let mut entries = Vec::<PreqDirectEntry>::with_capacity(keys.len());
    for &key in &keys {
        let (avail, tail4_enc, bag_mask) = unpack_preq_key(key);
        let tail4 = decode_tail4(tail4_enc);
        let (x, y) = crate::qstate::decode_avail_pair(avail);
        let choices = [(x, y), (y, x)]; // (remaining, old_q1)
        let mut by_piece: [Vec<u32>; crate::piece::PIECE_COUNT] =
            std::array::from_fn(|_| Vec::new());

        for (choice_idx, &(remaining, old_q1)) in choices.iter().enumerate() {
            if x == y && choice_idx == 1 {
                break;
            }
            let old_tail5 =
                crate::qstate::encode_tail5([old_q1, tail4[0], tail4[1], tail4[2], tail4[3]]);
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let old = crate::qstate::QChainState {
                    avail: crate::qstate::encode_avail_pair(placed, remaining),
                    tail5: old_tail5,
                    bag_mask,
                };
                let key = crate::qstate::qchain_key(old);
                if let Some(&qid) = q_index.key_to_id.get(&key) {
                    by_piece[placed as usize].push(qid as u32);
                }
            }
        }
        let mut total = 0usize;
        for v in by_piece.iter_mut() {
            v.sort_unstable();
            v.dedup();
            total += v.len();
        }
        entries.push(PreqDirectEntry {
            key,
            q7_by_piece: by_piece,
            total_q7: total,
        });
    }

    let mapped = qid_to_preq.iter().filter(|&&x| x != PREQ_MISSING).count();
    eprintln!(
        "built structural preq direct table concrete_qids={} mapped_qids={} preq_ids={} ratio={:.6} elapsed={:.3}s",
        q_index.states.len(),
        mapped,
        entries.len(),
        entries.len() as f64 / q_index.states.len().max(1) as f64,
        start.elapsed().as_secs_f64()
    );
    (qid_to_preq, entries)
}

#[derive(Debug)]
enum QidFieldAcc {
    Vec(Vec<u32>),
    Bitset { bits: Vec<u64> },
}

impl QidFieldAcc {
    fn new() -> Self {
        Self::Vec(Vec::new())
    }

    fn estimated_bytes(&self) -> usize {
        match self {
            QidFieldAcc::Vec(v) => v.capacity().saturating_mul(std::mem::size_of::<u32>()),
            QidFieldAcc::Bitset { bits } => {
                bits.capacity().saturating_mul(std::mem::size_of::<u64>())
            }
        }
    }

    fn finalized_vec(self) -> Vec<u32> {
        let mut out = Vec::<u32>::new();
        match self {
            QidFieldAcc::Vec(mut v) => {
                v.sort_unstable();
                v.dedup();
                out = v;
            }
            QidFieldAcc::Bitset { bits } => {
                out.reserve(bits.len().min(817_740));
                for (wi, mut w) in bits.into_iter().enumerate() {
                    while w != 0 {
                        let b = w.trailing_zeros() as usize;
                        w &= w - 1;
                        out.push((wi * 64 + b) as u32);
                    }
                }
            }
        }
        out
    }

    fn insert_all(&mut self, qids: &[u32], threshold: usize, bit_words: usize) {
        if qids.is_empty() {
            return;
        }
        match self {
            QidFieldAcc::Vec(v) => {
                v.extend_from_slice(qids);
                if v.len() >= threshold {
                    let mut bits = vec![0u64; bit_words];
                    for &qid in v.iter() {
                        let i = qid as usize;
                        bits[i >> 6] |= 1u64 << (i & 63);
                    }
                    *self = QidFieldAcc::Bitset { bits };
                }
            }
            QidFieldAcc::Bitset { bits } => {
                for &qid in qids {
                    let i = qid as usize;
                    bits[i >> 6] |= 1u64 << (i & 63);
                }
            }
        }
    }

    fn finalize_into(self, out: &mut Vec<u32>) -> u64 {
        out.clear();
        match self {
            QidFieldAcc::Vec(mut v) => {
                v.sort_unstable();
                v.dedup();
                let n = v.len() as u64;
                out.extend_from_slice(&v);
                n
            }
            QidFieldAcc::Bitset { bits } => {
                for (wi, mut w) in bits.into_iter().enumerate() {
                    while w != 0 {
                        let b = w.trailing_zeros() as usize;
                        w &= w - 1;
                        out.push((wi * 64 + b) as u32);
                    }
                }
                // The word scan is already sorted and contains no duplicates.
                out.len() as u64
            }
        }
    }
}

#[inline]
fn build_unique_preqs_for_block(
    input: &FieldChainLayer,
    qid_to_preq: &[u32],
    bi: usize,
) -> Vec<u32> {
    let qids = input.read_block_chains(bi);
    let mut preqs = Vec::<u32>::with_capacity(qids.len());
    for qid in qids {
        let pid = qid_to_preq[qid as usize];
        if pid != PREQ_MISSING {
            preqs.push(pid);
        }
    }
    preqs.sort_unstable();
    preqs.dedup();
    preqs
}

fn qpreq_compute_child_touch_from_preqs(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    preqs: &[u32],
    preq_entries: &[PreqDirectEntry],
    bi: usize,
) -> QLiveChildTouch {
    let meta = &input.metas[bi];
    let child_field = meta.field_id;
    let mut piece_active = [false; crate::piece::PIECE_COUNT];

    for &pid in preqs {
        let entry = &preq_entries[pid as usize];
        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let p = placed as usize;
            if piece_active[p] {
                continue;
            }
            if !entry.q7_by_piece[p].is_empty() && !rgraph.edges(child_field, placed).is_empty() {
                piece_active[p] = true;
            }
        }
        if piece_active.iter().all(|&x| x) {
            break;
        }
    }

    let mut parents = Vec::<FieldId>::new();
    for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
        if piece_active[placed as usize] {
            parents.extend_from_slice(rgraph.edges(child_field, placed));
        }
    }
    parents.sort_unstable();
    parents.dedup();

    let (min_parent, max_parent, median_parent) = if parents.is_empty() {
        (FieldId::MAX, FieldId::MAX, FieldId::MAX)
    } else {
        (
            parents[0],
            *parents.last().unwrap(),
            parents[parents.len() / 2],
        )
    };

    QLiveChildTouch {
        block_idx: bi,
        child_field,
        parents,
        min_parent,
        max_parent,
        median_parent,
    }
}

fn qpreq_build_counts_and_child_order_inputs(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qid_to_preq: &[u32],
    preq_entries: &[PreqDirectEntry],
) -> Result<(Vec<QLiveChildTouch>, Vec<u32>, Vec<u32>, u64, u64)> {
    let start = Instant::now();
    eprintln!(
        "q preq-direct streaming prepass start blocks={} input_states={} threads={}",
        input.metas.len(),
        input.total_states,
        rayon::current_num_threads()
    );
    let done = AtomicUsize::new(0);

    let items: Vec<(QLiveChildTouch, u32)> = (0..input.metas.len())
        .into_par_iter()
        .map(|bi| {
            let preqs = build_unique_preqs_for_block(input, qid_to_preq, bi);
            let count = preqs.len() as u32;
            let t = qpreq_compute_child_touch_from_preqs(rgraph, input, &preqs, preq_entries, bi);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 1024 == 0 || d == input.metas.len() {
                eprintln!(
                    "  q preq-direct streaming prepass blocks_done={}/{} elapsed={:.1}s",
                    d,
                    input.metas.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            (t, count)
        })
        .collect();

    let mut touches = Vec::<QLiveChildTouch>::with_capacity(items.len());
    let mut preq_counts = vec![0u32; input.metas.len()];
    let mut preq_states = 0u64;
    for (t, count) in items {
        preq_counts[t.block_idx] = count;
        preq_states += count as u64;
        touches.push(t);
    }

    let mut remaining = vec![0u32; crate::graph::NUM_FIELDS];
    let mut touched_edges = 0u64;
    let mut max_parents = 0usize;
    for t in &touches {
        touched_edges += t.parents.len() as u64;
        max_parents = max_parents.max(t.parents.len());
        for &p in &t.parents {
            remaining[p as usize] = remaining[p as usize].saturating_add(1);
        }
    }

    touches.sort_unstable_by(|a, b| {
        (
            a.median_parent,
            a.min_parent,
            a.max_parent,
            std::cmp::Reverse(a.parents.len()),
            a.child_field,
        )
            .cmp(&(
                b.median_parent,
                b.min_parent,
                b.max_parent,
                std::cmp::Reverse(b.parents.len()),
                b.child_field,
            ))
    });

    eprintln!(
        "q preq-direct streaming prepass done parent_fields={} child_parent_edges={} avg_parents_per_child={:.3} max_parents_per_child={} preq_states={} ratio_vs_input={:.6} elapsed={:.3}s",
        remaining.iter().filter(|&&c| c != 0).count(),
        touched_edges,
        touched_edges as f64 / input.metas.len().max(1) as f64,
        max_parents,
        preq_states,
        preq_states as f64 / input.total_states.max(1) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok((
        touches,
        remaining,
        preq_counts,
        input.total_states,
        preq_states,
    ))
}

fn qpreq_parent_weight(preq_counts: &[u32], touches: &[QLiveChildTouch]) -> Vec<u64> {
    let mut parent_weight = vec![0u64; crate::graph::NUM_FIELDS];
    for t in touches {
        let w = preq_counts[t.block_idx] as u64;
        for &p in &t.parents {
            parent_weight[p as usize] = parent_weight[p as usize].saturating_add(w.max(1));
        }
    }
    parent_weight
}

fn qpreq_build_frontier_order(
    preq_counts: &[u32],
    touches_summary_order: &[QLiveChildTouch],
    remaining0: &[u32],
) -> Vec<QLiveChildTouch> {
    let start = Instant::now();
    let mut by_child = touches_summary_order.to_vec();
    by_child.sort_unstable_by_key(|t| t.block_idx);
    let summary_order: Vec<usize> = touches_summary_order.iter().map(|t| t.block_idx).collect();
    let parent_weight = qpreq_parent_weight(preq_counts, &by_child);
    let children_of_parent = qlive_children_of_parent(&by_child);

    let mut remaining = remaining0.to_vec();
    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut processed = vec![false; by_child.len()];
    let mut heap = std::collections::BinaryHeap::<QLiveFrontierOrderItem>::new();
    let mut seed_pos = 0usize;
    let mut order = Vec::<usize>::with_capacity(by_child.len());
    let mut stale_pops = 0u64;
    let mut active_fields = 0u64;
    let mut active_weight = 0u128;
    let mut max_active_fields = 0u64;
    let mut max_active_weight = 0u128;
    let mut active_fields_sum = 0u128;
    let mut active_weight_sum = 0u128;
    let mut finalized = 0u64;

    let push_child = |ci: usize,
                      heap: &mut std::collections::BinaryHeap<QLiveFrontierOrderItem>,
                      processed: &[bool],
                      remaining: &[u32],
                      open: &[u8],
                      touches: &[QLiveChildTouch],
                      parent_weight: &[u64]| {
        if !processed[ci] {
            heap.push(qlive_frontier_order_key(
                &touches[ci],
                remaining,
                open,
                parent_weight,
                ci,
            ));
        }
    };

    while order.len() < by_child.len() {
        if heap.is_empty() {
            while seed_pos < summary_order.len() && processed[summary_order[seed_pos]] {
                seed_pos += 1;
            }
            if seed_pos < summary_order.len() {
                push_child(
                    summary_order[seed_pos],
                    &mut heap,
                    &processed,
                    &remaining,
                    &open,
                    &by_child,
                    &parent_weight,
                );
                seed_pos += 1;
            }
        }
        let ci = loop {
            let Some(item) = heap.pop() else {
                break None;
            };
            if processed[item.child] {
                continue;
            }
            let fresh = qlive_frontier_order_key(
                &by_child[item.child],
                &remaining,
                &open,
                &parent_weight,
                item.child,
            );
            if fresh != item {
                stale_pops += 1;
                heap.push(fresh);
                continue;
            }
            break Some(item.child);
        };
        let Some(ci) = ci else {
            continue;
        };

        processed[ci] = true;
        order.push(ci);
        let t = &by_child[ci];
        for &p in &t.parents {
            let pi = p as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active_fields += 1;
                active_weight += parent_weight[pi].max(1) as u128;
                if let Some(kids) = children_of_parent.get(&p) {
                    for &k in kids {
                        push_child(
                            k,
                            &mut heap,
                            &processed,
                            &remaining,
                            &open,
                            &by_child,
                            &parent_weight,
                        );
                    }
                }
            }
        }
        for &p in &t.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if let Some(kids) = children_of_parent.get(&p) {
                for &k in kids {
                    push_child(
                        k,
                        &mut heap,
                        &processed,
                        &remaining,
                        &open,
                        &by_child,
                        &parent_weight,
                    );
                }
            }
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active_fields -= 1;
                    active_weight = active_weight.saturating_sub(parent_weight[pi].max(1) as u128);
                }
                finalized += 1;
            }
        }
        max_active_fields = max_active_fields.max(active_fields);
        max_active_weight = max_active_weight.max(active_weight);
        active_fields_sum += active_fields as u128;
        active_weight_sum += active_weight;
        let done = order.len();
        if done % 4096 == 0 || done == by_child.len() {
            eprintln!(
                "  q preq-direct frontier order blocks_done={}/{} finalized_fields={} active_fields={} active_weight={} max_active_fields={} max_active_weight={} avg_active_fields={:.1} avg_active_weight={:.1} heap={} stale_pops={} elapsed={:.3}s",
                done, by_child.len(), finalized, active_fields, active_weight, max_active_fields, max_active_weight,
                active_fields_sum as f64 / done as f64, active_weight_sum as f64 / done as f64, heap.len(), stale_pops, start.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "q preq-direct frontier order done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_active_weight={} avg_active_weight={:.3} stale_pops={} elapsed={:.3}s",
        finalized,
        max_active_fields,
        active_fields_sum as f64 / order.len().max(1) as f64,
        max_active_weight,
        active_weight_sum as f64 / order.len().max(1) as f64,
        stale_pops,
        start.elapsed().as_secs_f64()
    );
    let mut ordered = Vec::<QLiveChildTouch>::with_capacity(by_child.len());
    for ci in order {
        ordered.push(by_child[ci].clone());
    }
    ordered
}

#[derive(Default, Clone, Copy)]
struct QPreqDirectStats {
    preq_sources: u64,
    raw_q7_candidates: u64,
    acc_inserts: u64,
}

#[derive(Debug, Clone, Copy)]
struct QPreqSpillRun {
    offset_qids: u64,
    count: u32,
}

struct QPreqAccSpillWriter {
    path: std::path::PathBuf,
    file: std::io::BufWriter<std::fs::File>,
    runs: HashMap<FieldId, Vec<QPreqSpillRun>>,
    total_qids: u64,
    spill_events: u64,
    spill_runs: u64,
    spilled_qids: u64,
    encode_buf: Vec<u8>,
}

impl QPreqAccSpillWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        Ok(Self {
            path: path.to_path_buf(),
            file,
            runs: HashMap::new(),
            total_qids: 0,
            spill_events: 0,
            spill_runs: 0,
            spilled_qids: 0,
            encode_buf: Vec::with_capacity(1 << 20),
        })
    }

    fn spill_acc(&mut self, field: FieldId, acc: QidFieldAcc) -> Result<u64> {
        use std::io::Write;
        let qids = acc.finalized_vec();
        if qids.is_empty() {
            return Ok(0);
        }
        let offset = self.total_qids;
        let bytes_len = qids.len().saturating_mul(3);
        self.encode_buf.resize(bytes_len, 0);
        for (i, &qid) in qids.iter().enumerate() {
            let j = i * 3;
            self.encode_buf[j] = (qid & 0xff) as u8;
            self.encode_buf[j + 1] = ((qid >> 8) & 0xff) as u8;
            self.encode_buf[j + 2] = ((qid >> 16) & 0xff) as u8;
        }
        self.file.write_all(&self.encode_buf)?;
        self.encode_buf.clear();
        let count = qids.len() as u32;
        self.runs.entry(field).or_default().push(QPreqSpillRun {
            offset_qids: offset,
            count,
        });
        self.total_qids += count as u64;
        self.spill_events += 1;
        self.spill_runs += 1;
        self.spilled_qids += count as u64;
        Ok(count as u64)
    }

    fn flush(&mut self) -> Result<()> {
        use std::io::Write;
        self.file.flush()?;
        Ok(())
    }
}

fn qpreq_spill_largest_until(
    accs: &mut HashMap<FieldId, QidFieldAcc>,
    total_hot_bytes: &mut usize,
    spill: &mut QPreqAccSpillWriter,
    hot_cap_bytes: usize,
    low_water_bytes: usize,
    protected: &[FieldId],
) -> Result<u64> {
    if *total_hot_bytes <= hot_cap_bytes {
        return Ok(0);
    }

    // Previous implementation selected one victim by scanning all hot accs,
    // then repeated that scan until low-water.  On layer6 this produced
    // ~35k victims in one sweep, so it performed tens of billions of HashMap
    // visits before writing only a few GiB.  Build one ordered victim list per
    // sweep instead: O(n log n + victims), not O(n * victims).
    let protected_set: std::collections::HashSet<FieldId> = protected.iter().copied().collect();
    let mut victims = Vec::<(usize, FieldId)>::with_capacity(accs.len().min(1_000_000));
    for (&field, acc) in accs.iter() {
        if protected_set.contains(&field) {
            continue;
        }
        let b = acc.estimated_bytes();
        if b != 0 {
            victims.push((b, field));
        }
    }
    victims.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut spilled = 0u64;
    let mut spilled_bytes = 0usize;
    let spilled_qids_before = spill.spilled_qids;
    let sweep_start = Instant::now();
    for (bytes, field) in victims {
        if *total_hot_bytes <= low_water_bytes {
            break;
        }
        let Some(acc) = accs.remove(&field) else {
            continue;
        };
        *total_hot_bytes = total_hot_bytes.saturating_sub(bytes);
        spilled_bytes = spilled_bytes.saturating_add(bytes);
        spill.spill_acc(field, acc)?;
        spilled += 1;
    }
    if spilled != 0 {
        let spilled_qids = spill.spilled_qids.saturating_sub(spilled_qids_before);
        eprintln!(
            "    q preq frontier-spill sweep spilled_runs={} spilled_qids={} spilled_hot_bytes={} hot_bytes_after={} elapsed={:.3}s",
            spilled,
            spilled_qids,
            spilled_bytes,
            *total_hot_bytes,
            sweep_start.elapsed().as_secs_f64()
        );
    }
    Ok(spilled)
}

fn qpreq_finalize_field_with_spills(
    field: FieldId,
    acc: Option<QidFieldAcc>,
    spill: &mut QPreqAccSpillWriter,
    qids_tmp: &mut Vec<u32>,
    read_buf: &mut Vec<u8>,
    spool: &mut QLiveSpoolWriter,
) -> Result<u64> {
    let runs = spill.runs.remove(&field).unwrap_or_default();
    if !runs.is_empty() {
        spill.flush()?;
    }
    let spill_file = if runs.is_empty() {
        None
    } else {
        Some(std::fs::File::open(&spill.path)?)
    };
    qids_tmp.clear();
    qpreq_materialize_close_parts(acc, &runs, spill_file.as_ref(), qids_tmp, read_buf)?;
    if qids_tmp.is_empty() {
        return Ok(0);
    }
    let n = qids_tmp.len() as u64;
    spool.write_field(field, qids_tmp)?;
    qids_tmp.clear();
    Ok(n)
}

struct QPreqCloseJob {
    field: FieldId,
    acc: Option<QidFieldAcc>,
    runs: Vec<QPreqSpillRun>,
    hot_bytes: usize,
}

fn qpreq_read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        let n = file.read_at(buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read from qpreq spill file",
            )
            .into());
        }
        offset += n as u64;
        let (_, rest) = buf.split_at_mut(n);
        buf = rest;
    }
    Ok(())
}

fn qpreq_materialize_close_parts(
    acc: Option<QidFieldAcc>,
    runs: &[QPreqSpillRun],
    spill_file: Option<&std::fs::File>,
    qids_tmp: &mut Vec<u32>,
    read_buf: &mut Vec<u8>,
) -> Result<()> {
    qids_tmp.clear();
    if !runs.is_empty() {
        let file = spill_file.expect("spill_file required for non-empty qpreq spill runs");
        for run in runs {
            let bytes_len = run.count as usize * 3;
            read_buf.resize(bytes_len, 0);
            qpreq_read_exact_at(file, read_buf, run.offset_qids * 3)?;
            for i in 0..run.count as usize {
                qids_tmp.push(crate::layer_store::read_u24_le_at(read_buf, i as u64));
            }
        }
    }
    if let Some(acc) = acc {
        let mut hot = acc.finalized_vec();
        qids_tmp.append(&mut hot);
    }
    if !qids_tmp.is_empty() {
        qids_tmp.sort_unstable();
        qids_tmp.dedup();
    }
    Ok(())
}

fn qpreq_drain_close_jobs_parallel(
    pending: &mut Vec<QPreqCloseJob>,
    spill: &mut QPreqAccSpillWriter,
    spool: &mut QLiveSpoolWriter,
) -> Result<u64> {
    if pending.is_empty() {
        return Ok(0);
    }
    let start = Instant::now();
    let jobs = std::mem::take(pending);
    let job_count = jobs.len();
    let run_count: usize = jobs.iter().map(|j| j.runs.len()).sum();
    let has_spills = run_count != 0;
    if has_spills {
        spill.flush()?;
    }
    let spill_file = if has_spills {
        Some(std::sync::Arc::new(std::fs::File::open(&spill.path)?))
    } else {
        None
    };

    let mut results: Vec<(FieldId, Vec<u32>)> = jobs
        .into_par_iter()
        .map(|job| -> Result<(FieldId, Vec<u32>)> {
            let mut qids = Vec::<u32>::new();
            let mut read_buf = Vec::<u8>::new();
            qpreq_materialize_close_parts(
                job.acc,
                &job.runs,
                spill_file.as_deref(),
                &mut qids,
                &mut read_buf,
            )?;
            Ok((job.field, qids))
        })
        .collect::<Result<Vec<_>>>()?;

    results.sort_unstable_by_key(|&(field, _)| field);
    let mut total = 0u64;
    for (field, qids) in results {
        if qids.is_empty() {
            continue;
        }
        total += qids.len() as u64;
        spool.write_field(field, &qids)?;
    }
    eprintln!(
        "    q preq frontier-spill parallel-close fields={} runs={} output_states={} elapsed={:.3}s",
        job_count,
        run_count,
        total,
        start.elapsed().as_secs_f64()
    );
    Ok(total)
}

pub fn external_step_q_preq_frontier_direct(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    output_prefix: &Path,
    work_dir: &Path,
    vec_to_bitset_threshold: usize,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();
    let bit_words = (q_index.states.len() + 63) / 64;

    eprintln!(
        "q preq frontier-direct step blocks={} input_states={} vec_to_bitset_threshold={} bitset_words={} threads={}",
        input.metas.len(), input.total_states, vec_to_bitset_threshold, bit_words, rayon::current_num_threads()
    );

    let (qid_to_preq, preq_entries) = build_preq_direct_table(q_index);
    let (touches_summary, remaining0, preq_counts, original_states, preq_states) =
        qpreq_build_counts_and_child_order_inputs(rgraph, input, &qid_to_preq, &preq_entries)?;
    let touches = qpreq_build_frontier_order(&preq_counts, &touches_summary, &remaining0);
    drop(touches_summary);
    drop(preq_counts);
    let mut remaining = remaining0;

    let spool_path = work_dir.join("qpreq_frontier_direct_spool.u24.bin");
    let _ = std::fs::remove_file(&spool_path);
    let mut spool = QLiveSpoolWriter::create(&spool_path)
        .with_context(|| format!("create {}", spool_path.display()))?;

    let mut accs = HashMap::<FieldId, QidFieldAcc>::with_capacity(100_000);
    let mut qids_tmp = Vec::<u32>::new();
    let mut stats = QPreqDirectStats::default();
    let mut output_states = 0u64;
    let mut finalized_fields = 0u64;

    for (done_idx, touch) in touches.iter().enumerate() {
        let bi = touch.block_idx;
        let child_field = touch.child_field;
        let preqs = build_unique_preqs_for_block(input, &qid_to_preq, bi);
        stats.preq_sources += preqs.len() as u64;

        for &pid in &preqs {
            let entry = &preq_entries[pid as usize];
            if entry.total_q7 == 0 {
                continue;
            }
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let q7s = &entry.q7_by_piece[placed as usize];
                if q7s.is_empty() {
                    continue;
                }
                let parents = rgraph.edges(child_field, placed);
                if parents.is_empty() {
                    continue;
                }
                stats.raw_q7_candidates += q7s.len() as u64 * parents.len() as u64;
                for &f7 in parents {
                    let acc = accs.entry(f7).or_insert_with(QidFieldAcc::new);
                    acc.insert_all(q7s, vec_to_bitset_threshold, bit_words);
                    stats.acc_inserts += q7s.len() as u64;
                }
            }
        }

        for &p in &touch.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                if let Some(acc) = accs.remove(&p) {
                    let n = acc.finalize_into(&mut qids_tmp);
                    spool.write_field(p, &qids_tmp)?;
                    output_states += n;
                    finalized_fields += 1;
                    qids_tmp.clear();
                }
            }
        }

        let done = done_idx + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q preq frontier-direct blocks_done={}/{} finalized_fields={} live_fields={} spooled_states={} preq_sources={} raw_q7_candidates={} elapsed={:.1}s",
                done, touches.len(), finalized_fields, accs.len(), output_states, stats.preq_sources, stats.raw_q7_candidates,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !accs.is_empty() {
        eprintln!(
            "q preq frontier-direct warning: finalizing {} remaining accs at end",
            accs.len()
        );
        let rest = std::mem::take(&mut accs);
        for (field, acc) in rest {
            let n = acc.finalize_into(&mut qids_tmp);
            spool.write_field(field, &qids_tmp)?;
            output_states += n;
            finalized_fields += 1;
            qids_tmp.clear();
        }
    }
    spool.flush()?;
    let metas = std::mem::take(&mut spool.metas);
    drop(spool);

    eprintln!(
        "q preq frontier-direct generation done original_states={} preq_states={} ratio={:.6} finalized_fields={} spooled_states={} raw_q7_candidates={} acc_inserts={} elapsed={:.1}s",
        original_states,
        preq_states,
        preq_states as f64 / original_states.max(1) as f64,
        finalized_fields,
        output_states,
        stats.raw_q7_candidates,
        stats.acc_inserts,
        start.elapsed().as_secs_f64()
    );

    let finished = qlive_assemble_spool(&spool_path, metas, output_prefix)?;
    let _ = std::fs::remove_file(&spool_path);
    println!("q_external_preq_frontier_direct original_states={original_states}");
    println!("q_external_preq_frontier_direct preq_states={preq_states}");
    println!(
        "q_external_preq_frontier_direct preq_ratio={:.6}",
        preq_states as f64 / original_states.max(1) as f64
    );
    println!(
        "q_external_preq_frontier_direct raw_q7_candidates={}",
        stats.raw_q7_candidates
    );
    println!(
        "q_external_preq_frontier_direct acc_inserts={}",
        stats.acc_inserts
    );
    println!("q_external_preq_frontier_direct finalized_fields={finalized_fields}");
    println!("q_external_preq_frontier_direct output_states={finished}");
    println!(
        "q_external_preq_frontier_direct removed_dupes={}",
        stats.raw_q7_candidates.saturating_sub(finished)
    );
    println!(
        "q_external_preq_frontier_direct output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external preq frontier-direct step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

pub fn external_step_q_preq_frontier_spill(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    output_prefix: &Path,
    work_dir: &Path,
    vec_to_bitset_threshold: usize,
    hot_cap_bytes: usize,
    inject_full_q_field: Option<FieldId>,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();
    let bit_words = (q_index.states.len() + 63) / 64;
    let low_water_bytes = hot_cap_bytes.saturating_mul(3) / 4;

    eprintln!(
        "q preq frontier-spill step blocks={} input_states={} vec_to_bitset_threshold={} bitset_words={} hot_cap_bytes={} low_water_bytes={} threads={}",
        input.metas.len(), input.total_states, vec_to_bitset_threshold, bit_words, hot_cap_bytes, low_water_bytes, rayon::current_num_threads()
    );

    let (qid_to_preq, preq_entries) = build_preq_direct_table(q_index);
    let (touches_summary, remaining0, preq_counts, original_states, preq_states) =
        qpreq_build_counts_and_child_order_inputs(rgraph, input, &qid_to_preq, &preq_entries)?;
    let touches = qpreq_build_frontier_order(&preq_counts, &touches_summary, &remaining0);
    drop(touches_summary);
    drop(preq_counts);
    let mut remaining = remaining0;

    let spool_path = work_dir.join("qpreq_frontier_spill_spool.u24.bin");
    let spill_path = work_dir.join("qpreq_frontier_acc_spill.u24.bin");
    let _ = std::fs::remove_file(&spool_path);
    let _ = std::fs::remove_file(&spill_path);
    let mut spool = QLiveSpoolWriter::create(&spool_path)
        .with_context(|| format!("create {}", spool_path.display()))?;
    let mut spill = QPreqAccSpillWriter::create(&spill_path)
        .with_context(|| format!("create {}", spill_path.display()))?;

    let mut accs = HashMap::<FieldId, QidFieldAcc>::with_capacity(200_000);
    let mut total_hot_bytes = 0usize;
    let mut pending_close = Vec::<QPreqCloseJob>::with_capacity(8192);
    let mut pending_close_hot_bytes = 0usize;
    let close_batch_fields = 8192usize;
    let close_batch_hot_bytes = hot_cap_bytes.saturating_div(8).max(512usize << 20);
    let mut stats = QPreqDirectStats::default();
    let mut output_states = 0u64;
    let mut finalized_fields = 0u64;
    let mut spill_sweeps = 0u64;
    let mut max_live_fields = 0usize;
    let mut max_hot_bytes = 0usize;

    for (done_idx, touch) in touches.iter().enumerate() {
        let bi = touch.block_idx;
        let child_field = touch.child_field;
        let preqs = build_unique_preqs_for_block(input, &qid_to_preq, bi);
        stats.preq_sources += preqs.len() as u64;

        for &pid in &preqs {
            let entry = &preq_entries[pid as usize];
            if entry.total_q7 == 0 {
                continue;
            }
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let q7s = &entry.q7_by_piece[placed as usize];
                if q7s.is_empty() {
                    continue;
                }
                let parents = rgraph.edges(child_field, placed);
                if parents.is_empty() {
                    continue;
                }
                stats.raw_q7_candidates += q7s.len() as u64 * parents.len() as u64;
                for &f7 in parents {
                    let acc = accs.entry(f7).or_insert_with(QidFieldAcc::new);
                    let before = acc.estimated_bytes();
                    acc.insert_all(q7s, vec_to_bitset_threshold, bit_words);
                    let after = acc.estimated_bytes();
                    if after >= before {
                        total_hot_bytes = total_hot_bytes.saturating_add(after - before);
                    } else {
                        total_hot_bytes = total_hot_bytes.saturating_sub(before - after);
                    }
                    stats.acc_inserts += q7s.len() as u64;
                }
            }
        }

        for &p in &touch.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                let acc = accs.remove(&p);
                let hot_bytes = acc.as_ref().map(|a| a.estimated_bytes()).unwrap_or(0);
                total_hot_bytes = total_hot_bytes.saturating_sub(hot_bytes);
                let runs = spill.runs.remove(&p).unwrap_or_default();
                if acc.is_some() || !runs.is_empty() {
                    pending_close_hot_bytes = pending_close_hot_bytes.saturating_add(hot_bytes);
                    pending_close.push(QPreqCloseJob {
                        field: p,
                        acc,
                        runs,
                        hot_bytes,
                    });
                    finalized_fields += 1;
                }
            }
        }

        if pending_close.len() >= close_batch_fields
            || pending_close_hot_bytes >= close_batch_hot_bytes
        {
            output_states +=
                qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
            pending_close_hot_bytes = 0;
        }

        if total_hot_bytes > hot_cap_bytes {
            let n = qpreq_spill_largest_until(
                &mut accs,
                &mut total_hot_bytes,
                &mut spill,
                hot_cap_bytes,
                low_water_bytes,
                &touch.parents,
            )?;
            if n != 0 {
                spill_sweeps += 1;
            }
        }

        max_live_fields = max_live_fields.max(accs.len());
        max_hot_bytes = max_hot_bytes.max(total_hot_bytes);

        let done = done_idx + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q preq frontier-spill blocks_done={}/{} finalized_fields={} live_fields={} hot_bytes={} max_hot_bytes={} spooled_states={} preq_sources={} raw_q7_candidates={} spill_events={} spill_runs={} spilled_qids={} spill_sweeps={} active_spilled_fields={} elapsed={:.1}s",
                done, touches.len(), finalized_fields, accs.len(), total_hot_bytes, max_hot_bytes,
                output_states, stats.preq_sources, stats.raw_q7_candidates,
                spill.spill_events, spill.spill_runs, spill.spilled_qids, spill_sweeps, spill.runs.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !pending_close.is_empty() {
        output_states +=
            qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
        pending_close_hot_bytes = 0;
    }
    if !accs.is_empty() || !spill.runs.is_empty() {
        eprintln!("q preq frontier-spill warning: finalizing remaining hot_accs={} spilled_fields={} at end", accs.len(), spill.runs.len());
        let fields: Vec<FieldId> = accs
            .keys()
            .copied()
            .chain(spill.runs.keys().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        for field in fields {
            let acc = accs.remove(&field);
            let hot_bytes = acc.as_ref().map(|a| a.estimated_bytes()).unwrap_or(0);
            total_hot_bytes = total_hot_bytes.saturating_sub(hot_bytes);
            let runs = spill.runs.remove(&field).unwrap_or_default();
            pending_close_hot_bytes = pending_close_hot_bytes.saturating_add(hot_bytes);
            pending_close.push(QPreqCloseJob {
                field,
                acc,
                runs,
                hot_bytes,
            });
            finalized_fields += 1;
            if pending_close.len() >= close_batch_fields
                || pending_close_hot_bytes >= close_batch_hot_bytes
            {
                output_states +=
                    qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
                pending_close_hot_bytes = 0;
            }
        }
        if !pending_close.is_empty() {
            output_states +=
                qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
        }
    }
    if let Some(field) = inject_full_q_field {
        let already_present = spool.metas.iter().any(|&(f, _, _)| f == field);
        if already_present {
            eprintln!(
                "q preq frontier-spill inject_full_q_field={} already exists in normal backward output; appending full-q union block",
                field
            );
        }
        let qids: Vec<u32> = (0..q_index.states.len() as u32).collect();
        let injected = qids.len() as u64;
        spool.write_field(field, &qids)?;
        output_states = output_states.saturating_add(injected);
        finalized_fields = finalized_fields.saturating_add(1);
        eprintln!(
            "q preq frontier-spill injected full-q terminal field={} qstates={} already_present={} spooled_states={} finalized_fields_counted={} elapsed={:.1}s",
            field,
            injected,
            already_present,
            output_states,
            finalized_fields,
            start.elapsed().as_secs_f64()
        );
    }

    spool.flush()?;
    let metas = std::mem::take(&mut spool.metas);
    drop(spool);
    spill.flush()?;

    eprintln!(
        "q preq frontier-spill generation done original_states={} preq_states={} ratio={:.6} finalized_fields={} spooled_states={} raw_q7_candidates={} acc_inserts={} max_live_fields={} max_hot_bytes={} spill_events={} spill_runs={} spilled_qids={} active_spilled_fields_end={} elapsed={:.1}s",
        original_states,
        preq_states,
        preq_states as f64 / original_states.max(1) as f64,
        finalized_fields,
        output_states,
        stats.raw_q7_candidates,
        stats.acc_inserts,
        max_live_fields,
        max_hot_bytes,
        spill.spill_events,
        spill.spill_runs,
        spill.spilled_qids,
        spill.runs.len(),
        start.elapsed().as_secs_f64()
    );

    let finished = qlive_assemble_spool(&spool_path, metas, output_prefix)?;
    let _ = std::fs::remove_file(&spool_path);
    let _ = std::fs::remove_file(&spill_path);
    println!("q_external_preq_frontier_spill original_states={original_states}");
    println!("q_external_preq_frontier_spill preq_states={preq_states}");
    println!(
        "q_external_preq_frontier_spill preq_ratio={:.6}",
        preq_states as f64 / original_states.max(1) as f64
    );
    println!(
        "q_external_preq_frontier_spill raw_q7_candidates={}",
        stats.raw_q7_candidates
    );
    println!(
        "q_external_preq_frontier_spill acc_inserts={}",
        stats.acc_inserts
    );
    println!("q_external_preq_frontier_spill finalized_fields={finalized_fields}");
    println!("q_external_preq_frontier_spill output_states={finished}");
    println!(
        "q_external_preq_frontier_spill removed_dupes={}",
        stats.raw_q7_candidates.saturating_sub(finished)
    );
    println!("q_external_preq_frontier_spill max_live_fields={max_live_fields}");
    println!("q_external_preq_frontier_spill max_hot_bytes={max_hot_bytes}");
    println!(
        "q_external_preq_frontier_spill spill_events={}",
        spill.spill_events
    );
    println!(
        "q_external_preq_frontier_spill spill_runs={}",
        spill.spill_runs
    );
    println!(
        "q_external_preq_frontier_spill spilled_qids={}",
        spill.spilled_qids
    );
    if let Some(field) = inject_full_q_field {
        println!("q_external_preq_frontier_spill injected_full_q_field={field}");
        println!(
            "q_external_preq_frontier_spill injected_full_q_states={}",
            q_index.states.len()
        );
    }
    println!(
        "q_external_preq_frontier_spill output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external preq frontier-spill step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );

    Ok(finished)
}

// -----------------------------------------------------------------------------
// Target-local sharded PreQ frontier-spill path.
//
// This backend treats the input/base child fields as a hypergraph:
//   child field -> set of target/parent fields touched by reverse edges.
// It partitions child fields by target-bucket locality, runs independent
// shard-local generators in parallel, and then field-wise unions shard partial
// outputs.  The point is to remove the single global accumulator bottleneck and
// to reduce shard-local live target sets / spill fan-in.

#[derive(Debug, Clone)]
struct QPreqShardMeta {
    touch: QLiveChildTouch,
    weight: u64,
    buckets: Vec<u32>,
    shard: usize,
    sig: [u64; 4],
}

#[derive(Debug, Default, Clone, Copy)]
struct QPreqShardStats {
    preq_sources: u64,
    raw_q7_candidates: u64,
    acc_inserts: u64,
    output_states: u64,
    finalized_fields: u64,
    max_live_fields: usize,
    max_hot_bytes: usize,
    spill_sweeps: u64,
    spill_events: u64,
    spill_runs: u64,
    spilled_qids: u64,
}

struct QPreqShardResult {
    shard: usize,
    spool_path: std::path::PathBuf,
    metas: Vec<(FieldId, u32, u64)>,
    stats: QPreqShardStats,
}

#[derive(Debug, Clone, Copy)]
pub struct QPreqTargetShardConfig {
    pub vec_to_bitset_threshold: usize,
    pub total_hot_cap_bytes: usize,
    pub num_shards: usize,
    pub target_bucket_size: u32,
    pub refine_passes: usize,
    pub inject_full_q_field: Option<FieldId>,
}

#[inline]
fn qpreq_splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn qpreq_target_buckets(parents: &[FieldId], bucket_size: u32) -> Vec<u32> {
    let bs = bucket_size.max(1);
    let mut b = Vec::<u32>::with_capacity(parents.len());
    for &p in parents {
        b.push(p / bs);
    }
    b.sort_unstable();
    b.dedup();
    b
}

fn qpreq_minhash_buckets(buckets: &[u32]) -> [u64; 4] {
    let seeds = [
        0x243f6a8885a308d3u64,
        0x13198a2e03707344u64,
        0xa4093822299f31d0u64,
        0x082efa98ec4e6c89u64,
    ];
    let mut sig = [u64::MAX; 4];
    for &b in buckets {
        let x = b as u64;
        for i in 0..4 {
            sig[i] = sig[i].min(qpreq_splitmix64(x ^ seeds[i]));
        }
    }
    sig
}

fn qpreq_build_target_local_shards(
    touches: Vec<QLiveChildTouch>,
    preq_counts: &[u32],
    num_shards0: usize,
    bucket_size: u32,
    refine_passes: usize,
) -> Vec<Vec<QLiveChildTouch>> {
    let start = Instant::now();
    let num_shards = num_shards0.max(1).min(touches.len().max(1));
    let bucket_size = bucket_size.max(1);
    eprintln!(
        "q preq target-shard partition start children={} shards={} bucket_size={} refine_passes={}",
        touches.len(),
        num_shards,
        bucket_size,
        refine_passes
    );

    let mut metas = Vec::<QPreqShardMeta>::with_capacity(touches.len());
    for t in touches {
        let preq = preq_counts[t.block_idx] as u64;
        let weight = preq.saturating_mul((t.parents.len() as u64).max(1)).max(1);
        let buckets = qpreq_target_buckets(&t.parents, bucket_size);
        let sig = qpreq_minhash_buckets(&buckets);
        metas.push(QPreqShardMeta {
            touch: t,
            weight,
            buckets,
            shard: 0,
            sig,
        });
    }

    metas.sort_unstable_by(|a, b| {
        (
            a.sig[0],
            a.sig[1],
            a.sig[2],
            a.sig[3],
            a.touch.median_parent,
            a.touch.child_field,
        )
            .cmp(&(
                b.sig[0],
                b.sig[1],
                b.sig[2],
                b.sig[3],
                b.touch.median_parent,
                b.touch.child_field,
            ))
    });

    let total_weight: u128 = metas.iter().map(|m| m.weight as u128).sum();
    let target_weight = (total_weight + num_shards as u128 - 1) / num_shards as u128;
    let soft_max_weight = target_weight.saturating_mul(108) / 100 + 1;
    let mut shard = 0usize;
    let mut cur_w = 0u128;
    for m in metas.iter_mut() {
        if shard + 1 < num_shards && cur_w >= target_weight {
            shard += 1;
            cur_w = 0;
        }
        m.shard = shard;
        cur_w += m.weight as u128;
    }

    let mut shard_weights = vec![0u128; num_shards];
    let mut bucket_loads: Vec<HashMap<u32, u32>> =
        (0..num_shards).map(|_| HashMap::new()).collect();
    for m in &metas {
        shard_weights[m.shard] += m.weight as u128;
        for &b in &m.buckets {
            *bucket_loads[m.shard].entry(b).or_insert(0) += 1;
        }
    }

    let eval_score = |m: &QPreqShardMeta,
                      sidx: usize,
                      shard_weights: &[u128],
                      bucket_loads: &[HashMap<u32, u32>]|
     -> i128 {
        let mut overlap = 0i128;
        let mut new_buckets = 0i128;
        for &b in &m.buckets {
            let c = bucket_loads[sidx].get(&b).copied().unwrap_or(0) as i128;
            overlap += c;
            if c == 0 {
                new_buckets += 1;
            }
        }
        let next_w = shard_weights[sidx].saturating_add(m.weight as u128);
        let overload = next_w.saturating_sub(soft_max_weight) as i128;
        overlap * 4096 - new_buckets * 256 - overload / 4096
    };

    for pass in 0..refine_passes {
        let mut moved = 0usize;
        let mut order: Vec<usize> = (0..metas.len()).collect();
        // Deterministic heavy-first refinement.  Moving heavy children first gives
        // most of the benefit without random non-reproducibility.
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(metas[i].weight));
        for i in order {
            let cur = metas[i].shard;
            let w = metas[i].weight as u128;
            let mut best = cur;
            let mut best_score = eval_score(&metas[i], cur, &shard_weights, &bucket_loads);
            for sidx in 0..num_shards {
                if sidx == cur {
                    continue;
                }
                if shard_weights[sidx].saturating_add(w) > soft_max_weight
                    && shard_weights[cur] <= target_weight
                {
                    continue;
                }
                let sc = eval_score(&metas[i], sidx, &shard_weights, &bucket_loads);
                if sc > best_score {
                    best_score = sc;
                    best = sidx;
                }
            }
            if best != cur {
                shard_weights[cur] = shard_weights[cur].saturating_sub(w);
                shard_weights[best] = shard_weights[best].saturating_add(w);
                for &b in &metas[i].buckets {
                    if let Some(x) = bucket_loads[cur].get_mut(&b) {
                        *x -= 1;
                        if *x == 0 {
                            bucket_loads[cur].remove(&b);
                        }
                    }
                    *bucket_loads[best].entry(b).or_insert(0) += 1;
                }
                metas[i].shard = best;
                moved += 1;
            }
        }
        eprintln!(
            "  q preq target-shard refine pass={}/{} moved={} elapsed={:.1}s",
            pass + 1,
            refine_passes,
            moved,
            start.elapsed().as_secs_f64()
        );
        if moved == 0 {
            break;
        }
    }

    let mut fanin = HashMap::<u32, u32>::new();
    for sidx in 0..num_shards {
        for &b in bucket_loads[sidx].keys() {
            *fanin.entry(b).or_insert(0) += 1;
        }
    }
    let mut fanin_vals: Vec<u32> = fanin.values().copied().collect();
    fanin_vals.sort_unstable();
    let q = |v: &[u32], p: f64| -> u32 {
        if v.is_empty() {
            return 0;
        }
        v[((v.len() - 1) as f64 * p).round() as usize]
    };
    let mut shard_child_counts = vec![0usize; num_shards];
    let mut shard_bucket_counts = vec![0usize; num_shards];
    for sidx in 0..num_shards {
        shard_bucket_counts[sidx] = bucket_loads[sidx].len();
    }
    for m in &metas {
        shard_child_counts[m.shard] += 1;
    }
    let min_w = shard_weights.iter().min().copied().unwrap_or(0);
    let max_w = shard_weights.iter().max().copied().unwrap_or(0);
    eprintln!(
        "q preq target-shard partition stats shards={} total_weight={} target_weight={} weight_min={} weight_max={} imbalance={:.3} bucket_fanin_p50={} p90={} max={} elapsed={:.3}s",
        num_shards, total_weight, target_weight, min_w, max_w,
        max_w as f64 / target_weight.max(1) as f64,
        q(&fanin_vals, 0.50), q(&fanin_vals, 0.90), fanin_vals.last().copied().unwrap_or(0),
        start.elapsed().as_secs_f64()
    );
    for sidx in 0..num_shards {
        eprintln!(
            "  q preq target-shard partition shard={} children={} weight={} target_buckets={}",
            sidx, shard_child_counts[sidx], shard_weights[sidx], shard_bucket_counts[sidx]
        );
    }

    let mut shards: Vec<Vec<QLiveChildTouch>> = (0..num_shards).map(|_| Vec::new()).collect();
    for m in metas {
        shards[m.shard].push(m.touch);
    }
    for (sidx, shard) in shards.iter_mut().enumerate() {
        // Shard-local ordering: use parent interval locality, then heavier parent-degree.
        // The target-local partition already did most of the work; this keeps close
        // lifetime sane without allocating NUM_FIELDS-sized greedy state per shard.
        shard.sort_unstable_by(|a, b| {
            (
                a.median_parent,
                a.min_parent,
                a.max_parent,
                std::cmp::Reverse(a.parents.len()),
                a.child_field,
            )
                .cmp(&(
                    b.median_parent,
                    b.min_parent,
                    b.max_parent,
                    std::cmp::Reverse(b.parents.len()),
                    b.child_field,
                ))
        });
        eprintln!(
            "  q preq target-shard local-order shard={} children={}",
            sidx,
            shard.len()
        );
    }
    shards
}

fn qpreq_run_one_target_shard(
    shard_id: usize,
    touches: Vec<QLiveChildTouch>,
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qid_to_preq: &[u32],
    preq_entries: &[PreqDirectEntry],
    work_dir: &Path,
    vec_to_bitset_threshold: usize,
    hot_cap_bytes: usize,
    bit_words: usize,
) -> Result<QPreqShardResult> {
    use anyhow::Context;
    let start = Instant::now();
    let shard_dir = work_dir.join(format!("target_shard_{:02}", shard_id));
    std::fs::create_dir_all(&shard_dir)?;
    let spool_path = shard_dir.join("partial_spool.u24.bin");
    let spill_path = shard_dir.join("partial_spill.u24.bin");
    let mut spool = QLiveSpoolWriter::create(&spool_path)
        .with_context(|| format!("create {}", spool_path.display()))?;
    let mut spill = QPreqAccSpillWriter::create(&spill_path)
        .with_context(|| format!("create {}", spill_path.display()))?;

    let mut remaining = HashMap::<FieldId, u32>::with_capacity(touches.len().saturating_mul(8));
    for t in &touches {
        for &p in &t.parents {
            *remaining.entry(p).or_insert(0) += 1;
        }
    }

    let shard_hot_cap = hot_cap_bytes.max(256usize << 20);
    let low_water_bytes = shard_hot_cap.saturating_mul(3) / 4;
    let mut accs = HashMap::<FieldId, QidFieldAcc>::with_capacity(remaining.len().min(200_000));
    let mut total_hot_bytes = 0usize;
    let mut pending_close = Vec::<QPreqCloseJob>::with_capacity(4096);
    let mut pending_close_hot_bytes = 0usize;
    let close_batch_fields = 4096usize;
    let close_batch_hot_bytes = shard_hot_cap.saturating_div(8).max(256usize << 20);
    let mut stats = QPreqShardStats::default();

    eprintln!(
        "q preq target-shard shard={} start children={} parent_fields={} hot_cap_bytes={} low_water_bytes={}",
        shard_id, touches.len(), remaining.len(), shard_hot_cap, low_water_bytes
    );

    for (done_idx, touch) in touches.iter().enumerate() {
        let bi = touch.block_idx;
        let child_field = touch.child_field;
        let preqs = build_unique_preqs_for_block(input, qid_to_preq, bi);
        stats.preq_sources += preqs.len() as u64;

        for &pid in &preqs {
            let entry = &preq_entries[pid as usize];
            if entry.total_q7 == 0 {
                continue;
            }
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let q7s = &entry.q7_by_piece[placed as usize];
                if q7s.is_empty() {
                    continue;
                }
                let parents = rgraph.edges(child_field, placed);
                if parents.is_empty() {
                    continue;
                }
                stats.raw_q7_candidates += q7s.len() as u64 * parents.len() as u64;
                for &f7 in parents {
                    // This child belongs to this shard, so every contribution it creates
                    // belongs in this shard's partial output.
                    let acc = accs.entry(f7).or_insert_with(QidFieldAcc::new);
                    let before = acc.estimated_bytes();
                    acc.insert_all(q7s, vec_to_bitset_threshold, bit_words);
                    let after = acc.estimated_bytes();
                    if after >= before {
                        total_hot_bytes = total_hot_bytes.saturating_add(after - before);
                    } else {
                        total_hot_bytes = total_hot_bytes.saturating_sub(before - after);
                    }
                    stats.acc_inserts += q7s.len() as u64;
                }
            }
        }

        for &p in &touch.parents {
            let Some(r) = remaining.get_mut(&p) else {
                continue;
            };
            if *r == 0 {
                continue;
            }
            *r -= 1;
            if *r == 0 {
                let acc = accs.remove(&p);
                let hot_bytes = acc.as_ref().map(|a| a.estimated_bytes()).unwrap_or(0);
                total_hot_bytes = total_hot_bytes.saturating_sub(hot_bytes);
                let runs = spill.runs.remove(&p).unwrap_or_default();
                if acc.is_some() || !runs.is_empty() {
                    pending_close_hot_bytes = pending_close_hot_bytes.saturating_add(hot_bytes);
                    pending_close.push(QPreqCloseJob {
                        field: p,
                        acc,
                        runs,
                        hot_bytes,
                    });
                    stats.finalized_fields += 1;
                }
            }
        }

        if pending_close.len() >= close_batch_fields
            || pending_close_hot_bytes >= close_batch_hot_bytes
        {
            stats.output_states +=
                qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
            pending_close_hot_bytes = 0;
        }

        if total_hot_bytes > shard_hot_cap {
            let n = qpreq_spill_largest_until(
                &mut accs,
                &mut total_hot_bytes,
                &mut spill,
                shard_hot_cap,
                low_water_bytes,
                &touch.parents,
            )?;
            if n != 0 {
                stats.spill_sweeps += 1;
            }
        }

        stats.max_live_fields = stats.max_live_fields.max(accs.len());
        stats.max_hot_bytes = stats.max_hot_bytes.max(total_hot_bytes);

        let done = done_idx + 1;
        if done % 4096 == 0 || done == touches.len() {
            eprintln!(
                "  q preq target-shard shard={} blocks_done={}/{} finalized_fields={} live_fields={} hot_bytes={} max_hot_bytes={} spooled_states={} preq_sources={} raw_q7_candidates={} spill_sweeps={} spill_runs={} spilled_qids={} active_spilled_fields={} elapsed={:.1}s",
                shard_id, done, touches.len(), stats.finalized_fields, accs.len(), total_hot_bytes,
                stats.max_hot_bytes, stats.output_states, stats.preq_sources, stats.raw_q7_candidates,
                stats.spill_sweeps, spill.spill_runs, spill.spilled_qids, spill.runs.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !pending_close.is_empty() {
        stats.output_states +=
            qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
        pending_close_hot_bytes = 0;
    }
    if !accs.is_empty() || !spill.runs.is_empty() {
        eprintln!(
            "q preq target-shard shard={} finalizing remaining hot_accs={} spilled_fields={}",
            shard_id,
            accs.len(),
            spill.runs.len()
        );
        let fields: Vec<FieldId> = accs
            .keys()
            .copied()
            .chain(spill.runs.keys().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        for field in fields {
            let acc = accs.remove(&field);
            let hot_bytes = acc.as_ref().map(|a| a.estimated_bytes()).unwrap_or(0);
            total_hot_bytes = total_hot_bytes.saturating_sub(hot_bytes);
            let runs = spill.runs.remove(&field).unwrap_or_default();
            pending_close_hot_bytes = pending_close_hot_bytes.saturating_add(hot_bytes);
            pending_close.push(QPreqCloseJob {
                field,
                acc,
                runs,
                hot_bytes,
            });
            stats.finalized_fields += 1;
            if pending_close.len() >= close_batch_fields
                || pending_close_hot_bytes >= close_batch_hot_bytes
            {
                stats.output_states +=
                    qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
                pending_close_hot_bytes = 0;
            }
        }
        if !pending_close.is_empty() {
            stats.output_states +=
                qpreq_drain_close_jobs_parallel(&mut pending_close, &mut spill, &mut spool)?;
        }
    }

    spool.flush()?;
    spill.flush()?;
    stats.spill_events = spill.spill_events;
    stats.spill_runs = spill.spill_runs;
    stats.spilled_qids = spill.spilled_qids;
    let metas = std::mem::take(&mut spool.metas);
    drop(spool);
    let _ = std::fs::remove_file(&spill_path);
    eprintln!(
        "q preq target-shard shard={} done partial_fields={} output_states={} preq_sources={} raw_q7_candidates={} acc_inserts={} max_live_fields={} max_hot_bytes={} spill_sweeps={} spill_runs={} spilled_qids={} elapsed={:.3}s",
        shard_id, metas.len(), stats.output_states, stats.preq_sources, stats.raw_q7_candidates,
        stats.acc_inserts, stats.max_live_fields, stats.max_hot_bytes, stats.spill_sweeps,
        stats.spill_runs, stats.spilled_qids, start.elapsed().as_secs_f64()
    );
    Ok(QPreqShardResult {
        shard: shard_id,
        spool_path,
        metas,
        stats,
    })
}

#[derive(Debug, Clone)]
struct QPreqMultiSpoolMeta {
    file_idx: usize,
    field: FieldId,
    count: u32,
    offset: u64,
}

#[derive(Debug)]
struct QPreqMergeGroup {
    field: FieldId,
    start: usize,
    end: usize,
    input_count: u64,
}

#[derive(Debug)]
struct QPreqMergeChunkResult {
    chunk: usize,
    chain_path: std::path::PathBuf,
    metas: Vec<crate::layer_store::FieldChainBlockMeta>,
    total_states: u64,
    merged_duplicate_fields: u64,
    max_fanin: usize,
    fanin_hist: [u64; 18],
    elapsed_sec: f64,
}

fn qpreq_write_final_index(
    index_path: &Path,
    metas: &[crate::layer_store::FieldChainBlockMeta],
    total_states: u64,
) -> Result<()> {
    use anyhow::Context;
    use std::io::{BufWriter, Write};
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(
        std::fs::File::create(index_path)
            .with_context(|| format!("failed to create {}", index_path.display()))?,
    );
    w.write_all(b"FCHNIDX1")?;
    w.write_all(&1u32.to_le_bytes())?;
    w.write_all(&(metas.len() as u32).to_le_bytes())?;
    w.write_all(&total_states.to_le_bytes())?;
    for meta in metas {
        w.write_all(&meta.field_id.to_le_bytes())?;
        w.write_all(&meta.count.to_le_bytes())?;
        w.write_all(&meta.chain_offset.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn qpreq_assemble_multi_spools(results: &[QPreqShardResult], output_prefix: &Path) -> Result<u64> {
    use anyhow::Context;
    use std::io::{BufReader, BufWriter, Write};
    let start = Instant::now();
    let spool_paths: Vec<std::path::PathBuf> =
        results.iter().map(|r| r.spool_path.clone()).collect();
    let mut entries = Vec::<QPreqMultiSpoolMeta>::new();
    for (file_idx, r) in results.iter().enumerate() {
        for &(field, count, offset) in &r.metas {
            entries.push(QPreqMultiSpoolMeta {
                file_idx,
                field,
                count,
                offset,
            });
        }
    }
    entries.sort_unstable_by_key(|m| (m.field, m.file_idx, m.offset));

    let mut groups = Vec::<QPreqMergeGroup>::new();
    let mut gi = 0usize;
    while gi < entries.len() {
        let field = entries[gi].field;
        let start_i = gi;
        let mut input_count = 0u64;
        while gi < entries.len() && entries[gi].field == field {
            input_count += entries[gi].count as u64;
            gi += 1;
        }
        groups.push(QPreqMergeGroup {
            field,
            start: start_i,
            end: gi,
            input_count,
        });
    }

    let total_input_count: u64 = groups.iter().map(|g| g.input_count).sum();
    let merge_workers = rayon::current_num_threads().max(1).min(groups.len().max(1));
    let target_per_worker = (total_input_count + merge_workers as u64 - 1) / merge_workers as u64;
    let mut chunks = Vec::<(usize, usize)>::new();
    let mut cs = 0usize;
    while cs < groups.len() {
        let mut ce = cs;
        let mut acc = 0u64;
        while ce < groups.len()
            && (acc < target_per_worker || ce == cs)
            && chunks.len() + 1 < merge_workers
        {
            acc += groups[ce].input_count;
            ce += 1;
        }
        if chunks.len() + 1 == merge_workers {
            ce = groups.len();
        }
        chunks.push((cs, ce));
        cs = ce;
    }

    eprintln!(
        "q preq target-shard final merge start partial_blocks={} files={} fields={} input_states={} workers={} output={}",
        entries.len(), spool_paths.len(), groups.len(), total_input_count, chunks.len(), output_prefix.display()
    );

    let merge_dir = output_prefix
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".{}.merge_chunks",
            output_prefix
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("qpreq_target_shard")
        ));
    let _ = std::fs::remove_dir_all(&merge_dir);
    std::fs::create_dir_all(&merge_dir)?;

    let chunk_results_vec: Vec<Result<QPreqMergeChunkResult>> = chunks
        .into_par_iter()
        .enumerate()
        .map(|(chunk_id, (g_start, g_end))| -> Result<QPreqMergeChunkResult> {
            let chunk_start = Instant::now();
            let files: Vec<std::fs::File> = spool_paths.iter()
                .map(|p| std::fs::File::open(p).with_context(|| format!("open partial spool {}", p.display())))
                .collect::<Result<_>>()?;
            let chain_path = merge_dir.join(format!("chunk_{:04}.chains.u24.bin", chunk_id));
            let mut writer = BufWriter::new(std::fs::File::create(&chain_path)
                .with_context(|| format!("create {}", chain_path.display()))?);
            let mut metas = Vec::<crate::layer_store::FieldChainBlockMeta>::with_capacity(g_end - g_start);
            let mut buf = Vec::<u8>::new();
            let mut qids = Vec::<u32>::new();
            let mut local_offset = 0u64;
            let mut local_total = 0u64;
            let mut merged_duplicate_fields = 0u64;
            let mut max_fanin = 0usize;
            let mut fanin_hist = [0u64; 18];

            for gidx in g_start..g_end {
                let g = &groups[gidx];
                qids.clear();
                for m in &entries[g.start..g.end] {
                    let bytes_len = m.count as usize * 3;
                    buf.resize(bytes_len, 0);
                    qpreq_read_exact_at(&files[m.file_idx], &mut buf, m.offset * 3)?;
                    for j in 0..m.count as u64 {
                        qids.push(crate::layer_store::read_u24_le_at(&buf, j));
                    }
                }
                let blocks = g.end - g.start;
                max_fanin = max_fanin.max(blocks);
                fanin_hist[blocks.min(fanin_hist.len() - 1)] += 1;
                if blocks > 1 {
                    merged_duplicate_fields += 1;
                    qids.sort_unstable();
                    qids.dedup();
                }
                for &qid in &qids {
                    crate::layer_store::write_u24_le(&mut writer, qid)?;
                }
                metas.push(crate::layer_store::FieldChainBlockMeta {
                    field_id: g.field,
                    count: qids.len() as u32,
                    chain_offset: local_offset,
                });
                local_offset += qids.len() as u64;
                local_total += qids.len() as u64;
            }
            writer.flush()?;
            eprintln!(
                "  q preq target-shard final merge chunk={} fields={} input_states={} output_states={} merged_duplicate_fields={} max_fanin={} elapsed={:.3}s",
                chunk_id, metas.len(), groups[g_start..g_end].iter().map(|g| g.input_count).sum::<u64>(),
                local_total, merged_duplicate_fields, max_fanin, chunk_start.elapsed().as_secs_f64()
            );
            Ok(QPreqMergeChunkResult {
                chunk: chunk_id,
                chain_path,
                metas,
                total_states: local_total,
                merged_duplicate_fields,
                max_fanin,
                fanin_hist,
                elapsed_sec: chunk_start.elapsed().as_secs_f64(),
            })
        })
        .collect();

    let mut chunk_results = Vec::<QPreqMergeChunkResult>::with_capacity(chunk_results_vec.len());
    for r in chunk_results_vec {
        chunk_results.push(r?);
    }
    chunk_results.sort_unstable_by_key(|r| r.chunk);

    let chains_path = output_prefix.with_extension("chains.u24.bin");
    let index_path = output_prefix.with_extension("field_index.bin");
    if let Some(parent) = chains_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut chains_out = BufWriter::new(
        std::fs::File::create(&chains_path)
            .with_context(|| format!("failed to create {}", chains_path.display()))?,
    );
    let mut final_metas =
        Vec::<crate::layer_store::FieldChainBlockMeta>::with_capacity(groups.len());
    let mut global_offset = 0u64;
    let mut finished = 0u64;
    let mut merged_duplicate_fields = 0u64;
    let mut max_fanin = 0usize;
    let mut fanin_hist = [0u64; 18];
    let mut max_chunk_elapsed = 0.0f64;

    let concat_start = Instant::now();
    for cr in &chunk_results {
        max_chunk_elapsed = max_chunk_elapsed.max(cr.elapsed_sec);
        let mut f = BufReader::new(std::fs::File::open(&cr.chain_path)?);
        std::io::copy(&mut f, &mut chains_out)?;
        for meta in &cr.metas {
            final_metas.push(crate::layer_store::FieldChainBlockMeta {
                field_id: meta.field_id,
                count: meta.count,
                chain_offset: global_offset + meta.chain_offset,
            });
        }
        global_offset += cr.total_states;
        finished += cr.total_states;
        merged_duplicate_fields += cr.merged_duplicate_fields;
        max_fanin = max_fanin.max(cr.max_fanin);
        for i in 0..fanin_hist.len() {
            fanin_hist[i] += cr.fanin_hist[i];
        }
    }
    chains_out.flush()?;
    qpreq_write_final_index(&index_path, &final_metas, finished)?;
    eprintln!(
        "q preq target-shard final merge concatenate chunks={} output_states={} elapsed={:.3}s",
        chunk_results.len(),
        finished,
        concat_start.elapsed().as_secs_f64()
    );
    for cr in &chunk_results {
        let _ = std::fs::remove_file(&cr.chain_path);
    }
    let _ = std::fs::remove_dir(&merge_dir);

    eprintln!(
        "q preq target-shard final merge done fields={} merged_duplicate_fields={} max_fanin={} fanin_hist={:?} output_states={} max_chunk_elapsed={:.3}s elapsed={:.3}s",
        final_metas.len(), merged_duplicate_fields, max_fanin, fanin_hist, finished,
        max_chunk_elapsed, start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

pub fn external_step_q_preq_frontier_spill_target_shard(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    output_prefix: &Path,
    work_dir: &Path,
    cfg: QPreqTargetShardConfig,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();
    let bit_words = (q_index.states.len() + 63) / 64;
    let num_shards = cfg.num_shards.max(1);
    let per_shard_hot = cfg
        .total_hot_cap_bytes
        .saturating_div(num_shards)
        .max(512usize << 20);
    eprintln!(
        "q preq target-shard step blocks={} input_states={} vec_to_bitset_threshold={} bitset_words={} total_hot_cap_bytes={} per_shard_hot_cap_bytes={} shards={} bucket_size={} refine_passes={} inject_full_q_field={:?} threads={}",
        input.metas.len(), input.total_states, cfg.vec_to_bitset_threshold, bit_words,
        cfg.total_hot_cap_bytes, per_shard_hot, num_shards, cfg.target_bucket_size,
        cfg.refine_passes, cfg.inject_full_q_field, rayon::current_num_threads()
    );

    let (qid_to_preq, preq_entries) = build_preq_direct_table(q_index);
    let (touches_summary, _remaining0, preq_counts, original_states, preq_states) =
        qpreq_build_counts_and_child_order_inputs(rgraph, input, &qid_to_preq, &preq_entries)?;
    let shards = qpreq_build_target_local_shards(
        touches_summary,
        &preq_counts,
        num_shards,
        cfg.target_bucket_size,
        cfg.refine_passes,
    );

    let results_vec: Vec<Result<QPreqShardResult>> = shards
        .into_par_iter()
        .enumerate()
        .map(|(shard_id, shard_touches)| {
            qpreq_run_one_target_shard(
                shard_id,
                shard_touches,
                rgraph,
                input,
                &qid_to_preq,
                &preq_entries,
                work_dir,
                cfg.vec_to_bitset_threshold,
                per_shard_hot,
                bit_words,
            )
        })
        .collect();
    let mut results = Vec::<QPreqShardResult>::with_capacity(results_vec.len() + 1);
    for r in results_vec {
        results.push(r?);
    }
    results.sort_unstable_by_key(|r| r.shard);

    let mut partial_states_sum = 0u64;
    let mut stats = QPreqShardStats::default();
    for r in &results {
        partial_states_sum += r.stats.output_states;
        stats.preq_sources += r.stats.preq_sources;
        stats.raw_q7_candidates += r.stats.raw_q7_candidates;
        stats.acc_inserts += r.stats.acc_inserts;
        stats.output_states += r.stats.output_states;
        stats.finalized_fields += r.stats.finalized_fields;
        stats.max_live_fields = stats.max_live_fields.max(r.stats.max_live_fields);
        stats.max_hot_bytes = stats.max_hot_bytes.max(r.stats.max_hot_bytes);
        stats.spill_sweeps += r.stats.spill_sweeps;
        stats.spill_events += r.stats.spill_events;
        stats.spill_runs += r.stats.spill_runs;
        stats.spilled_qids += r.stats.spilled_qids;
    }

    if let Some(field) = cfg.inject_full_q_field {
        let inject_path = work_dir.join("target_shard_inject_spool.u24.bin");
        let mut inject = QLiveSpoolWriter::create(&inject_path)?;
        let qids: Vec<u32> = (0..q_index.states.len() as u32).collect();
        inject.write_field(field, &qids)?;
        inject.flush()?;
        let metas = std::mem::take(&mut inject.metas);
        drop(inject);
        partial_states_sum = partial_states_sum.saturating_add(q_index.states.len() as u64);
        eprintln!(
            "q preq target-shard injected full-q terminal field={} qstates={} partial_states_sum={} elapsed={:.1}s",
            field, q_index.states.len(), partial_states_sum, start.elapsed().as_secs_f64()
        );
        results.push(QPreqShardResult {
            shard: usize::MAX,
            spool_path: inject_path,
            metas,
            stats: QPreqShardStats::default(),
        });
    }

    eprintln!(
        "q preq target-shard generation done original_states={} preq_states={} ratio={:.6} partial_states_sum={} raw_q7_candidates={} acc_inserts={} max_shard_live_fields={} max_shard_hot_bytes={} spill_sweeps={} spill_runs={} spilled_qids={} elapsed={:.1}s",
        original_states,
        preq_states,
        preq_states as f64 / original_states.max(1) as f64,
        partial_states_sum,
        stats.raw_q7_candidates,
        stats.acc_inserts,
        stats.max_live_fields,
        stats.max_hot_bytes,
        stats.spill_sweeps,
        stats.spill_runs,
        stats.spilled_qids,
        start.elapsed().as_secs_f64()
    );

    let finished = qpreq_assemble_multi_spools(&results, output_prefix)?;
    for r in &results {
        let _ = std::fs::remove_file(&r.spool_path);
    }

    println!("q_external_preq_frontier_spill_target_shard original_states={original_states}");
    println!("q_external_preq_frontier_spill_target_shard preq_states={preq_states}");
    println!(
        "q_external_preq_frontier_spill_target_shard preq_ratio={:.6}",
        preq_states as f64 / original_states.max(1) as f64
    );
    println!(
        "q_external_preq_frontier_spill_target_shard raw_q7_candidates={}",
        stats.raw_q7_candidates
    );
    println!(
        "q_external_preq_frontier_spill_target_shard acc_inserts={}",
        stats.acc_inserts
    );
    println!("q_external_preq_frontier_spill_target_shard partial_states_sum={partial_states_sum}");
    println!("q_external_preq_frontier_spill_target_shard output_states={finished}");
    println!(
        "q_external_preq_frontier_spill_target_shard removed_dupes={}",
        partial_states_sum.saturating_sub(finished)
    );
    println!(
        "q_external_preq_frontier_spill_target_shard max_shard_live_fields={}",
        stats.max_live_fields
    );
    println!(
        "q_external_preq_frontier_spill_target_shard max_shard_hot_bytes={}",
        stats.max_hot_bytes
    );
    println!(
        "q_external_preq_frontier_spill_target_shard spill_sweeps={}",
        stats.spill_sweeps
    );
    println!(
        "q_external_preq_frontier_spill_target_shard spill_events={}",
        stats.spill_events
    );
    println!(
        "q_external_preq_frontier_spill_target_shard spill_runs={}",
        stats.spill_runs
    );
    println!(
        "q_external_preq_frontier_spill_target_shard spilled_qids={}",
        stats.spilled_qids
    );
    if let Some(field) = cfg.inject_full_q_field {
        println!("q_external_preq_frontier_spill_target_shard injected_full_q_field={field}");
        println!(
            "q_external_preq_frontier_spill_target_shard injected_full_q_states={}",
            q_index.states.len()
        );
    }
    println!(
        "q_external_preq_frontier_spill_target_shard output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external preq frontier-spill-target-shard step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QLiveGreedyHeapItem {
    score: i128,
    child: usize,
}

impl Ord for QLiveGreedyHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.child.cmp(&self.child))
    }
}
impl PartialOrd for QLiveGreedyHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn qlive_greedy_score(touch: &QLiveChildTouch, remaining: &[u32], open: &[u8]) -> i128 {
    let mut close = 0i128;
    let mut reuse = 0i128;
    let mut new_open = 0i128;
    for &p in &touch.parents {
        let pi = p as usize;
        if remaining[pi] == 1 {
            close += 1;
        }
        if open[pi] != 0 {
            reuse += 1;
        } else {
            new_open += 1;
        }
    }
    close * 1_000_000 + reuse * 1_000 - new_open
}

fn qlive_weighted_greedy_score(
    touch: &QLiveChildTouch,
    remaining: &[u32],
    open: &[u8],
    parent_weight: &[u64],
) -> i128 {
    let mut close = 0i128;
    let mut reuse = 0i128;
    let mut new_open = 0i128;
    for &p in &touch.parents {
        let pi = p as usize;
        let w = parent_weight[pi].max(1) as i128;
        if remaining[pi] == 1 {
            close += w;
        }
        if open[pi] != 0 {
            reuse += w;
        } else {
            new_open += w;
        }
    }
    // Lexicographic-like weighted objective: close heavy open parents first,
    // then keep using already-open heavy parents, then avoid opening heavy parents.
    close * 1_000_000_000_000i128 + reuse * 1_000_000i128 - new_open
}

/// Dry-run scheduler for the lifetime idea.  It does not generate qclasses or
/// write any layer files.  It only uses the child-field -> parent-field graph,
/// then compares a cheap interval order with a close/reuse/new-open greedy order.
pub fn qlive_greedy_order_stats(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qgraph: &QRevClassGraph,
) -> Result<()> {
    use std::collections::BinaryHeap;
    let start = Instant::now();
    let (ordered_by_summary, remaining0) =
        qlive_build_child_order_and_counts(rgraph, input, qgraph);
    let mut by_child = ordered_by_summary.clone();
    by_child.sort_unstable_by_key(|t| t.block_idx);

    // Build parent -> source children adjacency.  Store it only for the active parent fields.
    let mut children_of_parent: HashMap<FieldId, Vec<usize>> =
        HashMap::with_capacity(remaining0.iter().filter(|&&c| c != 0).count());
    for (ci, t) in by_child.iter().enumerate() {
        for &p in &t.parents {
            children_of_parent.entry(p).or_default().push(ci);
        }
    }

    fn simulate_order(
        name: &str,
        order: &[usize],
        touches: &[QLiveChildTouch],
        remaining0: &[u32],
    ) {
        let mut remaining = remaining0.to_vec();
        let mut open = vec![0u8; crate::graph::NUM_FIELDS];
        let mut active: u64 = 0;
        let mut max_active: u64 = 0;
        let mut active_sum: u128 = 0;
        let mut finalized: u64 = 0;
        let st = Instant::now();
        for (i, &ci) in order.iter().enumerate() {
            let t = &touches[ci];
            for &p in &t.parents {
                let pi = p as usize;
                if open[pi] == 0 {
                    open[pi] = 1;
                    active += 1;
                }
            }
            for &p in &t.parents {
                let pi = p as usize;
                if remaining[pi] == 0 {
                    continue;
                }
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    if open[pi] != 0 {
                        open[pi] = 0;
                        active -= 1;
                    }
                    finalized += 1;
                }
            }
            max_active = max_active.max(active);
            active_sum += active as u128;
            let done = i + 1;
            if done % 1024 == 0 || done == order.len() {
                eprintln!(
                    "  q live order {name} blocks_done={}/{} finalized_fields={} active_fields={} max_active={} avg_active={:.1} elapsed={:.3}s",
                    done,
                    order.len(),
                    finalized,
                    active,
                    max_active,
                    active_sum as f64 / done as f64,
                    st.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "q live order {name} done finalized_fields={} max_active_fields={} avg_active_fields={:.3} elapsed={:.3}s",
            finalized,
            max_active,
            active_sum as f64 / order.len().max(1) as f64,
            st.elapsed().as_secs_f64()
        );
    }

    // Baseline: current median/min/max summary order.
    let summary_order: Vec<usize> = ordered_by_summary.iter().map(|t| t.block_idx).collect();
    simulate_order("summary", &summary_order, &by_child, &remaining0);

    // Greedy dry run: prefer children that close many currently-open parents,
    // then reuse open parents, then avoid opening new parents.
    let mut remaining = remaining0.clone();
    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut processed = vec![false; by_child.len()];
    let mut heap = BinaryHeap::<QLiveGreedyHeapItem>::new();
    let seed_order: Vec<usize> = summary_order.clone();
    let mut seed_pos = 0usize;
    let mut order = Vec::<usize>::with_capacity(by_child.len());

    let push_child = |ci: usize,
                      heap: &mut BinaryHeap<QLiveGreedyHeapItem>,
                      processed: &[bool],
                      remaining: &[u32],
                      open: &[u8],
                      touches: &[QLiveChildTouch]| {
        if !processed[ci] {
            let score = qlive_greedy_score(&touches[ci], remaining, open);
            heap.push(QLiveGreedyHeapItem { score, child: ci });
        }
    };

    let st = Instant::now();
    let mut active: u64 = 0;
    let mut max_active: u64 = 0;
    let mut active_sum: u128 = 0;
    let mut finalized: u64 = 0;
    let mut stale_pops: u64 = 0;

    while order.len() < by_child.len() {
        if heap.is_empty() {
            while seed_pos < seed_order.len() && processed[seed_order[seed_pos]] {
                seed_pos += 1;
            }
            if seed_pos < seed_order.len() {
                push_child(
                    seed_order[seed_pos],
                    &mut heap,
                    &processed,
                    &remaining,
                    &open,
                    &by_child,
                );
                seed_pos += 1;
            }
        }

        let ci = loop {
            let Some(item) = heap.pop() else {
                break None;
            };
            if processed[item.child] {
                continue;
            }
            let s2 = qlive_greedy_score(&by_child[item.child], &remaining, &open);
            if s2 != item.score {
                stale_pops += 1;
                heap.push(QLiveGreedyHeapItem {
                    score: s2,
                    child: item.child,
                });
                continue;
            }
            break Some(item.child);
        };
        let Some(ci) = ci else {
            continue;
        };

        processed[ci] = true;
        order.push(ci);
        let t = &by_child[ci];

        for &p in &t.parents {
            let pi = p as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active += 1;
                if let Some(kids) = children_of_parent.get(&p) {
                    for &k in kids {
                        push_child(k, &mut heap, &processed, &remaining, &open, &by_child);
                    }
                }
            }
        }
        for &p in &t.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if let Some(kids) = children_of_parent.get(&p) {
                for &k in kids {
                    push_child(k, &mut heap, &processed, &remaining, &open, &by_child);
                }
            }
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active -= 1;
                }
                finalized += 1;
            }
        }
        max_active = max_active.max(active);
        active_sum += active as u128;

        let done = order.len();
        if done % 1024 == 0 || done == by_child.len() {
            eprintln!(
                "  q live order greedy blocks_done={}/{} finalized_fields={} active_fields={} max_active={} avg_active={:.1} heap={} stale_pops={} elapsed={:.3}s",
                done,
                by_child.len(),
                finalized,
                active,
                max_active,
                active_sum as f64 / done as f64,
                heap.len(),
                stale_pops,
                st.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "q live order greedy done finalized_fields={} max_active_fields={} avg_active_fields={:.3} stale_pops={} elapsed={:.3}s total_elapsed={:.3}s",
        finalized,
        max_active,
        active_sum as f64 / order.len().max(1) as f64,
        stale_pops,
        st.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );

    // Weight parent fields by a cheap proxy for RAM cost.  A parent touched by
    // a child with many qchains is expected to receive more bag_tail classes.
    let mut parent_weight = vec![0u64; crate::graph::NUM_FIELDS];
    for t in &by_child {
        let w = input.metas[t.block_idx].count as u64;
        for &p in &t.parents {
            parent_weight[p as usize] = parent_weight[p as usize].saturating_add(w);
        }
    }

    fn simulate_order_weighted(
        name: &str,
        order: &[usize],
        touches: &[QLiveChildTouch],
        remaining0: &[u32],
        parent_weight: &[u64],
    ) {
        let mut remaining = remaining0.to_vec();
        let mut open = vec![0u8; crate::graph::NUM_FIELDS];
        let mut active_fields: u64 = 0;
        let mut active_weight: u128 = 0;
        let mut max_active_fields: u64 = 0;
        let mut max_active_weight: u128 = 0;
        let mut active_fields_sum: u128 = 0;
        let mut active_weight_sum: u128 = 0;
        let mut finalized: u64 = 0;
        let st = Instant::now();
        for (i, &ci) in order.iter().enumerate() {
            let t = &touches[ci];
            for &p in &t.parents {
                let pi = p as usize;
                if open[pi] == 0 {
                    open[pi] = 1;
                    active_fields += 1;
                    active_weight += parent_weight[pi].max(1) as u128;
                }
            }
            for &p in &t.parents {
                let pi = p as usize;
                if remaining[pi] == 0 {
                    continue;
                }
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    if open[pi] != 0 {
                        open[pi] = 0;
                        active_fields -= 1;
                        active_weight =
                            active_weight.saturating_sub(parent_weight[pi].max(1) as u128);
                    }
                    finalized += 1;
                }
            }
            max_active_fields = max_active_fields.max(active_fields);
            max_active_weight = max_active_weight.max(active_weight);
            active_fields_sum += active_fields as u128;
            active_weight_sum += active_weight;
            let done = i + 1;
            if done % 1024 == 0 || done == order.len() {
                eprintln!(
                    "  q live order {name} blocks_done={}/{} finalized_fields={} active_fields={} active_weight={} max_active_fields={} max_active_weight={} avg_active_fields={:.1} avg_active_weight={:.1} elapsed={:.3}s",
                    done,
                    order.len(),
                    finalized,
                    active_fields,
                    active_weight,
                    max_active_fields,
                    max_active_weight,
                    active_fields_sum as f64 / done as f64,
                    active_weight_sum as f64 / done as f64,
                    st.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "q live order {name} done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_active_weight={} avg_active_weight={:.3} elapsed={:.3}s",
            finalized,
            max_active_fields,
            active_fields_sum as f64 / order.len().max(1) as f64,
            max_active_weight,
            active_weight_sum as f64 / order.len().max(1) as f64,
            st.elapsed().as_secs_f64()
        );
    }

    simulate_order_weighted(
        "summary-weight",
        &summary_order,
        &by_child,
        &remaining0,
        &parent_weight,
    );
    simulate_order_weighted(
        "greedy-weight",
        &order,
        &by_child,
        &remaining0,
        &parent_weight,
    );

    // Weighted frontier greedy.  Same frontier mechanism, but score uses the
    // parent_weight proxy instead of pure field counts.
    let mut remaining = remaining0.clone();
    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut processed = vec![false; by_child.len()];
    let mut heap = BinaryHeap::<QLiveGreedyHeapItem>::new();
    let seed_order: Vec<usize> = summary_order.clone();
    let mut seed_pos = 0usize;
    let mut order_w = Vec::<usize>::with_capacity(by_child.len());

    let push_child_w = |ci: usize,
                        heap: &mut BinaryHeap<QLiveGreedyHeapItem>,
                        processed: &[bool],
                        remaining: &[u32],
                        open: &[u8],
                        touches: &[QLiveChildTouch],
                        parent_weight: &[u64]| {
        if !processed[ci] {
            let score = qlive_weighted_greedy_score(&touches[ci], remaining, open, parent_weight);
            heap.push(QLiveGreedyHeapItem { score, child: ci });
        }
    };

    let st = Instant::now();
    let mut active_fields: u64 = 0;
    let mut active_weight: u128 = 0;
    let mut max_active_fields: u64 = 0;
    let mut max_active_weight: u128 = 0;
    let mut active_fields_sum: u128 = 0;
    let mut active_weight_sum: u128 = 0;
    let mut finalized: u64 = 0;
    let mut stale_pops: u64 = 0;

    while order_w.len() < by_child.len() {
        if heap.is_empty() {
            while seed_pos < seed_order.len() && processed[seed_order[seed_pos]] {
                seed_pos += 1;
            }
            if seed_pos < seed_order.len() {
                push_child_w(
                    seed_order[seed_pos],
                    &mut heap,
                    &processed,
                    &remaining,
                    &open,
                    &by_child,
                    &parent_weight,
                );
                seed_pos += 1;
            }
        }

        let ci = loop {
            let Some(item) = heap.pop() else {
                break None;
            };
            if processed[item.child] {
                continue;
            }
            let s2 = qlive_weighted_greedy_score(
                &by_child[item.child],
                &remaining,
                &open,
                &parent_weight,
            );
            if s2 != item.score {
                stale_pops += 1;
                heap.push(QLiveGreedyHeapItem {
                    score: s2,
                    child: item.child,
                });
                continue;
            }
            break Some(item.child);
        };
        let Some(ci) = ci else {
            continue;
        };

        processed[ci] = true;
        order_w.push(ci);
        let t = &by_child[ci];
        for &p in &t.parents {
            let pi = p as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active_fields += 1;
                active_weight += parent_weight[pi].max(1) as u128;
                if let Some(kids) = children_of_parent.get(&p) {
                    for &k in kids {
                        push_child_w(
                            k,
                            &mut heap,
                            &processed,
                            &remaining,
                            &open,
                            &by_child,
                            &parent_weight,
                        );
                    }
                }
            }
        }
        for &p in &t.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if let Some(kids) = children_of_parent.get(&p) {
                for &k in kids {
                    push_child_w(
                        k,
                        &mut heap,
                        &processed,
                        &remaining,
                        &open,
                        &by_child,
                        &parent_weight,
                    );
                }
            }
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active_fields -= 1;
                    active_weight = active_weight.saturating_sub(parent_weight[pi].max(1) as u128);
                }
                finalized += 1;
            }
        }
        max_active_fields = max_active_fields.max(active_fields);
        max_active_weight = max_active_weight.max(active_weight);
        active_fields_sum += active_fields as u128;
        active_weight_sum += active_weight;
        let done = order_w.len();
        if done % 1024 == 0 || done == by_child.len() {
            eprintln!(
                "  q live order weighted-greedy blocks_done={}/{} finalized_fields={} active_fields={} active_weight={} max_active_fields={} max_active_weight={} avg_active_fields={:.1} avg_active_weight={:.1} heap={} stale_pops={} elapsed={:.3}s",
                done,
                by_child.len(),
                finalized,
                active_fields,
                active_weight,
                max_active_fields,
                max_active_weight,
                active_fields_sum as f64 / done as f64,
                active_weight_sum as f64 / done as f64,
                heap.len(),
                stale_pops,
                st.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "q live order weighted-greedy done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_active_weight={} avg_active_weight={:.3} stale_pops={} elapsed={:.3}s total_elapsed={:.3}s",
        finalized,
        max_active_fields,
        active_fields_sum as f64 / order_w.len().max(1) as f64,
        max_active_weight,
        active_weight_sum as f64 / order_w.len().max(1) as f64,
        stale_pops,
        st.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );

    // Direct frontier-delta greedy.  This evaluates the objective described by
    // the user: after choosing child C, the live frontier changes by
    //     delta = newly_loaded_weight(C) - finished_weight(C).
    // The heap key minimizes delta, then maximizes finish/reuse, then minimizes
    // new-open weight.  This is still a heuristic because candidate children are
    // restricted to the open frontier plus the summary seed order, but the score
    // directly targets peak active RAM proxy instead of indirectly rewarding close.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct QLiveFrontierHeapItem {
        neg_delta: i128,
        finish: i128,
        reuse: i128,
        neg_new_open: i128,
        neg_parent_count: i128,
        child: usize,
    }
    impl Ord for QLiveFrontierHeapItem {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            (
                self.neg_delta,
                self.finish,
                self.reuse,
                self.neg_new_open,
                self.neg_parent_count,
            )
                .cmp(&(
                    other.neg_delta,
                    other.finish,
                    other.reuse,
                    other.neg_new_open,
                    other.neg_parent_count,
                ))
                .then_with(|| other.child.cmp(&self.child))
        }
    }
    impl PartialOrd for QLiveFrontierHeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    fn qlive_frontier_key(
        touch: &QLiveChildTouch,
        remaining: &[u32],
        open: &[u8],
        parent_weight: &[u64],
    ) -> QLiveFrontierHeapItem {
        let mut finish = 0i128;
        let mut reuse = 0i128;
        let mut new_open = 0i128;
        for &p in &touch.parents {
            let pi = p as usize;
            let w = parent_weight[pi].max(1) as i128;
            if remaining[pi] == 1 {
                finish += w;
            }
            if open[pi] != 0 {
                reuse += w;
            } else {
                new_open += w;
            }
        }
        let delta = new_open - finish;
        QLiveFrontierHeapItem {
            neg_delta: -delta,
            finish,
            reuse,
            neg_new_open: -new_open,
            neg_parent_count: -(touch.parents.len() as i128),
            child: touch.block_idx, // overwritten by caller before push
        }
    }

    let mut remaining = remaining0.clone();
    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut processed = vec![false; by_child.len()];
    let mut heap = BinaryHeap::<QLiveFrontierHeapItem>::new();
    let seed_order: Vec<usize> = summary_order.clone();
    let mut seed_pos = 0usize;
    let mut order_f = Vec::<usize>::with_capacity(by_child.len());

    let push_child_f = |ci: usize,
                        heap: &mut BinaryHeap<QLiveFrontierHeapItem>,
                        processed: &[bool],
                        remaining: &[u32],
                        open: &[u8],
                        touches: &[QLiveChildTouch],
                        parent_weight: &[u64]| {
        if !processed[ci] {
            let mut item = qlive_frontier_key(&touches[ci], remaining, open, parent_weight);
            item.child = ci;
            heap.push(item);
        }
    };

    let st = Instant::now();
    let mut active_fields: u64 = 0;
    let mut active_weight: u128 = 0;
    let mut max_active_fields: u64 = 0;
    let mut max_active_weight: u128 = 0;
    let mut active_fields_sum: u128 = 0;
    let mut active_weight_sum: u128 = 0;
    let mut finalized: u64 = 0;
    let mut stale_pops: u64 = 0;

    while order_f.len() < by_child.len() {
        if heap.is_empty() {
            while seed_pos < seed_order.len() && processed[seed_order[seed_pos]] {
                seed_pos += 1;
            }
            if seed_pos < seed_order.len() {
                push_child_f(
                    seed_order[seed_pos],
                    &mut heap,
                    &processed,
                    &remaining,
                    &open,
                    &by_child,
                    &parent_weight,
                );
                seed_pos += 1;
            }
        }

        let ci = loop {
            let Some(item) = heap.pop() else {
                break None;
            };
            if processed[item.child] {
                continue;
            }
            let mut fresh =
                qlive_frontier_key(&by_child[item.child], &remaining, &open, &parent_weight);
            fresh.child = item.child;
            if fresh != item {
                stale_pops += 1;
                heap.push(fresh);
                continue;
            }
            break Some(item.child);
        };
        let Some(ci) = ci else {
            continue;
        };

        processed[ci] = true;
        order_f.push(ci);
        let t = &by_child[ci];
        for &p in &t.parents {
            let pi = p as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active_fields += 1;
                active_weight += parent_weight[pi].max(1) as u128;
                if let Some(kids) = children_of_parent.get(&p) {
                    for &k in kids {
                        push_child_f(
                            k,
                            &mut heap,
                            &processed,
                            &remaining,
                            &open,
                            &by_child,
                            &parent_weight,
                        );
                    }
                }
            }
        }
        for &p in &t.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if let Some(kids) = children_of_parent.get(&p) {
                for &k in kids {
                    push_child_f(
                        k,
                        &mut heap,
                        &processed,
                        &remaining,
                        &open,
                        &by_child,
                        &parent_weight,
                    );
                }
            }
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active_fields -= 1;
                    active_weight = active_weight.saturating_sub(parent_weight[pi].max(1) as u128);
                }
                finalized += 1;
            }
        }
        max_active_fields = max_active_fields.max(active_fields);
        max_active_weight = max_active_weight.max(active_weight);
        active_fields_sum += active_fields as u128;
        active_weight_sum += active_weight;
        let done = order_f.len();
        if done % 1024 == 0 || done == by_child.len() {
            eprintln!(
                "  q live order frontier-delta blocks_done={}/{} finalized_fields={} active_fields={} active_weight={} max_active_fields={} max_active_weight={} avg_active_fields={:.1} avg_active_weight={:.1} heap={} stale_pops={} elapsed={:.3}s",
                done,
                by_child.len(),
                finalized,
                active_fields,
                active_weight,
                max_active_fields,
                max_active_weight,
                active_fields_sum as f64 / done as f64,
                active_weight_sum as f64 / done as f64,
                heap.len(),
                stale_pops,
                st.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "q live order frontier-delta done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_active_weight={} avg_active_weight={:.3} stale_pops={} elapsed={:.3}s total_elapsed={:.3}s",
        finalized,
        max_active_fields,
        active_fields_sum as f64 / order_f.len().max(1) as f64,
        max_active_weight,
        active_weight_sum as f64 / order_f.len().max(1) as f64,
        stale_pops,
        st.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QLiveFrontierOrderItem {
    neg_delta: i128,
    finish: i128,
    reuse: i128,
    neg_new_open: i128,
    neg_parent_count: i128,
    child: usize,
}
impl Ord for QLiveFrontierOrderItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.neg_delta,
            self.finish,
            self.reuse,
            self.neg_new_open,
            self.neg_parent_count,
        )
            .cmp(&(
                other.neg_delta,
                other.finish,
                other.reuse,
                other.neg_new_open,
                other.neg_parent_count,
            ))
            .then_with(|| other.child.cmp(&self.child))
    }
}
impl PartialOrd for QLiveFrontierOrderItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn qlive_parent_weight(input: &FieldChainLayer, touches: &[QLiveChildTouch]) -> Vec<u64> {
    let mut parent_weight = vec![0u64; crate::graph::NUM_FIELDS];
    for t in touches {
        let w = input.metas[t.block_idx].count as u64;
        for &p in &t.parents {
            parent_weight[p as usize] = parent_weight[p as usize].saturating_add(w.max(1));
        }
    }
    parent_weight
}

fn qlive_children_of_parent(touches: &[QLiveChildTouch]) -> HashMap<FieldId, Vec<usize>> {
    let mut children_of_parent = HashMap::<FieldId, Vec<usize>>::with_capacity(1_500_000);
    for (ci, t) in touches.iter().enumerate() {
        for &p in &t.parents {
            children_of_parent.entry(p).or_default().push(ci);
        }
    }
    children_of_parent
}

fn qlive_frontier_order_key(
    touch: &QLiveChildTouch,
    remaining: &[u32],
    open: &[u8],
    parent_weight: &[u64],
    child: usize,
) -> QLiveFrontierOrderItem {
    let mut finish = 0i128;
    let mut reuse = 0i128;
    let mut new_open = 0i128;
    for &p in &touch.parents {
        let pi = p as usize;
        let w = parent_weight[pi].max(1) as i128;
        if remaining[pi] == 1 {
            finish += w;
        }
        if open[pi] != 0 {
            reuse += w;
        } else {
            new_open += w;
        }
    }
    let delta = new_open - finish;
    QLiveFrontierOrderItem {
        neg_delta: -delta,
        finish,
        reuse,
        neg_new_open: -new_open,
        neg_parent_count: -(touch.parents.len() as i128),
        child,
    }
}

fn qlive_build_frontier_order(
    input: &FieldChainLayer,
    touches_summary_order: &[QLiveChildTouch],
    remaining0: &[u32],
) -> Vec<QLiveChildTouch> {
    let start = Instant::now();
    let mut by_child = touches_summary_order.to_vec();
    by_child.sort_unstable_by_key(|t| t.block_idx);
    let summary_order: Vec<usize> = touches_summary_order.iter().map(|t| t.block_idx).collect();
    let parent_weight = qlive_parent_weight(input, &by_child);
    let children_of_parent = qlive_children_of_parent(&by_child);

    let mut remaining = remaining0.to_vec();
    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut processed = vec![false; by_child.len()];
    let mut heap = std::collections::BinaryHeap::<QLiveFrontierOrderItem>::new();
    let mut seed_pos = 0usize;
    let mut order = Vec::<usize>::with_capacity(by_child.len());
    let mut stale_pops = 0u64;
    let mut active_fields = 0u64;
    let mut active_weight = 0u128;
    let mut max_active_fields = 0u64;
    let mut max_active_weight = 0u128;
    let mut active_fields_sum = 0u128;
    let mut active_weight_sum = 0u128;
    let mut finalized = 0u64;

    let push_child = |ci: usize,
                      heap: &mut std::collections::BinaryHeap<QLiveFrontierOrderItem>,
                      processed: &[bool],
                      remaining: &[u32],
                      open: &[u8],
                      touches: &[QLiveChildTouch],
                      parent_weight: &[u64]| {
        if !processed[ci] {
            heap.push(qlive_frontier_order_key(
                &touches[ci],
                remaining,
                open,
                parent_weight,
                ci,
            ));
        }
    };

    while order.len() < by_child.len() {
        if heap.is_empty() {
            while seed_pos < summary_order.len() && processed[summary_order[seed_pos]] {
                seed_pos += 1;
            }
            if seed_pos < summary_order.len() {
                push_child(
                    summary_order[seed_pos],
                    &mut heap,
                    &processed,
                    &remaining,
                    &open,
                    &by_child,
                    &parent_weight,
                );
                seed_pos += 1;
            }
        }
        let ci = loop {
            let Some(item) = heap.pop() else {
                break None;
            };
            if processed[item.child] {
                continue;
            }
            let fresh = qlive_frontier_order_key(
                &by_child[item.child],
                &remaining,
                &open,
                &parent_weight,
                item.child,
            );
            if fresh != item {
                stale_pops += 1;
                heap.push(fresh);
                continue;
            }
            break Some(item.child);
        };
        let Some(ci) = ci else {
            continue;
        };

        processed[ci] = true;
        order.push(ci);
        let t = &by_child[ci];
        for &p in &t.parents {
            let pi = p as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active_fields += 1;
                active_weight += parent_weight[pi].max(1) as u128;
                if let Some(kids) = children_of_parent.get(&p) {
                    for &k in kids {
                        push_child(
                            k,
                            &mut heap,
                            &processed,
                            &remaining,
                            &open,
                            &by_child,
                            &parent_weight,
                        );
                    }
                }
            }
        }
        for &p in &t.parents {
            let pi = p as usize;
            if remaining[pi] == 0 {
                continue;
            }
            remaining[pi] -= 1;
            if let Some(kids) = children_of_parent.get(&p) {
                for &k in kids {
                    push_child(
                        k,
                        &mut heap,
                        &processed,
                        &remaining,
                        &open,
                        &by_child,
                        &parent_weight,
                    );
                }
            }
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active_fields -= 1;
                    active_weight = active_weight.saturating_sub(parent_weight[pi].max(1) as u128);
                }
                finalized += 1;
            }
        }
        max_active_fields = max_active_fields.max(active_fields);
        max_active_weight = max_active_weight.max(active_weight);
        active_fields_sum += active_fields as u128;
        active_weight_sum += active_weight;
        let done = order.len();
        if done % 4096 == 0 || done == by_child.len() {
            eprintln!(
                "  q live frontier order blocks_done={}/{} finalized_fields={} active_fields={} active_weight={} max_active_fields={} max_active_weight={} avg_active_fields={:.1} avg_active_weight={:.1} heap={} stale_pops={} elapsed={:.3}s",
                done,
                by_child.len(),
                finalized,
                active_fields,
                active_weight,
                max_active_fields,
                max_active_weight,
                active_fields_sum as f64 / done as f64,
                active_weight_sum as f64 / done as f64,
                heap.len(),
                stale_pops,
                start.elapsed().as_secs_f64()
            );
        }
    }

    eprintln!(
        "q live frontier order done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_active_weight={} avg_active_weight={:.3} stale_pops={} elapsed={:.3}s",
        finalized,
        max_active_fields,
        active_fields_sum as f64 / order.len().max(1) as f64,
        max_active_weight,
        active_weight_sum as f64 / order.len().max(1) as f64,
        stale_pops,
        start.elapsed().as_secs_f64()
    );

    let mut ordered = Vec::<QLiveChildTouch>::with_capacity(by_child.len());
    for ci in order {
        ordered.push(by_child[ci].clone());
    }
    ordered
}

#[derive(Debug, Clone, Copy)]
struct QLiveSpillRun {
    offset_records: u64,
    count: u32,
}

struct QLiveSpillWriter {
    path: std::path::PathBuf,
    file: std::io::BufWriter<std::fs::File>,
    total_records: u64,
    runs: HashMap<FieldId, Vec<QLiveSpillRun>>,
    run_count: u64,
}

impl QLiveSpillWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(path);
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        Ok(Self {
            path: path.to_path_buf(),
            file,
            total_records: 0,
            runs: HashMap::new(),
            run_count: 0,
        })
    }

    fn spill_acc(&mut self, field: FieldId, acc: HashMap<u32, u32>) -> Result<u64> {
        use std::io::Write;
        if acc.is_empty() {
            return Ok(0);
        }
        let offset = self.total_records;
        let count = acc.len() as u32;
        for (bag_tail, mask) in acc {
            self.file.write_all(&bag_tail.to_le_bytes())?;
            self.file.write_all(&mask.to_le_bytes())?;
        }
        self.total_records += count as u64;
        self.run_count += 1;
        self.runs.entry(field).or_default().push(QLiveSpillRun {
            offset_records: offset,
            count,
        });
        Ok(count as u64)
    }

    fn spill_pairs_ref<'a, I>(&mut self, field: FieldId, pairs: I) -> Result<u64>
    where
        I: IntoIterator<Item = (&'a u32, &'a u32)>,
    {
        use std::io::Write;
        let offset = self.total_records;
        let mut count: u32 = 0;
        for (bag_tail, mask) in pairs {
            self.file.write_all(&bag_tail.to_le_bytes())?;
            self.file.write_all(&mask.to_le_bytes())?;
            count += 1;
        }
        if count == 0 {
            return Ok(0);
        }
        self.total_records += count as u64;
        self.run_count += 1;
        self.runs.entry(field).or_default().push(QLiveSpillRun {
            offset_records: offset,
            count,
        });
        Ok(count as u64)
    }

    fn flush(&mut self) -> Result<()> {
        use std::io::Write;
        self.file.flush()?;
        Ok(())
    }
}

fn qlive_spill_one_largest_capped(
    accs: &mut HashMap<FieldId, HashMap<u32, u32>>,
    spill: &mut QLiveSpillWriter,
    live_classes: &mut u64,
    spill_count: &mut [u32],
    cold: &mut [u8],
    max_spills_per_parent: u32,
) -> Result<(u64, bool)> {
    let Some((&field, _)) = accs
        .iter()
        .filter(|(f, acc)| !acc.is_empty() && spill_count[**f as usize] < max_spills_per_parent)
        .max_by_key(|(_, acc)| acc.len())
    else {
        return Ok((0, false));
    };
    let Some(acc) = accs.remove(&field) else {
        return Ok((0, false));
    };
    let n = acc.len() as u64;
    *live_classes = live_classes.saturating_sub(n);
    spill.spill_acc(field, acc)?;
    let pi = field as usize;
    spill_count[pi] = spill_count[pi].saturating_add(1);
    let mut became_cold = false;
    if spill_count[pi] >= max_spills_per_parent && cold[pi] == 0 {
        cold[pi] = 1;
        became_cold = true;
    }
    Ok((n, became_cold))
}

fn qlive_finalize_field_with_spills(
    field: FieldId,
    mut acc: HashMap<u32, u32>,
    spill: &mut QLiveSpillWriter,
    qid_lookup: &[u32],
    qids: &mut Vec<u32>,
    spool: &mut QLiveSpoolWriter,
) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    if let Some(runs) = spill.runs.remove(&field) {
        spill.flush()?;
        let mut file = std::fs::File::open(&spill.path)?;
        let mut buf = vec![0u8; 0];
        for run in runs {
            let bytes_len = run.count as usize * 8;
            buf.resize(bytes_len, 0);
            file.seek(SeekFrom::Start(run.offset_records * 8))?;
            file.read_exact(&mut buf)?;
            let mut pos = 0usize;
            while pos < buf.len() {
                let bag_tail = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
                let mask = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
                pos += 8;
                *acc.entry(bag_tail).or_insert(0) |= mask;
            }
        }
    }
    qlive_finalize_field(field, acc, qid_lookup, qids, spool)
}

/// Dry-run SSD-spill scheduler simulation for the frontier-delta order.
///
/// This does not generate qclasses and does not write layer files.  It uses the
/// child->parent graph plus an inexpensive proxy for accumulator growth:
/// each child contribution to a touched parent adds `input_block_chain_count`
/// units to that parent's RAM accumulator.
///
/// The spill model is bounded by `max_spills_per_parent`:
/// - while spill_count[parent] < max_spills_per_parent, a parent may spill its
///   RAM accumulator to SSD and then continue accumulating in RAM later;
/// - when spill_count[parent] reaches max_spills_per_parent, the parent becomes
///   cold, and all later contributions are counted as SSD append-only direct
///   writes, never re-entering the RAM accumulator.
///
/// This models the intended RAM/SSD schedule more closely than unbounded
/// repeated spill: it controls parent-local RAM<->SSD round trips.
pub fn qlive_frontier_spill_sim(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    qgraph: &QRevClassGraph,
    cfg: QLiveFrontierSpillConfig,
) -> Result<()> {
    let start = Instant::now();
    let (touches_summary, remaining0) = qlive_build_child_order_and_counts(rgraph, input, qgraph);
    let touches = qlive_build_frontier_order(input, &touches_summary, &remaining0);
    let mut remaining = remaining0;

    let mut open = vec![0u8; crate::graph::NUM_FIELDS];
    let mut cold = vec![0u8; crate::graph::NUM_FIELDS];
    let mut spill_count = vec![0u8; crate::graph::NUM_FIELDS];

    let mut active_fields: u64 = 0;
    let mut max_active_fields: u64 = 0;
    let mut active_fields_sum: u128 = 0;

    // RAM accumulator proxy: field -> proxy class count currently held in RAM.
    let mut acc = HashMap::<FieldId, u64>::with_capacity(200_000);
    // SSD proxy: field -> proxy records written to SSD and later read at finalize.
    let mut spilled_by_field = HashMap::<FieldId, u64>::with_capacity(200_000);

    let mut live_classes: u64 = 0;
    let mut max_live_classes: u64 = 0;
    let mut live_classes_sum: u128 = 0;

    // Records written by flushing RAM accumulators.
    let mut spill_flush_records: u64 = 0;
    // Records written directly to SSD after a parent becomes cold.
    let mut cold_direct_records: u64 = 0;
    let mut merge_read_records: u64 = 0;
    let mut spill_events: u64 = 0;
    let mut parents_cold_ever: u64 = 0;
    let mut active_cold_parents: u64 = 0;
    let mut finalized_fields: u64 = 0;
    let mut closed_ram_records: u64 = 0;
    let target_after_global_spill = cfg.global_live_class_limit.saturating_mul(80) / 100;
    let max_spills = cfg.max_spills_per_parent.min(u8::MAX as u32) as u8;

    let spill_parent = |field: FieldId,
                        acc: &mut HashMap<FieldId, u64>,
                        spilled_by_field: &mut HashMap<FieldId, u64>,
                        live_classes: &mut u64,
                        spill_flush_records: &mut u64,
                        spill_events: &mut u64,
                        parents_cold_ever: &mut u64,
                        active_cold_parents: &mut u64,
                        cold: &mut [u8],
                        spill_count: &mut [u8]|
     -> u64 {
        let pi = field as usize;
        if cold[pi] != 0 {
            return 0;
        }
        let Some(n) = acc.remove(&field) else {
            return 0;
        };
        if n == 0 {
            return 0;
        }
        *live_classes = live_classes.saturating_sub(n);
        *spill_flush_records = spill_flush_records.saturating_add(n);
        {
            let entry = spilled_by_field.entry(field).or_insert(0);
            *entry = entry.saturating_add(n);
        }
        *spill_events += 1;
        if max_spills > 0 {
            spill_count[pi] = spill_count[pi].saturating_add(1);
            if spill_count[pi] >= max_spills {
                cold[pi] = 1;
                *parents_cold_ever += 1;
                *active_cold_parents += 1;
            }
        } else {
            // N=0 means no RAM re-use: first spill immediately makes the parent cold.
            cold[pi] = 1;
            *parents_cold_ever += 1;
            *active_cold_parents += 1;
        }
        n
    };

    eprintln!(
        "q live frontier-spill sim start ordered_blocks={} input_states={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={} note=proxy_count_is_input_chains_per_child_parent bounded_parent_roundtrips=true",
        touches.len(),
        input.total_states,
        cfg.global_live_class_limit,
        cfg.per_field_class_limit,
        cfg.max_spills_per_parent,
    );

    for (oi, touch) in touches.iter().enumerate() {
        let w = input.metas[touch.block_idx].count as u64;

        // Add this child's proxy contribution to every touched parent.
        for &parent in &touch.parents {
            let pi = parent as usize;
            if open[pi] == 0 {
                open[pi] = 1;
                active_fields += 1;
            }

            if cold[pi] != 0 {
                // Cold parent: do not rebuild RAM acc.  Append-only SSD proxy.
                cold_direct_records = cold_direct_records.saturating_add(w);
                {
                    let entry = spilled_by_field.entry(parent).or_insert(0);
                    *entry = entry.saturating_add(w);
                }
                continue;
            }

            let mut should_spill = false;
            {
                let e = acc.entry(parent).or_insert(0);
                *e = e.saturating_add(w);
                live_classes = live_classes.saturating_add(w);
                if cfg.per_field_class_limit > 0 && *e >= cfg.per_field_class_limit as u64 {
                    should_spill = true;
                }
            }
            if should_spill {
                spill_parent(
                    parent,
                    &mut acc,
                    &mut spilled_by_field,
                    &mut live_classes,
                    &mut spill_flush_records,
                    &mut spill_events,
                    &mut parents_cold_ever,
                    &mut active_cold_parents,
                    &mut cold,
                    &mut spill_count,
                );
            }
        }

        // If global RAM proxy exceeds the target, spill large/long-lived parents first.
        if cfg.global_live_class_limit > 0 && live_classes > cfg.global_live_class_limit {
            let target = target_after_global_spill.max(1);
            while live_classes > target {
                let Some((&field, &n)) = acc
                    .iter()
                    .filter(|(&field, &n)| n != 0 && cold[field as usize] == 0)
                    .max_by_key(|(&field, &n)| {
                        let rem = remaining[field as usize].max(1) as u64;
                        let longevity = 64u64.saturating_sub((rem + 1).leading_zeros() as u64);
                        // Prefer large, long-lived parents.  Slightly prefer parents
                        // with fewer previous spills to avoid exhausting N on small accs.
                        let remaining_spills = max_spills
                            .saturating_sub(spill_count[field as usize])
                            .max(1) as u64;
                        n.saturating_mul(longevity.max(1))
                            .saturating_mul(remaining_spills)
                    })
                else {
                    break;
                };
                if n == 0 {
                    break;
                }
                let before = live_classes;
                spill_parent(
                    field,
                    &mut acc,
                    &mut spilled_by_field,
                    &mut live_classes,
                    &mut spill_flush_records,
                    &mut spill_events,
                    &mut parents_cold_ever,
                    &mut active_cold_parents,
                    &mut cold,
                    &mut spill_count,
                );
                if live_classes == before {
                    break;
                }
            }
        }

        // Close parents whose last source child has now been processed.
        for &parent in &touch.parents {
            let pi = parent as usize;
            debug_assert!(remaining[pi] > 0);
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                if open[pi] != 0 {
                    open[pi] = 0;
                    active_fields = active_fields.saturating_sub(1);
                }
                if cold[pi] != 0 {
                    active_cold_parents = active_cold_parents.saturating_sub(1);
                }
                if let Some(n) = acc.remove(&parent) {
                    live_classes = live_classes.saturating_sub(n);
                    closed_ram_records = closed_ram_records.saturating_add(n);
                }
                if let Some(n) = spilled_by_field.remove(&parent) {
                    merge_read_records = merge_read_records.saturating_add(n);
                }
                finalized_fields += 1;
            }
        }

        max_active_fields = max_active_fields.max(active_fields);
        max_live_classes = max_live_classes.max(live_classes);
        active_fields_sum += active_fields as u128;
        live_classes_sum += live_classes as u128;

        let done = oi + 1;
        if done % 1024 == 0 || done == touches.len() {
            let total_write_records = spill_flush_records.saturating_add(cold_direct_records);
            eprintln!(
                "  q live frontier-spill sim blocks_done={}/{} finalized_fields={} active_fields={} max_active_fields={} live_classes={} max_live_classes={} avg_live_classes={:.1} spill_flush_records={} cold_direct_records={} total_write_records={} merge_read_records={} spill_events={} parents_cold_ever={} active_cold_parents={} ram_acc_fields={} elapsed={:.3}s",
                done,
                touches.len(),
                finalized_fields,
                active_fields,
                max_active_fields,
                live_classes,
                max_live_classes,
                live_classes_sum as f64 / done as f64,
                spill_flush_records,
                cold_direct_records,
                total_write_records,
                merge_read_records,
                spill_events,
                parents_cold_ever,
                active_cold_parents,
                acc.iter().filter(|(_, &n)| n != 0).count(),
                start.elapsed().as_secs_f64(),
            );
        }
    }

    let total_write_records = spill_flush_records.saturating_add(cold_direct_records);
    eprintln!(
        "q live frontier-spill sim done finalized_fields={} max_active_fields={} avg_active_fields={:.3} max_live_classes={} avg_live_classes={:.3} spill_flush_records={} cold_direct_records={} total_write_records={} merge_read_records={} closed_ram_records={} spill_events={} parents_cold_ever={} live_classes_end={} active_fields_end={} elapsed={:.3}s",
        finalized_fields,
        max_active_fields,
        active_fields_sum as f64 / touches.len().max(1) as f64,
        max_live_classes,
        live_classes_sum as f64 / touches.len().max(1) as f64,
        spill_flush_records,
        cold_direct_records,
        total_write_records,
        merge_read_records,
        closed_ram_records,
        spill_events,
        parents_cold_ever,
        live_classes,
        active_fields,
        start.elapsed().as_secs_f64(),
    );
    println!("q_live_frontier_spill_sim max_live_classes={max_live_classes}");
    println!(
        "q_live_frontier_spill_sim avg_live_classes={:.3}",
        live_classes_sum as f64 / touches.len().max(1) as f64
    );
    println!("q_live_frontier_spill_sim spill_flush_records={spill_flush_records}");
    println!("q_live_frontier_spill_sim cold_direct_records={cold_direct_records}");
    println!("q_live_frontier_spill_sim total_write_records={total_write_records}");
    println!("q_live_frontier_spill_sim merge_read_records={merge_read_records}");
    println!("q_live_frontier_spill_sim spill_events={spill_events}");
    println!("q_live_frontier_spill_sim parents_cold_ever={parents_cold_ever}");
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct QLiveFrontierSpillConfig {
    pub global_live_class_limit: u64,
    pub per_field_class_limit: usize,
    pub max_spills_per_parent: u32,
}

/// Experimental lifetime-based q-class external step.
///
/// Difference from `external_step_q_class`:
/// - Prepass computes, for every parent field, how many child fields can still
///   contribute to it.
/// - Child fields are processed in a parent-neighborhood order.
/// - Parent field accumulators `(bag_tail -> avail_mask)` are finalized and
///   evicted as soon as their last source child field has been processed.
///
/// This avoids writing every class record to bucket files, but may use more RAM
/// if the live parent-field frontier is large. It is intended as a direct test
/// of the field-lifetime idea.
pub fn external_step_q_live(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();

    let (touches, mut remaining) = qlive_build_child_order_and_counts(rgraph, input, qgraph);

    eprintln!("building compact qid lookup table for live materialization...");
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );

    let spool_path = work_dir.join("qlive_field_spool.u24.bin");
    let _ = std::fs::remove_file(&spool_path);
    let mut spool = QLiveSpoolWriter::create(&spool_path)?;

    let mut accs = HashMap::<FieldId, HashMap<u32, u32>>::with_capacity(1_000_000);
    let mut qids_tmp = Vec::<u32>::new();
    let mut stats = QLiveStats::default();
    let mut live_classes: u64 = 0;
    let mut finalized_fields: u64 = 0;
    let mut output_states: u64 = 0;

    eprintln!(
        "q live generation start ordered_blocks={} input_states={}",
        touches.len(),
        input.total_states
    );

    for (oi, touch) in touches.iter().enumerate() {
        let qids = input.read_block_chains(touch.block_idx);
        let child_field = touch.child_field;
        let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
            std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

        for qid in qids {
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                let parent_count = rgraph.edges(child_field, placed).len() as u64;
                if parent_count == 0 {
                    continue;
                }
                let map = &mut by_piece[placed as usize];
                for rec in recs {
                    stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                    *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                }
            }
        }

        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let map = &mut by_piece[placed as usize];
            if map.is_empty() {
                continue;
            }
            let parents = rgraph.edges(child_field, placed);
            for &parent in parents {
                let acc = accs
                    .entry(parent)
                    .or_insert_with(|| HashMap::with_capacity(map.len() * 2));
                for (&bag_tail, &mask) in map.iter() {
                    let e = acc.entry(bag_tail).or_insert(0);
                    if *e == 0 {
                        live_classes += 1;
                    }
                    *e |= mask;
                    stats.emitted_records += 1;
                    stats.local_chain_masked += mask.count_ones() as u64;
                }
            }
            map.clear();
        }

        // This child field's complete contribution has been incorporated.
        // Decrement each touched parent once, regardless of how many qclasses it received.
        for &parent in &touch.parents {
            let r = &mut remaining[parent as usize];
            debug_assert!(*r > 0);
            *r -= 1;
            if *r == 0 {
                if let Some(acc) = accs.remove(&parent) {
                    live_classes = live_classes.saturating_sub(acc.len() as u64);
                    let n =
                        qlive_finalize_field(parent, acc, &qid_lookup, &mut qids_tmp, &mut spool)
                            .with_context(|| format!("finalize live parent field {parent}"))?;
                    output_states += n;
                    finalized_fields += 1;
                }
            }
        }

        let done = oi + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q live blocks_done={}/{} finalized_fields={} live_fields={} live_classes={} output_states={} raw={} emitted={} elapsed={:.1}s",
                done,
                touches.len(),
                finalized_fields,
                accs.len(),
                live_classes,
                output_states,
                stats.raw,
                stats.emitted_records,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !accs.is_empty() {
        eprintln!(
            "q live warning: finalizing {} remaining accs at end",
            accs.len()
        );
        let rest = std::mem::take(&mut accs);
        for (field, acc) in rest {
            let n = qlive_finalize_field(field, acc, &qid_lookup, &mut qids_tmp, &mut spool)?;
            output_states += n;
            finalized_fields += 1;
        }
    }
    spool.flush()?;
    let metas = std::mem::take(&mut spool.metas);
    drop(spool);

    eprintln!(
        "q live generation done raw={} emitted_class_records={} local_chain_masked={} finalized_fields={} spooled_states={} elapsed={:.1}s",
        stats.raw,
        stats.emitted_records,
        stats.local_chain_masked,
        finalized_fields,
        output_states,
        start.elapsed().as_secs_f64()
    );

    let finished = qlive_assemble_spool(&spool_path, metas, output_prefix)?;
    let _ = std::fs::remove_file(&spool_path);
    println!("q_external_live raw_pairs={}", stats.raw);
    println!(
        "q_external_live emitted_class_records={}",
        stats.emitted_records
    );
    println!(
        "q_external_live local_chain_masked={}",
        stats.local_chain_masked
    );
    println!("q_external_live output_states={finished}");
    println!(
        "q_external_live removed_dupes={}",
        stats.raw.saturating_sub(finished)
    );
    println!(
        "q_external_live output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "q_external_live output_u24_plus_fp16_bytes={} ({:.3} GiB)",
        finished * 5,
        finished as f64 * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external live step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

/// Frontier-delta live q-class step with bounded SSD spill.
///
/// This keeps the lifetime-based finalization semantics, but it does not require
/// all live `(parent_field, bag_tail)` classes to remain in RAM.  Large per-field
/// accumulators, and the largest live accumulators when the global cap is
/// exceeded, are written to a single SSD spill file as compact `(bag_tail, mask)`
/// runs.  When a parent field closes, its RAM accumulator plus its spilled runs
/// are merged and materialized to qids.
pub fn external_step_q_live_frontier_spill(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    cfg: QLiveFrontierSpillConfig,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();

    let (touches_summary, remaining0) = qlive_build_child_order_and_counts(rgraph, input, qgraph);
    let touches = qlive_build_frontier_order(input, &touches_summary, &remaining0);
    let mut remaining = remaining0;

    eprintln!("building compact qid lookup table for live frontier spill materialization...");
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );

    let spool_path = work_dir.join("qlive_frontier_field_spool.u24.bin");
    let spill_path = work_dir.join("qlive_frontier_spill.classes.bin");
    let _ = std::fs::remove_file(&spool_path);
    let _ = std::fs::remove_file(&spill_path);
    let mut spool = QLiveSpoolWriter::create(&spool_path)?;
    let mut spill = QLiveSpillWriter::create(&spill_path)?;

    let mut accs = HashMap::<FieldId, HashMap<u32, u32>>::with_capacity(300_000);
    let mut qids_tmp = Vec::<u32>::new();
    let mut stats = QLiveStats::default();
    let mut live_classes: u64 = 0;
    let mut spilled_records: u64 = 0;
    let mut finalized_fields: u64 = 0;
    let mut output_states: u64 = 0;
    let mut cold_direct_records: u64 = 0;
    let mut spill_events: u64 = 0;
    let mut parents_cold_ever: u64 = 0;
    let mut spill_count = vec![0u32; crate::graph::NUM_FIELDS];
    let mut cold = vec![0u8; crate::graph::NUM_FIELDS];
    let target_after_global_spill = cfg.global_live_class_limit.saturating_mul(80) / 100;

    eprintln!(
        "q live frontier-spill generation start ordered_blocks={} input_states={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}",
        touches.len(),
        input.total_states,
        cfg.global_live_class_limit,
        cfg.per_field_class_limit,
        cfg.max_spills_per_parent
    );

    for (oi, touch) in touches.iter().enumerate() {
        let qids = input.read_block_chains(touch.block_idx);
        let child_field = touch.child_field;
        let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
            std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

        for qid in qids {
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                let parent_count = rgraph.edges(child_field, placed).len() as u64;
                if parent_count == 0 {
                    continue;
                }
                let map = &mut by_piece[placed as usize];
                for rec in recs {
                    stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                    *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                }
            }
        }

        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let map = &mut by_piece[placed as usize];
            if map.is_empty() {
                continue;
            }
            let parents = rgraph.edges(child_field, placed);
            for &parent in parents {
                let pi = parent as usize;
                if cold[pi] != 0 {
                    let n = spill.spill_pairs_ref(parent, map.iter())?;
                    cold_direct_records += n;
                    spilled_records += n;
                    stats.emitted_records += map.len() as u64;
                    stats.local_chain_masked +=
                        map.values().map(|m| m.count_ones() as u64).sum::<u64>();
                    continue;
                }

                let mut should_spill_parent = false;
                {
                    let acc = accs
                        .entry(parent)
                        .or_insert_with(|| HashMap::with_capacity(map.len() * 2));
                    for (&bag_tail, &mask) in map.iter() {
                        let e = acc.entry(bag_tail).or_insert(0);
                        if *e == 0 {
                            live_classes += 1;
                        }
                        *e |= mask;
                        stats.emitted_records += 1;
                        stats.local_chain_masked += mask.count_ones() as u64;
                    }
                    if cfg.per_field_class_limit > 0 && acc.len() >= cfg.per_field_class_limit {
                        should_spill_parent = true;
                    }
                }
                if should_spill_parent && spill_count[pi] < cfg.max_spills_per_parent {
                    if let Some(acc) = accs.remove(&parent) {
                        let n = acc.len() as u64;
                        live_classes = live_classes.saturating_sub(n);
                        spilled_records += spill.spill_acc(parent, acc)?;
                        spill_events += 1;
                        spill_count[pi] = spill_count[pi].saturating_add(1);
                        if spill_count[pi] >= cfg.max_spills_per_parent {
                            cold[pi] = 1;
                            parents_cold_ever += 1;
                        }
                    }
                }
            }
            map.clear();
        }

        if cfg.global_live_class_limit > 0 && live_classes > cfg.global_live_class_limit {
            let target = target_after_global_spill.max(1);
            while live_classes > target {
                let (n, became_cold) = qlive_spill_one_largest_capped(
                    &mut accs,
                    &mut spill,
                    &mut live_classes,
                    &mut spill_count,
                    &mut cold,
                    cfg.max_spills_per_parent,
                )?;
                if n == 0 {
                    break;
                }
                spilled_records += n;
                spill_events += 1;
                if became_cold {
                    parents_cold_ever += 1;
                }
            }
        }

        // This child field's complete contribution has been incorporated.
        // Decrement each touched parent once, regardless of qclass volume.
        for &parent in &touch.parents {
            let r = &mut remaining[parent as usize];
            debug_assert!(*r > 0);
            *r -= 1;
            if *r == 0 {
                let acc = accs.remove(&parent).unwrap_or_default();
                live_classes = live_classes.saturating_sub(acc.len() as u64);
                let n = qlive_finalize_field_with_spills(
                    parent,
                    acc,
                    &mut spill,
                    &qid_lookup,
                    &mut qids_tmp,
                    &mut spool,
                )
                .with_context(|| format!("finalize live frontier-spill parent field {parent}"))?;
                output_states += n;
                finalized_fields += 1;
            }
        }

        let done = oi + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q live frontier-spill blocks_done={}/{} finalized_fields={} live_fields={} live_classes={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} output_states={} raw={} emitted={} elapsed={:.1}s",
                done,
                touches.len(),
                finalized_fields,
                accs.len(),
                live_classes,
                spilled_records,
                cold_direct_records,
                spill_events,
                spill.run_count,
                parents_cold_ever,
                output_states,
                stats.raw,
                stats.emitted_records,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !accs.is_empty() || !spill.runs.is_empty() {
        eprintln!(
            "q live frontier-spill warning: finalizing remaining accs={} spilled_fields={} at end",
            accs.len(),
            spill.runs.len()
        );
        let fields: Vec<FieldId> = accs
            .keys()
            .copied()
            .chain(spill.runs.keys().copied())
            .collect();
        let mut fields = fields;
        fields.sort_unstable();
        fields.dedup();
        for field in fields {
            let acc = accs.remove(&field).unwrap_or_default();
            live_classes = live_classes.saturating_sub(acc.len() as u64);
            let n = qlive_finalize_field_with_spills(
                field,
                acc,
                &mut spill,
                &qid_lookup,
                &mut qids_tmp,
                &mut spool,
            )?;
            output_states += n;
            finalized_fields += 1;
        }
    }
    spill.flush()?;
    spool.flush()?;
    let spill_run_count = spill.run_count;
    let metas = std::mem::take(&mut spool.metas);
    drop(spool);
    drop(spill);

    eprintln!(
        "q live frontier-spill generation done raw={} emitted_class_records={} local_chain_masked={} finalized_fields={} spooled_states={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} spill_file_records={} live_classes_end={} elapsed={:.1}s",
        stats.raw,
        stats.emitted_records,
        stats.local_chain_masked,
        finalized_fields,
        output_states,
        spilled_records,
        cold_direct_records,
        spill_events,
        spill_run_count,
        parents_cold_ever,
        spilled_records,
        live_classes,
        start.elapsed().as_secs_f64()
    );

    let finished = qlive_assemble_spool(&spool_path, metas, output_prefix)?;
    let _ = std::fs::remove_file(&spool_path);
    let _ = std::fs::remove_file(&spill_path);
    println!("q_external_live_frontier_spill raw_pairs={}", stats.raw);
    println!(
        "q_external_live_frontier_spill emitted_class_records={}",
        stats.emitted_records
    );
    println!(
        "q_external_live_frontier_spill local_chain_masked={}",
        stats.local_chain_masked
    );
    println!("q_external_live_frontier_spill spilled_records={spilled_records}");
    println!("q_external_live_frontier_spill cold_direct_records={cold_direct_records}");
    println!("q_external_live_frontier_spill spill_events={spill_events}");
    println!(
        "q_external_live_frontier_spill spill_runs={}",
        spill_run_count
    );
    println!("q_external_live_frontier_spill parents_cold_ever={parents_cold_ever}");
    println!("q_external_live_frontier_spill output_states={finished}");
    println!(
        "q_external_live_frontier_spill removed_dupes={}",
        stats.raw.saturating_sub(finished)
    );
    println!(
        "q_external_live_frontier_spill output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "q_external_live_frontier_spill output_u24_plus_fp16_bytes={} ({:.3} GiB)",
        finished * 5,
        finished as f64 * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external live frontier-spill step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

fn qlive_write_closed_acc_run_only(
    field: FieldId,
    acc: HashMap<u32, u32>,
    class_tmp: &mut Vec<(u32, u32)>,
    class_spool: &mut QLiveClassSpoolWriter,
) -> Result<u64> {
    class_tmp.clear();
    class_tmp.extend(acc.into_iter().filter(|(_, mask)| *mask != 0));
    class_tmp.sort_unstable_by_key(|&(bag_tail, _)| bag_tail);
    let n = class_spool.write_field_classes(field, class_tmp)?;
    class_tmp.clear();
    Ok(n)
}

fn qlive_materialize_class_and_spill_runs(
    class_spool_path: &Path,
    class_metas: Vec<(FieldId, u32, u64)>,
    spill_path: &Path,
    mut spill_runs: HashMap<FieldId, Vec<QLiveSpillRun>>,
    mut closed_fields: Vec<FieldId>,
    qid_lookup: &[u32],
    output_prefix: &Path,
    qstate_count: usize,
) -> Result<u64> {
    use anyhow::bail;
    use std::io::{Read, Seek, SeekFrom};

    closed_fields.sort_unstable();
    closed_fields.dedup();
    for w in closed_fields.windows(2) {
        if w[0] >= w[1] {
            bail!("closed fields are not strictly sorted after dedup");
        }
    }

    let mut class_by_field =
        HashMap::<FieldId, (u32, u64)>::with_capacity(class_metas.len() * 2 + 1);
    for (field, count, offset) in class_metas {
        if class_by_field.insert(field, (count, offset)).is_some() {
            bail!("duplicate class spool meta for field {field}");
        }
    }

    let mut class_file = std::fs::File::open(class_spool_path)?;
    let mut spill_file = if spill_path.exists() {
        Some(std::fs::File::open(spill_path)?)
    } else {
        None
    };
    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut bytes = Vec::<u8>::new();
    let mut acc = HashMap::<u32, u32>::new();
    let mut qids = Vec::<u32>::new();
    let mut bitset = vec![0u64; (qstate_count + 63) / 64];
    let mut total = 0u64;
    let st = Instant::now();
    let total_fields = closed_fields.len();
    let mut materialized_class_records = 0u64;
    let mut materialized_spill_records = 0u64;
    let mut bitset_fields = 0u64;
    let mut vec_fields = 0u64;

    for (i, field) in closed_fields.into_iter().enumerate() {
        acc.clear();

        if let Some((count, offset)) = class_by_field.remove(&field) {
            let bytes_len = count as usize * 8;
            bytes.resize(bytes_len, 0);
            class_file.seek(SeekFrom::Start(offset * 8))?;
            class_file.read_exact(&mut bytes)?;
            let mut pos = 0usize;
            while pos < bytes.len() {
                let bag_tail = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                let mask = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
                pos += 8;
                *acc.entry(bag_tail).or_insert(0) |= mask;
            }
            materialized_class_records += count as u64;
        }

        if let Some(runs) = spill_runs.remove(&field) {
            let Some(sf) = spill_file.as_mut() else {
                bail!("spill runs exist but spill file is missing");
            };
            for run in runs {
                let bytes_len = run.count as usize * 8;
                bytes.resize(bytes_len, 0);
                sf.seek(SeekFrom::Start(run.offset_records * 8))?;
                sf.read_exact(&mut bytes)?;
                let mut pos = 0usize;
                while pos < bytes.len() {
                    let bag_tail = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                    let mask = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
                    pos += 8;
                    *acc.entry(bag_tail).or_insert(0) |= mask;
                }
                materialized_spill_records += run.count as u64;
            }
        }

        // If the class set is large, use a qid bitset.  This removes qid Vec growth,
        // repeated dedup, and produces sorted qids directly.  Small fields use Vec+sort
        // because clearing a 100KB bitset per field would dominate.
        let estimated_qids: usize = acc.values().map(|m| m.count_ones() as usize).sum();
        let field_count: u64;
        if estimated_qids >= 96_000 {
            bitset.fill(0);
            for (bag_tail, mut mask) in acc.drain() {
                while mask != 0 {
                    let avail = mask.trailing_zeros() as u8;
                    mask &= mask - 1;
                    let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
                    if qid != QID_MISSING {
                        let qi = qid as usize;
                        bitset[qi >> 6] |= 1u64 << (qi & 63);
                    }
                }
            }
            let mut n = 0u64;
            for (word_i, &word0) in bitset.iter().enumerate() {
                let mut word = word0;
                while word != 0 {
                    let b = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let qid = (word_i * 64 + b) as u32;
                    writer.push_pair_key(q_pair_key(field, qid))?;
                    n += 1;
                }
            }
            field_count = n;
            bitset_fields += 1;
        } else {
            qids.clear();
            qids.reserve(estimated_qids);
            for (bag_tail, mut mask) in acc.drain() {
                while mask != 0 {
                    let avail = mask.trailing_zeros() as u8;
                    mask &= mask - 1;
                    let qid = qid_from_lookup(qid_lookup, bag_tail, avail);
                    if qid != QID_MISSING {
                        qids.push(qid);
                    }
                }
            }
            qids.sort_unstable();
            qids.dedup();
            field_count = qids.len() as u64;
            for &qid in &qids {
                writer.push_pair_key(q_pair_key(field, qid))?;
            }
            vec_fields += 1;
        }
        total += field_count;

        let done = i + 1;
        if done % 65536 == 0 || done == total_fields {
            eprintln!(
                "  q live strong materialize fields_done={}/{} output_states={} class_records_read={} spill_records_read={} bitset_fields={} vec_fields={} elapsed={:.1}s",
                done,
                total_fields,
                total,
                materialized_class_records,
                materialized_spill_records,
                bitset_fields,
                vec_fields,
                st.elapsed().as_secs_f64()
            );
        }
    }
    writer.finish()
}

/// Frontier-delta live q-class step with bounded SSD spill and deferred materialization.
///
/// Difference from `external_step_q_live_frontier_spill`:
/// when a parent field closes, this writes a compact class run `(bag_tail, avail_mask)`
/// to a class spool instead of immediately materializing qids and writing final u24.
/// A second sequential phase materializes fields in field_id order.  This keeps the
/// generation hot loop focused on accumulation/spill scheduling and moves qid lookup,
/// qid sort/dedup, and final layer writes out of the child-processing loop.
pub fn external_step_q_live_frontier_spill_deferred(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    cfg: QLiveFrontierSpillConfig,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();

    let (touches_summary, remaining0) = qlive_build_child_order_and_counts(rgraph, input, qgraph);
    let touches = qlive_build_frontier_order(input, &touches_summary, &remaining0);
    let mut remaining = remaining0;

    let class_spool_path = work_dir.join("qlive_frontier_class_spool.bin");
    let spill_path = work_dir.join("qlive_frontier_deferred_spill.classes.bin");
    let _ = std::fs::remove_file(&class_spool_path);
    let _ = std::fs::remove_file(&spill_path);
    let mut class_spool = QLiveClassSpoolWriter::create(&class_spool_path)?;
    let mut spill = QLiveSpillWriter::create(&spill_path)?;

    let mut accs = HashMap::<FieldId, HashMap<u32, u32>>::with_capacity(300_000);
    let mut class_tmp = Vec::<(u32, u32)>::new();
    let mut stats = QLiveStats::default();
    let mut live_classes: u64 = 0;
    let mut spilled_records: u64 = 0;
    let mut finalized_fields: u64 = 0;
    let mut closed_class_records: u64 = 0;
    let mut cold_direct_records: u64 = 0;
    let mut spill_events: u64 = 0;
    let mut parents_cold_ever: u64 = 0;
    let mut spill_count = vec![0u32; crate::graph::NUM_FIELDS];
    let mut cold = vec![0u8; crate::graph::NUM_FIELDS];
    let target_after_global_spill = cfg.global_live_class_limit.saturating_mul(80) / 100;

    eprintln!(
        "q live frontier-spill-deferred generation start ordered_blocks={} input_states={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}",
        touches.len(),
        input.total_states,
        cfg.global_live_class_limit,
        cfg.per_field_class_limit,
        cfg.max_spills_per_parent
    );

    for (oi, touch) in touches.iter().enumerate() {
        let qids = input.read_block_chains(touch.block_idx);
        let child_field = touch.child_field;
        let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
            std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

        for qid in qids {
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                let parent_count = rgraph.edges(child_field, placed).len() as u64;
                if parent_count == 0 {
                    continue;
                }
                let map = &mut by_piece[placed as usize];
                for rec in recs {
                    stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                    *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                }
            }
        }

        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let map = &mut by_piece[placed as usize];
            if map.is_empty() {
                continue;
            }
            let parents = rgraph.edges(child_field, placed);
            let map_masked: u64 = map.values().map(|m| m.count_ones() as u64).sum();
            for &parent in parents {
                let pi = parent as usize;
                if cold[pi] != 0 {
                    let n = spill.spill_pairs_ref(parent, map.iter())?;
                    cold_direct_records += n;
                    spilled_records += n;
                    stats.emitted_records += map.len() as u64;
                    stats.local_chain_masked += map_masked;
                    continue;
                }

                let mut should_spill_parent = false;
                {
                    let acc = accs
                        .entry(parent)
                        .or_insert_with(|| HashMap::with_capacity(map.len() * 2));
                    for (&bag_tail, &mask) in map.iter() {
                        let e = acc.entry(bag_tail).or_insert(0);
                        if *e == 0 {
                            live_classes += 1;
                        }
                        *e |= mask;
                        stats.emitted_records += 1;
                        stats.local_chain_masked += mask.count_ones() as u64;
                    }
                    if cfg.per_field_class_limit > 0 && acc.len() >= cfg.per_field_class_limit {
                        should_spill_parent = true;
                    }
                }
                if should_spill_parent && spill_count[pi] < cfg.max_spills_per_parent {
                    if let Some(acc) = accs.remove(&parent) {
                        let n = acc.len() as u64;
                        live_classes = live_classes.saturating_sub(n);
                        spilled_records += spill.spill_acc(parent, acc)?;
                        spill_events += 1;
                        spill_count[pi] = spill_count[pi].saturating_add(1);
                        if spill_count[pi] >= cfg.max_spills_per_parent && cold[pi] == 0 {
                            cold[pi] = 1;
                            parents_cold_ever += 1;
                        }
                    }
                }
            }
            map.clear();
        }

        if cfg.global_live_class_limit > 0 && live_classes > cfg.global_live_class_limit {
            let target = target_after_global_spill.max(1);
            while live_classes > target {
                let (n, became_cold) = qlive_spill_one_largest_capped(
                    &mut accs,
                    &mut spill,
                    &mut live_classes,
                    &mut spill_count,
                    &mut cold,
                    cfg.max_spills_per_parent,
                )?;
                if n == 0 {
                    break;
                }
                spilled_records += n;
                spill_events += 1;
                if became_cold {
                    parents_cold_ever += 1;
                }
            }
        }

        for &parent in &touch.parents {
            let r = &mut remaining[parent as usize];
            debug_assert!(*r > 0);
            *r -= 1;
            if *r == 0 {
                let acc = accs.remove(&parent).unwrap_or_default();
                live_classes = live_classes.saturating_sub(acc.len() as u64);
                let n = qlive_write_closed_class_run_with_spills(
                    parent,
                    acc,
                    &mut spill,
                    &mut class_tmp,
                    &mut class_spool,
                )
                .with_context(|| {
                    format!("close live frontier-spill-deferred parent field {parent}")
                })?;
                closed_class_records += n;
                finalized_fields += 1;
            }
        }

        let done = oi + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q live frontier-spill-deferred blocks_done={}/{} finalized_fields={} live_fields={} live_classes={} closed_class_records={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} raw={} emitted={} elapsed={:.1}s",
                done,
                touches.len(),
                finalized_fields,
                accs.len(),
                live_classes,
                closed_class_records,
                spilled_records,
                cold_direct_records,
                spill_events,
                spill.run_count,
                parents_cold_ever,
                stats.raw,
                stats.emitted_records,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !accs.is_empty() || !spill.runs.is_empty() {
        eprintln!("q live frontier-spill-deferred warning: closing remaining accs={} spilled_fields={} at end", accs.len(), spill.runs.len());
        let fields: Vec<FieldId> = accs
            .keys()
            .copied()
            .chain(spill.runs.keys().copied())
            .collect();
        let mut fields = fields;
        fields.sort_unstable();
        fields.dedup();
        for field in fields {
            let acc = accs.remove(&field).unwrap_or_default();
            live_classes = live_classes.saturating_sub(acc.len() as u64);
            let n = qlive_write_closed_class_run_with_spills(
                field,
                acc,
                &mut spill,
                &mut class_tmp,
                &mut class_spool,
            )?;
            closed_class_records += n;
            finalized_fields += 1;
        }
    }
    spill.flush()?;
    class_spool.flush()?;
    let spill_run_count = spill.run_count;
    let class_metas = std::mem::take(&mut class_spool.metas);
    drop(class_spool);
    drop(spill);

    eprintln!(
        "q live frontier-spill-deferred generation done raw={} emitted_class_records={} local_chain_masked={} finalized_fields={} closed_class_records={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} spill_file_records={} live_classes_end={} elapsed={:.1}s",
        stats.raw,
        stats.emitted_records,
        stats.local_chain_masked,
        finalized_fields,
        closed_class_records,
        spilled_records,
        cold_direct_records,
        spill_events,
        spill_run_count,
        parents_cold_ever,
        spilled_records,
        live_classes,
        start.elapsed().as_secs_f64()
    );

    eprintln!(
        "q live frontier-spill-deferred materialize start class_records={} fields={}",
        closed_class_records,
        class_metas.len()
    );
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );
    let finished =
        qlive_materialize_class_spool(&class_spool_path, class_metas, &qid_lookup, output_prefix)?;
    let _ = std::fs::remove_file(&class_spool_path);
    let _ = std::fs::remove_file(&spill_path);

    println!(
        "q_external_live_frontier_spill_deferred raw_pairs={}",
        stats.raw
    );
    println!(
        "q_external_live_frontier_spill_deferred emitted_class_records={}",
        stats.emitted_records
    );
    println!(
        "q_external_live_frontier_spill_deferred local_chain_masked={}",
        stats.local_chain_masked
    );
    println!("q_external_live_frontier_spill_deferred closed_class_records={closed_class_records}");
    println!("q_external_live_frontier_spill_deferred spilled_records={spilled_records}");
    println!("q_external_live_frontier_spill_deferred cold_direct_records={cold_direct_records}");
    println!("q_external_live_frontier_spill_deferred spill_events={spill_events}");
    println!(
        "q_external_live_frontier_spill_deferred spill_runs={}",
        spill_run_count
    );
    println!("q_external_live_frontier_spill_deferred parents_cold_ever={parents_cold_ever}");
    println!("q_external_live_frontier_spill_deferred output_states={finished}");
    println!(
        "q_external_live_frontier_spill_deferred removed_dupes={}",
        stats.raw.saturating_sub(finished)
    );
    println!(
        "q_external_live_frontier_spill_deferred output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "q_external_live_frontier_spill_deferred output_u24_plus_fp16_bytes={} ({:.3} GiB)",
        finished * 5,
        finished as f64 * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external live frontier-spill-deferred step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}

/// Strong deferred frontier-spill step.  Defers spill-run merge as well as qid materialization out of the child-processing loop; materialization uses a hybrid Vec/bitset writer.
///
/// Difference from `external_step_q_live_frontier_spill`:
/// when a parent field closes, this writes a compact class run `(bag_tail, avail_mask)`
/// to a class spool instead of immediately materializing qids and writing final u24.
/// A second sequential phase materializes fields in field_id order.  This keeps the
/// generation hot loop focused on accumulation/spill scheduling and moves qid lookup,
/// qid sort/dedup, and final layer writes out of the child-processing loop.
pub fn external_step_q_live_frontier_spill_strong(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    q_index: &QStateIndex,
    qgraph: &QRevClassGraph,
    output_prefix: &Path,
    work_dir: &Path,
    cfg: QLiveFrontierSpillConfig,
) -> Result<u64> {
    std::fs::create_dir_all(work_dir)?;
    let start = Instant::now();

    let (touches_summary, remaining0) = qlive_build_child_order_and_counts(rgraph, input, qgraph);
    let touches = qlive_build_frontier_order(input, &touches_summary, &remaining0);
    let mut remaining = remaining0;

    let class_spool_path = work_dir.join("qlive_frontier_strong_class_spool.bin");
    let spill_path = work_dir.join("qlive_frontier_strong_spill.classes.bin");
    let _ = std::fs::remove_file(&class_spool_path);
    let _ = std::fs::remove_file(&spill_path);
    let mut class_spool = QLiveClassSpoolWriter::create(&class_spool_path)?;
    let mut spill = QLiveSpillWriter::create(&spill_path)?;

    let mut accs = HashMap::<FieldId, HashMap<u32, u32>>::with_capacity(300_000);
    let mut class_tmp = Vec::<(u32, u32)>::new();
    let mut closed_fields = Vec::<FieldId>::with_capacity(input.metas.len());
    let mut stats = QLiveStats::default();
    let mut live_classes: u64 = 0;
    let mut spilled_records: u64 = 0;
    let mut finalized_fields: u64 = 0;
    let mut closed_class_records: u64 = 0;
    let mut cold_direct_records: u64 = 0;
    let mut spill_events: u64 = 0;
    let mut parents_cold_ever: u64 = 0;
    let mut spill_count = vec![0u32; crate::graph::NUM_FIELDS];
    let mut cold = vec![0u8; crate::graph::NUM_FIELDS];
    let target_after_global_spill = cfg.global_live_class_limit.saturating_mul(80) / 100;

    eprintln!(
        "q live frontier-spill-strong generation start ordered_blocks={} input_states={} global_live_class_limit={} per_field_class_limit={} max_spills_per_parent={}",
        touches.len(),
        input.total_states,
        cfg.global_live_class_limit,
        cfg.per_field_class_limit,
        cfg.max_spills_per_parent
    );

    for (oi, touch) in touches.iter().enumerate() {
        let qids = input.read_block_chains(touch.block_idx);
        let child_field = touch.child_field;
        let mut by_piece: [HashMap<u32, u32>; crate::piece::PIECE_COUNT] =
            std::array::from_fn(|_| HashMap::<u32, u32>::with_capacity(4096));

        for qid in qids {
            for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
                let recs = qgraph.records_for(qid, placed);
                if recs.is_empty() {
                    continue;
                }
                let parent_count = rgraph.edges(child_field, placed).len() as u64;
                if parent_count == 0 {
                    continue;
                }
                let map = &mut by_piece[placed as usize];
                for rec in recs {
                    stats.raw += rec.avail_mask.count_ones() as u64 * parent_count;
                    *map.entry(rec.bag_tail).or_insert(0) |= rec.avail_mask;
                }
            }
        }

        for placed in 0u8..crate::piece::PIECE_COUNT as u8 {
            let map = &mut by_piece[placed as usize];
            if map.is_empty() {
                continue;
            }
            let parents = rgraph.edges(child_field, placed);
            let map_masked: u64 = map.values().map(|m| m.count_ones() as u64).sum();
            for &parent in parents {
                let pi = parent as usize;
                if cold[pi] != 0 {
                    let n = spill.spill_pairs_ref(parent, map.iter())?;
                    cold_direct_records += n;
                    spilled_records += n;
                    stats.emitted_records += map.len() as u64;
                    stats.local_chain_masked += map_masked;
                    continue;
                }

                let mut should_spill_parent = false;
                {
                    let acc = accs
                        .entry(parent)
                        .or_insert_with(|| HashMap::with_capacity(map.len() * 2));
                    for (&bag_tail, &mask) in map.iter() {
                        let e = acc.entry(bag_tail).or_insert(0);
                        if *e == 0 {
                            live_classes += 1;
                        }
                        *e |= mask;
                        stats.emitted_records += 1;
                        stats.local_chain_masked += mask.count_ones() as u64;
                    }
                    if cfg.per_field_class_limit > 0 && acc.len() >= cfg.per_field_class_limit {
                        should_spill_parent = true;
                    }
                }
                if should_spill_parent && spill_count[pi] < cfg.max_spills_per_parent {
                    if let Some(acc) = accs.remove(&parent) {
                        let n = acc.len() as u64;
                        live_classes = live_classes.saturating_sub(n);
                        spilled_records += spill.spill_acc(parent, acc)?;
                        spill_events += 1;
                        spill_count[pi] = spill_count[pi].saturating_add(1);
                        if spill_count[pi] >= cfg.max_spills_per_parent && cold[pi] == 0 {
                            cold[pi] = 1;
                            parents_cold_ever += 1;
                        }
                    }
                }
            }
            map.clear();
        }

        if cfg.global_live_class_limit > 0 && live_classes > cfg.global_live_class_limit {
            let target = target_after_global_spill.max(1);
            while live_classes > target {
                let (n, became_cold) = qlive_spill_one_largest_capped(
                    &mut accs,
                    &mut spill,
                    &mut live_classes,
                    &mut spill_count,
                    &mut cold,
                    cfg.max_spills_per_parent,
                )?;
                if n == 0 {
                    break;
                }
                spilled_records += n;
                spill_events += 1;
                if became_cold {
                    parents_cold_ever += 1;
                }
            }
        }

        for &parent in &touch.parents {
            let r = &mut remaining[parent as usize];
            debug_assert!(*r > 0);
            *r -= 1;
            if *r == 0 {
                let acc = accs.remove(&parent).unwrap_or_default();
                live_classes = live_classes.saturating_sub(acc.len() as u64);
                let n =
                    qlive_write_closed_acc_run_only(parent, acc, &mut class_tmp, &mut class_spool)
                        .with_context(|| {
                            format!("close live frontier-spill-strong parent field {parent}")
                        })?;
                closed_class_records += n;
                closed_fields.push(parent);
                finalized_fields += 1;
            }
        }

        let done = oi + 1;
        if done % 1024 == 0 || done == touches.len() {
            eprintln!(
                "  q live frontier-spill-strong blocks_done={}/{} finalized_fields={} live_fields={} live_classes={} closed_class_records={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} raw={} emitted={} elapsed={:.1}s",
                done,
                touches.len(),
                finalized_fields,
                accs.len(),
                live_classes,
                closed_class_records,
                spilled_records,
                cold_direct_records,
                spill_events,
                spill.run_count,
                parents_cold_ever,
                stats.raw,
                stats.emitted_records,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if !accs.is_empty() {
        eprintln!("q live frontier-spill-strong warning: closing remaining accs={} at end; spill runs are deferred metadata", accs.len());
        let mut fields: Vec<FieldId> = accs.keys().copied().collect();
        fields.sort_unstable();
        fields.dedup();
        for field in fields {
            let acc = accs.remove(&field).unwrap_or_default();
            live_classes = live_classes.saturating_sub(acc.len() as u64);
            let n = qlive_write_closed_acc_run_only(field, acc, &mut class_tmp, &mut class_spool)?;
            closed_class_records += n;
            closed_fields.push(field);
            finalized_fields += 1;
        }
    }
    spill.flush()?;
    class_spool.flush()?;
    let spill_run_count = spill.run_count;
    let spill_runs = std::mem::take(&mut spill.runs);
    let class_metas = std::mem::take(&mut class_spool.metas);
    drop(class_spool);
    drop(spill);

    eprintln!(
        "q live frontier-spill-strong generation done raw={} emitted_class_records={} local_chain_masked={} finalized_fields={} closed_class_records={} spilled_records={} cold_direct_records={} spill_events={} spill_runs={} parents_cold_ever={} spill_file_records={} live_classes_end={} elapsed={:.1}s",
        stats.raw,
        stats.emitted_records,
        stats.local_chain_masked,
        finalized_fields,
        closed_class_records,
        spilled_records,
        cold_direct_records,
        spill_events,
        spill_run_count,
        parents_cold_ever,
        spilled_records,
        live_classes,
        start.elapsed().as_secs_f64()
    );

    eprintln!(
        "q live frontier-spill-strong materialize start class_records={} fields={}",
        closed_class_records,
        class_metas.len()
    );
    let lookup_start = Instant::now();
    let qid_lookup = build_qid_lookup(q_index);
    eprintln!(
        "built qid lookup bytes={} elapsed={:.3}s",
        qid_lookup.len() * std::mem::size_of::<u32>(),
        lookup_start.elapsed().as_secs_f64()
    );
    let finished = qlive_materialize_class_and_spill_runs(
        &class_spool_path,
        class_metas,
        &spill_path,
        spill_runs,
        closed_fields,
        &qid_lookup,
        output_prefix,
        q_index.states.len(),
    )?;
    let _ = std::fs::remove_file(&class_spool_path);
    let _ = std::fs::remove_file(&spill_path);

    println!(
        "q_external_live_frontier_spill_strong raw_pairs={}",
        stats.raw
    );
    println!(
        "q_external_live_frontier_spill_strong emitted_class_records={}",
        stats.emitted_records
    );
    println!(
        "q_external_live_frontier_spill_strong local_chain_masked={}",
        stats.local_chain_masked
    );
    println!("q_external_live_frontier_spill_strong closed_class_records={closed_class_records}");
    println!("q_external_live_frontier_spill_strong spilled_records={spilled_records}");
    println!("q_external_live_frontier_spill_strong cold_direct_records={cold_direct_records}");
    println!("q_external_live_frontier_spill_strong spill_events={spill_events}");
    println!(
        "q_external_live_frontier_spill_strong spill_runs={}",
        spill_run_count
    );
    println!("q_external_live_frontier_spill_strong parents_cold_ever={parents_cold_ever}");
    println!("q_external_live_frontier_spill_strong output_states={finished}");
    println!(
        "q_external_live_frontier_spill_strong removed_dupes={}",
        stats.raw.saturating_sub(finished)
    );
    println!(
        "q_external_live_frontier_spill_strong output_u24_membership_bytes={} ({:.3} GiB)",
        finished * 3,
        finished as f64 * 3.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "q_external_live_frontier_spill_strong output_u24_plus_fp16_bytes={} ({:.3} GiB)",
        finished * 5,
        finished as f64 * 5.0 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "q external live frontier-spill-strong step elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(finished)
}
