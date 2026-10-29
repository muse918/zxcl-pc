use anyhow::{Context, Result};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use crate::graph::{FieldId, PIECE_COUNT};
use crate::layer_store::{FieldChainLayer, FieldChainWriter};
use crate::piece::FULL_BAG;
use crate::rgraph::ReverseGraph;
use crate::state::{decode_queue6, encode_queue6, is_valid_state, state_key, ChainState, StateIndex};

const DEFAULT_FLUSH_PAIRS: usize = 10_000_000;
const DEFAULT_MERGE_FANIN: usize = 128;
const RUN_MAGIC: &[u8; 8] = b"PAIRRUN1";

#[inline]
fn pair_key(field: FieldId, chain_id: u32) -> u64 {
    ((field as u64) << 32) | chain_id as u64
}

#[inline]
fn write_u64_le<W: Write>(w: &mut W, x: u64) -> Result<()> {
    w.write_all(&x.to_le_bytes())?;
    Ok(())
}

#[inline]
fn read_u64_le_from<R: Read>(r: &mut R) -> Result<Option<u64>> {
    let mut buf = [0u8; 8];
    let mut got = 0usize;
    while got < 8 {
        let n = r.read(&mut buf[got..])?;
        if n == 0 {
            if got == 0 { return Ok(None); }
            anyhow::bail!("truncated u64 in run file");
        }
        got += n;
    }
    Ok(Some(u64::from_le_bytes(buf)))
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

fn reverse_chain_state(new_chain: ChainState, out: &mut Vec<(u8, ChainState)>) {
    out.clear();
    let new_q = decode_queue6(new_chain.queue6);
    let revealed = new_q[5];
    let Some(old_bag) = reverse_bag(new_chain.bag_mask, revealed) else { return; };

    for placed in 0u8..PIECE_COUNT as u8 {
        let old_queue6 = prepend_front(placed, new_q);
        let old = ChainState { hold: new_chain.hold, queue6: old_queue6, bag_mask: old_bag };
        if is_valid_state(old) { out.push((placed, old)); }
    }

    let old_front = new_chain.hold;
    let old_queue6 = prepend_front(old_front, new_q);
    for placed in 0u8..PIECE_COUNT as u8 {
        if placed == old_front { continue; }
        let old = ChainState { hold: placed, queue6: old_queue6, bag_mask: old_bag };
        if is_valid_state(old) { out.push((placed, old)); }
    }
}

#[derive(Debug, Clone)]
pub struct ExternalConfig {
    pub flush_pairs: usize,
    pub merge_fanin: usize,
}

impl Default for ExternalConfig {
    fn default() -> Self {
        Self { flush_pairs: DEFAULT_FLUSH_PAIRS, merge_fanin: DEFAULT_MERGE_FANIN }
    }
}

#[derive(Debug, Clone)]
pub struct ExternalRunStats {
    pub input_states: u64,
    pub raw_pairs: u64,
    pub run_unique_pairs: u64,
    pub run_files: u64,
    pub final_unique_pairs: u64,
}

fn write_pair_run(path: &Path, pairs: &mut Vec<u64>) -> Result<u64> {
    if pairs.is_empty() { return Ok(0); }
    pairs.sort_unstable();
    pairs.dedup();
    let count = pairs.len() as u64;
    let file = File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    w.write_all(RUN_MAGIC)?;
    w.write_all(&count.to_le_bytes())?;
    for &x in pairs.iter() {
        write_u64_le(&mut w, x)?;
    }
    w.flush()?;
    Ok(count)
}

struct PairRunReader {
    path: PathBuf,
    reader: BufReader<File>,
    remaining: u64,
    current: Option<u64>,
}

impl PairRunReader {
    fn open(path: PathBuf) -> Result<Self> {
        let file = File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != RUN_MAGIC { anyhow::bail!("unsupported pair run format: {}", path.display()); }
        let mut cnt = [0u8; 8];
        reader.read_exact(&mut cnt)?;
        let remaining = u64::from_le_bytes(cnt);
        let mut out = Self { path, reader, remaining, current: None };
        out.advance()?;
        Ok(out)
    }

    fn advance(&mut self) -> Result<()> {
        if self.remaining == 0 {
            self.current = None;
            return Ok(());
        }
        self.current = read_u64_le_from(&mut self.reader)?;
        if self.current.is_none() { anyhow::bail!("truncated run file {}", self.path.display()); }
        self.remaining -= 1;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapItem {
    key: u64,
    reader_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key).then_with(|| self.reader_idx.cmp(&other.reader_idx))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

fn merge_runs_to_run(inputs: &[PathBuf], output: &Path) -> Result<u64> {
    let mut readers = Vec::<PairRunReader>::with_capacity(inputs.len());
    for p in inputs { readers.push(PairRunReader::open(p.clone())?); }

    let mut heap = BinaryHeap::<Reverse<HeapItem>>::new();
    for (i, r) in readers.iter().enumerate() {
        if let Some(key) = r.current { heap.push(Reverse(HeapItem { key, reader_idx: i })); }
    }

    let file = File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    w.write_all(RUN_MAGIC)?;
    w.write_all(&0u64.to_le_bytes())?; // patched by rewriting file is inconvenient; write temp body then rewrite header below not possible with BufWriter.

    let mut last = None;
    let mut count = 0u64;
    while let Some(Reverse(item)) = heap.pop() {
        let key = item.key;
        if last != Some(key) {
            write_u64_le(&mut w, key)?;
            last = Some(key);
            count += 1;
        }
        let r = &mut readers[item.reader_idx];
        r.advance()?;
        if let Some(next_key) = r.current {
            heap.push(Reverse(HeapItem { key: next_key, reader_idx: item.reader_idx }));
        }
    }
    w.flush()?;
    drop(w);

    // Patch count in header.
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::OpenOptions::new().write(true).open(output)?;
    file.seek(SeekFrom::Start(8))?;
    file.write_all(&count.to_le_bytes())?;
    file.flush()?;
    Ok(count)
}

fn merge_runs_to_field_chain(inputs: &[PathBuf], output_prefix: &Path) -> Result<u64> {
    let mut readers = Vec::<PairRunReader>::with_capacity(inputs.len());
    for p in inputs { readers.push(PairRunReader::open(p.clone())?); }

    let mut heap = BinaryHeap::<Reverse<HeapItem>>::new();
    for (i, r) in readers.iter().enumerate() {
        if let Some(key) = r.current { heap.push(Reverse(HeapItem { key, reader_idx: i })); }
    }

    let mut writer = FieldChainWriter::create(output_prefix)?;
    let mut last = None;
    let mut count = 0u64;

    while let Some(Reverse(item)) = heap.pop() {
        let key = item.key;
        if last != Some(key) {
            writer.push_pair_key(key)?;
            last = Some(key);
            count += 1;
        }
        let r = &mut readers[item.reader_idx];
        r.advance()?;
        if let Some(next_key) = r.current {
            heap.push(Reverse(HeapItem { key: next_key, reader_idx: item.reader_idx }));
        }
    }

    let written = writer.finish()?;
    debug_assert_eq!(count, written);
    Ok(count)
}

fn collect_run_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.exists() { return Ok(out); }
    for ent in fs::read_dir(dir)? {
        let ent = ent?;
        let path = ent.path();
        if path.extension().and_then(|s| s.to_str()) == Some("bin") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn merge_all_runs_to_field_chain(work_dir: &Path, output_prefix: &Path, fanin: usize) -> Result<u64> {
    let mut pass_dir = work_dir.join("runs");
    let mut runs = collect_run_files(&pass_dir)?;
    if runs.is_empty() { anyhow::bail!("no run files found in {}", pass_dir.display()); }
    let fanin = fanin.max(2);
    let mut pass = 0usize;

    while runs.len() > fanin {
        let next_dir = work_dir.join(format!("merge_pass_{pass:03}"));
        fs::create_dir_all(&next_dir)?;
        eprintln!("merge pass {pass}: {} runs -> {}", runs.len(), next_dir.display());
        let mut next_runs = Vec::<PathBuf>::new();
        for (group_idx, group) in runs.chunks(fanin).enumerate() {
            let out = next_dir.join(format!("run_{group_idx:012}.bin"));
            let start = Instant::now();
            let count = merge_runs_to_run(group, &out)?;
            eprintln!("  merged group {group_idx} inputs={} unique={} elapsed={:.1}s", group.len(), count, start.elapsed().as_secs_f64());
            next_runs.push(out);
            for p in group { let _ = fs::remove_file(p); }
        }
        // Try removing previous pass directory.
        let _ = fs::remove_dir_all(&pass_dir);
        pass_dir = next_dir;
        runs = next_runs;
        pass += 1;
    }

    eprintln!("final merge: {} runs -> {}", runs.len(), output_prefix.display());
    let start = Instant::now();
    let count = merge_runs_to_field_chain(&runs, output_prefix)?;
    eprintln!("final merge unique={} elapsed={:.1}s", count, start.elapsed().as_secs_f64());
    for p in &runs { let _ = fs::remove_file(p); }
    Ok(count)
}

fn flush_local_run(work_dir: &Path, run_counter: &AtomicU64, run_unique: &AtomicU64, pairs: &mut Vec<u64>) -> Result<()> {
    if pairs.is_empty() { return Ok(()); }
    let runs_dir = work_dir.join("runs");
    fs::create_dir_all(&runs_dir)?;
    let run_id = run_counter.fetch_add(1, AtomicOrdering::Relaxed);
    let path = runs_dir.join(format!("run_{run_id:012}.bin"));
    let count = write_pair_run(&path, pairs)?;
    run_unique.fetch_add(count, AtomicOrdering::Relaxed);
    pairs.clear();
    Ok(())
}

pub fn sample_external(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    state_index: &StateIndex,
    sample_states: u64,
) -> Result<()> {
    let start = Instant::now();
    let mut processed = 0u64;
    let mut raw = 0u64;
    let mut pairs = Vec::<u64>::new();
    let mut rev_buf = Vec::<(u8, ChainState)>::new();

    'outer: for (bi, meta) in input.metas.iter().enumerate() {
        let chains = input.read_block_chains(bi);
        for cid in chains {
            if processed >= sample_states { break 'outer; }
            let next_chain = state_index.states[cid as usize];
            reverse_chain_state(next_chain, &mut rev_buf);
            for &(placed, old_chain) in &rev_buf {
                let old_cid = *state_index.key_to_id.get(&state_key(old_chain)).expect("old chain missing") as u32;
                for &old_field in rgraph.edges(meta.field_id, placed) {
                    pairs.push(pair_key(old_field, old_cid));
                    raw += 1;
                }
            }
            processed += 1;
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    let unique = pairs.len() as u64;
    let ratio = if raw == 0 { 0.0 } else { unique as f64 / raw as f64 };
    let raw_per_input = if processed == 0 { 0.0 } else { raw as f64 / processed as f64 };
    let est_raw = raw_per_input * input.total_states as f64;
    let est_unique = est_raw * ratio;

    println!("sample input_states={processed}");
    println!("sample raw_pairs={raw}");
    println!("sample unique_pairs={unique}");
    println!("sample unique_ratio={ratio:.6}");
    println!("sample raw_per_input={raw_per_input:.3}");
    println!("estimated_full_raw_pairs={:.0}", est_raw);
    println!("estimated_full_unique_pairs={:.0}", est_unique);
    println!("estimated_full_u24_membership_bytes={:.0} ({:.3} GiB)", est_unique * 3.0, est_unique * 3.0 / 1024.0 / 1024.0 / 1024.0);
    println!("estimated_full_u24_plus_fp16_bytes={:.0} ({:.3} GiB)", est_unique * 5.0, est_unique * 5.0 / 1024.0 / 1024.0 / 1024.0);
    eprintln!("sample elapsed={:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

pub fn external_step(
    rgraph: &ReverseGraph,
    input: &FieldChainLayer,
    state_index: &StateIndex,
    output_prefix: &Path,
    work_dir: &Path,
    config: ExternalConfig,
) -> Result<ExternalRunStats> {
    fs::create_dir_all(work_dir)?;
    let runs_dir = work_dir.join("runs");
    if runs_dir.exists() {
        eprintln!("removing old runs dir: {}", runs_dir.display());
        fs::remove_dir_all(&runs_dir)?;
    }
    fs::create_dir_all(&runs_dir)?;

    let start = Instant::now();
    let raw = AtomicU64::new(0);
    let run_unique = AtomicU64::new(0);
    let run_counter = AtomicU64::new(0);
    let done_blocks = AtomicU64::new(0);

    let block_chunk = 8usize;
    eprintln!(
        "external generation blocks={} input_states={} threads={} flush_pairs={} block_chunk={}",
        input.metas.len(), input.total_states, rayon::current_num_threads(), config.flush_pairs, block_chunk
    );

    input.metas.par_chunks(block_chunk).try_for_each(|metas| -> Result<()> {
        let mut local = Vec::<u64>::with_capacity(config.flush_pairs.min(1_000_000));
        let mut rev_buf = Vec::<(u8, ChainState)>::new();

        for meta in metas {
            let start_elem = meta.chain_offset;
            for j in 0..meta.count as u64 {
                let cid = crate::layer_store::read_u24_le_at(&input.chains_bytes, start_elem + j);
                let next_chain = state_index.states[cid as usize];
                reverse_chain_state(next_chain, &mut rev_buf);
                for &(placed, old_chain) in &rev_buf {
                    let old_cid = *state_index.key_to_id.get(&state_key(old_chain)).expect("old chain missing") as u32;
                    for &old_field in rgraph.edges(meta.field_id, placed) {
                        local.push(pair_key(old_field, old_cid));
                        raw.fetch_add(1, AtomicOrdering::Relaxed);
                        if local.len() >= config.flush_pairs {
                            flush_local_run(work_dir, &run_counter, &run_unique, &mut local)?;
                        }
                    }
                }
            }
            let d = done_blocks.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if d % 100 == 0 || d == input.metas.len() as u64 {
                eprintln!(
                    "  external gen blocks_done={}/{} raw={} run_unique={} runs={} elapsed={:.1}s",
                    d,
                    input.metas.len(),
                    raw.load(AtomicOrdering::Relaxed),
                    run_unique.load(AtomicOrdering::Relaxed),
                    run_counter.load(AtomicOrdering::Relaxed),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        flush_local_run(work_dir, &run_counter, &run_unique, &mut local)?;
        Ok(())
    })?;

    let raw_pairs = raw.load(AtomicOrdering::Relaxed);
    let run_unique_pairs = run_unique.load(AtomicOrdering::Relaxed);
    let run_files = run_counter.load(AtomicOrdering::Relaxed);
    eprintln!(
        "external run generation done raw={} run_unique={} runs={} elapsed={:.1}s",
        raw_pairs, run_unique_pairs, run_files, start.elapsed().as_secs_f64()
    );

    let merge_start = Instant::now();
    let final_unique = merge_all_runs_to_field_chain(work_dir, output_prefix, config.merge_fanin)?;
    eprintln!("external final merge done final_unique={} elapsed={:.1}s", final_unique, merge_start.elapsed().as_secs_f64());

    let membership_bytes = final_unique * 3;
    let fp16_bytes = final_unique * 2;
    println!("external output_states={final_unique}");
    println!("external output_u24_membership_bytes={} ({:.3} GiB)", membership_bytes, membership_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("external future_fp16_values_bytes={} ({:.3} GiB)", fp16_bytes, fp16_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("external future_u24_plus_fp16_bytes={} ({:.3} GiB)", membership_bytes + fp16_bytes, (membership_bytes + fp16_bytes) as f64 / 1024.0 / 1024.0 / 1024.0);

    Ok(ExternalRunStats {
        input_states: input.total_states,
        raw_pairs,
        run_unique_pairs,
        run_files,
        final_unique_pairs: final_unique,
    })
}
