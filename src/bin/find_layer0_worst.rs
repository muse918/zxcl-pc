use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[path = "../graph.rs"]
mod graph;
#[path = "../layer_store.rs"]
mod layer_store;
#[path = "../piece.rs"]
mod piece;
#[path = "../qstate.rs"]
mod qstate;
#[path = "../state.rs"]
mod state;

use graph::EMPTY_FIELD_ID;
use piece::{mask_to_string, piece_char};
use qstate::{decode_avail_pair, decode_tail5, enumerate_qstates, QChainState};
use state::enumerate_states;

fn usage(program: &str) -> ! {
    eprintln!("usage: {program} <pruned_layer_dir> <mdp_out_dir> [scale=5040] [print_limit=200]");
    eprintln!("example: {program} data/forward_pruned data/one_loop");
    std::process::exit(2);
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn qstate_string(s: QChainState) -> String {
    let (a, b) = decode_avail_pair(s.avail);
    let tail = decode_tail5(s.tail5);
    let tail_s = tail.iter().map(|&p| piece_char(p)).collect::<String>();
    format!(
        "avail={{{},{}}} tail5={} bag_mask=0x{:02x} bag={}",
        piece_char(a), piece_char(b), tail_s, s.bag_mask, mask_to_string(s.bag_mask)
    )
}

fn choose_layer0_block(layer0: &layer_store::FieldChainLayer) -> Result<usize> {
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
        "could not identify layer0 empty-field block: metas_len={} EMPTY_FIELD_ID={} field0_absent",
        layer0.metas.len(),
        EMPTY_FIELD_ID
    );
}

fn read_u16_block(path: &Path, elem_offset: u64, count: usize) -> Result<Vec<u16>> {
    let mut f = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let byte_off = elem_offset.checked_mul(2).context("u16 byte offset overflow")?;
    let byte_len = count.checked_mul(2).context("u16 byte len overflow")?;
    let mut buf = vec![0u8; byte_len];
    f.seek(SeekFrom::Start(byte_off))?;
    f.read_exact(&mut buf)
        .with_context(|| format!("failed to read {} bytes from {} at byte offset {}", byte_len, path.display(), byte_off))?;
    let mut out = Vec::with_capacity(count);
    for ch in buf.chunks_exact(2) {
        out.push(u16::from_le_bytes([ch[0], ch[1]]));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        usage(args.get(0).map(String::as_str).unwrap_or("find_layer0_worst"));
    }

    let pruned_dir = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let scale: u32 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(5040);
    let print_limit: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(200);

    let layer0_prefix = pruned_dir.join("layer0q_pruned");
    let layer0 = layer_store::FieldChainLayer::open_index_only(&layer0_prefix)
        .with_context(|| format!("failed to open layer0 membership prefix {}", layer0_prefix.display()))?;
    let block_idx = choose_layer0_block(&layer0)?;
    let meta = &layer0.metas[block_idx];
    let chains = layer0.read_block_chains(block_idx);
    if chains.len() != meta.count as usize {
        anyhow::bail!("layer0 chain count mismatch: chains={} meta.count={}", chains.len(), meta.count);
    }

    let values_path = out_dir.join("layer0.values.u16.bin");
    let values = read_u16_block(&values_path, meta.chain_offset, meta.count as usize)?;

    eprintln!("enumerating qstates...");
    let state_index = enumerate_states();
    let q_index = enumerate_qstates(&state_index);

    let mut hist = BTreeMap::<u16, u64>::new();
    let mut min_raw = u16::MAX;
    let mut max_raw = 0u16;
    let mut worst = Vec::<(u32, u64, u16, QChainState)>::new();
    let mut best = Vec::<(u32, u64, u16, QChainState)>::new();
    let mut sum_raw = 0u64;
    let mut full_count = 0u64;
    let mut nonzero_count = 0u64;

    for (i, (&qid, &raw)) in chains.iter().zip(values.iter()).enumerate() {
        let Some(&qs) = q_index.states.get(qid as usize) else {
            anyhow::bail!("qid {} out of range {}", qid, q_index.states.len());
        };
        let value_offset = meta.chain_offset + i as u64;
        sum_raw += raw as u64;
        if raw != 0 { nonzero_count += 1; }
        if raw as u32 == scale { full_count += 1; }
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

    let count = chains.len() as u64;
    let den = count * scale as u64;
    let g = gcd_u64(sum_raw, den);

    println!("layer=0 field_id={} block_idx={} block_count={} chain_offset={}", meta.field_id, block_idx, meta.count, meta.chain_offset);
    println!("count={}", count);
    println!("sum_raw={}", sum_raw);
    println!("avg_raw={:.12}", sum_raw as f64 / count as f64);
    println!("avg_prob={:.12}", sum_raw as f64 / den as f64);
    println!("fraction={}/{}", sum_raw / g, den / g);
    println!("min_raw={} min_prob={:.12} worst_count={}", min_raw, min_raw as f64 / scale as f64, worst.len());
    println!("max_raw={} max_prob={:.12} best_count={}", max_raw, max_raw as f64 / scale as f64, best.len());
    println!("nonzero_count={}", nonzero_count);
    println!("full_count={} full_ratio={}/{} = {:.12}", full_count, full_count, count, full_count as f64 / count as f64);

    println!("hist raw,count ascending:");
    for (&raw, &n) in hist.iter() {
        println!("{} {} {:.12}", raw, n, raw as f64 / scale as f64);
    }

    println!("worst qchains{}:", if worst.len() > print_limit { " first_limit" } else { "" });
    for (qid, value_offset, raw, qs) in worst.iter().take(print_limit) {
        println!("raw={} qid={} value_offset={} {}", raw, qid, value_offset, qstate_string(*qs));
    }
    if worst.len() > print_limit {
        println!("... omitted {} more worst qchains", worst.len() - print_limit);
    }

    println!("best qchains{}:", if best.len() > print_limit { " first_limit" } else { "" });
    for (qid, value_offset, raw, qs) in best.iter().take(print_limit) {
        println!("raw={} qid={} value_offset={} {}", raw, qid, value_offset, qstate_string(*qs));
    }
    if best.len() > print_limit {
        println!("... omitted {} more best qchains", best.len() - print_limit);
    }

    Ok(())
}
