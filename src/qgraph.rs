#![allow(dead_code)]

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::piece::{piece_char, PIECE_COUNT};
use crate::qstate::{
    bag_tail_key, q_forward_after_reveal, qchain_key, reverse_qchain_state, QStateIndex,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QForwardRec {
    /// Random reveal piece that produced this child.
    pub reveal: u8,
    pub child_qid: u32,
}

#[derive(Debug, Clone)]
pub struct QForwardGraph {
    /// For parent qid, offsets[qid][p]..offsets[qid][p+1] is the forward range for placed piece p.
    pub offsets: Vec<[u32; PIECE_COUNT + 1]>,
    pub records: Vec<QForwardRec>,
    pub records_by_piece: [u64; PIECE_COUNT],
    pub max_records_per_piece: [usize; PIECE_COUNT],
    pub skipped_invalid_child: u64,
}

impl QForwardGraph {
    #[inline]
    pub fn records_for(&self, parent_qid: u32, placed: u8) -> &[QForwardRec] {
        debug_assert!((parent_qid as usize) < self.offsets.len());
        debug_assert!((placed as usize) < PIECE_COUNT);
        // SAFETY: `parent_qid` is bounded by qstate_count == offsets.len() (validated at
        // layer load) and `placed` is in 0..7. s <= e <= records.len() by construction in
        // build_qforward_graph. Innermost MDP hot loop; checks kept as debug_asserts.
        unsafe {
            let offs = self.offsets.get_unchecked(parent_qid as usize);
            let s = *offs.get_unchecked(placed as usize) as usize;
            let e = *offs.get_unchecked(placed as usize + 1) as usize;
            self.records.get_unchecked(s..e)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QRevClassRec {
    /// Encoded as tail5 | (bag_mask << 15). This is enough to reconstruct
    /// the parent qchain together with an avail id.
    pub bag_tail: u32,
    /// 28-bit mask of possible parent `avail` unordered pairs.
    pub avail_mask: u32,
}

#[derive(Debug, Clone)]
pub struct QRevClassGraph {
    /// For child qid, offsets[qid][p]..offsets[qid][p+1] is the record range for placed piece p.
    pub offsets: Vec<[u32; PIECE_COUNT + 1]>,
    pub records: Vec<QRevClassRec>,
    pub records_by_piece: [u64; PIECE_COUNT],
    pub masks_popcount_by_piece: [u64; PIECE_COUNT],
    pub max_records_per_piece: [usize; PIECE_COUNT],
    pub max_parent_avails_per_record: u32,
}

impl QRevClassGraph {
    #[inline]
    pub fn records_for(&self, child_qid: u32, placed: u8) -> &[QRevClassRec] {
        debug_assert!((child_qid as usize) < self.offsets.len());
        debug_assert!((placed as usize) < PIECE_COUNT);
        let offs = self.offsets[child_qid as usize];
        let s = offs[placed as usize] as usize;
        let e = offs[placed as usize + 1] as usize;
        &self.records[s..e]
    }
}

fn merge_piece_records(mut v: Vec<QRevClassRec>) -> Vec<QRevClassRec> {
    if v.is_empty() { return v; }
    v.sort_unstable_by_key(|r| r.bag_tail);
    let mut out = Vec::<QRevClassRec>::with_capacity(v.len());
    for rec in v {
        if let Some(last) = out.last_mut() {
            if last.bag_tail == rec.bag_tail {
                last.avail_mask |= rec.avail_mask;
                continue;
            }
        }
        out.push(rec);
    }
    out
}

pub fn build_qforward_graph(q_index: &QStateIndex) -> QForwardGraph {
    let mut offsets = Vec::<[u32; PIECE_COUNT + 1]>::with_capacity(q_index.states.len());
    let mut records = Vec::<QForwardRec>::new();
    let mut records_by_piece = [0u64; PIECE_COUNT];
    let mut max_records_per_piece = [0usize; PIECE_COUNT];
    let mut skipped_invalid_child = 0u64;

    for (qid, &parent) in q_index.states.iter().enumerate() {
        let mut offs = [0u32; PIECE_COUNT + 1];
        offs[0] = records.len() as u32;

        for placed in 0u8..PIECE_COUNT as u8 {
            let before = records.len();
            for reveal in 0u8..PIECE_COUNT as u8 {
                let Some(child) = q_forward_after_reveal(parent, placed, reveal) else { continue; };
                let key = qchain_key(child);
                let Some(&child_qid) = q_index.key_to_id.get(&key) else {
                    skipped_invalid_child += 1;
                    continue;
                };
                records.push(QForwardRec { reveal, child_qid: child_qid as u32 });
            }
            let n = records.len() - before;
            records_by_piece[placed as usize] += n as u64;
            max_records_per_piece[placed as usize] = max_records_per_piece[placed as usize].max(n);
            offs[placed as usize + 1] = records.len() as u32;
        }

        offsets.push(offs);
        if (qid + 1) % 100_000 == 0 || qid + 1 == q_index.states.len() {
            eprintln!(
                "  qforward graph build qids_done={}/{} records={}",
                qid + 1,
                q_index.states.len(),
                records.len()
            );
        }
    }

    QForwardGraph { offsets, records, records_by_piece, max_records_per_piece, skipped_invalid_child }
}

pub fn build_qrev_class_graph(q_index: &QStateIndex) -> QRevClassGraph {
    let mut offsets = Vec::<[u32; PIECE_COUNT + 1]>::with_capacity(q_index.states.len());
    let mut records = Vec::<QRevClassRec>::new();
    let mut records_by_piece = [0u64; PIECE_COUNT];
    let mut masks_popcount_by_piece = [0u64; PIECE_COUNT];
    let mut max_records_per_piece = [0usize; PIECE_COUNT];
    let mut max_parent_avails_per_record = 0u32;

    let mut rev_buf = Vec::new();

    for (qid, &child) in q_index.states.iter().enumerate() {
        let mut by_piece: [Vec<QRevClassRec>; PIECE_COUNT] = std::array::from_fn(|_| Vec::new());

        reverse_qchain_state(child, &mut rev_buf);
        for &(placed, parent_q) in &rev_buf {
            let parent_key = qchain_key(parent_q);
            if !q_index.key_to_id.contains_key(&parent_key) {
                continue;
            }
            let bt = bag_tail_key(parent_q);
            let bit = 1u32 << parent_q.avail;
            by_piece[placed as usize].push(QRevClassRec { bag_tail: bt, avail_mask: bit });
        }

        let mut offs = [0u32; PIECE_COUNT + 1];
        offs[0] = records.len() as u32;
        for p in 0..PIECE_COUNT {
            let merged = merge_piece_records(std::mem::take(&mut by_piece[p]));
            max_records_per_piece[p] = max_records_per_piece[p].max(merged.len());
            records_by_piece[p] += merged.len() as u64;
            for rec in merged {
                let pc = rec.avail_mask.count_ones();
                masks_popcount_by_piece[p] += pc as u64;
                max_parent_avails_per_record = max_parent_avails_per_record.max(pc);
                records.push(rec);
            }
            offs[p + 1] = records.len() as u32;
        }
        offsets.push(offs);

        if (qid + 1) % 100_000 == 0 || qid + 1 == q_index.states.len() {
            eprintln!(
                "  qrev graph build qids_done={}/{} records={}",
                qid + 1,
                q_index.states.len(),
                records.len()
            );
        }
    }

    QRevClassGraph {
        offsets,
        records,
        records_by_piece,
        masks_popcount_by_piece,
        max_records_per_piece,
        max_parent_avails_per_record,
    }
}

pub fn print_qforward_graph_stats(q_index: &QStateIndex, g: &QForwardGraph) {
    println!("qforward_graph qstates={}", q_index.states.len());
    println!("qforward_graph records={}", g.records.len());
    println!("qforward_graph estimated_record_bytes={}", g.records.len() as u64 * 8);
    println!("qforward_graph estimated_offsets_bytes={}", g.offsets.len() as u64 * 8 * 4);
    println!("qforward_graph skipped_invalid_child={}", g.skipped_invalid_child);
    for p in 0..PIECE_COUNT {
        println!(
            "qforward_graph piece={} records={} max_records_per_qid={}",
            piece_char(p as u8),
            g.records_by_piece[p],
            g.max_records_per_piece[p]
        );
    }
}

pub fn print_qrev_class_graph_stats(q_index: &QStateIndex, g: &QRevClassGraph) {
    println!("qrev_graph qstates={}", q_index.states.len());
    println!("qrev_graph records={}", g.records.len());
    println!("qrev_graph estimated_record_bytes={}", g.records.len() as u64 * 8);
    println!("qrev_graph estimated_offsets_bytes={}", g.offsets.len() as u64 * 8 * 4);
    println!("qrev_graph max_parent_avails_per_record={}", g.max_parent_avails_per_record);
    for p in 0..PIECE_COUNT {
        let recs = g.records_by_piece[p];
        let pop = g.masks_popcount_by_piece[p];
        let avg = if recs == 0 { 0.0 } else { pop as f64 / recs as f64 };
        println!(
            "qrev_graph piece={} records={} parent_avail_bits={} avg_mask_popcount={:.6} max_records_per_qid={}",
            piece_char(p as u8),
            recs,
            pop,
            avg,
            g.max_records_per_piece[p]
        );
    }
}

pub fn build_and_print_qrev_class_graph(q_index: &QStateIndex) -> Result<QRevClassGraph> {
    let g = build_qrev_class_graph(q_index);
    print_qrev_class_graph_stats(q_index, &g);
    Ok(g)
}

#[inline]
fn write_u32<W: Write>(w: &mut W, x: u32) -> Result<()> {
    w.write_all(&x.to_le_bytes())?;
    Ok(())
}

#[inline]
fn write_u64<W: Write>(w: &mut W, x: u64) -> Result<()> {
    w.write_all(&x.to_le_bytes())?;
    Ok(())
}

pub fn save_qforward_graph(path: &Path, g: &QForwardGraph) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    w.write_all(b"QFWDGR1\0")?;
    write_u32(&mut w, 1)?;
    write_u32(&mut w, g.offsets.len() as u32)?;
    write_u64(&mut w, g.records.len() as u64)?;
    for offs in &g.offsets {
        for &x in offs { write_u32(&mut w, x)?; }
    }
    for rec in &g.records {
        write_u32(&mut w, rec.child_qid)?;
        w.write_all(&[rec.reveal, 0, 0, 0])?;
    }
    w.flush()?;
    Ok(())
}

pub fn save_qrev_class_graph(path: &Path, g: &QRevClassGraph) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    w.write_all(b"QREVCL1\0")?;
    write_u32(&mut w, 1)?;
    write_u32(&mut w, g.offsets.len() as u32)?;
    write_u64(&mut w, g.records.len() as u64)?;
    for offs in &g.offsets {
        for &x in offs { write_u32(&mut w, x)?; }
    }
    for rec in &g.records {
        write_u32(&mut w, rec.bag_tail)?;
        write_u32(&mut w, rec.avail_mask)?;
    }
    w.flush()?;
    Ok(())
}

#[inline]
fn read_u32_at(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() { bail!("truncated qgraph u32"); }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_u64_at(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos + 8 > bytes.len() { bail!("truncated qgraph u64"); }
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

pub fn load_qrev_class_graph(path: &Path) -> Result<QRevClassGraph> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 24 || &bytes[0..8] != b"QREVCL1\0" {
        bail!("unsupported qrevclass graph format: {}", path.display());
    }
    let mut pos = 8usize;
    let version = read_u32_at(&bytes, &mut pos)?;
    if version != 1 { bail!("unsupported qrevclass version {version}"); }
    let qstates = read_u32_at(&bytes, &mut pos)? as usize;
    let record_count = read_u64_at(&bytes, &mut pos)? as usize;
    let expected = 24usize + qstates * (PIECE_COUNT + 1) * 4 + record_count * 8;
    if bytes.len() != expected {
        bail!("qrevclass size mismatch: got {}, expected {}", bytes.len(), expected);
    }
    let mut offsets = Vec::<[u32; PIECE_COUNT + 1]>::with_capacity(qstates);
    for _ in 0..qstates {
        let mut offs = [0u32; PIECE_COUNT + 1];
        for x in &mut offs { *x = read_u32_at(&bytes, &mut pos)?; }
        offsets.push(offs);
    }
    let mut records = Vec::<QRevClassRec>::with_capacity(record_count);
    for _ in 0..record_count {
        let bag_tail = read_u32_at(&bytes, &mut pos)?;
        let avail_mask = read_u32_at(&bytes, &mut pos)?;
        records.push(QRevClassRec { bag_tail, avail_mask });
    }

    let mut records_by_piece = [0u64; PIECE_COUNT];
    let mut masks_popcount_by_piece = [0u64; PIECE_COUNT];
    let mut max_records_per_piece = [0usize; PIECE_COUNT];
    let mut max_parent_avails_per_record = 0u32;
    for offs in &offsets {
        for p in 0..PIECE_COUNT {
            let a = offs[p] as usize;
            let b = offs[p + 1] as usize;
            records_by_piece[p] += (b - a) as u64;
            max_records_per_piece[p] = max_records_per_piece[p].max(b - a);
            for rec in &records[a..b] {
                let pc = rec.avail_mask.count_ones();
                masks_popcount_by_piece[p] += pc as u64;
                max_parent_avails_per_record = max_parent_avails_per_record.max(pc);
            }
        }
    }
    Ok(QRevClassGraph { offsets, records, records_by_piece, masks_popcount_by_piece, max_records_per_piece, max_parent_avails_per_record })
}
