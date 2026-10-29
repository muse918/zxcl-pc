use anyhow::{bail, Context, Result};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::graph::{FieldId, FieldGraph, NUM_FIELDS, PIECE_COUNT};

const MAGIC: &[u8; 8] = b"RGRPH002";

#[derive(Debug, Clone)]
pub struct ReverseGraph {
    pub offsets: Vec<[u32; PIECE_COUNT + 1]>,
    pub sources: Vec<FieldId>,
}

#[derive(Debug, Clone)]
pub struct ReverseGraphSummary {
    pub total_edges: u64,
    pub edge_count_by_piece: [u64; PIECE_COUNT],
    pub max_degree_by_piece: [u32; PIECE_COUNT],
}

impl ReverseGraph {
    #[inline]
    pub fn edges(&self, field: FieldId, piece: u8) -> &[FieldId] {
        let off = &self.offsets[field as usize];
        let a = off[piece as usize] as usize;
        let b = off[piece as usize + 1] as usize;
        &self.sources[a..b]
    }

    pub fn summary(&self) -> ReverseGraphSummary {
        let mut edge_count_by_piece = [0u64; PIECE_COUNT];
        let mut max_degree_by_piece = [0u32; PIECE_COUNT];

        for off in &self.offsets {
            for p in 0..PIECE_COUNT {
                let degree = off[p + 1] - off[p];
                edge_count_by_piece[p] += degree as u64;
                max_degree_by_piece[p] = max_degree_by_piece[p].max(degree);
            }
        }

        ReverseGraphSummary {
            total_edges: self.sources.len() as u64,
            edge_count_by_piece,
            max_degree_by_piece,
        }
    }
}

pub fn build_reverse_graph(graph: &FieldGraph) -> ReverseGraph {
    let mut counts = vec![[0u32; PIECE_COUNT]; NUM_FIELDS];

    for src in 0..NUM_FIELDS as FieldId {
        for p in 0..PIECE_COUNT as u8 {
            for &dst in graph.edges(src, p) {
                counts[dst as usize][p as usize] += 1;
            }
        }
    }

    let mut offsets = vec![[0u32; PIECE_COUNT + 1]; NUM_FIELDS];
    let mut total = 0u32;

    for f in 0..NUM_FIELDS {
        offsets[f][0] = total;
        for p in 0..PIECE_COUNT {
            total = total.checked_add(counts[f][p]).expect("reverse graph too large");
            offsets[f][p + 1] = total;
        }
    }

    let mut cursor = offsets.iter().map(|x| {
        let mut a = [0u32; PIECE_COUNT];
        a.copy_from_slice(&x[..PIECE_COUNT]);
        a
    }).collect::<Vec<_>>();

    let mut sources = vec![0u32; total as usize];

    for src in 0..NUM_FIELDS as FieldId {
        for p in 0..PIECE_COUNT as u8 {
            for &dst in graph.edges(src, p) {
                let idx = cursor[dst as usize][p as usize] as usize;
                sources[idx] = src;
                cursor[dst as usize][p as usize] += 1;
            }
        }
    }

    for f in 0..NUM_FIELDS {
        for p in 0..PIECE_COUNT {
            debug_assert_eq!(cursor[f][p], offsets[f][p + 1]);
        }
    }

    ReverseGraph { offsets, sources }
}

pub fn save_reverse_graph(path: impl AsRef<Path>, rg: &ReverseGraph) -> Result<()> {
    let path = path.as_ref();
    let file = fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);

    w.write_all(MAGIC)?;
    w.write_all(&(NUM_FIELDS as u32).to_le_bytes())?;
    w.write_all(&(rg.sources.len() as u64).to_le_bytes())?;

    for off in &rg.offsets {
        for &x in off {
            w.write_all(&x.to_le_bytes())?;
        }
    }

    for &src in &rg.sources {
        write_u24_le(&mut w, src)?;
    }

    w.flush()?;
    Ok(())
}

pub fn load_reverse_graph(path: impl AsRef<Path>) -> Result<ReverseGraph> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    parse_reverse_graph(&bytes)
}

pub fn parse_reverse_graph(bytes: &[u8]) -> Result<ReverseGraph> {
    let mut pos = 0usize;
    if bytes.len() < 20 || &bytes[0..8] != MAGIC {
        bail!("unsupported reverse graph format");
    }
    pos += 8;

    let num_fields = read_u32_le(&bytes[pos..pos + 4]) as usize;
    pos += 4;
    if num_fields != NUM_FIELDS {
        bail!("reverse graph NUM_FIELDS mismatch: {num_fields} != {NUM_FIELDS}");
    }

    let total_edges = read_u64_le(&bytes[pos..pos + 8]) as usize;
    pos += 8;

    let offsets_bytes = NUM_FIELDS * (PIECE_COUNT + 1) * 4;
    if pos + offsets_bytes > bytes.len() {
        bail!("truncated reverse graph offsets");
    }

    let mut offsets = Vec::<[u32; PIECE_COUNT + 1]>::with_capacity(NUM_FIELDS);
    for _ in 0..NUM_FIELDS {
        let mut off = [0u32; PIECE_COUNT + 1];
        for slot in &mut off {
            *slot = read_u32_le(&bytes[pos..pos + 4]);
            pos += 4;
        }
        offsets.push(off);
    }

    if pos + total_edges * 3 > bytes.len() {
        bail!("truncated reverse graph sources");
    }

    let mut sources = Vec::<FieldId>::with_capacity(total_edges);
    for _ in 0..total_edges {
        sources.push(read_u24_le(&bytes[pos..pos + 3]));
        pos += 3;
    }

    Ok(ReverseGraph { offsets, sources })
}

#[inline]
fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

#[inline]
fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

#[inline]
fn read_u24_le(bytes: &[u8]) -> u32 {
    (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16)
}

fn write_u24_le<W: Write>(w: &mut W, x: u32) -> Result<()> {
    debug_assert!(x < (1u32 << 24));
    w.write_all(&[(x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8, ((x >> 16) & 0xFF) as u8])?;
    Ok(())
}
