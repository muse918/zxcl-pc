#![allow(dead_code)]

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

pub type FieldId = u32;

pub const PTR_LENGTH: usize = 3;
pub const HASH_LENGTH: usize = 5;
pub const PIECE_COUNT: usize = 7;
pub const NUM_FIELDS: usize = 15_185_706;
pub const EMPTY_FIELD_ID: FieldId = (NUM_FIELDS as FieldId) - 1;
pub const MAX_HASH: u64 = 0xFF_FF_FF_FF_FF;
pub const TWO_LINE_HASH: u64 = 0xF_FF_FF;

#[derive(Debug, Clone)]
pub struct FieldGraph {
    pub hashes: Vec<u64>,
    pub offsets: Vec<[u32; PIECE_COUNT + 1]>,
    pub targets: Vec<FieldId>,
}

#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub num_fields: usize,
    pub total_edges: u64,
    pub edge_count_by_piece: [u64; PIECE_COUNT],
    pub max_degree_by_piece: [u32; PIECE_COUNT],
}

impl FieldGraph {
    pub fn load_graph_bin(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        parse_graph_bin(&bytes)
    }

    #[inline]
    pub fn edges(&self, field: FieldId, piece: u8) -> &[FieldId] {
        debug_assert!((field as usize) < self.offsets.len());
        debug_assert!((piece as usize) < PIECE_COUNT);
        // SAFETY: `field` always comes from graph/layer metadata (< NUM_FIELDS, and
        // `offsets` has exactly one entry per field) and `piece` is in 0..7. The offset
        // pair satisfies a <= b <= targets.len() by construction in parse_graph_bin, so
        // the sub-slice is in bounds. This accessor is on the innermost MDP hot loop;
        // the bounds checks are elided here and preserved as debug_asserts.
        unsafe {
            let off = self.offsets.get_unchecked(field as usize);
            let a = *off.get_unchecked(piece as usize) as usize;
            let b = *off.get_unchecked(piece as usize + 1) as usize;
            self.targets.get_unchecked(a..b)
        }
    }

    #[inline]
    pub fn hash(&self, field: FieldId) -> u64 {
        self.hashes[field as usize]
    }

    pub fn hash_lookup(&self, hash: u64) -> Option<FieldId> {
        self.hashes
            .iter()
            .position(|&h| h == hash)
            .map(|i| i as FieldId)
    }

    pub fn summary(&self) -> GraphSummary {
        let mut edge_count_by_piece = [0u64; PIECE_COUNT];
        let mut max_degree_by_piece = [0u32; PIECE_COUNT];

        for off in &self.offsets {
            for p in 0..PIECE_COUNT {
                let degree = off[p + 1] - off[p];
                edge_count_by_piece[p] += degree as u64;
                max_degree_by_piece[p] = max_degree_by_piece[p].max(degree);
            }
        }

        GraphSummary {
            num_fields: self.hashes.len(),
            total_edges: self.targets.len() as u64,
            edge_count_by_piece,
            max_degree_by_piece,
        }
    }
}

pub fn parse_graph_bin(bytes: &[u8]) -> Result<FieldGraph> {
    let mut pos = 0usize;
    let mut hashes = Vec::<u64>::with_capacity(NUM_FIELDS);
    let mut offsets = Vec::<[u32; PIECE_COUNT + 1]>::with_capacity(NUM_FIELDS);
    let mut targets = Vec::<FieldId>::new();

    for field in 0..NUM_FIELDS {
        if pos + HASH_LENGTH > bytes.len() {
            bail!("truncated graph at field {field} hash");
        }

        let hash = read_u40_be(&bytes[pos..pos + HASH_LENGTH]);
        pos += HASH_LENGTH;
        hashes.push(hash);

        let mut off = [0u32; PIECE_COUNT + 1];
        off[0] = targets.len() as u32;

        for p in 0..PIECE_COUNT {
            if pos >= bytes.len() {
                bail!("truncated graph at field {field} piece {p} degree");
            }
            let degree = bytes[pos] as usize;
            pos += 1;

            if pos + degree * PTR_LENGTH > bytes.len() {
                bail!("truncated graph at field {field} piece {p} targets");
            }

            for _ in 0..degree {
                let target = read_u24_le(&bytes[pos..pos + PTR_LENGTH]);
                pos += PTR_LENGTH;
                if target as usize >= NUM_FIELDS {
                    bail!("invalid target field id {target} at field {field} piece {p}");
                }
                targets.push(target);
            }

            off[p + 1] = targets.len() as u32;
        }

        offsets.push(off);
    }

    if pos != bytes.len() {
        eprintln!(
            "warning: graph.bin has trailing bytes: {}",
            bytes.len() - pos
        );
    }

    Ok(FieldGraph {
        hashes,
        offsets,
        targets,
    })
}

#[inline]
pub fn read_u40_be(bytes: &[u8]) -> u64 {
    debug_assert_eq!(bytes.len(), 5);
    ((bytes[0] as u64) << 32)
        | ((bytes[1] as u64) << 24)
        | ((bytes[2] as u64) << 16)
        | ((bytes[3] as u64) << 8)
        | (bytes[4] as u64)
}

#[inline]
pub fn read_u24_le(bytes: &[u8]) -> u32 {
    debug_assert_eq!(bytes.len(), 3);
    (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16)
}
