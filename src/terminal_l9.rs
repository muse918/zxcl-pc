//! On-demand L9 values. Membership remains authoritative even when no f32 L9
//! layer exists. The per-task cache is charged to the sweep's existing cache
//! allowance; oversized fields use positional reads instead of exceeding it.

use std::collections::{HashMap, VecDeque};
use std::fs::File;

use anyhow::{Context, Result};

use crate::graph::{FieldGraph, FieldId, EMPTY_FIELD_ID, PIECE_COUNT};
use crate::layer_store::FieldChainLayer;
use crate::qgraph::{QForwardGraph, QForwardRec};
use crate::{encode_policy_u16, read_exact_at_file, DenseLookupStats, POLICY_NONE_U16};

const CACHE_ENTRY_BYTES: usize = 512;

fn checked_edges(graph: &FieldGraph, field: FieldId, placed: u8) -> Result<&[FieldId]> {
    anyhow::ensure!(
        (placed as usize) < PIECE_COUNT,
        "invalid L9 placed piece {}",
        placed
    );
    let offsets = graph.offsets.get(field as usize).with_context(|| {
        format!(
            "L9 field {} outside graph domain {}",
            field,
            graph.offsets.len()
        )
    })?;
    graph
        .targets
        .get(offsets[placed as usize] as usize..offsets[placed as usize + 1] as usize)
        .context("invalid L9 graph edge offsets")
}

fn checked_recs(qfwd: &QForwardGraph, qid: u32, placed: u8) -> Result<&[QForwardRec]> {
    anyhow::ensure!(
        (placed as usize) < PIECE_COUNT,
        "invalid L9 placed piece {}",
        placed
    );
    let offsets = qfwd.offsets.get(qid as usize).with_context(|| {
        format!(
            "L9 qid {} outside qforward domain {}",
            qid,
            qfwd.offsets.len()
        )
    })?;
    qfwd.records
        .get(offsets[placed as usize] as usize..offsets[placed as usize + 1] as usize)
        .context("invalid L9 qforward record offsets")
}

fn checked_qid(qid: u32, qfwd: &QForwardGraph, terminal: &[[f32; PIECE_COUNT]]) -> Result<()> {
    anyhow::ensure!(
        terminal.len() == qfwd.offsets.len(),
        "L9 terminal table length {} != qforward domain {}",
        terminal.len(),
        qfwd.offsets.len()
    );
    anyhow::ensure!(
        (qid as usize) < qfwd.offsets.len(),
        "L9 qid {} outside qforward domain {}",
        qid,
        qfwd.offsets.len()
    );
    anyhow::ensure!(
        (qid as usize) < terminal.len(),
        "L9 qid {} outside terminal table {}",
        qid,
        terminal.len()
    );
    Ok(())
}

pub(crate) fn enabled_from_env() -> bool {
    std::env::var("FULL_MDP_L9_DIRECT").is_ok_and(|value| value == "1")
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode<'a> {
    Greedy(&'a [[u8; 2]]),
    Fixed(&'a File),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Source<'a> {
    pub layer: &'a FieldChainLayer,
    pub graph: &'a FieldGraph,
    pub qfwd: &'a QForwardGraph,
    pub terminal_reset_avg: &'a [[f32; PIECE_COUNT]],
    pub mode: Mode<'a>,
}

impl Source<'_> {
    fn validate_shapes(&self) -> Result<()> {
        let domain = self.qfwd.offsets.len();
        anyhow::ensure!(
            self.terminal_reset_avg.len() == domain,
            "L9 terminal table length {} != qforward domain {}",
            self.terminal_reset_avg.len(),
            domain
        );
        if let Mode::Greedy(legal) = self.mode {
            anyhow::ensure!(
                legal.len() == domain,
                "L9 legal table length {} != qforward domain {}",
                legal.len(),
                domain
            );
        }
        Ok(())
    }

    fn validate_metadata(&self) -> Result<()> {
        self.validate_shapes()?;
        let domain = self.qfwd.offsets.len();
        let mut end = 0u64;
        let mut previous_field = None;
        for meta in &self.layer.metas {
            anyhow::ensure!(
                (meta.field_id as usize) < self.graph.offsets.len(),
                "L9 field {} outside graph domain {}",
                meta.field_id,
                self.graph.offsets.len()
            );
            anyhow::ensure!(
                previous_field.is_none_or(|field| field < meta.field_id),
                "L9 metadata fields are not strictly sorted"
            );
            previous_field = Some(meta.field_id);
            anyhow::ensure!(
                meta.chain_offset == end,
                "L9 membership offsets are not contiguous"
            );
            anyhow::ensure!(
                (meta.count as usize) <= domain,
                "L9 field {} membership count {} exceeds qid domain {}",
                meta.field_id,
                meta.count,
                domain
            );
            end = end
                .checked_add(meta.count as u64)
                .context("L9 membership count overflow")?;
            // This also checks every graph offset before any legacy L9 kernel
            // can use its unchecked graph accessor after validate_membership.
            Finish::new(self.graph, meta.field_id)?;
        }
        anyhow::ensure!(
            end == self.layer.total_states,
            "L9 membership total count mismatch"
        );
        Ok(())
    }
}

fn validate_qid_bytes(
    bytes: &[u8],
    field: FieldId,
    domain: usize,
    previous: &mut Option<u32>,
) -> Result<()> {
    anyhow::ensure!(bytes.len() % 3 == 0, "truncated L9 membership qid");
    for raw in bytes.chunks_exact(3) {
        let qid = crate::layer_store::read_u24_le_at(raw, 0);
        anyhow::ensure!(
            (qid as usize) < domain,
            "L9 membership qid {} outside domain {} for field {}",
            qid,
            domain,
            field
        );
        anyhow::ensure!(
            previous.is_none_or(|prior| prior < qid),
            "L9 membership qids are not strictly sorted and unique for field {}",
            field
        );
        *previous = Some(qid);
    }
    Ok(())
}

fn validate_block_file(
    chains: &File,
    meta: &crate::layer_store::FieldChainBlockMeta,
    domain: usize,
) -> Result<()> {
    // Constant stack scratch is covered by SweepTuning's base task allowance.
    // It also permits complete validation when a block cannot fit in the cache.
    let mut bytes = [0u8; 12 * 1024];
    let mut remaining = (meta.count as u64)
        .checked_mul(3)
        .context("L9 membership size overflow")?;
    let mut offset = meta
        .chain_offset
        .checked_mul(3)
        .context("L9 membership offset overflow")?;
    let mut previous = None;
    while remaining != 0 {
        let count = (remaining as usize).min(bytes.len());
        read_exact_at_file(chains, &mut bytes[..count], offset)?;
        validate_qid_bytes(&bytes[..count], meta.field_id, domain, &mut previous)?;
        remaining -= count as u64;
        offset = offset
            .checked_add(count as u64)
            .context("L9 membership offset overflow")?;
    }
    Ok(())
}

/// Validate the membership stream once before invoking a legacy L9 kernel that
/// uses unchecked graph/qforward access. Does not retain a state-sized bitmap.
pub(crate) fn validate_membership(source: Source<'_>) -> Result<()> {
    source.validate_metadata()?;
    for qid in 0..source.qfwd.offsets.len() {
        for piece in 0..PIECE_COUNT {
            checked_recs(source.qfwd, qid as u32, piece as u8)?;
        }
    }
    if let Mode::Greedy(legal) = source.mode {
        for (qid, pieces) in legal.iter().enumerate() {
            for &piece in pieces {
                anyhow::ensure!(
                    piece == 255 || (piece as usize) < PIECE_COUNT,
                    "invalid L9 legal piece {} for qid {}",
                    piece,
                    qid
                );
                if piece != 255 {
                    checked_recs(source.qfwd, qid as u32, piece)?;
                }
            }
        }
    }
    let chains = File::open(&source.layer.chains_path)?;
    let expected = source
        .layer
        .total_states
        .checked_mul(3)
        .context("L9 chains size overflow")?;
    anyhow::ensure!(
        chains.metadata()?.len() == expected,
        "L9 membership file size mismatch"
    );
    for meta in &source.layer.metas {
        validate_block_file(&chains, meta, source.qfwd.offsets.len())?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Finish {
    mask: u8,
    first_edge: [u16; PIECE_COUNT],
}

impl Finish {
    fn new(graph: &FieldGraph, field: FieldId) -> Result<Self> {
        let mut out = Self {
            mask: 0,
            first_edge: [u16::MAX; PIECE_COUNT],
        };
        for piece in 0..PIECE_COUNT {
            if let Some(index) = checked_edges(graph, field, piece as u8)?
                .iter()
                .position(|&target| target == EMPTY_FIELD_ID)
            {
                anyhow::ensure!(
                    index < 256,
                    "L9 terminal edge index {} cannot be encoded in policy",
                    index
                );
                out.mask |= 1 << piece;
                out.first_edge[piece] = index as u16;
            }
        }
        Ok(out)
    }
}

pub(crate) fn finish_mask(graph: &FieldGraph, field: FieldId) -> Result<u8> {
    Ok(Finish::new(graph, field)?.mask)
}

fn greedy_with_finish(
    finish: Finish,
    qfwd: &QForwardGraph,
    legal: &[[u8; 2]],
    terminal_reset_avg: &[[f32; PIECE_COUNT]],
    qid: u32,
) -> Result<(f32, u16)> {
    checked_qid(qid, qfwd, terminal_reset_avg)?;
    anyhow::ensure!(
        legal.len() == qfwd.offsets.len(),
        "L9 legal table length {} != qforward domain {}",
        legal.len(),
        qfwd.offsets.len()
    );
    let available = legal
        .get(qid as usize)
        .context("L9 qid outside legal placement table")?;
    let mut best = 0.0f32;
    let mut code = POLICY_NONE_U16;
    // Preserve availability order and the original strict comparison. In
    // particular, negative/zero terminal candidates never become greedy actions.
    for &placed in available {
        if placed == 255 {
            continue;
        }
        anyhow::ensure!(
            (placed as usize) < PIECE_COUNT,
            "invalid L9 legal piece {} for qid {}",
            placed,
            qid
        );
        if finish.mask & (1 << placed) == 0 {
            continue;
        }
        if checked_recs(qfwd, qid, placed)?.is_empty() {
            continue;
        }
        let candidate = terminal_reset_avg[qid as usize][placed as usize];
        if candidate > best {
            best = candidate;
            code = encode_policy_u16(placed, finish.first_edge[placed as usize] as usize);
        }
    }
    Ok((best, code))
}

pub(crate) fn greedy_action(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    legal: &[[u8; 2]],
    terminal_reset_avg: &[[f32; PIECE_COUNT]],
    field: FieldId,
    qid: u32,
) -> Result<(f32, u16)> {
    greedy_with_finish(
        Finish::new(graph, field)?,
        qfwd,
        legal,
        terminal_reset_avg,
        qid,
    )
}

pub(crate) fn fixed_value(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    terminal_reset_avg: &[[f32; PIECE_COUNT]],
    field: FieldId,
    qid: u32,
    code: u16,
) -> Result<f32> {
    checked_qid(qid, qfwd, terminal_reset_avg)?;
    // Even a NONE action must not conceal an invalid membership field.
    checked_edges(graph, field, 0)?;
    if code == POLICY_NONE_U16 {
        return Ok(0.0);
    }
    let placed = (code >> 8) as u8;
    let edge_index = (code & 0xff) as usize;
    anyhow::ensure!(
        (placed as usize) < PIECE_COUNT,
        "corrupt policy code {:#06x} layer=9 field={} qid={}",
        code,
        field,
        qid
    );
    let edges = checked_edges(graph, field, placed)?;
    anyhow::ensure!(
        edge_index < edges.len(),
        "policy edge index {} out of range {} layer=9 field={} placed={} qid={}",
        edge_index,
        edges.len(),
        field,
        placed,
        qid
    );
    anyhow::ensure!(
        !checked_recs(qfwd, qid, placed)?.is_empty(),
        "policy places piece {} with no reveal records layer=9 field={} qid={}",
        placed,
        field,
        qid
    );
    anyhow::ensure!(
        edges[edge_index] == EMPTY_FIELD_ID,
        "layer 9 policy targets non-terminal field {} (field={} qid={})",
        edges[edge_index],
        field,
        qid
    );
    // This is evaluation of the selected policy, including signed Krylov inputs.
    Ok(terminal_reset_avg[qid as usize][placed as usize])
}

#[derive(Debug)]
struct CachedField {
    qids: Box<[u8]>,
    policy: Option<Box<[u8]>>,
    finish: Finish,
}

impl CachedField {
    fn bytes(&self) -> usize {
        // Includes map/deque buckets, allocator metadata and Finish. Arrays are
        // boxed and cache eviction happens before allocating a replacement.
        CACHE_ENTRY_BYTES + self.qids.len() + self.policy.as_ref().map_or(0, |bytes| bytes.len())
    }

    fn index(&self, qid: u32) -> Option<usize> {
        let mut low = 0;
        let mut high = self.qids.len() / 3;
        while low < high {
            let mid = low + (high - low) / 2;
            let candidate = crate::layer_store::read_u24_le_at(&self.qids, mid as u64);
            if candidate < qid {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low < self.qids.len() / 3
            && crate::layer_store::read_u24_le_at(&self.qids, low as u64) == qid)
            .then_some(low)
    }
}

#[derive(Debug)]
pub(crate) struct Lookup<'a> {
    source: Source<'a>,
    chains: File,
    cache: HashMap<FieldId, CachedField>,
    order: VecDeque<FieldId>,
    cache_bytes: usize,
    cache_cap: usize,
    collect_stats: bool,
    stats: DenseLookupStats,
}

impl<'a> Lookup<'a> {
    pub(crate) fn new(source: Source<'a>, cache_cap: usize, collect_stats: bool) -> Result<Self> {
        // A lookup is constructed for every scheduled L8 group. Full membership
        // validation belongs to the single production validate_membership pass;
        // this constructor remains O(1), with queried blocks checked on misses.
        source.validate_shapes()?;
        if let Mode::Fixed(file) = source.mode {
            let expected = source
                .layer
                .total_states
                .checked_mul(2)
                .context("L9 policy size overflow")?;
            anyhow::ensure!(
                file.metadata()?.len() == expected,
                "L9 policy size mismatch: expected {} bytes",
                expected
            );
        }
        let chains = File::open(&source.layer.chains_path)
            .with_context(|| format!("failed to open {}", source.layer.chains_path.display()))?;
        let expected = source
            .layer
            .total_states
            .checked_mul(3)
            .context("L9 chains size overflow")?;
        anyhow::ensure!(
            chains.metadata()?.len() == expected,
            "L9 membership file size mismatch"
        );
        // Keep half the explicitly planned task-cache allowance available for
        // retained hash/deque capacity and allocation overlap. The active entry
        // charge also reserves 512 bytes per field for metadata. Neither a large
        // field nor a long stream of cold fields can grow this cache unbounded.
        Ok(Self {
            source,
            chains,
            cache: HashMap::new(),
            order: VecDeque::new(),
            cache_bytes: 0,
            cache_cap: cache_cap / 2,
            collect_stats,
            stats: DenseLookupStats::default(),
        })
    }

    fn load_block(&mut self, field: FieldId) -> Result<Option<usize>> {
        let Ok(block) = self
            .source
            .layer
            .metas
            .binary_search_by_key(&field, |meta| meta.field_id)
        else {
            if self.collect_stats {
                self.stats.missing_fields += 1;
            }
            return Ok(None);
        };
        if self.cache.contains_key(&field) {
            if self.collect_stats {
                self.stats.cache_hits += 1;
            }
            return Ok(Some(block));
        }
        if self.collect_stats {
            self.stats.cache_misses += 1;
        }
        let meta = &self.source.layer.metas[block];
        anyhow::ensure!(
            (meta.count as usize) <= self.source.qfwd.offsets.len(),
            "L9 field {} membership count {} exceeds qid domain {}",
            field,
            meta.count,
            self.source.qfwd.offsets.len()
        );
        let end = meta
            .chain_offset
            .checked_add(meta.count as u64)
            .context("L9 membership block end overflow")?;
        anyhow::ensure!(
            end <= self.source.layer.total_states,
            "L9 membership block exceeds total states for field {}",
            field
        );
        let finish = Finish::new(self.source.graph, field)?;
        let qid_bytes = (meta.count as usize)
            .checked_mul(3)
            .context("L9 qid block size overflow")?;
        let policy_bytes = if matches!(self.source.mode, Mode::Fixed(_)) {
            (meta.count as usize)
                .checked_mul(2)
                .context("L9 policy block size overflow")?
        } else {
            0
        };
        let bytes = qid_bytes
            .checked_add(policy_bytes)
            .and_then(|n| n.checked_add(CACHE_ENTRY_BYTES))
            .context("L9 cache size overflow")?;
        if bytes > self.cache_cap {
            validate_block_file(&self.chains, meta, self.source.qfwd.offsets.len())?;
            return Ok(Some(block));
        }
        while self.cache_bytes.saturating_add(bytes) > self.cache_cap {
            let old = self
                .order
                .pop_front()
                .expect("L9 cache accounting mismatch");
            if let Some(entry) = self.cache.remove(&old) {
                self.cache_bytes -= entry.bytes();
                if self.collect_stats {
                    self.stats.evictions += 1;
                }
            }
        }
        let mut qids = vec![0u8; qid_bytes].into_boxed_slice();
        read_exact_at_file(
            &self.chains,
            &mut qids,
            meta.chain_offset
                .checked_mul(3)
                .context("L9 qid offset overflow")?,
        )?;
        validate_qid_bytes(&qids, field, self.source.qfwd.offsets.len(), &mut None)?;
        let policy = if let Mode::Fixed(file) = self.source.mode {
            let mut codes = vec![0u8; policy_bytes].into_boxed_slice();
            read_exact_at_file(
                file,
                &mut codes,
                meta.chain_offset
                    .checked_mul(2)
                    .context("L9 policy offset overflow")?,
            )?;
            Some(codes)
        } else {
            None
        };
        let entry = CachedField {
            qids,
            policy,
            finish,
        };
        self.cache_bytes += entry.bytes();
        debug_assert!(self.cache_bytes <= self.cache_cap);
        self.cache.insert(field, entry);
        self.order.push_back(field);
        if self.collect_stats {
            self.stats.sparse_blocks += 1;
            self.stats.materialized_states += meta.count as u64;
        }
        Ok(Some(block))
    }

    fn uncached_index(&self, block: usize, qid: u32) -> Result<Option<usize>> {
        let meta = &self.source.layer.metas[block];
        let mut low = 0usize;
        let mut high = meta.count as usize;
        let mut bytes = [0u8; 3];
        while low < high {
            let mid = low + (high - low) / 2;
            read_exact_at_file(
                &self.chains,
                &mut bytes,
                (meta.chain_offset + mid as u64) * 3,
            )?;
            let candidate = crate::layer_store::read_u24_le_at(&bytes, 0);
            anyhow::ensure!(
                (candidate as usize) < self.source.qfwd.offsets.len(),
                "L9 membership qid {} outside domain {} for field {}",
                candidate,
                self.source.qfwd.offsets.len(),
                meta.field_id
            );
            if candidate == qid {
                return Ok(Some(mid));
            }
            if candidate < qid {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(None)
    }

    fn value_in_loaded_field(&self, field: FieldId, block: usize, qid: u32) -> Result<f32> {
        checked_qid(qid, self.source.qfwd, self.source.terminal_reset_avg)?;
        let (index, finish, cached_code) = if let Some(entry) = self.cache.get(&field) {
            let Some(index) = entry.index(qid) else {
                return Ok(0.0);
            };
            let code = entry
                .policy
                .as_ref()
                .map(|codes| u16::from_le_bytes([codes[index * 2], codes[index * 2 + 1]]));
            (index, entry.finish, code)
        } else {
            let Some(index) = self.uncached_index(block, qid)? else {
                return Ok(0.0);
            };
            (index, Finish::new(self.source.graph, field)?, None)
        };
        match self.source.mode {
            Mode::Greedy(legal) => Ok(greedy_with_finish(
                finish,
                self.source.qfwd,
                legal,
                self.source.terminal_reset_avg,
                qid,
            )?
            .0),
            Mode::Fixed(file) => {
                let code = if let Some(code) = cached_code {
                    code
                } else {
                    let mut bytes = [0u8; 2];
                    let absolute = self.source.layer.metas[block].chain_offset + index as u64;
                    read_exact_at_file(file, &mut bytes, absolute * 2)?;
                    u16::from_le_bytes(bytes)
                };
                fixed_value(
                    self.source.graph,
                    self.source.qfwd,
                    self.source.terminal_reset_avg,
                    field,
                    qid,
                    code,
                )
            }
        }
    }

    pub(crate) fn get(&mut self, field: FieldId, qid: u32) -> Result<f32> {
        checked_qid(qid, self.source.qfwd, self.source.terminal_reset_avg)?;
        let Some(block) = self.load_block(field)? else {
            return Ok(0.0);
        };
        self.value_in_loaded_field(field, block, qid)
    }

    pub(crate) fn sum_recs(&mut self, field: FieldId, recs: &[QForwardRec]) -> Result<(f32, u64)> {
        // Add in reveal order, retaining missing children as zero terms. The
        // caller retains the original reveal count as its denominator.
        for rec in recs {
            checked_qid(
                rec.child_qid,
                self.source.qfwd,
                self.source.terminal_reset_avg,
            )?;
        }
        let Some(block) = self.load_block(field)? else {
            return Ok((0.0, recs.len() as u64));
        };
        let mut sum = 0.0f32;
        let mut missing = 0;
        for rec in recs {
            let value = self.value_in_loaded_field(field, block, rec.child_qid)?;
            sum += value;
            if value == 0.0 {
                missing += 1;
            }
        }
        Ok((sum, missing))
    }

    pub(crate) fn prewarm(&mut self, field: FieldId) -> Result<()> {
        self.load_block(field).map(|_| ())
    }

    pub(crate) fn stats_snapshot(&self) -> DenseLookupStats {
        self.stats.clone()
    }
}

#[cfg(test)]
#[path = "terminal_l9_tests.rs"]
mod tests;
