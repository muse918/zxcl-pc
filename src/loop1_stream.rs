//! Sequential compilation of fixed-policy L9 and a bounded whole-field L8 slab.
//!
//! Zero is ID 0; an L9 terminal formula is `zid + 1`. A terminal formula must
//! already contain the ordered legacy f32 average of `reward + reset[child]`.
//! PreRevealIndex is only an index for these shared formulas: this module never
//! moves the random-reveal boundary, sorts children, or merges coefficients.
//!
//! Mapping/table byte statistics describe logical payloads. Separate capacity
//! statistics charge retained Vec capacities and conservative HashMap bucket
//! storage, including overlapping tables during rehash. They exclude allocator
//! bookkeeping, page rounding, and borrowed graph/index data; none is an RSS
//! limit. The mapping cap covers both retained state maps when building L8.
//! The cache cap covers scan buffers, a terminal-use bitmap, and encoded target
//! memberships; fixed 8 KiB index-validation and three-byte lookup stack scratch
//! are excluded. The temporary average interner is separately bounded by its
//! entry count and is discarded after compilation. No existing giant input is
//! hashed. Keep inputs read-only: dev/inode/size/mtime/ctime and requested-path
//! bindings are checked at both ends of construction and before each apply.
//! Linux mutation notifications also reject writes inside one timestamp tick.
//! Capacity accounting reserves 4 KiB per retained Linux notification watcher
//! for its small kernel bookkeeping; this estimate is not an RSS guarantee.
//! Immutable borrows bind the in-memory graph/index/formulas for the cache's
//! lifetime; metadata checks cannot certify hostile writes with restored stats.

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::fs::{self, File, Metadata};
use std::mem::size_of;
use std::ops::Range;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(target_os = "linux")]
use std::{
    ffi::CString,
    os::fd::{AsRawFd, FromRawFd},
    os::unix::ffi::OsStrExt,
};

use crate::graph::{FieldGraph, FieldId, EMPTY_FIELD_ID, PIECE_COUNT};
use crate::layer_store::{FieldChainBlockMeta, FieldChainLayer};
use crate::pre_reveal::PreRevealIndex;
use crate::qgraph::{QForwardGraph, QForwardRec};
use crate::qid_rank::QidRankIndex;
use crate::qstate::QStateIndex;
use crate::DenseLookupStats;

const SCAN_STATES: usize = 64 * 1024;
const INDEX_MAGIC: &[u8; 8] = b"FCHNIDX1";

/// Formula-index interface also permits independent small compiler fixtures.
/// Reveal children must be in exactly the original qforward record order.
pub(crate) trait TerminalFormulaIndex: Any + Sync {
    fn terminal_count(&self) -> usize;
    fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32>;
    fn reveal_qids(&self, zid: u32) -> &[u32];
}

impl TerminalFormulaIndex for PreRevealIndex {
    fn terminal_count(&self) -> usize {
        self.len()
    }
    fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        PreRevealIndex::zid_for_placement(self, qid, placed)
    }
    fn reveal_qids(&self, zid: u32) -> &[u32] {
        PreRevealIndex::reveal_qids(self, zid)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Loop1Limits {
    pub max_mapping_bytes: u64,
    pub max_unique_averages: usize,
    pub max_cache_bytes: usize,
}

impl Default for Loop1Limits {
    fn default() -> Self {
        Self {
            max_mapping_bytes: 512 * 1024 * 1024,
            max_unique_averages: 2_000_000,
            max_cache_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Loop1Stats {
    pub scanned_states: u64,
    pub membership_bytes_read: u64,
    pub policy_bytes_read: u64,
    pub index_bytes_read: u64,
    pub membership_read_calls: u64,
    pub policy_read_calls: u64,
    pub index_read_calls: u64,
    pub mapping_bytes: u64,
    pub average_bytes: u64,
    pub membership_rank_bytes: u64,
    pub rank_build_peak_bytes: usize,
    pub rank_construction_seconds: f64,
    pub table_bytes: u64,
    pub retained_capacity_bytes: u64,
    pub compiler_peak_capacity_bytes: u64,
    pub interner_peak_capacity_bytes: u64,
    pub unique_averages: usize,
    pub unique_terminal_leaves: usize,
    pub retained_edges: u64,
    pub interner_peak_logical_bytes: u64,
    pub cache_peak_bytes: usize,
    pub missing_next_children: u64,
    pub construction_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    device: u64,
    inode: u64,
    size: u64,
    modified: i64,
    modified_ns: i64,
    changed: i64,
    changed_ns: i64,
    mode: u32,
}

impl FileStamp {
    fn of(meta: &Metadata) -> Self {
        Self {
            device: meta.dev(),
            inode: meta.ino(),
            size: meta.len(),
            modified: meta.mtime(),
            modified_ns: meta.mtime_nsec(),
            changed: meta.ctime(),
            changed_ns: meta.ctime_nsec(),
            mode: meta.mode(),
        }
    }
}

struct BoundFile {
    request: PathBuf,
    canonical: PathBuf,
    file: File,
    stamp: FileStamp,
    request_stamp: FileStamp,
    #[cfg(target_os = "linux")]
    mutation_watch: InputMutationWatch,
}

/// Metadata alone can miss a same-size write inside one filesystem clock
/// tick. Keep this queue unread: any mutation (or notification overflow) then
/// remains visible for the whole compiled object's lifetime. poll does not
/// consume notifications, so concurrent read-only checks cannot lose events.
#[cfg(target_os = "linux")]
pub(crate) struct InputMutationWatch(File);

#[cfg(target_os = "linux")]
impl InputMutationWatch {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let raw = unsafe { libc::inotify_init1(libc::IN_CLOEXEC | libc::IN_NONBLOCK) };
        ensure!(
            raw >= 0,
            "create input mutation watch: {}",
            std::io::Error::last_os_error()
        );
        let file = unsafe { File::from_raw_fd(raw) };
        let path_c =
            CString::new(path.as_os_str().as_bytes()).context("input path contains NUL")?;
        let mask = libc::IN_MODIFY
            | libc::IN_ATTRIB
            | libc::IN_CLOSE_WRITE
            | libc::IN_MOVE_SELF
            | libc::IN_DELETE_SELF
            | libc::IN_UNMOUNT;
        let descriptor =
            unsafe { libc::inotify_add_watch(file.as_raw_fd(), path_c.as_ptr(), mask) };
        ensure!(
            descriptor >= 0,
            "watch input {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        Ok(Self(file))
    }

    pub(crate) fn check(&self, path: &Path) -> Result<()> {
        let mut poll = libc::pollfd {
            fd: self.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let result = unsafe { libc::poll(&mut poll, 1, 0) };
            if result < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error).context("poll input mutation watch");
            }
            ensure!(
                result == 0 && poll.revents == 0,
                "input changed (mutation observed): {}",
                path.display()
            );
            return Ok(());
        }
    }
}

impl BoundFile {
    fn heap_bytes(&self) -> u64 {
        let bytes = (self.request.capacity() + self.canonical.capacity()) as u64;
        #[cfg(target_os = "linux")]
        let bytes = bytes + 4_096;
        bytes
    }
    fn open(path: &Path) -> Result<Self> {
        let request = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let canonical = fs::canonicalize(&request)?;
        #[cfg(target_os = "linux")]
        let mutation_watch = InputMutationWatch::open(&canonical)?;
        // Start observing mutations before capturing either path or handle
        // metadata, so writes during initialization invalidate the binding.
        let request_stamp = FileStamp::of(
            &fs::symlink_metadata(&request)
                .with_context(|| format!("stat input {}", request.display()))?,
        );
        let file =
            File::open(&request).with_context(|| format!("open input {}", request.display()))?;
        let meta = file.metadata()?;
        ensure!(
            meta.is_file(),
            "input is not a regular file: {}",
            request.display()
        );
        let out = Self {
            request,
            canonical,
            file,
            stamp: FileStamp::of(&meta),
            request_stamp,
            #[cfg(target_os = "linux")]
            mutation_watch,
        };
        out.check()?;
        Ok(out)
    }

    fn check(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        self.mutation_watch.check(&self.request)?;
        ensure!(
            FileStamp::of(&self.file.metadata()?) == self.stamp,
            "input handle changed: {}",
            self.request.display()
        );
        ensure!(
            FileStamp::of(
                &fs::metadata(&self.request)
                    .with_context(|| format!("input path changed: {}", self.request.display()))?
            ) == self.stamp,
            "input path changed: {}",
            self.request.display()
        );
        ensure!(
            FileStamp::of(&fs::symlink_metadata(&self.request)?) == self.request_stamp
                && fs::canonicalize(&self.request)? == self.canonical,
            "input path or symlink binding changed: {}",
            self.request.display()
        );
        Ok(())
    }
}

struct LayerInputs {
    index: BoundFile,
    membership: BoundFile,
    policy: BoundFile,
}

impl LayerInputs {
    fn heap_bytes(&self) -> u64 {
        self.index.heap_bytes() + self.membership.heap_bytes() + self.policy.heap_bytes()
    }
    fn open(layer: &FieldChainLayer, policy: &Path) -> Result<Self> {
        Ok(Self {
            index: BoundFile::open(&layer.index_path)?,
            membership: BoundFile::open(&layer.chains_path)?,
            policy: BoundFile::open(policy)?,
        })
    }
    fn check(&self) -> Result<()> {
        self.index.check()?;
        self.membership.check()?;
        self.policy.check()
    }
}

#[derive(Clone, Copy)]
enum ReadKind {
    Membership,
    Policy,
    Index,
}

fn read_at(
    file: &File,
    bytes: &mut [u8],
    offset: u64,
    kind: ReadKind,
    stats: &mut Loop1Stats,
) -> Result<()> {
    let mut done = 0;
    while done < bytes.len() {
        let at = offset
            .checked_add(done as u64)
            .context("input byte offset overflow")?;
        let n = match file.read_at(&mut bytes[done..], at) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        match kind {
            ReadKind::Membership => {
                stats.membership_read_calls += 1;
                stats.membership_bytes_read += n as u64;
            }
            ReadKind::Policy => {
                stats.policy_read_calls += 1;
                stats.policy_bytes_read += n as u64;
            }
            ReadKind::Index => {
                stats.index_read_calls += 1;
                stats.index_bytes_read += n as u64;
            }
        }
        ensure!(n != 0, "truncated input at byte offset {at}");
        done += n;
    }
    Ok(())
}

fn u24(raw: &[u8]) -> u32 {
    raw[0] as u32 | (raw[1] as u32) << 8 | (raw[2] as u32) << 16
}
fn u32le(raw: &[u8]) -> u32 {
    u32::from_le_bytes(raw.try_into().unwrap())
}
fn u64le(raw: &[u8]) -> u64 {
    u64::from_le_bytes(raw.try_into().unwrap())
}

fn checked_edges(graph: &FieldGraph, field: FieldId, placed: u8) -> Result<&[FieldId]> {
    ensure!(
        (placed as usize) < PIECE_COUNT,
        "invalid policy placed piece {placed}"
    );
    let offsets = graph
        .offsets
        .get(field as usize)
        .context("field outside graph domain")?;
    let start = offsets[placed as usize] as usize;
    let end = offsets[placed as usize + 1] as usize;
    ensure!(start <= end, "invalid graph edge offsets for field {field}");
    graph
        .targets
        .get(start..end)
        .context("graph edge offsets outside target array")
}

fn checked_recs(qfwd: &QForwardGraph, qid: u32, placed: u8) -> Result<&[QForwardRec]> {
    ensure!(
        (placed as usize) < PIECE_COUNT,
        "invalid policy placed piece {placed}"
    );
    let offsets = qfwd
        .offsets
        .get(qid as usize)
        .context("qid outside qforward domain")?;
    let (start, end) = (
        offsets[placed as usize] as usize,
        offsets[placed as usize + 1] as usize,
    );
    ensure!(
        start <= end,
        "invalid qforward record offsets for qid {qid}"
    );
    let recs = qfwd
        .records
        .get(start..end)
        .context("qforward offsets outside record array")?;
    ensure!(
        !recs.is_empty(),
        "policy places piece {placed} with no reveal records for qid {qid}"
    );
    ensure!(
        recs.len() <= PIECE_COUNT,
        "too many reveal records for qid {qid}"
    );
    let mut prior = None;
    for rec in recs {
        ensure!(
            (rec.reveal as usize) < PIECE_COUNT && prior.is_none_or(|p| p < rec.reveal),
            "qforward reveals are not strictly ordered for qid {qid}"
        );
        ensure!(
            (rec.child_qid as usize) < qfwd.offsets.len(),
            "qforward child outside qid domain"
        );
        prior = Some(rec.reveal);
    }
    Ok(recs)
}

fn validate_layer(
    layer: &FieldChainLayer,
    files: &LayerInputs,
    graph: &FieldGraph,
    domain: usize,
    stats: &mut Loop1Stats,
) -> Result<()> {
    let expected_index = (layer.metas.len() as u64)
        .checked_mul(16)
        .and_then(|n| n.checked_add(24))
        .context("index size overflow")?;
    ensure!(
        files.index.stamp.size == expected_index,
        "layer index file size mismatch"
    );
    ensure!(
        files.membership.stamp.size
            == layer
                .total_states
                .checked_mul(3)
                .context("membership size overflow")?,
        "membership file size mismatch"
    );
    ensure!(
        files.policy.stamp.size
            == layer
                .total_states
                .checked_mul(2)
                .context("policy size overflow")?,
        "policy file size mismatch"
    );
    let mut header = [0; 24];
    read_at(&files.index.file, &mut header, 0, ReadKind::Index, stats)?;
    ensure!(
        &header[..8] == INDEX_MAGIC && u32le(&header[8..12]) == 1,
        "unsupported layer index format"
    );
    ensure!(
        u32le(&header[12..16]) as usize == layer.metas.len()
            && u64le(&header[16..24]) == layer.total_states,
        "layer index header differs from loaded metadata"
    );
    let mut scratch = [0; 16 * 512];
    let mut end = 0u64;
    let mut prior = None;
    for start in (0..layer.metas.len()).step_by(512) {
        let count = (layer.metas.len() - start).min(512);
        read_at(
            &files.index.file,
            &mut scratch[..count * 16],
            24 + start as u64 * 16,
            ReadKind::Index,
            stats,
        )?;
        for i in 0..count {
            let meta = &layer.metas[start + i];
            let raw = &scratch[i * 16..i * 16 + 16];
            ensure!(
                u32le(&raw[..4]) == meta.field_id
                    && u32le(&raw[4..8]) == meta.count
                    && u64le(&raw[8..16]) == meta.chain_offset,
                "layer index differs from loaded metadata at block {}",
                start + i
            );
            ensure!(
                prior.is_none_or(|p| p < meta.field_id),
                "layer fields are not strictly sorted"
            );
            ensure!(
                meta.chain_offset == end,
                "layer field offsets are not contiguous"
            );
            ensure!(
                meta.count as usize <= domain,
                "field membership count exceeds qid domain"
            );
            ensure!(
                (meta.field_id as usize) < graph.offsets.len(),
                "layer field outside graph domain"
            );
            for placed in 0..PIECE_COUNT as u8 {
                for &target in checked_edges(graph, meta.field_id, placed)? {
                    ensure!(
                        target == EMPTY_FIELD_ID || (target as usize) < graph.offsets.len(),
                        "graph target outside field domain"
                    );
                }
            }
            end = end
                .checked_add(meta.count as u64)
                .context("layer state count overflow")?;
            prior = Some(meta.field_id);
        }
    }
    ensure!(
        end == layer.total_states,
        "layer total state count mismatch"
    );
    Ok(())
}

struct Domain<'a> {
    graph: &'a FieldGraph,
    q_index: &'a QStateIndex,
    qfwd: &'a QForwardGraph,
    pre: &'a dyn TerminalFormulaIndex,
}

impl Domain<'_> {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.q_index.states.len() == self.qfwd.offsets.len(),
            "qid/qforward domain mismatch"
        );
        ensure!(
            self.pre.terminal_count() < u32::MAX as usize,
            "terminal IDs exceed u32 domain"
        );
        Ok(())
    }
    fn matches(&self, other: &Domain<'_>) -> bool {
        std::ptr::eq(self.graph, other.graph)
            && std::ptr::eq(self.q_index, other.q_index)
            && std::ptr::eq(self.qfwd, other.qfwd)
            && self.pre.type_id() == other.pre.type_id()
            && std::ptr::eq(
                self.pre as *const dyn TerminalFormulaIndex as *const (),
                other.pre as *const dyn TerminalFormulaIndex as *const (),
            )
    }
}

fn mapping_bytes(count: u64) -> Result<u64> {
    count
        .checked_mul(4)
        .context("state mapping byte count overflow")
}

fn allocate_ids(count: u64) -> Result<Vec<u32>> {
    let count = usize::try_from(count).context("state count exceeds address space")?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(count)
        .context("allocate state mapping")?;
    Ok(ids)
}

struct ScanScratch {
    bytes: Vec<u8>,
    used: Vec<u8>,
    chunk_states: usize,
}

impl ScanScratch {
    fn new(terminal_count: usize, limits: Loop1Limits, stats: &mut Loop1Stats) -> Result<Self> {
        let bitmap_bytes = terminal_count
            .checked_add(7)
            .context("terminal bitmap overflow")?
            / 8;
        let available = limits
            .max_cache_bytes
            .checked_sub(bitmap_bytes)
            .context("cache cap cannot fit terminal-use bitmap")?;
        ensure!(
            available >= 5,
            "cache cap cannot fit one membership/policy scan record"
        );
        let chunk_states = (available / 5).min(SCAN_STATES);
        let scan_bytes = chunk_states * 5;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(scan_bytes)
            .context("allocate scan buffer")?;
        bytes.resize(scan_bytes, 0);
        let mut used = Vec::new();
        used.try_reserve_exact(bitmap_bytes)
            .context("allocate terminal-use bitmap")?;
        used.resize(bitmap_bytes, 0);
        stats.cache_peak_bytes = bitmap_bytes + scan_bytes;
        Ok(Self {
            bytes,
            used,
            chunk_states,
        })
    }
    fn mark(&mut self, id: u32) {
        if id != 0 {
            let z = (id - 1) as usize;
            self.used[z / 8] |= 1 << (z % 8);
        }
    }
    fn used_zids(&self, terminal_count: usize) -> Result<Vec<u32>> {
        let count: usize = self.used.iter().map(|b| b.count_ones() as usize).sum();
        let mut zids = Vec::new();
        zids.try_reserve_exact(count)
            .context("allocate used terminal list")?;
        for z in 0..terminal_count {
            if self.used[z / 8] & (1 << (z % 8)) != 0 {
                zids.push(z as u32);
            }
        }
        Ok(zids)
    }
    fn bytes(&self) -> usize {
        self.bytes.len() + self.used.len()
    }
}

fn read_chunk(
    files: &LayerInputs,
    offset: u64,
    count: usize,
    scratch: &mut ScanScratch,
    stats: &mut Loop1Stats,
) -> Result<()> {
    let (qids, rest) = scratch.bytes.split_at_mut(count * 3);
    read_at(
        &files.membership.file,
        qids,
        offset
            .checked_mul(3)
            .context("membership offset overflow")?,
        ReadKind::Membership,
        stats,
    )?;
    read_at(
        &files.policy.file,
        &mut rest[..count * 2],
        offset.checked_mul(2).context("policy offset overflow")?,
        ReadKind::Policy,
        stats,
    )
}

fn validate_qid(qid: u32, previous: &mut Option<u32>, domain: usize, field: FieldId) -> Result<()> {
    ensure!(
        (qid as usize) < domain,
        "membership qid {qid} outside domain {domain} for field {field}"
    );
    ensure!(
        previous.is_none_or(|p| p < qid),
        "membership qids are not strictly sorted and unique for field {field}"
    );
    *previous = Some(qid);
    Ok(())
}

fn decode_action<'a>(
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    field: FieldId,
    qid: u32,
    code: u16,
) -> Result<(FieldId, &'a [QForwardRec], u8)> {
    let placed = (code >> 8) as u8;
    let edge_index = (code & 255) as usize;
    let edges = checked_edges(graph, field, placed)?;
    let target = *edges.get(edge_index).with_context(|| {
        format!(
            "policy edge index {edge_index} out of range {} for field {field} qid {qid}",
            edges.len()
        )
    })?;
    Ok((target, checked_recs(qfwd, qid, placed)?, placed))
}

pub(crate) struct CompiledL9<'a> {
    domain: Domain<'a>,
    inputs: LayerInputs,
    metas: Vec<FieldChainBlockMeta>,
    state_ids: Vec<u32>,
    membership_ranks: Option<Vec<QidRankIndex>>,
    used_zids: Vec<u32>,
    stats: Loop1Stats,
}

impl<'a> CompiledL9<'a> {
    pub(crate) fn build(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        qfwd: &'a QForwardGraph,
        pre: &'a dyn TerminalFormulaIndex,
        layer: &FieldChainLayer,
        policy_path: &Path,
        limits: Loop1Limits,
    ) -> Result<Self> {
        let started = Instant::now();
        let domain = Domain {
            graph,
            q_index,
            qfwd,
            pre,
        };
        domain.validate()?;
        let bytes = mapping_bytes(layer.total_states)?;
        ensure!(
            bytes <= limits.max_mapping_bytes,
            "L9 mapping byte cap exceeded: {bytes} > {}",
            limits.max_mapping_bytes
        );
        let mut stats = Loop1Stats {
            mapping_bytes: bytes,
            ..Loop1Stats::default()
        };
        let mut scratch = ScanScratch::new(pre.terminal_count(), limits, &mut stats)?;
        let inputs = LayerInputs::open(layer, policy_path)?;
        validate_layer(layer, &inputs, graph, q_index.states.len(), &mut stats)?;
        let mut state_ids = allocate_ids(layer.total_states)?;
        for meta in &layer.metas {
            let mut previous = None;
            let mut position = 0usize;
            while position < meta.count as usize {
                let count = (meta.count as usize - position).min(scratch.chunk_states);
                read_chunk(
                    &inputs,
                    meta.chain_offset + position as u64,
                    count,
                    &mut scratch,
                    &mut stats,
                )?;
                for i in 0..count {
                    let qid = u24(&scratch.bytes[i * 3..i * 3 + 3]);
                    validate_qid(qid, &mut previous, q_index.states.len(), meta.field_id)?;
                    let p = count * 3 + i * 2;
                    let code = u16::from_le_bytes([scratch.bytes[p], scratch.bytes[p + 1]]);
                    let id = if code == u16::MAX {
                        0
                    } else {
                        let (target, recs, placed) =
                            decode_action(graph, qfwd, meta.field_id, qid, code)?;
                        ensure!(
                            target == EMPTY_FIELD_ID,
                            "L9 policy targets non-terminal field {target}"
                        );
                        let z = pre
                            .zid_for_placement(qid, placed)
                            .context("policy placement has no terminal formula")?;
                        ensure!(
                            (z as usize) < pre.terminal_count(),
                            "terminal formula outside domain"
                        );
                        let children = pre.reveal_qids(z);
                        ensure!(
                            children.len() == recs.len()
                                && children.iter().zip(recs).all(|(&q, r)| q == r.child_qid),
                            "terminal formula does not preserve qforward child order for qid {qid}"
                        );
                        z.checked_add(1).context("terminal ID overflow")?
                    };
                    scratch.mark(id);
                    state_ids.push(id);
                    stats.scanned_states += 1;
                }
                position += count;
            }
        }
        ensure!(
            state_ids.len() as u64 == layer.total_states,
            "L9 scan state count mismatch"
        );
        let used_zids = scratch.used_zids(pre.terminal_count())?;
        stats.unique_terminal_leaves = used_zids.len();
        stats.table_bytes = bytes
            + used_zids.len() as u64 * 4
            + layer.metas.len() as u64 * size_of::<FieldChainBlockMeta>() as u64;
        inputs.check()?;
        stats.construction_seconds = started.elapsed().as_secs_f64();
        let mut compiled = Self {
            domain,
            inputs,
            metas: layer.metas.clone(),
            state_ids,
            membership_ranks: None,
            used_zids,
            stats,
        };
        compiled.stats.retained_capacity_bytes = compiled.storage_bytes();
        Ok(compiled)
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.inputs.check()
    }

    /// Check a sparse caller's domain and opened input generations without
    /// requiring any terminal values. Shared rank indexes are mandatory so an
    /// ID-only shortcut never performs fallback membership reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn check_sparse_binding(
        &self,
        graph: &FieldGraph,
        qfwd: &QForwardGraph,
        pre: &dyn TerminalFormulaIndex,
        index: &File,
        membership: &File,
        policy: &File,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        let domain = &self.domain;
        ensure!(
            std::ptr::eq(graph, domain.graph)
                && std::ptr::eq(qfwd, domain.qfwd)
                && pre.type_id() == domain.pre.type_id()
                && std::ptr::eq(
                    pre as *const dyn TerminalFormulaIndex as *const (),
                    domain.pre as *const dyn TerminalFormulaIndex as *const (),
                ),
            "sparse L9 graph/qforward/formula binding differs from prepared domain"
        );
        ensure!(
            self.membership_ranks.is_some(),
            "sparse L9 shortcut requires shared membership-rank indexes"
        );
        for (name, actual, expected) in [
            ("index", index, &self.inputs.index.stamp),
            ("membership", membership, &self.inputs.membership.stamp),
            ("policy", policy, &self.inputs.policy.stamp),
        ] {
            ensure!(
                FileStamp::of(&actual.metadata()?) == *expected,
                "sparse L9 input binding mismatch: {name}"
            );
        }
        Ok(())
    }

    /// A terminal-formula ID lookup with no numeric terminal array, membership
    /// cache allocation, or per-state I/O. Check the retained inputs again
    /// after a batch; inputs must remain immutable throughout compilation.
    pub(crate) fn formula_lookup(&self) -> Result<CompiledL9FormulaLookup<'_>> {
        self.check_inputs_unchanged()?;
        ensure!(
            self.membership_ranks.is_some(),
            "L9 formula lookup requires shared membership-rank indexes"
        );
        Ok(CompiledL9FormulaLookup {
            cache: TargetCache::new(self, 0),
            io_stats: Loop1Stats::default(),
        })
    }

    pub(crate) fn stats(&self) -> &Loop1Stats {
        &self.stats
    }
    pub(crate) fn storage_bytes(&self) -> u64 {
        vec_capacity_bytes(&self.state_ids)
            + vec_capacity_bytes(&self.metas)
            + vec_capacity_bytes(&self.used_zids)
            + self.inputs.heap_bytes()
            + self.membership_ranks.as_ref().map_or(0, |ranks| {
                vec_capacity_bytes(ranks)
                    + ranks
                        .iter()
                        .map(|rank| rank.heap_bytes() as u64)
                        .sum::<u64>()
            })
    }
    /// Retain a policy-scoped membership/rank bitmap for each L9 field. The
    /// resulting immutable indexes are shared by all worker lookups, including
    /// workers with a zero task-cache allowance. Default compilation leaves
    /// these disabled so its original retained-memory contract stays intact.
    ///
    /// The retained cap includes bitmap/rank heaps and Vec element payload.
    /// The build-buffer cap covers the largest decoded field plus the bounded
    /// encoded scan buffer. Fixed three-byte endpoint scratch is excluded.
    /// Input versions are checked before any reads and again before commit.
    pub(crate) fn build_rank_index(
        &mut self,
        max_retained_bytes: u64,
        max_build_buffer_bytes: usize,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        if self.membership_ranks.is_some() {
            ensure!(
                self.stats.membership_rank_bytes <= max_retained_bytes,
                "retained membership-rank byte cap exceeded"
            );
            return Ok(());
        }
        let started = Instant::now();
        let mut extra = Loop1Stats::default();
        let mut retained = (self.metas.len() as u64)
            .checked_mul(size_of::<QidRankIndex>() as u64)
            .context("membership-rank metadata byte count overflow")?;
        ensure!(
            retained <= max_retained_bytes,
            "membership-rank retained byte cap exceeded"
        );
        let mut largest_qid_bytes = 0usize;
        // Endpoint-only preflight computes the exact trimmed bitmap budget
        // before allocating a field Vec or any retained bitmap.
        for meta in &self.metas {
            let qid_bytes = (meta.count as usize)
                .checked_mul(4)
                .context("membership-rank decoded field byte count overflow")?;
            largest_qid_bytes = largest_qid_bytes.max(qid_bytes);
            if meta.count == 0 {
                continue;
            }
            ensure!(
                qid_bytes
                    .checked_add(3)
                    .context("membership-rank build byte count overflow")?
                    <= max_build_buffer_bytes,
                "membership-rank build buffer cap cannot fit a decoded field and scan record"
            );
            let mut first_raw = [0u8; 3];
            let mut last_raw = [0u8; 3];
            read_at(
                &self.inputs.membership.file,
                &mut first_raw,
                meta.chain_offset
                    .checked_mul(3)
                    .context("membership-rank first offset overflow")?,
                ReadKind::Membership,
                &mut extra,
            )?;
            read_at(
                &self.inputs.membership.file,
                &mut last_raw,
                (meta.chain_offset + meta.count as u64 - 1)
                    .checked_mul(3)
                    .context("membership-rank last offset overflow")?,
                ReadKind::Membership,
                &mut extra,
            )?;
            let endpoints = [u24(&first_raw), u24(&last_raw)];
            ensure!(
                endpoints[0] <= endpoints[1]
                    && (endpoints[1] as usize) < self.domain.q_index.states.len(),
                "invalid membership-rank field endpoints"
            );
            retained = retained
                .checked_add(QidRankIndex::estimated_heap_bytes(&endpoints) as u64)
                .context("membership-rank retained byte count overflow")?;
            ensure!(
                retained <= max_retained_bytes,
                "membership-rank retained byte cap exceeded"
            );
        }
        let scan_bytes = if largest_qid_bytes == 0 {
            0
        } else {
            ((max_build_buffer_bytes - largest_qid_bytes) / 3).min(SCAN_STATES) * 3
        };
        let mut scan = Vec::new();
        scan.try_reserve_exact(scan_bytes)
            .context("allocate membership-rank scan buffer")?;
        scan.resize(scan_bytes, 0);
        let mut ranks = Vec::new();
        ranks
            .try_reserve_exact(self.metas.len())
            .context("allocate membership-rank field table")?;
        let mut actual_retained = self.metas.len() as u64 * size_of::<QidRankIndex>() as u64;
        for meta in &self.metas {
            let mut qids = Vec::new();
            qids.try_reserve_exact(meta.count as usize)
                .context("allocate membership-rank decoded field")?;
            let mut position = 0usize;
            while position < meta.count as usize {
                let count = (meta.count as usize - position).min(scan_bytes / 3);
                read_at(
                    &self.inputs.membership.file,
                    &mut scan[..count * 3],
                    (meta.chain_offset + position as u64)
                        .checked_mul(3)
                        .context("membership-rank scan offset overflow")?,
                    ReadKind::Membership,
                    &mut extra,
                )?;
                for raw in scan[..count * 3].chunks_exact(3) {
                    qids.push(u24(raw));
                }
                position += count;
            }
            extra.rank_build_peak_bytes =
                extra.rank_build_peak_bytes.max(qids.len() * 4 + scan_bytes);
            let next_retained = actual_retained
                .checked_add(QidRankIndex::estimated_heap_bytes(&qids) as u64)
                .context("membership-rank candidate byte count overflow")?;
            ensure!(
                next_retained <= max_retained_bytes,
                "membership-rank retained byte cap exceeded before allocation"
            );
            let index = QidRankIndex::from_sorted(&qids, self.domain.q_index.states.len())
                .map_err(anyhow::Error::msg)?;
            actual_retained = actual_retained
                .checked_add(index.heap_bytes() as u64)
                .context("membership-rank actual byte count overflow")?;
            ensure!(
                actual_retained <= max_retained_bytes,
                "membership-rank retained byte cap exceeded"
            );
            ranks.push(index);
        }
        ensure!(
            actual_retained == retained,
            "membership-rank preflight budget mismatch"
        );
        self.check_inputs_unchanged()?;
        self.stats.membership_bytes_read += extra.membership_bytes_read;
        self.stats.membership_read_calls += extra.membership_read_calls;
        self.stats.membership_rank_bytes = actual_retained;
        self.stats.rank_build_peak_bytes = extra.rank_build_peak_bytes;
        self.stats.rank_construction_seconds = started.elapsed().as_secs_f64();
        self.stats.construction_seconds += self.stats.rank_construction_seconds;
        self.stats.table_bytes += actual_retained;
        self.membership_ranks = Some(ranks);
        self.stats.retained_capacity_bytes = self.storage_bytes();
        Ok(())
    }
    pub(crate) fn state_ids(&self) -> &[u32] {
        &self.state_ids
    }
    pub(crate) fn state_count(&self) -> usize {
        self.state_ids.len()
    }
    pub(crate) fn metas(&self) -> &[FieldChainBlockMeta] {
        &self.metas
    }
    pub(crate) fn used_terminal_zids(&self) -> &[u32] {
        &self.used_zids
    }
    /// Bind one immutable numeric input for an apply. All source-version and
    /// terminal checks happen here, before constructing per-task lookups.
    pub(crate) fn prepare_f32<'s>(&'s self, terminals: &'s [f32]) -> Result<PreparedL9Values<'s>> {
        self.check_inputs_unchanged()?;
        validate_terminals(terminals, self.domain.pre.terminal_count(), &self.used_zids)?;
        Ok(PreparedL9Values {
            compiled: self,
            terminals,
        })
    }
    pub(crate) fn evaluate_f32(&self, terminals: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.evaluate_into_f32(terminals, &mut out, &mut Vec::new())?;
        Ok(out)
    }
    pub(crate) fn evaluate_into_f32(
        &self,
        terminals: &[f32],
        out: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        validate_terminals(terminals, self.domain.pre.terminal_count(), &self.used_zids)?;
        reserve_output(out, self.state_ids.len())?;
        scratch.clear();
        for &id in &self.state_ids {
            out.push(if id == 0 {
                0.0
            } else {
                terminals[id as usize - 1]
            });
        }
        Ok(())
    }
}

/// A per-apply token. Its immutable borrows keep the terminal values and
/// compiled graph alive while parallel task-local membership caches use them.
/// Inputs must remain read-only throughout the apply, as at construction.
#[derive(Clone, Copy)]
pub(crate) struct PreparedL9Values<'a> {
    compiled: &'a CompiledL9<'a>,
    terminals: &'a [f32],
}

impl std::fmt::Debug for PreparedL9Values<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedL9Values")
            .field("states", &self.compiled.state_count())
            .field("terminals", &self.terminals.len())
            .finish()
    }
}

impl<'a> PreparedL9Values<'a> {
    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.compiled.check_inputs_unchanged()
    }

    /// Bind a sparse traversal's opened L9 sources to this policy-scoped
    /// compiled token. Object identity prevents mixing structurally similar
    /// but distinct immutable domains; raw-handle stamps prevent mixing policy
    /// or membership generations. Shared rank indexes make cap-zero lookups
    /// perform no fallback membership reads during the sparse traversal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn check_sparse_binding(
        &self,
        graph: &FieldGraph,
        qfwd: &QForwardGraph,
        pre: &dyn TerminalFormulaIndex,
        index: &File,
        membership: &File,
        policy: &File,
    ) -> Result<()> {
        self.compiled
            .check_sparse_binding(graph, qfwd, pre, index, membership, policy)
    }

    /// Gather current terminal values in original L9 membership order for the
    /// existing RAM next-value lookup. NONE states remain positive zero.
    pub(crate) fn materialize_f32(self) -> Result<Vec<f32>> {
        self.compiled.evaluate_f32(self.terminals)
    }

    pub(crate) fn materialize_into_f32(self, out: &mut Vec<f32>) -> Result<()> {
        self.compiled
            .evaluate_into_f32(self.terminals, out, &mut Vec::new())
    }

    /// O(1), with no input scan, stat, file open, or cache allocation. The cap
    /// bounds encoded membership payload; a three-byte stack search handles
    /// fields larger than the cap, including cap zero.
    pub(crate) fn lookup(
        self,
        cache_cap_bytes: usize,
        collect_stats: bool,
    ) -> Result<CompiledL9Lookup<'a>> {
        let mut cache = TargetCache::new(self.compiled, cache_cap_bytes);
        cache.collect_stats = collect_stats;
        Ok(CompiledL9Lookup {
            values: self,
            cache,
            io_stats: Loop1Stats::default(),
        })
    }
}

pub(crate) struct CompiledL9FormulaLookup<'a> {
    cache: TargetCache<'a>,
    io_stats: Loop1Stats,
}

impl CompiledL9FormulaLookup<'_> {
    /// Return the original stored-policy terminal zid. An absent field/qid or
    /// POLICY_NONE returns None; this does not coalesce numerically equal
    /// formulas, change child order, or evaluate any floating-point values.
    pub(crate) fn get(&mut self, field: FieldId, qid: u32) -> Result<Option<u32>> {
        ensure!(
            (qid as usize) < self.cache.l9.domain.q_index.states.len(),
            "compiled L9 formula lookup qid {qid} outside domain"
        );
        let id = self.cache.find(field, qid, 0, &mut self.io_stats)?;
        Ok(id.checked_sub(1))
    }

    pub(crate) fn io_stats(&self) -> &Loop1Stats {
        &self.io_stats
    }
}

pub(crate) struct CompiledL9Lookup<'a> {
    values: PreparedL9Values<'a>,
    cache: TargetCache<'a>,
    io_stats: Loop1Stats,
}

impl std::fmt::Debug for CompiledL9Lookup<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledL9Lookup")
            .field("cache_bytes", &self.cache.bytes)
            .field("cache_cap_bytes", &self.cache.byte_cap)
            .finish_non_exhaustive()
    }
}

impl CompiledL9Lookup<'_> {
    fn check_qid(&self, qid: u32) -> Result<()> {
        ensure!(
            (qid as usize) < self.values.compiled.domain.q_index.states.len(),
            "compiled L9 lookup qid {qid} outside domain"
        );
        Ok(())
    }

    fn value_in_block(&mut self, block: usize, qid: u32) -> Result<f32> {
        let id = self.cache.find_in_block(block, qid, &mut self.io_stats)?;
        Ok(if id == 0 {
            0.0
        } else {
            self.values.terminals[id as usize - 1]
        })
    }

    pub(crate) fn get(&mut self, field: FieldId, qid: u32) -> Result<f32> {
        self.check_qid(qid)?;
        let Some(block) = self.cache.load_block(field, 0, &mut self.io_stats)? else {
            self.io_stats.missing_next_children += 1;
            return Ok(0.0);
        };
        self.value_in_block(block, qid)
    }

    /// Sum in the caller's original reveal order. The caller applies gamma and
    /// divides by the full reveal count. The count matches legacy lookup stats:
    /// any zero value, including NONE or a numeric zero, counts as missing.
    pub(crate) fn sum_recs(&mut self, field: FieldId, recs: &[QForwardRec]) -> Result<(f32, u64)> {
        for rec in recs {
            self.check_qid(rec.child_qid)?;
        }
        let Some(block) = self.cache.load_block(field, 0, &mut self.io_stats)? else {
            self.io_stats.missing_next_children += recs.len() as u64;
            return Ok((0.0, recs.len() as u64));
        };
        let mut sum = 0.0f32;
        let mut missing = 0u64;
        for rec in recs {
            let value = self.value_in_block(block, rec.child_qid)?;
            sum += value;
            ensure!(sum.is_finite(), "nonfinite compiled L9 reveal sum");
            if value == 0.0 {
                missing += 1;
            }
        }
        Ok((sum, missing))
    }

    pub(crate) fn prewarm(&mut self, field: FieldId) -> Result<()> {
        self.cache
            .load_block(field, 0, &mut self.io_stats)
            .map(|_| ())
    }
    pub(crate) fn prefetch_fields(&mut self, fields: &[FieldId]) -> Result<()> {
        for &field in fields {
            self.prewarm(field)?;
        }
        Ok(())
    }
    pub(crate) fn stats_snapshot(&self) -> DenseLookupStats {
        self.cache.lookup_stats.clone()
    }
    pub(crate) fn io_stats(&self) -> &Loop1Stats {
        &self.io_stats
    }
    pub(crate) fn cache_bytes(&self) -> usize {
        self.cache.bytes
    }
    pub(crate) fn cache_cap_bytes(&self) -> usize {
        self.cache.byte_cap
    }
    pub(crate) fn cache_peak_bytes(&self) -> usize {
        self.io_stats.cache_peak_bytes
    }
}

/// Fixed-width signature; all inactive slots are zero. Active child order and
/// zero children are retained exactly, including an all-zero average.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AverageSignature {
    pub denominator: u8,
    pub children: [u32; PIECE_COUNT],
}

struct CachedMembership {
    bytes: Vec<u8>,
    used: u64,
}
struct TargetCache<'a> {
    l9: &'a CompiledL9<'a>,
    blocks: HashMap<usize, CachedMembership>,
    byte_cap: usize,
    bytes: usize,
    clock: u64,
    collect_stats: bool,
    lookup_stats: DenseLookupStats,
}

impl<'a> TargetCache<'a> {
    fn new(l9: &'a CompiledL9<'a>, byte_cap: usize) -> Self {
        Self {
            l9,
            blocks: HashMap::new(),
            byte_cap,
            bytes: 0,
            clock: 0,
            collect_stats: false,
            lookup_stats: DenseLookupStats::default(),
        }
    }
    fn clear(&mut self) {
        self.blocks.clear();
        self.bytes = 0;
    }
    fn find(
        &mut self,
        field: FieldId,
        qid: u32,
        fixed_bytes: usize,
        stats: &mut Loop1Stats,
    ) -> Result<u32> {
        let Some(block) = self.load_block(field, fixed_bytes, stats)? else {
            stats.missing_next_children += 1;
            return Ok(0);
        };
        self.find_in_block(block, qid, stats)
    }
    fn load_block(
        &mut self,
        field: FieldId,
        fixed_bytes: usize,
        stats: &mut Loop1Stats,
    ) -> Result<Option<usize>> {
        let Ok(block) = self.l9.metas.binary_search_by_key(&field, |m| m.field_id) else {
            if self.collect_stats {
                self.lookup_stats.missing_fields += 1;
            }
            return Ok(None);
        };
        if self.l9.membership_ranks.is_some() {
            if self.collect_stats {
                self.lookup_stats.cache_hits += 1;
            }
            return Ok(Some(block));
        }
        self.clock = self.clock.checked_add(1).context("cache clock overflow")?;
        if let Some(cached) = self.blocks.get_mut(&block) {
            cached.used = self.clock;
            if self.collect_stats {
                self.lookup_stats.cache_hits += 1;
            }
            return Ok(Some(block));
        }
        if self.collect_stats {
            self.lookup_stats.cache_misses += 1;
        }
        let meta = &self.l9.metas[block];
        let byte_len = (meta.count as usize)
            .checked_mul(3)
            .context("target membership byte count overflow")?;
        if byte_len <= self.byte_cap {
            while self
                .bytes
                .checked_add(byte_len)
                .context("target cache byte count overflow")?
                > self.byte_cap
            {
                let old = *self
                    .blocks
                    .iter()
                    .min_by_key(|(_, v)| v.used)
                    .context("target cache accounting failure")?
                    .0;
                self.bytes -= self.blocks.remove(&old).unwrap().bytes.len();
                if self.collect_stats {
                    self.lookup_stats.evictions += 1;
                }
            }
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(byte_len)
                .context("allocate target membership cache")?;
            bytes.resize(byte_len, 0);
            read_at(
                &self.l9.inputs.membership.file,
                &mut bytes,
                meta.chain_offset
                    .checked_mul(3)
                    .context("target offset overflow")?,
                ReadKind::Membership,
                stats,
            )?;
            self.blocks
                .try_reserve(1)
                .context("allocate target cache entry")?;
            self.blocks.insert(
                block,
                CachedMembership {
                    bytes,
                    used: self.clock,
                },
            );
            self.bytes += byte_len;
            stats.cache_peak_bytes = stats.cache_peak_bytes.max(fixed_bytes + self.bytes);
            if self.collect_stats {
                self.lookup_stats.sparse_blocks += 1;
                self.lookup_stats.materialized_states += meta.count as u64;
            }
        }
        Ok(Some(block))
    }
    fn find_in_block(&mut self, block: usize, qid: u32, stats: &mut Loop1Stats) -> Result<u32> {
        let meta = &self.l9.metas[block];
        let position = if let Some(ranks) = &self.l9.membership_ranks {
            ranks[block].rank(qid)
        } else if let Some(cached) = self.blocks.get(&block) {
            binary_search_encoded(&cached.bytes, qid)
        } else {
            let (mut low, mut high) = (0usize, meta.count as usize);
            let mut found = None;
            let mut raw = [0; 3];
            while low < high {
                let middle = low + (high - low) / 2;
                read_at(
                    &self.l9.inputs.membership.file,
                    &mut raw,
                    (meta.chain_offset + middle as u64)
                        .checked_mul(3)
                        .context("target lookup offset overflow")?,
                    ReadKind::Membership,
                    stats,
                )?;
                match u24(&raw).cmp(&qid) {
                    std::cmp::Ordering::Less => low = middle + 1,
                    std::cmp::Ordering::Greater => high = middle,
                    std::cmp::Ordering::Equal => {
                        found = Some(middle);
                        break;
                    }
                }
            }
            found
        };
        if let Some(position) = position {
            let absolute = usize::try_from(meta.chain_offset + position as u64)
                .context("L9 mapping index overflow")?;
            self.l9
                .state_ids
                .get(absolute)
                .copied()
                .context("L9 mapping index outside table")
        } else {
            stats.missing_next_children += 1;
            Ok(0)
        }
    }
}

fn binary_search_encoded(bytes: &[u8], qid: u32) -> Option<usize> {
    let (mut low, mut high) = (0, bytes.len() / 3);
    while low < high {
        let middle = low + (high - low) / 2;
        match u24(&bytes[middle * 3..middle * 3 + 3]).cmp(&qid) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some(middle),
        }
    }
    None
}

fn vec_capacity_bytes<T>(values: &Vec<T>) -> u64 {
    values.capacity() as u64 * size_of::<T>() as u64
}

/// Conservative storage for the current std/hashbrown table. Its reported
/// capacity excludes empty buckets. Rounding that capacity up to a power of
/// two covers the bucket array; 64 bytes cover the trailing control group and
/// alignment. Allocator bookkeeping and page rounding are deliberately outside
/// this capacity budget, as they are for Vec. This is an implementation-bound
/// estimate, rather than a claim about arbitrary HashMap implementations.
fn hash_capacity_bytes<K, V>(capacity: usize) -> Result<u64> {
    if capacity == 0 {
        return Ok(0);
    }
    let buckets = capacity
        .checked_next_power_of_two()
        .context("hash bucket count overflow")?;
    hash_bucket_bytes::<K, V>(buckets)
}

fn hash_bucket_bytes<K, V>(buckets: usize) -> Result<u64> {
    (buckets as u64)
        .checked_mul(size_of::<(K, V)>() as u64 + 1)
        .and_then(|bytes| bytes.checked_add(64))
        .context("hash capacity byte count overflow")
}

/// Reserving `needed` entries can round to the next power of two and require
/// a further bucket doubling for the 7/8 load factor. Charge both possibilities
/// before calling try_reserve, including the old allocation during rehash.
fn hash_reserve_upper_bytes<K, V>(needed: usize) -> Result<u64> {
    if needed == 0 {
        return Ok(0);
    }
    let buckets = needed
        .checked_next_power_of_two()
        .and_then(|n| n.checked_mul(2))
        .context("hash reserve bucket count overflow")?
        .max(4);
    hash_bucket_bytes::<K, V>(buckets)
}

struct CompilerHeapBudget {
    max_bytes: u64,
    /// Retained L9, fixed L8 mappings/metadata/paths, and conservative maximum
    /// scan/cache/used-terminal scratch. Does not include average/interner Vecs.
    fixed_bytes: u64,
}

impl CompilerHeapBudget {
    fn check(&self, variable_bytes: u64, stats: &mut Loop1Stats, reason: &str) -> Result<()> {
        let peak = self
            .fixed_bytes
            .checked_add(variable_bytes)
            .context("compiler capacity byte count overflow")?;
        ensure!(
            peak <= self.max_bytes,
            "L8 compiler heap capacity cap exceeded before {reason}: {peak} > {}; scanned={} unique={}",
            self.max_bytes,
            stats.scanned_states,
            stats.unique_averages
        );
        stats.compiler_peak_capacity_bytes = stats.compiler_peak_capacity_bytes.max(peak);
        Ok(())
    }

    fn reserve_average(
        &self,
        averages: &mut Vec<AverageSignature>,
        interner: &mut HashMap<AverageSignature, u32>,
        stats: &mut Loop1Stats,
    ) -> Result<()> {
        let needed = averages
            .len()
            .checked_add(1)
            .context("average count overflow")?;
        let map_bytes = hash_capacity_bytes::<AverageSignature, u32>(interner.capacity())?;
        if needed > averages.capacity() {
            let next_capacity = needed
                .checked_next_power_of_two()
                .context("average capacity overflow")?;
            let next_bytes = (next_capacity as u64)
                .checked_mul(size_of::<AverageSignature>() as u64)
                .context("average capacity byte count overflow")?;
            self.check(
                vec_capacity_bytes(averages)
                    .checked_add(next_bytes)
                    .and_then(|n| n.checked_add(map_bytes))
                    .context("average reallocation byte count overflow")?,
                stats,
                "average Vec reallocation",
            )?;
            averages
                .try_reserve_exact(next_capacity - averages.len())
                .context("allocate unique averages")?;
            ensure!(
                averages.capacity() <= next_capacity,
                "average allocator exceeded requested capacity"
            );
        }
        if needed > interner.capacity() {
            let next_bytes = hash_reserve_upper_bytes::<AverageSignature, u32>(needed)?;
            self.check(
                vec_capacity_bytes(averages)
                    .checked_add(map_bytes)
                    .and_then(|n| n.checked_add(next_bytes))
                    .context("interner rehash byte count overflow")?,
                stats,
                "average interner rehash",
            )?;
            interner
                .try_reserve(1)
                .context("allocate average interner entry")?;
            ensure!(
                hash_capacity_bytes::<AverageSignature, u32>(interner.capacity())? <= next_bytes,
                "HashMap allocator exceeded conservative reserve capacity"
            );
        }
        let map_bytes = hash_capacity_bytes::<AverageSignature, u32>(interner.capacity())?;
        stats.interner_peak_capacity_bytes = stats.interner_peak_capacity_bytes.max(map_bytes);
        self.check(
            vec_capacity_bytes(averages) + map_bytes,
            stats,
            "average insertion",
        )
    }
}

pub(crate) struct CompiledL8<'a> {
    l9: &'a CompiledL9<'a>,
    inputs: LayerInputs,
    field_range: Range<usize>,
    state_offset: u64,
    metas: Vec<FieldChainBlockMeta>,
    state_ids: Vec<u32>,
    averages: Vec<AverageSignature>,
    used_zids: Vec<u32>,
    stats: Loop1Stats,
}

impl<'a> CompiledL8<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        qfwd: &'a QForwardGraph,
        pre: &'a dyn TerminalFormulaIndex,
        layer: &FieldChainLayer,
        policy_path: &Path,
        field_range: Range<usize>,
        l9: &'a CompiledL9<'a>,
        limits: Loop1Limits,
    ) -> Result<Self> {
        Self::build_with_heap_limit(
            graph,
            q_index,
            qfwd,
            pre,
            layer,
            policy_path,
            field_range,
            l9,
            limits,
            u64::MAX,
        )
    }

    /// Compile any contiguous whole-field range, including the entire L8.
    /// The capacity budget includes retained L9, both state maps, retained L8
    /// metadata, bounded cache/scratch, and conservative old+new allocations
    /// during Vec growth and HashMap rehash. Borrowed graph/index/formula data
    /// and allocator/page overhead remain outside this budget.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_with_heap_limit(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        qfwd: &'a QForwardGraph,
        pre: &'a dyn TerminalFormulaIndex,
        layer: &FieldChainLayer,
        policy_path: &Path,
        field_range: Range<usize>,
        l9: &'a CompiledL9<'a>,
        limits: Loop1Limits,
        max_compiler_heap_bytes: u64,
    ) -> Result<Self> {
        let started = Instant::now();
        let domain = Domain {
            graph,
            q_index,
            qfwd,
            pre,
        };
        domain.validate()?;
        ensure!(
            domain.matches(&l9.domain),
            "L8 and L9 must share the same immutable graph/index/formulas"
        );
        let is_empty_layer =
            layer.metas.is_empty() && layer.total_states == 0 && field_range == (0..0);
        ensure!(
            is_empty_layer
                || (field_range.start < field_range.end && field_range.end <= layer.metas.len()),
            "invalid L8 whole-field range"
        );
        ensure!(
            limits.max_unique_averages <= u32::MAX as usize,
            "average cap exceeds u32 ID domain"
        );
        let (state_offset, count) = if is_empty_layer {
            (0, 0)
        } else {
            let first = &layer.metas[field_range.start];
            let last = &layer.metas[field_range.end - 1];
            let count = last
                .chain_offset
                .checked_add(last.count as u64)
                .and_then(|n| n.checked_sub(first.chain_offset))
                .context("L8 selected state count overflow")?;
            (first.chain_offset, count)
        };
        let bytes = mapping_bytes(count)?;
        let combined = bytes
            .checked_add(l9.stats.mapping_bytes)
            .context("combined mapping byte count overflow")?;
        ensure!(
            combined <= limits.max_mapping_bytes,
            "combined L9/L8 mapping byte cap exceeded: {combined} > {}",
            limits.max_mapping_bytes
        );
        let mut stats = Loop1Stats {
            mapping_bytes: bytes,
            ..Loop1Stats::default()
        };
        l9.check_inputs_unchanged()?;
        let inputs = LayerInputs::open(layer, policy_path)?;
        validate_layer(layer, &inputs, graph, q_index.states.len(), &mut stats)?;
        // Charge the whole configured scan/cache payload, the maximum number
        // of cache hash buckets (including rehash overlap), and the largest
        // possible final used-zid list before allocating either state map.
        let cache_hash = hash_reserve_upper_bytes::<usize, CachedMembership>(l9.metas.len())?
            .checked_mul(2)
            .context("cache rehash byte count overflow")?;
        let budget = CompilerHeapBudget {
            max_bytes: max_compiler_heap_bytes,
            fixed_bytes: l9
                .storage_bytes()
                .checked_add(bytes)
                .and_then(|n| {
                    n.checked_add((field_range.len() * size_of::<FieldChainBlockMeta>()) as u64)
                })
                .and_then(|n| n.checked_add(pre.terminal_count() as u64 * 4))
                .and_then(|n| n.checked_add(limits.max_cache_bytes as u64))
                .and_then(|n| n.checked_add(cache_hash))
                .and_then(|n| n.checked_add(inputs.heap_bytes()))
                .context("fixed compiler capacity byte count overflow")?,
        };
        budget.check(0, &mut stats, "scan/cache and state mapping allocation")?;
        let mut scratch = ScanScratch::new(pre.terminal_count(), limits, &mut stats)?;
        let fixed_bytes = scratch.bytes();
        let mut cache = TargetCache::new(l9, limits.max_cache_bytes - fixed_bytes);
        let mut state_ids = allocate_ids(count)?;
        ensure!(
            vec_capacity_bytes(&state_ids) == bytes,
            "mapping allocator exceeded requested capacity"
        );
        let metas = layer.metas[field_range.clone()].to_vec();
        ensure!(
            metas.capacity() == field_range.len(),
            "metadata allocator exceeded requested capacity"
        );
        let mut averages = Vec::<AverageSignature>::new();
        let mut interner = HashMap::<AverageSignature, u32>::new();
        let mut last_progress = Instant::now();
        for meta in &layer.metas[field_range.clone()] {
            // A source field owns its target-membership cache; no all-qid map is retained.
            cache.clear();
            let mut previous = None;
            let mut position = 0usize;
            while position < meta.count as usize {
                let chunk = (meta.count as usize - position).min(scratch.chunk_states);
                read_chunk(
                    &inputs,
                    meta.chain_offset + position as u64,
                    chunk,
                    &mut scratch,
                    &mut stats,
                )?;
                for i in 0..chunk {
                    let qid = u24(&scratch.bytes[i * 3..i * 3 + 3]);
                    validate_qid(qid, &mut previous, q_index.states.len(), meta.field_id)?;
                    let p = chunk * 3 + i * 2;
                    let code = u16::from_le_bytes([scratch.bytes[p], scratch.bytes[p + 1]]);
                    let id = if code == u16::MAX {
                        0
                    } else {
                        let (target, recs, _) =
                            decode_action(graph, qfwd, meta.field_id, qid, code)?;
                        // Resolve the target field before interning; the raw edge index is not a key.
                        let mut signature = AverageSignature {
                            denominator: recs.len() as u8,
                            children: [0; PIECE_COUNT],
                        };
                        for (slot, rec) in recs.iter().enumerate() {
                            let child =
                                cache.find(target, rec.child_qid, fixed_bytes, &mut stats)?;
                            signature.children[slot] = child;
                            scratch.mark(child);
                        }
                        if let Some(&id) = interner.get(&signature) {
                            id
                        } else {
                            ensure!(
                                averages.len() < limits.max_unique_averages,
                                "unique average cap exceeded: scanned={} unique={} cap={}",
                                stats.scanned_states,
                                averages.len(),
                                limits.max_unique_averages
                            );
                            let id = u32::try_from(averages.len())
                                .context("average ID overflow")?
                                .checked_add(1)
                                .context("average ID overflow")?;
                            budget.reserve_average(&mut averages, &mut interner, &mut stats)?;
                            averages.push(signature);
                            interner.insert(signature, id);
                            stats.unique_averages = averages.len();
                            stats.retained_edges += signature.denominator as u64;
                            id
                        }
                    };
                    state_ids.push(id);
                    stats.scanned_states += 1;
                }
                position += chunk;
                if count >= 10_000_000 && last_progress.elapsed().as_secs() >= 10 {
                    eprintln!("loop1 L8 compile: scanned={}/{count} unique={} elapsed={:.1}s capacity_peak_bytes={} interner_capacity_bytes={}", stats.scanned_states, averages.len(), started.elapsed().as_secs_f64(), stats.compiler_peak_capacity_bytes, stats.interner_peak_capacity_bytes);
                    last_progress = Instant::now();
                }
            }
        }
        ensure!(
            state_ids.len() as u64 == count,
            "L8 scan state count mismatch"
        );
        let used_zids = scratch.used_zids(pre.terminal_count())?;
        stats.unique_averages = averages.len();
        stats.unique_terminal_leaves = used_zids.len();
        stats.average_bytes = averages.len() as u64 * size_of::<AverageSignature>() as u64;
        stats.interner_peak_logical_bytes =
            averages.len() as u64 * (size_of::<AverageSignature>() + size_of::<u32>()) as u64;
        stats.table_bytes = bytes
            + stats.average_bytes
            + used_zids.len() as u64 * 4
            + field_range.len() as u64 * size_of::<FieldChainBlockMeta>() as u64;
        drop(interner);
        inputs.check()?;
        l9.check_inputs_unchanged()?;
        stats.construction_seconds = started.elapsed().as_secs_f64();
        let mut compiled = Self {
            l9,
            inputs,
            field_range: field_range.clone(),
            state_offset,
            metas,
            state_ids,
            averages,
            used_zids,
            stats,
        };
        compiled.stats.retained_capacity_bytes = compiled.storage_bytes();
        Ok(compiled)
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.l9.check_inputs_unchanged()?;
        self.inputs.check()
    }
    pub(crate) fn stats(&self) -> &Loop1Stats {
        &self.stats
    }
    pub(crate) fn storage_bytes(&self) -> u64 {
        vec_capacity_bytes(&self.state_ids)
            + vec_capacity_bytes(&self.averages)
            + vec_capacity_bytes(&self.metas)
            + vec_capacity_bytes(&self.used_zids)
            + self.inputs.heap_bytes()
    }
    pub(crate) fn state_ids(&self) -> &[u32] {
        &self.state_ids
    }
    pub(crate) fn state_count(&self) -> usize {
        self.state_ids.len()
    }
    pub(crate) fn range(&self) -> Range<usize> {
        self.field_range.clone()
    }
    pub(crate) fn field_range(&self) -> Range<usize> {
        self.field_range.clone()
    }
    pub(crate) fn state_offset(&self) -> u64 {
        self.state_offset
    }
    pub(crate) fn metas(&self) -> &[FieldChainBlockMeta] {
        &self.metas
    }
    pub(crate) fn averages(&self) -> &[AverageSignature] {
        &self.averages
    }
    pub(crate) fn used_terminal_zids(&self) -> &[u32] {
        &self.used_zids
    }
    pub(crate) fn evaluate_f32(&self, terminals: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.evaluate_into_f32(terminals, &mut out, &mut Vec::new())?;
        Ok(out)
    }
    pub(crate) fn evaluate_into_f32(
        &self,
        terminals: &[f32],
        out: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        self.evaluate_averages_into_f32(terminals, scratch)?;
        reserve_output(out, self.state_ids.len())?;
        out.resize(self.state_ids.len(), 0.0);
        self.gather_range_into_f32(scratch, 0..self.state_ids.len(), out)
    }

    /// Evaluate unique average nodes only. No state-sized L8 output is
    /// allocated. Nodes run independently in parallel; each node keeps its
    /// original ordered f32 additions and original full denominator.
    pub(crate) fn evaluate_averages_f32(&self, terminals: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.evaluate_averages_into_f32(terminals, &mut out)?;
        Ok(out)
    }

    pub(crate) fn evaluate_averages_into_f32(
        &self,
        terminals: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        validate_terminals(
            terminals,
            self.l9.domain.pre.terminal_count(),
            &self.used_zids,
        )?;
        reserve_output(out, self.averages.len())?;
        out.resize(self.averages.len(), 0.0);
        out.par_iter_mut()
            .zip(self.averages.par_iter())
            .enumerate()
            .try_for_each(|(i, (out, signature))| -> Result<()> {
                let mut sum = 0.0f32;
                for &child in &signature.children[..signature.denominator as usize] {
                    sum += if child == 0 {
                        0.0
                    } else {
                        terminals[child as usize - 1]
                    };
                    ensure!(sum.is_finite(), "nonfinite average sum at index {i}");
                }
                let value = sum / signature.denominator as f32;
                ensure!(value.is_finite(), "nonfinite average at index {i}");
                *out = value;
                Ok(())
            })
    }

    /// Gather a local state range directly into caller-owned RAM or mmap. The
    /// range is relative to this compiled slab, not to the original layer.
    /// This does not allocate a state-sized intermediate output.
    pub(crate) fn gather_range_into_f32(
        &self,
        averages: &[f32],
        range: Range<usize>,
        out: &mut [f32],
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        ensure!(
            averages.len() == self.averages.len(),
            "average value length mismatch"
        );
        ensure!(
            range.start <= range.end && range.end <= self.state_ids.len(),
            "invalid L8 gather range"
        );
        ensure!(out.len() == range.len(), "L8 gather output length mismatch");
        out.par_iter_mut()
            .zip(self.state_ids[range].par_iter())
            .try_for_each(|(out, &id)| -> Result<()> {
                let value = if id == 0 {
                    0.0
                } else {
                    averages[id as usize - 1]
                };
                ensure!(
                    value.is_finite(),
                    "nonfinite average during L8 state gather"
                );
                *out = value;
                Ok(())
            })
    }

    /// Bounded gather for file-backed sinks. Only one `max_chunk_states` f32
    /// buffer is retained and each fully populated chunk is passed to `write`.
    /// Offsets are relative to the selected state slab. The caller publishes
    /// its output only after this method succeeds.
    pub(crate) fn gather_chunks_f32(
        &self,
        averages: &[f32],
        max_chunk_states: usize,
        mut write: impl FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        ensure!(
            max_chunk_states > 0,
            "L8 gather chunk size must be positive"
        );
        ensure!(
            averages.len() == self.averages.len(),
            "average value length mismatch"
        );
        let mut buffer = Vec::new();
        reserve_output(&mut buffer, max_chunk_states.min(self.state_ids.len()))?;
        for start in (0..self.state_ids.len()).step_by(max_chunk_states) {
            let end = start
                .saturating_add(max_chunk_states)
                .min(self.state_ids.len());
            buffer.resize(end - start, 0.0);
            self.gather_range_into_f32(averages, start..end, &mut buffer)?;
            write(start, &buffer)?;
        }
        self.check_inputs_unchanged()?;
        Ok(())
    }
}

fn validate_terminals(terminals: &[f32], expected: usize, used: &[u32]) -> Result<()> {
    ensure!(
        terminals.len() == expected,
        "terminal value length {} != {expected}",
        terminals.len()
    );
    for &z in used {
        ensure!(
            terminals[z as usize].is_finite(),
            "nonfinite terminal value at zid {z}"
        );
    }
    Ok(())
}

fn reserve_output(out: &mut Vec<f32>, len: usize) -> Result<()> {
    out.clear();
    out.try_reserve_exact(len)
        .context("allocate evaluation output")?;
    Ok(())
}

#[cfg(test)]
#[path = "loop1_stream_capacity_tests.rs"]
mod capacity_tests;
