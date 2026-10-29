//! Bounded, depth-first evaluation of the original fixed-policy boundary map.
//!
//! The only retained numeric intermediates are collision-evicting direct maps.
//! Eviction changes the work, never the value. Every sum retains qforward order
//! and its full denominator; missing children and POLICY_NONE are positive zero.
//! Read-only input mappings are virtual address space, not an RSS promise. Run
//! experiments under the external memory/CPU/output limits as well.

use anyhow::{ensure, Context, Result};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::graph::{FieldGraph, FieldId, EMPTY_FIELD_ID, PIECE_COUNT};
use crate::layer_store::{FieldChainBlockMeta, FieldChainLayer};
use crate::loop1_prepared::PreparedL9;
#[cfg(target_os = "linux")]
use crate::loop1_stream::InputMutationWatch;
use crate::loop1_stream::TerminalFormulaIndex;
use crate::loop1_stream::{CompiledL9Lookup, PreparedL9Values};
use crate::qgraph::QForwardGraph;

#[path = "loop1_sparse_dag.rs"]
pub(crate) mod dag;

const MAX_WORKERS: usize = 4;
const MAX_MEMO_BYTES: u64 = 8 << 30;
const FIELD_VALIDATION_SLOTS: usize = 4096;
const EMPTY_KEY: u64 = u64::MAX;
/// Unique bucket payload cap. Padded retirement coverage can be three times
/// larger; neither value is an unconditional cap on total process RSS.
pub(crate) const SOURCE_RESIDENCY_BUDGET_BYTES: u64 = 256 << 20;
pub(crate) const SOURCE_SCALAR_READ_CHARGE_BYTES: u64 = 128 << 10;
const MEMBERSHIP_VALIDATION_CHUNK_STATES: u64 = 65_536;

#[derive(Clone, Copy, Debug)]
pub(crate) enum SourceKind {
    Membership,
    Policy,
}

/// Bounded unique source-bucket window. Repeated reads retain their bucket until
/// retirement without spending another slot or forcing the same pages to
/// refault. The 256 MiB payload cap is unique bucket coverage, not read traffic.
/// With page size dividing the bucket size, one 128 KiB fringe on each side
/// gives at most 768 MiB of advised page intervals per worker (3 GiB across
/// four workers), before overlaps. A residency interpretation also assumes OS
/// fault-around stays inside those fringes; external RSS limits still apply.
pub(crate) struct SourceTouchWindow {
    buckets: Vec<u64>,
    seen: Vec<u64>,
    last_bucket: u64,
    flushes: u64,
    advised_ranges: u64,
    records: u64,
    dedup_hits: u64,
    unique_records: u64,
    retire_seconds: f64,
    policy_id: Option<usize>,
}

impl SourceTouchWindow {
    const MAX_RECORDS: usize =
        (SOURCE_RESIDENCY_BUDGET_BYTES / SOURCE_SCALAR_READ_CHARGE_BYTES) as usize;
    const BUCKET_MASK: u64 = (1u64 << 56) - 1;
    const SEEN_SLOTS: usize = Self::MAX_RECORDS * 2;

    pub(crate) fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            ensure!(
                page_size > 0 && SOURCE_SCALAR_READ_CHARGE_BYTES % page_size as u64 == 0,
                "source bucket size must be a multiple of the OS page size"
            );
        }
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(Self::MAX_RECORDS)
            .context("allocate bounded source touch window")?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(Self::SEEN_SLOTS)
            .context("allocate bounded exact source bucket set")?;
        seen.resize(Self::SEEN_SLOTS, EMPTY_KEY);
        Ok(Self {
            buckets,
            seen,
            last_bucket: EMPTY_KEY,
            flushes: 0,
            advised_ranges: 0,
            records: 0,
            dedup_hits: 0,
            unique_records: 0,
            retire_seconds: 0.0,
            policy_id: None,
        })
    }

    pub(crate) fn record(
        &mut self,
        policy: &SparsePolicy<'_>,
        depth: usize,
        kind: SourceKind,
        byte_offset: u64,
    ) -> Result<()> {
        let policy_id = policy as *const SparsePolicy<'_> as usize;
        ensure!(
            self.buckets.is_empty() || self.policy_id == Some(policy_id),
            "source touch window belongs to a different policy"
        );
        let layer = policy
            .layers
            .get(depth)
            .context("source touch layer outside domain")?;
        let (kind_id, width, map) = match kind {
            SourceKind::Membership => (0u64, 3u64, &layer.membership),
            SourceKind::Policy => (1u64, 2u64, &layer.policy),
        };
        ensure!(
            byte_offset
                .checked_add(width)
                .is_some_and(|end| end <= map.bytes().len() as u64),
            "source touch outside mapping"
        );
        let bucket = byte_offset / SOURCE_SCALAR_READ_CHARGE_BYTES;
        ensure!(bucket <= Self::BUCKET_MASK, "source touch bucket overflow");
        let map_id = depth as u64 * 2 + kind_id;
        let key = (map_id << 56) | bucket;
        self.record_key(policy, policy_id, key)
    }

    /// All source binding and bounds checks precede this private fast path.
    fn record_key(&mut self, policy: &SparsePolicy<'_>, policy_id: usize, key: u64) -> Result<()> {
        self.records += 1;
        if self.last_bucket == key {
            self.dedup_hits += 1;
            return Ok(());
        }
        let (present, mut slot) = self.seen_slot(key)?;
        if present {
            self.last_bucket = key;
            self.dedup_hits += 1;
            return Ok(());
        }
        if self.buckets.len() == Self::MAX_RECORDS {
            self.flush(policy)?;
            slot = self.seen_slot(key)?.1;
        }
        self.policy_id = Some(policy_id);
        self.seen[slot] = key;
        self.buckets.push(key);
        self.last_bucket = key;
        self.unique_records += 1;
        Ok(())
    }

    fn seen_slot(&self, key: u64) -> Result<(bool, usize)> {
        let mut slot = hash(key) as usize & (Self::SEEN_SLOTS - 1);
        for _ in 0..Self::SEEN_SLOTS {
            let existing = self.seen[slot];
            if existing == key {
                return Ok((true, slot));
            }
            if existing == EMPTY_KEY {
                return Ok((false, slot));
            }
            slot = (slot + 1) & (Self::SEEN_SLOTS - 1);
        }
        anyhow::bail!("bounded exact source bucket set exhausted")
    }

    pub(crate) fn flush(&mut self, policy: &SparsePolicy<'_>) -> Result<()> {
        if self.buckets.is_empty() {
            return Ok(());
        }
        ensure!(
            self.policy_id == Some(policy as *const SparsePolicy<'_> as usize),
            "source retirement window belongs to a different policy"
        );
        let start = Instant::now();
        self.buckets.sort_unstable();
        self.buckets.dedup();
        let mut index = 0;
        while index < self.buckets.len() {
            let map_id = self.buckets[index] >> 56;
            let first = self.buckets[index] & Self::BUCKET_MASK;
            let mut last = first;
            index += 1;
            while index < self.buckets.len()
                && self.buckets[index] >> 56 == map_id
                && self.buckets[index] & Self::BUCKET_MASK == last + 1
            {
                last += 1;
                index += 1;
            }
            let layer = policy
                .layers
                .get((map_id / 2) as usize)
                .context("source retirement layer outside domain")?;
            let map = if map_id & 1 == 0 {
                &layer.membership
            } else {
                &layer.policy
            };
            let begin = first
                .checked_mul(SOURCE_SCALAR_READ_CHARGE_BYTES)
                .context("source retirement offset overflow")?;
            let end = last
                .checked_add(1)
                .and_then(|last| last.checked_mul(SOURCE_SCALAR_READ_CHARGE_BYTES))
                .context("source retirement range overflow")?
                .min(map.bytes().len() as u64);
            // A u24 scalar may straddle a bucket boundary. The helper's
            // 128 KiB padding also retires that neighbor and fault-around PTEs.
            map.release_pages_with_fringe(begin, end - begin)?;
            self.advised_ranges += 1;
        }
        self.buckets.clear();
        self.seen.fill(EMPTY_KEY);
        self.last_bucket = EMPTY_KEY;
        self.flushes += 1;
        self.retire_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }

    pub(crate) fn flush_count(&self) -> u64 {
        self.flushes
    }
    pub(crate) fn range_count(&self) -> u64 {
        self.advised_ranges
    }
    pub(crate) fn record_count(&self) -> u64 {
        self.records
    }
    pub(crate) fn dedup_hit_count(&self) -> u64 {
        self.dedup_hits
    }
    /// Cumulative bucket admissions; a bucket is counted again after a flush.
    pub(crate) fn unique_record_count(&self) -> u64 {
        self.unique_records
    }
    pub(crate) fn retire_seconds(&self) -> f64 {
        self.retire_seconds
    }
    pub(crate) fn pending_charge_bytes(&self) -> u64 {
        // Compatibility name: this is pending UNIQUE bucket payload, not a
        // cumulative charge for repeated reads and not padded RSS coverage.
        self.buckets.len() as u64 * SOURCE_SCALAR_READ_CHARGE_BYTES
    }
    pub(crate) fn storage_bytes(&self) -> u64 {
        ((self.buckets.capacity() + self.seen.capacity()) * size_of::<u64>() + size_of::<Self>())
            as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
    mode: u32,
}

impl Stamp {
    fn of(m: &Metadata) -> Self {
        Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
            mode: m.mode(),
        }
    }
}

struct BoundMap {
    path: PathBuf,
    canonical: PathBuf,
    file: File,
    stamp: Stamp,
    request_stamp: Stamp,
    map: Option<Mmap>,
    #[cfg(target_os = "linux")]
    mutation_watch: InputMutationWatch,
}

impl BoundMap {
    fn open(path: &Path, expected: u64) -> Result<Self> {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let canonical = fs::canonicalize(&path)?;
        #[cfg(target_os = "linux")]
        let mutation_watch = InputMutationWatch::open(&canonical)?;
        let request_stamp = Stamp::of(&fs::symlink_metadata(&path)?);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file(),
            "input is not a regular file: {}",
            path.display()
        );
        ensure!(
            metadata.len() == expected,
            "input size mismatch: {} got {}, expected {expected}",
            path.display(),
            metadata.len()
        );
        ensure!(
            expected <= isize::MAX as u64,
            "input mapping exceeds addressable slice size"
        );
        let stamp = Stamp::of(&metadata);
        // SAFETY: the mapping is read-only. All source identities are retained
        // and checked around construction, validation and each application.
        // Callers must keep inputs immutable; metadata cannot defeat malicious
        // concurrent writes or prevent SIGBUS if another process truncates one.
        let map = if expected == 0 {
            None
        } else {
            Some(unsafe { MmapOptions::new().map(&file)? })
        };
        let result = Self {
            path,
            canonical,
            file,
            stamp,
            request_stamp,
            map,
            #[cfg(target_os = "linux")]
            mutation_watch,
        };
        result.check()?;
        Ok(result)
    }

    fn bytes(&self) -> &[u8] {
        self.map.as_deref().unwrap_or(&[])
    }

    fn check(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        self.mutation_watch.check(&self.path)?;
        ensure!(
            Stamp::of(&self.file.metadata()?) == self.stamp,
            "sparse loop1 input handle changed: {}",
            self.path.display()
        );
        ensure!(
            Stamp::of(&fs::symlink_metadata(&self.path)?) == self.request_stamp
                && fs::canonicalize(&self.path)? == self.canonical
                && Stamp::of(&fs::metadata(&self.path)?) == self.stamp,
            "sparse loop1 input path changed: {}",
            self.path.display()
        );
        Ok(())
    }

    /// Retire only this read-only file mapping's PTEs. The VMA remains mapped
    /// and concurrent readers may fault pages back in safely. No anonymous
    /// caches, owned numeric arrays or file contents are discarded.
    fn release_pages(&self, offset: u64, bytes: u64) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .context("mapped release range overflow")?;
        ensure!(
            end <= self.bytes().len() as u64,
            "mapped release exceeds input bounds"
        );
        if bytes == 0 {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            ensure!(page_size > 0, "query mapped-input page size");
            let page_size = page_size as u64;
            let start = offset / page_size * page_size;
            let count = usize::try_from(end - start)?;
            // SAFETY: start is page aligned within this live read-only mmap;
            // count ends inside its original byte length. MADV_DONTNEED keeps
            // the mapping valid. The immutable-input contract ensures every
            // concurrent refault observes exactly the same bytes.
            let result = unsafe {
                libc::madvise(
                    self.bytes().as_ptr().add(start as usize) as *mut libc::c_void,
                    count,
                    libc::MADV_DONTNEED,
                )
            };
            ensure!(
                result == 0,
                "retire mapped input pages {}: {}",
                self.path.display(),
                std::io::Error::last_os_error()
            );
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("bounded mapped-input residency requires Linux MADV_DONTNEED")
    }

    fn release_pages_with_fringe(&self, offset: u64, bytes: u64) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .context("mapped release range overflow")?;
        ensure!(
            end <= self.bytes().len() as u64,
            "mapped release exceeds input bounds"
        );
        if bytes == 0 {
            return Ok(());
        }
        // Linux file fault-around may install neighboring file PTEs. Retire
        // those fringes as well so many short disjoint field scans cannot
        // accumulate resident tails outside their requested state spans.
        let start = offset.saturating_sub(SOURCE_SCALAR_READ_CHARGE_BYTES);
        let end = end
            .saturating_add(SOURCE_SCALAR_READ_CHARGE_BYTES)
            .min(self.bytes().len() as u64);
        self.release_pages(start, end - start)
    }
}

struct MappedLayer<'a> {
    metas: &'a [FieldChainBlockMeta],
    total_states: u64,
    index: BoundMap,
    membership: BoundMap,
    policy: BoundMap,
}

impl<'a> MappedLayer<'a> {
    fn open(layer: &'a FieldChainLayer, policy: &Path, graph: &FieldGraph) -> Result<Self> {
        ensure!(
            layer.metas.len() <= graph.offsets.len(),
            "too many layer fields"
        );
        let index_size = (layer.metas.len() as u64)
            .checked_mul(16)
            .and_then(|n| n.checked_add(24))
            .context("index size overflow")?;
        let index = BoundMap::open(&layer.index_path, index_size)?;
        let bytes = index.bytes();
        ensure!(
            &bytes[..8] == b"FCHNIDX1" && read_u32(bytes, 8) == 1,
            "invalid sparse loop1 index header"
        );
        ensure!(
            read_u32(bytes, 12) as usize == layer.metas.len()
                && read_u64(bytes, 16) == layer.total_states,
            "stale layer index header"
        );
        let mut offset = 0u64;
        let mut previous = None;
        for (i, meta) in layer.metas.iter().enumerate() {
            ensure!(
                (meta.field_id as usize) < graph.offsets.len(),
                "layer field outside graph"
            );
            ensure!(
                previous.is_none_or(|p| p < meta.field_id),
                "layer fields are not sorted"
            );
            ensure!(meta.chain_offset == offset, "noncontiguous layer offsets");
            offset = offset
                .checked_add(u64::from(meta.count))
                .context("layer offset overflow")?;
            ensure!(
                read_u32(bytes, 24 + 16 * i) == meta.field_id
                    && read_u32(bytes, 28 + 16 * i) == meta.count
                    && read_u64(bytes, 32 + 16 * i) == meta.chain_offset,
                "layer metadata changed after loading"
            );
            previous = Some(meta.field_id);
        }
        ensure!(offset == layer.total_states, "layer total mismatch");
        let membership = BoundMap::open(
            &layer.chains_path,
            offset.checked_mul(3).context("membership size overflow")?,
        )?;
        let policy = BoundMap::open(
            policy,
            offset.checked_mul(2).context("policy size overflow")?,
        )?;
        let result = Self {
            metas: &layer.metas,
            total_states: offset,
            index,
            membership,
            policy,
        };
        result.check()?;
        result
            .index
            .release_pages(0, result.index.bytes().len() as u64)?;
        Ok(result)
    }

    fn check(&self) -> Result<()> {
        self.index.check()?;
        self.membership.check()?;
        self.policy.check()
    }

    #[inline]
    fn qid(&self, index: u64) -> u32 {
        let pos = index as usize * 3;
        let bytes = self.membership.bytes();
        u32::from(bytes[pos]) | (u32::from(bytes[pos + 1]) << 8) | (u32::from(bytes[pos + 2]) << 16)
    }

    #[inline]
    fn code(&self, index: u64) -> u16 {
        let pos = index as usize * 2;
        let bytes = self.policy.bytes();
        u16::from_le_bytes([bytes[pos], bytes[pos + 1]])
    }
}

fn read_u32(bytes: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap())
}

/// Coarse contiguous groups keep each worker's mapped reads sequential. Only
/// metadata ranges are retained; no membership or policy bytes are copied.
fn validation_ranges(metas: &[FieldChainBlockMeta]) -> Vec<std::ops::Range<usize>> {
    const TARGET_STATES: u64 = 4 << 20;
    const MAX_FIELDS: usize = 1024;
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut states = 0u64;
    for (index, meta) in metas.iter().enumerate() {
        states += u64::from(meta.count);
        if states >= TARGET_STATES || index + 1 - start >= MAX_FIELDS {
            ranges.push(start..index + 1);
            start = index + 1;
            states = 0;
        }
    }
    if start < metas.len() {
        ranges.push(start..metas.len());
    }
    ranges
}

/// Maps one immutable fixed policy. The formula index is used only to identify
/// exact ordered action expressions; no boundary coordinates are reduced.
pub(crate) struct SparsePolicy<'a> {
    graph: &'a FieldGraph,
    qfwd: &'a QForwardGraph,
    formulas: &'a dyn TerminalFormulaIndex,
    layers: Vec<MappedLayer<'a>>,
    qstate_count: usize,
    available_pieces: Vec<u8>,
    two_line_field: FieldId,
    fully_validated: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseLimits {
    pub workers: usize,
    pub total_memo_bytes: u64,
    pub progress: bool,
}

impl Default for SparseLimits {
    fn default() -> Self {
        Self {
            workers: MAX_WORKERS,
            total_memo_bytes: 4 << 30,
            progress: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SparseStats {
    pub roots: u64,
    pub state_requests: u64,
    pub evaluated_states: u64,
    pub state_hits: u64,
    pub state_collisions: u64,
    pub action_hits: u64,
    pub action_collisions: u64,
    pub missing_children: u64,
    /// Logical bytes inspected in mappings; not physical disk traffic.
    pub membership_bytes: u64,
    pub policy_bytes: u64,
    pub reachable_validation_bytes: u64,
    pub memo_payload_bytes: u64,
    pub field_validation_tag_bytes: u64,
    pub source_page_releases: u64,
    pub source_retired_ranges: u64,
    pub source_touch_records: u64,
    pub source_touch_unique_buckets: u64,
    pub source_touch_dedup_hits: u64,
    pub source_touch_window_bytes: u64,
    pub source_retire_seconds: f64,
    pub compiled_l9_averages: u64,
    pub compiled_l9_reveals: u64,
    pub compiled_l9_zero_terms: u64,
    pub compiled_l9_membership_bytes: u64,
    pub elapsed_seconds: f64,
}

impl SparseStats {
    fn add(&mut self, rhs: &Self) {
        self.roots += rhs.roots;
        self.state_requests += rhs.state_requests;
        self.evaluated_states += rhs.evaluated_states;
        self.state_hits += rhs.state_hits;
        self.state_collisions += rhs.state_collisions;
        self.action_hits += rhs.action_hits;
        self.action_collisions += rhs.action_collisions;
        self.missing_children += rhs.missing_children;
        self.membership_bytes += rhs.membership_bytes;
        self.policy_bytes += rhs.policy_bytes;
        self.reachable_validation_bytes += rhs.reachable_validation_bytes;
        self.memo_payload_bytes += rhs.memo_payload_bytes;
        self.field_validation_tag_bytes += rhs.field_validation_tag_bytes;
        self.source_page_releases += rhs.source_page_releases;
        self.source_retired_ranges += rhs.source_retired_ranges;
        self.source_touch_records += rhs.source_touch_records;
        self.source_touch_unique_buckets += rhs.source_touch_unique_buckets;
        self.source_touch_dedup_hits += rhs.source_touch_dedup_hits;
        self.source_touch_window_bytes += rhs.source_touch_window_bytes;
        self.source_retire_seconds += rhs.source_retire_seconds;
        self.compiled_l9_averages += rhs.compiled_l9_averages;
        self.compiled_l9_reveals += rhs.compiled_l9_reveals;
        self.compiled_l9_zero_terms += rhs.compiled_l9_zero_terms;
        self.compiled_l9_membership_bytes += rhs.compiled_l9_membership_bytes;
    }
}

pub(crate) struct SparseOutput {
    pub values: Vec<f32>,
    pub stats: SparseStats,
}

impl<'a> SparsePolicy<'a> {
    pub(crate) fn open(
        graph: &'a FieldGraph,
        qfwd: &'a QForwardGraph,
        formulas: &'a dyn TerminalFormulaIndex,
        layers: &'a [FieldChainLayer],
        policy_dir: &Path,
        two_line_field: FieldId,
    ) -> Result<Self> {
        ensure!(
            layers.len() == 10,
            "sparse loop1 requires exactly ten layers"
        );
        validate_domain(graph, qfwd, formulas)?;
        let available_pieces = qfwd
            .offsets
            .iter()
            .map(|row| {
                (0..PIECE_COUNT).fold(0u8, |mask, piece| {
                    mask | (u8::from(row[piece] < row[piece + 1]) << piece)
                })
            })
            .collect();
        ensure!(
            (two_line_field as usize) < graph.offsets.len(),
            "two-line field outside graph"
        );
        let mapped = layers
            .iter()
            .enumerate()
            .map(|(layer, input)| {
                MappedLayer::open(input, &crate::mdp_policy_u16_path(policy_dir, layer), graph)
            })
            .collect::<Result<Vec<_>>>()?;
        let result = Self {
            graph,
            qfwd,
            formulas,
            layers: mapped,
            qstate_count: qfwd.offsets.len(),
            available_pieces,
            two_line_field,
            fully_validated: false,
        };
        result.check_inputs_unchanged()?;
        Ok(result)
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        for input in &self.layers {
            input.check()?;
        }
        Ok(())
    }

    /// Drop resident PTEs from the read-only source mappings, including pages
    /// charged to another cgroup. This does not evict the system page cache.
    pub(crate) fn release_source_pages(&self) -> Result<()> {
        for layer in &self.layers {
            layer
                .index
                .release_pages(0, layer.index.bytes().len() as u64)?;
            layer
                .membership
                .release_pages(0, layer.membership.bytes().len() as u64)?;
            layer
                .policy
                .release_pages(0, layer.policy.bytes().len() as u64)?;
        }
        Ok(())
    }

    /// Release a checked contiguous state span plus 128 KiB fault-around
    /// fringes. Adjacent readers safely refault the immutable shared pages.
    pub(crate) fn release_source_range(&self, depth: usize, start: u64, count: u64) -> Result<()> {
        let layer = self
            .layers
            .get(depth)
            .context("mapped release layer outside domain")?;
        let end = start
            .checked_add(count)
            .context("mapped state release overflow")?;
        ensure!(
            end <= layer.total_states,
            "mapped state release exceeds layer"
        );
        layer.membership.release_pages_with_fringe(
            start.checked_mul(3).context("release offset overflow")?,
            count.checked_mul(3).context("release length overflow")?,
        )?;
        layer.policy.release_pages_with_fringe(
            start.checked_mul(2).context("release offset overflow")?,
            count.checked_mul(2).context("release length overflow")?,
        )
    }

    pub(crate) fn root_count(&self) -> u64 {
        self.layers[0].total_states
    }

    pub(crate) fn fully_validated(&self) -> bool {
        self.fully_validated
    }

    /// Validate *every* stored policy, including states unreachable from L0.
    /// Four workers scan contiguous field groups with bounded metadata. A
    /// single availability byte per quotient state is shared across all scans.
    pub(crate) fn validate_all(&mut self, progress: bool) -> Result<u64> {
        self.fully_validated = false;
        self.check_inputs_unchanged()?;
        self.release_source_pages()?;
        let start = Instant::now();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(MAX_WORKERS)
            .build()?;
        let completed = AtomicU64::new(0);
        let active_layer = AtomicU64::new(0);
        let stop = (Mutex::new(false), Condvar::new());
        let states = std::thread::scope(|scope| {
            if progress {
                scope.spawn(|| {
                    let mut stopped = stop.0.lock().unwrap();
                    while !*stopped {
                        let result = stop.1.wait_timeout(stopped, Duration::from_secs(10)).unwrap();
                        stopped = result.0;
                        if !*stopped {
                            let states = completed.load(Ordering::Relaxed);
                            eprintln!("loop1-sparse validate layer={} states={states} logical_read_bytes={} workers={MAX_WORKERS} seconds={:.1}",
                                active_layer.load(Ordering::Relaxed), states * 5, start.elapsed().as_secs_f64());
                        }
                    }
                });
            }
            let result = (|| -> Result<u64> {
                let mut states = 0u64;
                for (depth, layer) in self.layers.iter().enumerate() {
                    active_layer.store(depth as u64, Ordering::Relaxed);
                    let ranges = validation_ranges(layer.metas);
                    let count = pool.install(|| {
                        ranges
                            .par_iter()
                            .map(|range| -> Result<u64> {
                                let result = self.validate_field_range(depth, range.clone());
                                let first = &layer.metas[range.start];
                                let last = &layer.metas[range.end - 1];
                                let end = last.chain_offset + u64::from(last.count);
                                self.release_source_range(
                                    depth,
                                    first.chain_offset,
                                    end - first.chain_offset,
                                )?;
                                let count = result?;
                                completed.fetch_add(count, Ordering::Relaxed);
                                Ok(count)
                            })
                            .try_reduce(
                                || 0u64,
                                |left, right| {
                                    left.checked_add(right)
                                        .context("validated state count overflow")
                                },
                            )
                    })?;
                    ensure!(
                        count == layer.total_states,
                        "validated layer state count mismatch"
                    );
                    states = states
                        .checked_add(count)
                        .context("validated state count overflow")?;
                    if progress {
                        eprintln!("loop1-sparse validate layer={depth} complete states={states} workers={MAX_WORKERS} seconds={:.1}", start.elapsed().as_secs_f64());
                    }
                }
                Ok(states)
            })();
            *stop.0.lock().unwrap() = true;
            stop.1.notify_all();
            result
        });
        self.release_source_pages()?;
        let states = states?;
        self.check_inputs_unchanged()?;
        self.fully_validated = true;
        Ok(states)
    }

    fn validate_field_range(&self, depth: usize, range: std::ops::Range<usize>) -> Result<u64> {
        let layer = &self.layers[depth];
        let mut states = 0u64;
        for meta in &layer.metas[range] {
            // Whole-domain offsets, targets and formula identities were checked
            // on open. Cache the field's seven spans, then use the shared mask
            // instead of decoding an Action and calling the formula trait.
            let offsets = &self.graph.offsets[meta.field_id as usize];
            let starts: [usize; PIECE_COUNT] = std::array::from_fn(|piece| offsets[piece] as usize);
            let degrees: [usize; PIECE_COUNT] =
                std::array::from_fn(|piece| (offsets[piece + 1] - offsets[piece]) as usize);
            let mut previous = None;
            for i in 0..u64::from(meta.count) {
                let at = meta.chain_offset + i;
                let qid = layer.qid(at);
                ensure!(
                    (qid as usize) < self.qstate_count && previous.is_none_or(|p| p < qid),
                    "invalid membership at layer {depth} field {} offset {i}",
                    meta.field_id
                );
                previous = Some(qid);
                let code = layer.code(at);
                if code != u16::MAX {
                    let placed = (code >> 8) as usize;
                    ensure!(
                        placed < PIECE_COUNT,
                        "invalid policy piece layer={depth} field={} qid={qid}",
                        meta.field_id
                    );
                    let edge = (code & 255) as usize;
                    ensure!(
                        edge < degrees[placed],
                        "invalid policy edge layer={depth} field={} qid={qid}",
                        meta.field_id
                    );
                    ensure!(
                        self.available_pieces[qid as usize] & (1 << placed) != 0,
                        "policy has no reveal records layer={depth} field={} qid={qid}",
                        meta.field_id
                    );
                    if depth == 9 {
                        let target = self.graph.targets[starts[placed] + edge];
                        ensure!(
                            target == EMPTY_FIELD_ID,
                            "L9 policy targets non-terminal field {target} field={} qid={qid}",
                            meta.field_id
                        );
                    }
                }
            }
            states = states
                .checked_add(u64::from(meta.count))
                .context("validated field state count overflow")?;
        }
        Ok(states)
    }

    #[inline]
    fn decode(&self, depth: usize, field: FieldId, qid: u32, code: u16) -> Result<Action> {
        let placed = (code >> 8) as usize;
        ensure!(
            placed < PIECE_COUNT,
            "invalid policy piece layer={depth} field={field} qid={qid}"
        );
        let row = self.graph.offsets[field as usize];
        let edges = &self.graph.targets[row[placed] as usize..row[placed + 1] as usize];
        let target = *edges.get((code & 255) as usize).with_context(|| {
            format!("invalid policy edge layer={depth} field={field} qid={qid}")
        })?;
        let offsets = self.qfwd.offsets[qid as usize];
        let start = offsets[placed] as usize;
        let end = offsets[placed + 1] as usize;
        ensure!(
            start < end,
            "policy has no reveal records layer={depth} field={field} qid={qid}"
        );
        ensure!(
            depth != 9 || target == EMPTY_FIELD_ID,
            "L9 policy targets non-terminal field {target} field={field} qid={qid}"
        );
        let zid = self
            .formulas
            .zid_for_placement(qid, placed as u8)
            .context("validated action formula missing")?;
        Ok(Action {
            placed,
            target,
            zid,
            start,
            end,
            terminal: (depth == 4 && target == self.two_line_field)
                || (depth == 9 && target == EMPTY_FIELD_ID),
        })
    }

    /// Evaluate the first `root_limit` L0 states in authoritative file order;
    /// None means all roots and requires the full policy validation contract.
    /// Numeric memoization is always fresh for this application.
    pub(crate) fn apply(
        &self,
        terminal: &[[f32; PIECE_COUNT]],
        gamma: f32,
        root_limit: Option<usize>,
        limits: SparseLimits,
    ) -> Result<SparseOutput> {
        self.apply_impl(terminal, gamma, root_limit, limits, None)
    }

    /// A coherent gamma=1 application: both the raw L4 terminal table and the
    /// compiled L9 values come from the same checked reset/formula evaluation.
    /// Arbitrary caller-supplied terminal tables retain the original apply API.
    /// This timer includes formula and terminal-table evaluation.
    pub(crate) fn apply_prepared_l9(
        &self,
        prepared: &PreparedL9<'_>,
        reset: &[f32],
        include_reward: bool,
        root_limit: Option<usize>,
        limits: SparseLimits,
    ) -> Result<SparseOutput> {
        let start = Instant::now();
        self.check_inputs_unchanged()?;
        prepared.check_inputs_unchanged()?;
        let terminals = prepared.terminal_values(reset, include_reward)?;
        let table = prepared.terminal_table(&terminals)?;
        let source = prepared.compiled.prepare_f32(&terminals)?;
        let mut result = self.apply_impl(&table, 1.0, root_limit, limits, Some(source))?;
        prepared.check_inputs_unchanged()?;
        drop(table);
        drop(terminals);
        result.stats.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }

    fn apply_impl(
        &self,
        terminal: &[[f32; PIECE_COUNT]],
        gamma: f32,
        root_limit: Option<usize>,
        limits: SparseLimits,
        compiled_l9: Option<PreparedL9Values<'_>>,
    ) -> Result<SparseOutput> {
        let start = Instant::now();
        self.check_inputs_unchanged()?;
        self.release_source_pages()?;
        if let Some(source) = compiled_l9 {
            ensure!(gamma == 1.0, "sparse compiled L9 requires gamma=1");
            let layer = &self.layers[9];
            source.check_sparse_binding(
                self.graph,
                self.qfwd,
                self.formulas,
                &layer.index.file,
                &layer.membership.file,
                &layer.policy.file,
            )?;
        }
        ensure!(
            (0.0..=1.0).contains(&gamma),
            "gamma must be finite and in [0,1]"
        );
        ensure!(
            (1..=MAX_WORKERS).contains(&limits.workers),
            "sparse workers must be 1..={MAX_WORKERS}"
        );
        ensure!(
            limits.total_memo_bytes <= MAX_MEMO_BYTES,
            "sparse memo exceeds 8 GiB bounded experiment cap"
        );
        ensure!(
            terminal.len() == self.qstate_count && terminal.iter().flatten().all(|v| v.is_finite()),
            "invalid terminal table"
        );
        let total = usize::try_from(self.root_count()).context("L0 count exceeds usize")?;
        let count = root_limit.unwrap_or(total).min(total);
        ensure!(root_limit.is_some() || self.fully_validated,
            "all-root evaluation requires validate_all; reachable-only validation is a bounded probe");
        ensure!(
            count <= 4_000_000,
            "L0 output exceeds bounded experiment limit"
        );
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .context("allocate bounded L0 output")?;
        values.resize(count, 0.0f32);
        if count == 0 {
            return Ok(SparseOutput {
                values,
                stats: SparseStats {
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                    ..SparseStats::default()
                },
            });
        }
        let workers = limits.workers.min(count);
        let chunk = count.div_ceil(workers);
        let tasks = count.div_ceil(chunk);
        let per_worker_bytes = limits.total_memo_bytes / tasks as u64;
        // Allocate all caches before starting workers; failure cannot leave a
        // partially published result. Fixed validation tags add <=128 KiB.
        let mut caches = Vec::with_capacity(tasks);
        for _ in 0..tasks {
            caches.push(WorkerCaches::new(per_worker_bytes)?);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let progress = Progress::default();
        let stop = (Mutex::new(false), Condvar::new());
        let results = std::thread::scope(|scope| {
            if limits.progress {
                scope.spawn(|| {
                    let mut stopped = stop.0.lock().unwrap();
                    while !*stopped {
                        let result = stop.1.wait_timeout(stopped, Duration::from_secs(10)).unwrap();
                        stopped = result.0;
                        if !*stopped {
                            eprintln!("loop1-sparse apply roots={}/{} visits={} evaluated={} state_hits={} action_hits={} collisions={} seconds={:.1}",
                                progress.roots.load(Ordering::Relaxed), count,
                                progress.visits.load(Ordering::Relaxed), progress.evaluated.load(Ordering::Relaxed),
                                progress.state_hits.load(Ordering::Relaxed), progress.action_hits.load(Ordering::Relaxed),
                                progress.collisions.load(Ordering::Relaxed), start.elapsed().as_secs_f64());
                        }
                    }
                });
            }
            let result = pool.install(|| {
                values
                    .par_chunks_mut(chunk)
                    .zip(caches.into_par_iter())
                    .enumerate()
                    .map(|(task, (output, caches))| {
                        let mut worker = Worker {
                            policy: self,
                            terminal,
                            gamma,
                            caches,
                            stats: SparseStats::default(),
                            published: SparseStats::default(),
                            progress: &progress,
                            compiled_l9: compiled_l9
                                .map(|source| source.lookup(0, true))
                                .transpose()?,
                            source_touch_window: SourceTouchWindow::new()?,
                        };
                        for (i, value) in output.iter_mut().enumerate() {
                            let root = task * chunk + i;
                            let layer = &self.layers[0];
                            let block = layer.metas.partition_point(|m| {
                                m.chain_offset + u64::from(m.count) <= root as u64
                            });
                            let meta = layer
                                .metas
                                .get(block)
                                .context("root offset outside metadata")?;
                            worker.record_source_read(
                                0,
                                SourceKind::Membership,
                                root as u64 * 3,
                            )?;
                            let qid = layer.qid(root as u64);
                            *value = worker.eval(0, meta.field_id, qid)?;
                            worker.stats.roots += 1;
                            worker.publish();
                        }
                        worker.stats.memo_payload_bytes = worker.caches.payload_bytes();
                        worker.stats.field_validation_tag_bytes =
                            (worker.caches.validated_fields.capacity() * size_of::<u64>()) as u64;
                        worker.source_touch_window.flush(self)?;
                        worker.stats.source_page_releases =
                            worker.source_touch_window.flush_count();
                        worker.stats.source_retired_ranges =
                            worker.source_touch_window.range_count();
                        worker.stats.source_touch_records =
                            worker.source_touch_window.record_count();
                        worker.stats.source_touch_unique_buckets =
                            worker.source_touch_window.unique_record_count();
                        worker.stats.source_touch_dedup_hits =
                            worker.source_touch_window.dedup_hit_count();
                        worker.stats.source_touch_window_bytes =
                            worker.source_touch_window.storage_bytes();
                        worker.stats.source_retire_seconds =
                            worker.source_touch_window.retire_seconds();
                        if let Some(lookup) = worker.compiled_l9.as_ref() {
                            worker.stats.compiled_l9_membership_bytes =
                                lookup.io_stats().membership_bytes_read;
                        }
                        worker.publish();
                        Ok(worker.stats)
                    })
                    .collect::<Result<Vec<_>>>()
            });
            *stop.0.lock().unwrap() = true;
            stop.1.notify_all();
            result
        });
        self.release_source_pages()?;
        let results = results?;
        self.check_inputs_unchanged()?;
        if let Some(source) = compiled_l9 {
            source.check_inputs_unchanged()?;
        }
        let mut stats = SparseStats::default();
        for result in results {
            stats.add(&result);
        }
        stats.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(SparseOutput { values, stats })
    }
}

#[derive(Clone, Copy)]
struct Action {
    placed: usize,
    target: FieldId,
    zid: u32,
    start: usize,
    end: usize,
    terminal: bool,
}

fn validate_domain(
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    pre: &dyn TerminalFormulaIndex,
) -> Result<()> {
    ensure!(
        !graph.offsets.is_empty()
            && graph.offsets.len() == graph.hashes.len()
            && graph.offsets.len() < (1 << 24),
        "invalid sparse graph domain"
    );
    let mut cursor = 0usize;
    for row in &graph.offsets {
        ensure!(row[0] as usize == cursor, "noncontiguous graph offsets");
        for p in 0..PIECE_COUNT {
            let end = row[p + 1] as usize;
            ensure!(
                end >= cursor && end <= graph.targets.len(),
                "invalid graph offsets"
            );
            cursor = end;
        }
    }
    ensure!(cursor == graph.targets.len(), "unreferenced graph targets");
    ensure!(
        graph
            .targets
            .iter()
            .all(|&field| (field as usize) < graph.offsets.len() || field == EMPTY_FIELD_ID),
        "invalid graph target"
    );
    ensure!(
        !qfwd.offsets.is_empty()
            && qfwd.offsets.len() < (1 << 24)
            && qfwd.skipped_invalid_child == 0,
        "invalid sparse qforward domain"
    );
    ensure!(
        pre.terminal_count() <= u32::MAX as usize,
        "formula domain exceeds u32"
    );
    cursor = 0;
    for (qid, row) in qfwd.offsets.iter().enumerate() {
        ensure!(row[0] as usize == cursor, "noncontiguous qforward offsets");
        for placed in 0..PIECE_COUNT {
            let end = row[placed + 1] as usize;
            ensure!(
                end >= cursor && end <= qfwd.records.len(),
                "invalid qforward offsets"
            );
            let records = &qfwd.records[cursor..end];
            ensure!(records.len() <= PIECE_COUNT, "too many reveals");
            let mut previous = None;
            for rec in records {
                ensure!(
                    (rec.child_qid as usize) < qfwd.offsets.len()
                        && (rec.reveal as usize) < PIECE_COUNT
                        && previous.is_none_or(|r| r < rec.reveal),
                    "invalid ordered qforward support"
                );
                previous = Some(rec.reveal);
            }
            match pre.zid_for_placement(qid as u32, placed as u8) {
                Some(zid) => {
                    ensure!(
                        (zid as usize) < pre.terminal_count() && !records.is_empty(),
                        "invalid placement formula ID"
                    );
                    let children = pre.reveal_qids(zid);
                    ensure!(
                        children.len() == records.len()
                            && children
                                .iter()
                                .zip(records)
                                .all(|(&q, rec)| q == rec.child_qid),
                        "placement formula changes ordered children"
                    );
                }
                None => ensure!(records.is_empty(), "missing placement formula"),
            }
            cursor = end;
        }
    }
    ensure!(
        cursor == qfwd.records.len(),
        "unreferenced qforward records"
    );
    Ok(())
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    key: u64,
    value: f32,
    _padding: u32,
}

struct DirectCache {
    slots: Vec<Slot>,
}

fn hash(key: u64) -> u64 {
    let mut x = key;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

impl DirectCache {
    fn new(bytes: u64) -> Result<Self> {
        let maximum =
            usize::try_from(bytes / size_of::<Slot>() as u64).context("cache capacity overflow")?;
        let count = if maximum == 0 {
            0
        } else {
            1usize << maximum.ilog2()
        };
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .context("allocate bounded sparse cache")?;
        slots.resize(
            count,
            Slot {
                key: EMPTY_KEY,
                value: 0.0,
                _padding: 0,
            },
        );
        Ok(Self { slots })
    }

    #[inline]
    fn get(&self, key: u64) -> Option<f32> {
        if self.slots.is_empty() {
            return None;
        }
        let slot = self.slots[hash(key) as usize & (self.slots.len() - 1)];
        (slot.key == key).then_some(slot.value)
    }

    #[inline]
    fn insert(&mut self, key: u64, value: f32) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let index = hash(key) as usize & (self.slots.len() - 1);
        let old = self.slots[index].key;
        self.slots[index] = Slot {
            key,
            value,
            _padding: 0,
        };
        old != EMPTY_KEY && old != key
    }
}

struct WorkerCaches {
    states: DirectCache,
    actions: DirectCache,
    validated_fields: Vec<u64>,
}

impl WorkerCaches {
    fn new(bytes: u64) -> Result<Self> {
        let actions = DirectCache::new(bytes / 4)?;
        let states = DirectCache::new(bytes - bytes / 4)?;
        let mut validated_fields = Vec::new();
        validated_fields.try_reserve_exact(FIELD_VALIDATION_SLOTS)?;
        validated_fields.resize(FIELD_VALIDATION_SLOTS, EMPTY_KEY);
        Ok(Self {
            states,
            actions,
            validated_fields,
        })
    }

    fn payload_bytes(&self) -> u64 {
        ((self.states.slots.capacity() + self.actions.slots.capacity()) * size_of::<Slot>()) as u64
    }
}

#[derive(Default)]
struct Progress {
    roots: AtomicU64,
    visits: AtomicU64,
    evaluated: AtomicU64,
    state_hits: AtomicU64,
    action_hits: AtomicU64,
    collisions: AtomicU64,
}

struct Worker<'a, 'p> {
    policy: &'a SparsePolicy<'p>,
    terminal: &'a [[f32; PIECE_COUNT]],
    gamma: f32,
    caches: WorkerCaches,
    stats: SparseStats,
    published: SparseStats,
    progress: &'a Progress,
    compiled_l9: Option<CompiledL9Lookup<'a>>,
    source_touch_window: SourceTouchWindow,
}

impl Worker<'_, '_> {
    fn record_source_read(&mut self, depth: usize, kind: SourceKind, offset: u64) -> Result<()> {
        self.source_touch_window
            .record(self.policy, depth, kind, offset)
    }
    fn publish(&mut self) {
        self.progress
            .roots
            .fetch_add(self.stats.roots - self.published.roots, Ordering::Relaxed);
        self.progress.visits.fetch_add(
            self.stats.state_requests - self.published.state_requests,
            Ordering::Relaxed,
        );
        self.progress.evaluated.fetch_add(
            self.stats.evaluated_states - self.published.evaluated_states,
            Ordering::Relaxed,
        );
        self.progress.state_hits.fetch_add(
            self.stats.state_hits - self.published.state_hits,
            Ordering::Relaxed,
        );
        self.progress.action_hits.fetch_add(
            self.stats.action_hits - self.published.action_hits,
            Ordering::Relaxed,
        );
        self.progress.collisions.fetch_add(
            self.stats.state_collisions + self.stats.action_collisions
                - self.published.state_collisions
                - self.published.action_collisions,
            Ordering::Relaxed,
        );
        self.published = self.stats.clone();
    }

    /// The membership of each encountered field is validated before binary
    /// search even in reachable-only probes. Tags are bounded and may evict.
    fn find(&mut self, depth: usize, field: FieldId, qid: u32) -> Result<Option<u64>> {
        let policy = self.policy;
        let layer = &policy.layers[depth];
        let Ok(index) = layer.metas.binary_search_by_key(&field, |m| m.field_id) else {
            return Ok(None);
        };
        let meta = &layer.metas[index];
        if !self.policy.fully_validated {
            let key = ((depth as u64) << 32) | u64::from(field);
            let slot = hash(key) as usize & (FIELD_VALIDATION_SLOTS - 1);
            if self.caches.validated_fields[slot] != key {
                let mut previous = None;
                for start in
                    (0..u64::from(meta.count)).step_by(MEMBERSHIP_VALIDATION_CHUNK_STATES as usize)
                {
                    let count =
                        (u64::from(meta.count) - start).min(MEMBERSHIP_VALIDATION_CHUNK_STATES);
                    for i in start..start + count {
                        let member = layer.qid(meta.chain_offset + i);
                        ensure!(
                            (member as usize) < policy.qstate_count
                                && previous.is_none_or(|p| p < member),
                            "invalid reachable membership layer={depth} field={field}"
                        );
                        previous = Some(member);
                    }
                    policy.release_source_range(depth, meta.chain_offset + start, count)?;
                }
                self.stats.reachable_validation_bytes += u64::from(meta.count) * 3;
                self.caches.validated_fields[slot] = key;
            }
        }
        let mut lo = 0u64;
        let mut hi = u64::from(meta.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            self.record_source_read(depth, SourceKind::Membership, (meta.chain_offset + mid) * 3)?;
            let member = layer.qid(meta.chain_offset + mid);
            self.stats.membership_bytes += 3;
            if member < qid {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < u64::from(meta.count) {
            self.record_source_read(depth, SourceKind::Membership, (meta.chain_offset + lo) * 3)?;
            self.stats.membership_bytes += 3;
            if layer.qid(meta.chain_offset + lo) == qid {
                return Ok(Some(meta.chain_offset + lo));
            }
        }
        Ok(None)
    }

    fn eval(&mut self, depth: usize, field: FieldId, qid: u32) -> Result<f32> {
        self.stats.state_requests += 1;
        if self.stats.state_requests & ((1 << 16) - 1) == 0 {
            self.source_touch_window.flush(self.policy)?;
            self.publish();
        }
        if depth == 9 {
            if let Some(lookup) = self.compiled_l9.as_mut() {
                self.stats.compiled_l9_reveals += 1;
                return lookup.get(field, qid);
            }
        }
        let key = ((depth as u64) << 56) | (u64::from(field) << 32) | u64::from(qid);
        if let Some(value) = self.caches.states.get(key) {
            self.stats.state_hits += 1;
            return Ok(value);
        }
        let Some(index) = self.find(depth, field, qid)? else {
            self.stats.missing_children += 1;
            self.stats.state_collisions += u64::from(self.caches.states.insert(key, 0.0));
            return Ok(0.0);
        };
        self.stats.evaluated_states += 1;
        self.stats.policy_bytes += 2;
        self.record_source_read(depth, SourceKind::Policy, index * 2)?;
        let code = self.policy.layers[depth].code(index);
        let value = if code == u16::MAX {
            0.0
        } else {
            let action = self.policy.decode(depth, field, qid, code)?;
            if action.terminal {
                self.terminal[qid as usize][action.placed]
            } else {
                let action_key = ((depth as u64) << 56)
                    | (u64::from(action.target) << 32)
                    | u64::from(action.zid);
                if let Some(value) = self.caches.actions.get(action_key) {
                    self.stats.action_hits += 1;
                    value
                } else {
                    let sum = if depth == 8 && self.compiled_l9.is_some() {
                        // The existing lookup resolves one field, then sums in
                        // original reveal order with +0 for missing members.
                        let (sum, zero_terms) = self.compiled_l9.as_mut().unwrap().sum_recs(
                            action.target,
                            &self.policy.qfwd.records[action.start..action.end],
                        )?;
                        self.stats.compiled_l9_averages += 1;
                        self.stats.compiled_l9_reveals += (action.end - action.start) as u64;
                        self.stats.compiled_l9_zero_terms += zero_terms;
                        sum
                    } else {
                        let mut sum = 0.0f32;
                        for i in action.start..action.end {
                            let child = self.policy.qfwd.records[i].child_qid;
                            sum += self.eval(depth + 1, action.target, child)?;
                        }
                        sum
                    };
                    let scaled = if self.gamma == 1.0 {
                        sum
                    } else {
                        self.gamma * sum
                    };
                    let value = scaled / (action.end - action.start) as f32;
                    ensure!(
                        value.is_finite(),
                        "nonfinite sparse state layer={depth} field={field} qid={qid}"
                    );
                    self.stats.action_collisions +=
                        u64::from(self.caches.actions.insert(action_key, value));
                    value
                }
            }
        };
        self.stats.state_collisions += u64::from(self.caches.states.insert(key, value));
        Ok(value)
    }
}

/// Standalone experiment. It never publishes a solver-accepted value file.
pub(crate) fn run(args: &[String]) -> Result<()> {
    let setup_start = Instant::now();
    ensure!((7..=13).contains(&args.len()),
        "usage: loop1-sparse-probe <graph> <pruned> <policy> <x.f32|zero> <new-y.f32> [roots=32;0=all] [memo_mib=4096] [validation=reachable|full] [mode=affine|linear] [gamma=1] [workers=4]");
    let graph_path = PathBuf::from(&args[2]);
    let pruned_dir = PathBuf::from(&args[3]);
    let policy_dir = PathBuf::from(&args[4]);
    let initial = &args[5];
    let output = PathBuf::from(&args[6]);
    let roots: usize = args
        .get(7)
        .map_or(Ok(32), |s| s.parse())
        .context("invalid roots")?;
    let memo_mib: u64 = args
        .get(8)
        .map_or(Ok(4096), |s| s.parse())
        .context("invalid memo_mib")?;
    let validation = args.get(9).map_or("reachable", String::as_str);
    let mode = args.get(10).map_or("affine", String::as_str);
    let gamma: f32 = args
        .get(11)
        .map_or(Ok(1.0), |s| s.parse())
        .context("invalid gamma")?;
    let workers: usize = args
        .get(12)
        .map_or(Ok(4), |s| s.parse())
        .context("invalid workers")?;
    let use_compiled_l9 = std::env::var("PI_LOOP1_SPARSE_COMPILED_L9")
        .ok()
        .is_some_and(|value| {
            value != "0"
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("no")
        });
    ensure!(
        validation == "reachable" || validation == "full",
        "validation must be reachable or full"
    );
    ensure!(
        mode == "affine" || mode == "linear",
        "mode must be affine or linear"
    );
    ensure!(
        roots != 0 || validation == "full",
        "roots=0 requires validation=full"
    );
    ensure!(
        (0.0..=1.0).contains(&gamma),
        "gamma must be finite and in [0,1]"
    );
    ensure!(
        !use_compiled_l9 || gamma == 1.0,
        "PI_LOOP1_SPARSE_COMPILED_L9 requires gamma=1"
    );
    ensure!(
        (1..=MAX_WORKERS).contains(&workers),
        "workers must be 1..=4"
    );
    let memo_bytes = memo_mib
        .checked_mul(1 << 20)
        .context("memo size overflow")?;
    ensure!(memo_bytes <= MAX_MEMO_BYTES, "memo_mib must be <=8192");
    ensure!(
        !output.try_exists()?,
        "output must be a new path: {}",
        output.display()
    );
    let mut input_paths = vec![graph_path.clone()];
    for depth in 0..10 {
        let prefix = crate::q_pruned_prefix(&pruned_dir, depth);
        input_paths.extend([
            prefix.with_extension("field_index.bin"),
            prefix.with_extension("chains.u24.bin"),
            crate::mdp_policy_u16_path(&policy_dir, depth),
        ]);
    }
    if initial != "zero" {
        input_paths.push(PathBuf::from(initial));
    }
    // Membership/policy/index bindings below each own their mutation watch.
    // Guard graph and reset before parsing, including same-filesystem-tick writes.
    #[cfg(target_os = "linux")]
    let external_watches = std::iter::once(graph_path.clone())
        .chain((initial != "zero").then(|| PathBuf::from(initial)))
        .map(|path| Ok((path.clone(), InputMutationWatch::open(&path)?)))
        .collect::<Result<Vec<_>>>()?;
    let bindings = crate::loop1_prepared::SourceBindings::capture_paths(input_paths.clone())?;
    let graph = FieldGraph::load_graph_bin(&graph_path)?;
    let old = crate::enumerate_states();
    let q_index = crate::build_q_index_from_old(&old);
    drop(old);
    let qfwd = crate::build_qforward_graph(&q_index);
    let pre = crate::pre_reveal::PreRevealIndex::build(&q_index)?;
    // Enforce the production quotient/reveal semantics once, before using IDs.
    let _terminal_plan = if use_compiled_l9 {
        None
    } else {
        Some(crate::loop1_terminal::TerminalPlan::build(
            &q_index, &qfwd, &pre,
        )?)
    };
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("missing TWO_LINE_HASH")?;
    let layers = (0..10)
        .map(|depth| FieldChainLayer::open_index_only(crate::q_pruned_prefix(&pruned_dir, depth)))
        .collect::<Result<Vec<_>>>()?;
    let mut policy = SparsePolicy::open(&graph, &qfwd, &pre, &layers, &policy_dir, two_line)?;
    bindings.check()?;
    #[cfg(target_os = "linux")]
    for (path, watch) in &external_watches {
        watch.check(path)?;
    }
    ensure!(
        layers[0].metas.len() == 1 && graph.hash(layers[0].metas[0].field_id) == 0,
        "sparse probe requires one physical-empty L0 block"
    );
    ensure!(
        layers[0].total_states <= 4_000_000,
        "L0 exceeds bounded probe output cap"
    );
    let mut reset = vec![0.0f32; q_index.states.len()];
    if initial != "zero" {
        // L0 is physical empty field (hash 0), not the distinct PC sentinel.
        let input = crate::f32_ram::load(Path::new(initial), layers[0].total_states)?;
        let mut previous = None;
        for (i, value) in input.into_iter().enumerate() {
            ensure!(value.is_finite(), "nonfinite reset value at {i}");
            let qid = policy.layers[0].qid(i as u64);
            ensure!(
                (qid as usize) < reset.len() && previous.is_none_or(|p| p < qid),
                "L0 qids must be sorted, unique and in domain"
            );
            previous = Some(qid);
            reset[qid as usize] = value;
        }
    }
    bindings.check()?;
    #[cfg(target_os = "linux")]
    for (path, watch) in &external_watches {
        watch.check(path)?;
    }
    let compiled_start = Instant::now();
    let prepared_l9 = if use_compiled_l9 {
        Some(PreparedL9::build(
            &graph,
            &q_index,
            &qfwd,
            &pre,
            &layers,
            &policy_dir,
            crate::loop1_prepared::SourceBindings::capture_paths(input_paths)?,
        )?)
    } else {
        None
    };
    if let Some(prepared) = prepared_l9.as_ref() {
        eprintln!(
            "loop1-sparse compiled_l9_setup_seconds={:.6} compiled_l9_resident_bytes={}",
            compiled_start.elapsed().as_secs_f64(),
            prepared.storage_bytes()
        );
    }
    if validation == "full" {
        policy.validate_all(true)?;
    }
    eprintln!("loop1-sparse scope={} roots={} validation={} mode={} gamma={} workers={} memo_cap_bytes={} mmap_virtual_bytes={} compiled_l9={} setup_seconds={:.6} (logical reads do not measure physical IO)",
        if roots == 0 { "all-ten-layer-boundary" } else { "bounded-root-probe" },
        if roots == 0 { policy.root_count() } else { (roots as u64).min(policy.root_count()) },
        validation, mode, gamma, workers, memo_bytes,
        policy.layers.iter().map(|l| l.total_states * 5).sum::<u64>(), use_compiled_l9,
        setup_start.elapsed().as_secs_f64());
    let numeric_start = Instant::now();
    let limits = SparseLimits {
        workers,
        total_memo_bytes: memo_bytes,
        progress: true,
    };
    let result = if let Some(prepared) = prepared_l9.as_ref() {
        policy.apply_prepared_l9(
            prepared,
            &reset,
            mode == "affine",
            (roots != 0).then_some(roots),
            limits,
        )?
    } else {
        let terminal = crate::build_terminal_reset_avg_table_opt(
            &qfwd,
            &reset,
            gamma,
            q_index.states.len(),
            mode == "affine",
        );
        policy.apply(&terminal, gamma, (roots != 0).then_some(roots), limits)?
    };
    eprintln!(
        "loop1-sparse terminal_and_apply_seconds={:.6}",
        numeric_start.elapsed().as_secs_f64()
    );
    bindings.check()?;
    #[cfg(target_os = "linux")]
    for (path, watch) in &external_watches {
        watch.check(path)?;
    }
    if let Some(prepared) = prepared_l9.as_ref() {
        prepared.check_inputs_unchanged()?;
    }
    // create_new prevents overwriting any source or accepted solver value.
    // The adjacent temporary is also exclusive, then hard-link publication
    // provides atomic, no-replace creation of this new experiment artifact.
    let temporary = output.with_extension(format!("sparse-tmp-{}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let publication = (|| -> Result<()> {
        let mut writer = BufWriter::new(file);
        for value in &result.values {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        bindings.check()?;
        #[cfg(target_os = "linux")]
        for (path, watch) in &external_watches {
            watch.check(path)?;
        }
        policy.check_inputs_unchanged()?;
        if let Some(prepared) = prepared_l9.as_ref() {
            prepared.check_inputs_unchanged()?;
        }
        fs::hard_link(&temporary, &output)
            .with_context(|| format!("publish new probe output {}", output.display()))?;
        Ok(())
    })();
    let cleanup = fs::remove_file(&temporary);
    publication?;
    cleanup?;
    eprintln!(
        "loop1-sparse result {:?} output={}",
        result.stats,
        output.display()
    );
    Ok(())
}

#[cfg(test)]
#[path = "loop1_sparse_tests.rs"]
mod tests;
