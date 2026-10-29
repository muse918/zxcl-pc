#![allow(dead_code)]

use anyhow::{bail, ensure, Context, Result};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};

use crate::graph::FieldId;

const MAGIC: &[u8; 8] = b"FCHNIDX1";

#[derive(Debug, Clone)]
pub struct FieldChainBlockMeta {
    pub field_id: FieldId,
    pub count: u32,
    pub chain_offset: u64,
}

#[derive(Debug)]
pub struct FieldChainLayer {
    pub metas: Vec<FieldChainBlockMeta>,
    pub total_states: u64,
    pub index_path: PathBuf,
    pub chains_path: PathBuf,
    // For small layers, chains_bytes is populated by open().
    // For large q layers, open_index_only() leaves chains_bytes empty and uses
    // chains_file for per-block positional reads. This avoids keeping a 50+ GiB
    // chains.u24.bin resident while the frontier accumulators are live.
    pub chains_bytes: Vec<u8>,
    chains_file: Option<File>,
}

#[inline]
pub fn read_u24_le_at(bytes: &[u8], elem_idx: u64) -> u32 {
    let p = elem_idx as usize * 3;
    (bytes[p] as u32) | ((bytes[p + 1] as u32) << 8) | ((bytes[p + 2] as u32) << 16)
}

#[inline]
pub fn write_u24_le<W: Write>(w: &mut W, x: u32) -> Result<()> {
    ensure!(x < (1u32 << 24), "chain id exceeds u24 domain: {x}");
    w.write_all(&[(x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8, ((x >> 16) & 0xFF) as u8])?;
    Ok(())
}

#[inline]
fn read_u32_le(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() {
        bail!("truncated u32");
    }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_u64_le(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos + 8 > bytes.len() {
        bail!("truncated u64");
    }
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

impl FieldChainLayer {
    pub fn open(prefix: impl AsRef<Path>) -> Result<Self> {
        Self::open_impl(prefix, true)
    }

    pub fn open_index_only(prefix: impl AsRef<Path>) -> Result<Self> {
        Self::open_impl(prefix, false)
    }

    fn open_impl(prefix: impl AsRef<Path>, load_chains_into_ram: bool) -> Result<Self> {
        let prefix = prefix.as_ref();
        let index_path = prefix.with_extension("field_index.bin");
        let chains_path = prefix.with_extension("chains.u24.bin");

        let index_bytes = fs::read(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;

        if index_bytes.len() < 24 || &index_bytes[0..8] != MAGIC {
            bail!("unsupported field-chain index format: {}", index_path.display());
        }

        let mut pos = 8usize;
        let version = read_u32_le(&index_bytes, &mut pos)?;
        if version != 1 {
            bail!("unsupported field-chain index version {version}");
        }

        let block_count = read_u32_le(&index_bytes, &mut pos)? as usize;
        let total_states = read_u64_le(&index_bytes, &mut pos)?;

        let expected_index_bytes = block_count.checked_mul(16)
            .and_then(|n| n.checked_add(24))
            .context("field-chain index size overflow")?;
        if index_bytes.len() != expected_index_bytes {
            bail!(
                "field-chain index size mismatch: got {}, expected {}",
                index_bytes.len(),
                expected_index_bytes
            );
        }

        let mut metas = Vec::with_capacity(block_count);
        let mut prev_field = None;
        let mut sum = 0u64;

        for _ in 0..block_count {
            let field_id = read_u32_le(&index_bytes, &mut pos)?;
            let count = read_u32_le(&index_bytes, &mut pos)?;
            let chain_offset = read_u64_le(&index_bytes, &mut pos)?;

            if let Some(prev) = prev_field {
                if field_id <= prev {
                    bail!("field ids are not strictly sorted in index");
                }
            }
            prev_field = Some(field_id);
            if chain_offset != sum {
                bail!("field-chain offset is not contiguous at field {field_id}: got {chain_offset}, expected {sum}");
            }
            sum = sum.checked_add(count as u64)
                .context("field-chain offset/count overflow")?;
            if sum > total_states {
                bail!("field-chain block exceeds total_states at field {field_id}");
            }

            metas.push(FieldChainBlockMeta { field_id, count, chain_offset });
        }

        if sum != total_states {
            bail!("field-chain total_states mismatch: header={total_states}, sum={sum}");
        }

        let expected_chain_bytes = total_states
            .checked_mul(3)
            .context("chains.u24 expected byte size overflow")?;
        let actual_chain_bytes = fs::metadata(&chains_path)
            .with_context(|| format!("failed to stat {}", chains_path.display()))?
            .len();
        if actual_chain_bytes != expected_chain_bytes {
            bail!(
                "chains.u24 size mismatch: got {}, expected {}",
                actual_chain_bytes,
                expected_chain_bytes
            );
        }

        let (chains_bytes, chains_file) = if load_chains_into_ram {
            let chains_bytes = fs::read(&chains_path)
                .with_context(|| format!("failed to read {}", chains_path.display()))?;
            (chains_bytes, None)
        } else {
            let chains_file = File::open(&chains_path)
                .with_context(|| format!("failed to open {}", chains_path.display()))?;
            (Vec::new(), Some(chains_file))
        };

        Ok(Self { metas, total_states, index_path, chains_path, chains_bytes, chains_file })
    }

    #[cfg(unix)]
    fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = offset.checked_add(done as u64).context("chains.u24 read offset overflow")?;
            let n = match file.read_at(&mut buf[done..], at) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result.with_context(|| format!("failed to read chains.u24 at byte offset {}", at))?,
            };
            if n == 0 {
                bail!("unexpected EOF reading chains.u24 at byte offset {}", at);
            }
            done += n;
        }
        Ok(())
    }

    pub fn read_block_chains_into_with_buf(&self, block_idx: usize, out: &mut Vec<u32>, tmp_bytes: &mut Vec<u8>) {
        let meta = &self.metas[block_idx];
        out.clear();
        out.reserve(meta.count as usize);
        if !self.chains_bytes.is_empty() || meta.count == 0 {
            for i in 0..meta.count as u64 {
                out.push(read_u24_le_at(&self.chains_bytes, meta.chain_offset + i));
            }
            return;
        }

        #[cfg(unix)]
        {
            let file = self.chains_file.as_ref().expect("index-only layer missing chains_file");
            let byte_len = meta.count as usize * 3;
            let byte_off = meta.chain_offset * 3;
            tmp_bytes.resize(byte_len, 0);
            Self::read_exact_at(file, tmp_bytes.as_mut_slice(), byte_off).expect("failed to read q layer block from disk");
            for i in 0..meta.count as u64 {
                out.push(read_u24_le_at(tmp_bytes.as_slice(), i));
            }
            return;
        }

        #[cfg(not(unix))]
        {
            panic!("FieldChainLayer::open_index_only requires unix FileExt positional reads");
        }
    }

    pub fn read_block_chains_into(&self, block_idx: usize, out: &mut Vec<u32>) {
        let mut tmp_bytes = Vec::<u8>::new();
        self.read_block_chains_into_with_buf(block_idx, out, &mut tmp_bytes);
    }


    /// Reads the concatenated u24 chain bytes for a contiguous block range.
    /// Returns the chain_offset of the first block; callers can decode block-local qids by
    /// subtracting this base from each block's chain_offset.
    pub fn read_block_range_chains_bytes_into_with_buf(&self, start: usize, end: usize, tmp_bytes: &mut Vec<u8>) -> u64 {
        assert!(start < end && end <= self.metas.len());
        let first = &self.metas[start];
        let last = &self.metas[end - 1];
        let start_elem = first.chain_offset;
        let end_elem = last.chain_offset + last.count as u64;
        let byte_off = start_elem * 3;
        let byte_len = (end_elem - start_elem) as usize * 3;
        tmp_bytes.resize(byte_len, 0);

        if !self.chains_bytes.is_empty() || byte_len == 0 {
            let b0 = byte_off as usize;
            tmp_bytes.copy_from_slice(&self.chains_bytes[b0..b0 + byte_len]);
            return start_elem;
        }

        #[cfg(unix)]
        {
            let file = self.chains_file.as_ref().expect("index-only layer missing chains_file");
            Self::read_exact_at(file, tmp_bytes.as_mut_slice(), byte_off).expect("failed to read q layer block range from disk");
            return start_elem;
        }

        #[cfg(not(unix))]
        {
            panic!("FieldChainLayer::open_index_only requires unix FileExt positional reads");
        }
    }

    pub fn read_block_chains(&self, block_idx: usize) -> Vec<u32> {
        let meta = &self.metas[block_idx];
        let mut out = Vec::with_capacity(meta.count as usize);
        self.read_block_chains_into(block_idx, &mut out);
        out
    }

    pub fn print_stats(&self, label: &str) {
        let mut counts = self.metas.iter().map(|m| m.count as usize).collect::<Vec<_>>();
        counts.sort_unstable();
        let q = |p: f64| -> usize {
            if counts.is_empty() { return 0; }
            let idx = ((counts.len() - 1) as f64 * p).round() as usize;
            counts[idx]
        };

        println!("{label} index_path={}", self.index_path.display());
        println!("{label} chains_path={}", self.chains_path.display());
        println!("{label} blocks={}", self.metas.len());
        println!("{label} total_states={}", self.total_states);
        if let (Some(first), Some(last)) = (self.metas.first(), self.metas.last()) {
            println!("{label} field_range={}..={}", first.field_id, last.field_id);
        }
        if !counts.is_empty() {
            println!("{label} chains_per_field min={}", counts[0]);
            println!("{label} chains_per_field p50={}", q(0.50));
            println!("{label} chains_per_field p90={}", q(0.90));
            println!("{label} chains_per_field p99={}", q(0.99));
            println!("{label} chains_per_field max={}", counts[counts.len() - 1]);
        }

        let membership_bytes = self.total_states * 3;
        let fp16_bytes = self.total_states * 2;
        println!("{label} u24_membership_bytes={} ({:.3} GiB)", membership_bytes, membership_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("{label} future_fp16_values_bytes={} ({:.3} GiB)", fp16_bytes, fp16_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("{label} future_u24_plus_fp16_bytes={} ({:.3} GiB)", membership_bytes + fp16_bytes, (membership_bytes + fp16_bytes) as f64 / 1024.0 / 1024.0 / 1024.0);
    }
}

pub struct FieldChainWriter {
    index_path: PathBuf,
    chains_writer: BufWriter<File>,
    metas: Vec<FieldChainBlockMeta>,
    total_states: u64,
    open_field: Option<FieldId>,
    open_count: u32,
    open_offset: u64,
    last_chain: Option<u32>,
    failed: bool,
}

const U24_COPY_BUFFER_BYTES: usize = (1024 * 1024 / 3) * 3;

impl FieldChainWriter {
    /// Prevent index publication after a caller rejects shard metadata or an
    /// expected unique count outside the low-level append operation.
    pub(crate) fn invalidate(&mut self) {
        self.failed = true;
    }

    pub fn create(prefix: impl AsRef<Path>) -> Result<Self> {
        let prefix = prefix.as_ref();
        let index_path = prefix.with_extension("field_index.bin");
        let chains_path = prefix.with_extension("chains.u24.bin");
        if let Some(parent) = index_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = chains_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let chains_writer = BufWriter::new(
            File::create(&chains_path)
                .with_context(|| format!("failed to create {}", chains_path.display()))?,
        );
        Ok(Self {
            index_path,
            chains_writer,
            metas: Vec::new(),
            total_states: 0,
            open_field: None,
            open_count: 0,
            open_offset: 0,
            last_chain: None,
            failed: false,
        })
    }

    fn close_open_field(&mut self) {
        if let Some(field_id) = self.open_field.take() {
            self.metas.push(FieldChainBlockMeta {
                field_id,
                count: self.open_count,
                chain_offset: self.open_offset,
            });
            self.open_count = 0;
            self.last_chain = None;
        }
    }

    pub fn push_pair_key(&mut self, pair_key: u64) -> Result<()> {
        ensure!(!self.failed, "field-chain writer previously failed");
        let result = self.push_pair_key_impl(pair_key);
        self.failed |= result.is_err();
        result
    }

    fn check_field_order(&self, field: FieldId) -> Result<()> {
        let previous = self
            .open_field
            .or_else(|| self.metas.last().map(|meta| meta.field_id));
        ensure!(
            previous.is_none_or(|previous| field >= previous),
            "field ids are not sorted: {field} follows {previous:?}"
        );
        Ok(())
    }

    fn begin_field(&mut self, field: FieldId) {
        if self.open_field != Some(field) {
            self.close_open_field();
            self.open_field = Some(field);
            self.open_offset = self.total_states;
        }
    }

    fn push_pair_key_impl(&mut self, pair_key: u64) -> Result<()> {
        let field = (pair_key >> 32) as u32;
        let chain = pair_key as u32;
        ensure!(chain < (1u32 << 24), "chain id exceeds u24 domain: {chain}");
        self.check_field_order(field)?;
        let previous = if self.open_field == Some(field) {
            self.last_chain
        } else {
            None
        };
        if previous == Some(chain) {
            return Ok(());
        }
        ensure!(
            previous.is_none_or(|previous| chain > previous),
            "chain ids are not sorted in field {field}"
        );
        let next_total = self
            .total_states
            .checked_add(1)
            .context("field-chain total state count overflow")?;
        let prior_count = if self.open_field == Some(field) {
            self.open_count
        } else {
            0
        };
        let next_count = prior_count
            .checked_add(1)
            .context("field-chain field count overflow")?;

        self.begin_field(field);
        write_u24_le(&mut self.chains_writer, chain)?;
        self.total_states = next_total;
        self.open_count = next_count;
        self.last_chain = Some(chain);
        Ok(())
    }

    /// Append already encoded, sorted u24 qids for one field. Equal adjacent
    /// qids, including one repeated across calls, are deduplicated exactly as
    /// in push_pair_key. Descending qids or fields and partial u24 records are
    /// rejected. Normal unique input is validated once then copied as bytes.
    /// Returns the number of newly emitted states. Any failure poisons finish
    /// so a caller cannot publish the partially appended stream as an index.
    pub fn append_sorted_u24_block(&mut self, field: FieldId, bytes: &[u8]) -> Result<u64> {
        ensure!(!self.failed, "field-chain writer previously failed");
        let result = self.append_sorted_u24_block_impl(field, bytes);
        self.failed |= result.is_err();
        result
    }

    fn append_sorted_u24_block_impl(&mut self, field: FieldId, bytes: &[u8]) -> Result<u64> {
        ensure!(
            bytes.len() % 3 == 0,
            "encoded u24 block has a partial record"
        );
        self.check_field_order(field)?;
        let initial_previous = if self.open_field == Some(field) {
            self.last_chain
        } else {
            None
        };
        let mut previous = initial_previous;
        let mut unique = 0u64;
        let mut duplicates = false;
        for raw in bytes.chunks_exact(3) {
            let qid = raw[0] as u32 | (raw[1] as u32) << 8 | (raw[2] as u32) << 16;
            ensure!(
                previous.is_none_or(|previous| qid >= previous),
                "chain ids are not sorted in field {field}"
            );
            if previous == Some(qid) {
                duplicates = true;
            } else {
                unique += 1;
            }
            previous = Some(qid);
        }
        if unique == 0 {
            return Ok(0);
        }
        let next_total = self
            .total_states
            .checked_add(unique)
            .context("field-chain total state count overflow")?;
        let prior_count = if self.open_field == Some(field) {
            self.open_count
        } else {
            0
        };
        let added = u32::try_from(unique).context("field-chain field count exceeds u32")?;
        let next_count = prior_count
            .checked_add(added)
            .context("field-chain field count overflow")?;
        self.begin_field(field);
        if !duplicates {
            self.chains_writer.write_all(bytes)?;
        } else {
            let mut previous = initial_previous;
            let mut start = 0;
            for (position, raw) in bytes.chunks_exact(3).enumerate() {
                let qid = raw[0] as u32 | (raw[1] as u32) << 8 | (raw[2] as u32) << 16;
                if previous == Some(qid) {
                    self.chains_writer.write_all(&bytes[start..position * 3])?;
                    start = (position + 1) * 3;
                }
                previous = Some(qid);
            }
            self.chains_writer.write_all(&bytes[start..])?;
        }
        self.total_states = next_total;
        self.open_count = next_count;
        self.last_chain = previous;
        Ok(unique)
    }

    /// Copy a sorted encoded field range with at most one MiB of read scratch.
    /// The byte offset must be u24 aligned and the complete requested payload
    /// must exist. Keep the source immutable until this call returns.
    #[cfg(unix)]
    pub fn append_sorted_u24_file(
        &mut self,
        field: FieldId,
        source: &File,
        byte_offset: u64,
        count: u64,
    ) -> Result<u64> {
        self.append_sorted_u24_file_with_buf(field, source, byte_offset, count, &mut Vec::new())
    }

    /// As append_sorted_u24_file, reusing a bounded buffer across field ranges.
    /// Oversized caller scratch is released before allocation; retained scratch
    /// capacity never exceeds U24_COPY_BUFFER_BYTES.
    #[cfg(unix)]
    pub fn append_sorted_u24_file_with_buf(
        &mut self,
        field: FieldId,
        source: &File,
        byte_offset: u64,
        count: u64,
        scratch: &mut Vec<u8>,
    ) -> Result<u64> {
        ensure!(!self.failed, "field-chain writer previously failed");
        let result = self.append_sorted_u24_file_impl(field, source, byte_offset, count, scratch);
        self.failed |= result.is_err();
        result
    }

    #[cfg(unix)]
    fn append_sorted_u24_file_impl(
        &mut self,
        field: FieldId,
        source: &File,
        byte_offset: u64,
        count: u64,
        scratch: &mut Vec<u8>,
    ) -> Result<u64> {
        self.check_field_order(field)?;
        ensure!(
            byte_offset % 3 == 0,
            "encoded u24 source offset is not record aligned"
        );
        let bytes = count
            .checked_mul(3)
            .context("encoded u24 source byte count overflow")?;
        let end = byte_offset
            .checked_add(bytes)
            .context("encoded u24 source range overflow")?;
        let metadata = source.metadata()?;
        ensure!(
            metadata.is_file(),
            "encoded u24 source is not a regular file"
        );
        ensure!(
            end <= metadata.len(),
            "encoded u24 source payload is truncated: requested end {end}, size {}",
            metadata.len()
        );
        let destination = self.chains_writer.get_ref().metadata()?;
        ensure!(
            metadata.dev() != destination.dev() || metadata.ino() != destination.ino(),
            "encoded u24 source aliases writer output"
        );
        if scratch.capacity() > U24_COPY_BUFFER_BYTES {
            *scratch = Vec::new();
        }
        let buffer_bytes = bytes.min(U24_COPY_BUFFER_BYTES as u64) as usize;
        scratch.clear();
        scratch
            .try_reserve_exact(buffer_bytes)
            .context("allocate encoded u24 copy buffer")?;
        ensure!(
            scratch.capacity() <= U24_COPY_BUFFER_BYTES,
            "encoded u24 copy buffer exceeded capacity bound"
        );
        scratch.resize(buffer_bytes, 0);
        let mut position = 0u64;
        let mut emitted = 0u64;
        while position < bytes {
            let length = (bytes - position).min(buffer_bytes as u64) as usize;
            FieldChainLayer::read_exact_at(source, &mut scratch[..length], byte_offset + position)?;
            emitted = emitted
                .checked_add(self.append_sorted_u24_block_impl(field, &scratch[..length])?)
                .context("encoded u24 emitted state count overflow")?;
            position += length as u64;
        }
        Ok(emitted)
    }

    pub fn finish(mut self) -> Result<u64> {
        ensure!(
            !self.failed,
            "cannot finish field-chain writer after an append failure"
        );
        self.close_open_field();
        let block_count =
            u32::try_from(self.metas.len()).context("field-chain block count exceeds u32")?;
        self.chains_writer.flush()?;

        let mut w = BufWriter::new(
            File::create(&self.index_path)
                .with_context(|| format!("failed to create {}", self.index_path.display()))?,
        );

        w.write_all(MAGIC)?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&block_count.to_le_bytes())?;
        w.write_all(&self.total_states.to_le_bytes())?;

        for meta in &self.metas {
            w.write_all(&meta.field_id.to_le_bytes())?;
            w.write_all(&meta.count.to_le_bytes())?;
            w.write_all(&meta.chain_offset.to_le_bytes())?;
        }
        w.flush()?;
        Ok(self.total_states)
    }
}

#[cfg(test)]
#[path = "layer_store_writer_tests.rs"]
mod writer_tests;

pub fn save_field_chain_from_solver_keys(
    prefix: impl AsRef<Path>,
    keys: &[u64],
    mut unpack: impl FnMut(u64) -> (FieldId, crate::state::ChainState),
    state_index: &crate::state::StateIndex,
) -> Result<u64> {
    let mut writer = FieldChainWriter::create(prefix)?;

    let mut i = 0usize;
    while i < keys.len() {
        let (field_id, _) = unpack(keys[i]);
        let start = i;
        while i < keys.len() {
            let (f, _) = unpack(keys[i]);
            if f != field_id { break; }
            i += 1;
        }
        let end = i;
        let mut chain_ids = Vec::<u32>::with_capacity(end - start);
        for &key in &keys[start..end] {
            let (_field, chain) = unpack(key);
            let cid = *state_index.key_to_id.get(&crate::state::state_key(chain))
                .expect("chain missing in StateIndex") as u32;
            chain_ids.push(cid);
        }
        chain_ids.sort_unstable();
        chain_ids.dedup();
        for cid in chain_ids {
            let pair = ((field_id as u64) << 32) | cid as u64;
            writer.push_pair_key(pair)?;
        }
    }

    writer.finish()
}

#[cfg(test)]
mod index_validation_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn with_layer(offsets: &[u64], total: u64, check: impl FnOnce(&Path)) {
        let dir = std::env::temp_dir().join(format!("zxcl-index-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("layer");
        let mut bytes = MAGIC.to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend((offsets.len() as u32).to_le_bytes());
        bytes.extend(total.to_le_bytes());
        for (field, &offset) in offsets.iter().enumerate() {
            bytes.extend((field as u32).to_le_bytes());
            bytes.extend(1u32.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        fs::write(prefix.with_extension("field_index.bin"), bytes).unwrap();
        fs::write(prefix.with_extension("chains.u24.bin"), vec![0u8; total as usize * 3]).unwrap();
        check(&prefix);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_overlapping_gapped_and_overflowing_offsets() {
        for offsets in [&[0, 0][..], &[0, 2][..], &[1, 2][..], &[0, u64::MAX][..]] {
            with_layer(offsets, 2, |prefix| {
                assert!(FieldChainLayer::open(prefix).is_err());
                assert!(FieldChainLayer::open_index_only(prefix).is_err());
            });
        }
    }

    #[test]
    fn accepts_contiguous_offsets_and_block_range_reads() {
        with_layer(&[0, 1], 2, |prefix| {
            let layer = FieldChainLayer::open_index_only(prefix).unwrap();
            let mut bytes = Vec::new();
            assert_eq!(layer.read_block_range_chains_bytes_into_with_buf(0, 2, &mut bytes), 0);
            assert_eq!(bytes.len(), 6);
        });
    }

    #[test]
    fn rejects_blocks_exceeding_header_state_count() {
        with_layer(&[0, 1], 1, |prefix| {
            assert!(FieldChainLayer::open_index_only(prefix).is_err());
        });
    }
}
