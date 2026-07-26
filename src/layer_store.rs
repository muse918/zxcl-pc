#![allow(dead_code)]

use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
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
    debug_assert!(x < (1u32 << 24));
    w.write_all(&[
        (x & 0xFF) as u8,
        ((x >> 8) & 0xFF) as u8,
        ((x >> 16) & 0xFF) as u8,
    ])?;
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
            bail!(
                "unsupported field-chain index format: {}",
                index_path.display()
            );
        }

        let mut pos = 8usize;
        let version = read_u32_le(&index_bytes, &mut pos)?;
        if version != 1 {
            bail!("unsupported field-chain index version {version}");
        }

        let block_count = read_u32_le(&index_bytes, &mut pos)? as usize;
        let total_states = read_u64_le(&index_bytes, &mut pos)?;

        if index_bytes.len() != 24 + block_count * 16 {
            bail!(
                "field-chain index size mismatch: got {}, expected {}",
                index_bytes.len(),
                24 + block_count * 16
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
            sum += count as u64;

            metas.push(FieldChainBlockMeta {
                field_id,
                count,
                chain_offset,
            });
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

        Ok(Self {
            metas,
            total_states,
            index_path,
            chains_path,
            chains_bytes,
            chains_file,
        })
    }

    #[cfg(unix)]
    fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = offset + done as u64;
            let n = file
                .read_at(&mut buf[done..], at)
                .with_context(|| format!("failed to read chains.u24 at byte offset {}", at))?;
            if n == 0 {
                bail!("unexpected EOF reading chains.u24 at byte offset {}", at);
            }
            done += n;
        }
        Ok(())
    }

    pub fn read_block_chains_into_with_buf(
        &self,
        block_idx: usize,
        out: &mut Vec<u32>,
        tmp_bytes: &mut Vec<u8>,
    ) {
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
            let file = self
                .chains_file
                .as_ref()
                .expect("index-only layer missing chains_file");
            let byte_len = meta.count as usize * 3;
            let byte_off = meta.chain_offset * 3;
            tmp_bytes.resize(byte_len, 0);
            Self::read_exact_at(file, tmp_bytes.as_mut_slice(), byte_off)
                .expect("failed to read q layer block from disk");
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
    pub fn read_block_range_chains_bytes_into_with_buf(
        &self,
        start: usize,
        end: usize,
        tmp_bytes: &mut Vec<u8>,
    ) -> u64 {
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
            let file = self
                .chains_file
                .as_ref()
                .expect("index-only layer missing chains_file");
            Self::read_exact_at(file, tmp_bytes.as_mut_slice(), byte_off)
                .expect("failed to read q layer block range from disk");
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
        let mut counts = self
            .metas
            .iter()
            .map(|m| m.count as usize)
            .collect::<Vec<_>>();
        counts.sort_unstable();
        let q = |p: f64| -> usize {
            if counts.is_empty() {
                return 0;
            }
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
        println!(
            "{label} u24_membership_bytes={} ({:.3} GiB)",
            membership_bytes,
            membership_bytes as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!(
            "{label} future_fp16_values_bytes={} ({:.3} GiB)",
            fp16_bytes,
            fp16_bytes as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!(
            "{label} future_u24_plus_fp16_bytes={} ({:.3} GiB)",
            membership_bytes + fp16_bytes,
            (membership_bytes + fp16_bytes) as f64 / 1024.0 / 1024.0 / 1024.0
        );
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
}

impl FieldChainWriter {
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
        let field = (pair_key >> 32) as u32;
        let chain = pair_key as u32;
        debug_assert!(chain < (1u32 << 24));

        match self.open_field {
            None => {
                self.open_field = Some(field);
                self.open_offset = self.total_states;
                self.last_chain = None;
            }
            Some(f) if f != field => {
                debug_assert!(field > f);
                self.close_open_field();
                self.open_field = Some(field);
                self.open_offset = self.total_states;
            }
            Some(_) => {}
        }

        if self.last_chain == Some(chain) {
            return Ok(());
        }
        debug_assert!(self.last_chain.map_or(true, |c| chain > c));

        write_u24_le(&mut self.chains_writer, chain)?;
        self.total_states += 1;
        self.open_count += 1;
        self.last_chain = Some(chain);
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.close_open_field();
        self.chains_writer.flush()?;

        let mut w = BufWriter::new(
            File::create(&self.index_path)
                .with_context(|| format!("failed to create {}", self.index_path.display()))?,
        );

        w.write_all(MAGIC)?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(self.metas.len() as u32).to_le_bytes())?;
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
            if f != field_id {
                break;
            }
            i += 1;
        }
        let end = i;
        let mut chain_ids = Vec::<u32>::with_capacity(end - start);
        for &key in &keys[start..end] {
            let (_field, chain) = unpack(key);
            let cid = *state_index
                .key_to_id
                .get(&crate::state::state_key(chain))
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
