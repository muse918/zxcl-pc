//! Lossless, independently framed field values. Limits are per container;
//! callers must additionally reserve their aggregate artifact and worker RAM budgets.
//! Readers share their immutable file/index through an Arc. Caller-owned read
//! buffers and libz's bounded internal codec workspace are not part of index RAM.

use anyhow::{bail, ensure, Context, Result};
use std::ffi::CString;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{ErrorKind, Write};
use std::mem::size_of;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::layer_store::FieldChainBlockMeta;
#[cfg(target_os = "linux")]
use crate::loop1_stream::InputMutationWatch;

const MAGIC: &[u8; 8] = b"ZXCFV001";
const HEADER_BYTES: u64 = 64;
const ENTRY_BYTES: u64 = 48;
const INDEX_CHUNK_ENTRIES: usize = 1024;
const RAW: u8 = 0;
const ZLIB1: u8 = 1;
static NEXT_PENDING: AtomicU64 = AtomicU64::new(0);

#[link(name = "z")]
extern "C" {
    fn compressBound(source_len: libc::c_ulong) -> libc::c_ulong;
    fn compress2(
        dest: *mut u8,
        dest_len: *mut libc::c_ulong,
        source: *const u8,
        source_len: libc::c_ulong,
        level: libc::c_int,
    ) -> libc::c_int;
    fn uncompress2(
        dest: *mut u8,
        dest_len: *mut libc::c_ulong,
        source: *const u8,
        source_len: *mut libc::c_ulong,
    ) -> libc::c_int;
    fn adler32(adler: libc::c_ulong, buf: *const u8, len: libc::c_uint) -> libc::c_ulong;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompactLimits {
    pub max_field_bytes: usize,
    pub max_group_bytes: usize,
    pub max_logical_bytes: u64,
    /// Actual logical file length cap, including header and index, per file.
    pub max_file_bytes: u64,
    /// Both wire index bytes and actual retained Rust index capacity are capped.
    pub max_index_bytes: u64,
}

impl Default for CompactLimits {
    fn default() -> Self {
        Self {
            max_field_bytes: 817_740 * 4,
            max_group_bytes: 64 * 1024 * 1024,
            max_logical_bytes: 1 << 40,
            max_file_bytes: 32 << 30,
            max_index_bytes: 512 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CompactFileStats {
    pub raw_fields: u64,
    pub zlib_fields: u64,
    pub logical_bytes: u64,
    pub stored_payload_bytes: u64,
    pub file_bytes: u64,
    pub index_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct EncodedField {
    meta: FieldChainBlockMeta,
    width: u8,
    raw_len: u64,
    adler: u32,
    codec: u8,
    payload: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct EncodedGroup {
    width: u8,
    limits: CompactLimits,
    fields: Vec<EncodedField>,
    logical_bytes: u64,
    stored_bytes: u64,
    payload_capacity: usize,
}

fn valid_width(width: u8) -> Result<()> {
    ensure!(
        matches!(width, 2 | 4),
        "compact element width must be 2 or 4"
    );
    Ok(())
}

fn raw_bytes(width: u8, meta: &FieldChainBlockMeta, limits: CompactLimits) -> Result<usize> {
    let len = (meta.count as u64)
        .checked_mul(width as u64)
        .context("field length overflow")?;
    ensure!(
        len <= limits.max_field_bytes as u64,
        "field decoded length exceeds cap"
    );
    usize::try_from(len).context("field length exceeds address space")
}

fn same_meta(a: &FieldChainBlockMeta, b: &FieldChainBlockMeta) -> bool {
    a.field_id == b.field_id && a.count == b.count && a.chain_offset == b.chain_offset
}

fn next_meta(previous: &FieldChainBlockMeta, next: &FieldChainBlockMeta) -> Result<()> {
    ensure!(
        previous.field_id < next.field_id,
        "field ids must be strictly increasing"
    );
    let end = previous
        .chain_offset
        .checked_add(previous.count as u64)
        .context("logical ordinal overflow")?;
    ensure!(
        next.chain_offset == end,
        "field logical ordinals are not contiguous"
    );
    Ok(())
}

fn describe(width: u8, metas: &[FieldChainBlockMeta], limits: CompactLimits) -> Result<(u64, u64)> {
    valid_width(width)?;
    let mut total = 0u64;
    for (i, meta) in metas.iter().enumerate() {
        if i > 0 {
            next_meta(&metas[i - 1], meta)?;
        }
        meta.chain_offset
            .checked_add(meta.count as u64)
            .context("logical ordinal overflow")?;
        total = total
            .checked_add(raw_bytes(width, meta, limits)? as u64)
            .context("logical byte length overflow")?;
        ensure!(
            total <= limits.max_logical_bytes,
            "logical byte length exceeds cap"
        );
    }
    Ok((total, metas.first().map_or(0, |m| m.chain_offset)))
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut value = 1;
    for chunk in bytes.chunks(libc::c_uint::MAX as usize) {
        value = unsafe { adler32(value, chunk.as_ptr(), chunk.len() as libc::c_uint) };
    }
    value as u32
}

fn compression_bound(len: usize) -> Result<usize> {
    let input = libc::c_ulong::try_from(len).context("field length exceeds zlib domain")?;
    let bound = usize::try_from(unsafe { compressBound(input) })
        .context("zlib bound exceeds address space")?;
    ensure!(
        bound >= len && bound <= isize::MAX as usize,
        "invalid zlib compression bound"
    );
    Ok(bound)
}

fn resize_bounded(buffer: &mut Vec<u8>, len: usize, cap: usize) -> Result<()> {
    ensure!(
        len <= cap && len <= isize::MAX as usize,
        "buffer length exceeds cap"
    );
    // A caller may hand us an old larger buffer; release it rather than silently
    // retaining RAM above this reader's configured bound.
    if buffer.capacity() > cap {
        *buffer = Vec::new();
    }
    if buffer.capacity() < len {
        buffer
            .try_reserve_exact(len - buffer.len())
            .context("allocate bounded field buffer")?;
    }
    ensure!(
        buffer.capacity() <= cap,
        "buffer resident capacity exceeds cap"
    );
    buffer.resize(len, 0);
    Ok(())
}

pub(crate) fn encode_field(
    width: u8,
    meta: &FieldChainBlockMeta,
    raw: &[u8],
    limits: CompactLimits,
) -> Result<EncodedField> {
    valid_width(width)?;
    let raw_len = raw_bytes(width, meta, limits)?;
    meta.chain_offset
        .checked_add(meta.count as u64)
        .context("logical ordinal overflow")?;
    ensure!(
        raw_len as u64 <= limits.max_logical_bytes,
        "field logical length exceeds cap"
    );
    ensure!(
        raw.len() == raw_len,
        "field payload length does not match descriptor"
    );
    let mut payload = Vec::new();
    let codec = if raw.is_empty() {
        RAW
    } else {
        let bound = compression_bound(raw.len())?;
        ensure!(
            bound <= limits.max_group_bytes,
            "encoded field workspace exceeds group cap"
        );
        resize_bounded(&mut payload, bound, limits.max_group_bytes)?;
        let mut produced = bound as libc::c_ulong;
        let status = unsafe {
            compress2(
                payload.as_mut_ptr(),
                &mut produced,
                raw.as_ptr(),
                raw.len() as libc::c_ulong,
                1,
            )
        };
        ensure!(
            status == 0,
            "zlib level1 compression failed: status {status}"
        );
        let produced = usize::try_from(produced).context("zlib output length overflow")?;
        ensure!(produced <= bound, "zlib returned output beyond destination");
        if produced < raw.len() {
            payload.truncate(produced);
            ZLIB1
        } else {
            payload[..raw.len()].copy_from_slice(raw);
            payload.truncate(raw.len());
            RAW
        }
    };
    Ok(EncodedField {
        meta: meta.clone(),
        width,
        raw_len: raw_len as u64,
        adler: checksum(raw),
        codec,
        payload,
    })
}

impl EncodedGroup {
    pub(crate) fn new(width: u8, limits: CompactLimits) -> Result<Self> {
        valid_width(width)?;
        Ok(Self {
            width,
            limits,
            fields: Vec::new(),
            logical_bytes: 0,
            stored_bytes: 0,
            payload_capacity: 0,
        })
    }

    pub(crate) fn push_field(&mut self, meta: &FieldChainBlockMeta, raw: &[u8]) -> Result<()> {
        let len = raw_bytes(self.width, meta, self.limits)?;
        ensure!(
            raw.len() == len,
            "field payload length does not match descriptor"
        );
        let bound = if len == 0 { 0 } else { compression_bound(len)? };
        self.reserve_field(meta, len as u64, bound)?;
        let field = encode_field(self.width, meta, raw, self.limits)?;
        self.push_encoded_field(field)
    }

    // Includes conservative old/new Vec overlap when metadata must grow. The
    // usual encode_group path reserves its entire metadata array only once.
    fn reserve_field(
        &mut self,
        meta: &FieldChainBlockMeta,
        raw_len: u64,
        payload: usize,
    ) -> Result<()> {
        if let Some(previous) = self.fields.last() {
            next_meta(&previous.meta, meta)?;
        }
        meta.chain_offset
            .checked_add(meta.count as u64)
            .context("logical ordinal overflow")?;
        let logical = self
            .logical_bytes
            .checked_add(raw_len)
            .context("group logical length overflow")?;
        ensure!(
            logical <= self.limits.max_logical_bytes,
            "group logical bytes exceed cap"
        );
        let metadata = self
            .fields
            .capacity()
            .checked_mul(size_of::<EncodedField>())
            .context("group metadata length overflow")?;
        let next_payload = self
            .payload_capacity
            .checked_add(payload)
            .context("group payload capacity overflow")?;
        if self.fields.len() == self.fields.capacity() {
            let required = self
                .fields
                .len()
                .checked_add(1)
                .context("field count overflow")?;
            let available = self
                .limits
                .max_group_bytes
                .checked_sub(next_payload)
                .and_then(|n| n.checked_sub(metadata))
                .context("group allocation peak exceeds cap")?
                / size_of::<EncodedField>();
            ensure!(
                available >= required,
                "group metadata allocation peak exceeds cap"
            );
            let desired = self
                .fields
                .capacity()
                .saturating_mul(2)
                .max(4)
                .max(required)
                .min(available);
            self.fields
                .try_reserve_exact(desired - self.fields.len())
                .context("allocate bounded group metadata")?;
        }
        let retained = self
            .fields
            .capacity()
            .checked_mul(size_of::<EncodedField>())
            .and_then(|n| n.checked_add(next_payload))
            .context("group capacity overflow")?;
        ensure!(
            retained <= self.limits.max_group_bytes,
            "group resident capacity exceeds cap"
        );
        Ok(())
    }

    /// Accept an independently encoded field without copying or recompression.
    pub(crate) fn push_encoded_field(&mut self, field: EncodedField) -> Result<()> {
        ensure!(field.width == self.width, "encoded field width mismatch");
        ensure!(
            field.raw_len == raw_bytes(self.width, &field.meta, self.limits)? as u64,
            "encoded field length mismatch"
        );
        self.reserve_field(&field.meta, field.raw_len, field.payload.capacity())?;
        let stored = self
            .stored_bytes
            .checked_add(field.payload.len() as u64)
            .context("stored length overflow")?;
        let logical = self
            .logical_bytes
            .checked_add(field.raw_len)
            .context("group logical length overflow")?;
        self.payload_capacity = self
            .payload_capacity
            .checked_add(field.payload.capacity())
            .context("group capacity overflow")?;
        self.fields.push(field);
        self.logical_bytes = logical;
        self.stored_bytes = stored;
        Ok(())
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        self.fields.capacity() as u64 * size_of::<EncodedField>() as u64
            + self.payload_capacity as u64
    }
    pub(crate) fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }
    pub(crate) fn stored_bytes(&self) -> u64 {
        self.stored_bytes
    }
    pub(crate) fn field_count(&self) -> usize {
        self.fields.len()
    }
}

pub(crate) fn encode_group(
    width: u8,
    metas: &[FieldChainBlockMeta],
    raw: &[u8],
    limits: CompactLimits,
) -> Result<EncodedGroup> {
    let (total, _) = describe(width, metas, limits)?;
    ensure!(
        total == raw.len() as u64,
        "group payload length does not match descriptors"
    );
    let mut group = EncodedGroup::new(width, limits)?;
    let metadata_bytes = metas
        .len()
        .checked_mul(size_of::<EncodedField>())
        .context("group metadata size overflow")?;
    ensure!(
        metadata_bytes <= limits.max_group_bytes,
        "group metadata exceeds cap"
    );
    group
        .fields
        .try_reserve_exact(metas.len())
        .context("allocate bounded group metadata")?;
    ensure!(
        group.storage_bytes() <= limits.max_group_bytes as u64,
        "group metadata capacity exceeds cap"
    );
    let mut offset = 0;
    for meta in metas {
        let end = offset + raw_bytes(width, meta, limits)?;
        group.push_field(meta, &raw[offset..end])?;
        offset = end;
    }
    Ok(group)
}

#[derive(Debug)]
struct FrameMeta {
    meta: FieldChainBlockMeta,
    payload_offset: u64,
    stored_len: u64,
    raw_len: u64,
    adler: u32,
    codec: u8,
}

fn index_layout(field_count: u64, limits: CompactLimits) -> Result<(u64, u64, usize)> {
    let wire = field_count
        .checked_mul(ENTRY_BYTES)
        .context("index size overflow")?;
    ensure!(wire <= limits.max_index_bytes, "wire index exceeds cap");
    let count = usize::try_from(field_count).context("index count exceeds address space")?;
    let resident = count
        .checked_mul(size_of::<FrameMeta>())
        .context("resident index overflow")?;
    ensure!(
        resident as u64 <= limits.max_index_bytes && resident <= isize::MAX as usize,
        "resident index exceeds cap"
    );
    let payload_start = HEADER_BYTES
        .checked_add(wire)
        .context("header/index size overflow")?;
    ensure!(
        payload_start <= limits.max_file_bytes,
        "header/index file length exceeds cap"
    );
    Ok((wire, payload_start, count))
}

fn index_vec(count: usize, limits: CompactLimits) -> Result<Vec<FrameMeta>> {
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .context("allocate bounded field index")?;
    ensure!(
        entries.capacity() as u64 * size_of::<FrameMeta>() as u64 <= limits.max_index_bytes,
        "resident index capacity exceeds cap"
    );
    Ok(entries)
}

fn write_at(file: &File, mut bytes: &[u8], mut offset: u64) -> Result<()> {
    while !bytes.is_empty() {
        match file.write_at(bytes, offset) {
            Ok(0) => bail!("short compact file write"),
            Ok(n) => {
                bytes = &bytes[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e).context("write compact field file"),
        }
    }
    Ok(())
}

fn read_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> Result<()> {
    while !bytes.is_empty() {
        match file.read_at(bytes, offset) {
            Ok(0) => bail!("truncated compact field file"),
            Ok(n) => {
                bytes = &mut bytes[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e).context("read compact field file"),
        }
    }
    Ok(())
}

fn encode_entry(frame: &FrameMeta) -> [u8; 48] {
    let mut b = [0; 48];
    b[0..4].copy_from_slice(&frame.meta.field_id.to_le_bytes());
    b[4..8].copy_from_slice(&frame.meta.count.to_le_bytes());
    b[8..16].copy_from_slice(&frame.meta.chain_offset.to_le_bytes());
    b[16..24].copy_from_slice(&frame.payload_offset.to_le_bytes());
    b[24..32].copy_from_slice(&frame.stored_len.to_le_bytes());
    b[32..40].copy_from_slice(&frame.raw_len.to_le_bytes());
    b[40..44].copy_from_slice(&frame.adler.to_le_bytes());
    b[44] = frame.codec;
    b
}

fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}

/// Fresh-only output. Any failed append permanently prevents publication; Drop
/// removes only this writer's own pending file. finish never replaces a target.
pub(crate) struct CompactFieldWriter<'a> {
    path: PathBuf,
    pending: Option<PathBuf>,
    file: File,
    width: u8,
    metas: &'a [FieldChainBlockMeta],
    limits: CompactLimits,
    frames: Vec<FrameMeta>,
    payload_start: u64,
    end: u64,
    logical_start: u64,
    stats: CompactFileStats,
    failed: bool,
}

impl<'a> CompactFieldWriter<'a> {
    pub(crate) fn create(
        path: &Path,
        width: u8,
        metas: &'a [FieldChainBlockMeta],
        limits: CompactLimits,
    ) -> Result<Self> {
        let (logical_bytes, logical_start) = describe(width, metas, limits)?;
        let (index_bytes, payload_start, count) = index_layout(metas.len() as u64, limits)?;
        let frames = index_vec(count, limits)?;
        match fs::symlink_metadata(path) {
            Ok(_) => bail!("compact output already exists: {}", path.display()),
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(e).context("inspect compact output"),
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = path.file_name().context("compact output has no filename")?;
        let mut created = None;
        for _ in 0..64 {
            let mut pending_name = name.to_os_string();
            pending_name.push(format!(
                ".pending.{}.{}",
                std::process::id(),
                NEXT_PENDING.fetch_add(1, Ordering::Relaxed)
            ));
            let pending = parent.join(pending_name);
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&pending)
            {
                Ok(file) => {
                    created = Some((pending, file));
                    break;
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e).context("create fresh compact pending output"),
            }
        }
        let (pending, file) = created.context("could not create unique compact pending output")?;
        let writer = Self {
            path: path.to_path_buf(),
            pending: Some(pending),
            file,
            width,
            metas,
            limits,
            frames,
            payload_start,
            end: payload_start,
            logical_start,
            stats: CompactFileStats {
                logical_bytes,
                index_bytes,
                ..Default::default()
            },
            failed: false,
        };
        // Sparse reservation still counts its complete logical file length.
        writer
            .file
            .set_len(payload_start)
            .context("reserve bounded compact header/index")?;
        Ok(writer)
    }

    pub(crate) fn append_group(&mut self, group: &EncodedGroup) -> Result<()> {
        ensure!(
            !self.failed,
            "compact writer is poisoned by an earlier error"
        );
        let result = self.append_group_inner(group);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn append_group_inner(&mut self, group: &EncodedGroup) -> Result<()> {
        ensure!(group.width == self.width, "compact group width mismatch");
        ensure!(
            group.storage_bytes() <= self.limits.max_group_bytes as u64,
            "encoded group resident capacity exceeds writer cap"
        );
        let next = self
            .frames
            .len()
            .checked_add(group.fields.len())
            .context("field count overflow")?;
        ensure!(next <= self.metas.len(), "too many compact fields");
        let mut end = self.end;
        for (i, field) in group.fields.iter().enumerate() {
            ensure!(
                same_meta(&field.meta, &self.metas[self.frames.len() + i]),
                "compact field does not match expected membership descriptor"
            );
            ensure!(
                field.width == self.width
                    && field.raw_len == raw_bytes(self.width, &field.meta, self.limits)? as u64,
                "encoded field width/length mismatch"
            );
            ensure!(
                (field.codec == RAW && field.payload.len() as u64 == field.raw_len)
                    || (field.codec == ZLIB1 && (field.payload.len() as u64) < field.raw_len),
                "invalid encoded compact field"
            );
            end = end
                .checked_add(field.payload.len() as u64)
                .context("compact file size overflow")?;
            ensure!(
                end <= self.limits.max_file_bytes,
                "compact output file length exceeds cap"
            );
        }
        for field in &group.fields {
            write_at(&self.file, &field.payload, self.end)?;
            self.frames.push(FrameMeta {
                meta: field.meta.clone(),
                payload_offset: self.end,
                stored_len: field.payload.len() as u64,
                raw_len: field.raw_len,
                adler: field.adler,
                codec: field.codec,
            });
            self.end += field.payload.len() as u64;
            if field.codec == RAW {
                self.stats.raw_fields += 1;
            } else {
                self.stats.zlib_fields += 1;
            }
        }
        self.stats.stored_payload_bytes = self.end - self.payload_start;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<CompactFileStats> {
        ensure!(
            !self.failed,
            "compact writer is poisoned by an earlier error"
        );
        ensure!(
            self.frames.len() == self.metas.len(),
            "compact output is missing expected fields"
        );
        let mut index_buffer = [0u8; INDEX_CHUNK_ENTRIES * ENTRY_BYTES as usize];
        for (chunk_index, chunk) in self.frames.chunks(INDEX_CHUNK_ENTRIES).enumerate() {
            for (i, frame) in chunk.iter().enumerate() {
                let start = i * ENTRY_BYTES as usize;
                index_buffer[start..start + ENTRY_BYTES as usize]
                    .copy_from_slice(&encode_entry(frame));
            }
            write_at(
                &self.file,
                &index_buffer[..chunk.len() * ENTRY_BYTES as usize],
                HEADER_BYTES + chunk_index as u64 * INDEX_CHUNK_ENTRIES as u64 * ENTRY_BYTES,
            )?;
        }
        let mut header = [0; 64];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..16].copy_from_slice(&(self.width as u32).to_le_bytes());
        header[16..24].copy_from_slice(&(self.metas.len() as u64).to_le_bytes());
        header[24..32]
            .copy_from_slice(&(self.stats.logical_bytes / self.width as u64).to_le_bytes());
        header[32..40].copy_from_slice(&self.logical_start.to_le_bytes());
        header[40..48].copy_from_slice(&self.stats.index_bytes.to_le_bytes());
        header[48..56].copy_from_slice(&self.payload_start.to_le_bytes());
        header[56..64].copy_from_slice(&self.end.to_le_bytes());
        write_at(&self.file, &header, 0)?;
        self.file.flush().context("flush compact field output")?;
        self.file.sync_data().context("sync compact field output")?;
        ensure!(
            self.file.metadata()?.len() == self.end,
            "compact output length mismatch"
        );
        publish_no_replace(self.pending.as_ref().unwrap(), &self.path)?;
        self.pending = None;
        self.stats.file_bytes = self.end;
        Ok(self.stats)
    }
}

impl Drop for CompactFieldWriter<'_> {
    fn drop(&mut self) {
        if let Some(path) = self.pending.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn publish_no_replace(pending: &Path, path: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        let source =
            CString::new(pending.as_os_str().as_bytes()).context("pending path contains NUL")?;
        let target =
            CString::new(path.as_os_str().as_bytes()).context("output path contains NUL")?;
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        ensure!(
            result == 0,
            "publish compact output without replacement: {}",
            std::io::Error::last_os_error()
        );
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pending, path);
        bail!("atomic no-replace compact publication requires Linux renameat2")
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
impl FileStamp {
    fn of(m: &Metadata) -> Self {
        Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
        }
    }
}

struct ReaderInner {
    request: PathBuf,
    canonical: PathBuf,
    stamp: FileStamp,
    request_stamp: FileStamp,
    file: File,
    #[cfg(target_os = "linux")]
    watch: InputMutationWatch,
    width: u8,
    frames: Vec<FrameMeta>,
    limits: CompactLimits,
    logical_bytes: u64,
    file_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct CompactFieldReader {
    inner: Arc<ReaderInner>,
}

impl CompactFieldReader {
    pub(crate) fn open(
        path: &Path,
        width: u8,
        expected_metas: &[FieldChainBlockMeta],
        limits: CompactLimits,
    ) -> Result<Self> {
        valid_width(width)?;
        let request = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let canonical = fs::canonicalize(&request).context("resolve compact input")?;
        // Install the watch before opening/capturing the immutable input version.
        #[cfg(target_os = "linux")]
        let watch = InputMutationWatch::open(&canonical)?;
        let file = File::open(&canonical).context("open compact field input")?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "compact input is not a regular file");
        ensure!(
            metadata.len() <= limits.max_file_bytes,
            "compact input file length exceeds cap"
        );
        let stamp = FileStamp::of(&metadata);
        let request_stamp = FileStamp::of(&fs::symlink_metadata(&request)?);
        let mut header = [0; 64];
        read_at(&file, &mut header, 0)?;
        ensure!(&header[..8] == MAGIC, "invalid compact magic");
        ensure!(u32_at(&header, 8) == 1, "unsupported compact version");
        ensure!(
            u32_at(&header, 12) == width as u32,
            "compact element width mismatch"
        );
        let field_count = u64_at(&header, 16);
        let (index_bytes, payload_start, count) = index_layout(field_count, limits)?;
        ensure!(
            u64_at(&header, 40) <= limits.max_index_bytes,
            "declared index bytes exceed cap"
        );
        ensure!(
            u64_at(&header, 40) == index_bytes,
            "compact index byte length mismatch"
        );
        ensure!(
            u64_at(&header, 48) == payload_start,
            "compact payload start mismatch"
        );
        let file_bytes = u64_at(&header, 56);
        ensure!(
            file_bytes <= limits.max_file_bytes,
            "compact input file length exceeds cap"
        );
        ensure!(
            file_bytes == metadata.len() && file_bytes >= payload_start,
            "compact file length/EOF mismatch"
        );
        let logical_bytes = u64_at(&header, 24)
            .checked_mul(width as u64)
            .context("logical byte length overflow")?;
        ensure!(
            logical_bytes <= limits.max_logical_bytes,
            "compact logical bytes exceed cap"
        );
        ensure!(
            count == expected_metas.len(),
            "compact field count does not match expected membership"
        );
        let (expected_bytes, logical_start) = describe(width, expected_metas, limits)?;
        ensure!(
            logical_bytes == expected_bytes && u64_at(&header, 32) == logical_start,
            "compact logical domain does not match expected membership"
        );
        let mut frames = index_vec(count, limits)?;
        let mut end = payload_start;
        let mut index_buffer = [0u8; INDEX_CHUNK_ENTRIES * ENTRY_BYTES as usize];
        for (chunk_index, chunk) in expected_metas.chunks(INDEX_CHUNK_ENTRIES).enumerate() {
            read_at(
                &file,
                &mut index_buffer[..chunk.len() * ENTRY_BYTES as usize],
                HEADER_BYTES + chunk_index as u64 * INDEX_CHUNK_ENTRIES as u64 * ENTRY_BYTES,
            )?;
            for (i, expected) in chunk.iter().enumerate() {
                let b = &index_buffer[i * ENTRY_BYTES as usize..(i + 1) * ENTRY_BYTES as usize];
                let frame = FrameMeta {
                    meta: FieldChainBlockMeta {
                        field_id: u32_at(&b, 0),
                        count: u32_at(&b, 4),
                        chain_offset: u64_at(&b, 8),
                    },
                    payload_offset: u64_at(&b, 16),
                    stored_len: u64_at(&b, 24),
                    raw_len: u64_at(&b, 32),
                    adler: u32_at(&b, 40),
                    codec: b[44],
                };
                ensure!(
                    frame.raw_len <= limits.max_field_bytes as u64,
                    "field decoded length exceeds cap"
                );
                ensure!(
                    same_meta(&frame.meta, expected),
                    "compact field does not match expected membership descriptor"
                );
                ensure!(
                    frame.raw_len == raw_bytes(width, expected, limits)? as u64,
                    "compact field raw length mismatch"
                );
                ensure!(
                    b[45..48] == [0, 0, 0],
                    "nonzero compact index reserved bytes"
                );
                ensure!(
                    (frame.codec == RAW && frame.stored_len == frame.raw_len)
                        || (frame.codec == ZLIB1
                            && frame.stored_len > 0
                            && frame.stored_len < frame.raw_len),
                    "invalid compact codec or encoded length"
                );
                ensure!(
                    frame.payload_offset == end,
                    "compact payload frames are not contiguous"
                );
                end = end
                    .checked_add(frame.stored_len)
                    .context("compact payload length overflow")?;
                ensure!(end <= file_bytes, "compact payload extends beyond EOF");
                frames.push(frame);
            }
        }
        ensure!(end == file_bytes, "compact file has trailing bytes");
        let reader = Self {
            inner: Arc::new(ReaderInner {
                request,
                canonical,
                stamp,
                request_stamp,
                file,
                #[cfg(target_os = "linux")]
                watch,
                width,
                frames,
                limits,
                logical_bytes,
                file_bytes,
            }),
        };
        reader.check_inputs_unchanged()?;
        Ok(reader)
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        let inner = &self.inner;
        #[cfg(target_os = "linux")]
        inner.watch.check(&inner.request)?;
        ensure!(
            fs::canonicalize(&inner.request)? == inner.canonical
                && FileStamp::of(&fs::symlink_metadata(&inner.request)?) == inner.request_stamp
                && FileStamp::of(&fs::metadata(&inner.canonical)?) == inner.stamp
                && FileStamp::of(&inner.file.metadata()?) == inner.stamp,
            "compact input changed: {}",
            inner.request.display()
        );
        Ok(())
    }

    pub(crate) fn read_field_into(
        &self,
        index: usize,
        decoded: &mut Vec<u8>,
        encoded_scratch: &mut Vec<u8>,
    ) -> Result<()> {
        let result = self.read_field_inner(index, None, decoded, encoded_scratch);
        if result.is_err() {
            decoded.clear();
        }
        result
    }

    pub(crate) fn read_field_bound_into(
        &self,
        index: usize,
        expected: &FieldChainBlockMeta,
        decoded: &mut Vec<u8>,
        encoded_scratch: &mut Vec<u8>,
    ) -> Result<()> {
        let result = self.read_field_inner(index, Some(expected), decoded, encoded_scratch);
        if result.is_err() {
            decoded.clear();
        }
        result
    }

    fn read_field_inner(
        &self,
        index: usize,
        expected: Option<&FieldChainBlockMeta>,
        decoded: &mut Vec<u8>,
        encoded_scratch: &mut Vec<u8>,
    ) -> Result<()> {
        self.check_inputs_unchanged()?;
        let inner = &self.inner;
        let frame = inner
            .frames
            .get(index)
            .context("compact field index out of range")?;
        if let Some(expected) = expected {
            ensure!(
                same_meta(&frame.meta, expected),
                "compact lookup membership mismatch"
            );
        }
        let raw_len = usize::try_from(frame.raw_len).context("decoded field length overflow")?;
        resize_bounded(decoded, raw_len, inner.limits.max_field_bytes)?;
        if frame.codec == RAW {
            if encoded_scratch.capacity() > inner.limits.max_field_bytes {
                *encoded_scratch = Vec::new();
            }
            encoded_scratch.clear();
            read_at(&inner.file, decoded, frame.payload_offset)?;
        } else {
            let stored_len =
                usize::try_from(frame.stored_len).context("encoded field length overflow")?;
            resize_bounded(encoded_scratch, stored_len, inner.limits.max_field_bytes)?;
            read_at(&inner.file, encoded_scratch, frame.payload_offset)?;
            let mut produced =
                libc::c_ulong::try_from(raw_len).context("decoded length exceeds zlib domain")?;
            let mut consumed = libc::c_ulong::try_from(stored_len)
                .context("encoded length exceeds zlib domain")?;
            let status = unsafe {
                uncompress2(
                    decoded.as_mut_ptr(),
                    &mut produced,
                    encoded_scratch.as_ptr(),
                    &mut consumed,
                )
            };
            ensure!(status == 0, "zlib decompression failed: status {status}");
            ensure!(
                produced as u64 == frame.raw_len,
                "zlib decoded length mismatch"
            );
            ensure!(
                consumed as u64 == frame.stored_len,
                "zlib encoded stream has trailing bytes"
            );
        }
        ensure!(
            checksum(decoded) == frame.adler,
            "compact field Adler32 checksum mismatch"
        );
        self.check_inputs_unchanged()?;
        Ok(())
    }

    pub(crate) fn index_storage_bytes(&self) -> u64 {
        self.inner.frames.capacity() as u64 * size_of::<FrameMeta>() as u64
    }
    pub(crate) fn field_count(&self) -> usize {
        self.inner.frames.len()
    }
    pub(crate) fn logical_bytes(&self) -> u64 {
        self.inner.logical_bytes
    }
    pub(crate) fn file_bytes(&self) -> u64 {
        self.inner.file_bytes
    }
    pub(crate) fn element_width(&self) -> u8 {
        self.inner.width
    }
    pub(crate) fn meta(&self, index: usize) -> Option<&FieldChainBlockMeta> {
        self.inner.frames.get(index).map(|f| &f.meta)
    }
}

#[cfg(test)]
#[path = "compact_field_tests.rs"]
mod tests;
