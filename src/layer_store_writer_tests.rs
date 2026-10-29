use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "zxcl-writer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn prefix(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn encoded(qids: &[u32]) -> Vec<u8> {
    qids.iter()
        .flat_map(|qid| {
            assert!(*qid < 1 << 24);
            [*qid as u8, (*qid >> 8) as u8, (*qid >> 16) as u8]
        })
        .collect()
}

#[test]
fn encoded_blocks_match_original_pair_writer_bytes_and_dedup_across_calls() {
    let dir = TestDir::new();
    let reference = dir.prefix("reference");
    let bulk = dir.prefix("bulk");
    let blocks: [(u32, &[u32]); 4] = [
        (2, &[0, 255, 255, 256]),
        (2, &[256, 65_535, 65_536, (1 << 24) - 1]),
        (9, &[0, 0, 1, 2]),
        (10, &[0, (1 << 24) - 1]),
    ];
    let mut old = FieldChainWriter::create(&reference).unwrap();
    let mut new = FieldChainWriter::create(&bulk).unwrap();
    old.push_pair_key(2u64 << 32).unwrap();
    new.push_pair_key(2u64 << 32).unwrap();
    let mut appended = 0;
    for (field, qids) in blocks {
        for &qid in qids {
            old.push_pair_key((field as u64) << 32 | qid as u64)
                .unwrap();
        }
        appended += new.append_sorted_u24_block(field, &encoded(qids)).unwrap();
    }
    assert_eq!(appended, 10);
    assert_eq!(old.finish().unwrap(), 11);
    assert_eq!(new.finish().unwrap(), 11);
    for extension in ["field_index.bin", "chains.u24.bin"] {
        assert_eq!(
            fs::read(reference.with_extension(extension)).unwrap(),
            fs::read(bulk.with_extension(extension)).unwrap()
        );
    }
    let layer = FieldChainLayer::open(&bulk).unwrap();
    assert_eq!(
        layer
            .metas
            .iter()
            .map(|meta| (meta.field_id, meta.count, meta.chain_offset))
            .collect::<Vec<_>>(),
        [(2, 6, 0), (9, 3, 6), (10, 2, 9)]
    );
    assert_eq!(
        layer.read_block_chains(0),
        [0, 255, 256, 65_535, 65_536, (1 << 24) - 1]
    );
}

#[test]
fn invalid_encoded_blocks_and_pair_domain_poison_finish_before_index_publication() {
    for case in 0..6 {
        let dir = TestDir::new();
        let prefix = dir.prefix("invalid");
        let mut writer = FieldChainWriter::create(&prefix).unwrap();
        writer.push_pair_key(4u64 << 32 | 10).unwrap();
        let result = match case {
            0 => writer.append_sorted_u24_block(4, &[0]),
            1 => writer.append_sorted_u24_block(4, &encoded(&[11, 10])),
            2 => writer.append_sorted_u24_block(4, &encoded(&[9, 11])),
            3 => writer.append_sorted_u24_block(3, &encoded(&[0])),
            4 => writer.push_pair_key(4u64 << 32 | (1 << 24)).map(|_| 0),
            _ => writer.push_pair_key(3u64 << 32).map(|_| 0),
        };
        assert!(result.is_err(), "case {case}");
        assert!(writer.append_sorted_u24_block(8, &encoded(&[0])).is_err());
        assert!(writer.finish().is_err());
        assert!(!prefix.with_extension("field_index.bin").exists());
    }
    let mut bytes = Vec::new();
    assert!(write_u24_le(&mut bytes, 1 << 24).is_err());
    assert!(bytes.is_empty());

    let dir = TestDir::new();
    let prefix = dir.prefix("rejected-metadata");
    let mut writer = FieldChainWriter::create(&prefix).unwrap();
    writer
        .append_sorted_u24_block(0, &encoded(&[0, 1]))
        .unwrap();
    writer.invalidate();
    assert!(writer.finish().is_err());
    assert!(!prefix.with_extension("field_index.bin").exists());
}

#[cfg(unix)]
#[test]
fn encoded_file_ranges_stream_with_bounded_scratch_and_cross_chunk_dedup() {
    let dir = TestDir::new();
    let source_path = dir.prefix("source");
    let count = U24_COPY_BUFFER_BYTES / 3 + 9;
    let mut source_bytes = encoded(&[7, 8]);
    source_bytes.extend((0..count).flat_map(|position| {
        let qid = (position / 2) as u32;
        [qid as u8, (qid >> 8) as u8, (qid >> 16) as u8]
    }));
    source_bytes.extend(encoded(&[1, 2]));
    fs::write(&source_path, source_bytes).unwrap();
    let source = File::open(source_path).unwrap();
    let prefix = dir.prefix("streamed");
    let mut writer = FieldChainWriter::create(&prefix).unwrap();
    let mut scratch = vec![0u8; U24_COPY_BUFFER_BYTES * 2];
    let unique = (count + 1) / 2;
    assert_eq!(
        writer
            .append_sorted_u24_file_with_buf(3, &source, 6, count as u64, &mut scratch)
            .unwrap(),
        unique as u64
    );
    assert!(scratch.capacity() <= U24_COPY_BUFFER_BYTES);
    let last = (unique - 1) as u32;
    assert_eq!(
        writer
            .append_sorted_u24_block(3, &encoded(&[last, last + 1]))
            .unwrap(),
        1
    );
    assert_eq!(
        writer
            .append_sorted_u24_file(8, &source, source.metadata().unwrap().len(), 0)
            .unwrap(),
        0
    );
    assert_eq!(writer.finish().unwrap(), unique as u64 + 1);
    let layer = FieldChainLayer::open_index_only(&prefix).unwrap();
    assert_eq!(layer.metas.len(), 1);
    assert_eq!(
        layer.read_block_chains(0),
        (0..=last + 1).collect::<Vec<_>>()
    );
}

#[cfg(unix)]
#[test]
fn malformed_file_ranges_and_late_unsorted_payload_cannot_publish_an_index() {
    for case in 0..5 {
        let dir = TestDir::new();
        let source_path = dir.prefix("source");
        fs::write(&source_path, encoded(&[0, 1, 2])).unwrap();
        let source = File::open(source_path).unwrap();
        let prefix = dir.prefix("bad-range");
        let mut writer = FieldChainWriter::create(&prefix).unwrap();
        let (offset, count) = match case {
            0 => (1, 1),
            1 => (0, 4),
            2 => (u64::MAX, 1),
            3 => (0, u64::MAX),
            _ => (u64::MAX - 2, 1),
        };
        assert!(writer
            .append_sorted_u24_file(0, &source, offset, count)
            .is_err());
        assert!(writer.finish().is_err());
        assert!(!prefix.with_extension("field_index.bin").exists());
    }
    let dir = TestDir::new();
    let source_path = dir.prefix("late-descending");
    let count = U24_COPY_BUFFER_BYTES / 3;
    let mut raw = (0..count as u32)
        .flat_map(|qid| [qid as u8, (qid >> 8) as u8, (qid >> 16) as u8])
        .collect::<Vec<_>>();
    raw.extend([0, 0, 0]);
    fs::write(&source_path, raw).unwrap();
    let source = File::open(source_path).unwrap();
    let prefix = dir.prefix("bad-stream");
    let mut writer = FieldChainWriter::create(&prefix).unwrap();
    assert!(writer
        .append_sorted_u24_file(0, &source, 0, count as u64 + 1)
        .is_err());
    assert!(writer.total_states > 0); // The error occurs after a valid chunk.
    assert!(writer.finish().is_err());
    assert!(!prefix.with_extension("field_index.bin").exists());
}

#[cfg(unix)]
#[test]
fn encoded_file_copy_rejects_writer_output_aliases() {
    let dir = TestDir::new();
    let prefix = dir.prefix("alias");
    let mut writer = FieldChainWriter::create(&prefix).unwrap();
    let source = File::open(prefix.with_extension("chains.u24.bin")).unwrap();
    assert!(writer
        .append_sorted_u24_file(0, &source, 0, 0)
        .unwrap_err()
        .to_string()
        .contains("aliases"));
    assert!(writer.finish().is_err());
}
