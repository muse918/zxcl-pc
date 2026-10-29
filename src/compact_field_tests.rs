//! Independent wire fixtures and failure/publication regression coverage.
//! Payloads are bytes: floating-point NaNs, signed zero and policy sentinels
//! must survive without numerical interpretation or canonicalization.

use super::*;
use super::{CompactFieldReader as Reader, CompactFieldWriter as Writer};
use std::fs;
use std::path::Path;

#[link(name = "z")]
extern "C" {
    #[link_name = "compressBound"]
    fn fixture_compress_bound(source_len: libc::c_ulong) -> libc::c_ulong;
    #[link_name = "compress2"]
    fn fixture_compress2(
        dest: *mut u8,
        dest_len: *mut libc::c_ulong,
        source: *const u8,
        source_len: libc::c_ulong,
        level: libc::c_int,
    ) -> libc::c_int;
}

fn limits() -> CompactLimits {
    CompactLimits {
        max_field_bytes: 16 * 1024,
        max_group_bytes: 64 * 1024,
        max_logical_bytes: 64 * 1024,
        max_file_bytes: 1024 * 1024,
        max_index_bytes: 64 * 1024,
    }
}

fn meta(field_id: u32, count: u32, chain_offset: u64) -> crate::layer_store::FieldChainBlockMeta {
    crate::layer_store::FieldChainBlockMeta {
        field_id,
        count,
        chain_offset,
    }
}

fn adler(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

// Independent system-zlib level 9 encoding (the production encoder uses level
// 1). Header/index/checksum assembly below never calls the production writer.
fn fixture_zlib(raw: &[u8]) -> Vec<u8> {
    let mut encoded =
        vec![0; unsafe { fixture_compress_bound(raw.len() as libc::c_ulong) } as usize];
    let mut size = encoded.len() as libc::c_ulong;
    let result = unsafe {
        fixture_compress2(
            encoded.as_mut_ptr(),
            &mut size,
            raw.as_ptr(),
            raw.len() as libc::c_ulong,
            9,
        )
    };
    assert_eq!(result, 0, "independent zlib fixture compression");
    encoded.truncate(size as usize);
    encoded
}

#[derive(Clone)]
struct Frame {
    meta: crate::layer_store::FieldChainBlockMeta,
    decoded: Vec<u8>,
    encoded: Vec<u8>,
    codec: u8,
}

fn raw_frame(meta: crate::layer_store::FieldChainBlockMeta, bytes: &[u8]) -> Frame {
    Frame {
        meta,
        decoded: bytes.to_vec(),
        encoded: bytes.to_vec(),
        codec: 0,
    }
}

fn zlib_frame(meta: crate::layer_store::FieldChainBlockMeta, bytes: &[u8]) -> Frame {
    Frame {
        meta,
        decoded: bytes.to_vec(),
        encoded: fixture_zlib(bytes),
        codec: 1,
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

// Explicit public wire layout: a 64-byte header, ordered 48-byte descriptors,
// then contiguous field payloads. Keeping this separate catches writer/reader
// agreements that would otherwise hide the same wrong offset on both sides.
fn wire(width: u8, frames: &[Frame]) -> Vec<u8> {
    let payload_start = 64 + 48 * frames.len();
    let total_len = payload_start + frames.iter().map(|f| f.encoded.len()).sum::<usize>();
    let mut bytes = vec![0; total_len];
    bytes[..8].copy_from_slice(b"ZXCFV001");
    put_u32(&mut bytes, 8, 1);
    put_u32(&mut bytes, 12, u32::from(width));
    put_u64(&mut bytes, 16, frames.len() as u64);
    put_u64(
        &mut bytes,
        24,
        frames.iter().map(|f| u64::from(f.meta.count)).sum(),
    );
    put_u64(
        &mut bytes,
        32,
        frames.first().map_or(0, |f| f.meta.chain_offset),
    );
    put_u64(&mut bytes, 40, (48 * frames.len()) as u64);
    put_u64(&mut bytes, 48, payload_start as u64);
    put_u64(&mut bytes, 56, total_len as u64);
    let mut payload = payload_start;
    for (i, frame) in frames.iter().enumerate() {
        let index = 64 + i * 48;
        put_u32(&mut bytes, index, frame.meta.field_id);
        put_u32(&mut bytes, index + 4, frame.meta.count);
        put_u64(&mut bytes, index + 8, frame.meta.chain_offset);
        put_u64(&mut bytes, index + 16, payload as u64);
        put_u64(&mut bytes, index + 24, frame.encoded.len() as u64);
        put_u64(
            &mut bytes,
            index + 32,
            u64::from(frame.meta.count) * u64::from(width),
        );
        put_u32(&mut bytes, index + 40, adler(&frame.decoded));
        bytes[index + 44] = frame.codec;
        bytes[payload..payload + frame.encoded.len()].copy_from_slice(&frame.encoded);
        payload += frame.encoded.len();
    }
    bytes
}

fn metas(frames: &[Frame]) -> Vec<crate::layer_store::FieldChainBlockMeta> {
    frames.iter().map(|f| f.meta.clone()).collect()
}

fn word_payload(width: u8, count: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..count {
        if width == 4 {
            let bits = [
                0u32,
                0x8000_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x7fc1_2345,
                0xff80_0000,
                1,
                0xdead_beef,
            ][i % 8];
            bytes.extend_from_slice(&bits.to_le_bytes());
        } else {
            let code = [0xffffu16, 0, 0x0100, 0x0602, 0x8000, 0x00ff, 0x1234, 0x7f01][i % 8];
            bytes.extend_from_slice(&code.to_le_bytes());
        }
    }
    bytes
}

fn frames_fixture(width: u8) -> Vec<Frame> {
    let raw = if width == 4 {
        0x8182_f183u32.to_le_bytes().to_vec()
    } else {
        0xe297u16.to_le_bytes().to_vec()
    };
    vec![
        raw_frame(meta(2, 1, 37), &raw),
        raw_frame(meta(9, 0, 38), &[]),
        zlib_frame(meta(17, 512, 38), &word_payload(width, 512)),
    ]
}

fn assert_only_files(dir: &Path, expected: &[&str]) {
    let mut names: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut expected: Vec<_> = expected.iter().map(|name| name.to_string()).collect();
    expected.sort();
    assert_eq!(
        names, expected,
        "owned pending files must be removed after failure"
    );
}

#[test]
fn independent_raw_zlib_wire_preserves_words_and_reuses_shuffled_read_buffers() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    for width in [2, 4] {
        let frames = frames_fixture(width);
        assert!(frames[2].encoded.len() < frames[2].decoded.len());
        let expected = metas(&frames);
        let path = owned.path.join(format!("manual-{width}"));
        let bytes = wire(width, &frames);
        fs::write(&path, &bytes).unwrap();
        let reader = Reader::open(&path, width, &expected, limits()).unwrap();
        assert_eq!(reader.field_count(), 3);
        assert_eq!(reader.logical_bytes(), 513 * u64::from(width));
        assert_eq!(reader.file_bytes(), bytes.len() as u64);
        assert!((reader.index_storage_bytes() as u64) <= limits().max_index_bytes);
        reader.check_inputs_unchanged().unwrap();
        let clone = reader.clone();
        let mut decoded = vec![0xa5; 7];
        let mut scratch = vec![0x5a; 11];
        for index in [2, 0, 1, 2, 1, 0] {
            clone
                .read_field_into(index, &mut decoded, &mut scratch)
                .unwrap();
            assert_eq!(decoded, frames[index].decoded);
            reader
                .read_field_bound_into(index, &expected[index], &mut decoded, &mut scratch)
                .unwrap();
            assert_eq!(decoded, frames[index].decoded);
        }
        let mut wrong = expected[2].clone();
        wrong.chain_offset += 1;
        assert!(reader
            .read_field_bound_into(2, &wrong, &mut decoded, &mut scratch)
            .is_err());
        assert!(reader
            .read_field_into(3, &mut decoded, &mut scratch)
            .is_err());
        let reopened = Reader::open(&path, width, &expected, limits()).unwrap();
        reopened
            .read_field_into(2, &mut decoded, &mut scratch)
            .unwrap();
        assert_eq!(decoded, frames[2].decoded);
    }
}

#[test]
fn writer_roundtrip_preserves_float_policy_bits_zero_fields_and_empty_layers() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    for width in [2, 4] {
        let frames = frames_fixture(width);
        let expected = metas(&frames);
        let path = owned.path.join(format!("written-{width}"));
        let mut writer = Writer::create(&path, width, &expected, limits()).unwrap();
        for frame in &frames {
            let group = encode_group(
                width,
                std::slice::from_ref(&frame.meta),
                &frame.decoded,
                limits(),
            )
            .unwrap();
            writer.append_group(&group).unwrap();
        }
        writer.finish().unwrap();
        let wire_bytes = fs::read(&path).unwrap();
        assert_eq!(
            wire_bytes[64 + 44],
            0,
            "a single opaque word must use raw fallback"
        );
        assert_eq!(
            wire_bytes[64 + 2 * 48 + 44],
            1,
            "repeating words must exercise actual zlib storage"
        );
        let reader = Reader::open(&path, width, &expected, limits()).unwrap();
        let (mut decoded, mut scratch) = (Vec::new(), Vec::new());
        for index in [2, 1, 0] {
            reader
                .read_field_into(index, &mut decoded, &mut scratch)
                .unwrap();
            assert_eq!(decoded, frames[index].decoded);
        }
        let empty_path = owned.path.join(format!("empty-{width}"));
        Writer::create(&empty_path, width, &[], limits())
            .unwrap()
            .finish()
            .unwrap();
        let empty = Reader::open(&empty_path, width, &[], limits()).unwrap();
        assert_eq!(empty.field_count(), 0);
        assert_eq!(empty.logical_bytes(), 0);
        assert!(empty
            .read_field_into(0, &mut decoded, &mut scratch)
            .is_err());
    }
}

#[test]
fn membership_and_structural_corruption_are_rejected_before_payload_reads() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let path = owned.path.join("structure");
    let frames = frames_fixture(4);
    let expected = metas(&frames);
    let valid = wire(4, &frames);
    fs::write(&path, &valid).unwrap();
    assert!(Reader::open(&path, 2, &expected, limits()).is_err());
    for mutation in 0..3 {
        let mut foreign = expected.clone();
        match mutation {
            0 => foreign[2].field_id += 1,
            1 => foreign[2].count += 1,
            _ => foreign[2].chain_offset += 1,
        }
        assert!(Reader::open(&path, 4, &foreign, limits()).is_err());
    }
    for end in [0, 7, 31, 63, 64, 65, 64 + 48 * 3 - 1, valid.len() - 1] {
        fs::write(&path, &valid[..end]).unwrap();
        assert!(
            Reader::open(&path, 4, &expected, limits()).is_err(),
            "truncated at {end}"
        );
    }
    let mut invalid = Vec::<(&str, Vec<u8>)>::new();
    let mut add = |label, bytes| invalid.push((label, bytes));
    let mut bytes = valid.clone();
    bytes[0] ^= 1;
    add("magic", bytes);
    let mut bytes = valid.clone();
    put_u32(&mut bytes, 8, 2);
    add("version", bytes);
    let mut bytes = valid.clone();
    put_u32(&mut bytes, 12, 1);
    add("unsupported width", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 16, u64::MAX);
    add("field count overflow", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 24, u64::MAX);
    add("element count overflow", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 32, u64::MAX);
    add("logical ordinal overflow", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 40, u64::MAX);
    add("index length overflow", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 48, 63);
    add("payload points into header", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 56, u64::MAX);
    add("EOF overflow", bytes);
    let mut bytes = valid.clone();
    put_u32(&mut bytes, 64 + 48, 2);
    add("duplicate field", bytes);
    let mut bytes = valid.clone();
    put_u32(&mut bytes, 64 + 48 + 4, 1);
    add("count mismatch", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 48 + 8, 37);
    add("repeated logical ordinal", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 2 * 48 + 8, 39);
    add("skipped logical ordinal", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 16, 64);
    add("frame overlaps index", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 2 * 48 + 16, (64 + 3 * 48) as u64);
    add("frame overlaps prior payload", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 24, u64::MAX);
    add("stored length overflow", bytes);
    let mut bytes = valid.clone();
    put_u64(&mut bytes, 64 + 32, u64::MAX);
    add("raw length overflow", bytes);
    let mut bytes = valid.clone();
    bytes[64 + 44] = 2;
    add("unsupported codec", bytes);
    let mut bytes = valid.clone();
    bytes[64 + 45] = 1;
    add("reserved bits", bytes);
    let mut bytes = valid.clone();
    bytes.push(0);
    add("trailing EOF byte", bytes);
    for (label, bytes) in invalid {
        fs::write(&path, bytes).unwrap();
        assert!(
            Reader::open(&path, 4, &expected, limits()).is_err(),
            "accepted {label}"
        );
    }
}

#[test]
fn zlib_and_raw_reads_reject_corruption_trailing_input_and_wrong_decoded_lengths() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let path = owned.path.join("bad-payload");
    let descriptor = meta(7, 1024, 81);
    let expected = vec![descriptor.clone()];
    let baseline = zlib_frame(descriptor.clone(), &vec![0; 4096]);
    let mut cases = Vec::<(&str, Vec<u8>)>::new();
    for (label, len) in [
        ("short decoded output", 4092),
        ("oversized decoded output", 4100),
    ] {
        cases.push((
            label,
            wire(4, &[zlib_frame(descriptor.clone(), &vec![0; len])]),
        ));
    }
    for extra_member in [false, true] {
        let mut frame = baseline.clone();
        frame.encoded.extend_from_slice(if extra_member {
            &baseline.encoded
        } else {
            &[0x5a]
        });
        cases.push((
            if extra_member {
                "second zlib member"
            } else {
                "trailing compressed byte"
            },
            wire(4, &[frame]),
        ));
    }
    let mut frame = baseline.clone();
    *frame.encoded.last_mut().unwrap() ^= 1;
    cases.push(("zlib checksum", wire(4, &[frame])));
    let mut frame = baseline.clone();
    frame.encoded.truncate(frame.encoded.len() - 1);
    cases.push(("truncated zlib", wire(4, &[frame])));
    let mut bytes = wire(4, &[baseline]);
    put_u32(&mut bytes, 64 + 40, 0);
    cases.push(("independent decoded checksum", bytes));
    let raw = raw_frame(meta(7, 1, 81), &[0, 0, 0, 0x80]);
    let mut bytes = wire(4, &[raw]);
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&path, bytes).unwrap();
    let reader = Reader::open(&path, 4, &[meta(7, 1, 81)], limits()).unwrap();
    assert!(reader
        .read_field_into(0, &mut Vec::new(), &mut Vec::new())
        .is_err());
    for (label, bytes) in cases {
        fs::write(&path, bytes).unwrap();
        let reader = Reader::open(&path, 4, &expected, limits()).unwrap();
        let mut decoded = vec![0xa5; 9];
        let mut scratch = Vec::new();
        assert!(
            reader
                .read_field_into(0, &mut decoded, &mut scratch)
                .is_err(),
            "accepted {label}"
        );
    }
}

#[test]
fn configured_caps_reject_large_metadata_and_frames_before_allocation_or_decode() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let path = owned.path.join("limits");
    let frames = frames_fixture(4);
    let expected = metas(&frames);
    let bytes = wire(4, &frames);
    fs::write(&path, &bytes).unwrap();
    let mut cap = limits();
    cap.max_index_bytes = (48 * frames.len()) as u64;
    Reader::open(&path, 4, &expected, cap).unwrap();
    cap.max_index_bytes -= 1;
    assert!(Reader::open(&path, 4, &expected, cap).is_err());
    let mut cap = limits();
    cap.max_file_bytes = bytes.len() as u64;
    Reader::open(&path, 4, &expected, cap).unwrap();
    cap.max_file_bytes -= 1;
    assert!(Reader::open(&path, 4, &expected, cap).is_err());
    let mut cap = limits();
    cap.max_logical_bytes = 513 * 4 - 1;
    assert!(Reader::open(&path, 4, &expected, cap).is_err());
    let mut cap = limits();
    cap.max_field_bytes = 2047;
    assert!(Reader::open(&path, 4, &expected, cap).is_err());
    assert!(encode_group(
        4,
        &expected,
        &frames
            .iter()
            .flat_map(|f| f.decoded.iter().copied())
            .collect::<Vec<_>>(),
        cap
    )
    .is_err());
    let mut cap = limits();
    cap.max_group_bytes = 2051;
    assert!(encode_group(
        4,
        &expected,
        &frames
            .iter()
            .flat_map(|f| f.decoded.iter().copied())
            .collect::<Vec<_>>(),
        cap
    )
    .is_err());

    // A tiny physical file declares a four-GiB decoded frame. A bounded open
    // must reject it from metadata, without trying to allocate/decode that size.
    let huge = zlib_frame(meta(7, 1 << 30, 0), &[0; 4]);
    fs::write(&path, wire(4, &[huge.clone()])).unwrap();
    let mut cap = limits();
    cap.max_logical_bytes = u64::MAX;
    let error = Reader::open(&path, 4, &[huge.meta], cap)
        .err()
        .expect("huge declared decoded frame accepted");
    assert!(
        format!("{error:#}").contains("field"),
        "field cap must reject before decoding: {error:#}"
    );
    let mut bytes = wire(4, &[]);
    put_u64(&mut bytes, 16, 1 << 32);
    let declared_index_bytes = (1u64 << 32) * 48;
    put_u64(&mut bytes, 40, declared_index_bytes);
    put_u64(&mut bytes, 48, 64 + declared_index_bytes);
    put_u64(&mut bytes, 56, 64 + declared_index_bytes);
    fs::write(&path, bytes).unwrap();
    let mut cap = limits();
    cap.max_file_bytes = u64::MAX;
    let error = Reader::open(&path, 4, &[], cap)
        .err()
        .expect("huge declared index accepted");
    let message = format!("{error:#}");
    assert!(
        message.contains("index") && message.contains("cap"),
        "index cap must reject before membership/EOF checks or allocation: {message}"
    );

    let overflow = raw_frame(meta(7, 1, u64::MAX), &[1, 2, 3, 4]);
    let overflow_metas = metas(std::slice::from_ref(&overflow));
    fs::write(&path, wire(4, &[overflow])).unwrap();
    let error = Reader::open(&path, 4, &overflow_metas, limits())
        .err()
        .expect("overflowing logical ordinal accepted");
    assert!(format!("{error:#}").contains("ordinal"));
    let never_published = owned.path.join("ordinal-overflow");
    assert!(Writer::create(&never_published, 4, &overflow_metas, limits()).is_err());
    assert!(!never_published.exists());
}

#[test]
fn failed_append_poisons_writer_and_byte_cap_preserves_accepted_destination() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let poison = owned.path.join("poison");
    fs::create_dir(&poison).unwrap();
    let path = poison.join("output");
    let expected = vec![meta(2, 1, 19), meta(9, 1, 20)];
    let first = encode_group(4, &expected[..1], &[1, 2, 3, 4], limits()).unwrap();
    let second = encode_group(4, &expected[1..], &[5, 6, 7, 8], limits()).unwrap();
    let foreign = encode_group(4, &[meta(10, 1, 20)], &[5, 6, 7, 8], limits()).unwrap();
    let mut writer = Writer::create(&path, 4, &expected, limits()).unwrap();
    writer.append_group(&first).unwrap();
    assert!(writer.append_group(&foreign).is_err());
    assert!(
        writer.append_group(&second).is_err(),
        "error must permanently poison writer"
    );
    assert!(writer.finish().is_err());
    assert!(!path.exists());
    assert_only_files(&poison, &[]);

    let incomplete = owned.path.join("incomplete");
    fs::create_dir(&incomplete).unwrap();
    let path = incomplete.join("output");
    let mut writer = Writer::create(&path, 4, &expected, limits()).unwrap();
    writer.append_group(&first).unwrap();
    assert!(
        writer.finish().is_err(),
        "a prefix of the expected fields is not publishable"
    );
    assert!(!path.exists());
    assert_only_files(&incomplete, &[]);

    let capped = owned.path.join("capped");
    fs::create_dir(&capped).unwrap();
    let path = capped.join("accepted");
    let mut cap = limits();
    cap.max_file_bytes = 64 + 2 * 48 + 8 - 1;
    let mut writer = Writer::create(&path, 4, &expected, cap).unwrap();
    writer.append_group(&first).unwrap();
    fs::write(&path, b"previous accepted output").unwrap();
    assert!(writer.append_group(&second).is_err());
    assert!(writer.finish().is_err());
    assert_eq!(fs::read(&path).unwrap(), b"previous accepted output");
    assert_only_files(&capped, &["accepted"]);
    let exact = owned.path.join("exact");
    fs::create_dir(&exact).unwrap();
    cap.max_file_bytes += 1;
    let path = exact.join("output");
    let mut writer = Writer::create(&path, 4, &expected, cap).unwrap();
    writer.append_group(&first).unwrap();
    writer.append_group(&second).unwrap();
    writer.finish().unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), cap.max_file_bytes);
    assert_only_files(&exact, &["output"]);
}

#[test]
fn publication_never_replaces_existing_files_and_reader_detects_source_changes() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let expected = vec![meta(2, 1, 0)];
    let group = encode_group(4, &expected, &[0, 0, 0, 0x80], limits()).unwrap();
    let path = owned.path.join("accepted");
    let mut writer = Writer::create(&path, 4, &expected, limits()).unwrap();
    writer.append_group(&group).unwrap();
    fs::write(&path, b"accepted by another writer").unwrap();
    assert!(writer.finish().is_err());
    assert_eq!(fs::read(&path).unwrap(), b"accepted by another writer");
    assert_only_files(&owned.path, &["accepted"]);
    match Writer::create(&path, 4, &expected, limits()) {
        Err(_) => {}
        Ok(mut writer) => {
            writer.append_group(&group).unwrap();
            assert!(writer.finish().is_err());
        }
    }
    assert_eq!(fs::read(&path).unwrap(), b"accepted by another writer");
    assert_only_files(&owned.path, &["accepted"]);
    let readable = owned.path.join("readable");
    let mut writer = Writer::create(&readable, 4, &expected, limits()).unwrap();
    writer.append_group(&group).unwrap();
    writer.finish().unwrap();
    let reader = Reader::open(&readable, 4, &expected, limits()).unwrap();
    reader.check_inputs_unchanged().unwrap();
    let mut bytes = fs::read(&readable).unwrap();
    bytes.push(0);
    fs::write(&readable, bytes).unwrap();
    assert!(reader.check_inputs_unchanged().is_err());
}

#[test]
fn independent_and_written_indexes_cross_multiple_io_chunks_without_reordering() {
    let owned = crate::GreedyEmitScratch::create(&std::env::temp_dir()).unwrap();
    let mut cap = limits();
    cap.max_group_bytes = 512 * 1024;
    cap.max_index_bytes = 512 * 1024;
    cap.max_logical_bytes = 1024 * 1024;
    let mut frames = Vec::new();
    let mut ordinal = 23u64;
    for index in 0..2051u32 {
        let count = match index % 3 {
            0 => 0,
            1 => 1,
            _ => 128,
        };
        let descriptor = meta(10 + index * 2, count, ordinal);
        let word = (0x8000_0001u32 ^ index.wrapping_mul(0x0103_070b)).to_le_bytes();
        let raw: Vec<_> = (0..count).flat_map(|_| word).collect();
        frames.push(if count == 128 {
            zlib_frame(descriptor, &raw)
        } else {
            raw_frame(descriptor, &raw)
        });
        ordinal += u64::from(count);
    }
    let expected = metas(&frames);
    let manual_path = owned.path.join("manual-many-fields");
    fs::write(&manual_path, wire(4, &frames)).unwrap();
    let manual = Reader::open(&manual_path, 4, &expected, cap).unwrap();

    let written_path = owned.path.join("written-many-fields");
    let mut writer = Writer::create(&written_path, 4, &expected, cap).unwrap();
    for chunk in frames.chunks(64) {
        let mut group = EncodedGroup::new(4, cap).unwrap();
        for frame in chunk {
            // Workers can transfer ownership of an independently encoded field
            // into a group without recopying its payload.
            group
                .push_encoded_field(encode_field(4, &frame.meta, &frame.decoded, cap).unwrap())
                .unwrap();
        }
        writer.append_group(&group).unwrap();
    }
    writer.finish().unwrap();

    // Inspect the public wire descriptors independently of Reader, including
    // both 1024-entry I/O boundaries and the final partial chunk.
    let bytes = fs::read(&written_path).unwrap();
    let read32 = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let read64 = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    assert_eq!(&bytes[..8], b"ZXCFV001");
    assert_eq!(read64(16), frames.len() as u64);
    assert_eq!(read64(40), frames.len() as u64 * 48);
    assert_eq!(read64(56), bytes.len() as u64);
    let mut payload = 64 + frames.len() as u64 * 48;
    for (index, frame) in frames.iter().enumerate() {
        let entry = 64 + index * 48;
        assert_eq!(read32(entry), frame.meta.field_id);
        assert_eq!(read32(entry + 4), frame.meta.count);
        assert_eq!(read64(entry + 8), frame.meta.chain_offset);
        assert_eq!(read64(entry + 16), payload);
        assert_eq!(read64(entry + 32), frame.decoded.len() as u64);
        assert_eq!(read32(entry + 40), adler(&frame.decoded));
        assert_eq!(bytes[entry + 44], u8::from(frame.meta.count == 128));
        payload += read64(entry + 24);
    }
    assert_eq!(payload, bytes.len() as u64);

    let written = Reader::open(&written_path, 4, &expected, cap).unwrap();
    let clone = written.clone();
    let reopened = Reader::open(&written_path, 4, &expected, cap).unwrap();
    let mut decoded = vec![0xa5; 3];
    let mut scratch = vec![0x5a; 7];
    for index in [
        2050, 1025, 1023, 2048, 0, 1024, 2049, 2, 2047, 1, 1022, 1026, 2046,
    ] {
        for reader in [&manual, &clone, &reopened] {
            reader
                .read_field_bound_into(index, &expected[index], &mut decoded, &mut scratch)
                .unwrap();
            assert_eq!(decoded, frames[index].decoded, "field {index}");
        }
    }
}
