//! Bounded, warm raw/compact file measurements on separated real field spans.
//! Source access is limited to headers, selected field metadata and selected
//! f32/u16 payloads. No graph, queue index or full layer index is constructed.

use super::*;
use crate::compact_field::{encode_group, CompactFieldReader, CompactFieldWriter, CompactLimits};
use crate::layer_store::FieldChainBlockMeta;
use crate::loop1_whole_operator::InputGuard;
use std::os::unix::fs::MetadataExt;

const MIB: u64 = 1 << 20;
const MAX_SPAN_STATES: u64 = 1_000_000;
const MAX_FIELDS: usize = 4096;
const MAX_TOTAL_STATES: u64 = 15_000_000;
const MAX_SOURCE_READ: u64 = 256 * MIB;
const MAX_ARTIFACTS: u64 = 512 * MIB;
const SUMMARY_RESERVE: u64 = 2 * MIB;
const ANCHORS: [(u64, u64); 3] = [(1, 8), (1, 2), (7, 8)];

struct Config {
    pruned: PathBuf,
    values: PathBuf,
    policy: PathBuf,
    work: PathBuf,
    states: u64,
    fields: usize,
    cycles: usize,
    minimum_ms: u64,
}

fn argument(args: &[String], at: usize, default: u64, name: &str) -> Result<u64> {
    args.get(at)
        .map(|value| value.parse::<u64>())
        .transpose()
        .with_context(|| format!("invalid {name}"))
        .map(|value| value.unwrap_or(default))
}

impl Config {
    fn parse(args: &[String]) -> Result<Self> {
        anyhow::ensure!((6..=10).contains(&args.len()), "usage: compact-field-probe <pruned> <values> <policy> <work_parent> [max_span_states=1000000] [max_span_fields=512] [ABBA_cycles=2] [minimum_sample_ms=25]");
        let states = argument(args, 6, MAX_SPAN_STATES, "max_span_states")?;
        let fields = argument(args, 7, 512, "max_span_fields")?;
        let cycles = argument(args, 8, 2, "ABBA_cycles")?;
        let minimum_ms = argument(args, 9, 25, "minimum_sample_ms")?;
        anyhow::ensure!(
            (1..=MAX_SPAN_STATES).contains(&states),
            "max_span_states must be 1..=1000000"
        );
        anyhow::ensure!(
            (1..=MAX_FIELDS as u64).contains(&fields),
            "max_span_fields must be 1..=4096"
        );
        anyhow::ensure!((1..=4).contains(&cycles), "ABBA_cycles must be 1..=4");
        anyhow::ensure!(
            (10..=100).contains(&minimum_ms),
            "minimum_sample_ms must be 10..=100"
        );
        Ok(Self {
            pruned: PathBuf::from(&args[2]),
            values: PathBuf::from(&args[3]),
            policy: PathBuf::from(&args[4]),
            work: PathBuf::from(&args[5]),
            states,
            fields: fields as usize,
            cycles: cycles as usize,
            minimum_ms,
        })
    }
}

#[derive(Default)]
struct ReadBudget {
    bytes: u64,
}

impl ReadBudget {
    fn read(&mut self, file: &File, output: &mut [u8], at: u64) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(output.len() as u64)
            .context("source read counter overflow")?;
        anyhow::ensure!(
            self.bytes <= MAX_SOURCE_READ,
            "original source pread budget exceeded"
        );
        read_exact_at_file(file, output, at)
    }
}

struct Header {
    bytes: [u8; 24],
    fields: usize,
    states: u64,
}

fn header(file: &File, values: &File, policy: &File, budget: &mut ReadBudget) -> Result<Header> {
    let mut bytes = [0u8; 24];
    budget.read(file, &mut bytes, 0)?;
    anyhow::ensure!(
        &bytes[..8] == b"FCHNIDX1" && u32::from_le_bytes(bytes[8..12].try_into()?) == 1,
        "unsupported source field index header"
    );
    let fields = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
    let states = u64::from_le_bytes(bytes[16..24].try_into()?);
    anyhow::ensure!(
        fields > 0 && fields <= NUM_FIELDS && states > 0,
        "source header outside fixed field domain"
    );
    anyhow::ensure!(
        file.metadata()?.len() == 24 + fields as u64 * 16,
        "source index EOF differs from header"
    );
    anyhow::ensure!(
        values.metadata()?.len()
            == states
                .checked_mul(4)
                .context("source values length overflow")?
            && policy.metadata()?.len()
                == states
                    .checked_mul(2)
                    .context("source policy length overflow")?,
        "source values/policy lengths differ from header"
    );
    let mut edge = [0u8; 16];
    budget.read(file, &mut edge, 24)?;
    let first = metadata(&edge)?;
    anyhow::ensure!(
        first.chain_offset == 0,
        "source first field offset is nonzero"
    );
    budget.read(file, &mut edge, 24 + (fields as u64 - 1) * 16)?;
    let last = metadata(&edge)?;
    anyhow::ensure!(
        last.chain_offset.checked_add(u64::from(last.count)) == Some(states),
        "source last field does not reach EOF"
    );
    Ok(Header {
        bytes,
        fields,
        states,
    })
}

fn metadata(bytes: &[u8]) -> Result<FieldChainBlockMeta> {
    anyhow::ensure!(bytes.len() == 16, "truncated selected field metadata");
    let meta = FieldChainBlockMeta {
        field_id: u32::from_le_bytes(bytes[..4].try_into()?),
        count: u32::from_le_bytes(bytes[4..8].try_into()?),
        chain_offset: u64::from_le_bytes(bytes[8..16].try_into()?),
    };
    anyhow::ensure!(
        (meta.field_id as usize) < NUM_FIELDS && meta.count as usize <= CHECK_QSTATE_COUNT,
        "selected metadata exceeds field/qstate domain"
    );
    Ok(meta)
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 32 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn create_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(bytes)?;
    output.flush()?;
    output.sync_data()?;
    Ok(())
}

fn fresh_directory(parent: &Path) -> Result<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for attempt in 0..100 {
        let path = parent.join(format!(
            "compact-field-probe-{}-{stamp}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error).context("create owned compact profiler directory"),
        }
    }
    anyhow::bail!("could not create owned compact profiler directory")
}

struct Span {
    anchor: usize,
    ordinal: usize,
    metas: Vec<FieldChainBlockMeta>,
    raw: [Vec<u8>; 2],
    paths: [[PathBuf; 2]; 2],
}

struct LayerCase {
    layer: usize,
    dir: PathBuf,
    spans: Vec<Span>,
    omissions: Vec<String>,
    states: u64,
    source_fields: usize,
    source_states: u64,
}

fn select_layer(
    config: &Config,
    layer: usize,
    root: &Path,
    budget: &mut ReadBudget,
) -> Result<LayerCase> {
    let index =
        File::open(q_pruned_prefix(&config.pruned, layer).with_extension("field_index.bin"))?;
    let values = File::open(mdp_f32_value_path(&config.values, layer))?;
    let policy = File::open(mdp_policy_u16_path(&config.policy, layer))?;
    let h = header(&index, &values, &policy, budget)?;
    let dir = root.join(format!("layer{layer}"));
    fs::create_dir(&dir)?;
    create_file(&dir.join("source-index-header.bin"), &h.bytes)?;
    let mut spans: Vec<Span> = Vec::new();
    let mut omissions = Vec::new();
    let mut states = 0u64;
    for (anchor, (numerator, denominator)) in ANCHORS.into_iter().enumerate() {
        let ordinal = (h.fields as u64 * numerator / denominator) as usize;
        if spans
            .iter()
            .any(|span| ordinal < span.ordinal + span.metas.len())
        {
            omissions.push(format!("anchor{anchor}: overlaps earlier selected span"));
            continue;
        }
        let count = config.fields.min(h.fields - ordinal);
        let mut bytes = vec![0u8; count * 16];
        budget.read(&index, &mut bytes, 24 + ordinal as u64 * 16)?;
        let candidates = bytes
            .chunks_exact(16)
            .map(metadata)
            .collect::<Result<Vec<_>>>()?;
        let mut previous = if ordinal > 0 {
            let mut before = [0u8; 16];
            budget.read(&index, &mut before, 24 + (ordinal as u64 - 1) * 16)?;
            Some(metadata(&before)?)
        } else {
            None
        };
        for meta in &candidates {
            if let Some(before) = &previous {
                anyhow::ensure!(
                    before.field_id < meta.field_id
                        && before.chain_offset.checked_add(u64::from(before.count))
                            == Some(meta.chain_offset),
                    "selected metadata is not ordered/contiguous"
                );
            }
            anyhow::ensure!(
                meta.chain_offset
                    .checked_add(u64::from(meta.count))
                    .is_some_and(|end| end <= h.states),
                "selected field exceeds total source states"
            );
            previous = Some(meta.clone());
        }
        let mut selected = 0;
        let mut selected_states = 0u64;
        for meta in &candidates {
            let next = selected_states + u64::from(meta.count);
            if next > config.states {
                break;
            }
            selected_states = next;
            selected += 1;
        }
        if selected_states == 0 {
            omissions.push(format!(
                "anchor{anchor}: no nonempty whole field fits span cap"
            ));
            continue;
        }
        let metas = candidates[..selected].to_vec();
        let mut raw = [
            vec![0u8; usize::try_from(selected_states * 4)?],
            vec![0u8; usize::try_from(selected_states * 2)?],
        ];
        budget.read(&values, &mut raw[0], metas[0].chain_offset * 4)?;
        budget.read(&policy, &mut raw[1], metas[0].chain_offset * 2)?;
        let label = format!("span{anchor}");
        create_file(
            &dir.join(format!("{label}.source-metadata.bin")),
            &bytes[..selected * 16],
        )?;
        create_file(&dir.join(format!("{label}.original-values.raw")), &raw[0])?;
        create_file(&dir.join(format!("{label}.original-policy.raw")), &raw[1])?;
        let paths = [
            [
                dir.join(format!("{label}.trial-values.raw")),
                dir.join(format!("{label}.trial-policy.raw")),
            ],
            [
                dir.join(format!("{label}.trial-values.compact")),
                dir.join(format!("{label}.trial-policy.compact")),
            ],
        ];
        spans.push(Span {
            anchor,
            ordinal,
            metas,
            raw,
            paths,
        });
        states += selected_states;
    }
    anyhow::ensure!(
        !spans.is_empty(),
        "no bounded nonempty spans selected for layer {layer}"
    );
    Ok(LayerCase {
        layer,
        dir,
        spans,
        omissions,
        states,
        source_fields: h.fields,
        source_states: h.states,
    })
}

fn limits() -> CompactLimits {
    CompactLimits {
        max_field_bytes: CHECK_QSTATE_COUNT * 4,
        // The group includes owned frame buffers, spare capacity and metadata,
        // in addition to its at most 4,000,000 decoded payload bytes.
        max_group_bytes: 16 * MIB as usize,
        max_logical_bytes: MAX_SPAN_STATES * 4,
        max_file_bytes: 8 * MIB,
        max_index_bytes: 256 * 1024,
    }
}

#[derive(Clone, Default)]
struct FileSizes {
    logical: u64,
    payload: u64,
    file: u64,
    index: u64,
    raw_fields: u64,
    zlib_fields: u64,
}

fn write_pass(case: &LayerCase, compact: bool, kind: usize) -> Result<FileSizes> {
    let mut sizes = FileSizes::default();
    for span in &case.spans {
        let path = &span.paths[usize::from(compact)][kind];
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("remove only prior owned trial output"),
        }
        let width = if kind == 0 { 4 } else { 2 };
        if compact {
            let mut writer = CompactFieldWriter::create(path, width, &span.metas, limits())?;
            let group = encode_group(width, &span.metas, &span.raw[kind], limits())?;
            writer.append_group(&group)?;
            let stats = writer.finish()?;
            sizes.logical += stats.logical_bytes;
            sizes.payload += stats.stored_payload_bytes;
            sizes.file += stats.file_bytes;
            sizes.index += stats.index_bytes;
            sizes.raw_fields += stats.raw_fields as u64;
            sizes.zlib_fields += stats.zlib_fields as u64;
        } else {
            create_file(path, &span.raw[kind])?;
            sizes.logical += span.raw[kind].len() as u64;
            sizes.payload += span.raw[kind].len() as u64;
            sizes.file += fs::metadata(path)?.len();
            sizes.raw_fields += span.metas.len() as u64;
        }
    }
    Ok(sizes)
}

enum Reader {
    Raw(File),
    Compact(CompactFieldReader),
}

struct ReadSet {
    readers: Vec<Reader>,
    output: Vec<Vec<u8>>,
    decoded: Vec<u8>,
    encoded: Vec<u8>,
    kind: usize,
}

impl ReadSet {
    fn open(case: &LayerCase, compact: bool, kind: usize) -> Result<Self> {
        let mut readers = Vec::new();
        let mut output = Vec::new();
        for span in &case.spans {
            let path = &span.paths[usize::from(compact)][kind];
            readers.push(if compact {
                Reader::Compact(CompactFieldReader::open(
                    path,
                    if kind == 0 { 4 } else { 2 },
                    &span.metas,
                    limits(),
                )?)
            } else {
                Reader::Raw(File::open(path)?)
            });
            output.push(vec![0u8; span.raw[kind].len()]);
        }
        Ok(Self {
            readers,
            output,
            decoded: Vec::new(),
            encoded: Vec::new(),
            kind,
        })
    }

    fn pass(&mut self, case: &LayerCase, order: &[(usize, usize)]) -> Result<()> {
        for &(si, fi) in order {
            let span = &case.spans[si];
            let meta = &span.metas[fi];
            let width = if self.kind == 0 { 4usize } else { 2 };
            let count = meta.count as usize * width;
            let offset = usize::try_from(meta.chain_offset - span.metas[0].chain_offset)?
                .checked_mul(width)
                .context("local field byte offset overflow")?;
            match &self.readers[si] {
                Reader::Raw(file) => {
                    self.decoded.resize(count, 0);
                    read_exact_at_file(file, &mut self.decoded, offset as u64)?;
                }
                Reader::Compact(reader) => {
                    reader.read_field_bound_into(fi, meta, &mut self.decoded, &mut self.encoded)?
                }
            }
            anyhow::ensure!(self.decoded.len() == count, "restored field length changed");
            self.output[si][offset..offset + count].copy_from_slice(&self.decoded);
        }
        Ok(())
    }

    fn verify(&self, case: &LayerCase) -> Result<()> {
        for (si, span) in case.spans.iter().enumerate() {
            anyhow::ensure!(
                self.output[si] == span.raw[self.kind],
                "restored raw f32/u16 bytes differ"
            );
            if let Reader::Compact(reader) = &self.readers[si] {
                reader.check_inputs_unchanged()?;
            }
        }
        Ok(())
    }
}

fn orders(case: &LayerCase) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let sequential: Vec<_> = case
        .spans
        .iter()
        .enumerate()
        .flat_map(|(si, span)| (0..span.metas.len()).map(move |fi| (si, fi)))
        .collect();
    let mut shuffled = sequential.clone();
    let mut state = 0x9e3779b97f4a7c15u64 ^ case.layer as u64;
    for end in (1..shuffled.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        shuffled.swap(end, state as usize % (end + 1));
    }
    (sequential, shuffled)
}

fn cpu() -> Result<f64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the initialized timespec remains writable during clock_gettime.
    anyhow::ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time) } == 0,
        "cannot read process CPU clock"
    );
    Ok(time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9)
}

fn timed<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, f64, f64)> {
    let initial_cpu = cpu()?;
    let start = Instant::now();
    let value = operation()?;
    let wall = start.elapsed().as_secs_f64();
    let process_cpu = cpu()? - initial_cpu;
    Ok((value, wall, process_cpu))
}

fn median(values: &[f64]) -> Result<f64> {
    anyhow::ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite() && *v > 0.0),
        "invalid timing samples"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

fn phase(
    case: &LayerCase,
    kind: usize,
    name: &str,
    config: &Config,
    guard: &InputGuard,
    mut operation: impl FnMut(bool, usize) -> Result<(f64, f64)>,
    mut verify: impl FnMut(bool) -> Result<()>,
) -> Result<String> {
    let mut warm = [0.0; 2];
    let mut samples = Vec::new();
    let mut wall = [Vec::new(), Vec::new()];
    let mut process = [Vec::new(), Vec::new()];
    for compact in [false, true] {
        guard.check()?;
        let (seconds, cpu_seconds) = operation(compact, 1)?;
        verify(compact)?;
        guard.check()?;
        warm[usize::from(compact)] = seconds;
        println!("compact-field-probe warmup layer={} kind={} phase={} arm={} seconds={seconds:.9} process_cpu_seconds={cpu_seconds:.9} exact_bytes=1", case.layer, if kind==0 {"values"}else{"policy"}, name, if compact {"compact"} else {"raw"});
    }
    let fastest = warm[0].min(warm[1]);
    anyhow::ensure!(
        fastest.is_finite() && fastest > 0.0,
        "invalid warmup duration"
    );
    let inner = (((config.minimum_ms as f64 * 0.001) / fastest).ceil() as usize).clamp(1, 128);
    for cycle in 0..config.cycles {
        for (position, compact) in [false, true, true, false].into_iter().enumerate() {
            guard.check()?;
            let (seconds, cpu_seconds) = operation(compact, inner)?;
            verify(compact)?;
            guard.check()?;
            wall[usize::from(compact)].push(seconds / inner as f64);
            process[usize::from(compact)].push(cpu_seconds / inner as f64);
            let row = format!("{{\"cycle\":{cycle},\"position\":{position},\"arm\":{},\"inner_repetitions\":{inner},\"wall_seconds\":{seconds:.9},\"process_cpu_seconds\":{cpu_seconds:.9},\"logical_bytes\":{},\"logical_MiB_per_second\":{:.6},\"exact_restored_bytes\":true}}",
            json_string(if compact {"compact"} else {"raw"}), case.states*(if kind==0 {4}else{2})*inner as u64, (case.states*(if kind==0 {4}else{2})*inner as u64) as f64 / MIB as f64 / seconds);
            println!(
                "compact-field-probe sample layer={} kind={} phase={} json={row}",
                case.layer,
                if kind == 0 { "values" } else { "policy" },
                name
            );
            samples.push(row);
        }
    }
    let raw = median(&wall[0])?;
    let compact = median(&wall[1])?;
    let raw_cpu = median(&process[0])?;
    let compact_cpu = median(&process[1])?;
    let width = if kind == 0 { 4.0 } else { 2.0 };
    let result = format!("{{\"kind\":{},\"phase\":{},\"inner_repetitions\":{},\"samples_per_arm\":{},\"raw_median_seconds_per_pass\":{raw:.9},\"compact_median_seconds_per_pass\":{compact:.9},\"raw_median_process_cpu_seconds_per_pass\":{raw_cpu:.9},\"compact_median_process_cpu_seconds_per_pass\":{compact_cpu:.9},\"wall_speedup\":{:.9},\"cpu_speedup\":{:.9},\"raw_logical_MiB_per_second\":{:.6},\"compact_logical_MiB_per_second\":{:.6},\"samples\":[{}]}}",
        json_string(if kind==0 {"values"}else{"policy"}),json_string(name),inner,config.cycles*2,raw/compact,raw_cpu/compact_cpu,case.states as f64*width/MIB as f64/raw,case.states as f64*width/MIB as f64/compact,samples.join(","));
    println!(
        "compact-field-probe phase-summary layer={} kind={} phase={} json={result}",
        case.layer,
        if kind == 0 { "values" } else { "policy" },
        name
    );
    Ok(result)
}

fn verify_files(
    case: &LayerCase,
    compact: bool,
    kind: usize,
    order: &[(usize, usize)],
) -> Result<()> {
    let mut readers = ReadSet::open(case, compact, kind)?;
    readers.pass(case, order)?;
    readers.verify(case)
}

fn sizes_json(s: &FileSizes) -> String {
    format!("{{\"logical_bytes\":{},\"stored_payload_bytes\":{},\"file_bytes_including_header_index\":{},\"index_bytes\":{},\"raw_fields\":{},\"zlib_fields\":{}}}",s.logical,s.payload,s.file,s.index,s.raw_fields,s.zlib_fields)
}

fn profile(case: &LayerCase, config: &Config, guard: &InputGuard) -> Result<String> {
    let (sequential, shuffled) = orders(case);
    let mut kinds = Vec::new();
    let mut phases = Vec::new();
    let mut total_raw = 0u64;
    let mut total_compact = 0u64;
    for kind in 0..2 {
        let final_sizes = std::cell::RefCell::new([FileSizes::default(), FileSizes::default()]);
        phases.push(phase(
            case,
            kind,
            "encode_write",
            config,
            guard,
            |compact, inner| {
                let (sizes, wall, cpu) = timed(|| {
                    let mut sizes = FileSizes::default();
                    for _ in 0..inner {
                        sizes = write_pass(case, compact, kind)?;
                    }
                    Ok(sizes)
                })?;
                final_sizes.borrow_mut()[usize::from(compact)] = sizes;
                Ok((wall, cpu))
            },
            |compact| verify_files(case, compact, kind, &sequential),
        )?);
        let (mut raw, raw_open, raw_open_cpu) = timed(|| ReadSet::open(case, false, kind))?;
        let (mut compact, compact_open, compact_open_cpu) =
            timed(|| ReadSet::open(case, true, kind))?;
        // Retained handles and reusable buffers receive the same field order.
        let readers = std::cell::RefCell::new([&mut raw, &mut compact]);
        for (name, order) in [
            ("read_decode_sequential", &sequential),
            ("read_decode_shuffled", &shuffled),
        ] {
            phases.push(phase(
                case,
                kind,
                name,
                config,
                guard,
                |is_compact, inner| {
                    let mut readers = readers.borrow_mut();
                    let (_, wall, cpu) = timed(|| {
                        for _ in 0..inner {
                            readers[usize::from(is_compact)].pass(case, order)?;
                        }
                        Ok(())
                    })?;
                    Ok((wall, cpu))
                },
                |is_compact| readers.borrow()[usize::from(is_compact)].verify(case),
            )?);
        }
        drop(readers);
        let sizes = final_sizes.into_inner();
        total_raw += sizes[0].file;
        total_compact += sizes[1].file;
        anyhow::ensure!(
            sizes[0].logical == case.states * if kind == 0 { 4 } else { 2 }
                && sizes[0].logical == sizes[1].logical,
            "encoded logical byte count differs from selected states"
        );
        kinds.push(format!("{{\"kind\":{},\"element_width\":{},\"raw\":{},\"compact\":{},\"raw_over_compact_file_ratio\":{:.9},\"reader_open_seconds\":{{\"raw\":{raw_open:.9},\"compact\":{compact_open:.9}}},\"reader_open_process_cpu_seconds\":{{\"raw\":{raw_open_cpu:.9},\"compact\":{compact_open_cpu:.9}}}}}",
            json_string(if kind==0 {"values"}else{"policy"}),if kind==0 {4}else{2},sizes_json(&sizes[0]),sizes_json(&sizes[1]),sizes[0].file as f64/sizes[1].file as f64));
    }
    let mut spans = Vec::new();
    for span in &case.spans {
        let mut file_sizes = Vec::new();
        for kind in 0..2 {
            let r = CompactFieldReader::open(
                &span.paths[1][kind],
                if kind == 0 { 4 } else { 2 },
                &span.metas,
                limits(),
            )?;
            file_sizes.push(format!("{{\"width\":{},\"raw_path\":{},\"compact_path\":{},\"raw_bytes\":{},\"compact_bytes_including_header_index\":{},\"reader_index_storage_bytes\":{}}}",
                if kind==0 {4} else {2},json_string(&span.paths[0][kind].to_string_lossy()),json_string(&span.paths[1][kind].to_string_lossy()),span.raw[kind].len(),r.file_bytes(),r.index_storage_bytes()));
        }
        spans.push(format!("{{\"anchor\":{},\"header_ordinal_start\":{},\"header_ordinal_end\":{},\"fields\":{},\"states\":{},\"minimum_field_states\":{},\"maximum_field_states\":{},\"files\":[{}]}}",
            span.anchor,span.ordinal,span.ordinal+span.metas.len(),span.metas.len(),span.raw[0].len()/4,
            span.metas.iter().map(|m|m.count).min().unwrap_or(0),span.metas.iter().map(|m|m.count).max().unwrap_or(0),file_sizes.join(",")));
    }
    Ok(format!("{{\"layer\":{},\"source_fields\":{},\"source_states\":{},\"selected_states\":{},\"selected_fields\":{},\"raw_file_bytes\":{},\"compact_file_bytes_including_header_index\":{},\"raw_over_compact_file_ratio\":{:.9},\"kinds\":[{}],\"spans\":[{}],\"anchor_omissions\":[{}],\"phases\":[{}]}}",
        case.layer,case.source_fields,case.source_states,case.states,sequential.len(),total_raw,total_compact,total_raw as f64/total_compact as f64,kinds.join(","),
        spans.join(","),case.omissions.iter().map(|s|json_string(s)).collect::<Vec<_>>().join(","),phases.join(",")))
}

fn artifact_bytes(root: &Path) -> Result<u64> {
    let mut bytes = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        bytes = bytes
            .checked_add(if meta.is_dir() {
                artifact_bytes(&entry.path())?
            } else {
                meta.len()
            })
            .context("artifact accounting overflow")?;
    }
    Ok(bytes)
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let config = Config::parse(args)?;
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "run compact-field-probe in release mode"
    );
    let setup = Instant::now();
    let parent = fs::canonicalize(&config.work)?;
    anyhow::ensure!(parent.is_dir(), "work_parent must already be a directory");
    for source in [&config.pruned, &config.values, &config.policy] {
        anyhow::ensure!(
            !parent.starts_with(fs::canonicalize(source)?),
            "work_parent must be outside source datasets"
        );
    }
    let paths: Vec<_> = (3..=7)
        .flat_map(|layer| {
            [
                q_pruned_prefix(&config.pruned, layer).with_extension("field_index.bin"),
                mdp_f32_value_path(&config.values, layer),
                mdp_policy_u16_path(&config.policy, layer),
            ]
        })
        .collect();
    let guard = InputGuard::capture(paths.clone())?;
    let root = fresh_directory(&parent)?;
    let mut budget = ReadBudget::default();
    let mut cases = Vec::new();
    let mut states = 0u64;
    for layer in 3..=7 {
        guard.check()?;
        let case = select_layer(&config, layer, &root, &mut budget)?;
        states += case.states;
        anyhow::ensure!(
            states <= MAX_TOTAL_STATES,
            "global selected-state cap exceeded"
        );
        cases.push(case);
        guard.check()?;
    }
    // Original copies + both trial formats including per-field index/header and
    // two summary-link names. The codec uses RAW when compressed bytes grow.
    let fields: u64 = cases
        .iter()
        .flat_map(|c| &c.spans)
        .map(|s| s.metas.len() as u64)
        .sum();
    let bound = states * 18 + fields * (16 + 96) + 15 * 128 + 5 * 24 + SUMMARY_RESERVE + 8 * MIB;
    anyhow::ensure!(
        bound < MAX_ARTIFACTS && budget.bytes <= MAX_SOURCE_READ,
        "global source/artifact preflight exceeded"
    );
    let setup_seconds = setup.elapsed().as_secs_f64();
    println!("compact-field-probe setup selected_states={states} selected_fields={fields} original_source_pread_bytes={} setup_seconds={setup_seconds:.9} setup_excluded_from_timing=1 original_inputs_readonly=1 full_index_loaded=0 graph_loaded=0 qindex_loaded=0 output_dir={}",budget.bytes,root.display());
    let mut reports = Vec::new();
    for case in &cases {
        reports.push(profile(case, &config, &guard)?);
        guard.check()?;
        anyhow::ensure!(
            artifact_bytes(&root)? + SUMMARY_RESERVE < MAX_ARTIFACTS,
            "artifact cap exceeded"
        );
    }
    let sources=paths.iter().map(|p| -> Result<String>{let m=fs::metadata(p)?;Ok(format!("{{\"path\":{},\"canonical\":{},\"device\":{},\"inode\":{},\"bytes\":{},\"mtime_seconds\":{},\"mtime_nanoseconds\":{},\"ctime_seconds\":{},\"ctime_nanoseconds\":{}}}",json_string(&p.to_string_lossy()),json_string(&fs::canonicalize(p)?.to_string_lossy()),m.dev(),m.ino(),m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec()))}).collect::<Result<Vec<_>>>()?;
    let summary=format!("{{\"schema_version\":1,\"status\":\"PASS\",\"scope\":\"separated header-ordinal whole-field samples from L3 through L7\",\"selected_states\":{states},\"selected_fields\":{fields},\"original_source_pread_bytes\":{},\"source_pread_limit_bytes\":{MAX_SOURCE_READ},\"artifact_bound_bytes\":{bound},\"artifact_limit_bytes\":{MAX_ARTIFACTS},\"setup_seconds\":{setup_seconds:.9},\"ABBA_cycles\":{},\"samples_per_arm\":{},\"warm_cache\":true,\"raw_baseline\":\"flat sampled f32/u16 files with already resident selected metadata\",\"comparison\":\"same new binary feature comparison; not old production solver A/B\",\"sourcecopy_and_setup_excluded\":true,\"verification_outside_timing\":true,\"verification_scope\":\"restored byte comparisons and original-input guard checks\",\"restored_byte_comparison_outside_timing\":true,\"compact_decoder_checksum_and_source_checks_included\":true,\"raw_has_per_field_checksum\":false,\"retained_read_handles\":true,\"fixed_shuffle_seed\":\"0x9e3779b97f4a7c15 XOR layer\",\"source_versions_unchanged\":true,\"source_payloads_hashed\":false,\"exact_restored_bytes\":true,\"full_membership_or_policy_validation\":false,\"physical_disk_or_coldcache_measured\":false,\"full_PI_or_Vstar_speedup_measured\":false,\"policy_random_code_hotpath_measured\":false,\"full_PI_supported\":false,\"durable_write_policy\":\"flush and sync_data; compact atomic publication included\",\"read_open_and_index_load_timed_separately\":true,\"kind_timings_separate\":true,\"inner_repetition_cap\":128,\"source_files_written\":false,\"sources\":[{}],\"layers\":[{}]}}\n",budget.bytes,config.cycles,config.cycles*2,sources.join(","),reports.join(","));
    anyhow::ensure!(
        summary.len() < MIB as usize,
        "summary exceeds reserved allowance"
    );
    guard.check()?;
    let temporary = root.join(".summary.json.tmp");
    create_file(&temporary, summary.as_bytes())?;
    guard.check()?;
    fs::hard_link(&temporary, root.join("summary.json"))?;
    fs::remove_file(&temporary)?;
    anyhow::ensure!(
        artifact_bytes(&root)? < MAX_ARTIFACTS,
        "final artifact cap exceeded"
    );
    println!("compact-field-probe complete status=PASS exact_restored_bytes=1 source_versions_unchanged=1 original_source_pread_bytes={} artifacts_below_512MiB=1 summary={}",budget.bytes,root.join("summary.json").display());
    Ok(())
}
