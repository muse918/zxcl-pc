//! Bounded F3 -> F4 replay through the production target-owned pipeline.
//!
//! Both arms read the same real predecessor membership and intersect with the
//! same canonical next-layer fields. Only the temporary representation and its
//! production append path differ. No membership mapping or full payload hash is
//! needed; selected field blocks use bounded positional reads.

use super::*;
use crate::layer_store::{FieldChainLayer, FieldChainWriter};
use crate::loop1_whole_operator::InputGuard;
use std::os::unix::fs::MetadataExt;

const MIB: u64 = 1 << 20;
const MAX_TARGET_STATES: u64 = 16_000_000;
// Work bound only; input residency is bounded independently by one qid-domain field.
const MAX_SOURCE_STATES: u64 = 1_000_000_000;
const MAX_ARTIFACT_BYTES: u64 = 512 * MIB;
const COMPARE_BYTES: usize = 1 << 20;
const PRODUCTION_SHARDS: usize = 4096;

struct Config {
    graph: PathBuf,
    pruned: PathBuf,
    work: PathBuf,
    layer: usize,
    first_field: u32,
    fields: usize,
    repetitions: usize,
    workers: usize,
}

fn argument<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    default: &str,
    name: &str,
) -> Result<T> {
    let value = args.get(index).map(String::as_str).unwrap_or(default);
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {name}: {value}"))
}

impl Config {
    fn parse(args: &[String]) -> Result<Self> {
        anyhow::ensure!((5..=10).contains(&args.len()),
            "usage: index-io-probe <graph.bin> <pruned_layer_dir> <work_parent> [source_layer=3] [first_target_field=0] [target_fields=64] [repetitions=2] [workers=4]");
        let config = Self {
            graph: PathBuf::from(&args[2]),
            pruned: PathBuf::from(&args[3]),
            work: PathBuf::from(&args[4]),
            layer: argument(args, 5, "3", "source_layer")?,
            first_field: argument(args, 6, "0", "first_target_field")?,
            fields: argument(args, 7, "64", "target_fields")?,
            repetitions: argument(args, 8, "2", "repetitions")?,
            workers: argument(args, 9, "4", "workers")?,
        };
        anyhow::ensure!(
            (2..=6).contains(&config.layer),
            "source_layer must be a middle layer in 2..=6"
        );
        anyhow::ensure!(
            (config.first_field as usize) < NUM_FIELDS,
            "first_target_field is outside graph domain"
        );
        anyhow::ensure!(
            (1..=256).contains(&config.fields),
            "target_fields must be in 1..=256"
        );
        anyhow::ensure!(
            (2..=8).contains(&config.repetitions),
            "repetitions must be in 2..=8"
        );
        anyhow::ensure!(
            (1..=4).contains(&config.workers),
            "workers must be in 1..=4"
        );
        Ok(config)
    }
}

struct Group {
    shard: usize,
    targets: Vec<usize>,
    sources: Vec<usize>,
}

#[derive(Default, Debug)]
struct Sample {
    producer_seconds: f64,
    assembly_seconds: f64,
    total_seconds: f64,
    source_states: u64,
    target_states: u64,
    raw_candidates: u64,
    membership_hits: u64,
    output_states: u64,
    temporary_bytes: u64,
    final_bytes: u64,
    peak_logical_bytes: u64,
    peak_allocated_bytes: u64,
}

fn selected_target_groups(
    layer: &FieldChainLayer,
    first_field: u32,
    count: usize,
    workers: usize,
) -> Result<Vec<Group>> {
    anyhow::ensure!(
        count > 0 && workers > 0,
        "target field count and workers must be positive"
    );
    let mut cursor = layer
        .metas
        .partition_point(|meta| meta.field_id < first_field);
    anyhow::ensure!(cursor < layer.metas.len(), "target field range is empty");
    let group_count = workers.min(count);
    let mut groups = Vec::new();
    for owner in 0..group_count {
        let quota = count / group_count + usize::from(owner < count % group_count);
        loop {
            anyhow::ensure!(cursor < layer.metas.len(),
                "not enough distinct production target shards with {quota} fields; reduce target_fields or workers");
            let shard = field_range_shard(layer.metas[cursor].field_id, PRODUCTION_SHARDS);
            let end = layer.metas.partition_point(|meta| {
                field_range_shard(meta.field_id, PRODUCTION_SHARDS) <= shard
            });
            if end - cursor >= quota {
                let targets: Vec<_> = (cursor..cursor + quota).collect();
                anyhow::ensure!(
                    targets.iter().all(|&index| layer.metas[index].count > 0),
                    "target field range contains an empty field"
                );
                groups.push(Group {
                    shard,
                    targets,
                    sources: Vec::new(),
                });
                cursor = end;
                break;
            }
            cursor = end;
        }
    }
    Ok(groups)
}

fn states(layer: &FieldChainLayer, indices: &[usize]) -> Result<u64> {
    indices.iter().try_fold(0u64, |total, &index| {
        total
            .checked_add(u64::from(layer.metas[index].count))
            .context("selected state count overflow")
    })
}

fn expected_index(layer: &FieldChainLayer, selected: &[usize]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(24 + 16 * selected.len());
    bytes.extend_from_slice(b"FCHNIDX1");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(selected.len())?.to_le_bytes());
    bytes.extend_from_slice(&states(layer, selected)?.to_le_bytes());
    let mut offset = 0u64;
    for &index in selected {
        let meta = &layer.metas[index];
        bytes.extend_from_slice(&meta.field_id.to_le_bytes());
        bytes.extend_from_slice(&meta.count.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset = offset
            .checked_add(u64::from(meta.count))
            .context("canonical offset overflow")?;
    }
    Ok(bytes)
}

fn validate_metadata(layer: &FieldChainLayer) -> Result<()> {
    for meta in &layer.metas {
        anyhow::ensure!(
            (meta.field_id as usize) < NUM_FIELDS,
            "field outside graph domain: {}",
            meta.field_id
        );
        anyhow::ensure!(
            meta.count as usize <= CHECK_QSTATE_COUNT,
            "membership count exceeds qid domain at field {}",
            meta.field_id
        );
    }
    Ok(())
}

fn validate_selected_qids(layer: &FieldChainLayer, indices: &[usize]) -> Result<()> {
    let mut qids = Vec::new();
    let mut bytes = Vec::new();
    for &index in indices {
        layer.read_block_chains_into_with_buf(index, &mut qids, &mut bytes);
        let mut previous = None;
        for &qid in &qids {
            anyhow::ensure!(
                (qid as usize) < CHECK_QSTATE_COUNT && previous.is_none_or(|old| old < qid),
                "membership qids are not strictly ordered and in-domain at field {}",
                layer.metas[index].field_id
            );
            previous = Some(qid);
        }
    }
    Ok(())
}

fn assign_sources(
    layer: usize,
    graph: &FieldGraph,
    current: &FieldChainLayer,
    next: &FieldChainLayer,
    groups: &mut [Group],
    two_line: FieldId,
) -> Result<()> {
    let map: HashMap<_, _> = groups
        .iter()
        .flat_map(|group| group.targets.iter())
        .map(|&index| (next.metas[index].field_id, index))
        .collect();
    let mut sources = build_forward_target_owned_source_shards(
        layer,
        current,
        graph,
        &map,
        two_line,
        PRODUCTION_SHARDS,
    );
    for group in groups {
        group.sources = std::mem::take(&mut sources[group.shard]);
        if layer == 5 {
            group
                .sources
                .retain(|&index| current.metas[index].field_id != two_line);
        }
        anyhow::ensure!(
            !group.sources.is_empty(),
            "canonical target group has no source predecessor blocks"
        );
    }
    Ok(())
}

fn provenance(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        let meta = fs::metadata(path)?;
        eprintln!("index-io source path={:?} canonical={:?} dev={} inode={} bytes={} mtime_ns={} ctime_ns={}",
            path, fs::canonicalize(path)?, meta.dev(), meta.ino(), meta.len(),
            meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
            meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128);
    }
    Ok(())
}

fn disk_usage(path: &Path) -> Result<(u64, u64)> {
    let mut logical = 0u64;
    let mut allocated = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        let (bytes, blocks) = if meta.is_dir() {
            disk_usage(&entry.path())?
        } else {
            (
                meta.len(),
                meta.blocks()
                    .checked_mul(512)
                    .context("allocated disk bytes overflow")?,
            )
        };
        logical = logical
            .checked_add(bytes)
            .context("artifact byte count overflow")?;
        allocated = allocated
            .checked_add(blocks)
            .context("allocated byte count overflow")?;
    }
    Ok((logical, allocated))
}

fn observe_disk(path: &Path, sample: &mut Sample) -> Result<()> {
    let (logical, allocated) = disk_usage(path)?;
    sample.peak_logical_bytes = sample.peak_logical_bytes.max(logical);
    sample.peak_allocated_bytes = sample.peak_allocated_bytes.max(allocated);
    anyhow::ensure!(
        logical <= MAX_ARTIFACT_BYTES && allocated <= MAX_ARTIFACT_BYTES,
        "index I/O probe artifact budget exceeded: logical={logical} allocated={allocated}"
    );
    Ok(())
}

fn canonical_match(
    prefix: &Path,
    canonical: &FieldChainLayer,
    selected: &[usize],
    expected: &[u8],
) -> Result<()> {
    anyhow::ensure!(
        fs::read(prefix.with_extension("field_index.bin"))? == expected,
        "generated field index differs from canonical selected next layer"
    );
    let output = File::open(prefix.with_extension("chains.u24.bin"))?;
    let source = File::open(&canonical.chains_path)?;
    let expected_bytes = states(canonical, selected)?
        .checked_mul(3)
        .context("canonical byte count overflow")?;
    anyhow::ensure!(
        output.metadata()?.len() == expected_bytes,
        "generated membership length differs from canonical selection"
    );
    let mut a = vec![0u8; COMPARE_BYTES];
    let mut b = vec![0u8; COMPARE_BYTES];
    let mut output_offset = 0u64;
    for &index in selected {
        let meta = &canonical.metas[index];
        let source_offset = meta
            .chain_offset
            .checked_mul(3)
            .context("canonical read offset overflow")?;
        let count = u64::from(meta.count) * 3;
        let mut position = 0;
        while position < count {
            let length = (count - position).min(COMPARE_BYTES as u64) as usize;
            read_exact_at_file(&output, &mut a[..length], output_offset + position)?;
            read_exact_at_file(&source, &mut b[..length], source_offset + position)?;
            anyhow::ensure!(
                a[..length] == b[..length],
                "canonical replay mismatch at target field {} byte {}",
                meta.field_id,
                position
            );
            position += length as u64;
        }
        output_offset += count;
    }
    Ok(())
}

fn byte_match(a: &Path, b: &Path) -> Result<()> {
    for extension in ["field_index.bin", "chains.u24.bin"] {
        let first = File::open(a.with_extension(extension))?;
        let second = File::open(b.with_extension(extension))?;
        let count = first.metadata()?.len();
        anyhow::ensure!(
            count == second.metadata()?.len(),
            "format outputs have unequal {extension} lengths"
        );
        let mut left = vec![0u8; COMPARE_BYTES];
        let mut right = vec![0u8; COMPARE_BYTES];
        let mut at = 0;
        while at < count {
            let length = (count - at).min(COMPARE_BYTES as u64) as usize;
            read_exact_at_file(&first, &mut left[..length], at)?;
            read_exact_at_file(&second, &mut right[..length], at)?;
            anyhow::ensure!(
                left[..length] == right[..length],
                "format outputs differ in {extension} at byte {at}"
            );
            at += length as u64;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    ordinal: usize,
    format: TargetOwnedShardFormat,
    config: &Config,
    owned: &Path,
    graph: &FieldGraph,
    qfwd: &QForwardGraph,
    current: &FieldChainLayer,
    next: &FieldChainLayer,
    groups: &[Group],
    two_line: FieldId,
    pool: &rayon::ThreadPool,
    guard: &InputGuard,
) -> Result<(PathBuf, Sample)> {
    let start = Instant::now();
    guard.check()?;
    let directory = owned.join(format!("arm-{ordinal}"));
    fs::create_dir(&directory)?;
    let prefix = directory.join("selected");
    let qwords = qfwd.offsets.len().div_ceil(64);
    let producer_start = Instant::now();
    let results: Vec<Result<TargetOwnedShardOutput>> = pool.install(|| {
        groups
            .par_iter()
            .enumerate()
            .map(|(index, group)| {
                process_forward_target_owned_range_shard_with_format(
                    config.layer,
                    group.shard,
                    &group.sources,
                    &group.targets,
                    current,
                    graph,
                    qfwd,
                    next,
                    two_line,
                    PRODUCTION_SHARDS,
                    qwords,
                    &directory.join(format!("owner-{index}.tmp")),
                    format,
                )
            })
            .collect()
    });
    let mut sample = Sample {
        producer_seconds: producer_start.elapsed().as_secs_f64(),
        ..Sample::default()
    };
    let mut outputs = Vec::new();
    for result in results {
        let output = result?;
        anyhow::ensure!(
            output.format == format,
            "worker returned the wrong temporary format"
        );
        sample.source_states += output.stats.current_states;
        sample.target_states += output.stats.loaded_backward_states;
        sample.raw_candidates += output.stats.raw_qforward_candidates;
        sample.membership_hits += output.stats.membership_hits;
        sample.output_states += output.output_pairs;
        let bytes = fs::metadata(&output.path)?.len();
        let width = if format == TargetOwnedShardFormat::PairsU64 {
            8
        } else {
            3
        };
        anyhow::ensure!(
            bytes
                == output
                    .output_pairs
                    .checked_mul(width)
                    .context("temporary byte count overflow")?,
            "worker temporary format length mismatch"
        );
        sample.temporary_bytes += bytes;
        outputs.push(output);
    }
    observe_disk(owned, &mut sample)?;
    let assembly_start = Instant::now();
    let mut writer = FieldChainWriter::create(&prefix)?;
    let mut appended = 0u64;
    for output in outputs {
        let count = append_target_owned_output_to_writer(&output, &mut writer)?;
        anyhow::ensure!(
            count == output.output_pairs,
            "production append lost or duplicated states"
        );
        appended += count;
        observe_disk(owned, &mut sample)?;
        fs::remove_file(&output.path)?;
    }
    anyhow::ensure!(
        writer.finish()? == appended && appended == sample.output_states,
        "finished production writer count mismatch"
    );
    sample.assembly_seconds = assembly_start.elapsed().as_secs_f64();
    sample.final_bytes = fs::metadata(prefix.with_extension("field_index.bin"))?.len()
        + fs::metadata(prefix.with_extension("chains.u24.bin"))?.len();
    observe_disk(owned, &mut sample)?;
    guard.check()?;
    sample.total_seconds = start.elapsed().as_secs_f64();
    Ok((prefix, sample))
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let config = Config::parse(args)?;
    let setup = Instant::now();
    let current_prefix = q_pruned_prefix(&config.pruned, config.layer);
    let next_prefix = q_pruned_prefix(&config.pruned, config.layer + 1);
    let paths = vec![
        config.graph.clone(),
        current_prefix.with_extension("field_index.bin"),
        current_prefix.with_extension("chains.u24.bin"),
        next_prefix.with_extension("field_index.bin"),
        next_prefix.with_extension("chains.u24.bin"),
    ];
    let guard = InputGuard::capture(paths.clone())?;
    provenance(&paths)?;
    let current = FieldChainLayer::open_index_only(current_prefix)?;
    let next = FieldChainLayer::open_index_only(next_prefix)?;
    validate_metadata(&current)?;
    validate_metadata(&next)?;
    let mut groups =
        selected_target_groups(&next, config.first_field, config.fields, config.workers)?;
    let selected: Vec<_> = groups
        .iter()
        .flat_map(|group| group.targets.iter().copied())
        .collect();
    let target_states = states(&next, &selected)?;
    anyhow::ensure!(
        target_states > 0 && target_states <= MAX_TARGET_STATES,
        "selected target states must be in 1..={MAX_TARGET_STATES}: {target_states}"
    );
    let artifact_upper = target_states
        .checked_mul(14)
        .and_then(|n| n.checked_add((24 + selected.len() as u64 * 16) * 2 + MIB))
        .context("artifact bound overflow")?;
    anyhow::ensure!(
        artifact_upper <= MAX_ARTIFACT_BYTES,
        "selected replay exceeds artifact bound"
    );
    let expected = expected_index(&next, &selected)?;
    let graph = FieldGraph::load_graph_bin(&config.graph)?;
    let two_line = graph
        .hash_lookup(crate::graph::TWO_LINE_HASH)
        .context("TWO_LINE_HASH missing")?;
    assign_sources(config.layer, &graph, &current, &next, &mut groups, two_line)?;
    let source_states = groups.iter().try_fold(0u64, |total, group| {
        total
            .checked_add(states(&current, &group.sources)?)
            .context("source read count overflow")
    })?;
    anyhow::ensure!(source_states > 0 && source_states <= MAX_SOURCE_STATES,
        "selected predecessor states including group rereads must be in 1..={MAX_SOURCE_STATES}: {source_states}; reduce target_fields");
    let mut source_indices: Vec<_> = groups
        .iter()
        .flat_map(|group| group.sources.iter().copied())
        .collect();
    source_indices.sort_unstable();
    source_indices.dedup();
    validate_selected_qids(&current, &source_indices)?;
    validate_selected_qids(&next, &selected)?;
    guard.check()?;
    let old = enumerate_states();
    let quotient = enumerate_qstates(&old);
    drop(old);
    anyhow::ensure!(
        quotient.states.len() == CHECK_QSTATE_COUNT,
        "unexpected quotient domain size"
    );
    let qfwd = build_qforward_graph(&quotient);
    drop(quotient);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    guard.check()?;
    fs::create_dir_all(&config.work)?;
    let owned = GreedyEmitScratch::create(&config.work)?;
    eprintln!("index-io ready source_layer={} target_layer={} first_field={} last_field={} fields={} states={} unique_source_blocks={} source_states_with_group_rereads={} workers={} groups={} setup_seconds={:.6} artifact_upper_bytes={} artifact_cap_bytes={} production_shards=4096 membership_access=selected_block_pread inputs_warmed_by_validation=1 fsync=0",
        config.layer, config.layer + 1, next.metas[selected[0]].field_id, next.metas[*selected.last().unwrap()].field_id,
        selected.len(), target_states, source_indices.len(), source_states, config.workers, groups.len(), setup.elapsed().as_secs_f64(), artifact_upper, MAX_ARTIFACT_BYTES);
    for group in &groups {
        let fields: Vec<_> = group
            .targets
            .iter()
            .map(|&index| next.metas[index].field_id)
            .collect();
        eprintln!("index-io owner production_shard={} of={} target_fields={fields:?} source_blocks={} source_states={}",
            group.shard, PRODUCTION_SHARDS, group.sources.len(), states(&current, &group.sources)?);
    }
    let mut legacy = Vec::new();
    let mut compact = Vec::new();
    let mut reference: Option<PathBuf> = None;
    let mut ordinal = 0usize;
    let mut counters = None;
    for repetition in 0..config.repetitions {
        for format in [
            TargetOwnedShardFormat::PairsU64,
            TargetOwnedShardFormat::CompactU24,
            TargetOwnedShardFormat::CompactU24,
            TargetOwnedShardFormat::PairsU64,
        ] {
            let (prefix, sample) = run_arm(
                ordinal,
                format,
                &config,
                &owned.path,
                &graph,
                &qfwd,
                &current,
                &next,
                &groups,
                two_line,
                &pool,
                &guard,
            )?;
            anyhow::ensure!(
                sample.output_states == target_states && sample.target_states == target_states,
                "replayed states do not cover selected canonical F{}",
                config.layer + 1
            );
            let identity = (
                sample.source_states,
                sample.target_states,
                sample.raw_candidates,
                sample.membership_hits,
                sample.output_states,
            );
            if let Some(expected) = counters {
                anyhow::ensure!(
                    identity == expected,
                    "producer workloads differ between formats"
                );
            } else {
                counters = Some(identity);
            }
            canonical_match(&prefix, &next, &selected, &expected)?;
            if let Some(reference) = reference.as_ref() {
                byte_match(reference, &prefix)?;
            }
            guard.check()?;
            let source_read_bytes = (sample.source_states + sample.target_states) * 3;
            eprintln!("index-io sample ordinal={ordinal} repetition={repetition} format={format:?} fields={} states={} producer_seconds={:.6} assembly_seconds={:.6} total_seconds={:.6} source_membership_read_bytes={} temporary_write_bytes={} temporary_read_bytes={} final_write_bytes={} boundary_peak_logical_bytes={} boundary_peak_allocated_bytes={} canonical_match=1 reference_bytes_match={} raw_candidates={} membership_hits={}",
                selected.len(), sample.output_states, sample.producer_seconds, sample.assembly_seconds, sample.total_seconds,
                source_read_bytes, sample.temporary_bytes, sample.temporary_bytes, sample.final_bytes,
                sample.peak_logical_bytes, sample.peak_allocated_bytes, usize::from(reference.is_some()), sample.raw_candidates, sample.membership_hits);
            if format == TargetOwnedShardFormat::PairsU64 {
                legacy.push(sample.total_seconds);
            } else {
                compact.push(sample.total_seconds);
            }
            if reference.is_none() {
                reference = Some(prefix);
            } else {
                fs::remove_dir_all(prefix.parent().context("missing arm directory")?)?;
            }
            ordinal += 1;
        }
    }
    guard.check()?;
    eprintln!("index-io verified source_layer={} target_layer={} fields={} states={} samples_each={} median_u64_seconds={:.6} median_u24_seconds={:.6} total_pipeline_speedup={:.6} temporary_payload_byte_ratio=0.375 production_shards=4096 both_formats_byte_identical=1 traffic=logical_not_physical peak=samples_at_pipeline_boundaries input_setup_and_validation_timed_separately=1 output_comparison_excluded=1 scope=selected_real_middle_layer_replay_not_full_index_build cleanup=owned_artifacts",
        config.layer, config.layer + 1, selected.len(), target_states, legacy.len(), median(&legacy), median(&compact), median(&legacy) / median(&compact));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn illegal_probe_configuration_rejects_before_inputs_or_outputs() {
        let base: Vec<String> = [
            "zxcl",
            "index-io-probe",
            "/absent/graph",
            "/absent/layers",
            "/absent/work",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(Config::parse(&base).is_ok());
        for tail in [
            vec!["9"],
            vec!["3", "15185706"],
            vec!["3", "0", "0"],
            vec!["3", "0", "1", "1"],
            vec!["3", "0", "1", "2", "5"],
        ] {
            let mut args = base.clone();
            args.extend(tail.into_iter().map(String::from));
            assert!(Config::parse(&args).is_err());
        }
    }

    #[test]
    fn canonical_comparison_checks_rebased_index_and_payload_not_just_counts() {
        let owned = GreedyEmitScratch::create(&env::temp_dir()).unwrap();
        let canonical_prefix = owned.path.join("canonical");
        let mut writer = FieldChainWriter::create(&canonical_prefix).unwrap();
        for (field, qids) in [(2u32, vec![1u32]), (5, vec![2, 7]), (9, vec![0, 3])] {
            for qid in qids {
                writer.push_pair_key(q_pair_key(field, qid)).unwrap();
            }
        }
        writer.finish().unwrap();
        let canonical = FieldChainLayer::open_index_only(&canonical_prefix).unwrap();
        assert!(selected_target_groups(&canonical, 10, 1, 1).is_err());
        assert!(selected_target_groups(&canonical, 5, 3, 1).is_err());
        assert!(selected_target_groups(&canonical, 5, 2, 2).is_err());
        let selected = selected_target_groups(&canonical, 5, 2, 1)
            .unwrap()
            .remove(0)
            .targets;
        let expected = expected_index(&canonical, &selected).unwrap();
        let output = owned.path.join("output");
        fs::write(output.with_extension("field_index.bin"), &expected).unwrap();
        let payload: Vec<u8> = [2u32, 7, 0, 3]
            .iter()
            .flat_map(|qid| [*qid as u8, 0, 0])
            .collect();
        fs::write(output.with_extension("chains.u24.bin"), &payload).unwrap();
        canonical_match(&output, &canonical, &selected, &expected).unwrap();
        let mut changed = payload;
        changed[3] = 6;
        fs::write(output.with_extension("chains.u24.bin"), changed).unwrap();
        assert!(canonical_match(&output, &canonical, &selected, &expected).is_err());
    }
}
