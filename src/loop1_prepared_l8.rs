//! Policy-scoped full L8 compilation and bounded numeric materialization.
//!
//! The compiler retains IDs rather than numeric values across GMRES applies.
//! Each application evaluates shared ordered averages and gathers the ordinary
//! L8 vector into the same RAM, mmap, or file sinks consumed by the L7 kernel.

use anyhow::{ensure, Context, Result};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::Instant;

use crate::graph::FieldGraph;
use crate::layer_store::FieldChainLayer;
use crate::loop1_stream::{CompiledL8, CompiledL9, Loop1Limits};
use crate::pre_reveal::PreRevealIndex;
use crate::qgraph::QForwardGraph;
use crate::qstate::QStateIndex;
use crate::sweep_schedule::SweepTuning;
use crate::{FixedPolicySweep, PinnedLayer};

const MIB: u64 = 1024 * 1024;
const GATHER_CHUNK_STATES: usize = 2_000_000;

pub(crate) fn enabled_from_env() -> bool {
    std::env::var("PI_LOOP1_COMPILED_L8")
        .ok()
        .is_some_and(|value| {
            value != "0"
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("no")
        })
}

fn setting(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

fn mib_setting(name: &str, default: u64) -> Result<u64> {
    setting(name, default)?
        .checked_mul(MIB)
        .with_context(|| format!("{name} byte overflow"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build<'a>(
    graph: &'a FieldGraph,
    q_index: &'a QStateIndex,
    qfwd: &'a QForwardGraph,
    pre: &'a PreRevealIndex,
    layer: &FieldChainLayer,
    policy_path: &Path,
    l9: &'a CompiledL9<'a>,
) -> Result<CompiledL8<'a>> {
    let started = Instant::now();
    let limits = Loop1Limits {
        max_mapping_bytes: mib_setting("PI_LOOP1_L8_MAX_MAPPING_MIB", 4096)?,
        max_unique_averages: usize::try_from(setting(
            "PI_LOOP1_L8_MAX_UNIQUE_AVERAGES",
            32_000_000,
        )?)?,
        max_cache_bytes: usize::try_from(mib_setting("PI_LOOP1_L8_BUILD_CACHE_MIB", 32)?)?,
    };
    let requested_heap_limit = mib_setting("PI_LOOP1_L8_BUILD_MAX_MIB", 16_384)?;
    // The compiler cap includes its retained L9 input but not the pre-reveal
    // index and terminal plan. Reserve those conservatively, then clamp the
    // compiler's requested maximum to the caller's existing sweep allowance.
    let non_compiler_bytes = (q_index.states.len() as u64)
        .checked_add(pre.len() as u64)
        .and_then(|states| states.checked_mul(48))
        .context("compiled L8 common preparation memory overflow")?;
    let tuning = SweepTuning::from_env(0)?;
    ensure!(
        tuning.can_hold_layers(0, non_compiler_bytes, q_index.states.len()),
        "compiled L8 preparation exceeds FULL_MDP memory planning limit"
    );
    let (mut lower, mut upper) = (0u64, requested_heap_limit);
    while lower < upper {
        let width = upper - lower;
        let candidate = lower + width / 2 + width % 2;
        if tuning.can_hold_layers(candidate, non_compiler_bytes, q_index.states.len()) {
            lower = candidate;
        } else {
            upper = candidate - 1;
        }
    }
    let heap_limit = lower;
    let compiled = CompiledL8::build_with_heap_limit(
        graph,
        q_index,
        qfwd,
        pre,
        layer,
        policy_path,
        0..layer.metas.len(),
        l9,
        limits,
        heap_limit,
    )?;
    ensure!(
        compiled.state_offset() == 0 && compiled.state_count() as u64 == layer.total_states,
        "compiled L8 does not cover the complete layer"
    );
    eprintln!(
        "compiled-l8 states={} unique_averages={} mapping_bytes={} retained_capacity_bytes={} compiler_peak_capacity_bytes={} build_limit_bytes={} requested_build_limit_bytes={} build_elapsed={:.3}s",
        compiled.state_count(), compiled.averages().len(), compiled.stats().mapping_bytes,
        compiled.storage_bytes(), compiled.stats().compiler_peak_capacity_bytes,
        heap_limit, requested_heap_limit, started.elapsed().as_secs_f64()
    );
    Ok(compiled)
}

/// Materialize a complete ordinary L8 vector without an unbounded temporary
/// gather. All capacity checks precede creation or truncation of its sink.
pub(crate) fn materialize(
    compiled: &CompiledL8<'_>,
    terminals: &[f32],
    sweep: &FixedPolicySweep<'_>,
    tuning: &SweepTuning,
    prepared_resident_bytes: u64,
    mmap_spill: bool,
    ram_pair_max_bytes: u64,
) -> Result<Option<PinnedLayer>> {
    compiled.check_inputs_unchanged()?;
    let started = Instant::now();
    let value_bytes = (compiled.state_count() as u64)
        .checked_mul(4)
        .context("compiled L8 value bytes overflow")?;
    let average_bytes = (compiled.averages().len() as u64)
        .checked_mul(4)
        .context("compiled L8 average bytes overflow")?;
    ensure!(
        tuning.can_hold_layers(average_bytes, prepared_resident_bytes, sweep.qstate_count),
        "compiled L8 averages exceed FULL_MDP memory planning limit"
    );
    let ram_bytes = value_bytes
        .checked_add(average_bytes)
        .context("compiled L8 gather memory overflow")?;
    let use_ram = !mmap_spill
        && sweep.ram_next_layer
        && value_bytes <= ram_pair_max_bytes
        && tuning.can_hold_layers(ram_bytes, prepared_resident_bytes, sweep.qstate_count);
    // A file gather uses a bounded block. Reduce it when a narrow memory plan
    // can hold the averages but cannot also hold the usual 8 MB output block.
    let mut chunk_states = GATHER_CHUNK_STATES.min(compiled.state_count().max(1));
    if !use_ram && !mmap_spill {
        loop {
            let working = average_bytes
                .checked_add(chunk_states as u64 * 4)
                .context("compiled L8 file gather memory overflow")?;
            if tuning.can_hold_layers(working, prepared_resident_bytes, sweep.qstate_count) {
                break;
            }
            ensure!(
                chunk_states > 1,
                "compiled L8 file gather exceeds FULL_MDP memory planning limit"
            );
            chunk_states = (chunk_states / 2).max(1);
        }
    }
    // The retained averages and ID mapping have already been charged through
    // prepared_resident_bytes. Only this per-apply numeric average vector is new.
    let averages = compiled.evaluate_averages_f32(terminals)?;
    compiled.check_inputs_unchanged()?;
    sweep.check_prepared_inputs()?;
    let out_path = crate::mdp_f32_value_path(sweep.scratch_dir, 8);
    let (mut pin, mut sink) = if value_bytes == 0 {
        File::create(&out_path)?;
        (Some(PinnedLayer::Ram(Vec::new())), "compiled-empty")
    } else if use_ram {
        let mut output = Vec::new();
        output
            .try_reserve_exact(compiled.state_count())
            .context("allocate compiled L8 RAM gather")?;
        output.resize(compiled.state_count(), 0.0f32);
        compiled.gather_range_into_f32(&averages, 0..compiled.state_count(), &mut output)?;
        (Some(PinnedLayer::Ram(output)), "compiled-ram-gather")
    } else if mmap_spill {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&out_path)
            .with_context(|| format!("open compiled L8 mmap sink {}", out_path.display()))?;
        file.set_len(0)?;
        file.set_len(value_bytes)?;
        let mut map =
            unsafe { memmap2::MmapMut::map_mut(&file) }.context("map compiled L8 output")?;
        // Mapping base is page-aligned and length is exactly state_count * 4.
        let output = unsafe {
            std::slice::from_raw_parts_mut(map.as_mut_ptr() as *mut f32, compiled.state_count())
        };
        compiled.gather_range_into_f32(&averages, 0..compiled.state_count(), output)?;
        let _ = map.flush_async();
        (
            Some(PinnedLayer::Mapped(map.make_read_only()?)),
            "compiled-mmap-gather",
        )
    } else {
        use std::io::Write;
        let file = File::create(&out_path)
            .with_context(|| format!("create compiled L8 file sink {}", out_path.display()))?;
        file.set_len(value_bytes)?;
        let mut writer = std::io::BufWriter::new(file);
        compiled.gather_chunks_f32(&averages, chunk_states, |_, values| {
            #[cfg(target_endian = "little")]
            {
                // All f32 values are initialized. Viewing their representation
                // as bytes adds no allocation to the bounded gather block.
                let bytes = unsafe {
                    std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
                };
                writer.write_all(bytes)?;
            }
            #[cfg(target_endian = "big")]
            for &value in values {
                writer.write_all(&value.to_le_bytes())?;
            }
            Ok(())
        })?;
        writer.flush()?;
        drop(writer);
        (None, "compiled-file-gather")
    };
    drop(averages);
    // As with an ordinary spilled sweep layer, freeing construction scratch
    // may leave room to pin the finished numeric layer for its predecessor.
    if pin.is_none()
        && sweep.ram_next_layer
        && value_bytes <= sweep.ram_next_layer_max_bytes
        && tuning.can_hold_layers(value_bytes, prepared_resident_bytes, sweep.qstate_count)
    {
        pin = Some(PinnedLayer::Ram(crate::load_f32_values_vec(
            &out_path,
            compiled.state_count() as u64,
        )?));
        sink = "compiled-file-gather-ram-pin";
    }
    compiled.check_inputs_unchanged()?;
    eprintln!(
        "  fixed-policy layer=8 states={} unique_averages={} sink={} value_bytes={} elapsed={:.3}s",
        compiled.state_count(),
        compiled.averages().len(),
        sink,
        value_bytes,
        started.elapsed().as_secs_f64()
    );
    Ok(pin)
}
