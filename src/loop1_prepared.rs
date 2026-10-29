//! Policy-scoped L9 compilation for repeated full boundary-map applications.
//!
//! The full boundary coordinates and the ordered legacy f32 terminal formulas
//! are retained. This is a cached representation of one fixed policy, rebuilt
//! when the policy changes. Path and open-file identities are checked before
//! every application; inputs must remain immutable during a solve.

use anyhow::{ensure, Context, Result};
use std::fs::{self, File, Metadata};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::graph::FieldGraph;
use crate::layer_store::FieldChainLayer;
use crate::loop1_stream::{CompiledL9, Loop1Limits};
use crate::loop1_terminal::TerminalPlan;
use crate::pre_reveal::PreRevealIndex;
use crate::qgraph::QForwardGraph;
use crate::qstate::QStateIndex;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
    mode: u32,
}

impl Stamp {
    fn of(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
            mode: metadata.mode(),
        }
    }
}

struct BoundInput {
    request: PathBuf,
    canonical: PathBuf,
    file: File,
    stamp: Stamp,
    request_stamp: Stamp,
}

/// Snapshot file identities before loading inputs, then retain handles for the
/// solve. Metadata checks avoid hashing the multi-hundred-GiB source tables.
pub(crate) struct SourceBindings(Vec<BoundInput>);

impl SourceBindings {
    pub(crate) fn capture_paths(paths: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let mut inputs = Vec::new();
        for path in paths {
            let request = if path.is_absolute() {
                path
            } else {
                std::env::current_dir()?.join(path)
            };
            let request_stamp = Stamp::of(&fs::symlink_metadata(&request)?);
            let canonical = fs::canonicalize(&request)?;
            let file = File::open(&request)
                .with_context(|| format!("open compiled L9 input {}", request.display()))?;
            let metadata = file.metadata()?;
            ensure!(
                metadata.is_file(),
                "compiled L9 input is not a regular file: {}",
                request.display()
            );
            inputs.push(BoundInput {
                request,
                canonical,
                file,
                stamp: Stamp::of(&metadata),
                request_stamp,
            });
        }
        let out = Self(inputs);
        out.check()?;
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn capture(
        graph_path: &Path,
        layers: &[FieldChainLayer],
        policy_dir: &Path,
    ) -> Result<Self> {
        let mut paths = vec![graph_path.to_path_buf()];
        for (layer, input) in layers.iter().enumerate() {
            paths.extend([
                input.index_path.clone(),
                input.chains_path.clone(),
                crate::mdp_policy_u16_path(policy_dir, layer),
            ]);
        }
        Self::capture_paths(paths)
    }

    pub(crate) fn check(&self) -> Result<()> {
        for input in &self.0 {
            ensure!(
                Stamp::of(&input.file.metadata()?) == input.stamp,
                "compiled L9 source handle changed: {}",
                input.request.display()
            );
            ensure!(
                Stamp::of(&fs::symlink_metadata(&input.request)?) == input.request_stamp
                    && fs::canonicalize(&input.request)? == input.canonical
                    && Stamp::of(&fs::metadata(&input.request)?) == input.stamp,
                "compiled L9 source path changed: {}",
                input.request.display()
            );
        }
        Ok(())
    }
}

pub(crate) fn enabled_from_env() -> bool {
    std::env::var("PI_LOOP1_COMPILED_L9")
        .ok()
        .is_some_and(|value| {
            value != "0"
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("no")
        })
}

/// Reject destructive filesystem aliases before a solver creates scratch
/// files. An initial value may also be the accepted output because publishing
/// that output is atomic; graph, membership and policy inputs stay protected.
pub(crate) fn reject_solver_aliases(
    immutable_inputs: &[PathBuf],
    initial: Option<&Path>,
    output: &Path,
    scratch_dir: &Path,
) -> Result<()> {
    let applied = output.with_extension("applied.f32.bin");
    reject_aux_output_aliases(&applied, immutable_inputs, initial, output)?;
    for protected in immutable_inputs
        .iter()
        .map(PathBuf::as_path)
        .chain(initial)
        .chain([output, applied.as_path()])
    {
        crate::boundary_commands::reject_scratch_output_aliases(protected, scratch_dir)
            .with_context(|| {
                format!(
                    "fixed-policy scratch aliases protected path {}",
                    protected.display()
                )
            })?;
    }
    for destination in [output, applied.as_path()] {
        for input in immutable_inputs {
            ensure!(
                !crate::boundary_commands::paths_alias(destination, input)?,
                "fixed-policy output aliases immutable input: {} and {}",
                destination.display(),
                input.display()
            );
        }
    }
    if let Some(initial) = initial {
        ensure!(
            !crate::boundary_commands::paths_alias(&applied, initial)?,
            "fixed-policy applied output aliases initial value: {} and {}",
            applied.display(),
            initial.display()
        );
    }
    Ok(())
}

pub(crate) fn reject_aux_output_aliases(
    destination: &Path,
    immutable_inputs: &[PathBuf],
    initial: Option<&Path>,
    accepted: &Path,
) -> Result<()> {
    for protected in immutable_inputs
        .iter()
        .map(PathBuf::as_path)
        .chain(initial)
        .chain([accepted])
    {
        ensure!(
            !crate::boundary_commands::paths_alias(destination, protected)?,
            "fixed-policy auxiliary output aliases protected path: {} and {}",
            destination.display(),
            protected.display()
        );
    }
    Ok(())
}

fn mib_setting(name: &str, default: u64) -> Result<u64> {
    let value = match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .with_context(|| format!("invalid {name}"))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
    };
    value
        .checked_mul(1024 * 1024)
        .with_context(|| format!("{name} overflows bytes"))
}

pub(crate) struct PreparedL9<'a> {
    pre: &'a PreRevealIndex,
    plan: TerminalPlan,
    pub(crate) compiled: CompiledL9<'a>,
    bindings: SourceBindings,
    resident_bytes: u64,
}

impl<'a> PreparedL9<'a> {
    pub(crate) fn build(
        graph: &'a FieldGraph,
        q_index: &'a QStateIndex,
        qfwd: &'a QForwardGraph,
        pre: &'a PreRevealIndex,
        layers: &[FieldChainLayer],
        policy_dir: &Path,
        bindings: SourceBindings,
    ) -> Result<Self> {
        ensure!(layers.len() == 10, "compiled L9 needs exactly ten layers");
        bindings.check()?;
        let start = Instant::now();
        let plan = TerminalPlan::build(q_index, qfwd, pre)?;
        let limits = Loop1Limits {
            max_mapping_bytes: mib_setting("PI_LOOP1_L9_MAX_MAPPING_MIB", 512)?,
            max_cache_bytes: usize::try_from(mib_setting("PI_LOOP1_L9_BUILD_CACHE_MIB", 32)?)?,
            ..Loop1Limits::default()
        };
        let mut compiled = CompiledL9::build(
            graph,
            q_index,
            qfwd,
            pre,
            &layers[9],
            &crate::mdp_policy_u16_path(policy_dir, 9),
            limits,
        )?;
        compiled.build_rank_index(
            mib_setting("PI_LOOP1_L9_RANK_MAX_MIB", 64)?,
            limits.max_cache_bytes,
        )?;
        bindings.check()?;
        // Charge a conservative upper bound for retained pre-reveal vectors
        // (including their growth), plus the exact compiled logical payload.
        // Apply-time terminal values are charged as well. General allocator and
        // source-index overhead continue to use the scheduler's reserve.
        let resident_bytes = (q_index.states.len() as u64)
            .checked_add(pre.len() as u64)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(plan.storage_bytes() as u64 * 2))
            .and_then(|n| n.checked_add(compiled.storage_bytes()))
            .and_then(|n| n.checked_add(pre.len() as u64 * 4))
            .context("prepared L9 resident size overflow")?;
        eprintln!("compiled-l9 states={} terminal_formulas={} mapping_bytes={} membership_rank_bytes={} planned_resident_bytes={} build_elapsed={:.3}s",
            compiled.state_count(), compiled.used_terminal_zids().len(), compiled.stats().mapping_bytes,
            compiled.stats().membership_rank_bytes, resident_bytes, start.elapsed().as_secs_f64());
        Ok(Self {
            pre,
            plan,
            compiled,
            bindings,
            resident_bytes,
        })
    }

    pub(crate) fn check_inputs_unchanged(&self) -> Result<()> {
        self.bindings.check()?;
        self.compiled.check_inputs_unchanged()
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub(crate) fn terminal_values(&self, reset: &[f32], include_reward: bool) -> Result<Vec<f32>> {
        self.plan.evaluate_f32(reset, include_reward)
    }

    pub(crate) fn terminal_table(&self, terminals: &[f32]) -> Result<Vec<[f32; 7]>> {
        terminal_table(self.pre, terminals)
    }
}

pub(crate) fn terminal_table(pre: &PreRevealIndex, terminals: &[f32]) -> Result<Vec<[f32; 7]>> {
    ensure!(
        terminals.len() == pre.len(),
        "compiled L9 terminal count mismatch"
    );
    let mut table = vec![[0.0; 7]; pre.qid_to_zid.len()];
    for (qid, row) in table.iter_mut().enumerate() {
        for (placed, value) in row.iter_mut().enumerate() {
            if let Some(zid) = pre.zid_for_placement(qid as u32, placed as u8) {
                *value = terminals[zid as usize];
            }
        }
    }
    Ok(table)
}
