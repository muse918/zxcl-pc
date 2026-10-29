//! State/byte bounded work groups shared by greedy and fixed-policy sweeps.
//!
//! The allowance combines resident anonymous value layers, concurrent task
//! buffers, and caches (including a block that exceeds its cache cap). `reserve`
//! covers graph/index/GMRES allocations, cache metadata, allocator overhead and
//! the rest of the process. File mappings/page cache are reclaimable and are not
//! charged as pinned anonymous layers. This is a heap planning limit, not an
//! OS/cgroup memory limit.

use anyhow::{Context, Result};
use std::{env, ops::Range};

use crate::layer_store::FieldChainBlockMeta;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const BUFFER_BASE: u64 = 8 * MIB;
// Conservative across both kernels: grouped u24 input, decoded qids, per-block
// values/policies and both coalesced output buffers. Include Vec growth rather
// than only initialized lengths. Large blocks remain indivisible and are charged
// in full, even when they exceed the configured per-task/state limits.
const BYTES_PER_STATE: u64 = 40;
// Lookup scratch can retain decoded qids, raw qids and raw values independently.
// A miss/promotion also materializes its new block before evicting the old cache.
const LOOKUP_SCRATCH_BYTES_PER_QSTATE: u64 = 32;
// Sparse qids+values, their retained capacity and a cache-entry allowance. RAM
// densification needs at most one qid-domain-sized array as well.
const CACHE_BLOCK_BYTES_PER_QSTATE: u64 = 16;
const CACHE_ENTRY_ALLOWANCE: u64 = 512;
// Cache caps are payload budgets. Charge extra for retained capacity and allocation
// overlap; reserve additionally covers hash-table/deque metadata and the allocator.
const CACHE_MEMORY_FACTOR: u64 = 2;

pub(crate) struct SweepTuning {
    max_blocks: usize,
    max_states: u64,
    max_task_bytes: u64,
    memory_bytes: u64,
    reserve_bytes: u64,
    cache_bytes: usize,
}

pub(crate) struct SweepPlan {
    pub ranges: Vec<Range<usize>>,
    pub workers: usize,
    pub per_task_cache_bytes: usize,
    pub max_working_bytes: usize,
}

fn setting(name: &str, default: u64) -> Result<u64> {
    match env::var(name) {
        Ok(value) => parse_setting(name, &value),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err).with_context(|| format!("invalid {name}")),
    }
}

fn parse_setting(name: &str, value: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("invalid {name}={value}"))
}

impl SweepTuning {
    pub fn from_env(cache_bytes: usize) -> Result<Self> {
        Self::from_settings(cache_bytes, setting)
    }

    fn from_settings(cache_bytes: usize, read: impl Fn(&str, u64) -> Result<u64>) -> Result<Self> {
        let blocks = read("FULL_MDP_BLOCK_CHUNK", 512)?;
        let groups = read("FULL_MDP_LOOKUP_GROUP_CHUNKS", 4)?;
        let memory = read("FULL_MDP_MEMORY_MAX_GIB", 32)?;
        let reserve = read("FULL_MDP_MEMORY_RESERVE_GIB", (memory / 4).min(32))?;
        let max_states = read("FULL_MDP_TASK_MAX_STATES", 2_000_000)?;
        let task_mib = read("FULL_MDP_TASK_MAX_MIB", 64)?;
        anyhow::ensure!(
            blocks > 0 && groups > 0 && max_states > 0 && task_mib > 0,
            "sweep chunk, group, state and byte limits must be positive"
        );
        anyhow::ensure!(
            memory > reserve,
            "FULL_MDP_MEMORY_MAX_GIB must exceed its reserve"
        );
        Ok(Self {
            max_blocks: usize::try_from(
                blocks
                    .checked_mul(groups)
                    .context("sweep block limit overflow")?,
            )?,
            max_states,
            max_task_bytes: task_mib
                .checked_mul(MIB)
                .context("task byte limit overflow")?,
            memory_bytes: memory.checked_mul(GIB).context("memory limit overflow")?,
            reserve_bytes: reserve.checked_mul(GIB).context("reserve overflow")?,
            cache_bytes,
        })
    }

    fn working_bytes(states: u64, qstate_count: usize) -> Result<u64> {
        states
            .checked_mul(BYTES_PER_STATE)
            .and_then(|n| n.checked_add(BUFFER_BASE))
            .and_then(|n| {
                (qstate_count as u64)
                    .checked_mul(LOOKUP_SCRATCH_BYTES_PER_QSTATE)
                    .and_then(|scratch| n.checked_add(scratch))
            })
            .context("task working memory overflow")
    }

    fn cache_floor(qstate_count: usize) -> Result<u64> {
        (qstate_count as u64)
            .checked_mul(CACHE_BLOCK_BYTES_PER_QSTATE)
            .and_then(|n| n.checked_add(CACHE_ENTRY_ALLOWANCE))
            .context("cache floor overflow")
    }

    fn resident_bytes(&self, current_bytes: u64, next_bytes: u64) -> Result<u64> {
        self.reserve_bytes
            .checked_add(current_bytes)
            .and_then(|n| n.checked_add(next_bytes))
            .context("resident memory overflow")
    }

    /// Test a candidate anonymous RAM output before allocating it. The domain
    /// bounds the largest indivisible block and lookup scratch, so this includes
    /// the same single-task/cache floor as `plan`. The actual metadata plan should
    /// still be made before allocating the layer.
    pub fn can_hold_layers(
        &self,
        current_bytes: u64,
        next_bytes: u64,
        qstate_count: usize,
    ) -> bool {
        let required = (|| -> Result<u64> {
            let working =
                Self::working_bytes(qstate_count as u64, qstate_count)?.max(self.max_task_bytes);
            let cache = Self::cache_floor(qstate_count)?
                .checked_mul(CACHE_MEMORY_FACTOR)
                .context("cache working memory overflow")?;
            self.resident_bytes(current_bytes, next_bytes)?
                .checked_add(working)
                .and_then(|n| n.checked_add(cache))
                .context("layer fit memory overflow")
        })();
        required.is_ok_and(|n| n <= self.memory_bytes)
    }

    pub fn plan(
        &self,
        metas: &[FieldChainBlockMeta],
        current_bytes: u64,
        next_pinned_bytes: u64,
        requested_workers: usize,
        qstate_count: usize,
    ) -> Result<SweepPlan> {
        let resident = self.resident_bytes(current_bytes, next_pinned_bytes)?;
        let available = self.memory_bytes.checked_sub(resident)
            .context("resident layers exceed sweep memory budget; reduce RAM layer limits or increase FULL_MDP_MEMORY_MAX_GIB")?;
        if metas.is_empty() {
            // Keep a valid one-thread pool configuration without charging a task
            // that will never run, or reserving a cache that cannot be accessed.
            return Ok(SweepPlan {
                ranges: Vec::new(),
                workers: 1,
                per_task_cache_bytes: 0,
                max_working_bytes: 0,
            });
        }

        let mut ranges = Vec::new();
        let mut start = 0;
        let mut states = 0u64;
        let mut max_working = 0;
        for (i, meta) in metas.iter().enumerate() {
            anyhow::ensure!(
                meta.count as usize <= qstate_count,
                "field {} has {} qids, exceeds domain {}",
                meta.field_id,
                meta.count,
                qstate_count
            );
            let next = states
                .checked_add(meta.count as u64)
                .context("group state overflow")?;
            let bytes = Self::working_bytes(next, qstate_count)?;
            if i > start
                && (i - start >= self.max_blocks
                    || next > self.max_states
                    || bytes > self.max_task_bytes)
            {
                ranges.push(start..i);
                max_working = max_working.max(Self::working_bytes(states, qstate_count)?);
                start = i;
                states = 0;
            }
            states = states
                .checked_add(meta.count as u64)
                .context("group state overflow")?;
        }
        ranges.push(start..metas.len());
        max_working = max_working.max(Self::working_bytes(states, qstate_count)?);

        // Both lookup implementations retain one block even when the requested
        // cap is zero/smaller than that block. Never allocate concurrency under
        // the assumption that a zero cache setting means zero cache memory.
        let cache_floor = Self::cache_floor(qstate_count)?;
        let minimum_per_worker = cache_floor
            .checked_mul(CACHE_MEMORY_FACTOR)
            .and_then(|n| n.checked_add(max_working))
            .context("worker memory overflow")?;
        anyhow::ensure!(
            available >= minimum_per_worker,
            "sweep budget has {} bytes for a task needing {}; reduce RAM layer limits or increase FULL_MDP_MEMORY_MAX_GIB",
            available, minimum_per_worker
        );
        let workers = requested_workers
            .max(1)
            .min(ranges.len())
            .min(usize::try_from(available / minimum_per_worker).unwrap_or(usize::MAX));
        // Worker selection proves this subtraction cannot underflow. Divide out
        // the extra cache allowance before returning the actual payload cap.
        let cache_room = (available / workers as u64 - max_working) / CACHE_MEMORY_FACTOR;
        let per_task_cache_bytes =
            (self.cache_bytes / workers).min(usize::try_from(cache_room).unwrap_or(usize::MAX));
        Ok(SweepPlan {
            ranges,
            workers,
            per_task_cache_bytes,
            max_working_bytes: usize::try_from(max_working)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> SweepTuning {
        SweepTuning {
            max_blocks: 4,
            max_states: 200,
            max_task_bytes: 64 * MIB,
            memory_bytes: GIB,
            reserve_bytes: 128 * MIB,
            cache_bytes: 512 * MIB as usize,
        }
    }
    fn metas(counts: &[u32]) -> Vec<FieldChainBlockMeta> {
        let mut offset = 0;
        counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let m = FieldChainBlockMeta {
                    field_id: i as u32,
                    count,
                    chain_offset: offset,
                };
                offset += count as u64;
                m
            })
            .collect()
    }
    fn charged_bytes(
        t: &SweepTuning,
        p: &SweepPlan,
        current: u64,
        next: u64,
        domain: usize,
    ) -> u64 {
        t.reserve_bytes
            + current
            + next
            + p.workers as u64
                * (p.max_working_bytes as u64
                    + CACHE_MEMORY_FACTOR
                        * (p.per_task_cache_bytes as u64)
                            .max(SweepTuning::cache_floor(domain).unwrap()))
    }
    #[test]
    fn uneven_work_is_contiguous_and_state_bounded() {
        let m = metas(&[90, 90, 90, 500, 1, 1, 1, 1, 1]);
        let p = tuning().plan(&m, 0, 0, 20, 1000).unwrap();
        assert_eq!(p.ranges, vec![0..2, 2..3, 3..4, 4..8, 8..9]);
        assert!(p.ranges.windows(2).all(|w| w[0].end == w[1].start));
        assert_eq!(
            p.max_working_bytes as u64,
            SweepTuning::working_bytes(500, 1000).unwrap()
        );
    }
    #[test]
    fn block_count_bounds_zero_state_runs_without_empty_ranges() {
        let p = tuning().plan(&metas(&[0; 9]), 0, 0, 0, 0).unwrap();
        assert_eq!(p.ranges, vec![0..4, 4..8, 8..9]);
        assert_eq!(p.workers, 1);
        assert!(p.ranges.iter().all(|r| !r.is_empty()));
    }
    #[test]
    fn every_group_respects_both_bounds_unless_one_indivisible_block() {
        let counts = [1, 40, 0, 100, 99, 500, 0, 501, 2, 3, 10, 10, 199, 1];
        let m = metas(&counts);
        for state_limit in [1, 100, 200, 1000] {
            for byte_states in [0, 50, 200, 1000] {
                let mut t = tuning();
                t.max_states = state_limit;
                t.max_task_bytes = SweepTuning::working_bytes(byte_states, 1000).unwrap();
                let p = t.plan(&m, 0, 0, 12, 1000).unwrap();
                assert_eq!(p.ranges.first().unwrap().start, 0);
                assert_eq!(p.ranges.last().unwrap().end, counts.len());
                for range in &p.ranges {
                    assert!(!range.is_empty());
                    assert!(range.len() <= t.max_blocks);
                    let states = counts[range.clone()].iter().map(|&n| n as u64).sum::<u64>();
                    let bytes = SweepTuning::working_bytes(states, 1000).unwrap();
                    if states > t.max_states || bytes > t.max_task_bytes {
                        assert_eq!(range.len(), 1);
                    }
                    assert!(bytes <= p.max_working_bytes as u64);
                }
                assert!(p.ranges.windows(2).all(|r| r[0].end == r[1].start));
                assert!(charged_bytes(&t, &p, 0, 0, 1000) <= t.memory_bytes);
            }
        }
    }
    #[test]
    fn task_bytes_include_retained_lookup_scratch() {
        let mut t = tuning();
        t.max_states = 10_000_000;
        t.max_task_bytes = SweepTuning::working_bytes(100, 1000).unwrap();
        let p = t.plan(&metas(&[60, 60, 60]), 0, 0, 4, 1000).unwrap();
        assert_eq!(p.ranges, vec![0..1, 1..2, 2..3]);
    }
    #[test]
    fn oversized_block_is_isolated_and_charged_even_after_zero_count_blocks() {
        let mut t = tuning();
        t.max_task_bytes = SweepTuning::working_bytes(100, 1000).unwrap();
        let p = t
            .plan(&metas(&[0, 10, 500, 0, 10]), 0, 0, 20, 1000)
            .unwrap();
        assert_eq!(p.ranges, vec![0..2, 2..3, 3..5]);
        assert!(p.max_working_bytes as u64 > t.max_task_bytes);
    }
    #[test]
    fn concurrency_and_cache_share_one_budget() {
        let t = tuning();
        let p = t
            .plan(&metas(&[100; 100]), 700 * MIB, 40 * MIB, 64, 1000)
            .unwrap();
        assert!(charged_bytes(&t, &p, 700 * MIB, 40 * MIB, 1000) <= t.memory_bytes);
        assert!(p.workers < 64);
        assert!(p.per_task_cache_bytes < t.cache_bytes / p.workers);
        assert!(t.plan(&metas(&[100]), GIB, 0, 1, 1000).is_err());
    }
    #[test]
    fn zero_cache_still_charges_single_block_and_materialization() {
        let mut t = tuning();
        t.cache_bytes = 0;
        t.max_states = 100;
        let domain = 800_000;
        let minimum = SweepTuning::working_bytes(100, domain).unwrap()
            + CACHE_MEMORY_FACTOR * SweepTuning::cache_floor(domain).unwrap();
        t.memory_bytes = t.reserve_bytes + 3 * minimum - 1;
        let p = t.plan(&metas(&[100; 100]), 0, 0, 100, domain).unwrap();
        assert_eq!(p.workers, 2);
        assert_eq!(p.per_task_cache_bytes, 0);
        assert!(charged_bytes(&t, &p, 0, 0, domain) <= t.memory_bytes);
    }
    #[test]
    fn empty_layer_needs_no_task_or_cache() {
        let mut t = tuning();
        t.memory_bytes = t.reserve_bytes;
        let p = t.plan(&[], 0, 0, 0, usize::MAX).unwrap();
        assert!(p.ranges.is_empty());
        assert_eq!(
            (p.workers, p.per_task_cache_bytes, p.max_working_bytes),
            (1, 0, 0)
        );
        assert!(t.plan(&[], 1, 0, 0, 0).is_err());
    }
    #[test]
    fn impossible_single_block_and_invalid_domain_are_rejected() {
        let mut t = tuning();
        t.memory_bytes = t.reserve_bytes + BUFFER_BASE;
        assert!(t.plan(&metas(&[100]), 0, 0, 1, 1000).is_err());
        assert!(tuning().plan(&metas(&[1001]), 0, 0, 1, 1000).is_err());
        assert!(tuning().plan(&metas(&[1]), 0, 0, 1, usize::MAX).is_err());
        assert!(tuning().plan(&metas(&[1]), u64::MAX, 1, 1, 1000).is_err());
    }
    #[test]
    fn candidate_layer_fit_includes_domain_and_agrees_with_plan() {
        let mut t = tuning();
        let domain = 800_000;
        let headroom = SweepTuning::working_bytes(domain as u64, domain)
            .unwrap()
            .max(t.max_task_bytes)
            + CACHE_MEMORY_FACTOR * SweepTuning::cache_floor(domain).unwrap();
        t.memory_bytes = t.reserve_bytes + 100 * MIB + headroom;
        assert!(t.can_hold_layers(60 * MIB, 40 * MIB, domain));
        assert!(t
            .plan(&metas(&[domain as u32]), 60 * MIB, 40 * MIB, 20, domain)
            .is_ok());
        assert!(!t.can_hold_layers(60 * MIB + 1, 40 * MIB, domain));
        assert!(!t.can_hold_layers(u64::MAX, 1, domain));
        assert!(!t.can_hold_layers(0, 0, usize::MAX));
    }
    #[test]
    fn malformed_environment_numbers_are_rejected() {
        for value in ["", "bad", "-1", "1.5", " 12", "12 ", "18446744073709551616"] {
            let error = parse_setting("FULL_MDP_TASK_MAX_STATES", value).unwrap_err();
            assert!(error.to_string().contains("FULL_MDP_TASK_MAX_STATES"));
        }
        assert_eq!(parse_setting("FULL_MDP_TASK_MAX_STATES", "12").unwrap(), 12);
    }
    #[test]
    fn defaults_and_invalid_settings_are_checked_without_mutating_env() {
        let defaults = SweepTuning::from_settings(0, |_, default| Ok(default)).unwrap();
        assert_eq!(defaults.memory_bytes, 32 * GIB);
        assert_eq!(defaults.reserve_bytes, 8 * GIB);
        for (key, value) in [
            ("FULL_MDP_BLOCK_CHUNK", 0),
            ("FULL_MDP_LOOKUP_GROUP_CHUNKS", 0),
            ("FULL_MDP_TASK_MAX_STATES", 0),
            ("FULL_MDP_TASK_MAX_MIB", 0),
            ("FULL_MDP_MEMORY_MAX_GIB", 0),
            ("FULL_MDP_MEMORY_RESERVE_GIB", 32),
            ("FULL_MDP_BLOCK_CHUNK", u64::MAX),
            ("FULL_MDP_TASK_MAX_MIB", u64::MAX),
            ("FULL_MDP_MEMORY_MAX_GIB", u64::MAX),
        ] {
            assert!(
                SweepTuning::from_settings(0, |name, default| Ok(if name == key {
                    value
                } else {
                    default
                }))
                .is_err(),
                "accepted {key}={value}"
            );
        }
        assert!(SweepTuning::from_settings(0, |name, default| {
            if name == "FULL_MDP_TASK_MAX_STATES" {
                anyhow::bail!("invalid {name}=bad");
            }
            Ok(default)
        })
        .is_err());
    }
}
