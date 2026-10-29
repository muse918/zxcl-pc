//! Bounded reuse of an ordered fixed-policy action average within one task.
//!
//! A target field and a validated pre-reveal placement ID determine the complete
//! ordered child list, including the denominator. This cache stores that exact
//! f32 result for the current next-value layer. It never survives a kernel call.

use anyhow::{ensure, Context, Result};

use crate::graph::FieldId;
use crate::pre_reveal::PreRevealIndex;
use crate::qgraph::QForwardGraph;
use crate::qstate::QStateIndex;

enum PlacementIds<'a> {
    PreReveal(&'a PreRevealIndex),
    #[cfg(test)]
    Fixture(Vec<[u32; crate::PIECE_COUNT]>),
}

/// The production constructor checks every placement's ordered reveal support.
/// The borrow also binds the plan to the exact immutable qforward graph used by
/// the kernel; an unrelated graph cannot accidentally reuse these IDs.
pub(crate) struct ValidatedPlacements<'a> {
    ids: PlacementIds<'a>,
    qfwd: &'a QForwardGraph,
    qstate_count: usize,
}

impl<'a> ValidatedPlacements<'a> {
    pub(crate) fn build(
        q_index: &QStateIndex,
        qfwd: &'a QForwardGraph,
        pre: &'a PreRevealIndex,
    ) -> Result<Self> {
        // This validates the support, child order, denominator and placement
        // identity, rather than assuming any supplied graph has those properties.
        let _ = crate::loop1_terminal::TerminalPlan::build(q_index, qfwd, pre)?;
        Ok(Self {
            ids: PlacementIds::PreReveal(pre),
            qfwd,
            qstate_count: q_index.states.len(),
        })
    }

    pub(crate) fn check_graph(&self, qfwd: &QForwardGraph, qstate_count: usize) -> Result<()> {
        ensure!(
            std::ptr::eq(self.qfwd, qfwd) && self.qstate_count == qstate_count,
            "action-average placement plan belongs to a different qforward graph/domain"
        );
        Ok(())
    }

    /// Retained source payload required by this plan, including spare capacity.
    /// Production plans borrow PreRevealIndex without allocating another map;
    /// charge that shared index once (or conservatively twice if another plan
    /// already includes it). The qforward graph uses the existing common reserve.
    pub(crate) fn storage_bytes(&self) -> u64 {
        match &self.ids {
            PlacementIds::PreReveal(pre) => pre.storage_bytes(),
            #[cfg(test)]
            PlacementIds::Fixture(ids) => (ids.capacity() as u64)
                .saturating_mul(std::mem::size_of::<[u32; crate::PIECE_COUNT]>() as u64),
        }
    }

    #[inline]
    pub(crate) fn zid_for_placement(&self, qid: u32, placed: u8) -> Option<u32> {
        match &self.ids {
            PlacementIds::PreReveal(pre) => pre.zid_for_placement(qid, placed),
            #[cfg(test)]
            PlacementIds::Fixture(ids) => ids
                .get(qid as usize)?
                .get(placed as usize)
                .copied()
                .filter(|&id| id != u32::MAX),
        }
    }

    /// Tiny tests need arbitrary ordered reveals, rather than the full quotient
    /// domain. Canonicalize their exact child lists independently of pre-reveal
    /// indexing; this constructor is absent from production builds.
    #[cfg(test)]
    pub(crate) fn from_fixture(qfwd: &'a QForwardGraph) -> Result<Self> {
        use std::collections::HashMap;
        let qstate_count = qfwd.offsets.len();
        let mut expressions = HashMap::<Vec<u32>, u32>::new();
        let mut ids = Vec::with_capacity(qstate_count);
        let mut cursor = 0;
        for offsets in &qfwd.offsets {
            ensure!(
                offsets[0] as usize == cursor,
                "noncontiguous fixture qforward"
            );
            let mut row = [u32::MAX; crate::PIECE_COUNT];
            for placed in 0..crate::PIECE_COUNT {
                let start = offsets[placed] as usize;
                let end = offsets[placed + 1] as usize;
                let recs = qfwd
                    .records
                    .get(start..end)
                    .context("bad fixture qforward offsets")?;
                ensure!(
                    recs.len() <= 7,
                    "fixture reveal list exceeds seven children"
                );
                ensure!(
                    recs.iter().all(|r| (r.child_qid as usize) < qstate_count),
                    "fixture child outside domain"
                );
                if !recs.is_empty() {
                    let children: Vec<_> = recs.iter().map(|r| r.child_qid).collect();
                    let next_id = u32::try_from(expressions.len())?;
                    row[placed] = *expressions.entry(children).or_insert(next_id);
                }
            }
            cursor = offsets[crate::PIECE_COUNT] as usize;
            ids.push(row);
        }
        ensure!(
            cursor == qfwd.records.len(),
            "unreferenced fixture qforward records"
        );
        Ok(Self {
            ids: PlacementIds::Fixture(ids),
            qfwd,
            qstate_count,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ActionSource<'a> {
    pub(crate) placements: &'a ValidatedPlacements<'a>,
    /// Requested payload per task, carved out of the existing lookup allowance.
    pub(crate) requested_bytes: usize,
}

impl ActionSource<'_> {
    pub(crate) fn storage_bytes(&self) -> u64 {
        self.placements.storage_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CachedAverage {
    pub(crate) value: f32,
    pub(crate) missing_terms: u8,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Entry {
    key: u64,
    value: f32,
    missing_terms: u8,
    occupied: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CacheStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) collisions: u64,
    pub(crate) saved_reveals: u64,
    pub(crate) payload_bytes: usize,
}

pub(crate) struct ActionAverageCache {
    entries: Box<[Entry]>,
    mask: usize,
    stats: CacheStats,
}

impl ActionAverageCache {
    pub(crate) fn entry_bytes() -> usize {
        std::mem::size_of::<Entry>()
    }

    /// Keep the lookup's largest-block allowance before reserving cache slots.
    /// With cap=0 (or less than that floor), this returns no cache in O(1).
    pub(crate) fn partition(
        requested_bytes: usize,
        per_task_cache_cap: usize,
        qstate_count: usize,
    ) -> (Option<Self>, usize) {
        let lookup_floor = qstate_count.saturating_mul(16).saturating_add(512);
        let room = per_task_cache_cap.saturating_sub(lookup_floor);
        let bytes = requested_bytes.min(room);
        let slots = bytes / Self::entry_bytes();
        if slots == 0 {
            return (None, per_task_cache_cap);
        }
        // A power-of-two table makes the hot-path slot calculation a bit mask.
        let count = 1usize << (usize::BITS - 1 - slots.leading_zeros());
        let payload_bytes = count * Self::entry_bytes();
        let cache = Self {
            entries: vec![Entry::default(); count].into_boxed_slice(),
            mask: count - 1,
            stats: CacheStats {
                payload_bytes,
                ..CacheStats::default()
            },
        };
        (Some(cache), per_task_cache_cap - payload_bytes)
    }

    #[inline]
    fn key(field: FieldId, zid: u32) -> u64 {
        (u64::from(field) << 32) | u64::from(zid)
    }

    #[inline]
    fn slot(&self, key: u64) -> usize {
        // A noncryptographic finalizer avoids systematic field/qid collisions.
        let mut hash = key;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;
        hash as usize & self.mask
    }

    #[inline]
    pub(crate) fn get(
        &mut self,
        field: FieldId,
        zid: u32,
        reveal_count: usize,
    ) -> Option<CachedAverage> {
        let key = Self::key(field, zid);
        let entry = &self.entries[self.slot(key)];
        if entry.occupied && entry.key == key {
            self.stats.hits += 1;
            self.stats.saved_reveals += reveal_count as u64;
            Some(CachedAverage {
                value: entry.value,
                missing_terms: entry.missing_terms,
            })
        } else {
            self.stats.misses += 1;
            self.stats.collisions += u64::from(entry.occupied);
            None
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, field: FieldId, zid: u32, value: CachedAverage) {
        let key = Self::key(field, zid);
        let slot = self.slot(key);
        self.entries[slot] = Entry {
            key,
            value: value.value,
            missing_terms: value.missing_terms,
            occupied: true,
        };
    }

    pub(crate) fn stats(&self) -> CacheStats {
        self.stats
    }
}

#[cfg(test)]
#[path = "action_avg_cache_tests.rs"]
mod tests;
