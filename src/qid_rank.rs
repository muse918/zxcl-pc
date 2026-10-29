//! Membership and rank for sorted qids, without copying their associated values.
//!
//! A successful lookup returns the qid's original position in the sorted input.
//! The caller can add its field's chain offset and read the pinned value array.

#[derive(Debug)]
pub(crate) struct QidRankIndex {
    first_word: usize,
    words: Box<[u64]>,
    ranks: Box<[u32]>,
    count: usize,
    domain: usize,
}

impl QidRankIndex {
    /// Heap bytes needed for the inclusive word range spanned by `sorted_qids`.
    /// Endpoints suffice here; `from_sorted` validates order and domain before use.
    pub(crate) fn estimated_heap_bytes(sorted_qids: &[u32]) -> usize {
        match (sorted_qids.first(), sorted_qids.last()) {
            (Some(&first), Some(&last)) if first <= last => {
                ((last as usize / 64) - (first as usize / 64) + 1)
                    * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
            }
            _ => 0,
        }
    }

    pub(crate) fn from_sorted(sorted_qids: &[u32], domain: usize) -> Result<Self, &'static str> {
        if sorted_qids.len() > u32::MAX as usize {
            return Err("qid rank input exceeds u32 rank capacity");
        }
        if sorted_qids
            .last()
            .is_some_and(|&qid| qid as usize >= domain)
        {
            return Err("qid rank input is outside the qstate domain");
        }
        if sorted_qids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("qid rank input is not strictly sorted");
        }
        let first_word = sorted_qids.first().map_or(0, |&qid| qid as usize / 64);
        let word_count = sorted_qids
            .last()
            .map_or(0, |&qid| qid as usize / 64 + 1 - first_word);
        let mut words = vec![0u64; word_count].into_boxed_slice();
        for &qid in sorted_qids {
            words[qid as usize / 64 - first_word] |= 1u64 << (qid % 64);
        }
        let mut ranks = Vec::with_capacity(word_count);
        let mut rank = 0u32;
        for &word in &words {
            ranks.push(rank);
            rank += word.count_ones();
        }
        Ok(Self {
            first_word,
            words,
            ranks: ranks.into_boxed_slice(),
            count: sorted_qids.len(),
            domain,
        })
    }

    #[inline]
    pub(crate) fn rank(&self, qid: u32) -> Option<usize> {
        let qid = qid as usize;
        if qid >= self.domain {
            return None;
        }
        let word = (qid / 64).checked_sub(self.first_word)?;
        let bits = *self.words.get(word)?;
        let bit = 1u64 << (qid % 64);
        if bits & bit == 0 {
            return None;
        }
        Some(self.ranks[word] as usize + (bits & (bit - 1)).count_ones() as usize)
    }

    pub(crate) fn iter_qids(&self) -> impl Iterator<Item = u32> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(move |(word, &bits)| {
                let mut remaining = bits;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros();
                    remaining &= remaining - 1;
                    Some(((self.first_word + word) * 64 + bit as usize) as u32)
                })
            })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.words.as_ref()) + std::mem::size_of_val(self.ranks.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_handles_edges_gaps_and_domain() {
        let qids = [0, 1, 63, 64, 65, 127, 129, 1_023, 1_024, 1_088];
        let bitmap = QidRankIndex::from_sorted(&qids, 1_089).unwrap();
        for qid in 0..1_200 {
            assert_eq!(bitmap.rank(qid), qids.binary_search(&qid).ok(), "qid={qid}");
        }
        assert_eq!(bitmap.rank(u32::MAX), None);
        assert_eq!(bitmap.iter_qids().collect::<Vec<_>>(), qids);
        assert_eq!(bitmap.len(), qids.len());
        assert_eq!(
            bitmap.heap_bytes(),
            QidRankIndex::estimated_heap_bytes(&qids)
        );
    }

    #[test]
    fn trimmed_origin_empty_and_invalid_inputs() {
        let qids = [511, 512, 577, 639, 640];
        let bitmap = QidRankIndex::from_sorted(&qids, 641).unwrap();
        assert_eq!(bitmap.heap_bytes(), 4 * 12);
        assert_eq!(bitmap.rank(0), None);
        assert_eq!(bitmap.rank(510), None);
        assert_eq!(bitmap.iter_qids().collect::<Vec<_>>(), qids);
        let empty = QidRankIndex::from_sorted(&[], 0).unwrap();
        assert_eq!(empty.rank(0), None);
        assert_eq!(empty.heap_bytes(), 0);
        assert_eq!(empty.iter_qids().count(), 0);
        assert!(QidRankIndex::from_sorted(&[1, 1], 2).is_err());
        assert!(QidRankIndex::from_sorted(&[2, 1], 3).is_err());
        assert!(QidRankIndex::from_sorted(&[0, 64], 64).is_err());
        assert!(QidRankIndex::from_sorted(&[0], 0).is_err());
    }

    #[test]
    fn sparse_bitmap_and_dense_value_bits_match() {
        let domain = 100_003;
        let qids = (0..domain as u32)
            .filter(|qid| qid % 7 == 0 || qid % 13 == 3)
            .collect::<Vec<_>>();
        let bitmap = QidRankIndex::from_sorted(&qids, domain).unwrap();
        let values = qids
            .iter()
            .enumerate()
            .map(|(i, _)| match i % 5 {
                0 => -0.0,
                1 => 0.0,
                _ => (i as f32) / 17.0,
            })
            .collect::<Vec<_>>();
        let mut dense = vec![0.0f32; domain];
        for (i, qid) in bitmap.iter_qids().enumerate() {
            dense[qid as usize] = values[i];
        }
        for qid in 0..domain as u32 {
            let sparse = qids.binary_search(&qid).ok().map_or(0.0, |i| values[i]);
            let ranked = bitmap.rank(qid).map_or(0.0, |i| values[i]);
            assert_eq!(ranked.to_bits(), sparse.to_bits(), "bitmap qid={qid}");
            assert_eq!(
                dense[qid as usize].to_bits(),
                sparse.to_bits(),
                "dense qid={qid}"
            );
        }
    }

    /// `rustc --test -O src/qid_rank.rs -o /tmp/qid-rank-tests` or a release cargo test.
    /// A microbenchmark of lookups only; it does not predict end-to-end sweep speed.
    #[test]
    #[ignore]
    fn benchmark_index_modes_release() {
        use std::hint::black_box;
        use std::time::Instant;
        assert!(
            !cfg!(debug_assertions),
            "run this benchmark with --release / rustc -O"
        );
        const DOMAIN: usize = 817_740;
        const QUERIES: usize = 1_000_000;
        let mut rng = 0x7214_813b_aede_9077u64;
        let mut next_random = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as usize
        };
        for count in [32usize, 512, 8_192, 40_000, 100_000, 400_000] {
            let qids = (0..count)
                .map(|i| (i * DOMAIN / count) as u32)
                .collect::<Vec<_>>();
            let values = (0..count)
                .map(|i| (i % 997) as f32 / 997.0)
                .collect::<Vec<_>>();
            let build_start = Instant::now();
            let bitmap = QidRankIndex::from_sorted(&qids, DOMAIN).unwrap();
            let build = build_start.elapsed();
            let mut dense = vec![0.0f32; DOMAIN];
            for (i, &qid) in qids.iter().enumerate() {
                dense[qid as usize] = values[i];
            }
            for hit_pct in [0usize, 50, 100] {
                let queries = (0..QUERIES)
                    .map(|_| {
                        if next_random() % 100 < hit_pct {
                            qids[next_random() % count]
                        } else {
                            (next_random() % DOMAIN) as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let mut sums = [0.0f64; 3];
                let mut elapsed = [std::time::Duration::ZERO; 3];
                for mode in 0..3 {
                    let start = Instant::now();
                    let mut sum = 0.0f64;
                    for &qid in black_box(&queries) {
                        let v = match mode {
                            0 => qids.binary_search(&qid).ok().map_or(0.0, |i| values[i]),
                            1 => bitmap.rank(qid).map_or(0.0, |i| values[i]),
                            _ => dense[qid as usize],
                        };
                        sum += black_box(v) as f64;
                    }
                    elapsed[mode] = start.elapsed();
                    sums[mode] = black_box(sum);
                }
                assert_eq!(sums[0].to_bits(), sums[1].to_bits());
                assert_eq!(sums[0].to_bits(), sums[2].to_bits());
                eprintln!("qid-rank-bench count={count} forced_hit_pct={hit_pct} sparse_bytes={} bitmap_bytes={} dense_bytes={} bitmap_build_us={} sparse_ns={:.2} bitmap_ns={:.2} dense_ns={:.2}",
                    qids.len() * 4, bitmap.heap_bytes(), dense.len() * 4, build.as_micros(),
                    elapsed[0].as_nanos() as f64 / QUERIES as f64,
                    elapsed[1].as_nanos() as f64 / QUERIES as f64,
                    elapsed[2].as_nanos() as f64 / QUERIES as f64);
            }
        }
    }
}
