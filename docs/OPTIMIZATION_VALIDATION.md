# Solver optimization validation

This records the initial sweep optimization pass. Subsequent boundary/DAG changes
and their validation are documented in [BOUNDARY_DAG.md](BOUNDARY_DAG.md).

Validation date: 2026-10-29. The changes preserve the existing numerical contract;
no acceptance tolerance was relaxed. Evidence below covers unit/reference checks,
wrapper failure injection, actual GMRES CLI checks, and bounded real-data kernels.
It does **not** establish full-MDP runtime or peak RSS improvements.

## Validation matrix

| Change | Checked behavior and evidence |
|---|---|
| Certified resume and seed publication | `tests/test_solve_resume.py`: changed graph/index/chains/policy/parameters and copied completion records invalidate reuse; interrupted or preallocated outputs are not certified; inputs changed during a stage and retargeted symlinks are detected. Seed rebuilds start in a clean invocation-owned directory, and failure preserves the previously accepted seed. Hash caching uses file version metadata; completion records also bind the output destination. |
| Contiguous chain offsets | `src/layer_store.rs`: RAM and index-only loaders reject gaps, overlaps, count overflow and header/chain-size disagreement before grouped reads use offsets. Valid contiguous block ranges retain their original order. |
| GMRES warm start and publication | The wrapper passes the previous accepted value after round one. A converged start skips basis allocation/iterations, including the unused basis-cap check, but still runs the final affine f32 `T_pi(v)` sweep. Only finite candidates passing the actual infinity-norm residual gate are atomically published; failures preserve accepted output and receive no completion record. The wrapper default remains `PI_ACCEPT_BELLMAN_INF=3e-3` (direct CLI fallback: `1e-3`). |
| PI policy-only rolling values | `src/greedy_emit_tests.rs`: rolling RAM/file sweeps match full snapshots for values and policy bytes, retain layer 0, remove invocation-owned scratch, and preserve unrelated files. Standalone/final materialization still emits all layers. |
| Group lookup reuse and bulk policy reads | Greedy and fixed-policy tests compare grouped versus original/scalar paths across file, RAM and mmap sources/sinks, including empty layers, incomplete reads and invalid policy codes. They preserve first-action ties, reveal summation order, gamma application after the sum, and affine versus homogeneous fixed-policy behavior. Optional counters do not change output. |
| State/byte scheduler and combined heap planning | `src/sweep_schedule.rs`: contiguous coverage, uneven/empty groups, malformed settings, indivisible oversized fields, cache floors, retained lookup scratch, concurrent workers and candidate RAM-layer admission are checked. Workers/cache payload shrink or RAM output falls back to a file when needed. `FULL_MDP_MEMORY_RESERVE_GIB` must manually include the graph, indexes, **GMRES basis**, cache metadata and allocation overhead. This is a planning allowance, not a hard process/RSS cap; mappings and OS page cache are excluded from pinned anonymous-layer bytes. |
| Backward workspace and spill selection reuse | `src/qexternal_frontier_tests.rs`: reused child maps/read buffers, oversized-buffer release, spill ordering and membership output match independent allocating/scan references. The ignored entrypoint check exercises immediate, deferred and strong frontier modes on bounded fixtures. |
| Optional RAM bitmap+rank lookup | `src/qid_rank.rs` and `src/ram_qid_tests.rs`: sparse/bitmap/dense results match bitwise, including gaps, absent fields, nonzero original chain offsets, eviction and dense promotion. Bitmap metadata points into pinned values instead of copying them. `FULL_MDP_RAM_BITMAP_MIN_COUNT=0`/unset keeps it disabled; `128` opts in. |
| Safe f32 RAM loader | Validation found and fixed unsafe exposure of uninitialized destination storage. `src/f32_ram.rs` now reads into `MaybeUninit<f32>` allocation and exposes values only after every byte is initialized. Five tests cover exact little-endian bits (including NaN/infinity), empty files, wrong sizes/overflow, interrupted/partial reads, and EOF/error after partial initialization. |

## Test results and logs

All paths below are relative to the repository root. Logs are local evidence under
`work/optimization-validation/`; older debug/benchmark logs include failures and
interrupted experiments superseded by the named final logs.

| Check | Latest completed result | Evidence |
|---|---|---|
| Normal debug suite | 55 passed: 4 `find_layer0_worst` + 51 main; 6 ignored | `final-debug-tests.log` |
| Normal release suite | 55 passed: 4 `find_layer0_worst` + 51 main; 6 ignored | `final-release-tests.log` |
| Python manifest/resume suite | 14 passed | `python-resume-tests.log` |
| Actual GMRES CLI | 3 passed; 13.814 s | `gmres-runtime-tests.log` |
| Explicit ignored release suite | 6 passed, 0 failed; 38.80 s | `final-ignored-validation.log` |
| Release build / Clippy | Completed; Clippy emits existing/style warnings | `final-release-build.log`, `final-clippy.log` |

Shell syntax, formatting of new Rust modules and `git diff --check` also pass.

The Python resume tests use a mock solver to inject wrapper failures. The GMRES
runtime tests execute the actual binary against tiny ten-layer memberships and an
existing graph; their NONE policies exercise control flow and acceptance, not
nontrivial production convergence, and avoid constructing a full MDP.

## Bounded real-data results

The grouped comparison reads existing pruned membership, policy and next-layer
values, copying only whole current-layer blocks to temporary files. It compares
the original 512-block chunk boundaries with one shared lookup across those
chunks. Layer 2 contributes 361,207 states and layer 5 contributes 2,199,199 states:
**2,560,406 current-layer states**, with byte-identical greedy values/policies and
fixed-policy values. Setup is outside kernel timing; measured runs use ABBA order
after warmup, so they do not establish cold-cache behavior. Inputs are opened
read-only; source datasets and prior user files are preserved.

The final cache payload is 768 MiB per lookup, approximating the reference
16 GiB/20-worker setting. A prior 64 MiB experiment was intentionally interrupted
because repeated dense-field eviction measured unrepresentative cache thrashing.
It is not a completed performance result.

| Layer-5 kernel | Original misses → grouped | Materialized states, original → grouped |
|---|---:|---:|
| Greedy | 4,132 → 3,856 | 27,940,644 → 25,751,589 |
| Fixed policy | 2,831 → 2,657 | 23,953,882 → 22,198,602 |

Layer 2 counters are unchanged. Layer 5 eliminates some duplicate materialization,
but measured elapsed time is essentially unchanged at this scale; this is not
evidence of an end-to-end speedup. See `realdata-grouped-validation.log` and the
repeated comparison in `final-ignored-validation.log`.

The separate layer-8 RAM bitmap check processes **650,580 states**, pins the
83,839,296-byte next layer and uses a 512 MiB cache. All outputs match the file
reference byte-for-byte and the bitmap path creates ten bitmap blocks. With dense
promotion enabled (`dense_max=160`), sparse/bitmap measured seconds were
0.108326/0.102834 and 0.106923/0.104789; with promotion disabled they were
0.197380/0.109661 and 0.204739/0.141381. These are bounded kernel observations.

Bitmap construction still regresses cold fields: the synthetic RAM benchmark at
40,000 qids, three queries per field and a 16 MiB cache takes 6.979 ms sparse versus
16.768 ms bitmap across 48 fields. Warm lookup gains do not justify enabling it by
default. The same ignored suite measures allocating/reused child collection at
466.15/36.16 ms and scan/heap spill selection at 334.71/3.84 ms with 524,288 bytes
of heap metadata. These are synthetic component microbenchmarks, not solver timings.

## Reproduce

Run from `zxcl-pc`. The explicit ignored suite uses existing datasets and temporary
outputs; it does not rebuild the MDP or run policy iteration:

```bash
ZXCL_VALIDATION_GRAPH=/mnt/fast2/muse918/pcfinder/graph.bin \
ZXCL_VALIDATION_PRUNED=/mnt/fast1/muse918/pcfinder_layers/forward_pruned_4line_qopt_target_owned \
ZXCL_VALIDATION_VALUES=/mnt/fast1/muse918/pcfinder_layers/Vstar_fullV \
ZXCL_VALIDATION_POLICY=/mnt/fast1/muse918/pcfinder_qpreq/pi_policy_v5_f64 \
ZXCL_VALIDATION_REPETITIONS=1 \
timeout 360 cargo test --release --offline --bin zxcl -- \
  --ignored --test-threads=1 --nocapture \
  > work/optimization-validation/final-ignored-validation.log 2>&1
```

`ZXCL_VALIDATION_CACHE_MIB` is unset above (default 768); the grouped sample cap
defaults to 2,200,000 states per layer (layer 2 additionally caps at 512,000).
Use `ZXCL_VALIDATION_REPETITIONS=3` for more repetitions. Ordinary checks:

```bash
cargo test --offline
cargo test --release --offline
cargo build --release --offline --bin zxcl
cargo clippy --offline --all-targets
python3 -m unittest discover -s tests -p 'test_solve_resume.py' -v
ZXCL_RUNTIME_SOLVER="$PWD/target/release/zxcl" \
GRAPH_BIN=/mnt/fast2/muse918/pcfinder/graph.bin \
python3 -m unittest discover -s tests -p 'test_gmres_runtime.py' -v
```

## Limits

The production state space contains 86,329,867,487 states; one complete f32 snapshot
is about 321.6 GiB. No complete rebuild, full solve, full-runtime comparison or
peak-RSS validation was performed for these changes. Tests cover numerical
equivalence and bounded mechanisms, not completion at production scale.

The current field graph assumes exactly **15,185,706 fields** and u24 membership.
An SRS+/no-180 graph requires separate graph/domain/format adaptation and validation;
these results do not certify that variant. Hardware profiles and manual memory
reserves still need sizing for the intended GMRES iteration limit and live layers.
