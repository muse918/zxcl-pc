# Solver optimization validation

This records the initial sweep optimization pass and the later graph-to-index/V*
follow-up below. Boundary/DAG changes and their validation are documented in
[BOUNDARY_DAG.md](BOUNDARY_DAG.md) and [LOOP1_COMPILED.md](LOOP1_COMPILED.md).

Validation date: 2026-10-29. The changes preserve the existing numerical contract;
no acceptance tolerance was relaxed. Evidence below covers unit/reference checks,
wrapper failure injection, actual GMRES CLI checks, and bounded real-data kernels.
It does **not** establish full-MDP runtime or peak RSS improvements.

## Graph-to-index and V* solve follow-up (2026-10-29)

This stage targets `graph.bin` to MDP layer-index generation and GMRES within the
V* solve. The captured changes use compact temporary u24 membership, bulk writes,
index-only backward-layer loading, deduplicated GMRES source buckets and phase
timing logs. Historical `whole-003` measurements remain fixed-policy operator
evidence: their warm DAG E excludes production-wrapper work and is not a measured
V* solve time.

The captured solver SHA-256 is
`5364832638f850cd6771b5a8e9324b208e8bbc284cef8ba3f58bce7af44e7dd6`.
`work/limited-loop1/index-gmres-unit-002/` retains its binary/source identities,
source archive and limited-run summary. The completed checks are:

| Check | Result |
|---|---|
| Normal Rust suite | 207 passed |
| Separately selected ignored frontier check | 1 passed |
| Actual CLI regression cases | 24 passed; every selected case executed, none skipped |
| Source-backup regression suite | 16 passed; backup procedure is documented in [SOURCE_BACKUPS.md](SOURCE_BACKUPS.md) |
| Enclosing limited run | Return code 0; 206.996369500 s; monitored summed RSS peak 1,606,221,824 bytes; monitored new-write/scratch peak 808,178,446 bytes; no OOM events |

The earlier `index-gmres-unit-001` attempt had 195 Rust passes, three inotify
`EMFILE` failures and ten ignored tests. Parallel guard fixtures exhausted the
host's per-user 128-instance quota; one fixture used 93 watchers. The serial rerun
kept the RAII cleanup and input guards intact. The failed attempt is not counted
as a successful validation run.

The completed `work/limited-loop1/index-io-001/` replay used the original graph,
real F3 inputs and F4 outputs: **eight fields across four selected shards of the
production 4,096-shard partition**, producing **765,694 states**. Two ABBA cycles
gave four samples per format. Every output matched the canonical membership and
the alternate format byte-for-byte; input identities were unchanged.

| Selected F3-to-F4 replay | Original u64 pairs | Compact u24 |
|---|---:|---:|
| Median producer + temporary I/O + assembly, seconds | 0.261603 | 0.2420335 |
| Temporary payload written, bytes per state | 8 | 3 |
| Temporary write + read + final membership write, bytes per state | 19 | 9 |

The local timing ratio is **1.08085×**, and temporary payload is **62.5% smaller**.
This is a short, subsecond selected-field measurement with warm inputs and
buffered temporary I/O, without `fsync`. Graph/domain setup, partition planning,
input validation and canonical/cross-format comparisons are outside its timers.
The byte formulas cover membership payload, excluding headers, field indexes and
source-input reads; logical traffic is not physical device traffic. This remains
the initial short sample, rather than a complete index build.

The replay's terminal evidence is `index-io-001/evidence.json` and `summary.json`:
return code 0, **5.488551381 s** total, monitored summed RSS peak
**1,628,438,528 bytes** and monitored new-write/scratch peak **552,773,053 bytes**,
with no OOM events.

The broader `work/limited-loop1/index-io-002/` replay also completed with return
code 0. It used **64 real F3-to-F4 fields**, starting at field **12,942,851**:
16 fields per production shard owner **3491–3494**, producing **5,016,195 states**.
Two ABBA cycles again gave four samples per format, all matching canonical and
cross-format membership bytes exactly.

| Broader selected F3-to-F4 replay | Original u64 pairs | Compact u24 |
|---|---:|---:|
| Median producer + temporary I/O + assembly, seconds | 1.491783 | 1.4211485 |

The local timing ratio was **1.0497024×**, with the same temporary-payload byte
reduction and timing exclusions as the initial replay. It is a broader selected
replay, not a complete index build. `index-io-002/evidence.json` records terminal
PASS; total limited-run time was **15.28583805 s**, monitored summed RSS peaked at
**1,625,300,992 bytes** and monitored new-write/scratch at **612,996,543 bytes**,
with no OOM events.

These completed limited runs used a **32 GiB RAM budget, CPU affinity 0–3, a
32 GiB write budget and no time limit**. RSS and disk peaks are sampled
observations, not hard RSS/filesystem quotas.

A header-only audit counted **67,794,309,362 states** in affected F4–F7 layers.
**If those same memberships are rebuilt through this path**, the payload formula
changes from `19N = 1,288,091,877,878` to `9N = 610,148,784,258` logical bytes,
saving **677,943,093,620 bytes (about 631.38 GiB)**. This is conditional byte
accounting from existing headers, not measured full-build I/O or runtime.

The preproduction `work/limited-loop1/window-prefix-32-001/` comparison passed
**364 final-evidence checks**. It used only the **first 32 original roots**, ABBA
old/new/new/old order with two samples per arm, and **reachable-only validation**,
not all stored policies. Both affine and signed outputs were finite and matched
all 32 corresponding `whole-003` values bit-for-bit, with identical semantic and
cache-read counters.

| 32-root window comparison, median seconds | Original | Deduplicated |
|---|---:|---:|
| Whole probe | 17.7647 | 15.2814 |
| DAG construction B | 6.12153 | 5.18654 |
| Raw affine application | 3.91695 | 3.13180 |
| Raw signed application | 3.85390 | 3.20719 |

Warm DAG E stayed around **5 ms** in both modes. Both made **16,590,366 raw reads**;
deduplication retained 30,012 unique bucket records and suppressed 16,560,354
duplicates, reducing flushes from **8,108 to 23** and advised ranges from
**132,342 to 14,144**. The completed run took **66.620 s**, with monitored summed
RSS peak **5,167,513,600 bytes**, monitored new-write/scratch peak
**552,760,186 bytes** and no OOM events. Its `final-evidence.json` records the
audit. These prefix timings do not predict full production GMRES performance.

### GMRES warm-start precheck: full-boundary production acceptance

`work/limited-loop1/whole-gmres-001/` was **intentionally interrupted**, not
completed or timed out. It validated all **86,329,867,487 stored policy states in
238.98 s**, then was stopped with **197,078 DAG roots compiled**. Its summary
records interruption/130 after **897.627 s**, monitored summed RSS peak
**5,110,247,424 bytes**, monitored new-write/scratch peak **559,776,698 bytes** and
confirmed termination. `cancellation-reason.json` records the decision; this run
produced no completed GMRES performance result.

The reason is recorded in
[warmstart-analysis.json](../work/limited-loop1/warmstart-analysis.json). Comparing
the saved full `whole-003` affine result `T_pi(warm)` with its warm vector gives
**`||T_pi(warm) - warm||_inf = 0.00146484375`**, below the unchanged **0.003**
acceptance threshold. All 32 recorded input identities match the current inputs.
This is a derivation from historical solver output, **not a new solver
measurement**. It motivates checking an already acceptable warm start before
paying the policy-specific L9/DAG construction cost.

The implemented opt-in **`PI_GMRES_RAW_WARM_START=1`** defaults to off and requires
WHOLE mode with `gamma=1`. Its authoritative raw precheck runs before L9/DAG
preparation. Passing the same absolute infinity-norm gate (0.003 in this run)
returns `ACCEPTED_WARM_START`, zero Krylov iterations and `dag_built=0`, with
`relative_residual=not_computed`. This accepts a fixed-policy
warm start within the configured tolerance; it does not establish optimality.
Rejection pays an extra raw precheck cost, then continues through normal GMRES
and its original raw/DAG final bitwise gate.

This revision passed `work/limited-loop1/index-gmres-unit-003/` validation with
solver SHA-256
`2e54d2849ebe1bf1a0e95c6e0a81a05da22de423cab99af52a2d894d00e41a80`:

| Warm-precheck revision check | Result |
|---|---|
| Normal Rust suite | 209 passed: 9 auxiliary + 200 main |
| Separately selected ignored frontier check | 1 passed |
| Actual CLI regression cases | 29 passed: 17 WHOLE + 9 L8 + 3 legacy; every selected case executed, none skipped |
| Source-backup regression suite | 16 passed |
| Captured validation assets | Rust/Cargo and all six Python validation-asset hashes archived and unchanged before/after |
| Enclosing limited run | Return code 0; 300.541737914 s; monitored summed RSS peak 1,605,947,392 bytes; monitored new-write/scratch peak 829,097,258 bytes; no OOM events |

The run used the same 32 GiB/four-CPU/32 GiB/unlimited-time limits. These checks
validate the raw-precheck implementation and fixtures. The earlier index and
prefix performance measurements remain bound to the recorded **unit-002**
binary; they are not fresh measurements of this **unit-003** binary.

[GitHub CI for commit `05a62a5`](https://github.com/muse918/zxcl-pc/actions/runs/40863391716)
also passed. The workflow uses `cargo test --workspace -- --test-threads=1`,
matching the validated serial fixture mode that avoids the host inotify quota
failure; the guards were not relaxed.

Production `work/limited-loop1/whole-gmres-002/` **completed with return code 0**.
It validated all **86,329,867,487 stored policy states** and evaluated the
original raw affine operator for **all 817,740 roots** using the unit-003 binary.
RAW warm precheck and WHOLE were 1; L9 tail, compiled-L8/L9 and action memo were
0, with the reference memo setting at 4096 MiB and four workers under the same
resource limits. The supplied warm vector was accepted with
**`||T_pi(warm) - warm||_inf = 0.00146484375 <= 0.003`** and republished unchanged.
All output and applied values were finite. Their hashes matched the recorded warm
vector and historical full raw result exactly.

| Completed production phase | Seconds |
|---|---:|
| Full stored-policy validation | 226.465798 |
| Original raw numeric evaluation, including dense reset and terminal table | 1805.436230052 |
| Raw policy application reported within that evaluation | 1805.375333216 |
| Warm-start preflight total | 2031.954399547 |
| Limited-run controller total | 2037.115697734 |

The raw numeric evaluation took about **30.09 minutes**; total limited-run time
was about **33 minutes 57 seconds**. Startup includes graph/domain loading,
formula indexing and full stored-policy validation; the OS file cache is not
claimed cold. Status was **`ACCEPTED_WARM_START`**, with **zero Krylov iterations,
no DAG construction and relative residual `not_computed`**. No compiled-L9
shortcut or DAG bitwise gate executed on this accepted raw path. This is an
actual full-boundary fixed-policy warm-start acceptance, not iterative
convergence or an optimal-V* result.

The [final evidence](../work/limited-loop1/whole-gmres-002/final-evidence.json)
records monitored summed RSS peak **5,357,322,240 bytes** (about 4.989 GiB),
monitored new-write/scratch peak **573,662,192 bytes** (about 0.534 GiB), CPU
affinity 0–3 and zero OOM/OOM-kill events. The cgroup recorded **5,825,341
memory-max events** under file-cache charge/reclaim pressure; these were not OOM
events. All 32 recorded input stat/lstat identities stayed unchanged; large
payloads were not rehashed by this run. Final cleanup passed: the service was
inactive, its cgroup gone and the saved solver process identity gone.

The independent [same-work audit](../work/limited-loop1/whole-gmres-002/same-work-audit.json)
passed **201 checks**, including every applied f32 value, preserved warm values,
18 matching mathematical/cache/logical-read counters and captured identities.
For the same **16,401,897,372 source-touch records**, deduplication reduced flushes
from **8,016,142 to 67,262** and advised ranges from **169,884,035 to 58,154,702**.

For historical context, `whole-003` raw S was **3091.580989225 seconds** for the
same warm vector and policy, with all 32 input identities matching. These are
different historical runs, not a controlled A/B comparison. The completed
`whole-003` multi-case probe has a different workload from this CLI, so its total
runtime is not a previous GMRES CLI baseline. The demonstrated change here is
acceptance through the original raw gate before policy-specific DAG construction.

The wrapper inherits these opt-ins. Point `SOLVER_BIN` to the captured executable:
its existing helper only builds when the selected binary is missing, so an old
`target/release/zxcl` is not evidence for this revision. From `zxcl-pc`, this
example previews configuration without starting PI:

```bash
SOLVER_BIN="$PWD/work/limited-loop1/index-gmres-unit-003/solver.bin" \
  PI_POLICY_EVAL_BACKEND=gmres PI_LOOP1_WHOLE_DAG=1 \
  PI_GMRES_RAW_WARM_START=1 MDP_GAMMA=1 PI_ACCEPT_BELLMAN_INF=3e-3 \
  RAYON_NUM_THREADS=4 FULL_MDP_MEMORY_MAX_GIB=32 \
  ./scripts/solve_mdp.sh env
```

Each PI round starts a separate fixed-policy solver process and requires a new
policy-specific DAG if one is needed. Completion records bind the executable,
all `PI_*` settings and inputs. Greedy policy improvement and final layer emission
are not accelerated or validated by this run. Complete policy files (about
160.8 GiB) and final value layers (about 321.6 GiB) exceed the current 32 GiB
experimental write budget; this configuration preview is not a full-PI run.

Source backup instructions and verification are in
[SOURCE_BACKUPS.md](SOURCE_BACKUPS.md). Verified snapshots are kept under
`/home/muse918/pcfinder-source-backups/zxcl-pc/`, with `LATEST.json` and an active
15-minute timer. The validated source is also available on the
[source-backup branch](https://github.com/muse918/zxcl-pc/tree/codex/source-backup-20261029).

No complete V* solve, PI speedup, other kick-table variant or final ten-layer
materialization is established by this stage's completed tests, index replays,
prefix comparison and accepted production warm start.

## Greedy action reuse and V* convergence follow-up (2026-10-29)

This stage targets greedy policy improvement within V* iteration. It retains the
original **817,740 boundary coordinates** and uses no left/right reflection.
The earlier 33-minute-57-second result accepted an existing fixed-policy warm
start with zero Krylov iterations; it did not measure V* convergence.

`FULL_MDP_GREEDY_ACTION_CACHE_MIB` defaults to **0 (off)**. A positive setting
reserves a bounded greedy action memo from the existing task lookup-cache budget.
It reuses the ordered action average keyed by resolved target and pre-reveal ID
(`zid`), preserving missing probability mass, f32 reveal order and first-action
ties. Its scope is a greedy kernel invocation, not reuse across changing value vectors.

The captured binary in `work/limited-loop1/vstar-greedy-001/` has SHA-256
`2d1baa1843f26a04311aad8ff143678a106f0244aa9594328e676676cfc2dba0`.
The completed release checks passed **215 Rust tests** (9 auxiliary + 206 main),
with **10 ignored**, and **23 wrapper tests**. The tiny discounted ten-layer
value-iteration fixture does not exercise fixed-policy GMRES evaluation or
establish production PI convergence.

The L5 comparison selected original block range **2231368..2232336**, covering
**968 whole fields and 1,990,756 states**. It used the same validated binary with
the feature toggled, an equal **768 MiB total task cache budget**, and either no
action memo or **1 MiB** charged within that budget. Two ABBA cycles supplied
four measured samples per arm after warmup. This is a **sequential greedy
kernel**, with four-CPU affinity and an immutable existing L6 value snapshot;
the full next layer was not pinned or fully membership-validated by this probe.

The second comparison, `work/limited-loop1/vstar-greedy-l2-002/`, selected
**361,207 states across two whole L2 fields**, starting at original block 0.
It reused the same validated binary and 768 MiB/1 MiB settings, without another
build or test-suite run. Two ABBA cycles again supplied four samples per arm.
All retained f32 values were finite; value bits and u16 policy bytes matched.

| Bounded greedy kernel, median wall seconds | Memo off | 1 MiB action memo | Timing ratio | Time decrease |
|---|---:|---:|---:|---:|
| L5: 1,990,756 states | 1.400708552 | 1.244658535 | 1.125375767× | 11.14079% |
| L2: 361,207 states | 1.622773734 | 1.404232418 | 1.155630444× | 13.4671465% |

Median process CPU seconds were **1.400600203 → 1.244523933** for L5 and
**1.609633011 → 1.392310994** for L2. These are local kernel measurements.
Every greedy f32 value bit, u16 policy byte and mathematical counter matched.
In L5, each arm evaluated **6,757,636 actions**; the memo recorded **2,299,689
hits** and saved **10,714,263 reveal evaluations**. Next-layer materialization stayed at
**24,125,237 states**, and logical requested source bytes stayed at
**174,848,927**. Thus this sample shows saved action/reveal work, not reduced
source I/O; physical device traffic was not measured. The OS cache is not claimed
cold. Setup (**3.019766 s**), including placement-plan construction
(**0.222516 s**), and output comparisons are outside kernel timing; setup
amortization was not measured.

The [final evidence](../work/limited-loop1/vstar-greedy-001/final-evidence.json)
and [cleanup audit](../work/limited-loop1/vstar-greedy-001/cleanup-audit.json)
passed. All eight recorded source metadata identities stayed unchanged; the
source payloads were not fully hashed. The probe took **17.364 s** and the
bounded slab analysis **4.047 s**. The enclosing limited run took **291.530 s**,
including wrapper tests (**111.242 s**), Rust tests (**100.143 s**) and release
build (**57.017 s**); that total is not a greedy-kernel time. Monitored summed
RSS peaked at **1,577,222,144 bytes**, and monitored new-write/scratch at
**843,166,776 bytes**, under 32 GiB RAM, CPU affinity 0–3, 32 GiB writes and no
time limit. Memory-max, OOM and OOM-kill events were zero; the service was inactive
and its cgroup gone after completion.

The [L2 final evidence](../work/limited-loop1/vstar-greedy-l2-002/final-evidence.json)
also passed its root audit. Each arm evaluated **10,977,983 actions**; the memo
recorded **3,868,463 hits** and saved **13,567,112 reveal evaluations**. Next-layer
materialization stayed at **22,034,910 states** and logical requested source bytes
at **155,327,991**. Six input versions, the captured binary and the earlier
validation-source hashes stayed unchanged. The limited run took **18.410102194 s**,
with monitored summed RSS peak **1,745,272,832 bytes** and monitored write-root
peak **613,115,622 bytes**, including retained artifacts. It used the same limits
and recorded zero memory-max/OOM/OOM-kill events. Cleanup confirmed the inactive
service, gone cgroup and gone worker process. Reusing the validated binary kept
the two measured probe runs to about **36 seconds** combined; the earlier
build/test timings remain separate.

A later [0-versus-4 MiB L5 candidate](../work/limited-loop1/greedy-memo-001/final-evidence.json)
reused the same validated `vstar-greedy-001` binary, slab, 768 MiB budget and two
ABBA cycles, without rebuilding or rerunning tests. Its own paired medians were
**1.553689162 → 1.556934122 s**, about **0.21% slower**, with no meaningful gain.
The 4 MiB memo recorded **2,370,386 hits**, versus 2,299,689 in the earlier 1 MiB
run; those separate runs are not a controlled 1-versus-4 MiB comparison. Probe
time was **19.116306 s** and limited-run time **19.923725 s**, with monitored
summed RSS peak **1,586,917,376 bytes**, write-root peak **838,940,852 bytes**,
zero memory events and cleanup PASS. The capacity increase was rejected, and
planned 16 MiB/L2 capacity follow-ups were dropped.

The **L5 slab's** storage analysis compares **11,944,536 raw value/policy
bytes** with **1,198,936 bytes** for independently compressed fields, including
an assumed **31,040-byte index/header**. This is a hypothetical format estimate:
no corresponding production format or decoder is implemented, and no full-layer
compression ratio is inferred. Every sampled new value matched the old candidate
snapshot bit-for-bit; **1,340 of 1,990,756 policy codes (0.06731%)** differed.
Old witness consistency and action advantages were not verified, so these
differences do not establish a convergence result.

The full V* goal remains active and its ETA is unknown. This bounded greedy
measurement does not establish a complete sweep or PI speedup. Complete raw
policies require about **160.8 GiB**, and adjacent temporary value storage can
exceed **80 GiB**, above the current 32 GiB experimental write budget. Production
storage integration and further access-pattern evidence are needed before a full PI run.

The proposed production integration, **not implemented**, uses `CompactGroupSink` to append
existing per-field value/policy buffers in deterministic group order into compact
blobs. Values would use the `DenseF32ValueLookup` field-read hook, and policies
bounded per-worker decoded caches, with RAW fallback or ZLIB1 and strict
size/consumption checks. The design uses bounded field/group buffers without a
whole-layer arena. The standalone probe below supplies initial separated L3–L7
encode/decode CPU and size measurements. Production integration and a 32 GiB
full-run storage/time bound remain unverified.

## Compact field codec probe (2026-10-29)

`compact-field-001` implemented and measured a standalone RAW/ZLIB1 field codec;
it has **not been adopted by production greedy/GMRES**. VALUE storage fell by a
factor of **6.84**, but the measured warm sequential read/decode cost rose by
**16.84×**. This establishes a storage/read-cost tradeoff, not a solve-time gain.

The CLI is `compact-field-probe <pruned> <values> <policies> <work_parent>
[max_span_states=1000000] [max_span_fields=512] [ABBA_cycles=2]
[minimum_sample_ms=25]`. It reads selected field metadata and f32/u16 payloads;
it builds no graph/queue domain or complete MDP index. This run selected three
separated whole-field spans near header ordinals 1/8, 1/2 and 7/8 in each of
L3–L7: **15 spans, 14,573,805 states and 3,520 fields**, with **87,566,230 original
source bytes read by `pread`**. Exact span bounds are retained in the
[probe summary](../work/limited-loop1/compact-field-001/compact-field-probe-1724455-1793277716819287101-0/summary.json).

| Layer | Sampled states / fields | VALUE bytes, raw → compact | POLICY bytes, raw → compact |
|---|---:|---:|---:|
| L3 | 2,823,472 / 49 | 11,293,888 → 1,759,556 | 5,646,944 → 1,268,457 |
| L4 | 2,988,030 / 1,222 | 11,952,120 → 1,020,221 | 5,976,060 → 975,799 |
| L5 | 2,873,494 / 1,098 | 11,493,976 → 1,253,918 | 5,746,988 → 807,437 |
| L6 | 2,994,913 / 857 | 11,979,652 → 1,889,453 | 5,989,826 → 463,376 |
| L7 | 2,893,896 / 294 | 11,575,584 → 2,599,054 | 5,787,792 → 317,028 |

Compact sizes include headers/indexes; raw sizes are flat payloads with selected
metadata already resident. Across these samples, VALUE ratios range from
**4.45–11.72×** (aggregate **58,295,220 → 8,522,202 bytes**, 6.84×), and POLICY
from **4.45–18.26×** (**29,147,610 → 3,832,097 bytes**, 7.61×).

Two warm ABBA cycles gave four samples per arm, with retained reader handles and
normalized repeated passes. Summing per-layer median sequential passes gives
VALUE **0.019876070 → 0.334760300 s** and POLICY **0.009269854 → 0.234653087 s**;
compact read/decode was 16.84× and 25.31× slower respectively. Encode/write was
also slower (2.47× VALUE, 1.76× POLICY), including flush/`sync_data` and compact
atomic publication. Reader-open/index-load time was measured separately.

These timings include compact decoder checksums and its own source-guard checks;
the raw baseline has no per-field checksum. Source copying/setup and restored-byte
comparisons are outside the timed phases. Restored bytes matched exactly,
compact checksums passed, and all 15 original-source metadata identities stayed
unchanged; source payloads were not fully hashed. Shuffled whole-field reads do
not measure the policy random-code hot path. Physical device traffic, cold-cache
performance, a complete storage footprint and V* convergence remain unmeasured.

The captured solver SHA-256 is
`2f44fc207b63e6abb23f4b10a1c1ea4c02dd9221dc3c94172f38369fda5326c1`.
`rust-tests.log` records **223 passes** (9 auxiliary + 214 main), with **10
ignored**. Rust tests took **104.886 s**, release build **62.445 s**, and the
probe **54.939 s**. The enclosing limited run completed in **223.504 s**, with
monitored summed RSS peak **928,026,624 bytes** and monitored write-root peak
**882,546,517 bytes** across the retained global work root. Limits were 32 GiB
RAM, CPU affinity 0–3, 32 GiB writes and no time limit; all memory events were
zero. The [final evidence](../work/limited-loop1/compact-field-001/final-evidence.json)
and [cleanup audit](../work/limited-loop1/compact-field-001/cleanup-audit.json)
passed. No production full-PI or V* timing improvement is claimed.

For the captured binary and existing read-only inputs, use a fresh run directory:

```bash
python3 scripts/run_limited_experiment.py \
  --mem-gib 32 --cpus 4 --disk-gib 32 --seconds 0 \
  --output-dir /path/new-compact-run -- \
  "$PWD/work/limited-loop1/compact-field-001/solver.bin" compact-field-probe \
  /path/pruned /path/values /path/policies /path/new-compact-run 1000000 512 2 25
```

## Initial sweep validation matrix

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

## Initial sweep test results and logs

The initial-pass paths below are relative to the repository root. Its logs are
local evidence under `work/optimization-validation/`; older debug/benchmark logs
include failures and interrupted experiments superseded by the named final logs.

| Check | Completed initial-pass result | Evidence |
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

## Initial sweep limits and remaining full-solve scope

The production state space contains 86,329,867,487 states; one complete f32 snapshot
is about 321.6 GiB. The initial sweep pass performed no complete rebuild, full
solve, full-runtime comparison or peak-RSS validation; its tests covered numerical
equivalence and bounded mechanisms. The later full-boundary fixed-policy raw
acceptance and its observed resource peaks are documented above. A complete index
rebuild, iterative GMRES convergence, complete PI and final-layer runtime
comparison remain unverified by these results.

The current field graph assumes exactly **15,185,706 fields** and u24 membership.
An SRS+/no-180 graph requires separate graph/domain/format adaptation and validation;
these results do not certify that variant. Hardware profiles and manual memory
reserves still need sizing for the intended GMRES iteration limit and live layers.
