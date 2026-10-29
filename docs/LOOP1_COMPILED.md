# Shared fixed-policy loop1 expressions

This path targets the cost of one boundary-to-boundary PC application (`loop1`).
It keeps the existing 817,740-coordinate solver boundary and uses no left/right
reflection. Further boundary condensation remains a deferred deployment idea in
[BOUNDARY_DAG.md](BOUNDARY_DAG.md#priorities-and-deferred-deployment-ideas).

The complete L9 layer is streamed into a flat state-to-expression ID array.
L8 can then share equal ordered averaging expressions. Production opt-ins reuse
L9, compile complete L8, or compile the original whole-boundary operator for a
fixed policy. The measurements below include completed full-L9/L8 and original
whole-boundary fixed-policy compilation, bitwise comparisons and repeated
evaluation costs, alongside earlier bounded probes. End-to-end full-solve and
policy-iteration performance remain unproven.

## Numerical and reuse contract

At `gamma=1`, a fixed L9 policy selects either zero or a terminal formula. A
terminal formula retains the original computation order:

```text
sum(reward + reset[child_qid], in original reveal order) / reveal_count
```

`reward` is one for affine applications and zero for signed linear Krylov
applications. The implementation does not move reward addition outside the
average. Equal numerical values alone do not merge expressions. L8 keys retain
the denominator, child order and multiplicity; missing pruned members contribute
zero with the original denominator. The stored policy action and resolved target
field determine the children.

The 205,170 pre-reveal groups identify reusable terminal formulas here. They do
not replace the solver boundary or change its arithmetic. Every L9 state has ID
zero or `terminal_id + 1`, so the 20,959,824-state production L9 layer needs
83,839,296 bytes for its flat `u32` mapping, without a per-state hash-map entry.
The shared pre-reveal index has approximately 16,363,624 bytes of logical payload;
the terminal plan has 4,091,700 bytes. Metadata and allocator overhead are separate.

L9 construction validates its membership and policy once. Input identity/version checks
bind the reusable plan to its source files and run before applications; changing
a source requires rebuilding. The plan is reusable across GMRES applications of
the same fixed policy, including changing and signed boundary vectors. Policy
improvement requires a new plan. Plans are process-local; there is no persistent
compiled cache.

## Whole-boundary solver opt-in

`PI_LOOP1_WHOLE_DAG=1` selects the whole fixed-policy DAG in
`pi-policy-eval-gmres` and `apply-fixed-policy-boundary`. It defaults to off,
requires Linux and `gamma=1`, and rejects enabled compiled-L8 or nonzero action-memo
settings before input loading. It validates all stored policies and preserves
every stored L0 root in its original order; terminal resets use the physical-empty
first-block prefix. The plan is reused across signed/affine applications of the
same policy.

| Whole-operator setting | Default |
|---|---:|
| `PI_LOOP1_WHOLE_MEMO_MIB` | 4096, maximum 4096 |
| `PI_LOOP1_WHOLE_REFERENCE_MEMO_MIB` | 4096, maximum 8192 |
| `PI_LOOP1_WHOLE_MAX_NODES` | 5,000,000 |
| `PI_LOOP1_WHOLE_MAX_EDGES` | 25,000,000 |
| `PI_LOOP1_WHOLE_L9_TAIL` | 1 |

Standalone publication and GMRES final verification recompute every root through
the raw numeric policy path, using the original terminal-table builder, fresh
bounded caches and no DAG or compiled L9. They require finite bitwise equality;
GMRES also retains the actual Bellman residual gate. Input versions are checked
again after the pending output is flushed and fsynced, before atomic publication.
Whole-mode GMRES clamps `PI_GMRES_BASIS_MAX_GIB` to at most **16 GiB**.

For a prebuilt solver and existing inputs, use a fresh output directory; keep
both the new value and scratch below it:

```bash
python3 scripts/run_limited_experiment.py \
  --mem-gib 32 --cpus 4 --disk-gib 32 --seconds 0 \
  --output-dir /path/new-whole-run -- \
  env PI_LOOP1_WHOLE_DAG=1 PI_LOOP1_COMPILED_L8=0 \
    PI_LOOP1_ACTION_CACHE_MIB=0 PI_LOOP1_WHOLE_L9_TAIL=1 \
    PI_LOOP1_WHOLE_MEMO_MIB=4096 PI_LOOP1_WHOLE_REFERENCE_MEMO_MIB=4096 \
    PI_GMRES_BASIS_MAX_GIB=16 FULL_MDP_MEMORY_MAX_GIB=32 RAYON_NUM_THREADS=4 \
    /path/zxcl pi-policy-eval-gmres \
    /path/graph.bin /path/pruned /path/policy /path/current-l0.f32 \
    /path/new-whole-run/value.f32 /path/new-whole-run/scratch \
    1.0 200 1e-6 1
```

The command interface has passed fixture validation, including one actual Krylov
iteration. The all-root probe also passed both full 817,740-root comparisons;
see the completed all-root results below. A complete optimal-MDP solve, other
kicktable variants and final ten-layer value tables are not established by
these checks.

## L9/L8 sweep opt-ins

`PI_LOOP1_COMPILED_L9=1` enables prepared L9 evaluation in
`pi-policy-eval-gmres` and `apply-fixed-policy-boundary`. It defaults to off and
requires `gamma=1`.
For example, from `zxcl-pc`, using existing inputs and new output/scratch paths:

```bash
PI_LOOP1_COMPILED_L9=1 target/release/zxcl pi-policy-eval-gmres \
  /path/graph.bin /path/pruned /path/policy \
  /path/current-l0.f32 /path/new-l0.f32 /path/new-scratch \
  1.0 200 1e-6 16
```

| Variable | Purpose |
|---|---|
| `PI_LOOP1_L9_MAX_MAPPING_MIB` | Flat ID mapping construction cap; default 512 MiB |
| `PI_LOOP1_L9_RANK_MAX_MIB` | Retained L9 membership rank-index cap; default 64 MiB |
| `PI_LOOP1_L9_BUILD_CACHE_MIB` | Temporary membership buffer/cache cap during construction; default 32 MiB |

The plan is built once for that fixed-policy evaluation and retained across
operator applications. With only compiled L9 enabled and the RAM plan permitting,
each application gathers a
complete L9 `f32` array from the compiled IDs and current terminal values. L8 then
uses its original RAM lookup kernel. This retains numeric L9 materialization
while removing repeated L9 policy decoding and membership validation.

A shared, persistent qid membership/rank index also maps an L8 lookup directly to
its L9 state ID. Its construction happens once per policy; workers and repeated
GMRES applications share it. This direct lookup supplies the fallback when a
pinned L9 value array does not fit the RAM plan, without repeated membership-file
searches or L9 value materialization. The generic compiled-L9 lookup can still
use the existing per-task membership cache when no rank index is supplied; the
production opt-in builds the rank index. The other layers retain their existing
sweep kernels.

RAM gathering requires `FULL_MDP_RAM_NEXT_LAYER` (enabled by default), no
`FULL_MDP_MMAP_SPILL`, and the L9 numeric array fitting both
`FULL_MDP_RAM_NEXT_LAYER_MAX_GIB` (default 96 GiB) and
`FULL_MDP_RAM_PAIR_MAX_GIB` (default 88 GiB), together with the scheduler's
combined resident-memory check. Otherwise the compiled rank-direct path is used.
Layer logs distinguish `sink=compiled-ram-gather value_bytes=83839296` from
`sink=compiled-rank-direct value_bytes=0` on the production dataset.

The GMRES final publication check uses the original legacy affine sweep independently
of the prepared L9 path, materializes L9 even if `FULL_MDP_L9_DIRECT=1`, and retains
the existing Bellman residual threshold. Compiled resident payload and the
gathered numeric L9 array are charged to the sweep scheduler's memory plan;
retained compiled payload remains charged during the final legacy check.
Accounting is a logical allocation estimate, not a hard process RSS limit.

`PI_LOOP1_COMPILED_L8=1` additionally enables the complete-L8 compiler in both
commands and automatically enables compiled L9. It is also off by default and
requires `gamma=1`. Construction covers all L8 fields, retaining state IDs and
shared ordered averages. Each application evaluates the shared averages once and
gathers the ordinary numeric L8 vector into the existing RAM, mmap or file sink
for the original L7 kernel. In this mode L9 supplies terminal expressions without
materializing a numeric L9 layer. The GMRES final original affine verification bypasses
both compiled layers and the action memo described below.

| Complete-L8 variable | Purpose |
|---|---|
| `PI_LOOP1_L8_MAX_MAPPING_MIB` | Combined L9/L8 ID-mapping cap; default 4096 MiB |
| `PI_LOOP1_L8_MAX_UNIQUE_AVERAGES` | Maximum shared L8 averages; default 32,000,000 |
| `PI_LOOP1_L8_BUILD_CACHE_MIB` | Construction cache cap; default 32 MiB |
| `PI_LOOP1_L8_BUILD_MAX_MIB` | Compiler heap-cap request; default 16,384 MiB, further limited by the sweep memory plan |

This integration makes the compiled methods usable during a full fixed-policy solve.
A complete new 86,329,867,487-state solve has not been run, and a full-solve speedup
is not established by the two-layer measurements below. Complete L9/L8 is now
compiled and measured on the production dataset; the remaining layers are
outside that measurement. A complete L7 compiler remains separate work.

## Complete L9/L8 measurement

`work/limited-loop1/full-l8-001/` completed successfully with all **1,047,622,806
L8 states** and **20,959,824 L9 states**. Complete L8 shares **63,168 ordered
averages** under the stored fixed policy. This is a complete two-layer result,
separate from the earlier 27-field L8 slab and from a complete boundary-to-boundary
loop1 application.

The run used four workers for both paths and the original production fixed-policy
kernels as the reference. After warmup, three ABBA repetitions produced six
samples per path in each case. The figures below are **arithmetic means**, unlike
the medians in the earlier slab probes. The comparison reused two preallocated
L8 output buffers, one per path. Allocating and zeroing these buffers and checking
all output values for finiteness and bitwise equality are outside the application
timers. There were no source or scratch-value writes.

| Complete L9/L8 case | Original seconds | Compiled seconds | Tail speedup | B break-even applications |
|---|---:|---:|---:|---:|
| Actual L0, affine reward | 45.972297037 | 0.674248079 | 68.183059671× | 8 |
| Independent signed input, linear reward | 47.128337318 | 0.821526418 | 57.366794634× | 7 |

Every full L8 output was finite and bit-identical to the freshly computed
original result during warmup and after every paired sample in both cases. The
original path computes complete numeric L9 before L8; the compiled path evaluates
the shared terminal/average expressions and gathers complete numeric L8. Most of
the compiled application time is gathering: 0.635189211 seconds affine and
0.777230519 seconds linear, versus 0.000948334 and 0.001235913 seconds for unique
average evaluation.

Total policy-scoped construction **B was 323.204851679 seconds**, including
0.154151142 seconds for the pre-reveal index, 0.060326726 for the terminal plan,
1.705007741 for L9 and 321.285365541 for complete L8. Common setup was reported
separately at 2.349901375 seconds. The stated break-even counts compare
`B + k E` with `k S` for this fixed policy and this complete L9/L8 numeric phase;
policy changes require rebuilding. They do not establish a GMRES or whole-loop
speedup, and the original final verification remains necessary.

| Capacity measurement | Bytes |
|---|---:|
| Combined L9/L8 state-ID mappings | 4,274,330,520 |
| Retained capacity including L9/L8, pre-reveal index and terminal plan | 4,314,882,868 |
| Compiler peak capacity estimate | 4,334,594,036 |

These are allocation/capacity measurements, not process RSS. The external
cgroup memory-charge limit was **32 GiB**, CPUs **0–3**, a monitored **32 GiB disk allowance with
a 31 GiB stop threshold**, and no time limit. The job completed in **1075.518773254
seconds**, with peak monitored experiment write usage of **472,360,030 bytes**,
and no memory-max, OOM or OOM-kill events. One live sample recorded
`memory.current=19,002,601,472` bytes (about 17.7 GiB), zero swap and affinity to
CPUs 0–3; that sample is **not a measured memory peak**.
The cgroup limit and that single sample do not certify a 32 GiB total-RSS ceiling;
the later [whole-loop attempt](#stopped-all-root-attempt-and-rss-enforcement)
exposed the distinction.

The saved Vstar snapshots were checked only for lineage. With the supplied
actual L0 input, freshly computed original f32 values differed in **5,853,951 L9
states** and **252,713,065 L8 states**, with no nonfinite values. The snapshots
therefore are not an exact reference for this calculation. Those differences
are not a compiled-method validation failure: the bitwise acceptance reference
was the fresh original kernel output using the same inputs and policy.

`output.log`, `summary.json`, `identity.json` and `live-resources.json` retain the
results and provenance. The run used a copied solver and recorded binary/source
hashes plus input/settings identities. These results apply to that captured
binary and policy. Complete loop1 evaluation across all layers, all original
boundary roots and a full solve remain unproven; subsequent sparse/DAG work needs
its own build, tests and actual execution.

## Historical bounded L9/L8 probe

The probe compiles complete L9 and a contiguous range of whole L8 fields, bounded
by a state cap. It compares both affine and signed linear results with the
existing production kernels, using the supplied boundary vector and two
independent signed vectors. All complete-L9 and selected-L8 outputs must be
finite and bit-identical as `f32`.

```bash
cargo build --release --offline --bin zxcl
ZXCL_LOOP1_MATERIALIZED=1 target/release/zxcl loop1-probe \
  /mnt/fast2/muse918/pcfinder/graph.bin \
  /mnt/fast1/muse918/pcfinder_layers/forward_pruned_4line_qopt_target_owned \
  /mnt/fast1/muse918/pcfinder_qpreq/pi_policy_v5_f64 \
  /mnt/fast1/muse918/pcfinder_layers/Vstar_fullV/layer0.values.f32.bin \
  2000000 3
```

The L8 cap defaults to 2,000,000 states, may be set up to 4,000,000, and never splits
a field. Repetition count defaults to five and must be at least three. Probe
copies live in its owned temporary directory; production membership, policy and
value inputs are read-only.

`ZXCL_LOOP1_MATERIALIZED=1` adds the current default materialized-L9 baseline to
the direct-L9 baseline. All measured paths run sequentially; the legacy paths use
the existing production kernels. Warmup and paired sampling are included. These
timings do not represent the full production worker schedule.

The printed costs are:

- **B:** pre-reveal-index, terminal-plan, complete-L9 ID/rank-index and bounded-L8
  construction, including source validation and mapping construction.
- **B L9 only:** the same construction through L9, excluding L8 expression
  compilation. This is the setup cost used for the production-path comparison.
- **P:** the RAM production L9-only route on the same L8 slab: source-version
  checks, full terminal generation and table gathering/finite validation,
  numeric L9 gathering from compiled IDs, then the original L8 RAM lookup
  kernel. It does not use L8 expression sharing.
- **R:** the rank-based direct fallback. The same original L8 policy kernel
  queries compiled L9 IDs through the retained membership rank index without
  gathering numeric L9. It is checked for bitwise correctness separately.
- **E:** version checks, selected terminal generation, shared L8 expression
  evaluation and output gathering.
- **S materialized:** the legacy terminal table, full L9 materialization and
  the same L8 slab evaluated through RAM lookup.
- **S direct:** the legacy terminal table, full L9 membership validation and
  policy traversal, then the same L8 slab with direct L9 lookup.

`prep_seconds` separately reports graph/quotient/qforward construction,
source-index loading, reset-vector creation and the owned L8 slab copy. It
explicitly excludes pre-reveal-index construction, which is charged to B.

Compare `B + k E` with `k S` for the L8-sharing prototype, and
`B_L9_only + k P` with `k S` for the production L9-only route. Break-even is
`ceil(B / (S - application_cost))` only when the application is cheaper than S.
It applies to this fixed policy and slab, after the separately reported common
probe setup. P and materialized S use the same original L8 policy kernel, with
their own PSSP samples rather than borrowing the baseline from E's ABBA samples.

## Historical 27-field L8 measurements

The validation dataset has 20,959,824 L9 states and the selected L8 slab contains
1,949,970 states in 27 fields. It produces 51,510 unique L8 averaging expressions
with 183,702 retained child edges. L9 exposes 205,170 terminal expressions plus
zero. Its 15,339,200-byte membership rank index is retained separately from the
83,839,296-byte flat state-ID mapping and included in the printed table and
construction totals. Complete L9 retains 100,000,776 bytes of compiled tables;
L9 plus the L8 slab retains 110,184,216 bytes. The 4,091,700-byte terminal plan
brings the latter total to 114,275,916 bytes, excluding common source-domain data
such as the pre-reveal index and the numeric L9 gather buffer.

The completed run is `work/loop1-validation/production-probe-gather.log`:

| Construction/setup measurement | Seconds |
|---|---:|
| Common probe setup, excluding pre-reveal index | 2.715991928 |
| Pre-reveal index, included in both B values | 0.171703740 |
| Terminal plan, included in both B values | 0.066898541 |
| Membership rank index, included in both B values | 0.150981746 |
| B L9 only | 2.208858686 |
| B including the L8 prototype | 2.938209531 |

Construction scanned 177,739,026 source bytes: 131,609,454 membership bytes,
45,819,588 policy bytes and 309,984 index bytes. The rank-index construction
buffer peaked at 1,039,728 logical bytes under its 32 MiB cap.

The production-path comparison below uses three PSSP repetitions, six samples per
path per case, and median seconds. P and S both run the original L8 RAM lookup
kernel. The last column uses B L9 only.

| Boundary vector | Reward | P | S materialized | P L8 kernel | S L8 kernel | Break-even applications |
|---|---:|---:|---:|---:|---:|---:|
| Actual L0 | 0 | 0.476283986 | 0.915604789 | 0.339821891 | 0.336407274 | 6 |
| Actual L0 | 1 | 0.460212312 | 0.908193417 | 0.333212162 | 0.329446020 | 5 |
| Independent signed | 0 | 0.461181493 | 0.915164954 | 0.333670042 | 0.333043830 | 5 |
| Independent signed | 1 | 0.470899190 | 0.922669636 | 0.340006909 | 0.335716375 | 5 |
| Independent signed Krylov | 0 | 0.463753423 | 0.915623513 | 0.334382374 | 0.335811651 | 5 |
| Independent signed Krylov | 1 | 0.464550724 | 0.929472319 | 0.335185177 | 0.336669534 | 5 |

P took **0.460212312–0.476283986 seconds**, compared with its paired S at
**0.908193417–0.929472319 seconds**, and amortized B L9 only after **5–6
applications**. That approximately twofold result covers **complete L9 plus the
27-field L8 slab**. It is not a measured speedup for complete L8, one complete
loop1, or a full solve. P leaves the L8 kernel unchanged; the full L8 layer has
1,047,622,806 states, compared with 1,949,970 in this slab.

The separate L8-sharing prototype uses three ABBA repetitions and six samples per
path per case. Each E column has its own paired baseline; values are median
seconds.

| Boundary vector | Reward | E with materialized comparison | S materialized | E with direct comparison | S direct |
|---|---:|---:|---:|---:|---:|
| Actual L0 | 0 | 0.011085018 | 0.920486470 | 0.015320393 | 1.955116162 |
| Actual L0 | 1 | 0.011098273 | 0.913728687 | 0.011007347 | 1.911057779 |
| Independent signed | 0 | 0.010682009 | 0.899473200 | 0.010967814 | 1.908952121 |
| Independent signed | 1 | 0.010954263 | 0.918074484 | 0.011132372 | 1.920395132 |
| Independent signed Krylov | 0 | 0.010690479 | 0.916232717 | 0.011111659 | 1.906641788 |
| Independent signed Krylov | 1 | 0.010730711 | 0.919984205 | 0.010955934 | 1.896829350 |

E took **0.010682009–0.015320393 seconds**; its paired S materialized took
**0.899473200–0.920486470 seconds**, and S direct took
**1.896829350–1.955116162 seconds**. B amortized after **four applications**
against materialization or **two** against direct L9, in all six cases. This
prototype gains from L8 expression sharing that the production opt-in does not
use when only `PI_LOOP1_COMPILED_L9=1` is enabled. The complete-L8 opt-in uses that
representation across the whole layer, whose cost and compression cannot be
extrapolated from this slab.

All six cases were finite and bit-identical for complete L9, the bounded L8 slab,
the production numeric gather and the rank-direct fallback. Source versions were
unchanged. Process-lifetime `VmHWM` was **1,559,486,464 bytes**.

Logical construction payload, cache and interner accounting excludes
allocator/hash capacity overhead; process-lifetime `VmHWM` also includes common
loading and correctness-validation buffers and is not an isolated compiler peak.

The earlier `real-probe.log` measures the L8-sharing prototype before the retained
rank index and before pre-reveal-index construction was charged to B. Its timing
and amortization figures are historical and are not the current production-path
measurement. A first production lookup using repeated binary membership searches
was slower than materialization on the measured slab. Persistent rank indexing
improved the complete measured L9/L8 tail but still slowed the L8 leaf kernel
relative to its RAM lookup. Since full L8 is much larger than the probe slab, the
RAM production path therefore gathers numeric L9 and retains the original L8 RAM
lookup. A tail-only timing gain does not establish a complete-layer gain.

## Bounded L5 action-cache experiment

`PI_LOOP1_ACTION_CACHE_MIB` requests a task-local memo of exact ordered
fixed-policy action averages; it defaults to zero (disabled). Its key uses the
resolved target field and validated placement/reveal identity. Cached values
live only for the current kernel invocation and next-value layer, preserving
child order, missing-member zero terms, the full denominator and signed values.
The memo takes its payload from the existing per-task lookup-cache allowance;
enabling it does not add an equal-sized extra lookup budget.

The real-data L5 experiment used **1,990,756 states in 968 contiguous whole
fields**, field-index range `2231368..2232336`. It ran the production fixed-policy
kernel sequentially against the existing file-backed L6 snapshot, without pinning
the complete next layer. This is a different scope from the 27-field L8 probe
above. In the first experiment both arms had a **768 MiB total cache allowance**;
the memo arm reserved **16 MiB** of that allowance for action reuse. Both used
identical field groups.
After warmup, three ABBA repetitions produced six samples per arm.

| First L5 experiment | Baseline | 16 MiB action memo |
|---|---:|---:|
| Median wall seconds | 0.634405438 | 0.658862805 |
| Median process CPU seconds | 0.634365001 | 0.658789736 |
| Action-cache hits | 0 | 620,572 (31.1726801%) |
| Next-layer blocks loaded | 2,521 | 2,521 |
| Logical requested source bytes | 156,778,115 | 156,778,115 |

The memo was **3.86% slower** despite its hit rate and 2,797,655 saved reveal
lookups. Both arms requested the same 9,953,780 current-membership/policy bytes
and 146,824,335 next-membership/value bytes: the existing lookup cache already
retained the required blocks. Physical disk I/O was not measured. All actual
snapshot outputs were bit-identical. A separate independent scalar fixture used
48 signed states, 214 retained next states and deliberately absent members; both
arms matched bitwise at `gamma=1` and `gamma=0.625`.

Logs and run metadata are in
`work/limited-loop1/action-l5-001/{output.log,summary.json,run-config.json}`.
The run completed with unchanged sources and no source-file writes. Setup took
3.445688006 seconds, including 0.223892987 seconds for placement validation, and
is excluded from the application timings. Setup amortization and a full-loop
speedup were not measured. With the slower application in this sample, no
positive setup cost can be amortized against the baseline. This result leaves
the action memo disabled by default; a cache hit alone is not evidence of a
useful optimization.

The **1 MiB** follow-up, recorded in
`work/limited-loop1/sparse-ab-and-action1/output.log`, used the same 968 fields,
1,990,756 states and equal 768 MiB total allowance. Warmup and three ABBA
repetitions again produced six measured samples per arm:

| Follow-up L5 measurement | Baseline | 1 MiB action memo |
|---|---:|---:|
| Median wall seconds | 0.627495153 | 0.585602958 |
| Median process CPU seconds | 0.627255968 | 0.585369803 |
| Action-cache hits | 0 | 542,722 (27.2621055%) |
| Next-layer blocks loaded | 2,521 | 2,521 |
| Logical requested source bytes | 156,778,115 | 156,778,115 |

This paired follow-up reduced application time by **6.676%**, a measured
**1.071536859×** speedup on that L5 slab. It saved 2,497,630 reveal lookups but
requested the same 146,824,335 next-membership/value bytes and 9,953,780 current
membership/policy bytes. All actual snapshot outputs and the separate signed
scalar fixtures at `gamma=1`/`0.625` remained bit-identical. Setup took
2.761245995 seconds, including 0.215950464 seconds of placement validation, and
is excluded from application timing; setup amortization was not measured.

The 1 MiB and 16 MiB results are separate paired experiments. They show that the
larger hit rate did not produce the faster application in these samples; they
do not establish an optimal cache size or a whole-loop benefit. The action memo
remains disabled by default until broader validation.

These bounded runs used an aggregate cgroup-v2 **32 GiB memory-charge limit**, CPU
affinity to logical CPUs **0–3**, and no elapsed-time limit. The CPU quota
controller was unavailable, so CPU enforcement was affinity. Its experiment
write root had a monitored **32 GiB disk allowance** with a **31 GiB stop
threshold**, rather than a filesystem quota. Peak monitored usage was 332,174,805
bytes for the 16 MiB run and 479,515,744 bytes for the combined uncached-sparse/
1 MiB memo job. Work toward a complete loop1 application remains active and
unproven under these limits. Neither this L5 result nor the earlier L9/L8 tail
measurements establish its eventual speed or successful completion.

## Bounded sparse numeric probe

`loop1-sparse-probe` is an experimental standalone numeric evaluator. It follows
the fixed policy through the ten layers and keeps bounded state/action memos,
recomputing entries after collisions rather than retaining every state. Numeric
memos are fresh for each application; this is separate from the reusable compiled
L9/L8 expression representation and leaves boundary coordinates unchanged.

Two actual probes evaluated only the first **32 boundary roots**, in affine mode
at `gamma=1` with four workers. Each produced a **128-byte output**. One requested
a 256 MiB memo cap, retaining 201,326,592 bytes of memo payload plus 131,072 bytes
of field-validation tags; the other disabled numeric memoization.

| Single-run diagnostic | 256 MiB cap | No memo |
|---|---:|---:|
| State requests | 867,775 | 1,567,392 |
| Evaluated states | 862,964 | 1,567,392 |
| State-cache hits | 4,811 | 0 |
| Action-cache hits | 75,354 | 0 |
| Reported elapsed seconds | 1.085231578 | 0.676369195 |
| Separately reported common setup seconds | 3.274946 | 3.261005 |

The cached and uncached outputs passed `cmp`, including every f32 bit. The check
is recorded by the successful chained command and summary in
`work/limited-loop1/sparse-ab-and-action1/`; the cached probe is in
`work/limited-loop1/sparse-32-m256/`. The runs occurred separately, with no paired
warmup/order control, and evaluation elapsed time includes fresh memo startup.
These figures are diagnostics, not a fair timing comparison or evidence of a
speedup. Fewer evaluated states alone does not establish lower runtime.

Both runs used **reachable validation**, checking membership of encountered
fields and reached policy entries. They did not validate all policy entries or
evaluate all 817,740 boundary roots. Read-only source mappings totaled
431,649,337,435 virtual bytes; that is not resident RAM usage, and logical reads
do not measure physical I/O. Running all roots (`roots=0`) requires full input
validation. No all-root sparse completion or full-loop performance is established
by these 32-root outputs.

## Bounded fixed-policy DAG probes

`loop1-dag-probe` now compiles ordered expressions for selected original boundary
roots using the sparse policy reader and its immutable-input/validation contract.
Bounded state/action construction memos may discard entries after collisions;
this causes recomputation, while equivalent expressions remain globally shared.
The resulting DAG is retained for repeated numeric applications of the same
fixed policy. Compiled L9 supplies exact terminal leaves. Original reveal order,
denominators, missing zero terms and reward-inside-terminal arithmetic remain
unchanged; there is no boundary reduction or reflection.

Two completed probes used `validation=reachable`, five repeated DAG applications
per case and a separately computed sparse scalar reference:

| Compiled roots | DAG nodes | Edges | Requested memo cap | Actual memo payload |
|---|---:|---:|---:|---:|
| 32 | 24,486 | 26,719 | 256 MiB | 201,326,592 bytes |
| 1,024 | 41,519 | 61,940 | 4 GiB | 3,221,225,472 bytes |

| Compiled roots | Case | Scalar roots bit-checked | B seconds | E median seconds | Scalar seconds | Reported break-even applications |
|---|---|---:|---:|---:|---:|---:|
| 32 | Actual affine | 32 | 3.216082185 | 0.006097222 | 0.889877334 | 4 |
| 32 | Signed linear | 32 | 3.216082185 | 0.006239773 | 0.731964485 | 5 |
| 1,024 | Actual affine | 1,024 | 18.158147708 | 0.005743463 | 6.141819322 | 3 |
| 1,024 | Signed linear | 32 | 18.158147708 | 0.005598040 | 2.621499111 | Not comparable |

All 32 outputs matched bitwise in both cases of the first probe. The second
probe checked all 1,024 actual-affine outputs, but independently checked only
32 signed-linear outputs. Its signed E timing evaluates all 1,024 roots, so the
code correctly reports no break-even against the 32-root scalar measurement.
All DAG outputs were finite. The 1,024-root run's `executable-reference.json`
verifies that it used the same executable as the saved `dag32-001/solver.bin`
and refers to that binary's source/input identity record.

These are bounded probe results, not matched full-loop timing comparisons. E is
the median of five applications of an already compiled plan; each scalar S is
one separate numeric reference execution, including fresh memo startup and
encountered-field validation. Neither probe performs full stored-policy
validation. Even where root counts match, the printed `ceil(B/(S-E))` applies to
that probe's preparation/application scope; it is not measured full-solve
amortization. Logs are in `work/limited-loop1/dag32-001/` and `dag1024-001/`.

The DAG has separate limits of five million nodes, 25 million edges and at most
4 GiB of construction memo. An actual early-guard check set
`PI_LOOP1_DAG_MAX_NODES=0` with missing input paths: it rejected the invalid cap
before loading those paths and created no output.

## Stopped all-root attempt and RSS enforcement

The all-root attempt in `work/limited-loop1/whole-001/` requested **all 817,740
original roots**, **full validation of the 86,329,867,487 stored policy states**,
a 4 GiB memo cap and `baseline=full`, requiring full-root scalar comparisons for
both actual-affine and signed-linear cases. It rebuilt release after the focused
DAG checks, captured a copied solver, `source.tar.gz` and binary/source/input
identities, and configured input-version comparisons before and after execution.
Its settings were a **32 GiB
cgroup memory-charge cap**, CPUs **0–3**, a monitored **32 GiB disk allowance with
a 31 GiB stop threshold**, and no time limit.

The run was stopped and **no whole-loop result was accepted**. Although the
cgroup memory cap was configured correctly, solver RSS was observed around
**51.2 GiB**, and later around **104 GiB**. Shared read-only file-backed mmap pages
can count in the solver's RSS while their memory charge belongs elsewhere.
Consequently the cgroup charge limit did not enforce the user's **32 GiB RSS
limit**. These observations are not an RSS peak measurement, and the attempt
must not be used as resource-compliant proof.

The runner reported interruption with return code **130** after **337.572358
seconds**, but its initial cancellation failed to stop the systemd service's
children. The solver continued until it was explicitly killed with
`systemctl --user kill --kill-who=all --signal=KILL` for that job's service. The
subsequent process check showed only a solver zombie with **RSS 0**. An interrupted
runner summary alone therefore did not establish that computation had stopped.

The log reached full validation of all stored policy states in about **245
seconds** and compilation of **18,781 roots** before the explicit stop. That is
progress from an invalidated attempt, not an accepted all-root DAG, application
cost or full-solve result. The subsequent residency/RSS/cleanup repairs are
described below; they do not make the stopped run compliant retrospectively.
Earlier numerical comparisons retain their stated scopes, but a cgroup-only
limit does not certify the requested RSS bound.

## RSS repairs and the second all-root attempt

The mapped sparse reader now releases source-page residency with Linux
`MADV_DONTNEED` during validation and bounded lookup work. This drops resident
pages from the solver's mappings while preserving read-only source contents;
the file cache may retain the underlying pages. The runner also
monitors **summed process RSS**, including nested job cgroups, every **50 ms**.
Shared pages count once per process in this sum. For a 32 GiB budget it stops at
**28 GiB RSS**, leaving 4 GiB of headroom; the 32 GiB cgroup charge cap remains
separate. Sampling can overshoot the stop threshold, so this is monitored RSS
enforcement, not a kernel hard RSS quota.

Missing RSS information still fails closed. A temporary `VmRSS` gap during
exec/exit caused the earlier `unit-007` run to stop safely after 19 limiter tests.
The reader now falls back to `/proc/.../statm` only with stable process identity;
permission failures or otherwise uninspectable live processes remain errors.
The completed `work/limited-loop1/unit-008/` run records:

| Repaired bounded verification | Result |
|---|---|
| Runner tests | 21 passed |
| Rust tests | 182 passed: 178 main + 4 auxiliary; 10 ignored |
| Release build | Passed, 54.65 s |
| Enclosing run | Completed, 182.178 s |
| Monitored summed-process RSS peak | 809,529,344 bytes, about 772 MiB |
| Cgroup events | No memory-max, OOM or OOM-kill events |

`work/limited-loop1/resource-smoke-001/result.json` records two actual controls
checks. With a **256 MiB** memory budget and **224 MiB** RSS stop threshold, the
RSS test returned `rss-limit`/124 at a monitored peak of **244,453,376 bytes**, below
256 MiB, without cgroup memory-max or OOM events. CPU affinity was 0–3. A separate
SIGINT cancellation used a child and grandchild that ignored SIGTERM; it returned
130 with `termination.confirmed=true`, and both descendants were checked gone.
The result retains failed systemctl kill/stop diagnostics; the fallback cleanup
nevertheless confirmed termination. These checks verify descendant termination
after the interrupt.

The retry in `work/limited-loop1/whole-002/` used a frozen copied binary with
source and input identities captured. During validation, its log had completed
through L3, reporting **17,466,975,495 validated states in 42.2 seconds**. One
validation-stage sample in `live-resources-validation.json` recorded solver
`VmRSS=1,388,552 KiB` (about **1.32 GiB**), CPU affinity 0–3 and zero swap. The
cgroup charge was at its 32 GiB cap and memory-max events were nonzero under
reclaim pressure, with no OOM or OOM-kill events. That is a **single solver RSS
sample, not a peak**, and it does not establish successful completion or the
retry's maximum RSS.

This retry was subsequently interrupted deliberately because compilation was
too slow; it did **not** time out. It completed full validation of all
**86,329,867,487 states in 208.2 seconds**, then reported **29 compiled roots and
296,864 visits in 216.6 seconds**. Earlier source had reported about 1,180 roots
and 2.9 million visits in ten seconds. These are diagnostic progress records
from different versions, not a matched benchmark. Source analysis strongly
points to repeated source-wide `MADV_DONTNEED` walks and about 850 MiB of retained
page tables as the bottleneck; that attribution is not based on a performance
trace.

The final summary records interruption/130 after **487.881307984 seconds**, with
`termination.confirmed=true`, an empty termination-error list and no solver
remaining in the host process check. Monitored summed-process RSS peaked at
**4,731,580,416 bytes** (about 4.4 GiB), below the 28 GiB RSS stop threshold and
32 GiB budget; monitored write usage peaked at **498,254,508 bytes**. This is a
successful cleanup and bounded-RSS observation for the stopped run, not a
completed whole-loop result. No whole-loop output was accepted.

## Targeted residency and the completed all-root probe

The tested replacement records at most **2,048 `u64` source-bucket entries**
(16 KiB payload), with 128 KiB buckets and the existing conservative 256 MiB
read-charge window. Retirement sorts, deduplicates and coalesces those entries,
then advises only the padded ranges actually touched. Source-wide retirement is
reserved for phase boundaries. Coverage includes 18 sparse and 12 DAG tests;
retirement counters and time are reported separately.

`work/limited-loop1/whole-003/` completed with return code 0 using a frozen binary
with saved source and input identities, all original roots and full baselines
for both vectors. Construction covered **all 817,740 roots**, with **2,782,565 DAG
nodes** and **10,715,097 edges**. Reported B is **5035.596176711 seconds**;
validation of all **86,329,867,487 stored policy states** took
**206.695887924 seconds**, so B plus validation is **5242.292064635 seconds**.
Total reported setup, including common preparation, is **5245.723532 seconds**.
Both full-root comparisons passed finite and bitwise checks:

| Completed case | Bounded numeric DFS S seconds | DAG E median seconds | Reported break-even applications |
|---|---:|---:|---:|
| Actual affine, all 817,740 roots bit-identical | 3091.580989225 | 0.049296621 | 2 |
| Signed linear, all 817,740 roots bit-identical | 3180.442059102 | 0.049688517 | 2 |

S is **one bounded numeric DFS reference measurement per case**, not the old
materialized layer sweep or a full PI run; E is the median of **five warm
applications per case**, using a prepared dense reset vector. Dense reset
preparation is excluded from both timings. Additional production-wrapper
allocation/scatter work is outside E, so E is not a cold CLI runtime or an
end-to-end GMRES matvec latency. The two-application break-even has this
fixed-policy/probe scope.

The completed limited run took **11521.378719811095 seconds**. Monitored
summed-process RSS peaked at **5,319,933,952 bytes** (about 4.955 GiB), and monitored
new-write/scratch usage peaked at **510,732,740 bytes** (about 0.4757 GiB). The run
used CPU affinity 0–3, a 32 GiB RAM budget, zero swap, a 32 GiB write budget and no
time limit, with no OOM events. The terminal summary, saved identity, input
post-checks and probe output are present. The earlier single
`live-resources-dag-early.json` RSS sample is superseded by the completed
monitored peak.

The same-binary limited CLI regression in
`work/limited-loop1/whole-krylov-001/` also passed: the selected nonzero-warm-start
fixture performed **one actual Krylov iteration**, then passed the original raw
bitwise gate, a zero Bellman residual check and exact zero-output publication.
This checks production integration on that fixture. A complete real-data
GMRES/Krylov solve, complete PI, final value tables, other kick-table variants and
the remaining original five-step success criteria are not established by these
checks. Those broader full-solve results remain unverified and are separate from
the current complete-loop1 application-cost goal.

## Validation history

The following completed checks cover the earlier L9 integration and L8 slab
probe. Logs are under `work/loop1-validation/`. The probe's bitwise checks cover complete
L9 and the full bounded L8 slab across three input vectors and both reward modes.
Unit tests cover ordered terminal arithmetic, signed values, missing probability
mass, stream limits, malformed inputs and source mutation/replacement. These
checks do not substitute for a complete production solve or a full-layer L8/L7
compression measurement.

| Verification | Result | Log |
|---|---|---|
| Normal Rust suite before the final rank/gather additions | 138 passed (4 auxiliary + 134 main), 9 ignored | `all-unit-tests.log` |
| Focused loop1 suite, including three new rank tests | 25 passed, 1 ignored | `ranked-unit-tests.log` |
| Final gathered production fixture: independent scalar comparison, signed modes and source changes | 1 passed, 86.69 s | `gather-production-unit.log` |
| Resume/certificate regression suite | 19 passed | `resume-tests.log` |
| Actual release CLI regression suite | 20 passed, 62.685 s | `all-runtime-tests.log` |
| Complete-L9 and bounded-L8 real-data probe | All six vector/reward combinations finite and bit-identical, including gather and rank-direct paths | `production-probe-gather.log` |
| Final release build | Passed | `gather-release-build.log` |
| `git diff --check` | Passed | — |

The focused and single-fixture runs overlap with the normal suite; their counts
are not additive. The final rank/gather changes were checked by the focused
fixtures, actual release CLI and real-data comparisons rather than a second full
normal-suite run at that stage.

The subsequent full-L8/action-cache stage used the 32 GiB cgroup/four-CPU runner.
Its completed `work/limited-loop1/unit-004/output.log` records **160 Rust tests
passed** (4 auxiliary and 156 main), **10 ignored**, and **13 runner tests passed**.
That composite run then failed a resume test exposing a same-tick mutation whose
recorded metadata was unchanged. The guard now fingerprints regular files up to
8 MiB on each signature check, with bounded reads and metadata checks around the
read. Four regression tests were added, bringing the resume suite to 23.

The corrected run in `work/limited-loop1/checks-005/` completed successfully:

| Corrected bounded verification | Result |
|---|---|
| Release build | Passed, 54.25 s |
| Complete-L8/action-cache actual CLI fixtures | 9 passed, 45.465 s |
| Resume/certificate suite, including the four new fingerprint tests | 23 passed, 113.465 s |
| Enclosing limited run | Return code 0, 213.84 s, no OOM or memory-limit events |

The nine CLI fixtures include the corrected verbosity-dependent GMRES assertion
that failed in the earlier `runtime-001` run. They check original final
verification, standalone RAM/file/mmap sinks, empty layers, memory caps,
gamma rejection, malformed unreachable policies, bitwise comparison failure and
action-cache-only discounted operation. Historical and corrected checks overlap;
counts are reported per run.

The complete-L9/L8 production-data run in `work/limited-loop1/full-l8-001/` has
also completed successfully, as detailed above, including every L8 output in both
affine and signed linear cases. This adds complete two-layer evidence, not a
whole-loop or full-solve result.

The later `work/limited-loop1/unit-006/` run passed **179 Rust tests** (175 main
and 4 auxiliary), with **10 ignored**, and completed in 130.86 seconds. The
`release-004/` run passed the **11 focused DAG tests** and built release in
54.60 seconds. These runs validate the tested DAG implementation and precede
the first all-root attempt, which reran focused checks before execution. That
attempt was subsequently stopped for the RSS/cleanup issue above and produced
no accepted whole-loop result.
The later residency/limiter phase's completed checks are `unit-008/`: 182 Rust
tests and 21 runner tests passed, with the separate actual RSS/cancellation
smokes detailed above. The `whole-002/` retry was subsequently stopped for the
compilation-cost issue above.

`work/limited-loop1/unit-009/` completed successfully in **322.299291666 seconds**:
**189 Rust tests** passed (185 main + 4 auxiliary), with 10 ignored; release built
in **57.19 seconds**; actual CLI suites passed **11 WHOLE tests** (54.582 seconds),
**9 L8 tests** (47.831 seconds) and **3 legacy GMRES tests** (13.880 seconds).
Monitored summed-process RSS peaked at **1,593,856,000 bytes**, with no cgroup
memory-max or OOM events. This verifies the tested production interface and
targeted-retirement fixtures.

The later `work/limited-loop1/whole-003/` execution completed with return code 0.
Both actual-affine and signed-linear comparisons passed finite and bitwise checks
for **all 817,740 original roots**, with the completed timings and monitored
resource peaks recorded above. This establishes the full-root fixed-policy DAG
probe for its captured binary and inputs.

The subsequent same-binary `work/limited-loop1/whole-krylov-001/` run completed
with return code 0. Exactly **one selected CLI test** ran and passed in **4.561
seconds**, with no skipped tests, errors or failures. It checks a nonzero warm
start, exactly one Krylov iteration, the original raw-bit final gate, zero Bellman
residual and exact zero-output publication. The enclosing limited run took
**5.232635600 seconds**, with monitored summed-process RSS peaking at
**1,584,619,520 bytes**, CPU affinity 0–3, a 32 GiB RAM budget, no time limit and
no OOM events. Its executable hash matches the frozen `whole-003/` binary.
Together with the earlier 11 WHOLE CLI cases this covers 12 cases across the two
runs; **the 12 cases were not rerun as a suite**, and the 189-test Rust count is
unchanged. A complete real-data solve or PI run is not part of these checks.
Later source changes require their own build and validation; recorded results
apply to the binaries and settings captured for their runs.

The final independent [completion audit](../work/limited-loop1/whole-003/completion-audit.json)
passed all 21 checks for the complete-loop1 application-cost goal. It verified
the frozen binary and source archive, Rust/Cargo sources unchanged at that audit time,
all 32 input identities, both full-root bit comparisons, timing arithmetic and
the finite published output. The [cleanup audit](../work/limited-loop1/whole-003/cleanup-audit.json)
also confirmed all four current/earlier stopped experiment units inactive, their
cgroups gone and no matching solver or launcher processes. The
[evidence record](../work/limited-loop1/whole-003/final-evidence.json) collects the
measurements and their scope; broader solve and PI claims remain separate.
