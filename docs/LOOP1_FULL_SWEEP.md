# Complete loop1 applications

The optimization target is one complete boundary-to-boundary application of a
stored policy through layers L9 to L0. A smaller GMRES boundary is deferred, and
left/right reflection is excluded. The complete L9/L8 measurements in
[LOOP1_COMPILED.md](LOOP1_COMPILED.md) are separate from the completed all-root
affine and signed applications below. Both match an independent bounded numeric
reference. Their final provenance/output/cleanup audit passed for this complete
loop1 application-cost goal; these measurements establish no complete solve or
policy-iteration speedup.

## Current experiment limits

The user has confirmed these limits for ongoing experiments:

| Resource | Limit |
|---|---|
| Aggregate RAM | 32 GiB |
| CPU allocation | 4 CPUs |
| Combined new outputs and scratch | 32 GiB |
| Wall time | Unlimited |

These limits apply to complete-loop optimization experiments as well as bounded
probes. They do not reduce the objective to tail-only acceleration. Use
`scripts/run_limited_experiment.py` for resource enforcement; do not launch the
128 GiB or 224 GiB profiles for current experiments. Both exceed the confirmed
limits. The larger profile below is retained as unused capacity-planning data.

The runner uses a transient user service with an actual cgroup
`memory.max=34359738368`, no swap allowance, whole-group OOM handling and four
allowed CPUs. It verifies these settings before starting the command. The CPU
quota controller is unavailable on the current user hierarchy, so the actual
four-CPU affinity is the CPU restriction; the runner records that enforcement
mode rather than claiming a working CPU quota. Wall time is unlimited with
`--seconds 0`, which is now the default.

The aborted `whole-001` run exposed a gap: that cgroup limit was present, yet the
solver's actual RSS exceeded 32 GiB. Shared read-only file-backed mapping pages
can already be charged to another cgroup while still appearing in the solver's
RSS. Thus `memory.max=32 GiB` and a small `memory.current` do not prove that the
process RSS stays within 32 GiB.

The corrected read-only input path records actual scalar reads in a bounded
2048-entry `u64` array (16 KiB of payload), using 128 KiB source buckets. It
coalesces touched buckets and releases only those ranges, with padding, through
`MADV_DONTNEED`. Whole-source retirement runs at phase boundaries. This replaces
the repeated source-wide retirement that made the stopped `whole-002` compiler
too slow, while continuing to bound resident mapping pages. The runner
has both parent and worker RSS monitors at 50 ms intervals, summing processes in
the service and its subgroups. Shared pages are conservatively counted once per
process; threads are not counted again. The default 32 GiB allowance stops at
28 GiB summed RSS, leaving 4 GiB of headroom. This is a sampled guard that can
overshoot between observations, not a hard kernel RSS limit. Transiently missing
RSS counters use an identity-checked `statm` fallback; monitoring fails closed if
it cannot establish a valid live-process measurement.

Termination uses the supported `systemctl --kill-who=all` option, with direct
cgroup-process killing as a fallback, and verifies that the service has no live
processes before reporting confirmed cleanup. Unit and live smoke checks of
these corrections are recorded below. The historical `whole-003` probe completed
the full-boundary operator checks and its own resource record; see its
[final evidence](../work/limited-loop1/whole-003/final-evidence.json).
The later `whole-gmres-001` attempt was intentionally interrupted. Its successor,
`whole-gmres-002`, completed full-boundary raw warm-start acceptance with zero
Krylov iterations and no DAG construction; see the
[production record](OPTIMIZATION_VALIDATION.md#gmres-warm-start-precheck-full-boundary-production-acceptance).
The user-confirmed RAM, CPU, output/scratch and unlimited-time constraints
remain unchanged.

All outputs must be directed into the fresh `--output-dir` or an explicitly
declared `--write-root`. The runner includes logs, temporary files and default
build/cache directories in its disk monitoring. It checks combined logical or
allocated file sizes every 0.5 seconds and applies a per-file size limit. This
32 GiB budget has a 31 GiB stop threshold, leaving 1 GiB of headroom. The general
headroom is `max(1 MiB, min(1 GiB, budget / 32))`. This monitor is not a filesystem
quota and can briefly overshoot; use bounded writers that check their intended
output size before creating large files. Do not use
the weaker `--allow-process-limits` fallback for these experiments.

The full-layer representation is incompatible with the disk limit once
it spills L7: that layer alone requires 35,679,165,760 f32 bytes, or 33.229 GiB.
The complete-layer table below identifies still larger middle layers. A new
whole-loop route therefore needs bounded intermediates rather than materializing
all intermediate layers onto disk. The default-off whole-operator route now has
a successful production release build and CLI fixture coverage, using bounded
memoization and shared ordered expressions. Its options, command and original
numeric acceptance check are documented in
[LOOP1_COMPILED.md](LOOP1_COMPILED.md). The historical `whole-003` probe completed
full-boundary correctness and application-cost checks against a bounded numeric
reference, as recorded in its
[final evidence](../work/limited-loop1/whole-003/final-evidence.json).
Those historical checks do not establish production GMRES convergence or runtime.
The later full-boundary raw warm-start acceptance and its measured CLI runtime
are recorded separately in
[OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md#gmres-warm-start-precheck-full-boundary-production-acceptance).

## Unused 224 GiB capacity-planning profile

`config/loop1-224gb.env` is an untested, unused larger-machine profile that is
not permitted under the current experiment limits. Its arithmetic was checked
against the approximately 251.302 GiB host. It sources the unchanged
`reference-128gb.env`, then changes only four solve settings:

| Setting | Reference | Larger profile |
|---|---:|---:|
| `FULL_MDP_MEMORY_MAX_GIB` | 128 | 224 |
| `FULL_MDP_MEMORY_RESERVE_GIB` | 32 | 48 |
| `FULL_MDP_RAM_PAIR_MAX_GIB` | 88 | 160 |
| `FULL_MDP_RAM_NEXT_LAYER_MAX_GIB` | 96 | 160 |

The 224 GiB planning budget leaves 176 GiB after its reserve. The reserve covers
graph/index allocations, the GMRES basis when applicable, cache metadata,
allocator overhead and other live allocations outside the charged value buffers
and worker payloads. It must be sized for the actual invocation. Other processes
also consume host memory. Physical RAM exceeds the configured budget by
approximately 27.302 GiB; this arithmetic is not a guaranteed free-memory margin
or a process RSS bound. The scheduler is a heap planner; file cache and mappings
have separate operating-system accounting.

No automatic machine detection selects this profile, and it must not be sourced
for current experiments. It preserves the reference 20 threads, task limits,
cache budget, `FULL_MDP_RAM_NEXT_LAYER=1` and `FULL_MDP_MMAP_SPILL=0`. It does not
enable compiled L9 or change policy actions, state membership, reveal order, the
solver boundary or floating-point expressions. The RAM allowance changes whether
a layer is accumulated directly in RAM or written and loaded from scratch.

## Metadata planning check

The following is a planning result from the current ten production field-index
files, not a measured runtime or RSS result. Their magic/version, contiguous
offsets, sorted field IDs, counts, total states and membership-file byte lengths
were checked without reading membership or policy bodies. The scheduling
calculation follows `SweepTuning` with the profiles' 817,740-qid domain, 2,048
block group limit, 2,000,000-state limit, 64 MiB task limit, 16 GiB lookup cache and
20 requested workers. These figures describe the uncompiled fixed-policy path;
retained compiled tables must also be charged when enabled. Neither column is
the permitted resource-limited experiment configuration.

| Layer | States | f32 GiB | 128 GiB planned sink | 224 GiB planned sink |
|---|---:|---:|---|---|
| L9 | 20,959,824 | 0.078 | RAM | RAM |
| L8 | 1,047,622,806 | 3.903 | RAM | RAM |
| L7 | 8,919,791,440 | 33.229 | RAM | RAM |
| L6 | 19,662,015,314 | 73.247 | file | RAM |
| L5 | 21,684,252,636 | 80.780 | file | RAM |
| L4 | 17,528,249,972 | 65.298 | file | RAM |
| L3 | 14,671,958,083 | 54.657 | file | RAM |
| L2 | 2,702,659,080 | 10.068 | RAM | RAM |
| L1 | 91,540,592 | 0.341 | RAM | RAM |
| L0 | 817,740 | 0.003 | RAM | RAM |

The largest simultaneous pair is L5 + L6: 165,385,071,800 bytes, or
154.026850872 GiB. Under the unused 224 GiB profile, it fits the 160 GiB pair
allowance and 176 GiB operator allowance with the scheduler's minimum
worker/cache requirements. All non-L0 layers would retain 20 workers; L0 has only
one group. The planner would reduce each worker's lookup-cache payload to about
530.513 MiB at the largest pair, and to about 734.004 MiB at L4. Other non-L0
layers would use 819.2 MiB. Its maximum charged estimate, including the 48 GiB
reserve and twofold cache allocation allowance, is 224 GiB. This is not a
measurement of allocation or residency.

According to the reference-profile plan, L6 through L3 would write
294,185,904,020 scratch bytes (273.981973547 GiB), then read the same amount when
pinning the completed layer. The larger profile could remove a total of
588,371,808,040 logical scratch-transfer bytes (547.963947095 GiB) per application.
This is a potential transfer reduction, not a measured result. Physical device
traffic can differ because of page cache and filesystem behavior. The complete
backward calculation and graph/membership/policy reads remain. The planning
result establishes neither a speedup nor the absence of memory pressure.

## Resource-limited comparison procedure

Run bounded candidates sequentially with the confirmed limits. Do not overlap
compute or builds with timed runs. Record available memory, swap activity,
competing jobs and the actual experiment process cgroup. Check the experiment's
`memory.max`, `memory.swap.max`, allowed CPUs and ancestor limits. Also monitor
the actual experiment process tree's RSS and verify that interrupt/limit handling
terminates every worker. Cgroup charged memory alone is insufficient for shared
input mappings, as the aborted `whole-001` run demonstrated. Compare
`memory.events` before and after each run; historical ancestor OOM counters are
not evidence that the measured experiment caused an OOM.

Record the exact executable hash, source revision/diff identity, solve
environment, graph identity, all ten membership index/data and policy file
identities, boundary input identity, output/scratch paths and storage devices.
Keep inputs read-only. Capture identities before and after the run and reject a
comparison if they changed. The working tree currently has no committed
baseline, so a commit hash alone is insufficient to identify source.

Use fresh, invocation-owned output directories. The runner requires that its
`--output-dir` not already exist. Supply the candidate command after `--`, with
these explicit runner options: `--mem-gib 32 --cpus 4 --disk-gib 32 --seconds 0`.
The command must use bounded memory and intermediate storage; invoking an
unchanged complete materialized sweep is not a valid experiment under these
limits. Keep input files, boundary vector, gamma and resource limits equal across
comparisons. Record every algorithm or executable change. The standalone
`apply-fixed-policy-boundary` command now accepts the same operator options as
GMRES: `PI_LOOP1_COMPILED_L9=1` prepares L9, `PI_LOOP1_COMPILED_L8=1` prepares L8
and also enables compiled L9, and `PI_LOOP1_ACTION_CACHE_MIB` requests the bounded
action-average cache. Compiled L9/L8 require `gamma=1`. Record these options and
their construction cost separately; enabling them does not make complete-layer
materialization fit the current output/scratch limit.

The separately built whole-operator opt-in avoids full intermediate layer
materialization. See [LOOP1_COMPILED.md](LOOP1_COMPILED.md) for its production
settings and invocation. Passing its small CLI fixtures does not establish a
successful all-root production-data solve.

The standalone command rejects nonfinite input/output values and output/input
aliases. Its optional `compare.f32` argument is now a required finite, bitwise
comparison when supplied, including signed-zero bits. A mismatch fails the
command before publication. Source bindings are checked before publication for
prepared paths, and the complete finite output is written to an owned temporary
file, flushed and synced, then atomically renamed into place. An existing output
is preserved when a comparison fails. These are current source behaviors; their
complete production-data coverage still requires a successful full-run check.

Check successful exit, complete boundary coverage and required state counts.
Record setup/compilation separately from evaluation, total application time,
wall time, memory peak, cgroup event deltas, swap changes and process/device I/O
where available. The original `fixed-policy apply done` timer excludes
graph/qforward setup and terminal-table preparation; command wall time includes
them. Report both when applicable. Check the runner's `summary.json`,
`worker-result.json`, cgroup counters and observed disk peak. Logical scratch
file lengths do not establish physical device writes.

Require the complete 817,740-element boundary result to be finite and compare
every byte with a valid reference output, using `cmp` or an equivalent exact
check. The standalone CLI's comparison also requires finite bit-identical
values; its numeric maximum-difference report alone is insufficient without a
successful exit. A tail fixture or a converged residual alone is insufficient to
establish bit-identical complete
applications. Signed linear applications need independent checks when their
kernel changes; the existing standalone materialized command measures affine
mode.

Record cache conditions and run order. A single sequential pair supplies an
observed timing with order/page-cache effects unresolved. Use matched
repetitions or reversed order before claiming a stable speedup. No complete-loop
speedup is established until the complete-run evidence is available under the
confirmed resource limits.

## Current resource-limited validation

These completed runs were configured with the confirmed resource limits:

| Check | Result | Record directory under `work/limited-loop1/` |
|---|---|---|
| Resource-runner tests | 21 passed | `unit-008/` |
| Rust unit suites | 189 passed: 185 main and 4 auxiliary; 10 ignored | `unit-009/` |
| Production release build | Passed, 57.19 seconds | `unit-009/` |
| Whole-operator CLI fixtures | 11 passed, 54.582 seconds | `unit-009/` |
| Complete-L8/action-cache CLI fixtures | 9 passed, 47.831 seconds | `unit-009/` |
| GMRES CLI fixtures | 3 passed, 13.880 seconds | `unit-009/` |
| Resume/certificate suite | 23 passed | `checks-005/` |

The enclosing `unit-009/` run completed in 322.299291666 seconds, with a sampled
summed-RSS peak of 1,593,856,000 bytes, zero memory-max/OOM/OOM-kill events and
return code zero. It covers the targeted mapping-retirement change and the
production whole-operator integration. The earlier passing suites did not detect
the shared-mapping RSS and cleanup gaps; none of the later checks retroactively
certifies the aborted `whole-001` run's resource compliance.

The live checks in `resource-smoke-001/result.json` exercised a 256 MiB cgroup
with slow memory growth. The RSS guard returned `rss-limit`/124 at a sampled
244,453,376-byte peak, below the hard cgroup allowance; memory-max/OOM/OOM-kill
counters stayed zero. A separate cancellation case had a child and grandchild
that ignored SIGTERM. SIGINT returned 130 and cleanup was confirmed with both
processes gone, including the direct cgroup fallback when systemd bus commands
failed. These small checks verify the exercised limit and cleanup paths, not a
complete-loop application.

The earlier L8 source-mutation test failure in `unit-002/` was repaired and is
superseded by the successful `unit-006/` run. The CLI suite includes strict
bitwise comparison failure, signed-zero handling and preservation of an accepted
output when comparison fails. Recorded results apply to the captured source,
binary and settings of each run. The later production integration is now covered
by `unit-009/` and its CLI fixtures, while further source changes require their
own build and checks.

The production-data run `full-l8-001/` also completed successfully. It evaluates
complete L9 and all **1,047,622,806 L8 states**, with every L8 output finite and
bit-identical to freshly computed original kernels for actual affine and
independent signed linear inputs. Policy-scoped construction took
**323.204851679 seconds**. The complete two-layer application measured
**57.37–68.18 times faster**, with construction amortized after 7–8 applications.
The canonical timing, arithmetic-mean sampling, preallocated-output exclusions,
construction breakdown and resource-accounting details are in
[LOOP1_COMPILED.md](LOOP1_COMPILED.md#complete-l9l8-measurement). This is a complete
L9/L8 result; it establishes neither whole-loop nor full-solve acceleration.

## Full-run evidence

The 128 GiB, 20-CPU baseline was **aborted at the user's request**, not completed.
The operator stopped process 79183 after approximately 24 minutes with L7 still
in progress; it exited 130. The run produced no complete boundary output or
complete-loop timing. Its last completed-layer records were L9 at 0.048 seconds
and L8 at 15.900 seconds. These partial observations were made outside the
subsequently confirmed 32 GiB/four-CPU limits. They are not a four-CPU baseline
and do not establish the full cost of loop1.

The partial log, copied executable and input identity record remain in
`work/full-loop1-validation/baseline-128g/`. The unused 224 GiB profile has no
complete-run evidence. The 547.964 GiB scratch-transfer figure is a metadata
estimate, not a measurement from either profile.

The subsequent `work/limited-loop1/whole-001/` experiment targeted all 817,740
boundary roots, complete source-policy/membership validation across
86,329,867,487 stored states, and a complete original baseline for actual affine
and independent signed linear inputs. It requested five matched repetitions and
a 4096 MiB expression/memo allowance, with the 32 GiB cgroup limit, four CPUs,
32 GiB monitored disk/31 GiB stop threshold and no wall-time limit.

**This run was aborted and is not resource-compliant evidence.** The operator
observed solver RSS around 51.2 GiB despite the correctly configured 32 GiB cgroup
limit. The runner's interrupted summary reports return code 130 and
337.572358096 seconds, but its cleanup attempted the unsupported
`systemctl --kill-whom` option and left the service's solver running. A later host
process check observed approximately 104 GiB RSS. The operator explicitly killed
all processes in `zxcl-experiment-run-b3f714d0d941.service` using the supported
`--kill-who=all` option; the subsequent solver state was a zombie with RSS zero.
The interrupted summary's elapsed time is not a complete-loop runtime or proof
that cleanup succeeded.

The partial output log reached validation of all 86,329,867,487 stored states at
245.0 seconds and a later compilation progress record for 18,781 roots. Neither
is a completed boundary-to-boundary application. There is no accepted whole-loop
boundary output, complete original-baseline comparison or whole-loop timing from
this run. The read-only mapping residency and resource-runner fixes were then
implemented and checked as described above.

The replacement `work/limited-loop1/whole-002/` was then deliberately stopped for
slow compilation, with return code 130 after 487.881307984 seconds. Its summary
records `termination.confirmed=true`, no termination errors, sampled summed-RSS
peak 4,731,580,416 bytes and monitored write peak 498,254,508 bytes. The host
process check confirmed no remaining solver. This establishes successful cleanup
for the stopped retry, not a completed application or accepted timing.

Source inspection pointed to repeated `MADV_DONTNEED` walks over approximately
402 GiB of mapped inputs, together with about 850 MiB of page tables, as the
likely cause of the compiler slowdown. This is an inference from source and
process observations, not a performance-trace result. The targeted touched-range
retirement described above replaces that hot path; whole-source retirement is
retained at phase boundaries.

The next frozen experiment, `work/limited-loop1/whole-003/`, uses a 4096 MiB memo
allowance, copied executable and captured source/input identities. It has now
constructed **all 817,740 boundary roots**, retaining **2,782,565 expression
nodes and 10,715,097 edges**. Full policy/membership validation covered all
86,329,867,487 stored states.

Both the actual-affine and independent signed-linear cases completed finite,
bitwise comparisons of every one of the 817,740 boundary outputs against a
freshly evaluated bounded numeric DFS reference. Construction and complete input
validation were shared by the two cases:

| Measurement | Seconds |
|---|---:|
| Policy-scoped construction B | 5035.596176711 |
| Complete policy/membership validation V | 206.695887924 |
| B + V | 5242.292064635 |

| Complete-boundary case | Numeric DFS S, one sample, seconds | DAG E, median of five warm samples, seconds |
|---|---:|---:|
| Actual affine | 3091.580989225 | 0.049296621 |
| Independent signed linear | 3180.442059102 | 0.049688517 |

The S/E application timings exclude preparation of the dense reset vector.
The calculated `ceil((B + V) / (S - E))` is **two applications in each case** for
this fixed policy and measured reference. Each S is a single measurement; each E
is a five-sample warm median. This comparison uses a bounded numeric DFS reference under the
current limits, **not the earlier full-layer materialized sweep**. It establishes
no end-to-end GMRES, solve or policy-iteration speedup. A policy change requires
rebuilding, and common setup and final acceptance work remain separate costs.

The job reached terminal `completed` status with return code zero after
11,521.378719811095 seconds, including construction, validation, both complete
numeric references and repeated DAG evaluations. Its summary records:

| Resource observation | Value |
|---|---:|
| Sampled summed-process RSS peak | 5,319,933,952 bytes, about 4.955 GiB |
| Monitored experiment write/scratch peak | 510,732,740 bytes, about 0.4757 GiB |
| CPU affinity | 0–3 |
| Cgroup charged-memory allowance | 32 GiB |
| Swap allowance and final swap use | Zero |
| Disk allowance / monitored stop threshold | 32 GiB / 31 GiB |
| Wall-time limit | Unlimited |
| OOM / OOM-kill events | Zero / zero |

The memory `max` event counter was nonzero under file-cache charge/reclaim
pressure; it is distinct from process RSS and OOM counters. The RSS peak is the
monitor's observed peak, rather than a proof about unobserved intervals. The
complete `inputs-after.json` reports unchanged input identities, and the
published experimental `boundary.f32.bin` is 3,270,960 bytes, exactly 817,740 f32
values. It is an experimental fixed-policy result, not a claim of an optimal
boundary value or complete solver convergence.

A separate small CLI case in `work/limited-loop1/whole-krylov-001/` used the same
frozen binary and passed its one selected nonzero-warm-start test: selected one,
executed one, skipped zero, failures/errors zero. The test asserts one actual
Krylov iteration, independent raw finite/bitwise verification, zero Bellman
residual and exact-zero publication. It took 4.561 seconds; the enclosing
resource-limited job returned zero in 5.232635600 seconds with observed RSS peak
1,584,619,520 bytes, CPU affinity 0–3 and no OOM events. This is a small integration
fixture, not evidence of convergence of a full production GMRES solve.

**Both complete-boundary cases and the small Krylov integration case passed.**
The independent final audit passed all 21 checks for this loop1 application-cost
scope. All 32 input identities were unchanged; the published 817,740-value output
was finite and matched its recorded hash; current and earlier stopped experiment
units had no remaining cgroups or matching processes. The
[completion audit](../work/limited-loop1/whole-003/completion-audit.json),
[evidence record](../work/limited-loop1/whole-003/final-evidence.json) and
[cleanup audit](../work/limited-loop1/whole-003/cleanup-audit.json) retain the exact
scope and limitations.
Detailed implementation and other measurements are documented in
[LOOP1_COMPILED.md](LOOP1_COMPILED.md). No full solve, policy-iteration, other
kicktable variant or optimal-value claim follows from these fixed-policy
application measurements.
