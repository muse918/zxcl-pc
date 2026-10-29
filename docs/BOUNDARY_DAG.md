# Experimental pre-reveal boundary and DAG

All five proposed mechanisms have implementations: a smaller pre-reveal boundary,
direct terminal L9 evaluation, structural expression sharing, reduced policy
evaluation with optional failure deflation, and certified interval pruning.
They are experimental. Existing policy evaluation remains the wrapper default;
production performance of this reduced-boundary backend is unproven. The later
original-coordinate WHOLE backend has completed full-root fixed-policy checks
and repeated-evaluation measurements in [LOOP1_COMPILED.md](LOOP1_COMPILED.md).

## Boundary and numerical contract

`PreRevealIndex` partitions the **817,740** quotient states into **205,170** groups.
A group contains the available pair, four known queued pieces, and the bag before
the last random reveal. Each quotient state occurs exactly once, and children
retain ascending revealed-piece order, including bag refill and duplicate pairs.
The 205,170 count follows from enumeration and the complete inverse-reveal
partition, rather than dividing by a fixed bag size.

Let `R` average the post-reveal quotient values within those groups. For a fixed
policy at `gamma=1`, let `K` map a pre-reveal continuation to the ten-layer
quotient boundary. Then

```text
v = K u,       u = 1 + R v,
P = R K,       (I - P) u = 1.
```

Choices that observe a reveal remain inside its expectation: `E[max(...)]` cannot
be replaced by `max(E[...])`. Absent pruned members contribute zero with the full
original reveal denominator; their probability is never redistributed.

This is exact algebra over real numbers. Moving a reveal average across reward
addition changes floating-point rounding, even with ordered f32 DAG evaluation.
Reconstruction therefore always ends with a fresh original affine **f32**
`T_pi(v)` sweep and the existing publication gate. Finite values and the true
reduced residual are also checked. The acceptance tolerance is unchanged:
`PI_ACCEPT_BELLMAN_INF=3e-3` in the wrapper, `1e-3` for the direct CLI fallback.
Failure preserves the accepted destination.
Scratch/output alias checks compare canonical names and device/inode identity,
including existing hard links, before any scratch mutation.

## Direct L9 and ordered expression DAG

`FULL_MDP_L9_DIRECT=1` skips the materialized L9 value layer during fixed-policy
evaluation and policy-only greedy emission. L8 queries L9 on demand. Original L9
membership and policy validation remain authoritative, and greedy emission still
writes the L9 policy. Full/final snapshot emission retains all ten value layers.
The flag defaults to `0`. The materialized L9 value layer in existing validation
data is 83,839,296 bytes (about 80 MiB).

`Compiler` traverses only descendants of requested roots, memoizes
`(layer, field, qid)`, and interns ordered `Zero`, `Terminal`, `Average`, and
`Maximum` expressions. Keys include child order, multiplicity, and denominator;
equal numerical values alone never merge. First-action ties and zero/NONE
baselines are retained. Fixed policies have linear reachable nodes and accept
signed f64 Krylov vectors; their transpose reverses the same shared DAG.

The DAG backend compiles once per policy per process and reuses it for operator
applications. It has no persistent cache. Construction can still visit too many
states or compress poorly. With construction cost `B`, DAG application cost `E`,
and sweep cost `S`, compare `B + k E` against `k S`; measured break-even exists
only when `S > E`. A bounded late-layer probe cannot establish full-root savings.

## Reduced policy evaluation

Run from `zxcl-pc`; paths below are placeholders for existing inputs and new
outputs. These commands are interfaces, not a request to start a production run.

```bash
cargo run --release --offline --bin zxcl -- pre-reveal-check

target/release/zxcl boundary-dag-probe \
  /path/graph.bin /path/pruned /path/policy 8 16 /path/l0.f32 10

PI_REDUCED_OPERATOR=dag PI_REDUCED_METHOD=gmres \
target/release/zxcl pi-policy-eval-reduced \
  /path/graph.bin /path/pruned /path/policy \
  /path/accepted-v.f32 /path/new-v.f32 /path/scratch \
  1.0 300 1e-6 16

# Wrapper opt-in; its existing gmres backend remains the default.
PI_POLICY_EVAL_BACKEND=reduced PI_REDUCED_OPERATOR=dag \
PI_REDUCED_METHOD=deflated FULL_MDP_L9_DIRECT=1 \
bash scripts/solve_mdp.sh solve
```

The direct CLI accepts `zero` instead of an initial file. It requires exactly
`gamma=1` and complete, unique layer-0 empty-field reset membership containing
every quotient qid. Initial/output values retain membership-file order; explicit
qid mappings handle averaging and reconstruction. `sweep` rounds terminal inputs
to f32 and uses existing sweeps; `dag` applies compiled f64 expressions in memory.
Neither changes the final legacy verification requirement.

The physical empty reset field is identified by board hash `0` (field `0` in the
validation graph). The legacy `EMPTY_FIELD_ID` names a distinct PC terminal
sentinel. Reduced roots use the actual layer-0 field after checking its geometry.

| Setting | Default | Meaning |
|---|---:|---|
| `PI_POLICY_EVAL_BACKEND` | `gmres` | Wrapper: `gmres` or opt-in `reduced` |
| `PI_REDUCED_OPERATOR` | `sweep` | Reduced operator: `sweep` or `dag` |
| `PI_REDUCED_METHOD` | `gmres` | Reduced solve: `gmres` or `deflated`; deflation requires DAG |
| `PI_REDUCED_RESTART` | `40` | Restarted GMRES cycle length |
| `PI_GMRES_BASIS_MAX_GIB` | `24` | Existing Krylov workspace allowance |
| `PI_DAG_MAX_STATES` | `2000000` | Maximum memoized compiler states |
| `PI_DAG_MAX_NODES` | `5000000` | Maximum interned expression nodes |
| `PI_DAG_MAX_EDGES` | `25000000` | Maximum stored expression edges |
| `PI_DAG_FIELD_CACHE_MIB` | `64` | Membership/policy cache payload allowance |

Caps fail explicitly; there is no unbounded fallback. Oversized field blocks use
positional reads. These are logical allocation/payload caps, **not a hard RSS
limit**. Size `FULL_MDP_MEMORY_RESERVE_GIB` manually for graph/index storage, DAG
maps and metadata, Krylov vectors, live layers, allocator overhead, and other
resident data. The basis allowance does not include all operator allocations.

## Failure deflation and certificates

Failure `d` is propagated directly through missing mass and zero nodes, avoiding
cancellation in `1 - P 1`. For strictly positive normalized `mu`, define
`Q = P + d mu^T` and solve two matrix-free stabilized systems:

```text
(I - Q^T + mu 1^T) nu = mu,      1^T nu = 1,
C = 1 / (nu^T d),
(I - Q + 1 mu^T) h = 1 - C d,   mu^T h = 0,
u = C 1 + h.
```

This does not assume `P` is rank one or has only one slow mode. The implementation
checks row mass, positivity, normalization, both true system residuals, and the
original `(I-P)u=1` residual. Closed classes and zero stationary failure are
rejected. Deflated CLI publication additionally requires an outwardly verified
finite positive supersolution and weighted contraction for that fixed policy,
then reconstruction and the original f32 acceptance gate. A small residual by
itself is neither a properness proof nor a global value-error bound.

`certified_bounds` evaluates nonnegative interval boxes with outward f64 rounding
and checks gradual underflow. `prune` removes a Maximum child only when its upper
bound is strictly below a witness's lower bound; ties remain. Its guarantee is
restricted to the supplied box. `certify_and_prune` verifies a finite strictly
positive supersolution for the complete intended operator and a weighted
contraction factor below one before claiming the same optimal fixed point.
Any residual-error bound uses that certificate's conditioning. A certificate for
a fixed policy or partial sampled roots does not certify full-game optimality.
Pruning compacts the retained root subgraphs so orphan expressions are not
evaluated on later applications.

## Bounded validation and remaining work

`boundary-dag-probe` accepts layer `7`, `8`, or `9`, a root cap, and optional
repetitions (default `10`); replace the policy directory with `greedy` for greedy
sampling. The ignored real-data test compares bounded samples with independent
recursive and existing kernel references. Logs are under
`work/boundary-validation/`.

The real-data test uses 32 roots from each of L9 and L8 and eight from L7, in
both fixed and greedy modes. Ordered f32/f64 DAG outputs match an independent
recursive reference; L9/L8 f32 values and greedy policy codes also match existing
kernels. Signed fixed-policy inputs and transpose dot products are checked.
These are 72 sampled roots per mode, not a traversal of complete layers.

Representative f64 results below use signed inputs for fixed policy and varying
positive inputs for greedy. `S` is the sampled recursive reference, with bounded
membership/policy caching; it is **not** a full production sweep. Setup is
excluded. Three ABBA repetitions follow warmup.

| Sample | Visited states | DAG nodes / edges | Build B (ms) | Reference S (ms) | DAG E (ms) | B/(S-E) |
|---|---:|---:|---:|---:|---:|---:|
| L8 fixed | 151 | 152 / 119 | 66.19 | 8.859 | 0.158 | 7.6 |
| L7 fixed | 152 | 153 / 144 | 38.82 | 0.183 | 0.099 | 465.8 |
| L8 greedy | 165 | 304 / 428 | 77.42 | 0.193 | 0.101 | 847.3 |
| L7 greedy | 245 | 313 / 495 | 30.48 | 0.187 | 0.094 | 325.1 |

The small samples do not show a large expression-count compression. Build cost
can dominate, and these break-even figures do not establish a faster PI run.
Point-box greedy pruning removes 152 branches at L8 (428 to 275 edges) and 176
at L7 (495 to 312 edges), with no changed f64 root values. Node counts shrink
only from 304 to 303 and 313 to 311. Those proofs apply to the tested input point.

Validation on 2026-10-29 completed as follows. Ordinary suites leave eight
explicit-data/benchmark tests ignored; the explicit release run passes all eight.

| Check | Result | Log under `work/boundary-validation/` |
|---|---|---|
| Debug and release suites | 116 passed each: 4 auxiliary + 112 main | `final-debug-tests.log`, `final-release-tests.log` |
| Manifest/resume failures and backend switching | 19 passed | `resume-tests.log` |
| Actual legacy GMRES CLI | 3 passed | `actual-gmres-tests.log` |
| Actual reduced CLI: sweep, DAG, deflated, rejection, caps and hard links | 8 passed | `actual-reduced-tests.log` |
| Boundary enumeration and sampled probe commands | Passed | `pre-reveal-cli.log`, `probe-fixed-cli.log`, `probe-greedy-cli.log` |
| Direct-L9 production greedy/fixed fixtures | 2 passed | `direct-greedy-test.log`, `direct-fixed-test.log` |
| Explicit ignored release checks | 8 passed, 78.94 s | `ignored-real-tests.log` |
| Release build, Clippy, new-module formatting, shell syntax | Passed; Clippy retains style warnings | `final-release-build.log`, `final-clippy.log` |

The graph-backed CLI fixtures use a complete reset vector and NONE policies.
Nontrivial numerical behavior is checked separately with small nonsymmetric,
near-unit and multiple-slow-mode matrices. A strict slow-mode fixture asks for
precision below the f64 subtraction floor and is explicitly rejected. Its
successful counterpart derives the inner tolerance from `EPSILON * lifetime`;
the independent original-equation tolerance remains unchanged. This does not
establish nontrivial production policy convergence.

An actual full-boundary attempt with the stored production policy reaches the
default **2,000,000-state** compiler cap and stops after 24.70 s. Node/edge caps
remain 5,000,000/25,000,000 and the field cache is 64 MiB. The complete reset
membership and physical empty field pass validation; zero Krylov iterations are
requested. The preexisting output vector remains byte-identical. See
`full-boundary-budget-probe.log` and `.json`. A preliminary 1,000,000-state trial
also hits its cap in 17.12 s.

This demonstrates a safe budget rejection, **not a successful full DAG build**.
This older reduced-boundary DAG backend cannot evaluate the whole policy under
its default state cap. It remains opt-in; increasing that cap has not been
validated as a solution. The sweep backend still performs original full-layer
work. The later original-coordinate WHOLE backend uses a separate bounded
compiler and has completed the full-root checks linked above.

To repeat the explicit checks, use the dataset environment from
[the initial validation report](OPTIMIZATION_VALIDATION.md#reproduce), set
`ZXCL_VALIDATION_REPETITIONS=3`, and run:

```bash
cargo test --release --offline --bin zxcl -- --ignored --test-threads=1 --nocapture
ZXCL_RUNTIME_SOLVER="$PWD/target/release/zxcl" \
GRAPH_BIN=/mnt/fast2/muse918/pcfinder/graph.bin \
python3 -m unittest discover -s tests -p 'test_reduced_runtime.py' -v
```

Interval pruning currently runs **after DAG construction**. Certified avoidance
of building the 86,329,867,487-state production space is not implemented. For this
reduced-boundary backend, no full production run, complete DAG compression
measurement, end-to-end speed comparison, or peak-RSS validation is claimed.
The later original-coordinate WHOLE measurements are separate evidence. The
existing field graph/domain assumptions remain; SRS+/no-180 adaptation is outside
this implementation.

## Priorities and deferred deployment ideas

The current performance target is the cost of **one boundary-to-boundary PC
application (loop1)**: state visits, expression operations, membership/policy
reads and intermediate value I/O. A smaller outer GMRES vector is not evidence
of a cheaper loop1 application. Left/right reflection is excluded.

The compiler's two-million-state limit counts visited `(layer, field, qid)`
states before expression sharing. Its rejection does not establish that the
eventual unique-expression DAG is large. A useful follow-up experiment is a
bounded, streamed reachable-frontier/bottom-up compiler with disk-backed state
to expression mappings. Measure construction B, retained nodes/edges, repeated
application E, memory and I/O against the existing loop1 sweep S. Repeated
validation of immutable membership/policy should also be separated from numeric
applications only with explicit input-version binding.

The streamed full-L9 fixed-policy compiler and bounded L8 slab experiment are
implemented separately in [LOOP1_COMPILED.md](LOOP1_COMPILED.md). L9 has
20,959,824 states in the validation dataset, but each expression is Zero or
Terminal(zid), so at most 205,171 unique expressions. A flat u32
state-to-expression table needs 83,839,296 bytes, without one hash-map entry per
state. The first 1,949,970-state L8 slab shares 51,510 ordered averages; this does
not establish compression or speedup for the complete L8/L7 layers or loop1.

Further boundary condensation is deferred as a possible memory-saving option
for constrained deployment. These are mathematical ideas, not implemented
backends or measured loop1 speedups:

- Retaining only full-bag pre-reveal states leaves 15,120 unknowns. For a proper
  fixed policy, exact Schur elimination restores the other 190,050 values, but
  requires applying `(I-P_EE)^-1`; cost and dense fill may dominate.
- Retaining bag sizes 7 and 2 leaves 32,760 unknowns. With phase `phi=4*k mod 7`,
  a 5- or 10-placement PC advances phase by 1 or 2. The other five phases are
  acyclic until this retained set is reached, so `P_EE^5=0` and elimination uses
  a finite polynomial. The resulting macro application can still cost more.

Any deployment benefit must account for boundary/Krylov memory separately from
the graph, policy, DAG and scratch storage. Further condensation is not part of
the current loop1 optimization work.
