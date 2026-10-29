# Build and solve the MDP

MDP construction and MDP solving are independent, restartable pipelines:

- `scripts/build_mdp.sh`: `graph.bin` to `forward_pruned/`
- `scripts/solve_mdp.sh`: `forward_pruned/` to the stable policy and full `V*`

Read the entire guide before starting. The outputs are very large and some stages take
hours or days.

## 1. Hardware and storage

My run used 24 physical CPU cores, 128 GiB of usable RAM and separate NVMe scratch.
Construction and solving put different kinds of load on storage:

| Phase | Minimum | Recommended |
|---|---|---|
| MDP build | 32 GiB RAM, 8 cores, 500 GiB NVMe scratch, 1.5 TiB persistent storage | 128 GiB RAM, 20+ cores, 2 TiB+ NVMe |
| MDP solve | 32 GiB RAM, 8 cores, 1 TiB NVMe active space plus archive storage | 128 GiB RAM, 20+ cores, 2 TiB+ NVMe plus result storage |

The 32 GiB profile trades memory for substantially more spill I/O and runtime. Load a
provided profile before starting either pipeline:

```bash
source config/low-memory-32gb.env
# or
source config/reference-128gb.env
```

The maximum forward-pruned value layer is about 81 GiB. The full pruned state set is
about 86.3 billion states, and one complete f32 value snapshot is about 321 GiB.

## 2. Configure paths

Keep persistent and disposable I/O on different devices when possible:

```bash
export ZXCL_DATA_DIR=/mnt/large/zxcl-pc
export ZXCL_SCRATCH_DIR=/mnt/nvme/zxcl-pc
export RAYON_NUM_THREADS=20
```

Optional tuning variables and their defaults are shown by:

```bash
./scripts/build_mdp.sh env
./scripts/solve_mdp.sh env
```

Path variables are independently overridable, so active indexes, spill and archival
results can live on different devices without symlinks:

| Variable | Purpose |
|---|---|
| `ZXCL_DATA_DIR` | Default persistent root |
| `ZXCL_SCRATCH_DIR` | Spill and temporary full-layer root |
| `GRAPH_BIN`, `RGRAPH_BIN`, `QGRAPH_PREFIX` | Graph input/output locations |
| `BACKWARD_DIR`, `PRUNED_DIR` | Constructed MDP locations |
| `ONE_LOOP_DIR`, `SEED_DIR`, `RESULTS_DIR` | Solver output locations |
| `SOLVER_BIN` | Prebuilt solver executable |

The remaining variables printed by `env` control thread counts, memory caps, shard
sizes, caches, GMRES limits and numerical tolerances. The checked-in profiles are
starting points for different machines; every value can be overridden.

Solver stages reuse results only when their completion record matches the graph,
pruned index, numerical inputs and current output destination. Files without a valid
completion record are rebuilt. The first identity check hashes the graph and pruned
chain files; later checks cache those hashes using file version metadata. Policy
improvement deletes its own temporary rolling scratch directory after the stage, and
preserves files outside that directory.

### Sweep work and memory limits

Greedy and fixed-policy sweeps share a state/byte-bounded scheduler. Adjacent chunks
reuse their next-layer lookup cache, while the aggregate state and byte limits keep
large field blocks from turning a lookup group into an unbounded allocation.

| Variable | Purpose |
|---|---|
| `FULL_MDP_BLOCK_CHUNK` | Maximum field blocks in one base chunk |
| `FULL_MDP_LOOKUP_GROUP_CHUNKS` | Maximum adjacent chunks sharing one lookup |
| `FULL_MDP_TASK_MAX_STATES` | Group state limit; default 2,000,000 |
| `FULL_MDP_TASK_MAX_MIB` | Estimated group working-byte limit; default 64 MiB |
| `FULL_MDP_MEMORY_MAX_GIB` | Combined heap planning budget; default 32 GiB |
| `FULL_MDP_MEMORY_RESERVE_GIB` | Reserve for graph, indexes, GMRES basis, cache metadata and allocation overhead |
| `FULL_MDP_CACHE_STATS`, `FULL_MDP_ACTION_STATS` | Optional cache/action counters in the actual sweep paths |
| `FULL_MDP_RAM_BITMAP_MIN_COUNT` | Experimental bitmap+rank threshold; `0`/unset disables it, `128` enables memory-beneficial candidates |

An indivisible field larger than a task limit runs alone and is charged in full.
The scheduler reduces worker count and per-worker cache payload when resident value
layers consume the budget; a candidate RAM output can fall back to a file sink.
The provided profiles set explicit 128 GiB and 28 GiB planning budgets. The reserve
must cover the GMRES basis and other live allocations for the selected run. File
mappings and OS page cache are reclaimable and excluded from pinned anonymous layer
bytes, so this planner does not replace an OS process/cgroup limit.

Bitmap+rank stores membership and offsets into the pinned values, avoiding a copy of
the values. It preserves reveal order and supports promotion to the existing dense
cache. It stays opt-in because building a new bitmap can outweigh faster lookups for
cold fields. Measure with the same cache budget before enabling it for a long run.

See [OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md) for the regression
checks, bounded real-data comparisons and measurement limits.
The experimental pre-reveal boundary, direct L9 lookup and DAG backends are
described separately in [BOUNDARY_DAG.md](BOUNDARY_DAG.md).
The full-boundary compiled L9 opt-in and bounded L8 expression-sharing probe are
described in [LOOP1_COMPILED.md](LOOP1_COMPILED.md).

## 3. Obtain and verify the field graph

```bash
./scripts/download_graph.sh
./scripts/build_mdp.sh prepare
```

`prepare` builds the release solver, validates `graph.bin`, constructs `rgraph.bin`,
and writes the quotient-chain forward/reverse graphs.

## 4. Construct the state space

```bash
./scripts/build_mdp.sh backward
./scripts/build_mdp.sh prune
```

`backward` starts at the terminal perfect-clear field, constructs layers 9 and 8 in
memory, then expands layers 7 through 0 with the external-memory frontier-spill path.
The resulting `backward/layerKq.*` files are a terminal-reachable superset.

`prune` intersects that superset with forward reachability from the empty boundary and
the forced-continuation condition. The resulting
`forward_pruned/layerKq_pruned.*` files define the final MDP index.

Validate the expected layer count before solving:

```bash
for k in $(seq 0 9); do
  test -s "$ZXCL_DATA_DIR/forward_pruned/layer${k}q_pruned.field_index.bin"
  test -s "$ZXCL_DATA_DIR/forward_pruned/layer${k}q_pruned.chains.u24.bin"
done
```

At this point MDP construction is complete. `forward_pruned/` is the only build output
required by the solver; retain or archive `backward/` if reproducibility is required.

The compact temporary-membership path, bulk writes and index-only backward loading
have a bounded real F3-to-F4 comparison in the
[graph-to-index validation record](OPTIMIZATION_VALIDATION.md#graph-to-index-and-v-solve-follow-up-2026-10-29).
That replay does not establish complete-build runtime.

## 5. Produce the u16 initialization seed

```bash
./scripts/solve_mdp.sh one-loop
```

This initialization does not evaluate all ten layers. It treats every surviving
layer-4 state as a full-valued boundary (`5040/5040`) and evaluates layers 3, 2, 1
and 0 backward in exact u16 units of `1/5040`. One f32 Bellman sweep then produces
`seed_l0/layer0.values.f32.bin`, which is used only to bootstrap policy iteration.

The normal defaults are therefore `ONE_LOOP_START_LAYER=3` and
`ONE_LOOP_BASE_FULL_NEXT_LAYER=4`. Both remain environment variables for experiments;
using `9` and `none` requests the substantially more expensive full ten-layer u16
calculation and is not required by the public solve pipeline.

## 6. Solve the infinite-horizon MDP

```bash
./scripts/solve_mdp.sh solve
```

For each round, the script:

1. performs one greedy argmax sweep and records a u16 action per state;
2. evaluates that fixed policy with matrix-free GMRES;
3. performs the next greedy improvement;
4. stops only when all ten policy files are byte-identical across consecutive rounds.

The safety limit is eight PI rounds (`MAX_PI_ROUNDS=8`). The solver stops earlier when
two consecutive policies are identical. Increase the limit only if your run reaches
it. GMRES starts from the previous accepted policy value, using the initialization
seed only for the first round. An already converged start skips Krylov iterations,
while the final actual f32 Bellman residual check still runs. Accepted values are
published atomically after that check, and failed or interrupted results receive no
completion certificate.

PI improvement uses `FULL_MDP_POLICY_ONLY=1`: it retains policies and layer-0 values,
uses RAM for adjacent value layers when they fit, and spills larger layers into an
invocation-owned temporary directory. This avoids retaining a full value snapshot
for every PI round. Standalone emission and `finalize` still materialize all ten
value layers by default.

The accepted f32 Bellman infinity-norm floor defaults to `3e-3`. Lower values may be
unreachable at V around 4,000 because a few f32 ULPs already exceed `1e-3`.

The current GMRES source-bucket and phase-timing changes are tracked in the
[V* solve follow-up](OPTIMIZATION_VALIDATION.md#graph-to-index-and-v-solve-follow-up-2026-10-29).
It records a completed production fixed-policy warm-start acceptance through the
full raw operator, with zero Krylov iterations. Complete solve performance remains
unmeasured; warm fixed-policy DAG timings alone do not establish it.

## 7. Materialize the final result

```bash
./scripts/solve_mdp.sh finalize
```

This writes:

- `results/vstar_boundary.f32.bin`: 817,740 quotient-boundary values
- `results/vstar_full/layerK.values.f32.bin`: all ten materialized value layers
- `results/policy_final/layerK.policy.u16.bin`: final greedy policy

`finalize` recomputes the greedy policy while materializing the layers and requires it
to be byte-identical to the stable policy recorded by `solve`.

## 8. Run both pipelines

After configuration and graph download, all stages can be run in order:

```bash
./scripts/build_mdp.sh all 2>&1 | tee build-mdp.log
./scripts/solve_mdp.sh all 2>&1 | tee solve-mdp.log
```

## 9. Materialize from the bundled layer-0 V*

The repository includes `precomputed/vstar_layer0.f32.bin`. Once `forward_pruned/`
has been built, PI and GMRES can be skipped:

```bash
./scripts/solve_mdp.sh from-vstar 2>&1 | tee materialize-vstar.log
```

This verifies the bundled file's size and SHA-256, writes it as
`results/vstar_boundary.f32.bin`, performs one backward argmax sweep to generate
layers 1–9 and the greedy policy, and preserves the supplied vector as layer 0.

An alternate boundary vector can be selected with `VSTAR_LAYER0`. Alternate files
must contain exactly 817,740 little-endian f32 values; only the bundled file has a
fixed checksum enforced by the script.

For long unattended runs, use your system's service manager and set an explicit memory
limit. On a NUMA machine, binding to the large-memory node can materially reduce
cross-node traffic.

## 10. Expected scale

My completed run had:

- 817,740 quotient boundary states
- approximately 86.3 billion total pruned states
- approximately 4,338.53 mean optimal boundary value

The solver decides when to stop by comparing consecutive policies. The mean value is
only a quick sanity check.
