#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${ZXCL_DATA_DIR:-"$ROOT/data"}
SCRATCH=${ZXCL_SCRATCH_DIR:-"$ROOT/work"}
GRAPH=${GRAPH_BIN:-"$DATA/graph.bin"}
PRUNED=${PRUNED_DIR:-"$DATA/forward_pruned"}
ONE_LOOP=${ONE_LOOP_DIR:-"$DATA/one_loop"}
SEED_DIR=${SEED_DIR:-"$DATA/seed_l0"}
RESULTS=${RESULTS_DIR:-"$DATA/results"}
DEFAULT_BIN="$ROOT/target/release/zxcl"
BIN=${SOLVER_BIN:-"$DEFAULT_BIN"}
BUNDLED_VSTAR="$ROOT/precomputed/vstar_layer0.f32.bin"
VSTAR_INPUT=${VSTAR_LAYER0:-"$BUNDLED_VSTAR"}
VSTAR_SIZE=3270960
VSTAR_SHA256=40fece5d8b8f5ea4c57e69624f91ee9fb20a2be7a61093d5dbbd9889d4c5fff1
GRAPH_SHA256=7dc434e3611684756911480d6d338d37bf8ca5ade263611d651fa8709e0294f4

export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-20}
export FULL_MDP_RAM_NEXT_LAYER=${FULL_MDP_RAM_NEXT_LAYER:-1}
export FULL_MDP_RAM_NEXT_LAYER_MAX_GIB=${FULL_MDP_RAM_NEXT_LAYER_MAX_GIB:-96}
export FULL_MDP_RAM_PAIR_MAX_GIB=${FULL_MDP_RAM_PAIR_MAX_GIB:-88}
export FULL_MDP_L9_DIRECT=${FULL_MDP_L9_DIRECT:-0}
export PI_POLICY_EVAL_BACKEND=${PI_POLICY_EVAL_BACKEND-gmres}
export PI_REDUCED_OPERATOR=${PI_REDUCED_OPERATOR-sweep}
export PI_REDUCED_METHOD=${PI_REDUCED_METHOD-gmres}
export PI_GMRES_AITKEN=${PI_GMRES_AITKEN:-1}
export PI_ACCEPT_BELLMAN_INF=${PI_ACCEPT_BELLMAN_INF:-3e-3}
export PI_GMRES_BASIS_MAX_GIB=${PI_GMRES_BASIS_MAX_GIB:-24}
export PI_GMRES_SNAPSHOT_EVERY=${PI_GMRES_SNAPSHOT_EVERY:-0}
export FULL_MDP_MMAP_SPILL=${FULL_MDP_MMAP_SPILL:-0}
export FULL_MDP_BLOCK_CHUNK=${FULL_MDP_BLOCK_CHUNK:-512}
export FULL_MDP_SPARSE_LOOKUP_MAX_COUNT=${FULL_MDP_SPARSE_LOOKUP_MAX_COUNT:-65536}
export FULL_MDP_RAM_DENSE_MIN_COUNT=${FULL_MDP_RAM_DENSE_MIN_COUNT:-512}
export FULL_MDP_RAM_DENSE_HEAT_MIN=${FULL_MDP_RAM_DENSE_HEAT_MIN:-64}
export FULL_MDP_RAM_DENSE_MAX_BLOCKS=${FULL_MDP_RAM_DENSE_MAX_BLOCKS:-160}
export ONE_LOOP_MDP_BLOCK_CHUNK=${ONE_LOOP_MDP_BLOCK_CHUNK:-64}
export ONE_LOOP_MDP_BATCH_CHUNKS=${ONE_LOOP_MDP_BATCH_CHUNKS:-$((RAYON_NUM_THREADS * 4))}
export FULL_MDP_BATCH_CHUNKS=${FULL_MDP_BATCH_CHUNKS:-$((RAYON_NUM_THREADS * 4))}
export FULL_MDP_PROGRESS_EVERY_BATCHES=${FULL_MDP_PROGRESS_EVERY_BATCHES:-20}
export FULL_MDP_PREWARM_FIELD_TARGETS=${FULL_MDP_PREWARM_FIELD_TARGETS:-0}
export FULL_MDP_SYNC_LAYER=${FULL_MDP_SYNC_LAYER:-0}
export FULL_MDP_CACHE_STATS=${FULL_MDP_CACHE_STATS:-0}
export FULL_MDP_ACTION_STATS=${FULL_MDP_ACTION_STATS:-0}
export FULL_MDP_WORK_DIR=${FULL_MDP_WORK_DIR:-"$SCRATCH/seed-rolling"}
if [[ -z ${FULL_MDP_LOOKUP_GROUP_CHUNKS+x} ]]; then
  if [[ "$FULL_MDP_RAM_NEXT_LAYER" == 1 ]]; then
    export FULL_MDP_LOOKUP_GROUP_CHUNKS=4
  else
    export FULL_MDP_LOOKUP_GROUP_CHUNKS=1
  fi
fi

CACHE_GIB=${CACHE_GIB:-16}
ONE_LOOP_CACHE_GIB=${ONE_LOOP_CACHE_GIB:-8}
SEED_CACHE_GIB=${SEED_CACHE_GIB:-32}
MDP_GAMMA=${MDP_GAMMA:-1.0}
ONE_LOOP_SCALE=${ONE_LOOP_SCALE:-5040}
# The initialization seed only needs layers 0..3.  Layer 4 is treated as a
# full-valued boundary (5040/5040), matching the seed used by the reference run.
ONE_LOOP_START_LAYER=${ONE_LOOP_START_LAYER:-3}
ONE_LOOP_BASE_FULL_NEXT_LAYER=${ONE_LOOP_BASE_FULL_NEXT_LAYER:-4}
SEED_MAX_ITERS=${SEED_MAX_ITERS:-1}
SEED_TOL_ABS=${SEED_TOL_ABS:--1}
SEED_TOL_MEAN=${SEED_TOL_MEAN:--1}
MAX_PI_ROUNDS=${MAX_PI_ROUNDS:-8}
GMRES_MAX_ITERS=${GMRES_MAX_ITERS:-300}
GMRES_REL_TOL=${GMRES_REL_TOL:-1e-6}

validate_policy_backend() {
  case "$PI_POLICY_EVAL_BACKEND" in
    gmres) ;;
    reduced)
      case "$PI_REDUCED_OPERATOR" in
        sweep|dag) ;;
        *) echo "invalid PI_REDUCED_OPERATOR=$PI_REDUCED_OPERATOR; expected sweep or dag" >&2; exit 2 ;;
      esac
      case "$PI_REDUCED_METHOD" in
        gmres) ;;
        deflated)
          [[ "$PI_REDUCED_OPERATOR" == dag ]] || {
            echo "PI_REDUCED_METHOD=deflated requires PI_REDUCED_OPERATOR=dag" >&2; exit 2;
          }
          ;;
        *) echo "invalid PI_REDUCED_METHOD=$PI_REDUCED_METHOD; expected gmres or deflated" >&2; exit 2 ;;
      esac
      python3 - "$MDP_GAMMA" <<'PY'
import math
import sys

try:
    gamma = float(sys.argv[1])
except ValueError:
    gamma = float('nan')
if not math.isfinite(gamma) or gamma != 1.0:
    sys.exit(f"PI_POLICY_EVAL_BACKEND=reduced requires finite MDP_GAMMA=1; got {sys.argv[1]!r}")
PY
      ;;
    *) echo "invalid PI_POLICY_EVAL_BACKEND=$PI_POLICY_EVAL_BACKEND; expected gmres or reduced" >&2; exit 2 ;;
  esac
}

need_file() {
  [[ -s "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }
}

ensure_solver() {
  if [[ -x "$BIN" ]]; then
    return 0
  fi
  if [[ "$BIN" != "$DEFAULT_BIN" ]]; then
    echo "custom SOLVER_BIN is not executable: $BIN" >&2
    exit 1
  fi
  cargo build --manifest-path "$ROOT/Cargo.toml" --release -p zxcl
}

layer_pair_exists() {
  [[ -s "$1.field_index.bin" && -s "$1.chains.u24.bin" ]]
}

pruned_complete() {
  local k
  for k in $(seq 0 9); do layer_pair_exists "$PRUNED/layer${k}q_pruned" || return 1; done
}

policies_equal() {
  local a=$1 b=$2 k
  for k in $(seq 0 9); do
    cmp -s "$a/layer${k}.policy.u16.bin" "$b/layer${k}.policy.u16.bin" || return 1
  done
}

policy_complete() {
  local dir=$1 k
  for k in $(seq 0 9); do [[ -s "$dir/layer${k}.policy.u16.bin" ]] || return 1; done
}

show_env() {
  cat <<EOF
ROOT=$ROOT
ZXCL_DATA_DIR=$DATA
ZXCL_SCRATCH_DIR=$SCRATCH
GRAPH_BIN=$GRAPH
PRUNED_DIR=$PRUNED
ONE_LOOP_DIR=$ONE_LOOP
SEED_DIR=$SEED_DIR
RESULTS_DIR=$RESULTS
SOLVER_BIN=$BIN
VSTAR_LAYER0=$VSTAR_INPUT
RAYON_NUM_THREADS=$RAYON_NUM_THREADS
CACHE_GIB=$CACHE_GIB
ONE_LOOP_CACHE_GIB=$ONE_LOOP_CACHE_GIB
SEED_CACHE_GIB=$SEED_CACHE_GIB
MDP_GAMMA=$MDP_GAMMA
ONE_LOOP_SCALE=$ONE_LOOP_SCALE
ONE_LOOP_START_LAYER=$ONE_LOOP_START_LAYER
ONE_LOOP_BASE_FULL_NEXT_LAYER=$ONE_LOOP_BASE_FULL_NEXT_LAYER
SEED_MAX_ITERS=$SEED_MAX_ITERS
SEED_TOL_ABS=$SEED_TOL_ABS
SEED_TOL_MEAN=$SEED_TOL_MEAN
MAX_PI_ROUNDS=$MAX_PI_ROUNDS
GMRES_MAX_ITERS=$GMRES_MAX_ITERS
GMRES_REL_TOL=$GMRES_REL_TOL
FULL_MDP_RAM_NEXT_LAYER=$FULL_MDP_RAM_NEXT_LAYER
FULL_MDP_RAM_NEXT_LAYER_MAX_GIB=$FULL_MDP_RAM_NEXT_LAYER_MAX_GIB
FULL_MDP_RAM_PAIR_MAX_GIB=$FULL_MDP_RAM_PAIR_MAX_GIB
FULL_MDP_L9_DIRECT=$FULL_MDP_L9_DIRECT
FULL_MDP_MEMORY_MAX_GIB=${FULL_MDP_MEMORY_MAX_GIB:-32}
FULL_MDP_MEMORY_RESERVE_GIB=${FULL_MDP_MEMORY_RESERVE_GIB:-auto}
FULL_MDP_TASK_MAX_STATES=${FULL_MDP_TASK_MAX_STATES:-2000000}
FULL_MDP_TASK_MAX_MIB=${FULL_MDP_TASK_MAX_MIB:-64}
FULL_MDP_RAM_BITMAP_MIN_COUNT=${FULL_MDP_RAM_BITMAP_MIN_COUNT:-0}
PI_POLICY_EVAL_BACKEND=$PI_POLICY_EVAL_BACKEND
PI_REDUCED_OPERATOR=$PI_REDUCED_OPERATOR
PI_REDUCED_METHOD=$PI_REDUCED_METHOD
PI_REDUCED_RESTART=${PI_REDUCED_RESTART:-40}
PI_DAG_MAX_STATES=${PI_DAG_MAX_STATES:-2000000}
PI_DAG_MAX_NODES=${PI_DAG_MAX_NODES:-5000000}
PI_DAG_MAX_EDGES=${PI_DAG_MAX_EDGES:-25000000}
PI_DAG_FIELD_CACHE_MIB=${PI_DAG_FIELD_CACHE_MIB:-64}
PI_GMRES_BASIS_MAX_GIB=$PI_GMRES_BASIS_MAX_GIB
PI_ACCEPT_BELLMAN_INF=$PI_ACCEPT_BELLMAN_INF
PI_GMRES_AITKEN=$PI_GMRES_AITKEN
PI_GMRES_SNAPSHOT_EVERY=$PI_GMRES_SNAPSHOT_EVERY
FULL_MDP_MMAP_SPILL=$FULL_MDP_MMAP_SPILL
FULL_MDP_BLOCK_CHUNK=$FULL_MDP_BLOCK_CHUNK
FULL_MDP_SPARSE_LOOKUP_MAX_COUNT=$FULL_MDP_SPARSE_LOOKUP_MAX_COUNT
FULL_MDP_RAM_DENSE_MIN_COUNT=$FULL_MDP_RAM_DENSE_MIN_COUNT
FULL_MDP_RAM_DENSE_HEAT_MIN=$FULL_MDP_RAM_DENSE_HEAT_MIN
FULL_MDP_RAM_DENSE_MAX_BLOCKS=$FULL_MDP_RAM_DENSE_MAX_BLOCKS
ONE_LOOP_MDP_BLOCK_CHUNK=$ONE_LOOP_MDP_BLOCK_CHUNK
ONE_LOOP_MDP_BATCH_CHUNKS=$ONE_LOOP_MDP_BATCH_CHUNKS
FULL_MDP_BATCH_CHUNKS=$FULL_MDP_BATCH_CHUNKS
FULL_MDP_LOOKUP_GROUP_CHUNKS=$FULL_MDP_LOOKUP_GROUP_CHUNKS
FULL_MDP_PROGRESS_EVERY_BATCHES=$FULL_MDP_PROGRESS_EVERY_BATCHES
FULL_MDP_PREWARM_FIELD_TARGETS=$FULL_MDP_PREWARM_FIELD_TARGETS
FULL_MDP_SYNC_LAYER=$FULL_MDP_SYNC_LAYER
FULL_MDP_CACHE_STATS=$FULL_MDP_CACHE_STATS
FULL_MDP_ACTION_STATS=$FULL_MDP_ACTION_STATS
FULL_MDP_WORK_DIR=$FULL_MDP_WORK_DIR
EOF
}

MANIFEST="$ROOT/scripts/solve_manifest.py"
HASH_CACHE=${SOLVE_HASH_CACHE_DIR:-"$SCRATCH/solve-identities"}
CONTEXT=""
INPUT_SNAPSHOT=""

manifest() {
  # A solve may take days. Reusing the startup context after an input changes
  # would certify results for data the solver did not consistently read.
  case $1 in
    check|record|identity|copy|snapshot)
      if [[ -n "$INPUT_SNAPSHOT" ]]; then
        python3 "$MANIFEST" guard "$INPUT_SNAPSHOT" || return $?
      fi
      ;;
  esac
  python3 "$MANIFEST" "$@"
}
stage_key() { manifest key "$CONTEXT" "$@"; }

record_completion() {
  local dependencies=$1 completion=$2
  shift 2
  if [[ -n "$dependencies" ]]; then manifest guard "$dependencies"; fi
  manifest record "$completion" "$@"
  # Also catch a dependency changing while output sizes/finite values were read.
  if ! manifest guard "$INPUT_SNAPSHOT" ||
      { [[ -n "$dependencies" ]] && ! manifest guard "$dependencies"; }; then
    rm -f "$completion"
    return 1
  fi
}

initialize_run() {
  validate_policy_backend
  ensure_solver
  need_file "$GRAPH"
  pruned_complete || { echo "forward-pruned MDP is incomplete: $PRUNED" >&2; exit 1; }
  # The first run reads every input once; later runs use content hashes cached
  # against device/inode/size/mtime/ctime. Separate graph variants stay separate.
  local inputs
  INPUT_SNAPSHOT=$(manifest snapshot-inputs "$GRAPH" "$PRUNED" "$BIN")
  inputs=$(manifest context "$GRAPH" "$PRUNED" "$BIN" "$HASH_CACHE")
  manifest guard "$INPUT_SNAPSHOT"
  CONTEXT=$(manifest key "$inputs" "gamma=$MDP_GAMMA")
}

seed_key() {
  stage_key seed "$ONE_LOOP_SCALE" "$ONE_LOOP_START_LAYER" "$ONE_LOOP_BASE_FULL_NEXT_LAYER" \
    "$ONE_LOOP_CACHE_GIB" "$SEED_CACHE_GIB" "$SEED_MAX_ITERS" "$SEED_TOL_ABS" "$SEED_TOL_MEAN"
}

build_seed() (
  # solve-full-mdp-l0-f32 intentionally resumes files in its output directory.
  # An uncertified seed must instead start from the requested one-loop values,
  # even when an older configuration left same-sized input/checkpoint files.
  local completion=$1 key=$2 build
  build=$(mktemp -d "$SEED_DIR/.seed-build.XXXXXX")
  trap 'rm -rf -- "$build"' EXIT
  manifest guard "$INPUT_SNAPSHOT"
  "$BIN" solve-one-loop-mdp-u16 \
    "$GRAPH" "$PRUNED" "$ONE_LOOP" "$ONE_LOOP_SCALE" "$ONE_LOOP_CACHE_GIB" \
    "$ONE_LOOP_START_LAYER" "$ONE_LOOP_BASE_FULL_NEXT_LAYER"
  manifest guard "$INPUT_SNAPSHOT"
  "$BIN" solve-full-mdp-l0-f32 "$GRAPH" "$PRUNED" "$build" \
    "$MDP_GAMMA" "$SEED_MAX_ITERS" "$SEED_CACHE_GIB" "one_loop:$ONE_LOOP" \
    "$SEED_TOL_ABS" "$SEED_TOL_MEAN"
  # Validate before replacing an accepted seed. A failed/interrupted build leaves
  # its old value and certificate intact; retries never resume the scratch files.
  record_completion "" "$build/seed.complete.json" "$CONTEXT" "$key" "$PRUNED" seed "$build/layer0.values.f32.bin"
  manifest copy "$build/layer0.values.f32.bin" "$SEED_DIR/layer0.values.f32.bin"
  record_completion "" "$completion" "$CONTEXT" "$key" "$PRUNED" seed "$SEED_DIR/layer0.values.f32.bin"
)

one_loop() {
  initialize_run
  local completion="$SEED_DIR/seed.complete.json" key
  key=$(seed_key)
  if manifest check "$completion" "$CONTEXT" "$key"; then
    echo "validated one-loop seed already exists"
    return 0
  fi
  mkdir -p "$ONE_LOOP" "$SEED_DIR"
  build_seed "$completion" "$key"
}

solve() {
  # An output file alone never certifies a successful seed or policy evaluation.
  one_loop
  local seed="$SEED_DIR/layer0.values.f32.bin"
  mkdir -p "$RESULTS" "$SCRATCH/pi"

  local previous_policy="" previous_value="" round policy value emit_dir
  local key completion improve_from input_id policy_id warm_start dependencies boundary_snapshot
  for round in $(seq 0 $((MAX_PI_ROUNDS - 1))); do
    policy="$RESULTS/policy_$round"
    value="$RESULTS/v_pi${round}.f32.bin"
    emit_dir="$SCRATCH/pi/emit_$round"
    mkdir -p "$policy" "$emit_dir"

    improve_from=${previous_value:-$seed}
    boundary_snapshot=$(manifest snapshot "$improve_from")
    input_id=$(manifest identity "$improve_from" "$HASH_CACHE")
    manifest guard "$boundary_snapshot"
    key=$(stage_key policy "$round" "$input_id" "$CACHE_GIB")
    completion="$policy/policy.complete.json"
    if manifest check "$completion" "$CONTEXT" "$key"; then
      echo "validated policy round $round already exists"
    else
      rm -f "$completion"
      dependencies=$boundary_snapshot
      FULL_MDP_POLICY_ONLY=1 "$BIN" emit-full-mdp-layers-from-l0 \
        "$GRAPH" "$PRUNED" "$improve_from" "$emit_dir" "$MDP_GAMMA" "$CACHE_GIB" "$policy"
      record_completion "$dependencies" "$completion" "$CONTEXT" "$key" "$PRUNED" policy "$policy"
    fi
    policy_id=$(manifest identity "$completion" "$HASH_CACHE")

    dependencies=""
    if [[ -n "$previous_policy" ]]; then
      dependencies=$(manifest snapshot "$previous_value" "$previous_policy"/layer*.policy.u16.bin "$policy"/layer*.policy.u16.bin)
    fi
    if [[ -n "$previous_policy" ]] && policies_equal "$previous_policy" "$policy"; then
      key=$(stage_key stable "$policy_id" "$input_id")
      if manifest check "$RESULTS/stable.complete.json" "$CONTEXT" "$key"; then
        echo "validated policy stable at improvement round $round"
        return 0
      fi
      rm -f "$RESULTS/stable.complete.json"
      manifest copy "$previous_value" "$RESULTS/vstar_boundary.f32.bin"
      printf '%s\n' "$previous_policy" > "$RESULTS/STABLE_POLICY.pending"
      mv "$RESULTS/STABLE_POLICY.pending" "$RESULTS/STABLE_POLICY"
      printf '%s\n' "$round" > "$RESULTS/STABLE_ROUND.pending"
      mv "$RESULTS/STABLE_ROUND.pending" "$RESULTS/STABLE_ROUND"
      record_completion "$dependencies" "$RESULTS/stable.complete.json" "$CONTEXT" "$key" "$PRUNED" stable \
        "$RESULTS/vstar_boundary.f32.bin" "$previous_policy" "$RESULTS/STABLE_POLICY" "$RESULTS/STABLE_ROUND"
      echo "policy stable at improvement round $round"
      return 0
    fi

    warm_start=$improve_from
    key=$(stage_key evaluation "$round" "$policy_id" "$input_id" "$GMRES_MAX_ITERS" "$GMRES_REL_TOL" "$CACHE_GIB")
    completion="$value.complete.json"
    if manifest check "$completion" "$CONTEXT" "$key"; then
      echo "validated evaluated value round $round already exists"
    else
      rm -f "$completion"
      dependencies=$(manifest snapshot "$warm_start" "$policy"/layer*.policy.u16.bin)
      "$BIN" "pi-policy-eval-$PI_POLICY_EVAL_BACKEND" \
        "$GRAPH" "$PRUNED" "$policy" "$warm_start" "$value" "$SCRATCH/pi/eval_$round" \
        "$MDP_GAMMA" "$GMRES_MAX_ITERS" "$GMRES_REL_TOL" "$CACHE_GIB"
      record_completion "$dependencies" "$completion" "$CONTEXT" "$key" "$PRUNED" value "$value"
    fi

    previous_policy=$policy
    previous_value=$value
  done

  echo "policy did not stabilize within MAX_PI_ROUNDS=$MAX_PI_ROUNDS" >&2
  echo "increase MAX_PI_ROUNDS and retain the existing policy/value directories" >&2
  exit 1
}

finalize() {
  initialize_run
  manifest check "$RESULTS/stable.complete.json" "$CONTEXT" - || {
    echo "no validated stable result for these inputs; run solve first" >&2; exit 1;
  }
  local vstar="$RESULTS/vstar_boundary.f32.bin" stable_policy key completion dependencies
  stable_policy=$(head -1 "$RESULTS/STABLE_POLICY")
  key=$(stage_key finalize "$(manifest identity "$RESULTS/stable.complete.json" "$HASH_CACHE")" "$CACHE_GIB")
  completion="$RESULTS/final.complete.json"
  if manifest check "$completion" "$CONTEXT" "$key"; then
    echo "validated final values and policy already exist"
    return 0
  fi
  rm -f "$completion"
  dependencies=$(manifest snapshot "$vstar" "$stable_policy"/layer*.policy.u16.bin "$RESULTS/STABLE_POLICY" "$RESULTS/STABLE_ROUND")
  mkdir -p "$RESULTS/vstar_full" "$RESULTS/policy_final"
  FULL_MDP_POLICY_ONLY=0 "$BIN" emit-full-mdp-layers-from-l0 \
    "$GRAPH" "$PRUNED" "$vstar" "$RESULTS/vstar_full" \
    "$MDP_GAMMA" "$CACHE_GIB" "$RESULTS/policy_final"
  if ! policies_equal "$stable_policy" "$RESULTS/policy_final"; then
    echo "final greedy policy differs from the stability certificate" >&2
    exit 1
  fi
  record_completion "$dependencies" "$completion" "$CONTEXT" "$key" "$PRUNED" full "$RESULTS/vstar_full" "$RESULTS/policy_final"
  echo "V* and the stable policy were materialized for all ten layers"
}

materialize_boundary() (
  local source_snapshot=$1 completion=$2 key=$3 build dependencies
  mkdir -p "$SCRATCH" "$RESULTS/vstar_full" "$RESULTS/policy_final"
  build=$(mktemp -d "$SCRATCH/.from-vstar.XXXXXX")
  trap 'rm -rf -- "$build"' EXIT
  # The supplied file may itself be vstar_full/layer0.values.f32.bin, which the
  # emitter overwrites. Freeze its content before opening any output files.
  manifest copy "$VSTAR_INPUT" "$build/boundary.f32.bin"
  record_completion "$source_snapshot" "$build/input.complete.json" "$CONTEXT" "$key" "$PRUNED" value "$build/boundary.f32.bin"
  dependencies=$(manifest snapshot "$build/boundary.f32.bin")
  FULL_MDP_POLICY_ONLY=0 "$BIN" emit-full-mdp-layers-from-l0 \
    "$GRAPH" "$PRUNED" "$build/boundary.f32.bin" "$RESULTS/vstar_full" \
    "$MDP_GAMMA" "$CACHE_GIB" "$RESULTS/policy_final"
  manifest copy "$build/boundary.f32.bin" "$RESULTS/vstar_boundary.f32.bin"
  manifest copy "$build/boundary.f32.bin" "$RESULTS/vstar_full/layer0.values.f32.bin"
  policy_complete "$RESULTS/policy_final" || {
    echo "failed to materialize the complete greedy policy" >&2; exit 1;
  }
  record_completion "$dependencies" "$completion" "$CONTEXT" "$key" "$PRUNED" full-boundary "$RESULTS/vstar_full" "$RESULTS/policy_final" "$RESULTS/vstar_boundary.f32.bin"
)

from_vstar() {
  initialize_run
  need_file "$VSTAR_INPUT"
  local actual_size expected_size key completion dependencies
  dependencies=$(manifest snapshot "$VSTAR_INPUT")
  actual_size=$(wc -c < "$VSTAR_INPUT")
  expected_size=$(( $(manifest states "$PRUNED" 0) * 4 ))
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "layer-0 V* size mismatch: got $actual_size bytes, expected $expected_size" >&2
    exit 1
  fi
  if [[ "$VSTAR_INPUT" == "$BUNDLED_VSTAR" ]]; then
    echo "$VSTAR_SHA256  $VSTAR_INPUT" | sha256sum --check --status || {
      echo "bundled layer-0 V* checksum mismatch" >&2; exit 1;
    }
    echo "$GRAPH_SHA256  $GRAPH" | sha256sum --check --status || {
      echo "graph.bin does not match the graph used by the bundled layer-0 V*" >&2; exit 1;
    }
  fi
  key=$(stage_key from-vstar "$(manifest identity "$VSTAR_INPUT" "$HASH_CACHE")" "$CACHE_GIB")
  manifest guard "$dependencies"
  completion="$RESULTS/from-vstar.complete.json"
  if manifest check "$completion" "$CONTEXT" "$key"; then
    echo "validated supplied boundary materialization already exists"
    return 0
  fi
  rm -f "$completion"
  materialize_boundary "$dependencies" "$completion" "$key"
  echo "supplied layer-0 V* was materialized into all ten layers"
}

all() {
  one_loop
  solve
  finalize
}

case ${1:-env} in
  env) show_env ;;
  one-loop) one_loop ;;
  solve) solve ;;
  finalize) finalize ;;
  from-vstar) from_vstar ;;
  all) all ;;
  *) echo "usage: $0 {env|one-loop|solve|finalize|from-vstar|all}" >&2; exit 2 ;;
esac
