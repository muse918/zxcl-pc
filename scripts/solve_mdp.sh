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

one_loop() {
  ensure_solver
  need_file "$GRAPH"
  pruned_complete || { echo "forward-pruned MDP is incomplete: $PRUNED" >&2; exit 1; }
  if [[ -s "$SEED_DIR/layer0.values.f32.bin" ]]; then
    echo "one-loop seed already exists"
    return 0
  fi
  mkdir -p "$ONE_LOOP" "$SEED_DIR"
  "$BIN" solve-one-loop-mdp-u16 \
    "$GRAPH" "$PRUNED" "$ONE_LOOP" "$ONE_LOOP_SCALE" "$ONE_LOOP_CACHE_GIB" \
    "$ONE_LOOP_START_LAYER" "$ONE_LOOP_BASE_FULL_NEXT_LAYER"
  "$BIN" solve-full-mdp-l0-f32 "$GRAPH" "$PRUNED" "$SEED_DIR" \
    "$MDP_GAMMA" "$SEED_MAX_ITERS" "$SEED_CACHE_GIB" "one_loop:$ONE_LOOP" \
    "$SEED_TOL_ABS" "$SEED_TOL_MEAN"
  need_file "$SEED_DIR/layer0.values.f32.bin"
}

solve() {
  ensure_solver
  pruned_complete || { echo "forward-pruned MDP is incomplete: $PRUNED" >&2; exit 1; }
  local seed="$SEED_DIR/layer0.values.f32.bin"
  need_file "$seed"
  mkdir -p "$RESULTS" "$SCRATCH/pi"

  local previous_policy="" previous_value="" round policy value emit_dir
  for round in $(seq 0 $((MAX_PI_ROUNDS - 1))); do
    policy="$RESULTS/policy_$round"
    value="$RESULTS/v_pi${round}.f32.bin"
    emit_dir="$SCRATCH/pi/emit_$round"
    mkdir -p "$policy" "$emit_dir"

    local improve_from=${previous_value:-$seed}
    if policy_complete "$policy"; then
      echo "policy round $round already exists"
    else
      "$BIN" emit-full-mdp-layers-from-l0 \
        "$GRAPH" "$PRUNED" "$improve_from" "$emit_dir" "$MDP_GAMMA" "$CACHE_GIB" "$policy"
    fi

    if [[ -n "$previous_policy" ]] && policies_equal "$previous_policy" "$policy"; then
      cp "$previous_value" "$RESULTS/vstar_boundary.f32.bin"
      printf '%s\n' "$previous_policy" > "$RESULTS/STABLE_POLICY"
      printf '%s\n' "$round" > "$RESULTS/STABLE_ROUND"
      echo "policy stable at improvement round $round"
      return 0
    fi

    if [[ -s "$value" ]]; then
      echo "evaluated value round $round already exists"
    else
      "$BIN" pi-policy-eval-gmres \
        "$GRAPH" "$PRUNED" "$policy" "$seed" "$value" "$SCRATCH/pi/eval_$round" \
        "$MDP_GAMMA" "$GMRES_MAX_ITERS" "$GMRES_REL_TOL" "$CACHE_GIB"
    fi

    previous_policy=$policy
    previous_value=$value
  done

  echo "policy did not stabilize within MAX_PI_ROUNDS=$MAX_PI_ROUNDS" >&2
  echo "increase MAX_PI_ROUNDS and retain the existing policy/value directories" >&2
  exit 1
}

finalize() {
  ensure_solver
  pruned_complete || { echo "forward-pruned MDP is incomplete: $PRUNED" >&2; exit 1; }
  local vstar="$RESULTS/vstar_boundary.f32.bin"
  need_file "$vstar"
  need_file "$RESULTS/STABLE_POLICY"
  local stable_policy
  stable_policy=$(head -1 "$RESULTS/STABLE_POLICY")
  policy_complete "$stable_policy" || { echo "stable policy is incomplete: $stable_policy" >&2; exit 1; }
  mkdir -p "$RESULTS/vstar_full" "$RESULTS/policy_final"

  "$BIN" emit-full-mdp-layers-from-l0 \
    "$GRAPH" "$PRUNED" "$vstar" "$RESULTS/vstar_full" \
    "$MDP_GAMMA" "$CACHE_GIB" "$RESULTS/policy_final"

  if ! policies_equal "$stable_policy" "$RESULTS/policy_final"; then
    echo "final greedy policy differs from the stability certificate" >&2
    exit 1
  fi

  echo "V* and the stable policy were materialized for all ten layers"
}

from_vstar() {
  ensure_solver
  need_file "$GRAPH"
  pruned_complete || { echo "forward-pruned MDP is incomplete: $PRUNED" >&2; exit 1; }
  need_file "$VSTAR_INPUT"

  local actual_size
  actual_size=$(wc -c < "$VSTAR_INPUT")
  if [[ "$actual_size" != "$VSTAR_SIZE" ]]; then
    echo "layer-0 V* size mismatch: got $actual_size bytes, expected $VSTAR_SIZE" >&2
    exit 1
  fi
  if [[ "$VSTAR_INPUT" == "$BUNDLED_VSTAR" ]]; then
    echo "$VSTAR_SHA256  $VSTAR_INPUT" | sha256sum --check --status || {
      echo "bundled layer-0 V* checksum mismatch" >&2
      exit 1
    }
    echo "$GRAPH_SHA256  $GRAPH" | sha256sum --check --status || {
      echo "graph.bin does not match the graph used by the bundled layer-0 V*" >&2
      exit 1
    }
  fi

  mkdir -p "$RESULTS/vstar_full" "$RESULTS/policy_final"
  cp "$VSTAR_INPUT" "$RESULTS/vstar_boundary.f32.bin"
  "$BIN" emit-full-mdp-layers-from-l0 \
    "$GRAPH" "$PRUNED" "$VSTAR_INPUT" "$RESULTS/vstar_full" \
    "$MDP_GAMMA" "$CACHE_GIB" "$RESULTS/policy_final"

  # The sweep's layer 0 is T(V*). Keep the canonical supplied boundary vector as
  # layer 0, while layers 1..9 and the greedy policy come from the sweep.
  cp "$VSTAR_INPUT" "$RESULTS/vstar_full/layer0.values.f32.bin"
  policy_complete "$RESULTS/policy_final" || {
    echo "failed to materialize the complete greedy policy" >&2
    exit 1
  }
  echo "bundled layer-0 V* was materialized into all ten layers"
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
