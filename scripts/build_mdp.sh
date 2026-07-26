#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${ZXCL_DATA_DIR:-"$ROOT/data"}
SCRATCH=${ZXCL_SCRATCH_DIR:-"$ROOT/work"}
GRAPH=${GRAPH_BIN:-"$DATA/graph.bin"}
RGRAPH=${RGRAPH_BIN:-"$DATA/rgraph.bin"}
QGRAPH=${QGRAPH_PREFIX:-"$DATA/qgraph/qchain"}
BACKWARD=${BACKWARD_DIR:-"$DATA/backward"}
PRUNED=${PRUNED_DIR:-"$DATA/forward_pruned"}
DEFAULT_BIN="$ROOT/target/release/zxcl"
BIN=${SOLVER_BIN:-"$DEFAULT_BIN"}

export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-20}

LIVE_LIMIT=${LIVE_LIMIT:-600000000}
PER_FIELD_LIMIT=${PER_FIELD_LIMIT:-3000000}
MAX_SPILLS=${MAX_SPILLS:-2}
PRUNE_SHARDS=${PRUNE_SHARDS:-4096}
PRUNE_HOT_CAP_GIB=${PRUNE_HOT_CAP_GIB:-64}
PRUNE_VEC_TO_BITSET_THRESHOLD=${PRUNE_VEC_TO_BITSET_THRESHOLD:-8192}
PRUNE_TARGET_BUCKET_SIZE=${PRUNE_TARGET_BUCKET_SIZE:-65536}
PRUNE_REFINE_PASSES=${PRUNE_REFINE_PASSES:-4}

export FWD_PRUNE_MAX_OPEN_SHARDS=${FWD_PRUNE_MAX_OPEN_SHARDS:-256}
export FWD_PRUNE_CHUNK_BLOCKS=${FWD_PRUNE_CHUNK_BLOCKS:-64}
export FWD_PRUNE_DIRECT_FULL_BITSET_MAX_NEXT_LAYER=${FWD_PRUNE_DIRECT_FULL_BITSET_MAX_NEXT_LAYER:-3}
export FWD_PRUNE_TARGET_OWNED=${FWD_PRUNE_TARGET_OWNED:-1}
export FWD_PRUNE_TARGET_LOCAL_PAR_BATCH=${FWD_PRUNE_TARGET_LOCAL_PAR_BATCH:-20}

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

show_env() {
  cat <<EOF
ROOT=$ROOT
ZXCL_DATA_DIR=$DATA
ZXCL_SCRATCH_DIR=$SCRATCH
GRAPH_BIN=$GRAPH
RGRAPH_BIN=$RGRAPH
QGRAPH_PREFIX=$QGRAPH
BACKWARD_DIR=$BACKWARD
PRUNED_DIR=$PRUNED
SOLVER_BIN=$BIN
RAYON_NUM_THREADS=$RAYON_NUM_THREADS
LIVE_LIMIT=$LIVE_LIMIT
PER_FIELD_LIMIT=$PER_FIELD_LIMIT
MAX_SPILLS=$MAX_SPILLS
PRUNE_SHARDS=$PRUNE_SHARDS
PRUNE_HOT_CAP_GIB=$PRUNE_HOT_CAP_GIB
PRUNE_VEC_TO_BITSET_THRESHOLD=$PRUNE_VEC_TO_BITSET_THRESHOLD
PRUNE_TARGET_BUCKET_SIZE=$PRUNE_TARGET_BUCKET_SIZE
PRUNE_REFINE_PASSES=$PRUNE_REFINE_PASSES
FWD_PRUNE_MAX_OPEN_SHARDS=$FWD_PRUNE_MAX_OPEN_SHARDS
FWD_PRUNE_CHUNK_BLOCKS=$FWD_PRUNE_CHUNK_BLOCKS
FWD_PRUNE_DIRECT_FULL_BITSET_MAX_NEXT_LAYER=$FWD_PRUNE_DIRECT_FULL_BITSET_MAX_NEXT_LAYER
FWD_PRUNE_TARGET_OWNED=$FWD_PRUNE_TARGET_OWNED
FWD_PRUNE_TARGET_LOCAL_PAR_BATCH=$FWD_PRUNE_TARGET_LOCAL_PAR_BATCH
EOF
}

prepare() {
  need_file "$GRAPH"
  mkdir -p "$(dirname "$RGRAPH")" "$(dirname "$QGRAPH")" "$SCRATCH"
  ensure_solver
  "$BIN" build-rgraph "$GRAPH" "$RGRAPH"
  "$BIN" build-qgraph "$QGRAPH"
}

backward() {
  ensure_solver
  need_file "$RGRAPH"
  need_file "$QGRAPH.qrevclass.bin"
  mkdir -p "$BACKWARD" "$SCRATCH/backward-seed" "$SCRATCH/backward"

  if ! layer_pair_exists "$BACKWARD/layer8q" || ! layer_pair_exists "$BACKWARD/layer9q"; then
    "$BIN" build-prefix-q "$RGRAPH" "$SCRATCH/backward-seed"
    local k suffix
    for k in 9 8; do
      for suffix in field_index.bin chains.u24.bin; do
        mv "$SCRATCH/backward-seed/layer${k}q_direct.$suffix" "$BACKWARD/layer${k}q.$suffix"
      done
    done
  fi

  local layer next
  for layer in 7 6 5 4 3 2 1 0; do
    next=$((layer + 1))
    if layer_pair_exists "$BACKWARD/layer${layer}q"; then
      echo "backward layer $layer already exists"
      continue
    fi
    mkdir -p "$SCRATCH/backward/layer${layer}"
    "$BIN" external-step-q-live-frontier-spill \
      "$RGRAPH" "$QGRAPH.qrevclass.bin" \
      "$BACKWARD/layer${next}q" "$BACKWARD/layer${layer}q" \
      "$SCRATCH/backward/layer${layer}" \
      "$LIVE_LIMIT" "$PER_FIELD_LIMIT" "$MAX_SPILLS"
  done
}

prune() {
  ensure_solver
  if pruned_complete; then
    echo "all forward-pruned layers already exist"
    return 0
  fi
  local k
  for k in $(seq 0 9); do layer_pair_exists "$BACKWARD/layer${k}q" || { echo "missing backward layer $k" >&2; exit 1; }; done
  mkdir -p "$PRUNED" "$SCRATCH/prune"
  "$BIN" forward-prune-q-frontier-target-shard \
    "$GRAPH" "$BACKWARD" "$PRUNED" "$SCRATCH/prune" \
    "$PRUNE_SHARDS" "$PRUNE_VEC_TO_BITSET_THRESHOLD" "$PRUNE_HOT_CAP_GIB" \
    "$PRUNE_TARGET_BUCKET_SIZE" "$PRUNE_REFINE_PASSES"
}

all() {
  prepare
  backward
  prune
}

case ${1:-env} in
  env) show_env ;;
  prepare) prepare ;;
  backward) backward ;;
  prune) prune ;;
  all) all ;;
  *) echo "usage: $0 {env|prepare|backward|prune|all}" >&2; exit 2 ;;
esac
