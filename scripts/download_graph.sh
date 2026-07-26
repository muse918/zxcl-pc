#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${ZXCL_DATA_DIR:-"$ROOT/data"}
OUT=${GRAPH_BIN:-"$DATA/graph.bin"}
URL=${GRAPH_URL:-"https://drive.usercontent.google.com/download?id=1XEYrDFhatN-McOcpTpWuAYq8ubyanJtq&export=download&confirm=t"}
EXPECTED_SIZE=510917451
SHA256=7dc434e3611684756911480d6d338d37bf8ca5ade263611d651fa8709e0294f4

mkdir -p "$(dirname "$OUT")"
if [[ -f "$OUT" ]]; then
  echo "using existing $OUT"
else
  echo "downloading graph.bin to $OUT"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$OUT.part" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$OUT.part" "$URL"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
  mv "$OUT.part" "$OUT"
fi

actual_size=$(wc -c < "$OUT")
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
  echo "graph.bin size mismatch: got $actual_size bytes, expected $EXPECTED_SIZE" >&2
  exit 1
fi

echo "$SHA256  $OUT" | sha256sum --check --status || {
  echo "graph.bin checksum mismatch: $OUT" >&2
  exit 1
}
echo "graph.bin verified: $OUT"
