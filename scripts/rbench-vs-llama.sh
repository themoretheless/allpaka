#!/usr/bin/env bash
# Paired allpaka vs llama.cpp throughput via `allpaka rbench` (rbench schema).
set -euo pipefail

model=${1:?usage: rbench-vs-llama.sh MODEL [PP=480] [TG=32] [REPEATS=5]}
pp=${2:-480}
tg=${3:-32}
repeats=${4:-5}
root=$(cd "$(dirname "$0")/.." && pwd)
allpaka=${ALLPAKA_BIN:-$root/target/release/allpaka}
llama=${LLAMA_BENCH:-llama-bench}
out=${BENCH_OUTPUT_DIR:-$root/.rbench/llama-compare-$(date +%s)}

[[ -x "$allpaka" ]] || {
  echo "building release allpaka..." >&2
  (cd "$root" && cargo build --release -p allpaka-cli)
  allpaka=$root/target/release/allpaka
}
command -v "$llama" >/dev/null || [[ -x "$llama" ]]

exec "$allpaka" rbench "$model" \
  --pp "$pp" \
  --tg "$tg" \
  --repeats "$repeats" \
  --allpaka "$allpaka" \
  --llama-bench "$llama" \
  --out "$out"
