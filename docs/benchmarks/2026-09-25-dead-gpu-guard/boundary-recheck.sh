#!/bin/bash
# Is the window-span cliff the same boundary it was at 19:20Z, when the
# machine had 84% free RAM and no other session? Correctness observables
# only - exit code, window census, argmax - never the rates printed here.
set -uo pipefail
BIN="/private/tmp/allpaka-guard/target/release/allpaka"
ROOT="/Users/themoretheless/Documents/Sources/allpaka"
M="$ROOT/models/qwen3-235b-a22b-instruct-2507-Q2_K_XL.gguf"
OUT="$ROOT/docs/benchmarks/2026-09-25-dead-gpu-guard"
for gib in 12 10 8; do
  f="$OUT/recheck-cap${gib}"
  ALLPAKA_GPU_WINDOW_GIB=$gib \
  ALLPAKA_BENCH_PP=64 ALLPAKA_BENCH_TG=8 ALLPAKA_BENCH_SKIP_MTP=1 \
  ALLPAKA_BENCH_REPORT="$f-report.json" \
    "$BIN" bench --engine "$M" > "$f-log.txt" 2>&1
  code=$?
  echo "cap ${gib}GiB $(date -u +%H:%M:%SZ) exit $code" \
    "$(grep -o 'in [0-9]* windows' "$f-log.txt" | head -1)" \
    "$(grep -o 'prefill argmax=[0-9]*' "$f-log.txt")" \
    "$(grep -o 'failed during [a-z ]*: [0-9]*' "$f-log.txt" | head -1)" \
    "free $(( $(vm_stat | awk '/Pages free/{print $3}' | tr -d '.') * 16384 / 1073741824 ))GiB"
done
