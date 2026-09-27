#!/bin/bash
# Does the command-buffer guard tell a dead GPU from a fast one, both ways?
# One process per window cap, same binary, same model, tiny measurement so the
# run is about correctness and not throughput: cap 32 is the shipped arm that
# executed everything, cap 6 is the cliff arm whose buffers the driver drops.
set -uo pipefail
BIN="/private/tmp/allpaka-guard/target/release/allpaka"
ROOT="/Users/themoretheless/Documents/Sources/allpaka"
M="$ROOT/models/qwen3-235b-a22b-instruct-2507-Q2_K_XL.gguf"
OUT="$ROOT/docs/benchmarks/2026-09-25-dead-gpu-guard"
for gib in 32 6; do
  ALLPAKA_GPU_WINDOW_GIB=$gib \
  ALLPAKA_BENCH_PP=64 ALLPAKA_BENCH_TG=8 ALLPAKA_BENCH_SKIP_MTP=1 \
  ALLPAKA_BENCH_REPORT="$OUT/guard-cap${gib}-report.json" \
    "$BIN" bench --engine "$M" > "$OUT/guard-cap${gib}-log.txt" 2>&1
  code=$?
  echo "cap ${gib}GiB $(date -u +%TH%MZ) exit $code" \
    "$(grep -o 'in [0-9]* windows' "$OUT/guard-cap${gib}-log.txt" | head -1)" \
    "$(grep -o 'prefill argmax=[0-9]*' "$OUT/guard-cap${gib}-log.txt")" \
    "$(grep -c 'command buffer failed' "$OUT/guard-cap${gib}-log.txt")" logged
done
