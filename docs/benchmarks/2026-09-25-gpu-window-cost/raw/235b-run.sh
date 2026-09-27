#!/bin/bash
# Bisect the window cap at which the 235B's logits go uniform (prefill argmax
# landing on the last row = all logits tied or NaN). One process per cap, warm
# page cache, everything else identical to the census runs.
set -uo pipefail
ROOT="/Users/themoretheless/Documents/Sources/allpaka"
OUT="/Users/themoretheless/Documents/Sources/allpaka/.airbug-bench/235b-window-bisect"
M="$ROOT/models/qwen3-235b-a22b-instruct-2507-Q2_K_XL.gguf"
for gib in 32 24 16 12 10 8 6 4; do
  export ALLPAKA_GPU_WINDOW_GIB=$gib
  ALLPAKA_BENCH_PP=64 ALLPAKA_BENCH_TG=8 ALLPAKA_BENCH_SKIP_MTP=1 \
  ALLPAKA_BENCH_REPORT="$OUT/cap${gib}-report.json" \
    "$ROOT/target/release/allpaka" bench --engine "$M" > "$OUT/cap${gib}-log.txt" 2>&1
  echo "cap ${gib}GiB $(date -u +%T) exit $? $(grep -o 'in [0-9]* windows' $OUT/cap${gib}-log.txt | head -1) $(grep -o 'prefill argmax=[0-9]*' $OUT/cap${gib}-log.txt)"
done
