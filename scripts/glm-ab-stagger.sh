#!/bin/bash
# Cool-machine GLM A/B for mul_mv_id grid + SHARED_STAGGER.
# Must run outside Cursor agent (Metal=nil under seatbelt).
set -euo pipefail
export ALLPAKA_BENCH_SKIP_MTP=1 ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=32
export ALLPAKA_PROFILE=max-performance
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-/Users/themoretheless/Documents/Sources/allpaka/models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf}
BIN=${BIN:-$ROOT/target/release/allpaka}
OUT=${OUT:-$ROOT/.airbug-bench/glm-mvid-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

metal=$(swift -e 'import Metal; print(MTLCreateSystemDefaultDevice()?.name ?? "nil")')
echo "Metal: $metal" | tee "$OUT/metal.txt"
if [[ $metal == nil ]]; then
  echo "no Metal device — abort (do not trust CPU-fallback numbers)" >&2
  exit 1
fi

run_allpaka() {
  local label=$1
  shift
  echo "=== allpaka $label ==="
  env "$@" "$BIN" bench --engine "$MODEL" | tee "$OUT/allpaka-${label}.log"
}

{
  echo "=== mvbench (MV_ID on) ==="
  (cd "$ROOT" && cargo test -p allpaka-backend --release --test gpu_glm_mvbench -- --ignored --nocapture) \
    | tee "$OUT/mvbench-on.log" | rg 'GB/s|SKIP|idx |error|panic' || true

  run_allpaka mvid-on
  sleep 2
  run_allpaka mvid-off ALLPAKA_MV_ID=0
  sleep 2
  run_allpaka stagger-on ALLPAKA_SHARED_STAGGER=1
  sleep 2
  echo "=== llama ==="
  llama-bench -m "$MODEL" -p 480 -n 32 -r 1 -b 2048 -ub 512 | tee "$OUT/llama.log"
  echo "DONE -> $OUT"
  echo "Cool matrix baseline (2026-09-12): allpaka decode 39.0 / llama 45.5 = 0.86×"
  echo "Ship if mvid-on decode >= llama; else ALLPAKA_MV_ID=0 and capture Q8/Q4 BW."
} 2>&1 | tee "$OUT/full.log"
