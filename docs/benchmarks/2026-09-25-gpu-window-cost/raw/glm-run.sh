#!/bin/bash
# Single-variable test of the 235B window cliff: a 46.6 GiB mapping at cap 3 GiB
# has a ~138 GiB total mapped span (more than the 108.7 GiB that broke on the
# 235B) while its unique bytes are 44% smaller. If mapped span is the variable,
# this breaks; if the large mapping is, it does not.
set -uo pipefail
ROOT="/Users/themoretheless/Documents/Sources/allpaka"
OUT="/Users/themoretheless/Documents/Sources/allpaka/.airbug-bench/glm-window-cliff"
M="$ROOT/models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
for gib in default 3 8; do
  if [ "$gib" = default ]; then unset ALLPAKA_GPU_WINDOW_GIB; else export ALLPAKA_GPU_WINDOW_GIB=$gib; fi
  ALLPAKA_BENCH_PP=64 ALLPAKA_BENCH_TG=8 ALLPAKA_BENCH_SKIP_MTP=1 \
  ALLPAKA_BENCH_REPORT="$OUT/glm-$gib-report.json" \
    "$ROOT/target/release/allpaka" bench --engine "$M" > "$OUT/glm-$gib-log.txt" 2>&1
  echo "glm cap ${gib} $(date -u +%T) exit $? $(grep -o 'attached, [0-9.]* GiB.*' $OUT/glm-$gib-log.txt | head -1) $(grep -o 'residency: windows=[0-9]* set=[a-z]*' $OUT/glm-$gib-log.txt) $(grep -o 'prefill argmax=[0-9]*' $OUT/glm-$gib-log.txt) $(grep -o 'gpu during prefill: [0-9]* waits, [0-9]* dispatches' $OUT/glm-$gib-log.txt)"
done
