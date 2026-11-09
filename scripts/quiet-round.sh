#!/bin/bash
# Full quiet-machine measurement round. Refuses to run under load.
# usage: quiet-round.sh   (run from the perf-elim worktree)
set -u
cd "$(dirname "$0")/.."
top=$(ps -A -o %cpu -r | awk 'NR==2{print $1}')
if awk -v t="$top" 'BEGIN{exit !(t>=60)}'; then
  echo "NOISY: top process ${top}% >= 60 - aborting"; exit 1
fi
D=.rbench/quiet-$(date +%H%M)
mkdir -p "$D"
M6=~/.cache/allpaka/models/Qwen3-0.6B-Q8_0.gguf
M8=~/.cache/allpaka/models/Qwen3-8B-Q4_K_M.gguf
M30=~/.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf

echo "=== mmbench q6_k vs llama (12288->4096, m=480) + q4_k control ==="
if [ ! -x target/release/examples/mmbench ]; then
  ~/.cargo/bin/cargo build --release --example mmbench -p allpaka-backend 2>&1 | tail -1
fi
for fmt in q6_k q4_k; do
  MM_FMT=$fmt MM_NIN=12288 MM_NOUT=4096 MM_M=480 MM_LLAMA=1 \
    ./target/release/examples/mmbench 2>/dev/null | tee "$D/mmbench-$fmt.txt"
done

echo "=== 8B pp stage split (ALLPAKA_PF_SPLIT=1) ==="
ALLPAKA_PF_SPLIT=1 ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=1 ALLPAKA_BENCH_SKIP_MTP=1 \
  ./target/release/allpaka bench --engine "$M8" 2>/dev/null | tee "$D/pf-split-8b.txt" | tail -30

echo "=== full matrix (3 models, REPEATS=3) ==="
REPEATS=3 ALLPAKA_BENCH_SKIP_MTP=1 BENCH_OUTPUT_DIR="$PWD/$D/matrix" \
  scripts/bench-matrix.sh "$M6" "$M8" "$M30" 2>&1 | tail -40
echo "=== done: $D ==="
