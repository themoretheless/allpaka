#!/bin/bash
# Interleaved noisy-machine A/B: decode tg tok/s, alternating arms so
# stationary load hits both. Report per-pair deltas + sign count.
# usage: ab-decode.sh <model> <envA-name>   (A = flag off, B = flag on)
set -u
M="$1"
FLAG="${2:-ALLPAKA_SWFUSE}"
PAIRS="${3:-6}"
cd "$(dirname "$0")/.."
run() {
  env ${1:+$1=1} ALLPAKA_BENCH_PP=8 ALLPAKA_BENCH_TG=32 ALLPAKA_BENCH_SKIP_MTP=1 \
    ./target/release/allpaka bench --engine "$M" 2>/dev/null | awk '/^  decode/ {print $7}'
}
# warm the page cache / pipelines once
run "" >/dev/null
pos=0; neg=0
for i in $(seq 1 "$PAIRS"); do
  a=$(run ""); b=$(run "$FLAG")
  d=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.3f", (b-a)/a*100}')
  s=$(awk -v d="$d" 'BEGIN{print (d>0)?"+":"-"}')
  [ "$s" = "+" ] && pos=$((pos+1)) || neg=$((neg+1))
  echo "pair $i: off=$a on=$b  delta=${d}% ($s)"
done
echo "signs: +$pos / -$neg for $FLAG on $(basename "$M")"
