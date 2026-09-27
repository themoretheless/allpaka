#!/usr/bin/env bash
# Task #25 driver: ten AB/BA pairs of the 35B's decode wall, arms `ALLPAKA_SWFUSE`
# unset (shipped since 2026-09-24) against `=1` (the default-ON policy every
# published row for this model measured). Protocol is fixed in
# preregistration.txt beside this file; the command line is verbatim from
# ../qwen35-35b-m4-max-pending.md.
#
# The binary is copied out to $PINNED before the first pair, because this is a
# shared worktree and another session can rebuild target/release/allpaka mid-run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-$ROOT/.airbug-bench/q35-postflip-$(date +%Y%m%d-%H%M%S)}
PAIRS=${PAIRS:-10}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
mkdir -p "$OUT"
cp "$ROOT/target/release/allpaka" "$OUT/allpaka"
BIN=$OUT/allpaka
{
  echo "pinned binary sha256: $(shasum -a 256 "$BIN" | awk '{print $1}')"
  echo "git HEAD: $(git -C "$ROOT" rev-parse HEAD)"
  echo "dirty rust at start:"; git -C "$ROOT" status --porcelain '*.rs' | sed 's/^/  /'
  echo "model: $MODEL"
  echo "start: $(date -u +%FT%TZ)  $(uptime)"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

export ALLPAKA_BENCH_SKIP_MTP=1 ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=32

arm() { # $1 = label (plain|fused), $2 = pair index
  local label=$1 i=$2
  echo "--- pair $i arm $label  $(uptime)"
  if [[ $label == fused ]]; then export ALLPAKA_SWFUSE=1; else unset ALLPAKA_SWFUSE; fi
  ALLPAKA_BENCH_REPORT="$OUT/allpaka-${label}-${i}.json" \
    "$BIN" bench --engine "$MODEL" >"$OUT/allpaka-${label}-${i}.txt" 2>&1
  unset ALLPAKA_SWFUSE
}

# The llama arm is not in preregistration.txt: it is added because the pending
# doc's 1.35x row was measured in the fused regime, and restating it needs a
# llama number from THIS window, not arithmetic across two. One repeat per pair,
# placed between the two allpaka arms so drift hits both equally.
arm_llama() {
  local i=$1
  echo "--- pair $i arm llama  $(uptime)"
  llama-bench -m "$MODEL" -p 0 -n 32 -d 1 -r 1 -ngl 99 -ctk f16 -ctv f16 -o json \
    >"$OUT/llama-tg-$i.json" 2>"$OUT/llama-tg-$i.txt"
}

# Each pair carries the shipped arm twice, at both ends, so the pair gets its own
# noise bar: |plain2/plain - 1| on decode ms/token is what the window lets this
# instrument resolve for the SAME policy, and a pair whose repeaters disagree is
# not evidence about a 0.3 ms difference between policies. See Amendment 2.
for ((i = 1; i <= PAIRS; i++)); do
  if ((i % 2)); then arm plain "$i"; sleep 1; arm_llama "$i"; sleep 1; arm fused "$i"; sleep 1; arm plain2 "$i"
  else arm plain2 "$i"; sleep 1; arm fused "$i"; sleep 1; arm_llama "$i"; sleep 1; arm plain "$i"; fi
done
echo "DONE -> $OUT" 2>&1 | tee -a "$OUT/provenance.txt"
