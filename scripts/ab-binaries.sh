#!/usr/bin/env bash
# Interleaved A/B of two allpaka builds on one model: A B A B ..., so slow
# drift and background load hit both arms alike. Prints prefill and decode
# tok/s per pair, the per-pair delta of B against A, and whether both builds
# produced the same prefill logits hash (logits-fnv).
#
# usage: scripts/ab-binaries.sh <allpaka-A> <allpaka-B> <model.gguf> [pairs] [pp] [tg]
set -uo pipefail
A="$1"; B="$2"; M="$3"; PAIRS="${4:-6}"; PP="${5:-512}"; TG="${6:-128}"
run() {
    ALLPAKA_BENCH_PP="$PP" ALLPAKA_BENCH_TG="$TG" ALLPAKA_BENCH_SKIP_MTP=1 \
        "$1" bench --engine "$M" 2>/dev/null |
        awk '/^  prefill .*tok\/s/ {p=$(NF-1)} /^  decode .*tok\/s/ {d=$(NF-1)}
             /logits-fnv=/ {split($NF, f, "="); h=f[2]} END {print p, d, h}'
}
run "$A" >/dev/null; run "$B" >/dev/null   # warm page cache and kernels
printf 'pair  A_pp     B_pp     d_pp%%   A_tg     B_tg     d_tg%%   logits\n'
for i in $(seq 1 "$PAIRS"); do
    read -r ap at ah < <(run "$A")
    read -r bp bt bh < <(run "$B")
    same=$([ "$ah" = "$bh" ] && echo same || echo "DIFF $ah $bh")
    awk -v i="$i" -v ap="$ap" -v bp="$bp" -v at="$at" -v bt="$bt" -v s="$same" 'BEGIN {
        printf "%-5d %-8s %-8s %+6.2f  %-8s %-8s %+6.2f  %s\n", i, ap, bp, (bp-ap)/ap*100, at, bt, (bt-at)/at*100, s }'
done
