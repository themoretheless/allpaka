#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: bench-compare.sh MODEL [PP=480] [TG=32] [REPEATS=5]}
pp=${2:-480}
tg=${3:-32}
repeats=${4:-5}
allpaka=${ALLPAKA_BIN:-./target/release/allpaka}
llama=${LLAMA_BENCH:-llama-bench}
delay=${BENCH_DELAY:-0}

command -v jq >/dev/null
command -v "$llama" >/dev/null
[[ -x "$allpaka" ]]
[[ "$pp" =~ ^[0-9]+$ && "$tg" =~ ^[0-9]+$ && "$repeats" =~ ^[1-9][0-9]*$ ]]

tmp=$(mktemp -d "${TMPDIR:-/tmp}/allpaka-bench.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

median() {
    sort -n "$1" | awk '{ v[NR]=$1 } END {
        if (NR % 2) print v[(NR+1)/2];
        else print (v[NR/2] + v[NR/2+1]) / 2;
    }'
}

for ((i=1; i<=repeats; i++)); do
    echo "run $i/$repeats: allpaka" >&2
    out=$(ALLPAKA_BENCH_PP="$pp" ALLPAKA_BENCH_TG="$tg" \
        "$allpaka" bench --engine "$model" 2>&1)
    grep -q "gpu path decode: attempts=$tg successes=$tg declines=0" <<<"$out"
    awk '$1 == "prefill" && $3 == "tok" { print $(NF-1) }' <<<"$out" >>"$tmp/allpaka_pp"
    awk '$1 == "decode" && $3 == "tok" { print $(NF-1) }' <<<"$out" >>"$tmp/allpaka_tg"

    echo "run $i/$repeats: llama.cpp" >&2
    "$llama" -m "$model" -p "$pp" -n "$tg" -r 1 -o json \
        >"$tmp/llama-$i.json" 2>"$tmp/llama-$i.err"
    jq -er --argjson n "$pp" '.[] | select(.n_prompt == $n) | .avg_ts' \
        "$tmp/llama-$i.json" >>"$tmp/llama_pp"
    jq -er --argjson n "$tg" '.[] | select(.n_gen == $n) | .avg_ts' \
        "$tmp/llama-$i.json" >>"$tmp/llama_tg"
    sleep "$delay"
done

ap=$(median "$tmp/allpaka_pp")
at=$(median "$tmp/allpaka_tg")
lp=$(median "$tmp/llama_pp")
lt=$(median "$tmp/llama_tg")
rp=$(awk -v a="$ap" -v l="$lp" 'BEGIN { printf "%.2f", a/l }')
rt=$(awk -v a="$at" -v l="$lt" 'BEGIN { printf "%.2f", a/l }')

printf '\n| Metric | llama.cpp median | allpaka median | allpaka/llama |\n'
printf '| --- | ---: | ---: | ---: |\n'
printf '| pp%s prefill tok/s | %.1f | %.1f | %sx |\n' "$pp" "$lp" "$ap" "$rp"
printf '| tg%s decode tok/s | %.1f | %.1f | %sx |\n' "$tg" "$lt" "$at" "$rt"
printf '\nRaw samples (tok/s):\n'
printf 'allpaka pp: %s\n' "$(paste -sd, "$tmp/allpaka_pp")"
printf 'llama.cpp pp: %s\n' "$(paste -sd, "$tmp/llama_pp")"
printf 'allpaka tg: %s\n' "$(paste -sd, "$tmp/allpaka_tg")"
printf 'llama.cpp tg: %s\n' "$(paste -sd, "$tmp/llama_tg")"
