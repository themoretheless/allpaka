#!/usr/bin/env bash
# Run `allpaka bench --engine` several times, one process per run, and fail
# unless every run prints the same prefill logits hash and the same greedy
# decode stream hash.
#
#   scripts/check-prefill-determinism.sh <model.gguf> [runs] [pp]
#
# ALLPAKA_BIN picks the binary (default target/release/allpaka[.exe]). Any
# ALLPAKA_* switch in the environment reaches the runs, so a single mode can be
# checked: ALLPAKA_GGML=0 scripts/check-prefill-determinism.sh model.gguf 5 128
set -euo pipefail

model=${1:?usage: check-prefill-determinism.sh <model.gguf> [runs] [pp]}
runs=${2:-5}
pp=${3:-512}

here=$(cd "$(dirname "$0")/.." && pwd)
bin=${ALLPAKA_BIN:-$here/target/release/allpaka}
[ -x "$bin" ] || bin="$bin.exe"
[ -x "$bin" ] || { echo "error: no allpaka binary at $bin; build with cargo build --release -p allpaka-cli" >&2; exit 2; }

export ALLPAKA_BENCH_PP=$pp
export ALLPAKA_BENCH_TG=${ALLPAKA_BENCH_TG:-32}
export ALLPAKA_BENCH_SKIP_MTP=1

first=""
status=0
for run in $(seq 1 "$runs"); do
    out=$("$bin" bench --engine "$model" 2>&1) || { echo "$out" | tail -5 >&2; echo "run $run: bench failed" >&2; exit 2; }
    prefill=$(echo "$out" | grep -o 'argmax=[0-9]* logits-fnv=[0-9a-f]*' | head -1)
    decode=$(echo "$out" | grep -o 'tokens-fnv=[0-9a-f]*' | head -1)
    [ -n "$prefill" ] && [ -n "$decode" ] || { echo "run $run: no hashes in the bench output" >&2; exit 2; }
    line="$prefill $decode"
    echo "run $run: $line"
    if [ -z "$first" ]; then
        first=$line
    elif [ "$line" != "$first" ]; then
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "FAIL: pp$pp runs disagree" >&2
else
    echo "PASS: $runs runs at pp$pp agree"
fi
exit "$status"
