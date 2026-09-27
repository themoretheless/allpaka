#!/bin/bash
# Run the post-flip llama matrix, but only inside a window the instrument can be
# trusted in. The harness gate proves the work happened; it cannot prove the
# machine was free, so the decision to measure at all lives here.
#
# preregistration.txt, Gates section: 5 consecutive clean samples 60 s apart
# (load < 5.0, no compiler, no foreign bench, no indexer burning CPU now), cap
# 180 min. On cap expiry this exits non-zero and no artifact is written.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
NEED=${NEED:-5}
POLL=${POLL:-60}
CAP_MIN=${CAP_MIN:-180}
LOAD_MAX=${LOAD_MAX:-5.0}
INDEXERS=(corespotlightd managedcorespotlightd mds mds_stores mdworker_shared
          photoanalysisd filecoordinationd)
MODELS=(
    models/qwen3-0.6b-Q8_0.gguf
    models/qwen3-30b-a3b-Q4_K_M.gguf
    models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
)
OUT=$HERE/series-1

busy_names() {
    pgrep -xl cargo; pgrep -xl rustc; pgrep -xl llama-bench
}
# A resident indexer is normal; one working this instant is not.
indexer_busy() {
    local n
    for n in "${INDEXERS[@]}"; do
        ps -Aceo pcpu,comm | awk -v n="$n" '$2 == n && $1 + 0 > 40 {print n" "$1}'
    done
}
sample() {
    local load rivals idx
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    rivals=$(busy_names | tr '\n' ' ')
    idx=$(indexer_busy | tr '\n' ' ')
    printf '%s  load %5s' "$(date -u +%H:%M:%SZ)" "$load"
    [[ -n $rivals ]] && printf '  builders:%s' "$rivals"
    [[ -n $idx ]] && printf '  indexer:%s' "$idx"
    awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { exit (l + 0 < m) ? 0 : 1 }' \
        && [[ -z $rivals && -z $idx ]]
}

clean=0
end=$(( $(date +%s) + 60 * CAP_MIN ))
while true; do
    if out=$(sample); then
        clean=$((clean + 1))
        printf '%s  clean %s/%s\n' "$out" "$clean" "$NEED"
        [[ $clean -ge $NEED ]] && break
    else
        clean=0
        printf '%s  not clean - not measuring yet\n' "$out"
    fi
    [[ $(date +%s) -gt $end ]] && { echo "cap ${CAP_MIN}min reached; campaign did NOT run"; exit 3; }
    sleep "$POLL"
done

echo "=== window reached ==="
cd "$ROOT" || exit 1
for m in "${MODELS[@]}"; do
    [[ -f $m ]] || { echo "missing model: $m"; exit 1; }
done
echo "allpaka sha256 $(shasum -a 256 target/release/allpaka | awk '{print $1}')"
echo "head $(git rev-parse --short HEAD)  tree $(git status --porcelain | grep -cv '^??') tracked edits"
echo "llama $(llama-bench --version 2>/dev/null | head -1)"

mkdir -p "$OUT"
REPEATS=${REPEATS:-5} PP=${PP:-480} TG=${TG:-32} BENCH_OUTPUT_DIR="$OUT" \
    scripts/bench-matrix.sh "${MODELS[@]}" 2>"$OUT/matrix-1.stderr.txt" |
    tee "$OUT/matrix-1.txt"
rc=${PIPESTATUS[0]}
echo "=== bench-matrix exit $rc ; artifacts in $OUT ==="
exit $rc
