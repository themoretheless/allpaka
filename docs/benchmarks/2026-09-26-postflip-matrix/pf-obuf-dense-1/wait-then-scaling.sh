#!/usr/bin/env bash
# Chain the scaling ladder BEHIND the in-flight series-1 gate, then wait for its
# own clean window. One quiet period therefore feeds both runs in a defined order
# instead of having them contend for it - a ladder running alongside series-1
# would produce exactly the depressed-rows result chunk-1.txt documents.
#
# Same gate semantics as ../wait-then-run.sh (preregistration.txt, Gates):
# 5 consecutive clean samples 60 s apart, load < 5.0, no compiler, no foreign
# bench, no indexer burning CPU. Cap here is longer because the wait is stacked.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
NEED=${NEED:-5}
POLL=${POLL:-60}
CAP_MIN=${CAP_MIN:-300}
LOAD_MAX=${LOAD_MAX:-5.0}
SERIES_PID=${SERIES_PID:-}
BIN=${BIN:-/tmp/pf-obuf-target/release/allpaka}
MODEL=${MODEL:-models/qwen3-0.6b-Q8_0.gguf}
OUT=${OUT:-$HERE/scaling-1}
INDEXERS=(corespotlightd managedcorespotlightd mds mds_stores mdworker_shared
          photoanalysisd filecoordinationd)

busy_names() {
    pgrep -xl cargo; pgrep -xl rustc; pgrep -xl llama-bench; pgrep -xl allpaka
    pgrep -f "bench-matrix.sh" | sed 's/^/matrix-pid:/'
}
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
    [[ -n $rivals ]] && printf '  rivals:%s' "$rivals"
    [[ -n $idx ]] && printf '  indexer:%s' "$idx"
    awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { exit (l + 0 < m) ? 0 : 1 }' \
        && [[ -z $rivals && -z $idx ]]
}

# Do not even start counting clean samples while series-1 owns the machine.
if [[ -n $SERIES_PID ]] && kill -0 "$SERIES_PID" 2>/dev/null; then
    echo "waiting for series-1 pid $SERIES_PID to exit"
    while kill -0 "$SERIES_PID" 2>/dev/null; do sleep "$POLL"; done
    echo "series-1 exited at $(date -u +%H:%M:%SZ); now gating on load"
fi

cd "$ROOT" || exit 1
[[ -x $BIN ]] || { echo "missing binary $BIN - refusing to measure"; exit 1; }
[[ -f $MODEL ]] || { echo "missing model $MODEL"; exit 1; }

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
    [[ $(date +%s) -gt $end ]] && { echo "cap ${CAP_MIN}min reached; ladder did NOT run"; exit 3; }
    sleep "$POLL"
done

echo "=== window reached at $(date -u +%FT%TZ) ==="
mkdir -p "$OUT"
{ echo "window opened $(date -u +%FT%TZ)"; sysctl -n vm.loadavg; } >"$OUT/window.txt"
REPS=${REPS:-5} "$HERE/scaling-1.sh" "$MODEL" "${REPS:-5}" "$BIN" "$OUT" \
    >"$OUT/ladder-1.txt" 2>&1
echo "=== scaling-1 exit $? ; see $OUT/ladder-1.txt ==="
