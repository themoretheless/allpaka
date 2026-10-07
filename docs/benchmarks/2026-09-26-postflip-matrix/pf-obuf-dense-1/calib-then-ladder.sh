#!/usr/bin/env bash
# Gate on the GPU itself, not on load average.
#
# Tonight's measurements were destroyed by a GUI app co-tenanting the Metal
# device (Kimi Helper (Renderer), 76% CPU) while load sat at 5.6-7.4, no
# compiler ran, and every rule in the campaign's Gates section reported clean.
# GPU executing time for one fixed shape doubled inside a run. Load is a proxy
# for CPU pressure; this device is the resource, so measure the resource.
#
# Calibration shape: pp1024, one rep. Quiet baseline on this build/model is
# ~101 ms of `executing` (scaling-1 rep1 102, attn_mm A/B 101); the contaminated
# evening ran 136-150 ms. Accept below 112 ms (+10% on baseline), 3 consecutive
# samples, then re-check AFTER the ladder and mark the run uncertified if the
# window moved - a clean sample before a run proves nothing about its middle.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
BIN=${BIN:-/tmp/pf-obuf-target/release/allpaka}
MODEL=${MODEL:-models/qwen3-0.6b-Q8_0.gguf}
BASE=${BASE:-101}
LIMIT_PCT=${LIMIT_PCT:-10}
NEED=${NEED:-3}
POLL=${POLL:-120}
CAP_MIN=${CAP_MIN:-240}
LADDER=${LADDER:-"1024 1536 2048 2560 3072 4096 8192"}
REPS=${REPS:-5}
OUT=${OUT:-$HERE/scaling-calib-1}
MAX=$((BASE + BASE * LIMIT_PCT / 100))

calib() {
    ALLPAKA_PF_OBUF_DENSE=1 ALLPAKA_BENCH_PP=1024 ALLPAKA_BENCH_TG=1 \
        ALLPAKA_BENCH_REPORT=/tmp/calib-report.json "$BIN" bench --engine "$MODEL" 2>&1 |
        grep -oE "executing [0-9]+ ms" | head -1 | grep -oE "[0-9]+"
}
tenants() {
    ps -Aceo pcpu,comm | awk '$1 + 0 > 30 && $0 !~ /WindowServer/ {print $2" "$1}' |
        tr '\n' ' '
}

cd "$ROOT" || exit 1
[[ -x $BIN ]] || { echo "missing $BIN"; exit 1; }
mkdir -p "$OUT" || exit 1
echo "gate: calib shape pp1024, accept <= ${MAX} ms executing (baseline ${BASE} +${LIMIT_PCT}%)"

ok=0
end=$(( $(date +%s) + 60 * CAP_MIN ))
while true; do
    ms=$(calib)
    printf '%s  calib %sms  load %s  busy:%s\n' "$(date -u +%H:%M:%SZ)" "${ms:-none}" \
        "$(sysctl -n vm.loadavg | awk '{print $2}')" "$(tenants)"
    if [[ -n $ms ]] && ((ms <= MAX)); then
        ok=$((ok + 1)); [[ $ok -ge $NEED ]] && break
    else
        ok=0
    fi
    [[ $(date +%s) -gt $end ]] && { echo "cap ${CAP_MIN}min; ladder did NOT run"; exit 3; }
    sleep "$POLL"
done

CAL_BEFORE=${ms}
echo "=== window reached $(date -u +%FT%TZ) (calib ${CAL_BEFORE}ms) ==="
LADDER="$LADDER" PROBE_FA=0 "$HERE/scaling-1.sh" "$MODEL" "$REPS" "$BIN" "$OUT" \
    >"$OUT/ladder-1.txt" 2>&1
rc=$?
CAL_AFTER=$(calib)
echo "post-run calib ${CAL_AFTER}ms vs pre-run ${CAL_BEFORE}ms"
{
    echo "pre-run calib ${CAL_BEFORE} ms   post-run calib ${CAL_AFTER} ms   accept <= ${MAX} ms"
    if [[ -n $CAL_AFTER ]] && ((CAL_AFTER <= MAX)); then
        echo "window held across the run"
    else
        echo "WINDOW DID NOT HOLD - the run straddled GPU pressure; treat as provisional"
    fi
} >"$OUT/window.txt"
"$HERE/fit-scaling.py" "$OUT/ladder-1.txt" 2>&1 | tee -a "$OUT/window.txt"
echo "=== exit $rc ; artifacts in $OUT ==="
