#!/usr/bin/env bash
# Paired prefill SCALING ladder: allpaka (dense-obuf fix ON) vs llama-bench over
# a log-spaced PP ladder, repeat-major so each repeat samples the whole curve and
# machine drift shows up as a repeat effect rather than a slope.
#
# Question it answers (chunk-1.txt section 4): is the long-context deficit a bad
# quadratic (attention) term or a bad per-token (linear) term? Two PP points made
# that undecidable; six, with 5 paired repeats, should not.
#
# cost/token = A + B*(L/2).  A = linear/weights term, B = per-key attention term.
#
# Run from the repo root. Campaign-local, not a repo script.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
model=${1:?model}
reps=${2:-5}
wt=${3:?allpaka binary with the dense-obuf fix}
out=${4:?outdir}
LADDER=${LADDER:-"256 512 1024 2048 4096 8192"}
tg=${TG:-32}
mkdir -p "$out" || exit 1
echo "ladder:$LADDER reps:$reps model:$model"
echo "allpaka binary sha256 $(shasum -a 256 "$wt" | awk '{print $1}') head $(git rev-parse --short HEAD)"

for ((i = 1; i <= reps; i++)); do
    if ((i % 2 == 1)); then order="ap ll"; else order="ll ap"; fi
    for pp in $LADDER; do
        for arm in $order; do
            if [[ $arm == ap ]]; then
                ALLPAKA_PF_OBUF_DENSE=1 ALLPAKA_BENCH_PP=$pp ALLPAKA_BENCH_TG=$tg \
                    ALLPAKA_BENCH_REPORT="$out/ap-$pp-$i.json" "$wt" bench --engine "$model" \
                    >"$out/ap-$pp-$i.log" 2>&1
            else
                llama-bench -m "$model" -p "$pp" -n 0 -d 0 -r 1 -ngl 99 -ctk f16 -ctv f16 \
                    -o json >"$out/ll-$pp-$i.json" 2>/dev/null
            fi
        done
        a=$(jq -r '[.measurements[]|select(.name=="prefill")][0].summary.median' "$out/ap-$pp-$i.json")
        l=$(jq -r '.[0].samples_ts[0]' "$out/ll-$pp-$i.json")
        e=$(grep -o "executing [0-9]* ms" "$out/ap-$pp-$i.log" | head -1 | awk '{print $2}')
        c=$(grep -o "during prefill: [0-9]* waits" "$out/ap-$pp-$i.log" | head -1 | awk '{print $3}')
        printf 'rep %s pp %5s  allpaka %8.1f (exec %4sms commits %s)  llama %8.1f  ratio %.4f  load %s\n' \
            "$i" "$pp" "$a" "$e" "$c" "$l" \
            "$(awk -v a="$a" -v l="$l" 'BEGIN{printf "%.4f", a/l}')" \
            "$(sysctl -n vm.loadavg | awk '{print $2}')"
    done
done
echo "=== done; artifacts in $out ==="

# Is the baseline arm llama's best? Every campaign row has been compared against
# a llama-bench that reports flash_attn:-1 (auto), and auto is not documented as
# on-or-off for the Metal backend. If auto means OFF, the llama numbers this goal
# is chasing are a handicapped attention path, and closing "the gap" would be
# closing less than it looks. Probed at both ends, one invocation each, after the
# ladder so it cannot bias the paired rows.
echo "--- llama -fa probe (what did flash_attn:-1 resolve to?) ---"
for pp in 256 8192; do
    for fa in default on off; do
        flag=()
        [[ $fa != default ]] && flag=(-fa "$fa")
        llama-bench -m "$model" -p "$pp" -n 0 -d 0 -r 1 -ngl 99 -ctk f16 -ctv f16 \
            ${flag[@]+"${flag[@]}"} -o json >"$out/fa-$fa-$pp.json" 2>/dev/null
        printf 'pp %5s -fa %-7s %10.1f t/s  (reported flash_attn %s)\n' \
            "$pp" "$fa" "$(jq -r '.[0].samples_ts[0]' "$out/fa-$fa-$pp.json")" \
            "$(jq -r '.[0].flash_attn' "$out/fa-$fa-$pp.json")"
    done
done

"$HERE/fit-scaling.py" "$out/ladder-1.txt" 2>&1 | tee "$out/fit-1.txt"
