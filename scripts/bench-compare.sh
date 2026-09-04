#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: bench-compare.sh MODEL [PP=480] [TG=32] [REPEATS=5]}
pp=${2:-480}
tg=${3:-32}
repeats=${4:-5}
allpaka=${ALLPAKA_BIN:-./target/release/allpaka}
llama=${LLAMA_BENCH:-llama-bench}
[[ "$pp" =~ ^[1-9][0-9]*$ && "$tg" =~ ^[1-9][0-9]*$ && "$repeats" =~ ^[1-9][0-9]*$ ]]
command -v jq >/dev/null
command -v "$llama" >/dev/null
[[ -x "$allpaka" ]]
# Keep raw artifacts, including failed runs, for reproducibility.
out=${BENCH_OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/allpaka-bench.XXXXXX")}
mkdir -p "$out"
out=$(cd "$out" && pwd)
sha=$(shasum -a 256 "$model" | awk '{print $1}')

run_allpaka() {
    ALLPAKA_BENCH_PP="$pp" ALLPAKA_BENCH_TG="$tg" \
        ALLPAKA_BENCH_REPORT="$out/allpaka-$1.json" \
        "$allpaka" bench --engine "$model" >"$out/allpaka-$1.log" 2>&1
    jq -e --argjson pp "$pp" --argjson tg "$tg" '
      ([.measurements[] | select(.name=="prefill")][0] |
       .tokens==$pp and .context_tokens==0 and (.input_tokens|length)==$pp) and
      ([.measurements[] | select(.name=="decode")][0] |
       .tokens==$tg and .context_tokens==$pp and (.input_tokens|length)==$tg and
       .fast_path.attempts==$tg and .fast_path.successes==$tg and .fast_path.declines==0)
    ' "$out/allpaka-$1.json" >/dev/null
}

run_llama() {
    "$llama" -m "$model" -p "$pp" -n 0 -d 0 -r 1 -ngl 99 -ctk f16 -ctv f16 -o json \
        >"$out/llama-pp-$1.json" 2>"$out/llama-pp-$1.log"
    "$llama" -m "$model" -p 0 -n "$tg" -d "$pp" -r 1 -ngl 99 -ctk f16 -ctv f16 -o json \
        >"$out/llama-tg-$1.json" 2>"$out/llama-tg-$1.log"
}

for ((i=1; i<=repeats; i++)); do
    printf 'pair %s/%s\n' "$i" "$repeats" >&2
    if ((i % 2)); then run_allpaka "$i"; run_llama "$i";
    else run_llama "$i"; run_allpaka "$i"; fi
done

jq -n --arg model "$model" --arg sha "$sha" --argjson pp "$pp" --argjson tg "$tg" \
    --slurpfile ap <(jq -s '.' "$out"/allpaka-*.json) \
    --slurpfile lp <(jq -s 'add' "$out"/llama-pp-*.json) \
    --slurpfile lt <(jq -s 'add' "$out"/llama-tg-*.json) '
  def stats: sort | {samples: ., median: (if length%2==1 then .[length/2|floor]
    else (.[length/2-1]+.[length/2])/2 end), min: .[0], max: .[-1]};
  {model: $model, model_sha256: $sha, pp: $pp, tg: $tg,
   comparison_validated: false,
   limitations: ["llama-bench generates its own token stream; MoE routing is not identical",
                 "KV precision must be verified against the allpaka capability report"],
   allpaka_prefill: ([$ap[0][].measurements[]|select(.name=="prefill")|.summary.median]|stats),
   allpaka_decode: ([$ap[0][].measurements[]|select(.name=="decode")|.summary.median]|stats),
   llama_prefill: ([$lp[0][]|select(.n_prompt==$pp and .n_gen==0 and .n_depth==0)|.samples_ts[]]|stats),
   llama_decode: ([$lt[0][]|select(.n_prompt==0 and .n_gen==$tg and .n_depth==$pp)|.samples_ts[]]|stats)}
' >"$out/comparison.json"
cat "$out/comparison.json"
printf '\nArtifacts: %s\n' "$out"
