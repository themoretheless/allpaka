#!/usr/bin/env bash
# Run paired allpaka vs llama.cpp benches across several GGUF models.
# Usage:
#   scripts/bench-matrix.sh models/qwen3-30b-a3b-Q4_K_M.gguf \
#       models/qwen3.5-35b-a3b-Q4_K_M.gguf models/glm-4.5-air-Q4_K_M.gguf
# Optional env: PP TG REPEATS BENCH_OUTPUT_DIR ALLPAKA_BIN LLAMA_BENCH
set -euo pipefail

[[ $# -ge 1 ]] || {
    echo "usage: $0 MODEL [MODEL...]" >&2
    exit 1
}

root=$(cd "$(dirname "$0")/.." && pwd)
compare=$root/scripts/bench-compare.sh
[[ -x $compare ]] || chmod +x "$compare"

pp=${PP:-480}
tg=${TG:-32}
repeats=${REPEATS:-5}
out=${BENCH_OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/allpaka-matrix.XXXXXX")}
mkdir -p "$out"
out=$(cd "$out" && pwd)

entries=$out/entries.jsonl
: >"$entries"

for model in "$@"; do
    [[ -f $model ]] || {
        echo "missing model: $model" >&2
        exit 1
    }
    name=$(basename "$model")
    name=${name%.gguf}
    model_out=$out/$name
    mkdir -p "$model_out"
    printf '=== %s ===\n' "$name" >&2
    BENCH_OUTPUT_DIR=$model_out \
        "$compare" "$model" "$pp" "$tg" "$repeats" | tee "$model_out/comparison.stdout.json"
    # bench-compare writes comparison.json into BENCH_OUTPUT_DIR
    jq -n --arg name "$name" --slurpfile c "$model_out/comparison.json" \
        '{name:$name, comparison:$c[0]}' >>"$entries"
done

jq -n --argjson pp "$pp" --argjson tg "$tg" --slurpfile rows <(jq -s '.' "$entries") '
  def ratio(a; b): if (b == null or b == 0) then null else a / b end;
  {
    pp: $pp,
    tg: $tg,
    models: [$rows[0][] | {
      name: .name,
      model: .comparison.model,
      allpaka_prefill_median: .comparison.allpaka_prefill.median,
      llama_prefill_median: .comparison.llama_prefill.median,
      prefill_ratio: ratio(.comparison.allpaka_prefill.median; .comparison.llama_prefill.median),
      allpaka_decode_median: .comparison.allpaka_decode.median,
      llama_decode_median: .comparison.llama_decode.median,
      decode_ratio: ratio(.comparison.allpaka_decode.median; .comparison.llama_decode.median)
    }]
  }
' | tee "$out/matrix.json"

printf '\nMatrix artifacts: %s\n' "$out"
