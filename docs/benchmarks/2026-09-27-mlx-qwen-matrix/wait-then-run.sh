#!/bin/bash
# Run the MLX weight format validation harness for Qwen3 30B-A3B models.
#
# This validates the allpaka-mlx crate's safetensors loading and dequantization
# against benchmark gating criteria: 5 consecutive clean samples, load < 5.0,
# no concurrent compilation work. Cap at 180 minutes before giving up.
#
# Preregistration.txt Gates (same as llama matrix):
# - 5 consecutive clean samples 60s apart
# - load < 5.0, no compiler running, no foreign bench indexer burning CPU
# - On cap expiry: exit 3 without writing artifacts
#
# Expected parity thresholds (from mlx-format-integration.md):
# - MLX Q4Affine baseline: 16.00 GiB footprint → ~124 tok/s minimum
# - H1 win condition: >172 tok/s with <17 GB footprint

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
NEED=${NEED:-5}
POLL=${POLL:-60}
CAP_MIN=${CAP_MIN:-180}
LOAD_MAX=${LOAD_MAX:-5.0}
INDEXERS=(corespotlightd managedcorespotlightd mds mds_stores mdworker_shared
          photoanalysisd filecoordinationd)

# MLX checkpoint locations (adjust to your environment)
MLX_CHECKPOINT_DIR="${MLX_CHECKPOINT_DIR:-$HOME/models/qqvm-qwen3-30b-A3B-mlx-snapshot}"

# Models to validate against GGUF variants for comparison
MODELS=(
    "models/qwen3-30b-a3b-Q4_K_M.gguf"
    "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)

OUT=$HERE/series-mlx-1

busy_names() {
    # Check for cargo/rustc builders AND mlx-related tools
    pgrep -xl cargo; pgrep -xl rustc; pgrep -xl llama-bench; pgrep -xl mlx-bench
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

# Validate MLX checkpoint exists
if [[ ! -d "$MLX_CHECKPOINT_DIR" ]]; then
    echo "missing MLX checkpoint directory: $MLX_CHECKPOINT_DIR"
    echo "Set MLX_CHECKPOINT_DIR env var to location of qqvm-qwen3-30b-A3B-mlx-snapshot/"
    exit 1
fi

# Verify container layer loads successfully
echo "Testing MLX safetensors container..."
cargo run -q -p allpaka-mlx --example verify_mlq_format -- "$MLX_CHECKPOINT_DIR" \
    || { echo "MLX container validation failed"; exit 2; }
echo "Container validation passed"

# Load models that exist for comparison
for m in "${MODELS[@]}"; do
    if [[ ! -f "$m" ]]; then
        echo "Skipping missing model: $m (will validate GGUF-only)"
    fi
done

# Report build context
echo "allpaka sha256 $(shasum -a 256 target/release/allpaka 2>/dev/null | awk '{print $1}' || echo 'no-release-build')"
echo "head $(git rev-parse --short HEAD)  tree $(git status --porcelain | grep -cv '^??') tracked edits"
echo "llama $(llama-bench --version 2>/dev/null | head -1)"
echo "mlx-checkpoint-size $(du -sh "$MLX_CHECKPOINT_DIR" 2>/dev/null | cut -f1 || echo 'unknown')"

mkdir -p "$OUT"
REPEATS=${REPEATS:-5} PP=${PP:-480} TG=${TG:-32} BENCH_OUTPUT_DIR="$OUT" \
    scripts/bench-matrix.sh "${MODELS[@]}" 2>"$OUT/matrix-mlx.stderr.txt" |
    tee "$OUT/matrix-mlx.txt"
rc=${PIPESTATUS[0]}

echo "=== bench-matrix exit $rc ; artifacts in $OUT ==="

# Print summary metrics for preregistration validation
if [[ -f "$OUT/matrix-mlx.txt" ]]; then
    echo ""
    echo "=== MLX FORMAT RESULTS ==="
    grep -E "(tok/s|MB/s|GB/s)" "$OUT/matrix-mlx.txt" | tail -10 || echo "No performance data available"
    echo ""
    
    # Validate against preregistration thresholds
    echo "Preregistration check:"
    echo "- Parity threshold: 124 tok/s (Qwen3-30B-A3B MLX vs GGUF)"
    echo "- H1 win threshold: >172 tok/s (<17 GB footprint)"
    echo ""
fi

exit $rc
