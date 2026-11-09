#!/usr/bin/env bash
# Type-check the engine crates for every GPU backend selection, not only the
# one the host compiles: Metal (native macOS), stub (no GPU) and CUDA.
# The stub and CUDA arms are cfg-gated out on macOS, so a plain `cargo check`
# never sees them.
#
# Needs a non-macOS std: rustup target add x86_64-pc-windows-msvc
set -euo pipefail

target=${ALLPAKA_CHECK_TARGET:-x86_64-pc-windows-msvc}
crates=(-p allpaka-dtype -p allpaka-gguf -p allpaka-mlx -p allpaka-backend -p allpaka-model -p allpaka-core)

echo "== native" >&2
cargo check "${crates[@]}" --all-targets
echo "== stub ($target)" >&2
cargo check "${crates[@]}" --target "$target"
echo "== cuda ($target)" >&2
cargo check "${crates[@]}" --target "$target" --features allpaka-backend/cuda
