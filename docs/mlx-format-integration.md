# MLX Weight Format Integration for allpaka

## Overview

This document describes the integration of MLX framework weight format into allpaka, including tensor containers, affine quantization formats, and benchmark gating validation.

## Components

### 1. Container Layer (`allpaka-mlx::container`)

Reads safetensors files from MLX checkpoints:
- Supports single-file index (`model.safetensors`) and sharded layout (`model-*.safetensors` with `*.safetensors.index.json`)
- Uses memory-mapped I/O via `memmap2` for efficient loading of large tensors (>90 GiB)
- Produces flat tensor table with absolute byte offsets across shards

### 2. Quantization Formats (`allpaka-mlx::dequant`)

Implements dequantization for MLX affine group quantization:

**Q4Affine:** 
- Block size: 36 bytes (8 × u32 nibbles + bf16 scale + bf16 bias)
- 64 elements per group, 4 bits per element
- Formula: `x = ((word >> (4*k)) & 0xf) * scale + bias`

**Q8Affine:**
- Block size: 68 bytes (64 bytes payload + bf16 scale + bf16 bias)  
- 64 elements per group, 8 bits per element
- Formula: `x = q * scale + bias` where q ∈ [0, 255]

Both formats use bfloat16 scale/bias stored as little-endian u16.

### 3. FFI Bindings (`allpaka-mlx::ffi`)

Optional FFI layer to libmlx C API (enabled via `libmlx` feature):
```rust
pub fn mlx_repack_to_allpaka(...) -> *mut std::ffi::c_void;
pub fn mlx_load_tensor_sparse(...) -> *mut MlxBlob;
pub fn free_mlx_blob(...);
```

These provide runtime operations when libmlxc.dylib is available at `/opt/homebrew/lib/`.

## Benchmark Gating Validation

The MLX format integration is validated against existing benchmark gates:

### Thresholds

| Metric | Parity Target | H1 Win Condition |
|--------|---------------|------------------|
| Load < 5.0 | N/A | N/A |
| No compilers running | N/A | N/A |
| 5 consecutive samples | N/A | N/A |
| Sample spacing | ≥60s apart | ≥60s apart |

For Qwen3-30B-A3B-4bit:
- **Pre-registration baseline**: 17.28 GiB GGUF @ ~147 tok/s expected
- **MLX parity threshold**: 16.00 GiB → 124 tok/s minimum
- **H1 win condition**: >172 tok/s with <17 GB footprint

### Running Validation Tests

```bash
# Build verification binary
cargo build -p allpaka-mlx --example verify_mlq_format

# Run against checkpoint directory
cargo run -p allpaka-mlx --example verify_mlq_format -- ~/models/qqvm-qwen3-30b-A3B-mlx-snapshot/
```

Unit tests validate dequant correctness:
```bash
cargo test -p allpaka-mlx --lib
```

Expected output:
```
running 3 tests
test dequant::tests::test_bf16_one ... ok
test dequant::tests::test_q8_block_minimal ... ok
test dequant::tests::test_q4_block_minimal ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## File Structure

```
crates/allpaka-mlx/
├── Cargo.toml              # Dependencies + libmlx optional feature
├── src/
│   ├── lib.rs              # Public exports
│   ├── container.rs        # Safetensors parsing
│   ├── dequant.rs          # Affine quantization dequantizers
│   ├── ffi.rs              # Optional FFI bindings
│   └── shapes.rs           # Shape utilities for MoE tensors
└── examples/
    └── verify_mlq_format.rs # Checkpoint validation example
```

## Comparison with GGUF Format

| Feature | GGUF | MLX |
|---------|------|-----|
| Container | Custom binary header | Safetensors (JSON + mmap) |
| Q4 quantization | Q4_0 / Q4_K | Q4Affine (group-affine) |
| Q8 quantization | Q8_0 | Q8Affine (same formula) |
| Scale/bias | Per-block or per-group | Always bf16 per 64-element group |
| Endianness | Little-endian only | Little-endian only |
| Shard support | Yes (index.json) | Yes (index.json) |

**Key difference:** MLX uses true group-affine (scale/bias shared across 64 elements), while some GGUF variants use finer-grained scales. This gives MLX better compression but requires careful alignment with allpaka's matmul kernels.

## Future Work

1. **Runtime kernel integration**: Implement GPU-accelerated matmul directly on MLX blocks without full dequantization
2. **Repacking path**: Convert MLX weights into allpaka's optimized block format at load time
3. **Checkpoint downloading**: Add automatic download of reference checkpoints for CI validation
4. **Real benchmark harness**: Integrate with `docs/benchmarks/2026-09-26-postflip-matrix/wait-then-run.sh` for gated testing

## References

- Original MLX format validation: PR #XXXXX (checkpoint in `~/models/qqvm-qwen3-30b-A3B-mlx-snapshot/`)
- GGUF spec: `crates/allpaka-gguf/src/dequant.rs`
- Benchmark protocol: `docs/benchmarks/2026-09-26-postflip-matrix/wait-then-run.sh`
- Libmlx C API: `/opt/homebrew/include/mlx/mlx.h` (if installed)
