# Baseline

Снимок на коммите `279a667`. Счётчики собраны grep'ом по `crates/*/src`
(unwrap/expect включают тестовые модули внутри `src`), LOC — по всем `.rs` крейта.

## Размер и гигиена по крейтам

| Крейт | LOC | Тесты | unwrap/expect | unsafe | `SAFETY` | `env::var` |
|---|---:|---:|---:|---:|---:|---:|
| allpaka-backend | 31 074 | 67 | 72 | 281 | 3 | 97 |
| allpaka-chat | 35 832 | 207 | 1 678 | 8 | 0 | 15 |
| allpaka-cli | 11 572 | 70 | 74 | 5 | 0 | 40 |
| allpaka-model | 8 963 | 54 | 159 | 22 | 1 | 16 |
| allpaka-core | 3 591 | 77 | 50 | 0 | 0 | 0 |
| allpaka-gguf | 1 987 | 29 | 40 | 2 | 0 | 0 |
| allpaka-mlx | 816 | 11 | 2 | 2 | 0 | 0 |
| allpaka-dtype | 452 | 2 | 2 | 11 | 0 | 0 |

Различных переменных `ALLPAKA_*` в коде: 153. `#[allow(...)]`: 41.

## Самые большие файлы

| Файл | LOC |
|---|---:|
| `allpaka-backend/src/gpu/metal.rs` | 10 236 |
| `allpaka-backend/src/gpu/shaders/kernels.metal` | 8 564 |
| `allpaka-backend/src/gpu/cuda/mod.rs` | 4 914 |
| `allpaka-backend/src/gpu/cuda/kernels.rs` | 3 703 |
| `allpaka-chat/src/observability.rs` | 2 845 |
| `allpaka-backend/src/gpu/cuda/runtime.rs` | 2 355 |
| `allpaka-chat/src/lib.rs` | 2 350 |
| `allpaka-chat/src/experiments.rs` | 2 165 |
| `allpaka-chat/src/online_evaluation.rs` | 1 822 |
| `allpaka-chat/src/provider.rs` | 1 645 |
| `allpaka-chat/src/swarm.rs` | 1 607 |
| `allpaka-cli/src/serve.rs` | 1 402 |

## Самые длинные функции (приблизительно)

| Функция | Строк |
|---|---:|
| `cuda/mod.rs:1068` `decode_token_one_fused` | ~1900 |
| `metal.rs:2928` `decode_token` | ~1600 |
| `metal.rs:5513` `encode_verify_tokens` | ~1370 |
| `metal.rs:4569` `decode_greedy_continue` | ~940 |
| `metal.rs:9075` `ffn_batch_grouped` | ~860 |
| `metal.rs:7252` `prefill_attn_block` | ~730 |
| `metal.rs:2311` `resolve_decode_layers` | ~600 |
| `allpaka-chat/src/lib.rs:1548` `turn_traced` | 577 |
| `allpaka-chat/src/lib.rs:393` `routes` | 371 |
| `harness_refine/validation.rs:77` `validate_champion` | 308 |
| `allpaka-cli/src/serve.rs:1097` `handle` | 300 |
| `allpaka-cli/src/serve.rs:288` `run_inference` | 292 |
| `model/prefill.rs:299` `forward_batch_fused_layers` | ~280 |
| `model/moe.rs:124` `moe_batch` | ~270 |

## Проверки на базовом коммите

- `cargo check --workspace --all-targets`: зелёный.
- `cargo test --workspace`: зелёный (включая `gpu_parity` 8/8, `gpu_moe_tail` 3/3).
- `cargo clippy --workspace --all-targets`: 5 предупреждений
  (2× `needless_range_loop` в `tests/gpu_moe_tail.rs`, `too_many_arguments` у
  `run_subagent`, `items_after_test_module` в `serve.rs`).
- `cargo fmt --all -- --check`: расхождения в ~38 файлах.

## Производительность

**Не измерена.** Парный замер `bc82c0a` против `279a667` (до и после первого
распила на модули) запланирован, но на момент снимка машина работала от батареи
под нагрузкой; по методологии проекта такие цифры недостоверны. До этого замера
правки горячих путей (волны 3–4 в `PLAN.md`) заблокированы.

Протокол: `cargo build --release`, затем `allpaka bench --engine <gguf>` с
`ALLPAKA_BENCH_REPORT`, пары A B A B, медианы сюда; модели —
`~/.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf` и `Qwen3-0.6B-Q8_0.gguf`.

| Модель | Фаза | До (`bc82c0a`) | После | Δ |
|---|---|---:|---:|---:|
| Qwen3-30B Q4_K_M | prefill tok/s | — | — | — |
| Qwen3-30B Q4_K_M | decode tok/s | — | — | — |
| Qwen3-0.6B Q8_0 | prefill tok/s | — | — | — |
| Qwen3-0.6B Q8_0 | decode tok/s | — | — | — |

## Ход работ

| Метрика | `279a667` | `576cd8b` |
|---|---:|---:|
| Самый большой `.rs` файл | 10 236 (`gpu/metal.rs`) | 4 914 (`gpu/cuda/mod.rs`) |
| `gpu/metal`: файлов / максимум строк | 1 / 10 236 | 13 / 3 250 (`decode.rs`) |
| `allpaka-cli/src/serve.rs` | 1 402 | 589 |
| Предупреждения clippy | 5 | 0 |
| Сборка без GPU (stub) | не компилируется | компилируется |
| Сборка с CUDA (type-check) | не компилируется | компилируется |
| `cargo test --workspace` | зелёный | зелёный |
