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

### До и после рефакторинга (2026-11-10)

Бинарь `bc82c0a` (A) против `54b19cc` (B), pp480 + tg32, 4 пары с чередованием
порядка, медиана пар, tok/s.

| Модель | Фаза | A | B | B / A |
|---|---|---:|---:|---:|
| Qwen3-0.6B Q8_0 | prefill | 8 277 | 8 296 | 1.00 |
| Qwen3-0.6B Q8_0 | decode | 376.8 | 388.4 | 1.03 |
| Qwen3-8B Q4_K_M | prefill | 263.6 | 264.9 | 1.01 |
| Qwen3-8B Q4_K_M | decode | 38.6 | 39.5 | 1.03 |
| Qwen3-30B-A3B Q4_K_M | prefill | 323.1 | 340.9 | 1.06 |
| Qwen3-30B-A3B Q4_K_M | decode | 72.6 | 75.7 | 1.04 |

Вывод: признаков регрессии от распила нет, B не медленнее A ни в одной строке.
Обратное («стало быстрее») из этих данных не следует: разброс внутри одной
стороны доходит до 2× (8B decode 23–50).

**Абсолютным числам этого замера верить нельзя.** Оба бинаря в 3–4.4 раза
медленнее записанного для того же коммита `bc82c0a` в
`docs/benchmarks/allpaka-vs-llama-metal-status.md` (30B prefill 1 460–1 500,
8B prefill 836): часы GPU показывают 1 340–1 520 мс исполнения на prefill 30B
против ~321 мс. Причина на стороне машины: в простое, без наших процессов,
`ioreg` показывал загрузку GPU 46–75% (WindowServer и GPU-процессы нескольких
Electron-приложений), параллельно шли сборки других проектов. llama.cpp в том же
окне просела в 1.5–3.5 раза с большим разбросом.

Следствие: сравнение с llama.cpp из этого окна непригодно (кроме 0.6B, где
нагрузка мала и результат согласуется с записью: prefill 1.16×, decode 1.28×),
а A/B-эксперименты с эффектом меньше ~10% сейчас неразличимы. Перемерить на
машине со свободным GPU: `scripts/bench-matrix.sh` и пары A B A B.

### Серия переключателей под нагрузкой (2026-11-10): неразличимо

11 конфигураций по 6 раундов на 8B и 30B, порядок сдвигается каждый раунд;
`base` и `base2` идентичны и измеряют шум. Отношение к `base` в том же раунде.

| Модель, фаза | `base2` / `base`: медиана (мин–макс) | Самое большое отклонение среди переключателей |
|---|---|---|
| 8B prefill | 0.99 (0.80–1.06) | `mv_id0` 1.06 |
| 8B decode | 0.875 (0.56–1.12) | `mv_tg64` 0.77, `mm_pipe0` 1.13 |
| 30B prefill | 0.97 (0.63–1.07) | `mv_tg64` 1.08, `mm_pipe0` 0.93 |
| 30B decode | 1.06 (0.76–1.53) | `attn_s16` 1.14, `mv_tg256` 1.14 |

Две одинаковые конфигурации расходятся по медиане на величину до 12.5%, а в
одной паре до 1.5 раза. Ни один переключатель (`MV_TG`, `ATTN_S8`, `Q8_NR0`,
`MM_K64`, `MMID_K64`, `MM_PIPE`, `MV_ID`, `SHARED_STAGGER`) не выходит за этот
коридор, часть «эффектов» видна на фазах, которые переключатель не затрагивает.
Это не отрицательный результат по переключателям, а отсутствие результата:
серию нужно повторить на свободной машине. Умолчания не менялись.

## Ход работ

| Метрика | `279a667` | после волны 2 |
|---|---:|---:|
| Самый большой `.rs` файл (без тестов и текста ядер) | 10 236 (`gpu/metal.rs`) | 3 250 (`gpu/metal/decode.rs`) |
| `gpu/metal`: файлов / максимум строк | 1 / 10 236 | 13 / 3 250 |
| `gpu/cuda/mod.rs` | 4 914 | 254 (+6 модулей, максимум 1 914 в `decode_fused.rs`) |
| `gpu/cuda/runtime.rs` | 2 355 | 41 (+7 модулей, максимум 827) |
| `allpaka-chat/src/lib.rs` | 2 350 | 348 |
| `allpaka-chat/src/observability.rs` | 2 845 | 8 модулей, максимум 526 (тесты 1 053) |
| `allpaka-cli/src/serve.rs` | 1 402 | 589 |
| `allpaka-core`: `plan.rs` / `fleet.rs` | 628 / 440 | 259 / 279 |
| Предупреждения clippy | 5 | 0 |
| `cargo fmt --check` | ~38 файлов с расхождениями | чисто |
| Сборка без GPU (stub) | не компилируется | компилируется |
| Сборка с CUDA (type-check) | не компилируется | компилируется |
| `cargo test --workspace` | зелёный | зелёный |
| UI-тесты Studio | 58 файлов без раннера | 58/58 через `scripts/studio-ui-tests/run.sh` |

Все переносы проверены сравнением строк: отличаются только заголовки модулей,
объявления `mod`/`use` и пути `include_str!` (на каталог глубже).
