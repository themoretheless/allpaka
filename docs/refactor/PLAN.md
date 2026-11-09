# План рефакторинга

Связанные документы: [BASELINE](BASELINE.md), [INVENTORY](INVENTORY.md),
[ARCHITECTURE](ARCHITECTURE.md). Размер: S (часы), M (день), L (несколько дней).

## Инварианты

1. Поведение, HTTP-контракты, форматы файлов не меняются без явного решения.
2. Перенос кода, правка логики и форматирование — разные коммиты.
3. Каждый коммит собирается; перед push — `cargo test --workspace` и clippy.
4. Горячие пути правятся только с парным замером A B A B и greedy-паритетом.
5. Баг, найденный по дороге, чинится отдельным коммитом с тестом.
6. Ядра `.metal` / CUDA C и реестр отрицательных результатов не трогаются без
   новых данных. Megakernel не оживляется.

## Волна 0. Базовая линия и страховка

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 0.1 | `docs/refactor/*` | S | — | done |
| 0.2 | Парный бенчмарк `bc82c0a` против `279a667` + `verify --decode` | S | — | отложено: нужна сеть питания и спокойная машина |
| 0.3 | Проверка `gpu/stub.rs` против API бэкендов | S | низкий | todo |
| 0.4 | Характеризационные тесты: `route_gated_into_slice` (граница 256), `Config::from_gguf`, `dq_fmt`/`row_bytes`, `valid_id`, `state_file` | M | — | todo |

## Волна 1. Дефекты и фундамент

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 1.1 | B1: стаб приведён к API, compile-time проверка сигнатур | S | низкий | todo |
| 1.2 | B2: >256 экспертов — константа и явный fallback | S | низкий | todo |
| 1.3 | B3: `total_cmp` вместо `partial_cmp().unwrap()` | S | низкий | todo |
| 1.4 | B4: именованные таблицы `dq_fmt` и тест | S | низкий | todo |
| 1.5 | B5, B6 и все некэшированные чтения env → `RuntimePolicy` / `DebugFlags` | M | низкий | todo |
| 1.6 | Оставшиеся предупреждения clippy | S | низкий | todo |
| 1.7 | `cargo fmt --all` отдельным коммитом | S | — | todo |
| 1.8 | `// SAFETY:` для `unsafe impl` и срезов из сырых указателей | M | — | todo |

## Волна 2. Структурные перемещения (без изменения логики)

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 2.1 | `gpu/metal.rs` → `metal/{device,pipelines,select,telemetry,matvec,attn,ffn,decode,verify,prefill,mega}` | L | низкий–средний | todo |
| 2.2 | `gpu/cuda/mod.rs` → `{api,attn,ffn,decode,decode_fused,prefill}` | M | низкий | todo |
| 2.3 | `gpu/cuda/runtime.rs` → `runtime/{device,telemetry,select,gemm,q8,l2,matvec}` | M | низкий | todo |
| 2.4 | `gpu/common.rs`: общие `row_bytes`, окна весов, телеметрия, `attn_shape_ok` | M | низкий | todo |
| 2.5 | backend импортирует `GgmlType` из dtype; MLX-деквант → mlx; `anyhow` из dtype | S | низкий | todo |
| 2.6 | `core`: распил `plan.rs` и `fleet.rs` | M | низкий | todo |
| 2.7 | `cli`: `serve.rs` → `serve/{inference,stream,router}`; диспетчер из `main.rs` | S | низкий | todo |
| 2.8 | `chat`: `lib.rs` → `bootstrap`, `api/*`, `runtime/*` | M | низкий | todo |
| 2.9 | `chat`: `observability.rs` → `observability/*` | M | низкий | todo |
| 2.10 | `scripts/`: подкаталоги, общий harness, `node --test` | S | низкий | todo |
| 2.11 | `model/lib.rs`: сквозные тесты → `tests/` | S | — | todo |

## Волна 3. Логические рефакторинги

Не-горячий код:

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 3.1 | chat: `ApiError: IntoResponse` вместо 161 ручной ошибки | M | низкий | todo |
| 3.2 | chat: вся запись через `storage::atomic_write*` (B7, B8) | M | средний | todo |
| 3.3 | chat: убрать дубли `digest`, времени, обёрток sha256, bounded-read | S | низкий | todo |
| 3.4 | chat: `valid_id` — одно правило (B9) | S | **требует решения** | todo |
| 3.5 | chat: строковые status/kind/action → enum с прежними словами на проводе | M | средний | todo |
| 3.6 | chat: `turn_traced` → стадии хода | L | высокий | todo |
| 3.7 | chat: контекстные структуры вместо 8–14 параметров (`swarm`, `run_subagent`) | S | низкий | todo |
| 3.8 | chat: блокирующий fs из async → `spawn_blocking` | M | средний | todo |
| 3.9 | chat: реестр инструментов вместо `match name`; токены отмены в experiments, judge_runs, memory_extract, swarm | M | средний | todo |
| 3.10 | cli: таблица маршрутов вместо `starts_with` | M | средний | todo |
| 3.11 | core: `PoolPlacement`, общий co-residency, `PlanError` | M | низкий | todo |
| 3.12 | model: один `Architecture` enum; `ModelError`; дедупликация `loader.rs` | M | низкий | todo |
| 3.13 | gguf/mlx/dtype: типизированные ошибки | M | низкий | todo |

Горячий код (каждый пункт — отдельный A/B, заблокировано до 0.2):

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 3.20 | model: `ffn_residual()` вместо 4 копий | S | низкий | todo |
| 3.21 | model: `AttnMode` + `AttnInputs` | S | низкий | todo |
| 3.22 | model: `QuantMat::desc()` | S | низкий | todo |
| 3.23 | model: `GpuEligibility` при загрузке | M | средний | todo |
| 3.24 | model: `enum Mixer`, `Norm`, общий `Block`, `RoutingPolicy` | L | средний | todo |
| 3.25 | backend: `decode_token` → функции кодирования стадий | L | средний | todo |
| 3.26 | backend: `decode_token_one_fused` → контекстная структура | L | средний | todo |
| 3.27 | backend: request-структуры вместо 13–19 параметров | M | низкий | todo |

## Волна 4. Оптимизация по замерам

| # | Гипотеза | Размер | Статус |
|---|---|---|---|
| 4.1 | Кэш дескрипторов `token_layers` на сессию/модель | M | todo |
| 4.2 | Эффект от снятия некэшированных чтений env на CUDA-путях | S | todo |
| 4.3 | Что мешает не-Neox RoPE (Llama, Mistral) идти быстрыми GPU-путями | M | todo |
| 4.4 | Studio: блокирующий I/O и contention на блокировках — по профилю | M | todo |

Отрицательные результаты записываются в реестр проекта.

## Волна 5. Чистка и отчёт

| # | Задача | Размер | Статус |
|---|---|---|---|
| 5.1 | Megakernel и проигравшие варианты ядер → `experimental` | M | **требует решения** |
| 5.2 | Справочник `ALLPAKA_*`, генерируемый из `RuntimePolicy` | M | todo |
| 5.3 | `docs/roadmap.md`: закрыть D6, обновить D3/D4 | S | todo |
| 5.4 | Тесты для непокрытых модулей; общие фикстуры Rust ↔ Python SDK | M | todo |
| 5.5 | Таблица до/после в BASELINE, остаточный бэклог | S | todo |

## Требует решения

1. **3.4** `valid_id`: какое правило эталонное (лимит длины, набор символов).
2. **5.1** удалять ли недостижимый megakernel-код или держать за feature.
3. **3.1** если в ответах об ошибках найдётся несогласованность формата.
4. Любой шаг, замедляющий горячий путь больше шума (±3%).
