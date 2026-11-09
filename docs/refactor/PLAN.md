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
| 0.2 | Парный бенчмарк `bc82c0a` против `main` + `verify --decode` | S | — | частично: A/B сделан, регрессии нет (BASELINE); абсолютные числа и сравнение с llama.cpp непригодны из-за занятого GPU, `verify --decode` не запускался. Перемерить на свободной машине |
| 0.3 | Проверка `gpu/stub.rs` против API бэкендов | S | низкий | done: `scripts/check-backends.sh` |
| 0.4 | Характеризационные тесты: `route_gated_into_slice` (граница 256), `Config::from_gguf`, `dq_fmt`/`row_bytes`, `valid_id`, `state_file` | M | — | todo |

## Волна 1. Дефекты и фундамент

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 1.1 | B1: стаб приведён к API; проверка — `scripts/check-backends.sh` (stub и CUDA) | S | низкий | done (`4b12e1a`, `523c5e9`) |
| 1.2 | B2: >256 экспертов | S | низкий | не дефект: буфер по `n_used` (top-k), а не по числу экспертов; `moe.rs` при >256 уходит в кучу. Осталась только именованная константа |
| 1.3 | B3: `total_cmp` вместо `partial_cmp().unwrap()` | S | низкий | done (`704972c`) |
| 1.4 | B4: именованные таблицы `dq_fmt` и тест | S | низкий | todo |
| 1.5 | B5 и все некэшированные чтения env → `RuntimePolicy` / `DebugFlags` | M | низкий | wip: B5 done (`9e538b5`); B6 не дефект (аргумент `_def` дублирует соглашение о флаге, таблица согласована); CUDA и Metal — todo |
| 1.6 | Оставшиеся предупреждения clippy | S | низкий | done (`9bd193d`): 0 предупреждений |
| 1.7 | `cargo fmt --all` отдельным коммитом | S | — | done: `fmt --check` проходит |
| 1.8 | `// SAFETY:` для `unsafe impl` и срезов из сырых указателей | M | — | todo |

## Волна 2. Структурные перемещения (без изменения логики)

| # | Задача | Размер | Риск | Статус |
|---|---|---|---|---|
| 2.1 | `gpu/metal.rs` → `metal/{device,region,pipelines,select,telemetry,matvec,attn,args,ffn_grouped,decode,verify,prefill,mega}` | L | низкий–средний | done (`4b8ccc9` и следующий): чистый перенос, проверен сравнением строк; `mod.rs` 527 строк. Pipelines остались внутри `impl Gpu` в `device.rs` |
| 2.2 | `gpu/cuda/mod.rs` → `{region,attn,decode,decode_fused,prefill,ffn_grouped}` | M | низкий | done (`bd55b83`): 4 914 → 254 строки в `mod.rs`; проверено только type-check'ом |
| 2.3 | `gpu/cuda/runtime.rs` → `runtime/{device,select,embed,gemm,q8,l2,matvec}` | M | низкий | done (`bd55b83`) |
| 2.4 | `gpu/common.rs`: общие `row_bytes`, окна весов, телеметрия, `attn_shape_ok` | M | низкий | todo |
| 2.5 | backend импортирует `GgmlType` из dtype; MLX-деквант → mlx; `anyhow` из dtype | S | низкий | wip: импорт done; перенос MLX-декванта и `anyhow` — todo |
| 2.6 | `core`: поиск из `plan.rs` и `fleet.rs` → `plan/search.rs`, `fleet/search.rs` | M | низкий | done (`ce57524`). Длина `search_tree` (190 строк) — задача 3.11 |
| 2.7 | `cli`: `serve.rs` → `serve/{inference,router}`; диспетчер из `main.rs` | S | низкий | wip: inference и router вынесены (`576cd8b`), `serve.rs` 1 402 → 589; `handle` (300 строк) и `main.rs` требуют правки логики → волна 3 |
| 2.8 | `chat`: `lib.rs` → `api/{router,handlers}`, `runtime/{actor,turn,subagent,exec}` | M | низкий | done (`8c5bb55`): `lib.rs` 2 350 → 348 |
| 2.9 | `chat`: `observability.rs` → `observability/{model,trace,ingest,store,receipts,guardrails,api,tests}` | M | средний | done (`cb9a9ec`): `mod.rs` 63 строки кода |
| 2.10 | `scripts/`: UI-тесты Studio в `studio-ui-tests/` с раннером | S | низкий | done (`d6fe2db`): 58 тестов, `run.sh`. Остальные скрипты оставлены на месте: на них ~130 ссылок в документах и внешней автоматизации, перенос не окупается. Общий harness для 58 тестов — todo |
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

## Заметки по ходу работ

- Распил `2140b62..4fbe7ab` не был чистым переносом: вместе с ним приехал новый
  код (пул потоков, AVX2-пути, бинарный протокол pipeline, trie в prefix-кэше) и
  две регрессии сборки под cfg, невидимые на macOS (B1, B10). Поэтому перед
  каждым push теперь запускается `scripts/check-backends.sh`.
- Перенос кода проверяется сравнением мультимножеств строк старого файла и новых
  модулей (с точностью до `pub(super)`): отличаться должны только заголовки
  модулей и объявления `mod`/`use`.
- Два пункта аудита оказались не дефектами (B2, B6), см. INVENTORY.

## Требует решения

1. **3.4** `valid_id`: какое правило эталонное (лимит длины, набор символов).
2. **5.1** удалять ли недостижимый megakernel-код или держать за feature.
3. **3.1** если в ответах об ошибках найдётся несогласованность формата.
4. Любой шаг, замедляющий горячий путь больше шума (±3%).
