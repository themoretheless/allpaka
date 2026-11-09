# Инвентаризация

Результат трёх аудитов на коммите `279a667`. Номера строк относятся к этому
коммиту и сдвигаются по мере работ. Статус: `todo`, `wip`, `done`, `отложено`.

## Вероятные дефекты (чинятся отдельными коммитами с тестом)

| # | Где | Что | Статус |
|---|---|---|---|
| B1 | `backend/src/gpu/stub.rs` | API отстал от бэкендов: нет `ffn_batch_grouped`, `attend_project`, region-функций, `stats`, `gpu_time_stats`, `minimum_kv_capacity`; есть неиспользуемые старые имена. Подтверждено компиляцией: сборка без GPU не проходила. Регрессия внесена распилом `c983860` | done (`4b12e1a`) |
| B2 | `model/prefill.rs:312,422` | Стековый буфер на 256 экспертов; при большем числе `k` молча обрезается | не дефект: буфер держит top-k (`n_used`), а не всех экспертов |
| B3 | `backend/gpu/metal.rs:1279` | `partial_cmp().unwrap()` паникует на NaN | done (`704972c`) |
| B4 | `metal.rs:4542` и `cuda/runtime.rs:738` | `dq_fmt`: разные коды формата на бэкендах (Metal BF16=6; CUDA Q5_0=6, BF16=9), нет теста соответствия ядрам | todo |
| B5 | `model/prefill.rs:53` | `ALLPAKA_DUMP_HIDDEN` читается из окружения на каждый `forward_batch` | done (`9e538b5`) |
| B6 | `backend/src/runtime.rs:16-29` | Хелперы `on/off/set/present` игнорируют аргумент `_def` | не дефект: умолчание задаёт само соглашение флага, таблица согласована |
| B10 | `backend/src/gpu/cuda/runtime.rs:680,739` | CUDA-сборка не компилировалась: `match` по `GgmlType` не покрывал `MlxQ4`/`MlxQ8`, добавленные с `allpaka-dtype` | done (`523c5e9`) |
| B7 | `chat/state_file.rs` | Чтение-изменение-запись целого списка без блокировки: потеря обновлений при одновременных сохранениях | todo. Сама запись атомарна (temp + rename); открытым остаётся только гонка чтение-изменение-запись |
| B8 | ~20 модулей chat | Неатомарный `fs::write` | в основном ложная находка: почти все вызовы в тестах. Реальные два места исправлены (`e3e5dce`) |
| B11 | `scripts/test-studio.py:2575` | Интеграционный тест Studio падал с `41269b6` (20 октября): тот коммит дал участнику swarm завершающий запрос без инструментов после исчерпания лимита шагов, а тест и `docs/studio.md` описывали прежнее поведение. Код оставлен, тест и документ приведены к нему; тест проходит целиком (121 проверка) | done |
| B9 | chat, 8 копий `valid_id` | Правила расходятся (лимит 80 против 100, hex-only в `lib.rs`) | требует решения |

## allpaka-backend

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `gpu/metal.rs` | 10 236 | Весь бэкенд в одном файле; `decode_token` ~1600 строк, 8 ответственностей; 67 unsafe без `SAFETY`; 13 некэшированных чтений env; 38 `pub fn` при 0 `pub(crate)` | P0 | todo |
| `gpu/cuda/mod.rs` | 4 914 | `decode_token_one_fused` ~1900 строк с замыканиями; 11 некэшированных чтений env; 66 unsafe без `SAFETY`; нет тестов | P0 | todo |
| `gpu/cuda/runtime.rs` | 2 355 | Лаунчеры с 11–18 параметрами; чтение env на каждый запуск matvec (:1554–2347); статики и `GPU` полностью `pub`; 5× `allow(dead_code)` | P0 | todo |
| `gpu/cuda/ggml.rs` | 549 | 32 unsafe без `SAFETY`; функции с 12–16 параметрами | P2 | todo |
| `gpu/types.rs` | 418 | `decode_token_outcome` 155 строк; нет тестов | P2 | todo |
| `gpu/stub.rs` | 164 | См. B1 | P0 | todo |
| `quantmat/mod.rs` | 902 | Функции с 13–19 параметрами (:486–629); 20 unwrap | P1 | todo |
| `quantmat/{dot_f32,dot_neon,q8_acts}.rs` | ~960 | Нет тестов; unsafe почти без `SAFETY` | P2 | todo |
| `ops/*` | ~1 440 | Тесты только в `mod.rs`; unsafe без `SAFETY` в norm/blas | P2 | todo |
| `runtime.rs` | 152 | См. B6; покрывает только Metal-флаги | P1 | todo |
| Дубли Metal ↔ CUDA | — | телеметрические атомики, окна весов, `row_bytes`, `attn_shape_ok`, `SharedRegion`, `scale_min_k4` в 4 местах | P1 | todo |
| Недостижимое | — | megakernel (`mega_enabled()` всегда false); варианты, задокументированные медленнее: `RTOPK`, `CFUSE`, `RFUSE`, `DECODE_GUFUSE`, `SWFUSE`, CUDA `GGML_MMVQ`, `K5120`; `Q2_ILV` | требует решения | todo |

## allpaka-dtype, allpaka-gguf, allpaka-mlx

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `dtype/lib.rs` | 452 | MLX-деквант (:362–405) живёт не в том крейте; `anyhow` ради одной функции; 11 unsafe без `SAFETY` | P1 | todo |
| backend → dtype | — | `GgmlType` импортируется транзитом через `allpaka_gguf` | P1 | todo |
| `gguf/{metadata,dequant,vision,lib}.rs` | ~1 660 | 40 unwrap/expect в библиотеке; `anyhow` вместо типизированных ошибок | P2 | todo |
| `gguf/tensors.rs` | 317 | Нет тестов | P2 | todo |
| `mlx/{container,lib,shapes}.rs` | ~380 | Нет unit-тестов | P3 | todo |

## allpaka-model

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `model/mod.rs` | 202 | 51 `pub(crate)`-элемент: подмодули расширяют один `Model` и лезут во внутренности друг друга; 12 env-хелперов | P1 | todo |
| `model/decode.rs` | 762 | «норма → FFN → residual» трижды (:487, :543, :589); `token_layers` (:213, 185 строк) пересобирается на каждый токен; 4× `expect("gdn arch")` подряд | P1, горячий | todo |
| `model/prefill.rs` | 610 | `forward_batch_fused_layers` 279 строк; B2, B5; `n_used <= 8` захардкожен | P1, горячий | todo |
| `model/moe.rs` | 641 | `moe_batch` 271 строка; `Ffn::Moe` 12 полей (политика роутинга размазана); буферы `[_; 256]` в 4 местах | P1, горячий | todo |
| `model/attention.rs` | 551 | `run_cpu_attention_from_qkv`: 15 параметров, 2 bool-флага (`false, true` на вызовах) | P1, горячий | todo |
| `model/loader.rs` | 455 | `load_range` 262 строки; сборка trunk и MTP слоёв — копипаст (:117–170 против :211–245); нет тестов | P2 | todo |
| `model/mtp.rs` | 162 | Копия decode-блока внимания (:45–110); `MtpLayer` дублирует 6 полей `Layer` | P2 | todo |
| `Layer` | — | Сумма типов, замаскированная под структуру (`attn`/`gdn` взаимоисключающие `Option`) | P1 | todo |
| Пригодность GPU-пути | — | Проверка `Neox && !cpu_attn && !no_tokenbuf` в 6 местах | P1 | todo |
| `config.rs` + `requirements.rs` | 450 | Две таксономии архитектур на строках, могут расходиться; `config.rs` без тестов | P2 | todo |
| `kv.rs` | 848 | 16 unsafe без `SAFETY`; `unsafe impl Send/Sync for Region` без обоснования; заявленный шов под MLA без типа | P2 | todo |
| Ошибки | — | `anyhow` во всей библиотеке; «GPU отказался» неотличим от ошибки | P2 | todo |
| `lib.rs` | 942 | ~910 строк — сквозные тесты; место им в `tests/` | P3 | todo |

## allpaka-core

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `plan.rs` | 628 | `search_tree` 190 строк, `build_plan` 102; `Verdict::Infeasible { reason: String }` | P2 | todo |
| `fleet.rs` | 440 | `search` 112, `fleet` 102 (валидация и поиск вместе); `Result<_, String>` | P2 | todo |
| `fleet.rs` ↔ `replicate.rs` | — | `Placement` и `Replica` почти дублируют друг друга; масштабирование co-residency повторено | P2 | todo |
| `cost.rs`, `presets.rs` | ~200 | Нет прямых тестов | P3 | todo |

## allpaka-cli

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `serve.rs` | 1 402 | `handle` 300, `run_inference` 292, `dispatch_connection` 181 (цепочка `starts_with`); тестовый модуль посреди файла | P1 | todo |
| `main.rs` | 773 | `main` 235 строк; нет тестов | P2 | todo |
| `serve/http.rs` | — | Ошибки JSON собираются вручную на местах | P2 | todo |
| `serve/rag_tools.rs` | 413 | `match tool_name`; нет тестов | P2 | todo |
| `serve/resources.rs`, `launch.rs`, `http.rs`, `airbug.rs` | — | Глобальные статики (`STATE`, `BIND`, `MAX_BODY`) | P3 | todo |
| `bench.rs`, `pipeline.rs`, `verify.rs` | ~1 080 | Функции по 175–192 строки; нет тестов | P3 | todo |

## allpaka-chat

| Модуль | LOC | Запахи | Приоритет | Статус |
|---|---:|---|---|---|
| `lib.rs` | 2 350 | `turn_traced` 577 строк; `routes` 371; `run` 178; 57× `lock().unwrap()`; 31 ручная ошибка; 16 обработчиков с `Json<Value>`; нет тестов | P0 | todo |
| `observability.rs` | 2 845 | Шесть ответственностей в файле; собственная атомарная запись; 56 вызовов `std::fs` при async-обработчиках | P1 | todo |
| Ошибки API | — | 161 ручной `error(StatusCode…)`, нет `ApiError: IntoResponse` | P1 | todo |
| Хранение | — | Пять способов: `storage`, `state_file`, ручной temp+rename, голый `fs::write`, `data_lock`; см. B7, B8 | P1 | todo |
| Дубли | — | `valid_id` ×8, `digest` ×5, время ×3, обёртки `valid_sha256_hex` ×3, bounded-read ×3 | P1 | todo |
| Строковые статусы | — | background, observability, experiments, matrix_jobs, review_queues, guardrail_policies, online_evaluation, online_sources, model_evaluators, harness_refine/types, types (role, source), prompts | P2 | todo |
| `swarm.rs` | 1 607 | Функции с 10–14 параметрами; нет отмены | P2 | todo |
| `background.rs` | 1 136 | 95 unwrap/expect вне тестов; статус строкой | P2 | todo |
| `experiments.rs`, `judge_runs.rs`, `memory_extract.rs` | ~3 360 | Нет токенов отмены; блокирующий fs | P2 | todo |
| `tools.rs` | 548 | `match name`; статическая карта `LOCKS` | P2 | todo |
| Без тестов | — | `lib.rs`, `harness_refine/{engine,workspace,store,validation,http}` (частично покрыты `tests.rs`), `judge_runs`, `plugins`, `history`, `rag`, `state_file` | P2 | todo |
| Python SDK | 2 796 | Хеши guardrail-политик пересчитываются независимо от Rust, общих фикстур нет | P3 | todo |

## scripts/

86 файлов в одном каталоге; 61 `test-studio-*-ui.js` повторяет одну обвязку
(`fs` + `vm` + чтение `web/app.js`); общего раннера нет, CI их не запускает.
Приоритет P2, статус todo.

## Устаревшее в документации

- `docs/roadmap.md` D6: уже сделано, `PlanItem.status` — enum `PlanStatus`
  (`chat/src/types.rs:298`).
- README: описаны 9 переменных `ALLPAKA_*` из 153.
