# Целевая архитектура

## Крейты и направление зависимостей

```
allpaka-dtype   типы тензоров и скалярные преобразования; без зависимостей проекта
   ↑
allpaka-gguf    чтение GGUF, CPU-деквант        allpaka-mlx   safetensors + MLX-деквант
   ↑
allpaka-backend CPU-ядра (ops, quantmat), GPU-бэкенды (metal, cuda, stub), политика рантайма
   ↑
allpaka-model   прямой проход: загрузка, prefill, decode, KV, спекуляция, токенизатор
   ↑
allpaka-cli     бинарник: serve, bench, verify, watch          allpaka-core  планировщик кластера (чистая математика)
allpaka-chat    Studio: axum-приложение, хранилища, оценка, harness
```

Правила:
- Зависимости только вверх по схеме; `allpaka-core` не знает о движке.
- Backend берёт `GgmlType` из `allpaka-dtype`, а не транзитом через gguf.
- Библиотечные крейты возвращают типизированные ошибки; `anyhow` остаётся в
  бинарнике и в композиционных корнях.
- По умолчанию `pub(crate)`; `pub` только то, что реально потребляет другой крейт.

## Backend: один контракт, три реализации

`gpu/mod.rs` — единственная публичная поверхность. За ней `metal/`, `cuda/`,
`stub.rs` с одинаковым набором функций; совпадение сигнатур проверяется при
компиляции. Общее для бэкендов (`row_bytes`, окна весов, телеметрия,
`attn_shape_ok`) живёт в `gpu/common.rs`.

```
gpu/metal/   device  pipelines  select  telemetry  matvec  attn  ffn  decode  verify  prefill  mega
gpu/cuda/    api  attn  ffn  decode  decode_fused  prefill
             runtime/{device, telemetry, select, gemm, q8, l2, matvec}
```

Переключатели `ALLPAKA_*` читаются один раз в `RuntimePolicy`
(`backend/src/runtime.rs`); горячие пути получают готовые флаги. Из него же
генерируется справочник переменных.

Варианты ядер, проигравшие замер, изолируются в `experimental`, чтобы A/B
оставался возможным, но не мешал чтению основного пути.

## Model: слой как сумма типов

Сейчас `Layer` — структура с взаимоисключающими `Option`-полями, а вариации
архитектур размазаны по условиям. Цель:

```rust
struct Block { mixer: Mixer, ffn: Ffn, attn_norm: Norm, ffn_norm: Norm }
enum Mixer { Attention { weights, qk_norm: Option<QkNorm>, bias }, Gdn(GdnLayer) }
struct Norm { f32: Vec<f32>, raw: &'a [u8] }          // CPU- и GPU-представление вместе
struct Router { mat, bias, policy: RoutingPolicy }    // вместо 5 полей в Ffn::Moe
enum Architecture { Qwen3, Qwen3Moe, Qwen35, Llama3, Mistral, DeepSeekMla, .. }
struct GpuEligibility { token_buffer, fused_prefill, gpu_attend, .. } // один раз при загрузке
```

`MtpLayer` строится из того же `Block`. Швы под новые семейства: `RopeSpec`
(стиль, построитель частот, секции) и `KvLayout` (линейный f16 сейчас, MLA
позже). Трейт вводится только при появлении четвёртой разновидности mixer.

## CLI serve

`serve.rs` остаётся композиционным корнем. Внутри: `router` (таблица маршрутов
вместо цепочки `starts_with`), `inference` (prefill, sampling), `stream` (SSE),
плюс уже существующие `http`, `messages`, `sampler`, `session`, `template`,
`resources`, `memory`.

## Studio (allpaka-chat)

```
bootstrap        сборка App из хранилищ: единственный композиционный корень
api/             router, sessions, plan, background: тонкие axum-обработчики
runtime/         actor (очередь действий сессии), turn (стадии хода), subagent
storage          единственный путь записи на диск (атомарно), id, sha256, время
observability/   model, store, ingest, aggregate, trace, http
```

Сквозные решения:
- Ошибки API: один `ApiError: IntoResponse`; обработчики возвращают `Result`.
- Запись состояния: только `storage::atomic_write*`; чтение-изменение-запись
  списков под одной блокировкой.
- Статусы и виды: enum с прежними словами на проводе.
- Блокирующий файловый ввод-вывод из async — через `spawn_blocking`.
- Ход (`turn`) — конвейер стадий: вызов провайдера → цикл инструментов →
  guardrails → сохранение; каждая стадия тестируется отдельно.

## Скрипты

```
scripts/studio-ui-tests/  harness.js + *.test.js, запуск `node --test`
scripts/serving-tests/    scripts/bench/    scripts/eval/    scripts/git/
```
