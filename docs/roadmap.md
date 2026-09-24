# Roadmap

Единственный источник правды по развитию allpaka. Правила ниже — часть
документа, а не пожелание:

1. Строка без критерия готовности (**DoD**) не считается задачей.
2. Число без ссылки на артефакт не считается результатом.
3. У каждого внешнего источника либо «актуален», либо «заменён X».

Сверено с рабочим деревом и с wiki-снимком. Даты здесь не проставляются:
источник правды по времени — git, здесь фиксируется состояние.

## Провенанс: что с чем стало

| Было | Где жило | Стало |
|---|---|---|
| 8 пунктов рантайма | `docs/runtime-roadmap.md` | Трек A (там же детализация таблицы) |
| Ближайшие шаги | `README.md`, «Что дальше» | Трек B (из README теперь ссылка сюда) |
| Приоритеты P0/P1/P2 | `docs/research/chat-2026-09-21/README.md` | Трек C (P0 закрыт, P1/P2 перенесены) |
| Статус движка | `wiki://allpaka-engine-status` | Часть строк закрыта кодом (профили, autotune-кэш, multi-model serve, отмена/дедлайны); остаток «remaining architecture work» разложен по A3–A7. Страница — снимок на 2026-09-02 |
| `codex://allpaka/gpu-recovery/2026-08-29` | RAG-корпус (`updates`) | **Заменён**: `wiki://allpaka-engine-status` объявляет его superseded. В планировании не используется |
| MTP | `docs/mtp.md` | «Снято / парковка» с условием возврата |
| Опровергнутые оптимизации | `docs/moe-prefill.md` | «Снято / не делать без новых данных» |
| `google/ax` (Go-оркестратор агентских нагрузок на Kubernetes) | внешний, актуален на 2026-09-24 | Заимствовано ровно одно: форма статуса (`phase` как одно слово, `conditions` как деталь, `watch` как переходы) → `allpaka watch`. Песочницы, egress-allowlist, `suspend`/`resume` акторов, Redis-контролплейн и `atespace` не перенесены: у allpaka узкое место - пропускная способность памяти и хвост задержки линка, а не планирование кластера. `suspend`/`resume` к тому же закрыты на уровне сессии в Studio |
| Структура репо и метрики «непонятности» | вне репо, в рабочем плане | Разбор по шести метрикам проведён и закрыт коммитами начиная с `90f259d` (гинтлинок, `state_file`, кластеры Studio, `/health`, README). Сам документ с метриками в `docs/` заведён не был: `roadmap.md` объявлен единственным источником правды, а `docs/benchmarks` уже занимает 1042 из 1207 трекаемых файлов |

---

## Трек A — движок и сервинг

Источник: `docs/runtime-roadmap.md`. Статусы сверены с кодом.

### A1. Сопоставимые бенчмарки — в работе

Точный token replay, KV-точность, matched contexts. Часть сделана: метаданные
работы, строгий GPU-coverage, alternating context-matched прогоны.

**DoD:** token replay против llama.cpp на одном и том же токен-стриме;
артефакт по стабильной JSON-схеме с явным флагом comparability. Для MoE
идентичные формы тензоров не означают идентичный роутинг экспертов — это
должно быть отражено в отчёте.

**Пруф:** `docs/benchmarks/qwen3-30b-m4-max-2026-09-02.md`, `docs/benchmarks/*.json`.

### A2. Изолированные сессии, отмена, дедлайны, backpressure — сделано

**Пруф:** `scripts/test-serving-controls.py`; контракт — `docs/serving.md`
(`request_id`, `timeout_ms`, `session_id`, `POST /v1/requests/{id}/cancel`, 408/429).

### A3. Единый бюджет памяти — в работе

Сделано: reservations для mapped-весов, полной capacity prefix-cache и живых
сессий; отказ до старта стрима; освобождение; `peak_reserved_bytes` в `/stats`.

Осталось: единый KV/scratch admission. GPU scratch, вспомогательные аллокации
модели и HTTP-буферы не покрыты — это reservation-счётчики, а **не** hard-cap
процесса.

**DoD:** либо покрыть эти статьи, либо честно назвать границу в `docs/serving.md`
и в самом ответе API (сейчас формулировка уже честная — довести до конца).

### A4. KV-блочный prefix cache + hit/miss — ожидает

Сейчас `/stats` отдаёт residency (`prefix_cache_entries`, `prefix_cache_bytes`),
а не попадания.

**DoD:** hit/miss и reused-token счётчики под нагрузкой; reuse наблюдаем через
`usage.prompt_tokens_details.cached_tokens`.

### A5. Step scheduler + chunked prefill — ожидает

Сейчас `batching_mode: model-aware-admission`, `kernel_batching: false`, инференс
сериализован одним владельцем.

**DoD:** chunked prefill плюс метрика максимальной decode-задержки под
конкурентной нагрузкой; задержка ограничена, а не «как получится».

### A6. Fused batching независимых сессий на GPU — ожидает, строго после A5

**DoD:** parity-тесты батча против последовательного выполнения; `/stats`
перестаёт сообщать `kernel_batching: false`.

### A7. Workload-aware autotune — ожидает

Профили `safe` / `balanced` / `max-performance` и версионированный кэш autotune
уже есть.

**DoD:** выбор профиля под рабочую нагрузку (latency / throughput / memory), а
не только под пару модель+устройство.

### A8. Structured output + валидированные tool calls — ожидает

**DoD:** constrained decoding по JSON-схеме инструмента и явный отказ при
несоответствии аргументов; тест на «модель выдала мусор», а не только happy path.

---

## Трек B — распределённый инференс

Источник: `README.md`, «Что дальше». Планировщик (`plan.rs`, `fleet.rs`,
`fabric.rs`) есть; исполнителя разреза нет: `allpaka launch` печатает команды
для llama-server, а разрез собственного движка по слоям через сеть отсутствует.

### B1. Loopback-разрез на два процесса

**DoD:** одна модель разрезана по слоям между двумя процессами на одной машине;
greedy-стрим бит-совместим с монолитным прогоном.

**Exit-критерий трека:** если бит-паритет недостижим — трек закрывается, модель
стоимости в `plan.rs` остаётся как есть, а README переписывается без обещания
исполнителя. Это точка честного отказа, а не «продолжим как-нибудь».

### B2. Разрез по измеренному линку

**DoD:** predicted tok/s по `allpaka.toml` против measured на mac↔PC в пределах
заявленного допуска; неизмеренный стык по-прежнему блокирует разрез.
Напоминание из README: хоп односторонний, на проход платится p99, а не среднее.

### B3. Спекуляция на разрезе + запуск по конфигу

**DoD:** спекуляция поверх B2 не меняет сравнение «разрез vs монолит» (она
множитель к обеим конфигурациям, а не лечение задержки); запуск идёт по конфигу
планировщика вместо печати чужих команд.

Форма наблюдения за таким запуском уже есть для одного сервера: `allpaka watch`
(`crates/allpaka-cli/src/watch.rs`) сводит опрос ручек в `phase` плюс
`conditions` и печатает переходы. Разрез унаследует тот же словарь, а не
новый: `Reachable` становится `StageReachable` на каждый узел, и к нему
добавляются `LinkVerified` (стык измерен, иначе по правилам B2 разрез
запрещён) и `LayersPlaced`. Пока подъём разреза не запускается программно
(`launch` печатает команды), кода для этих условий нет и заводить их заранее
нельзя - классификатор обязан оставаться функцией над измеренным.

---

## Трек C — Studio

Источник: `docs/research/chat-2026-09-21/README.md` (P1/P2). P0 закрыт.

### C1. Поиск/RAG по папкам проекта — ожидает

**DoD:** цитаты с provenance (файл + строки), видимый бюджет контекста, выбор
файлов через `@`; тест на актуальность — изменённый файл не выдаётся из кэша как
свежий.

Расхождение к устранению в этом же пункте: `docs/studio.md` пишет, что MCP «в
этот цикл пока не включён», но `crates/allpaka-chat/src/mcp.rs` — уже работающий
streamable-HTTP клиент к локальному `rag-mcp` (префикс `rag_`, только read-only
инструменты при `read_only`). Документ надо привести в соответствие.

### C2. Checkpoints/undo, git diff, безопасные патчи, shell — ожидает

**DoD:** отмена с последующей сверкой фактического состояния; применение патча
проверяется байт-в-байт на исходной копии — по образцу уже существующего
контрактного теста `edit_file` (apply возвращённого diff через `git apply`).

### C3. Извлечение PDF/DOCX/XLSX/OCR/URL/аудио/видео — ожидает

Сейчас: PDF/DOCX/XLSX, OCR, аудио, видео и архивы не извлекаются
(`docs/studio.md`).

**DoD:** реальное извлечение, лимиты MIME/размера, сохранение исходника.

### C4. Стоимость/лимиты, сравнение моделей, фоновые задачи, память — ожидает

**DoD:** точные usage/cost и бюджеты (сейчас `usage` относится к последнему
запросу, суммарных бюджетов нет) и долговечный планировщик с восстановлением
после рестарта.

---

## Снято / не делать без новых данных

- **MTP-спекуляция в `serve`.** Бит-экзактна, но ~0.8× от plain (`docs/mtp.md`),
  в `serve` не подключена. Условие возврата: либо verify дешевле (глобальные
  matvec-ядра или one-buffer драфт-шаг ~2 мс → ~1 мс), либо явный DoD «≥1.15×
  plain»; не выполнено — путь уходит в `docs/negative-results`.
- **Tensor parallel** вместо pipeline: all-reduce внутри каждого слоя — хуже.
- **Сжатие активаций** между узлами: 10 КБ — микросекунды против миллисекунд
  задержки.
- **Порт `kernel_mul_mm_id_q4_K`** из llama.cpp — закрыто инспекцией и
  измерением (`docs/moe-prefill.md`).
- Всё остальное из «What was tried and falsified» в `docs/moe-prefill.md` и из
  реестра отрицательных результатов.

## Сквозные инварианты

1. Любая цифра производительности имеет артефакт в `docs/benchmarks/` и
   воспроизводимую команду.
2. Замер — парные/back-to-back прогоны; вывод из одного числа не принимается
   (методология: `wiki://allpaka-measurement-methodology`).
3. Отрицательный результат публикуется, а не забывается; образец — раздел
   «What was tried and falsified» в `docs/moe-prefill.md`.
4. Регрессия ловится до мержа: все проверки, которые уже существуют в
   репозитории, подключены к CI (см. D1).

---

## Трек D — гигиена (сквозной)

### D1. CI: подключить существующие проверки

Факт: `.github/workflows/ci.yml` — один job `macos-14` с
`cargo test --workspace` и кэшем cargo. Не подключены: `clippy`, `fmt`, тесты
`allpaka-chat`, `scripts/test-serving-controls.py`,
`scripts/test-serving-memory.py`, `scripts/test-studio.py`, node-проверки Studio.
В самом файле есть комментарий, что `plugin_smoke` скипает engine/RAG-кейсы —
для них нужен self-hosted runner и `RAG_MCP_BIN`/`RAG_DB_PATH`.

**Блокер среды:** `.github/` — скрытый путь, инструменты записи этого воркспейса
его не редактируют («Only relative, non-hidden workspace paths are allowed»).
Патч ниже применяется вручную либо после явного исключения для `.github/`.

**DoD:** всё, что не требует Metal/GGUF, выполняется в CI и является обязательной
проверкой. `clippy -D warnings` включается после D4, иначе job будет красным
из-за известного долга.

### D2. Self-hosted Metal-раннер

**DoD:** CI покрывает GPU-кернелы и smoke-тесты с настоящими моделями, а не
только CPU-часть.

### D3. Распил `crates/allpaka-backend/src/gpu/`

`metal.rs` → device/memory, реестр pipeline, пути исполнения, диагностика, без
изменения измеренного поведения. **DoD:** существующие parity-тесты зелёные,
цифры бенчмарка не сдвинулись за пределы дрейфа.

### D4. Clippy-долг backend/model

Objective-C macro `cfg` expansion и старые internals. **DoD:**
`cargo clippy --workspace --all-targets -- -D warnings` проходит, после чего
проверка становится обязательной в CI (D1).

### Допущения, требующие подтверждения

- Паритет Windows/CUDA проверяется по релизным пунктам, а не на каждом этапе.
  Иначе стоимость проверки каждого пункта удваивается.
- Порядок исполнения по умолчанию: A → B → C, при этом D1 идёт первым как
  самый дешёвый и разблокирующий.

---

## Приложение: предлагаемый CI (черновик, не применён)

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request: {}

jobs:
  test:                       # без GPU: быстро, кандидат в обязательные
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: "rustfmt, clippy" }
      - name: Cargo cache
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: cargo-${{ runner.os }}-${{ hashFiles('Cargo.lock') }}
          restore-keys: cargo-${{ runner.os }}-
      - run: cargo fmt --all -- --check
      - run: cargo test --workspace
      - run: cargo test -p allpaka-chat --offline
      - run: cargo build -p allpaka-cli --no-default-features --offline
      - run: python3 scripts/test-studio.py
      - run: node --check crates/allpaka-chat/web/app.js
      - run: node scripts/test-studio-content.cjs
      # после D4: cargo clippy --workspace --all-targets -- -D warnings

  gpu:                        # self-hosted Metal; цель D2, до неё не блокирует PR
    if: github.event_name == 'push'
    runs-on: [self-hosted, macos, metal]
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --workspace -- --test-threads=1
      - run: python3 scripts/test-serving-controls.py
      - run: python3 scripts/test-serving-memory.py
```
