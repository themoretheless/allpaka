# Roadmap

Единственный план развития allpaka. Сюда сведены и отсюда удалены:
`docs/runtime-roadmap.md`, `docs/refactor/PLAN.md`,
`docs/code-review-2026-11-07.md`, разделы «Next steps / Future work / Still
open / Open gaps» из `docs/MLX-INTEGRATION-FINAL.md`,
`docs/mlx-format-integration.md`, `docs/decode-opts.md`,
`docs/benchmarks/matrix-2026-09-12.md`,
`docs/benchmarks/allpaka-vs-llama-metal-status.md` и «Что дальше» из README.
Реализованное в план не переносилось — история в git.

Правила документа:

1. Строка без критерия готовности (**DoD**) не считается задачей.
2. Число без ссылки на артефакт не считается результатом.
3. Сделанный пункт удаляется из плана, а не помечается «done».

Сверено с деревом на `1584ad5`.

## Где мы сейчас

- **CUDA (RTX 5090, Qwen3-32B Q4_K_M):** decode ~110% llama.cpp, prefill
  105–110% (без повтора CUDA-графа; с графом — до 112%).
  Артефакт: `docs/benchmarks/2026-11-10-allpaka-vs-llama-rtx5090.md`.
- **Metal (M4 Max):** 30B pp ~0.90×, GLM-Air decode ~0.86×, 235B decode ~0.72×.
  Артефакт: `docs/benchmarks/allpaka-vs-llama-metal-status.md`.
- **Сервинг:** изолированные сессии, отмена, дедлайны, backpressure, блочный
  prefix cache с hit/miss — есть (`docs/serving.md`). Инференс сериализован
  одним владельцем, `kernel_batching: false`.
- **Распределённый движок:** разрез по слоям на TCP-стадии, gateway с
  токенизатором и `pipeline-plan` есть; loopback-регрессия
  `scripts/test-distributed.py` сравнивает с монолитом
  (`docs/distributed-execution.md`). Квалификации на двух машинах нет.

---

## Трек P — производительность движка

### P1. CUDA: prefill без повтора графа

Граф повторяется только при одинаковой форме чанка и базе KV; длинный промпт
по чанкам его не повторяет, поэтому реальный показатель — колонка «no graph».

**DoD:** pp на чанкованном промпте ≥ llama.cpp в парном замере (3 раунда
интерлив), argmax и logits-fnv стабильны между раундами.

### P2. CUDA: недетерминизм prefill pp128 без ggml

В режиме без ggml при pp128 хэш логитов и argmax меняются от запуска к запуску;
pp512 и decode стабильны.

**DoD:** причина найдена (порядок редукции / гонка), pp128 детерминирован, тест
на повторяемость хэша.

### P3. CUDA: сборка `allpaka_ggml.dll` воспроизводима из репозитория

Сейчас DLL собирается локальными неотслеживаемыми `.bat`
(`third_party/build_ggml_cuda.bat`, `link_ggml_cuda.bat`,
`rebuild_ggml_cuda_fa.bat`, `compile_mmq_ptx.bat`, `.compile_mmvq_tmp.bat`), а
`third_party/llama.cpp` не закреплён.

**DoD:** один закоммиченный скрипт + закреплённая ревизия llama.cpp; чистый
клон собирает DLL, которую ждут `d5dea0f`/`3815e23`; временные `.bat` удалены.

### P4. Metal: GLM-Air decode ≥ llama

Эксперты доминируют (down ~33%, gate_up ~25%), shared expert добавляет Q5+Q8
трафик. Не замерены на холодной машине: `ALLPAKA_MV_ID` (по умолчанию ON),
`ALLPAKA_SHARED_STAGGER=1`, `ALLPAKA_MV_TG` ∈ {64,128,256}. Харнесс —
`scripts/glm-ab-stagger.sh` (запускать вне песочницы агента).

Правило: `MV_ID` остаётся ON только если холодный GLM decode ≥ llama, иначе
`ALLPAKA_MV_ID=0`. Если после A/B всё ещё <1.0× — Metal GPU capture одного
токена: эффективный GB/s indexed Q8_0 down `[4096×1408]` и Q4_K gate/up
`[1408×4096]` против `mul_mv_q8_0`/`mul_mv_q4_K` llama, плюс простой между
стадиями. Новых fuse-флагов до этих цифр не добавлять.

**DoD:** холодный A/B GLM decode ≥ 1.0× llama с артефактом, либо отрицательный
результат с GPU capture в реестре.

### P5. Metal: 235B decode (Q2_K путь)

Строго после P4. **DoD:** decode ≥ 0.9× llama на холодной машине, артефакт.

### P6. Metal: остаток pp на 30B

pp480 — 481 dispatch в одном command buffer, GPU занят 321/322 мс; остаток
~10% — пропускная способность mm-ядер внутри графа (на изолированном
микробенче паритет с llama). Открыто: аномалия `mmbench` q6_k, перемерить на
тихой машине (`MM_FMT=q6_k MM_NIN=12288 MM_NOUT=4096 MM_M=480 MM_LLAMA=1` +
контроль q4_k той же формы).

**DoD:** объяснённый (capture) разрыв in-graph vs microbench и либо ≥1.0× pp,
либо отрицательный результат.

### P7. Перемеры, устаревшие после смены дефолтов

- qwen35-35B: SWFUSE стал opt-in 2026-09-26, строки
  `docs/benchmarks/qwen35-35b-m4-max-pending.md` сняты в fused-режиме.
- Сравнение с llama.cpp на 8B/30B и перемер BASELINE на тихом Mac (порядок в
  `docs/bench-vs-llama-remote.md`).

**DoD:** свежие парные артефакты, pending-файлы закрыты или переименованы.

### P8. Новые модели в матрице

1. Dense Qwen (`qwen3`/`qwen35` dense) среднего размера.
2. Gemma — только после attention с head_dim=512.
3. Опционально MTP-35B для паритета спекуляции.

**DoD:** модель проходит greedy-паритет и попадает в матрицу с артефактом.

### P9. Не-Neox RoPE (Llama, Mistral) на быстрых путях

Пять проверок `RopeStyle::Neox` в model закрывают быстрые пути; GPU-ядра
реализуют только Neox. Путь без правки ядер: переставить строки Wq/Wk при
загрузке ценой копии этих весов.

**DoD:** Llama-модель идёт по GPU-пути с greedy-паритетом против CPU-эталона.

---

## Трек A — сервинг

### A1. Сопоставимые бенчмарки: token replay против llama.cpp

Есть: метаданные нагрузки, строгий GPU-coverage, alternating context-matched
прогоны, `scripts/llama-matched-bench.cpp`.

**DoD:** replay одного и того же токен-стрима в обоих движках; артефакт по
стабильной JSON-схеме с флагом comparability; для MoE в отчёте отражено, что
одинаковые формы тензоров не означают одинаковый роутинг экспертов.

### A2. Единый бюджет памяти: KV/scratch admission

Есть reservations для mapped-весов, prefix-cache и сессий,
`peak_reserved_bytes` в `/stats`. Не покрыты GPU scratch, вспомогательные
аллокации модели и HTTP-буферы — это счётчики, не hard-cap процесса.

**DoD:** покрыть эти статьи либо явно назвать границу в `docs/serving.md` и в
ответе API.

### A3. Step scheduler + chunked prefill

**DoD:** chunked prefill и метрика максимальной decode-задержки под
конкурентной нагрузкой; задержка ограничена.

### A4. Fused batching независимых сессий на GPU — строго после A3

**DoD:** parity-тесты батча против последовательного выполнения; `/stats`
перестаёт сообщать `kernel_batching: false`.

### A5. Workload-aware autotune

Профили `safe`/`balanced`/`max-performance` и версионированный кэш есть.

**DoD:** выбор профиля под нагрузку (latency / throughput / memory), а не только
под модель+устройство.

### A6. Structured output + валидированные tool calls

**DoD:** constrained decoding по JSON-схеме инструмента, явный отказ при
несоответствии аргументов; тест на мусорный вывод модели.

### A7. MLX-формат в рантайме

Загрузка, деквант и харнесс есть (`docs/mlx-format-integration.md`).

**DoD (по шагам):** matmul прямо по MLX-блокам без полного декванта, либо
перепаковка в формат allpaka при загрузке; гейтинговый прогон
`docs/benchmarks/2026-09-27-mlx-qwen-matrix/wait-then-run.sh` с артефактом;
CI-проверка на скачиваемом эталонном чекпоинте.

---

## Трек B — распределённый инференс

### B1. Квалификация на двух машинах

**DoD:** одна модель разрезана между mac и PC; greedy-стрим совпадает с
монолитом; predicted tok/s по `allpaka.toml` против measured в пределах
заявленного допуска. Хоп односторонний — платится p99, а не среднее;
неизмеренный стык блокирует разрез.

### B2. Запуск по конфигу и наблюдение

`pipeline-plan` выдаёт манифест, но не поднимает стадии.

**DoD:** запуск стадий по манифесту без ручного копирования аргументов;
`allpaka watch` расширяется словарём `StageReachable`/`LinkVerified`/
`LayersPlaced` — только после того, как подъём стал программным.

### B3. Спекуляция на разрезе

**DoD:** спекуляция поверх B1 не меняет сравнение «разрез vs монолит» (она
множитель к обеим конфигурациям).

---

## Трек C — Studio

### C1. Поиск/RAG по папкам проекта

**DoD:** цитаты с provenance (файл + строки), видимый бюджет контекста, выбор
файлов через `@`; изменённый файл не выдаётся из кэша как свежий. В том же
пункте: `docs/studio.md` всё ещё пишет, что MCP «пока не включён», хотя
`crates/allpaka-chat/src/mcp.rs` — работающий клиент к `rag-mcp`.

### C2. Checkpoints/undo, git diff, безопасные патчи, shell

**DoD:** отмена со сверкой фактического состояния; патч проверяется
байт-в-байт по образцу контрактного теста `edit_file`.

### C3. Извлечение PDF/DOCX/XLSX/OCR/URL/аудио/видео

**DoD:** реальное извлечение, лимиты MIME/размера, сохранение исходника.

### C4. Стоимость и бюджеты в чате

Фоновые задачи с восстановлением после рестарта и сводка стоимости по трассам
уже есть. **DoD:** суммарные usage/cost по беседе и бюджеты (сейчас `usage` —
последний запрос), fallback/provider routing.

### C5. Браузерная квалификация памяти

В `docs/jcode-features.md` у консолидации памяти и проверки источников
помечены pending: browser-квалификация, HTTP-кейсы rejection/provider-call-count,
семантический синтез. **DoD:** pending-пометки закрыты прогоном или переведены
в явные ограничения.

---

## Трек D — гигиена и корректность

### D1. CI: подключить существующие проверки

Сейчас `.github/workflows/ci.yml` — один job `macos-14` с `cargo test
--workspace`. Не подключены `fmt`, `clippy`, `allpaka-chat`, Studio-скрипты,
node-проверки. Clippy на workspace уже чист (Metal и CUDA).

**DoD:** всё, что не требует GPU/GGUF, выполняется в CI и обязательно, включая
`cargo clippy --workspace --all-targets -- -D warnings`. Черновик — в приложении.

### D2. Self-hosted GPU-раннеры

**DoD:** CI покрывает GPU-ядра и smoke-тесты с моделями на Metal; желательно
также Windows/CUDA-раннер, раз основная работа по скорости идёт там.

### D3. Metal: `decode_token` → стадии

1 657 строк горячего пути; вместе с ним — request-структуры для Metal-лаунчеров
и билдер dispatch-бойлерплейта (`set_pipeline + set_buffer×N + dispatch`).
Нужен Mac. **DoD:** A B A B без регрессии за пределы шума, greedy-паритет.

### D4. Soundness-FIXME

- `gpu/cuda/pdl.rs` читает приватные поля cudarc по угаданным смещениям —
  собирать аргументы запуска самим.
- `gpu/metal/ffn_grouped.rs:216` — гонка рестейджинга GLM router bias
  (вероятный баг GLM fused prefill). Нужен Mac.
- `gpu/metal/verify.rs:361` — дамп `ALLPAKA_VERIFY_DEBUG`.
- `tests/gpu_parity.rs:101,180` — `&mut` поверх общего `&'static` среза при
  параллельных тестах.

**DoD:** `grep 'FIXME(soundness)'` пуст, на каждый — тест или обоснование.

### D5. CPU-эталон расходится с llama.cpp

`allpaka verify` на x86, Qwen2.5-7B-Instruct-Q4_K_M, промпт по умолчанию:
argmax llama.cpp 12095, наш 135958; топ-10 логитов ниже на ~9–13. Было и до
рефакторинга. Воспроизведение: `llama-server -m <7B> --port 8080`, затем
`allpaka verify --model <7B> --decode 32`.

**DoD:** причина найдена, argmax совпадает, регрессионный тест.

### D6. Мусор и мёртвый код

- `gpu/shaders/kernels.metal` — неиспользуемая копия `metal_kernels.rs`.
- Одноразовые debug-пробы (`ALLPAKA_*_ZERO`, `ROWS_DEBUG`, `TOKENBUF_DEBUG`,
  …): либо результат задокументирован с датой и проба удаляется, либо остаётся
  как инструмент в `docs/env-reference.md`. Требует решения владельца по
  каждой.
- Неотслеживаемые артефакты в корне и `docs/benchmarks/` (`sizeof_q8.obj`,
  `overlay-off-*.out/.err`, `nsys-decode-32b.*`) — закоммитить как артефакт или
  удалить.

**DoD:** перечисленного нет в дереве, `git status` чист после сборки.

---

## Ждёт решения владельца

1. `memory_extract` отвечает 409 при удалении беседы (`0d194ed`) — оставить или
   откатить.
2. Сужение префиксных маршрутов `serve` (`/statsx`, `/v1/modelsx` сейчас
   матчатся) и ответ 405.
3. Решения по умолчанию: два правила id (`d702ed3`); megakernel и проигравшие
   варианты остаются на месте (megakernel выключен константой).
4. Любой шаг, замедляющий горячий путь больше шума (±3% на тихой машине).

## Снято / не делать без новых данных

- **MTP-спекуляция в `serve`.** Бит-экзактна, но ~0.8× от plain (`docs/mtp.md`).
  Возврат: verify дешевле (one-buffer драфт-шаг ~2 мс → ~1 мс) или DoD «≥1.15×
  plain».
- **Megakernel / `moe_ffn_mega`.** Device-atomic busy-wait вешал GPU-планировщик
  Mac. Возврат только с encoder-barrier стадиями и тестом, который не может
  повесить GPU.
- **RFUSE, GUFUSE, CFUSE, RTOPK, SHARED_TAIL, NR0≠2, Q8 K-split+shmem** —
  `docs/decode-opts.md`.
- **Порт `kernel_mul_mm_id_q4_K`** и прочее из «What was tried and falsified» —
  `docs/moe-prefill.md`.
- **Tensor parallel** вместо pipeline; **сжатие активаций** между узлами.
- **Кэш дескрипторов `token_layers`** — эффект ниже шума.

## Сквозные инварианты

1. Любая цифра производительности имеет артефакт в `docs/benchmarks/` и
   воспроизводимую команду; каждый бенчмарк allpaka идёт парно со свежим
   прогоном llama.cpp.
2. Замер — парные/back-to-back прогоны (A B A B); вывод из одного числа не
   принимается.
3. Горячие пути правятся только с парным замером и greedy-паритетом.
4. Поведение, HTTP-контракты и форматы файлов не меняются без явного решения.
5. Перенос кода, правка логики и форматирование — разные коммиты; баг по
   дороге — отдельный коммит с тестом.
6. Отрицательный результат публикуется, а не забывается.

## Приложение: предлагаемый CI (черновик, не применён)

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request: {}

jobs:
  test:                       # без GPU: быстро, кандидат в обязательные
    runs-on: macos-14
    env: { RUSTFLAGS: "" }
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
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace
      - run: cargo test -p allpaka-chat --offline
      - run: cargo build -p allpaka-cli --no-default-features --offline
      - run: python3 scripts/test-studio.py
      - run: node --check crates/allpaka-chat/web/app.js
      - run: node scripts/test-studio-content.cjs

  gpu:                        # self-hosted Metal; цель D2, до неё не блокирует PR
    if: github.event_name == 'push'
    runs-on: [self-hosted, macos, metal]
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --workspace -- --test-threads=1
      - run: python3 scripts/test-serving-controls.py
      - run: python3 scripts/test-serving-memory.py
```
