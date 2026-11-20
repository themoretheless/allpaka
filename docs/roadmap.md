# Roadmap

Единственный план развития, рефакторинга и оптимизации `allpaka`. Сюда сведены
все оставшиеся задачи из прежних дорожных карт, рефакторинг-инвентаризаций и
аудитов оптимизации; отдельные плановые файлы (`docs/runtime-roadmap.md`,
`docs/refactor/PLAN.md`, `docs/refactor/INVENTORY.md`,
`docs/refactor/ARCHITECTURE.md`, `docs/code-review-2026-11-07.md`,
`docs/jcode-features.md`, `docs/research/chat-2026-09-21/capability-coverage.md`
и дублирующие секции «Next steps / Future work / Still open / Open gaps / Что
дальше» в остальных документах) удалены. Реализованное в план не переносится —
история в git.

Правила документа:

1. Строка без критерия готовности (**DoD**) не считается задачей.
2. Число без ссылки на артефакт не считается результатом.
3. Сделанный пункт удаляется из плана, а не помечается «done».

Сверено с деревом на `919a4cf`.

---

## Что осталось — сводка

| Где делать | Пункты |
|---|---|
| Везде (код, без спецжелеза) | A6, A7 (контейнер/ядра CPU), C1–C2, D3–D7 |
| Нужен Mac (Metal) | P4, P5, P6, P7, P9 (Metal), D3 (замер A/B), D2 (Metal-раннер) |
| Нужен Windows + CUDA GPU | P8, P9 (CUDA), A5, D4 (проверка CUDA), D2 (CUDA-раннер) |
| Нужна вторая машина | B1 → B3 |
| Нужен доступ владельца к GitHub / решению | D1 (branch protection), D2 (регистрация раннеров), D6 (debug-пробы) |

---

## Где мы сейчас (фактическая база, не задачи)

- **CUDA (RTX 5090):** Qwen3-32B Q4_K_M decode ~111% llama.cpp, prefill pp512
  ~108%, pp2048 ~103.5% (`docs/benchmarks/2026-11-10-allpaka-vs-llama-rtx5090.md`,
  `docs/benchmarks/2026-11-13-cuda-prefill-determinism-rtx5090.md`). В матрице с
  паритетом: Qwen3-32B, Qwen2.5-7B, Llama-3.1-8B, Gemma-3-12B (включая окно SWA
  за пределами 1024 токенов, `docs/benchmarks/2026-11-15-verify-gemma3-12b-rtx5090.md`).
- **Metal (M4 Max):** Qwen3.6-35B prefill 1.03× / decode 1.35×; Qwen3-30B
  prefill ~0.90× (+`MMID_NARROW`), GLM-Air decode ~0.86×, Qwen3-235B Q2_K_XL
  decode ~0.72× (`docs/benchmarks/allpaka-vs-llama-metal-status.md`). Все 10
  форматов GGUF поддерживают GPU-эмбеддинги, `MM_PIPE` и `RopeStyle::Norm`.
- **Сервинг (`allpaka serve`):** изолированные и параллельные именованные
  сессии, отмена, дедлайны, backpressure, блочный `PrefixCache`, чередование до
  четырёх длинных запросов (`serve/interleave.rs`) и fused multi-session decode
  `Model::forward_multi` (`kernel_batching: true` для плотных моделей, ×2.05 на
  четырёх сессиях; `docs/serving.md`,
  `docs/benchmarks/2026-11-14-serve-two-long-7b-rtx5090.md`).
- **Распределённый движок:** бинарный TCP-протокол со стадиями по слоям,
  батчевый `forward_batch`, DP-планировщик `pipeline-plan`, `pipeline-up`,
  `watch --manifest` и loopback-регрессия `scripts/test-distributed.py`
  (`docs/distributed-execution.md`).
- **Studio (`allpaka-chat`):** мультипровайдерный чат с запасной маршрутизацией
  (`fallback`), серверный разбор `@путь[:START-END]`, извлечение документов
  (PDF/DOCX/XLSX/PPTX/ODF/HTML/URL), verified writes / checkpoints / `undo` /
  `git_diff`, PTY-терминалы, worktrees, Forge PR, Swarm и Agent Team DAG,
  паритет с `deepseek-ai/deepseek-harness` (`docs/deepseek-harness-parity.md`),
  наблюдаемость и онлайн-оценки (`docs/studio.md`,
  `docs/observability/opik-parity.md`).
- **CI:** расширенный `.github/workflows/ci.yml` (`fmt`, `check-backends.sh`,
  `clippy -D warnings` + Windows CUDA cross-clippy, `cargo test --workspace`,
  63 UI-проверки Studio и 8 Python E2E-наборов) зелёный на `macos-latest` (`919a4cf`).

---

## Трек P — производительность движка

### P4. Metal: GLM-Air decode ≥ llama

Эксперты доминируют (down ~33%, gate_up ~25%), shared expert добавляет Q5+Q8
трафик. Не замерены на холодной машине: `ALLPAKA_MV_ID` (по умолчанию ON),
`ALLPAKA_SHARED_STAGGER=1`, `ALLPAKA_MV_TG` ∈ {64, 128, 256}. Харнесс —
`scripts/glm-ab-stagger.sh`.

Правило: `MV_ID` остаётся ON только если холодный GLM decode ≥ llama, иначе
`ALLPAKA_MV_ID=0`. Если после A/B всё ещё <1.0× — Metal GPU capture одного
токена: эффективный GB/s indexed Q8_0 down `[4096×1408]` и Q4_K gate/up
`[1408×4096]` против `mul_mv_q8_0`/`mul_mv_q4_K` llama, плюс простой между
стадиями. Новых fuse-флагов до этих цифр не добавлять.

**DoD:** холодный A/B GLM decode ≥ 1.0× llama с артефактом в `docs/benchmarks/`,
либо задокументированный отрицательный результат с GPU capture.

### P5. Metal: 235B decode (Q2_K путь)

Строго после P4 (векторизованный `matvec_q2_k_mv` уже в `main`, нужен сквозной
холодный прогон 235B против llama.cpp).

**DoD:** decode ≥ 0.9× llama на холодной машине с парным артефактом в
`docs/benchmarks/`.

### P6. Metal: остаток prefill на 30B и аномалия `mmbench` `q6_k`

pp480 — 481 dispatch в одном command buffer, GPU занят 321/322 мс; остаток ~10%
внутри графа при паритете изолированного микробенча. Перемерить на тихой машине
`MM_FMT=q6_k MM_NIN=12288 MM_NOUT=4096 MM_M=480 MM_LLAMA=1` против контроля
`q4_k` той же формы.

**DoD:** объяснённый через GPU capture разрыв in-graph vs microbench и либо
≥ 1.0× pp, либо опубликованный отрицательный результат.

### P7. Перемеры на тихом Mac после смены дефолтов

- `qwen35-35B`: `SWFUSE` стал opt-in 2026-09-26, цифры в
  `docs/benchmarks/qwen35-35b-m4-max-pending.md` сняты в старом fused-режиме.
- `GLM-4.5-Air`: перемер `docs/benchmarks/glm45-air-m4-max-pending.md` на
  холодной машине.
- Парный перемер матрицы 8B/30B против llama.cpp (`docs/bench-vs-llama-remote.md`).

**DoD:** свежие парные артефакты на тихой машине; файлы `*-pending.md` заменены
финальными отчётами.

### P8. Новые модели в матрице

1. **Gemma 4 (`head_dim = 512`)** — добавить поддержку ядер внимания/RoPE для
   `head_dim = 512` и прогнать `allpaka verify` при появлении эталонного GGUF.
2. **MTP-35B на CUDA** — проверка greedy/verify паритета спекулятивного блока
   `nextn` против монолита.

**DoD:** каждая модель проходит `allpaka verify` / greedy-паритет и попадает в
матрицу с артефактом.

### P9. Оставшиеся высокоприоритетные точки ускорения GPU и CPU

Точки из глубокого аудита `allpaka-backend` и `allpaka-model`, которые ещё не
реализованы и не были опровергнуты замерами:

1. **CUDA MoE и GDN на GPU без проходов через CPU:**
   - `fuse_decode_eligible` (`crates/allpaka-backend/src/gpu/cuda/decode.rs`)
     сейчас отклоняет все модели с `TokenFfn::Moe` и `gdn`, откатывая их на
     послойный decode без CUDA Graph и без `decode_greedy_continue`.
   - `cuda::ffn_batch_grouped` (`gpu/cuda/ffn_grouped.rs`) и `cuda::ffn_batch`
     (`gpu/cuda/mod.rs`) скачивают активации/роутер на CPU, считают `swiglu` на
     CPU и вызывают `gpu.stream.alloc` трижды на каждый слой.
2. **Metal `encode_verify_tokens` (`m = 2..8`) для `hd == 128` и `Dense` FFN:**
   в `crates/allpaka-backend/src/gpu/metal/verify.rs` перевести циклы `0..m`
   поштучных `qk_prep`, `attend` и `TILE=1` `Dense` FFN на уже существующие
   батчевые `qk_prep_batch`, `attend_rows` и `TILE=m` (как уже сделано для
   `hd == 256`), а в `decode_greedy_continue` поддержать GDN и расчёт RoPE на
   GPU из `ROPE_INV_FREQ`.
3. **Квантованный и страничный KV-кэш (`Q8_0` / `PagedKv`):** `Feature::PagedKv`
   в `capability.rs` и хранение KV в `Q8_0` для сокращения трафика памяти на
   длинных контекстах и разделения физических страниц префикса между сессиями.
4. **CPU батчевый `QuantMat::matmul` (`m > 1`) и `x86_64` SIMD:**
   - перевести `QuantMat::matmul` при `m > 1` (`quantmat/mod.rs`) на
     предварительную квантизацию всех `m` строк в `Q8Acts` с обходом weight-outer
     вместо скалярного `row_dot` и распараллелить `matmul_many` по блокам строк
     на все потоки пула, а не `items.len()`;
   - добавить `AVX2`/`AVX-VNNI` в `q8_acts::dot_i8` и `F16C`+`FMA` в
     `ops::f16::{dot, axpy}` на `x86_64`.
5. **Холодный старт и накладные расходы рантайма:** дисковый кэш PTX/CUBIN для
   NVRTC (`cuda/runtime/device.rs`), пропуск повторного `cuCtxSetCurrent` в
   `cuda/pdl.rs`, предкомпиляция/ленивая сборка пайплайнов `kernels.metal`,
   zero-copy парсинг заголовка GGUF из `Mmap` и пропуск аллокации 300k строк
   словаря в `gguf::metadata::read` / `VisionCensus::read`, кэширование
   `Vec<TokenLayer>` на сессию/модель и ленивое выделение страниц `Region` без
   синхронного `alloc_zeroed` гигабайтных буферов.

**DoD (для каждого подпункта отдельно):** парный замер A B A B с выигрышем за
пределами шума (артефакт в `docs/benchmarks/`), бит-в-бит / greedy-паритет на
тестах `gpu_parity` и `decode_paths_agree`.

---

## Трек A — сервинг и форматы

### A5. Autotune: кандидаты, различимые на CUDA

Выбор профиля (`--workload latency|throughput|memory`) работает
(`docs/benchmarks/2026-11-14-autotune-workloads-rtx5090.md`), но переключает
только Metal-флаги, поэтому на CUDA все три профиля дают одинаковую скорость и
память.

**DoD:** среди кандидатов `autotune` есть параметры, реально меняющие CUDA-путь
(размер чанка prefill, повтор графа, бюджет кэша f16-весов, число потоков/варпов),
и выбор между `latency`, `throughput` и `memory` статистически отделим от шума с
артефактом.

### A6. Частичные попадания `PrefixCache`, точный бюджет гибридных сессий и prefill без лишних логитов

1. В `crates/allpaka-cli/src/serve/session.rs` (`exact_prefix_snapshot`)
   использовать частичные совпадения префиксного дерева (`hit.matched_tokens <
   key.len()`), чтобы общий системный промпт не пересчитывался с нуля при новом
   сообщении.
2. В `serve/inference.rs` считать промежуточные чанки prefill (`0..n-1`) и
   `prewarm_prompt` через `forward_batch_hidden`, не вычисляя и не отбрасывая
   выходную голову `logits` на каждом чанке.
3. В `serve/memory.rs` (`session_bytes`) учитывать фактическое число активных
   attention- и GDN-слоёв гибридной модели вместо умножения обоих кэшей на все
   `n_layers` (сейчас завышает оценку памяти сессии в 3–4 раза на `qwen35moe`).

**DoD:** интеграционный тест частичного hit префикс-кэша в `serve`, точное
совпадение оценки `session_bytes` с фактической аллокацией `KvCache`+`SsmCache`
на гибридной модели и замер времени длинного prefill.

### A7. MLX-формат в рантайме и исправление контейнера `allpaka-mlx`

В `allpaka-mlx` сейчас открыты дефекты чтения реальных чекпоинтов HuggingFace /
`mlx-lm` и отсутствует прямое исполнение `MlxQ4`/`MlxQ8` в `allpaka-backend`:

1. **Контейнер и FFI (`crates/allpaka-mlx/src/`):**
   - убрать округление `data_start` вверх до 32 байт в `parse_safetensors_header`
     (`container.rs:200-210`), сдвигающее начало данных стандартных
     `.safetensors` файлов;
   - возвращать ошибку на инвертированных `end_offset < offset`, не мутировать
     `TensorInfo::name` префиксом шарда (хранить `shard: usize` в `TensorInfo`),
     поддержать fallback на отсортированный `model-*.safetensors` без
     `index.json`;
   - удалить дублирующий `enum GgmlType` из `ffi.rs:97-119`, где `F16` ошибочно
     маппится в `FLOAT32`.
2. **Рантайм (`allpaka-backend` / `allpaka-model`):** matmul/matvec прямо по
   блокам `MlxQ4`/`MlxQ8` (или перепаковка при загрузке без полного декванта в
   F32) + гейтинговый прогон `docs/benchmarks/2026-09-27-mlx-qwen-matrix/wait-then-run.sh`.

**DoD:** unit-тест на невыровненном стандартном `.safetensors` без ручного
паддинга, рабочий инференс MLX-чекпоинта в рантайме, артефакт гейтингового
прогона и CI-проверка на эталонной фикстуре.

---

## Трек B — распределённый инференс

### B1. Квалификация на двух машинах и постоянные соединения gateway

1. Удерживать пул открытых TCP-соединений стадий в `crates/allpaka-cli/src/pipeline.rs`
   (`serve_chat`) вместо переподключения и handshake на каждый HTTP-запрос,
   буферизовать запись фреймов в `BufWriter` и исправить срез `strip_prefix` при
   стриминге `<tool_call>` с пробелами.
2. Прогнать реальный разрез одной модели между Mac и PC по измеренному линку
   (`pipeline-up` + `watch --manifest`).

**DoD:** одна модель разрезана между двумя физическими машинами; greedy-стрим
совпадает с монолитом; `predicted tok/s` по `allpaka.toml` против `measured` в
пределах заявленного допуска (с учётом p99 хопа); артефакт замера.

### B3. Спекуляция поверх разреза

**DoD:** спекуляция поверх B1 сохраняет бит-в-бит greedy-паритет и не ухудшает
отношение «разрез vs монолит» с артефактом замера.

---

## Трек C — Studio (`allpaka-chat`)

### C1. Асинхронная запись сессий и индекс каталогов (без блокировки Tokio)

1. Вынести периодический `persist` во время стриминга (`runtime/turn/model.rs`)
   из-под `Mutex` сессии и с потока Tokio в фоновую очередь / `spawn_blocking`,
   исключить сериализацию многомегабайтных `images`/`swarm` в промежуточный
   `serde_json::Value` ради `.remove()` в `provider::body_with_effort`.
2. Кэшировать индекс сессий, правил `online_evaluation` и ревизий хранилищ
   (`evaluation`, `prompts`, `memory`) в памяти вместо полного чтения всех JSON
   и пересчёта SHA-256 при каждом `list()`, поиске `/api/history` и завершении
   трейса (`enqueue_trace`); убрать тройной `self.load` в `memory::save` и
   двойной парсинг через `Value` в `strict_json::parse`.
3. Объединить дублирующийся код файловых хранилищ (`evaluation`, `prompts`,
   `memory`, `guardrail_policies`, `model_evaluators`, `online_sources`) в
   обобщённый `VerifiedStore<T>`.

**DoD:** ни один `sync_all()` / `fs::read_dir` не выполняется напрямую на
async-воркере Tokio во время стриминга ответа; тест производительности на
каталоге из ≥500 сессий/трейсов без квадратичной деградации; все 63 UI и 8
Python E2E-тестов зелёные.

### C2. Нативная десктопная оболочка и встроенные локальные анализаторы медиа (опционально)

Все серверные, агентные и веб-возможности Studio реализованы; внешними швами
(`automation_seams.rs`, `extract.rs`) остаются:

1. Легковесная десктопная обёртка (Tauri / WebView) поверх локального сервера
   Studio с системным треем и глобальным хоткеем.
2. Опциональный встроенный OCR / STT через локальный движок `allpaka` при
   отсутствии внешних утилит `tesseract` / `ffmpeg` / `whisper`.

**DoD:** собираемый десктопный бандл на macOS/Windows и/или локальный fallback
транскрибации с E2E-тестом без внешних CLI в `PATH`.

---

## Трек D — корректность, безопасность и гигиена

### D1. Защита ветки `main` в GitHub

Расширенный CI на `macos-latest` зелёный, но ветка `main` в GitHub всё ещё не
защищена (`Branch not protected`, HTTP 404).

**DoD:** в правилах ветки `main` на GitHub джоба `test` отмечена обязательной
(`required status checks`) перед слиянием.

### D2. Self-hosted GPU-раннеры (Metal и Windows/CUDA)

На облачном `macos-latest` GPU паравиртуализован и нет локальных весов моделей,
а Windows/CUDA проверяется в облаке только через cross-`clippy`.

**DoD:** зарегистрирован self-hosted Metal-раннер (и по возможности
Windows/CUDA-раннер), прогоняющий `cargo test --workspace -- --test-threads=1` и
GPU/serving smoke-тесты на реальных моделях.

### D3. Metal: декомпозиция `metal/decode.rs` (3 288 строк) на стадии

`crates/allpaka-backend/src/gpu/metal/decode.rs` насчитывает 3 288 строк и
дублирует послойное кодирование команд между `decode_token` и
`decode_greedy_continue` (плюс пересечение с `verify.rs`). Убрать аллокации
временных `Vec` в обёртке `memory_barrier_with_resources` (~200 аллокаций на
токен) и вынести единый послойный энкодер стадий.

**DoD:** `metal/decode.rs` разбит на читаемые стадии без дублирования цикла
слоёв, нулевые heap-аллокации на барьерах энкодера, парный замер A B A B на Mac
без регрессии за пределы шума и greedy-паритет.

### D4. Дефекты корректности GPU, гибридных моделей и памяти (найдены при аудите `HEAD`)

Каждый пункт чинится отдельным коммитом со своим регрессионным тестом:

1. **Падение `tests/dense_forward.rs` с `--features cuda` (`detach_mapping`):**
   `Model::load` передаёт в `gpu::attach` кучевые буферы `_owned_weights`
   (`Box<[f32]>` после перестановки голов Llama в NEOX), не выровненные по
   16 КиБ, а `CudaGpu::detach_mapping` (`cuda/runtime/device.rs:618`) округляет
   конец диапазона вверх до 16 КиБ и выбивает окна соседних живых аллокаций.
2. **CUDA GPU-эмбеддинги зануляют 4 из 10 форматов:** `cuda::can_embed_on_gpu`
   (`cuda/mod.rs:131-136`) разрешает все 10 типов `dq_fmt`, но ядра
   `embed_row_f32` и `embed_row_f32_dtoken` (`cuda/kernels.rs:3952-4135`) не
   обрабатывают `Q2_K` (`4`), `Q3_K` (`5`), `Q5_0` (`6`) и `BF16` (`9`), молча
   оставляя нули в активациях (а для `Q8_0`/`F16`/`F32` крутят цикл в 1 потоке).
3. **CUDA MoE prefill игнорирует `shared_gate`:** `cuda/ffn_grouped.rs:115-119`
   всегда складывает общий эксперт с весом `1.0`, игнорируя
   `GroupedShared::gate_out` (`sigmoid(gate · hs)` на `qwen35moe`).
4. **Глобальное состояние GPU и префилла при нескольких моделях/потоках:**
   `set_rope_inv_freq`, `set_rope_inv_freq_swa` и `set_quant_gemm` мутируют
   глобальные синглтоны GPU при загрузке второй модели (например, draft-модели в
   `Speculator` или `ModelRegistry`), а состояние префилла (`pf_x`, `pf_hs`,
   `pf_active`) отпускает `Mutex<Gpu>` между `prefill_begin` и `prefill_end` без
   RAII-гарда `PrefillSessionGuard`.
5. **Утечка старых буферов в `MTLResidencySet` и отсутствие пиннинга KV/SSM:**
   `ensure_prefill`, `ensure_arenas` и `ensure_route` (`metal/device.rs:966-1027`)
   не вызывают `removeAllocation:` перед заменой выросших буферов, а
   `wrap_region` (`metal/region.rs`) не добавляет `SharedRegion` KV/SSM в
   `residency`; попутно убрать мёртвую запись `*yp.add(wts_at + *n_used) = 1.0`
   в `metal/decode.rs:914, 2519`.
6. **Гибридные GDN-модели и спекуляция в `allpaka-model`:**
   - `Config::from_gguf` (`config.rs:259-302`): обрезка хвостовых слоёв проверяет
     только `attn_q.weight` (отрезая валидные GDN-слои с `attn_qkv.weight`),
     `has_attn_bias` проверяет только `blk.0`, а `mrope_sections` паникует при
     `< 4` элементах;
   - `Speculator::round` (`speculate.rs:51-154`): не сохраняет/не откатывает
     слоты `SsmCache` при отклонении драфт-токенов на гибридных моделях и иначе
     разрешает ничьи в `argmax` (`>` против `max_by`);
   - `try_forward_greedy_chain` (`model/decode.rs:707`) передаёт `ssm: None`, а
     спаны `profile::span` в `model/decode.rs:408, 622` и `moe.rs:250`
     дропаются сразу в `;` с нулевой длительностью.
7. **Переполнение/рекурсия в `allpaka-gguf`:** ограничить глубину рекурсии
   `skip_array` на вложенных массивах (`elem_type == 9`, `gguf/src/metadata.rs:454`)
   и заменить `saturating_mul` + `as usize` на проверяемую арифметику в
   `TensorInfo::elements` / `GgufFile::dequant` (`gguf/src/tensors.rs:26, 257`).

**DoD:** все 7 подпунктов закрыты отдельными коммитами с тестами;
`cargo test --workspace` проходит **включая** `--features cuda` (в т.ч.
`tests/dense_forward.rs`).

### D5. Безопасность, процессы и гонки состояния в `allpaka-backend`, `allpaka-cli` и `allpaka-chat`

1. **Убрать UB `std::env::set_var("PATH", ...)` в рантайме CUDA:** в
   `cuda/ggml.rs:126` и `cuda/runtime/device.rs:930` заменить мутацию окружения
   многопоточного процесса на `AddDllDirectory` / `LoadLibraryExW`
   (`LOAD_WITH_ALTERED_SEARCH_PATH`).
2. **Закрыть неаутентифицированный `POST /v1/launch` (`allpaka-cli`):** в
   `serve/router.rs` и `serve/launch.rs` требовать локальный токен/флаг
   разрешения, убрать предсказуемый путь `/tmp/allpaka-relaunch.sh` без `O_EXCL`
   и экранировать невалидный JSON в `<tool_call>` (`serve/messages.rs:89-91`).
3. **Группы процессов при таймауте команд Studio:** в `runtime/exec.rs:59-73` и
   `automation_seams.rs` запускать дочерние процессы в отдельной группе
   (`process_group(0)`) и завершать всю группу (`killpg`), чтобы внучатые
   процессы `/bin/sh -c` не зависали сиротами при таймауте.
4. **Константное сравнение секретов, SSRF и атомарность в `allpaka-chat` / `pipeline`:**
   - перевести проверку ключа кластера в `pipeline.rs:371` и
     `chat/src/remote_worker.rs:38-43` на константное по времени сравнение;
   - подключить SSRF-проверку (`validate_and_resolve_public_url`) к URL
     кастомных провайдеров, MCP-плагинов и `remote_worker` (с явным opt-in для
     локальной сети);
   - перевести `webhook.rs` и `credentials.rs` на сериализованное обновление
     через `state_file` (и не хранить секреты вебхуков открытым текстом), а
     `tools::write_verified` — на атомарную запись через временный файл;
   - заменить `.lock().unwrap()` в обработчиках Studio на `unwrap_or_else(|e|
     e.into_inner())` / контролируемую ошибку.

**DoD:** ни одного вызова `env::set_var` вне однопоточного старта/тестов, тесты
на убийство дерева процессов по таймауту, конкурентные мутации `webhooks` /
`credentials` без потери записей, отказ `POST /v1/launch` без авторизации.

### D6. Одноразовые debug-пробы, кэширование env в `model` и артефакты в `docs/benchmarks/`

1. Одноразовые debug-переключатели (`ALLPAKA_*_ZERO`, `ALLPAKA_ROWS_DEBUG`,
   `ALLPAKA_TOKENBUF_DEBUG`, `ALLPAKA_VERIFY_DEBUG`, `ALLPAKA_MEGA_DEBUG` и др.):
   по решению владельца либо зафиксировать вывод в `docs/` и удалить ветки из
   кода, либо оставить штатными в `docs/env-reference.md`.
2. В `crates/allpaka-model/src/model/mod.rs` перевести чтение переключателей
   декодирования (`cpu_attn`, `no_tokenbuf`, `no_argmax`, `greedy_chain`) с
   потокенного `std::env::var` на структуру конфигурации модели/сессии (чтобы и
   не дергать ОС-лок окружения на каждый токен, и не ломать тесты глобальным
   `OnceLock`).
3. Удалить случайно закоммиченные исполняемые бинарники `allpaka` / `gpu_ffnbench`
   (~45 МБ суммарно) и `.pyc` из `docs/benchmarks/2026-09-*`, оставив только
   текстовые/JSON отчёты и скрипты воспроизведения.

**DoD:** нет `std::env::var` на горячем пути токена, в `docs/benchmarks/` нет
скомпилированных бинарников и `__pycache__`, `docs/env-reference.md` синхронен с
`env_reference_is_current`.

### D7. Архитектурная чистота границ крейтов и тестов

1. Перенести скалярные блок-деквантизаторы (`dequant_q*_block`, `scale_min_k4`)
   из `allpaka-gguf::dequant` в `allpaka-dtype`, полностью разорвав зависимость
   `allpaka-backend -> allpaka-gguf`.
2. Подключить либо удалить неиспользуемые пути `execution::ExecutionPolicy` и
   подключить `memory::Budget` / `telemetry::PhaseTrace` к аллокациям GPU-арен и
   CUDA/CPU стадиям; документировать оставшиеся `// SAFETY:` в `allpaka-backend`.
3. Довести типизированный `ModelError` в `allpaka-model` (вместо `anyhow` в
   библиотечном API) и enum-статусы в `observability/model.rs` и `judge_runs.rs`.
4. Перевести оставшиеся 5 UI-тестов Studio (`test-studio-{agents-team,interactions,ops-automation,sandbox-presets,terminals}-ui.js`)
   на общий `scripts/studio-ui-tests/harness.js` и добавить прямые unit-тесты в
   модули без `#[cfg(test)]` (`gguf/tensors.rs`, `mlx/container.rs`,
   `core/{cost,presets}.rs`, `chat/{judge_runs,plugins,history,rag}.rs`).

**DoD:** `cargo tree -p allpaka-backend` не содержит `allpaka-gguf`, все 63
UI-теста работают через `harness.js`, `check-backends.sh` и `clippy -D warnings`
зелёные.

---

## Снято / не делать без новых данных

- **MTP-спекуляция в `serve`.** Бит-экзактна, но ~0.8× от plain (`docs/mtp.md`).
  Возврат только при удешевлении verify / one-buffer драфт-шага с DoD «≥ 1.15×
  plain».
- **Megakernel / `moe_ffn_mega`.** Межтредгрупповой busy-wait на `atomic_uint`
  вешал планировщик Metal и macOS. Возврат только на стадиях с барьерами
  энкодера и тестом, исключающим зависание GPU.
- **Опровергнутые на M4 Max переключатели Metal:** дефолтный `SWFUSE` (регрессия
  на 235B Q3_K), `DUAL_GW` (`DECODE_GUFUSE` / `ALLPAKA_DUAL`), `CFUSE`, `RFUSE`
  под `normflag`, `RTOPK`, `SHARED_TAIL`, `NR0 ≠ 2` для Q8, `Q8 K-split+shmem`,
  тайлы `mm64` / `mm_bn64` / `mmid_k64` и порт `kernel_mul_mm_id_q4_K` —
  зафиксированы в `docs/decode-opts.md` и `docs/moe-prefill.md`.
- **Tensor parallel** вместо pipeline и **сжатие активаций** между узлами.

---

## Сквозные инварианты

1. Любая цифра производительности имеет артефакт в `docs/benchmarks/` и
   воспроизводимую команду; каждый бенчмарк `allpaka` идёт парно со свежим
   прогоном `llama.cpp`.
2. Замер — парные/чередующиеся прогоны (`A B A B`); вывод из одного числа не
   принимается, шаг с замедлением горячего пути больше шума (±3% на тихой
   машине) не вливается.
3. Горячие пути правятся только с парным замером и greedy-паритетом.
4. Поведение, HTTP-контракты и форматы файлов не меняются без явного решения.
5. Перенос кода, правка логики и форматирование — разные коммиты; баг по
   дороге — отдельный коммит с тестом.
6. Отрицательный результат публикуется, а не забывается.
