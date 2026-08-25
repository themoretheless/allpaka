---
name: allpaka
description: Локальный LLM-inference движок allpaka. Использовать, когда пользователь просит запустить локальную модель, сделать запрос к allpaka serve, проверить статус движка, прогнать текст через локальный inference или воспользоваться встроенными RAG-инструментами rag_search/rag_read.
---

# Allpaka Engine

Локальный LLM-inference движок (собственный Rust/Metal backend, GGUF-модели) с OpenAI-совместимым chat API и встроенным RAG tool-calling. Вся работа — через сабкоманды самого бинарника `allpaka` (Rust, без внешних зависимостей).

## Ключевые пути

- Репозиторий движка: `/Users/themoretheless/Documents/Sources/allpaka`
- Бинарник: `/Users/themoretheless/Documents/Sources/allpaka/target/release/allpaka` (если нет — `cargo build --release -p allpaka-cli`; cargo: `~/.cargo/bin/cargo`)
- Модели: `/Users/themoretheless/Documents/Sources/allpaka/models/`
  - `qwen3-235b-a22b-instruct-2507-Q2_K_XL.gguf` — основная (прогрев ~50 с, занимает ~83 GiB)
  - `qwen3-30b-a3b-Q4_K_M.gguf` — средняя
  - `qwen3-0.6b-Q8_0.gguf` — быстрая, для тестов
- Endpoint по умолчанию: `http://127.0.0.1:8099/v1/chat/completions`

## Когда использовать

- «запусти allpaka / локальную модель», «спроси у локальной модели», «прогони через allpaka»
- проверка, жив ли сервер; тесты RAG tool-calling (`rag_search` / `rag_read`)
- бенчмарки и сравнение конфигураций движка

## Как пользоваться

Все действия — сабкоманды бинарника (путь см. выше; далее `allpaka`):

```bash
allpaka status                                   # /health + /stats: жива ли служба и какая модель
allpaka serve --model <путь.gguf> [--bind 127.0.0.1:8099]   # запустить сервер (на переднем плане)
allpaka chat "текст запроса" [--rag] [--max-tokens 800] [--system "..."] [--model-name qwen3]
allpaka rag-test                                 # end-to-end тест RAG tool-loop, exit != 0 при регрессе
```

- Запуск сервера: `allpaka serve --model ...` — долгоживущий процесс; если пользователь просил поднять сервер, оставить его работать (не убивать после ответа).
- `chat --rag` передаёт схемы инструментов `rag_search`/`rag_read`; serve выполняет tool-loop сам (до 2 итераций, настраивается `ALLPAKA_RAG_MAX_TOOL_ROUNDS`).
- Переменные окружения serve: `ALLPAKA_RAG_TOOLS=0` выключает RAG; `ALLPAKA_RAG_BACKEND=auto|mcp|grep` — бэкенд поиска (по умолчанию auto: rag-mcp с BM25-индексом, иначе grep); `RAG_MCP_BIN` / `RAG_DB_PATH` — пути к rag-mcp и его DuckDB; `ALLPAKA_RAG_NOTES_DIR` — директория заметок, `ALLPAKA_RAG_AUTO_TOOLS=1` — авто-инжект схем. По умолчанию заметки берутся из `~/.claude/projects/-Users-themoretheless-Documents-Sources-allpaka/memory`.
- Автотест движка и RAG: `cargo test -p allpaka-cli --test plugin_smoke -- --test-threads=1` (поднимает serve сам, на отдельном порту).

## Правила

- Если сервер уже запущен пользователем (status отвечает), НЕ перезапускать и не останавливать его — работать с текущим.
- 235B грузится ~50 секунд и занимает почти всю память: перед запуском большой модели предупредить пользователя; для быстрых проверок использовать 0.6b.
- Не выдумывать ответы «от имени» локальной модели: всегда реально вызывать `allpaka chat` и показывать, что вернул сервер.
- Логи serve: `serve.log` в репозитории (при ручном запуске) или куда перенаправил пользователь.
