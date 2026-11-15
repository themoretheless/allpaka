# Паритет Allpaka с `deepseek-ai/deepseek-harness`

Этот документ фиксирует архитектуру, границы безопасности, HTTP API, CLI-режимы, Web UI и Python SDK, реализующие полный функциональный паритет Allpaka с [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness).

## Матрица подсистем

| Подсистема | Модуль Rust / файл | Инструменты / CLI / UI |
|---|---|---|
| Выгрузка больших выводов (Spill) | [`crates/allpaka-chat/src/spill.rs`](../crates/allpaka-chat/src/spill.rs) | Пагинация `GET /api/sessions/:id/spills` и `/:spill_id`, изоляция по сессиям |
| Песочница ОС и профили прав | [`crates/allpaka-chat/src/sandbox.rs`](../crates/allpaka-chat/src/sandbox.rs) | `SandboxMode` (`unsandboxed`, `read_only`, `workspace_write`), `ApprovalPolicy`, 4 пресета `GET /api/permission-presets` (`sandbox-exec` на macOS, `bwrap` на Linux, fail-closed) |
| Интерактивные подтверждения и вопросы | [`crates/allpaka-chat/src/interaction.rs`](../crates/allpaka-chat/src/interaction.rs) | Инструмент `ask_user_question`, поля сессии `pending_approval` / `pending_question`, `POST /api/sessions/:id/interaction`, баннер в [`web/harness-ext.js`](../crates/allpaka-chat/web/harness-ext.js) |
| Патчи, glob и чекпоинты / undo | [`crates/allpaka-chat/src/patch.rs`](../crates/allpaka-chat/src/patch.rs) | Инструменты `apply_patch`, `working_directory`, автоматический захват `CheckpointManager`, `GET /api/sessions/:id/checkpoints`, `POST /api/sessions/:id/checkpoints/:cid/undo` |
| Интерактивные PTY-терминалы | [`crates/allpaka-chat/src/terminal.rs`](../crates/allpaka-chat/src/terminal.rs) | Изолированные терминалы модели и пользователя, маршруты `/api/sessions/:id/terminals*` |
| Языковые серверы (LSP) | [`crates/allpaka-chat/src/lsp.rs`](../crates/allpaka-chat/src/lsp.rs) | LSP JSON-RPC клиент по stdio |
| Git Worktrees и Forge PR/MR | [`crates/allpaka-chat/src/worktree.rs`](../crates/allpaka-chat/src/worktree.rs), [`forge.rs`](../crates/allpaka-chat/src/forge.rs) | Инструмент `create_worktree` и `forge_pr`, маршруты `/api/sessions/:id/worktrees`, `/worktree`, `/forge-pr` |
| Удалённые корни по SSH | [`crates/allpaka-chat/src/ssh.rs`](../crates/allpaka-chat/src/ssh.rs) | Проверка опций (запрет Option Injection), `POST /api/ssh/test` |
| Субагенты и Agent Team DAG | [`crates/allpaka-chat/src/runtime/subagent.rs`](../crates/allpaka-chat/src/runtime/subagent.rs), [`agent_team.rs`](../crates/allpaka-chat/src/agent_team.rs) | Почтовый ящик, `list/message/interrupt` субагентов (`/api/sessions/:id/agents*`), задачи DAG с проверкой циклов (`/api/sessions/:id/team`) |
| Programmatic Tool Calling (PTC) | [`crates/allpaka-chat/src/ptc.rs`](../crates/allpaka-chat/src/ptc.rs) | Инструмент `run_code` с loopback token-bridge (`studio.call_tool`), запрет рекурсии в `run_code` и `ExecutionPolicy::files_only()` |
| Декларативные Workflows и Ralph-цикл | [`crates/allpaka-chat/src/workflow.rs`](../crates/allpaka-chat/src/workflow.rs) | Хранение `/api/workflows`, запуск `/api/sessions/:id/workflow`, цикл проверки |
| Веб-поиск, документы и seams | [`crates/allpaka-chat/src/web_tools.rs`](../crates/allpaka-chat/src/web_tools.rs), [`documents.rs`](../crates/allpaka-chat/src/documents.rs), [`automation_seams.rs`](../crates/allpaka-chat/src/automation_seams.rs) | SSRF-фильтр, извлечение PDF/DOCX/XLSX/PPTX/ODF/HTML/XML (`POST /api/documents/extract`), STT/Browser/Computer seams (`POST /api/voice/transcribe`) |
| Профили, портативные бандлы, reload | [`crates/allpaka-chat/src/bundles.rs`](../crates/allpaka-chat/src/bundles.rs) | Built-in профили (`default`, `coding-safe`, `autonomous-goal`, `research`), `/api/profiles*`, `/api/bundles/export`, `/api/bundles/import`, `/api/harness/reload` |
| Автоматизация: Schedules, Webhooks, ACP, LiveShare, OTLP | [`schedule.rs`](../crates/allpaka-chat/src/schedule.rs), [`webhook.rs`](../crates/allpaka-chat/src/webhook.rs), [`acp.rs`](../crates/allpaka-chat/src/acp.rs), [`liveshare.rs`](../crates/allpaka-chat/src/liveshare.rs), [`otel.rs`](../crates/allpaka-chat/src/otel.rs) | Cron/интервалы, HMAC-SHA256 webhook-триггеры (`X-Allpaka-Signature-256`), ACP JSON-RPC stdio, LiveShare read-only с SHA-256 хешированием токенов, экспорт OTLP JSON |
| CLI, Web UI и Python SDK | [`crates/allpaka-cli/src/main.rs`](../crates/allpaka-cli/src/main.rs), [`web/harness-ext.js`](../crates/allpaka-chat/web/harness-ext.js), [`sdk/python/allpaka_studio.py`](../sdk/python/allpaka_studio.py) | Флаги `--profile`, `--headless`, `--prompt`, `--dump-config`, `--acp`; модальное окно «Операции»; методы SDK без внешних зависимостей |

## Ключевые режимы CLI (`allpaka studio`)

```bash
# Вывести активную конфигурацию и встроенные профили как JSON
allpaka studio --dump-config --profile coding-safe

# Одиночный headless-прогон промпта с JSON-результатом в stdout
allpaka studio --headless --profile coding-safe --prompt "Проверь состояние проекта"

# Сервер Agent Client Protocol (ACP) JSON-RPC 2.0 по stdio
allpaka studio --acp --workspace /path/to/repo
```

Полный перечень флагов определён в [`crates/allpaka-cli/src/main.rs`](../crates/allpaka-cli/src/main.rs) и [`crates/allpaka-cli/src/dispatch.rs`](../crates/allpaka-cli/src/dispatch.rs).

## Инварианты безопасности

1. **Fail-closed песочница (`sandbox.rs`)**: при выборе `read_only` или `workspace_write` отсутствие системного бэкенда (`sandbox-exec` / `bwrap`) возвращает ошибку вместо тихого запуска без изоляции; директория данных Studio защищена от чтения и записи песочницей.
2. **Ограниченный Swarm и политики (`tools.rs`, `runtime/turn.rs`)**: участники Swarm получают только [`agentgrep`, `list_files`, `read_file`]; при `ExecutionPolicy::files_only()` и на вложенных/ограниченных ходах интерактивные, командные, командно-сетевые и team-инструменты не выдаются.
3. **Внешние триггеры (`router.rs`, `webhook.rs`, `liveshare.rs`)**:
   - Webhooks `/api/webhooks/trigger/:wid` проверяют константную во времени подпись `X-Allpaka-Signature-256` и не требуют `X-Allpaka-Client: studio`.
   - LiveShare хранит только SHA-256 от capability-токена; отзыв по `share.id` (`DELETE /api/sessions/:id/shares/:sid`) немедленно закрывает доступ к `/api/shares/:token` и `/share/:token`.
4. **Строгий CSP (`router.rs`)**: UI соблюдает `default-src 'self'; script-src 'self'; style-src 'self'` — без встроенных `<script>`, `onclick=` и `style=`.
5. **Ограниченная распаковка и SSRF-защита (`documents.rs`, `web_tools.rs`, `otel.rs`)**: ZIP/PDF имеют жёсткие лимиты на размер/сжатие; loopback/private/metadata IP заблокированы по умолчанию (для локальных OTLP-тестов используется переменная `ALLPAKA_OTEL_ALLOW_LOCAL=1`).

## Верификация и быстрый старт Python SDK

```python
from sdk.python.allpaka_studio import Studio

studio = Studio("http://127.0.0.1:8100")
presets = studio.list_permission_presets()
bundle = studio.export_bundle("default")
wh = studio.create_webhook("ci-hook", "Проверь событие: {{payload}}")
studio.trigger_webhook(wh["webhook"]["id"], wh["secret"], {"event": "push"})
```

- Скрипт UI VM-тестов: [`scripts/studio-ui-tests/run.sh`](../scripts/studio-ui-tests/run.sh) (покрывает 63 проверки, включая пресеты, интеракции, терминалы, Team DAG и CSP).
- Скрипт SDK-тестов: [`scripts/test-studio-sdk.py`](../scripts/test-studio-sdk.py) (116 тестов).
- Сквозной E2E-прогон: [`scripts/test-studio.py`](../scripts/test-studio.py).
