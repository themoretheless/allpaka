---
name: gh-pr-status-watch
description: Опрос check-ов pull request через gh — один проход таблицей или NDJSON, либо слежение за одним PR до завершения CI. Использовать, когда нужно дождаться зелёного статуса или понять, какой check висит.
---

# Статус check-ов PR

Скрипт `scripts/gh-pr-status-watch.sh` (в составе плагина — `bin/gh-pr-status-watch.sh`,
рядом обязательная `bin/scripts-common.sh`). Только чтение: вызывает `gh pr view`
и ничего не меняет в PR и в репозитории.

## Использование

```sh
scripts/gh-pr-status-watch.sh --once 123            # один опрос, ровная таблица
scripts/gh-pr-status-watch.sh --once --json 123     # машиночитаемо
scripts/gh-pr-status-watch.sh 123                   # следить до завершения check-ов
scripts/gh-pr-status-watch.sh --interval=30s --timeout=20m 123
scripts/gh-pr-status-watch.sh --once 12 34 56       # несколько PR за один проход
```

Флаги: `--once`, `--json`, `--interval=DURATION` (default `10s`), `--timeout=DURATION`
(0 — без лимита), `--repo=owner/name`, `-h`.

Слежение в реальном времени имеет смысл только для одного PR; при нескольких аргументах
скрипт делает один проход и выходит.

## Вывод

Таблица — по строке на check: имя, `status` (в нижнем регистре), `conclusion`.
Пустой conclusion у незавершённого check-а показывается как `…`, отмеченные ✗ —
упавшие (`failure|timed_out|cancelled`), ✓ — `success|skipped|neutral`. Над строкой
итог: `success`, `pending`, `failure` или `none`.

`--json` печатает по одному объекту `gh pr view` на PR (NDJSON). Для массива:

```sh
scripts/gh-pr-status-watch.sh --once --json 12 34 | jq -s 'map({n: .number, s: .statusCheckRollup})'
```

## Коды выхода

- `0` — check-и зелёные или их нет;
- `1` — есть неуспешный check (или PR недоступен);
- `2` — таймаут слежения;
- `3` — нет `gh auth`;
- `4` — неверные аргументы.

Удобно как gate перед слиянием:

```sh
scripts/gh-pr-status-watch.sh --once "$PR" || exit 1
```

## Как сводится статус

Скрипт сворачивает массив `statusCheckRollup`, а не читает `.statusCheckRollup.conclusion`
(у массива такого поля нет, и такой путь даёт пустоту). Приоритет: любой
`failure|timed_out|cancelled|action_required` → `failure`; иначе любой незавершённый
(`queued|in_progress|pending|waiting`) → `pending`; иначе все
`success|skipped|neutral` → `success`; пустой массив → `none`.

У `gh pr checks --json` другие имена полей: доступны `bucket`, `completedAt`,
`description`, `event`, `link`, `name`, `startedAt`, `state`, `workflow` — ни `status`,
ни `conclusion`, ни `url` там нет, gh отсекает запрос ещё до обращения к API. Поэтому
скрипт читает `statusCheckRollup` один раз из `gh pr view`.

## Границы

- Никаких `gh pr merge`, `gh pr edit`, комментариев: для слияния есть gh-auto-merge.
- Не долбить API: интервал не ниже секунды, при 403/429 сообщить о лимите
  (`gh api rate_limit --jq '.rate'`), а не уменьшать интервал.
- Ошибки gh показывать как есть: 404, `not logged in`, `no such remote`.
- Токен не печатать; `gh auth status` достаточно.

## Окружение

Обычный shell: `gh` в PATH и авторизован (`gh auth login` или `GH_TOKEN`/`GITHUB_TOKEN`),
`jq`, git-репозиторий с GitHub remote для вывода `owner/name`. Ни MCP-сервера, ни контейнера.
