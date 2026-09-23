---
name: gh
description: GitHub через CLI gh — PR, issues, Actions и их логи, релизы, gh api. Использовать, когда нужен статус PR или CI, создание/просмотр пул-реквеста, работа с issues, поиск по репозиториям, или сырой запрос к GitHub API.
---

# GitHub CLI (gh)

Все действия — настоящие команды `gh` в репозитории пользователя.

## Шаг 0: авторизация

```sh
gh auth status
```

Если не авторизован — сказать прямо и предложить `gh auth login` (интерактивно,
выполняет пользователь) либо токен в переменной `GH_TOKEN` / `GITHUB_TOKEN`.

Токен — секрет: не печатать его в чат, не подставлять в команды текстом,
`gh auth token` не выводить без явной просьбы.

## Репозиторий

```sh
gh repo view --json nameWithOwner,defaultBranchRef,visibility
gh repo view owner/name --json nameWithOwner      # явный репозиторий
gh repo clone owner/name
```

Не угадывать `owner/name` — брать из `gh repo view` или `git remote -v`
(удобно, когда рядом живёт git-плагин).

## Pull requests

```sh
gh pr list --state open --limit 20 --json number,title,author,isDraft
gh pr view <n> --json number,title,state,mergeable,reviewDecision,statusCheckRollup
gh pr diff <n>
gh pr checks <n>            # добавить --watch при ожидании
gh pr status                # что касается текущей ветки
```

Изменяющие (только по явной просьбе):

```sh
gh pr create --title "..." --body "..."      # или --fill, если пользователь согласен
gh pr comment <n> --body "..."
gh pr merge <n> --squash
gh pr close <n>
```

Перед `pr create` показать пользователю заголовок и тело — они уйдут публично.
Перед `pr merge` проверить `gh pr checks <n>` и `mergeable`.

## Issues

```sh
gh issue list --state open --limit 20 --json number,title,labels
gh issue view <n> --json number,title,body,state
gh issue create --title "..." --body "..."    # только по просьбе
gh issue comment <n> --body "..."             # только по просьбе
```

## Actions и CI

```sh
gh run list --limit 10 --json databaseId,workflowName,status,conclusion,headBranch
gh run view <id>
gh run view <id> --log-failed        # логи только упавших джобов
gh run watch <id>
gh workflow list
gh workflow run <name>               # только по просьбе: это меняет состояние CI
```

## Релизы и прочее

```sh
gh release list --limit 10
gh release create <tag> --notes "..."   # только по просьбе, публикация необратима
gh label list
gh search prs --repo owner/name "query" --limit 10
gh search repos "query" --limit 10
```

## gh api (сырые запросы)

```sh
gh api repos/{owner}/{repo}/pulls/<n> --jq '.title, .state'
gh api --paginate repos/{owner}/{repo}/issues --jq '.[].number'
gh api repos/{owner}/{repo}/commits -f sha=... -f message=...
```

`--jq` вместо внешнего `jq`; `--paginate` для полных списков. Метод по умолчанию
GET: если команда без `-f/-F/--method`, это чтение. Не выполнять
`--method DELETE` и `-X PATCH/POST` без явного запроса.

## Правила безопасности

- **По умолчанию только чтение:** `list`, `view`, `diff`, `checks`, `run view`,
  `api` без тела запроса.
- Изменяющие операции — `pr create/merge/close/comment`, `issue create/close/comment`,
  `release create/delete`, `workflow run`, `repo delete`, `repo edit`, `push` —
  только по прямой просьбе и с показом того, что уйдёт наружу.
- Всё, что создаётся, видно другим людям: комментарии, PR, issues, релизы.
  Черновик текста показывать до отправки.
- Не выполнять `gh repo delete`, `gh release delete`, `gh pr merge --admin`,
  удаление веток и тегов без отдельного подтверждения.
- Уважать rate limit: при 403/429 не долбить повторами, а сообщить
  (`gh api rate_limit --jq '.rate'`).
- Ошибки `gh` показывать как есть: «not logged in», «HTTP 404», «no such remote» —
  это ответ, а не повод додумывать состояние репозитория.

## Никакой изоляции: обычный shell

Плагин **не поднимает MCP-сервер** и не использует контейнер: нужна обычная
командная строка на машине, где `gh` уже авторизован.

```sh
command -v gh && gh --version
gh auth status
```

Единственное требование к окружению — доступ агента к командной строке. Если
клиент его не даёт (только MCP-инструменты), скилл бесполезен; MCP-обёртка GitHub
здесь сознательно не используется.
