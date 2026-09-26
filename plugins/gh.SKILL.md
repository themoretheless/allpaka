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

### Basic Operations

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

### Auto-Merge with Preconditions

**Script:** `scripts/gh-auto-merge.sh` (implements smart merge logic)

**Preconditions checked before merge:**
1. PR is open and not a draft
2. `mergeable` is true (gh returns a boolean here; false means conflicts or a
   non-applicable state — `mergeStateStatus` carries the reason)
3. Every check in the `statusCheckRollup` array is green. It is an array of
   `{name,status,conclusion}`, and `conclusion` is empty while checks run, so
   `.statusCheckRollup.conclusion` is not a valid path
4. `reviewDecision` is surfaced, and required reviews are gh's to enforce, not ours

**Commands:**
```sh
# Dry-run: print the decision and the exact gh command, change nothing
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>

# Squash (default), or an explicit strategy
scripts/gh-auto-merge.sh <PR_NUMBER>
scripts/gh-auto-merge.sh --merge --delete-branch <PR_NUMBER>

# Wait up to 45 min for checks to turn green
scripts/gh-auto-merge.sh --timeout=45m <PR_NUMBER>

# Let GitHub do the merge once checks pass
scripts/gh-auto-merge.sh --auto <PR_NUMBER>
```

There is no `--strategy` option — gh has none either; `--squash`, `--merge` and
`--rebase` are the real flags.

**Safety:**
- Never bypass branch protection rules (`--admin` is not used)
- Fail fast if preconditions are not met
- Show a preview of what will be merged
- Exit codes: 0=merged or dry-run ok, 1=precondition failed, 2=timeout or bad
  arguments, 3=no gh auth

### PR Status Monitoring & Watch Mode

**Script:** `scripts/gh-pr-status-watch.sh`

**Continuous check monitoring:**
```sh
# Watch one PR until checks turn green (or a check fails)
scripts/gh-pr-status-watch.sh <PR_NUMBER>

# Several PRs at once, positionally
scripts/gh-pr-status-watch.sh 12 34

# One poll, then exit
scripts/gh-pr-status-watch.sh --once <PR_NUMBER>

# Machine-readable output: one `gh pr view` object per PR, NDJSON
scripts/gh-pr-status-watch.sh --once --json <PR_NUMBER> | jq -s .
```

**Output:**
- Table (default): one row per check — name, status, conclusion. `…` marks a check
  that has not finished, `✗` a failed one, `✓` a passed or skipped one
- `--json`: the raw `gh pr view` object per PR, so `statusCheckRollup` stays an array
- Polls every `--interval=DURATION` (default `10s`), stops at `--timeout`

Exit codes: 0 green, 1 a check failed, 2 timeout, 3 no gh auth, 4 bad arguments.

**Use cases:**
- Monitor long-running CI builds
- Alert when PR becomes blocked by failing tests
- Track slow checks that exceed expected duration

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

### Self-Heal Analysis on CI Failure

**Script:** `scripts/ci-self-heal-analyzer.sh` (analysis only — it never applies anything)

**Failure patterns it recognises:** formatting, compile errors, clippy, test
failures, the ring/`target-cpu` toolchain assert, transient network and resource
failures, plus a step-name fallback when the log matches nothing.

**Commands:**
```sh
# Table of findings (default)
scripts/ci-self-heal-analyzer.sh <RUN_ID>

# Markdown / JSON, and save the report under .qoder/reports/
scripts/ci-self-heal-analyzer.sh --output=md --save <RUN_ID>

# Skip pulling step logs (names only, faster)
scripts/ci-self-heal-analyzer.sh --no-log <RUN_ID>

# Print a draft issue body and the gh issue create command, create nothing
scripts/ci-self-heal-analyzer.sh --output=issue <RUN_ID>
```

Exit codes: 0 findings, 1 patterns not recognised, 2 bad arguments, 3 no gh auth
or run unavailable.

**Safety boundaries:**
- ❌ NO automatic commits, no pushes, no `git add`
- ❌ NO `cargo fmt` and no `cargo clippy --fix` — this repo is not fmt-clean, a
  workspace-wide format reflows other people's files
- ✅ rustfmt is suggested per file, with the exact `--edition` from the log
- ✅ only `gh run rerun --failed` is offered as a mutation, labelled as one

**Example output** (`--output=table` renders a markdown table too; confidence is
`высокая`/`средняя`/`низкая`):
```text
| Категория | Что найдено | Команда | Уверенность | Побочный эффект |
|---|---|---|---|---|
| formatting | rustfmt хочет переформатировать: crates/x/src/lib.rs | `rustfmt --edition 2021 crates/x/src/lib.rs` | высокая | локальные правки |
| toolchain-cpu-flags | ring проверяет фичи CPU: build идёт с -C target-cpu=native из .cargo/config.toml | `RUSTFLAGS="" cargo build --workspace` | высокая | локальная сборка |
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
- Из скриптов автоматизации меняет что-либо только `gh-auto-merge.sh`, и у него есть
  `--dry-run`: показывать решение и команду до действия. `gh-pr-status-watch.sh` и
  `ci-self-heal-analyzer.sh` только читают; у `git-worktree-manager.sh` `prune` принимает
  `--dry-run`, а `remove` не трогает ветку.

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
