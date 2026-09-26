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
1. PR has no conflicts (`mergeable=true`)
2. All required status checks pass (`statusCheckRollup.conclusion == success`)
3. Required reviews present (check via `reviewDecision` field)
4. Branch protection rules respected

**Commands:**
```sh
# Dry-run: check conditions without merging
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>

# Merge with squash strategy (default)
scripts/gh-auto-merge.sh --strategy=squash <PR_NUMBER>

# Merge with timeout (wait up to 30 min for checks)
scripts/gh-auto-merge.sh --timeout=30m --strategy merge <PR_NUMBER>
```

**Safety:**
- Never bypass branch protection rules
- Fail fast if preconditions not met
- Show preview of what will be merged
- Exit codes: 0=merged, 1=failed precondition, 2=timeout, 3=not authorized

### PR Status Monitoring & Watch Mode

**Script:** `scripts/gh-pr-status-watch.sh`

**Continuous check monitoring:**
```sh
# Watch single PR until all checks pass
scripts/gh-pr-status-watch.sh <PR_NUMBER>

# Batch mode for multiple PRs
scripts/gh-pr-status-watch.sh --batch pr-list.txt

# JSON output for CI integration
scripts/gh-pr-status-watch.sh --json <PR_NUMBER> | jq '.checks[]'
```

**Output formats:**
- Markdown table (default): human-readable status per check
- JSON: machine-parseable for automation tools
- Updates every N seconds (configurable interval)

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

**Script:** `scripts/ci-self-heal-analyzer.sh` (conservative suggestion mode)

**Analyzes common failure patterns:**
- Formatting issues (`cargo fmt --check`)
- Clippy warnings (`cargo clippy`)
- Test expectation mismatches
- Linting errors

**Commands:**
```sh
# Generate fix suggestions as markdown table
scripts/ci-self-heal-analyzer.sh --output=table <RUN_ID>

# Output diff snippets for manual application
scripts/ci-self-heal-analyzer.sh --output=diff <RUN_ID>

# Optional: create draft issue with analysis
scripts/ci-self-heal-analyzer.sh --output=issue <RUN_ID>
```

**Safety boundaries:**
- ❌ NO automatic commits
- ❌ NO pushing changes anywhere
- ✅ Only generates suggestions in stdout/file
- ✅ Includes clear "how to apply" instructions

**Example output:**
```markdown
## CI Failure Analysis

| Issue Type | Files Affected | Fix Command | Confidence |
|------------|----------------|-------------|------------|
| formatting | backend/src/lib.rs | `cargo fmt --package allpaka-backend` | 100% |
| clippy | src/cli.rs | `cargo clippy --fix --package allpaka-cli` | 95% |

To apply fixes manually:
  $ cargo fmt
  $ cargo clippy --fix
  $ git commit -m "chore: self-heal CI fixes"
```

## Релизы и прочее

```sh
gh release list --limit 10
gh release create <tag> --notes "..."   # только по просьбе, публикация необратима
gh label list
gh search prs --repo owner/name "query" --limit 10
gh search repos "query" --limit 10
```

### PR Lifecycle Hooks

**Script:** `scripts/gh-pr-lifecycle-hooks.sh`

**Automated maintenance operations:**
- Close stale PRs (inactive >30 days, configurable)
- Label management based on code owners
- Weekly status reports via GitHub Issues

**Commands:**
```sh
# Find and close stale PRs (>30 days inactive)
scripts/gh-pr-lifecycle-hooks.sh stale-pr-cleanup --age-threshold=30d --action=close|comment

# Get weekly PR dashboard
scripts/gh-pr-lifecycle-hooks.sh weekly-dashboard --output=.qoder/reports/pr-weekly.md

# Manage labels based on ownership patterns
scripts/gh-pr-lifecycle-hooks.sh sync-labels --codeowners=CODEOWNERS
```

**Safety:**
- Only acts on PRs explicitly marked as stale by timeout
- Always previews what would be changed before acting
- Requires explicit confirmation for destructive actions
- Logs all changes to audit trail

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
- Скрипты автоматизации (`gh-auto-merge.sh`, `gh-pr-lifecycle-hooks.sh`) имеют флаги `--dry-run`:
  использовать для проверки перед фактическим действием

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
