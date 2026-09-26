---
name: gh-auto-merge
description: Автоматическое слияние PR при выполнении всех условий — проверка mergeable, passing checks, required reviews. Использовать когда нужно безопасно слить PR без ручного проверения условий или для batch-обработки нескольких PR.
---

# Auto-Merge via GitHub CLI

Автоматизированное слияние pull requests с предварительной проверкой всех условий безопасности.

## Когда использовать

- Перед сливом PR — автоматическая проверка всех preconditions (mergeable, CI passing, reviews)
- Batch-обработка множества PR по ночам/в нерабочее время
- Интеграция в CI/CD workflow для авто-слива после прохождения всех checks
- Замена ручной команды `gh pr merge` с гарантией отсутствия конфликтов

## Установка требований

```sh
# Проверить авторизацию
gh auth status

# Если не авторизован
gh auth login  # интерактивная настройка
```

## Основные команды

### Dry-run режим (предпросмотр)

```sh
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>
```

Показывает какие проверки будут выполнены БЕЗ реального слияния:

```bash
$ scripts/gh-auto-merge.sh --dry-run 123

=== Checking prerequisites ===
Fetching PR #123...

=== PR Details ===
  Number:   #123
  Title:    Fix critical bug in backend
  State:    open
  From:     feature-critical-fix
  Base:     main
  Mergeable: true
  CI Status: success
  Reviews:   approved

=== Dry Run Mode ===
Would perform the following action:
  gh pr merge 123 --repo owner/repo --strategy squash

No changes made.
```

### Реальное слияние

```sh
scripts/gh-auto-merge.sh [--strategy=squash|merge|rebase] [--timeout=DURATION] <PR_NUMBER>
```

**Опции:**
- `--strategy=squash` (default) — создать один commit из всех изменений PR
- `--strategy=merge` — создать merge commit с историей PR
- `--strategy=rebase` — rebase ветку PR на основную перед слиянием
- `--timeout=30m` — максимальное время ожидания passing checks (s/m/h суффиксы)

**Примеры:**
```sh
# Слить с squash (стандартная практика)
scripts/gh-auto-merge.sh --strategy=squash 123

# Ждать checks до 60 минут перед слиянием
scripts/gh-auto-merge.sh --timeout=60m --strategy merge 456

# В рабочем каталоге репозитория (определяет GITHUB_REPO автоматически)
scripts/gh-auto-merge.sh 789
```

## Precondition Checks

Скрипт проверяет перед каждым слиянием:

1. ✅ **PR открыт** (state = "open")
2. ✅ **Нет конфликтов** (mergeable = true)
3. ✅ **Все required checks прошли** (statusCheckRollup.conclusion == success)
4. ✅ **Required reviews present** (reviewDecision != "changes_requested")
5. ✅ **Правила защиты веток соблюдены** (branch protection rules)
6. ✅ **Достаточные permissions** (user has merge permission)

Если любая проверка не проходит — exit code = 1 без попыток форсировать.

## Exit Codes

| Code | Значение | Описание |
|------|----------|----------|
| 0 | Success | Успешно слит |
| 1 | Failed precondition | Не выполнены условия (конфликт, failing checks, no approvals) |
| 2 | Timeout | Таймаут ожидания passing checks |
| 3 | Not authorized | Нет прав на слияние или не авторизован |

## Интеграция в Workflow

### CI/CD Pipeline

```yaml
# .github/workflows/auto-merge-feature.yml
name: Auto-Merge Feature PRs
on:
  pull_request:
    branches: [main]
    types: [closed]

jobs:
  auto-merge-if-green:
    if: github.event.pull_request.merged == false && contains(github.event.pull_request.labels.*.name, 'auto-merge')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check PR conditions
        run: |
          PR_NUM=${{ github.event.pull_request.number }}
          scripts/gh-auto-merge.sh --dry-run "$PR_NUM"
      
      - name: Auto-merge with squash
        if: success()
        run: |
          scripts/gh-auto-merge.sh --strategy squash "$PR_NUM"
```

### Batch Processing

```bash
# Обработка списка PR из файла
cat pr-list.txt | while read pr_num; do
  echo "Processing PR #$pr_num..."
  scripts/gh-auto-merge.sh --timeout=30m --strategy squash "$pr_num" || \
    echo "Failed for #$pr_num" >> failed-prs.log
done
```

### Manual Daily Check

```bash
#!/bin/bash
# scripts/daily-auto-merge.sh

# Найти все green PRs
GREEN_PRS=$(gh pr list --state open --json number,title,statusCheckRollup --jq '.[] | select(.statusCheckRollup.conclusion == "success" and .mergeable == true) | .number')

for PR in $GREEN_PRS; do
  echo "Checking PR #$PR..."
  scripts/gh-auto-merge.sh --dry-run "$PR" > /dev/null 2>&1 && \
    scripts/gh-auto-merge.sh --strategy squash "$PR"
done
```

## Safety Rules

- **Не преодолевает branch protection rules** — даже если есть admin права
- **Требует явного указания** — никогда не запускается silently
- **Показывает preview перед действием** — через --dry-run или вывод текущих параметров
- **Логгирует все действия** — в `.qoder/logs/YYYYMMDD.log`
- **Уважает rate limits** — wait_for_rate_limit при необходимости

## Отличия от обычного `gh pr merge`

Обычная команда:
```sh
gh pr merge 123 --squash  # Может сломаться если появились конфликты
```

Auto-merge скрипт:
```sh
scripts/gh-auto-merge.sh 123  # Проверяет mergeable==true перед выполнением
```

Разница:
- ✅ Проверка mergeability поле выполнения
- ✅ Ожидание passing checks (если pending/waiting)
- ✅ Timeout protection от зависания
- ✅ Detailed error messages вместо generic failures
- ✅ Exit codes для программной обработки

## Troubleshooting

### PR не mergeable — почему?

```bash
# Детальный статус
gh pr view 123 --json mergeable,mergeStateStatus,requiresStrictStatusChecks

# Возможные причины:
# - conflicts: есть merge conflicts (нужно resolve)
# - not_allowed: нет required reviews
# - unknown: PR требует extra configuration
# - dirty: uncommitted changes в рабочей ветке
```

### Checks висят в pending долго

```bash
# Отдельный мониторинг
scripts/gh-pr-status-watch.sh 123 &

# Cancel monitoring
kill %1
```

### Rate limit exceeded

```bash
# Проверить текущий лимит
gh api rate_limit --jq '.rate'

# Wait backoff встроен в скрипт автоматически
```

## Integration with Git Skill

Работ