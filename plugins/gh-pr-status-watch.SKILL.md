---
name: gh-pr-status-watch
description: Непрерывный мониторинг статусов проверок PR с real-time updates. Использовать когда нужно отслеживать прогресс CI checks, ждать passing builds, или мониторить multiple PRs одновременно.
---

# PR Status Watch via GitHub CLI

Непрерывное слежение за статусами проверок pull requests с автоматическим обновлением статуса.

## Когда использовать

- Ожидание прохождения long-running CI builds (test suites >10 мин)
- Мониторинг multiple PRs параллельно перед merge decisions
- Alerting при failing checks для быстрого triage
- Интеграция в CI dashboards для machine-parseable status
- Terminal watch mode с实时更新

## Установка требований

```sh
# Проверить авторизацию
gh auth status

# Если не авторизован
gh auth login
```

## Основные команды

### Watch Mode (Continuous Monitoring)

```sh
scripts/gh-pr-status-watch.sh [--interval=DURATION] <PR_NUMBER>
```

**Примеры:**
```bash
# Monitor single PR until all checks complete
scripts/gh-pr-status-watch.sh 123

# Custom poll interval (5 seconds)
scripts/gh-pr-status-watch.sh --interval=5s 123

# Exit after one poll (no continuous watching)
scripts/gh-pr-status-watch.sh --interval=0s 123
```

### Batch Mode (Multiple PRs)

```sh
cat pr-list.txt | xargs -I{} scripts/gh-pr-status-watch.sh {}
```

Или напрямую:
```bash
scripts/gh-pr-status-watch.sh 123 456 789
```

### JSON Output (Machine Parseable)

```sh
scripts/gh-pr-status-watch.sh --json <PR_NUMBER> | jq '.checks[]'
```

**Output format:**
```json
{
  "pr": 123,
  "checks": [
    {
      "name": "Build & Test",
      "workflowName": "CI",
      "status": "completed",
      "conclusion": "success",
      "detail": "",
      "url": "https://github.com/..."
    },
    {
      "name": "Security Scan",
      "workflowName": "Security",
      "status": "pending",
      "conclusion": null,
      "detail": "Waiting for queue position #3",
      "url": "https://github.com/..."
    }
  ]
}
```

## Markdown Output Format

Human-readable table with automatic refresh:

```
==================================================
2026-09-26 14:32:15 | Monitoring PRs on owner/repo
PRs: 123
Interval: 10s
==================================================

── PR #123 ──
┌─────────────┬──────────────────┬─────────────┬─────────────┬──────────────────────────┐
│ PR №       │ Check Name        │ Status      │ Result      │ Detail                   │
├─────────────┼──────────────────┼─────────────┼─────────────┼──────────────────────────┤
│ #123        │ Build & Test      │ completed   │ success     │                          │
│ #123        │ Security Scan     │ completed   │ failure     │ 2 high-severity issues   │
│ #123        │ Documentation     │ pending     │ n/a         │ Waiting for queue        │
│ #123        │ Lint              │ completed   │ success     │                          │
└─────────────┴──────────────────┴─────────────┴─────────────┴──────────────────────────┘

📊 Summary: 2 passing, 1 failing, 1 pending
⚠️  WARNING: Some checks are failing!
```

## Integration in Scripts

### Wait for All Checks to Pass

```bash
#!/bin/bash
# wait-for-green.sh

PR_NUM=$1
timeout=3600  # 1 hour max

echo "Waiting for PR #$PR_NUM to become green..."

while true; do
  STATUS=$(scripts/gh-pr-status-watch.sh --json "$PR_NUM" | \
    jq '[.checks[] | select(.status == "completed")] | length')
  
  FAILED=$(scripts/gh-pr-status-watch.sh --json "$PR_NUM" | \
    jq '[.checks[] | select(.status == "completed" and .conclusion == "failure")] | length')
  
  if [[ $FAILED -eq 0 ]] && [[ $STATUS -gt 0 ]]; then
    echo "✓ All checks passed!"
    break
  fi
  
  sleep 30
done
```

### CI Dashboard Integration

```python
# ci_dashboard.py
import subprocess
import json

def get_pr_status(pr_num):
    result = subprocess.run(
        ['bash', 'scripts/gh-pr-status-watch.sh', '--json', str(pr_num)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def render_status(pr_num):
    data = get_pr_status(pr_num)
    print(f"\nPR #{pr_num}")
    print("-" * 50)
    
    for check in data['checks']:
        emoji = "✅" if check['conclusion'] == 'success' else \
                "❌" if check['conclusion'] == 'failure' else \
                "⏳"
        print(f"{emoji} {check['name']}: {check['conclusion'] or 'pending'}")
    
    summary(data['checks'])

def summary(checks):
    passing = sum(1 for c in checks if c['conclusion'] == 'success')
    failing = sum(1 for c in checks if c['conclusion'] == 'failure')
    pending = sum(1 for c in checks if c['status'] != 'completed')
    
    print(f"\nSummary: {passing} passing, {failing} failing, {pending} pending")
```

### Slack/Discord Notifications

```bash
#!/bin/bash
# notify-failing-checks.sh

PR_NUM=$1
SLACK_WEBHOOK=$2

for check in $(scripts/gh-pr-status-watch.sh --json "$PR_NUM" | \
               jq -r '.checks[] | select(.status == "completed" and .conclusion == "failure") | .name'); do
  curl -X POST "$SLACK_WEBHOOK" -H 'Content-type: application/json' \
    --data "{
      \"attachments\":[{
        \"color\":\"danger\",
        \"text\":\"⚠️ Failing check in PR #$PR_NUM: \`$check\`\"
      }]}"
done
```

## Configuration

### Environment Variables

```bash
# Set default polling interval
export DEFAULT_INTERVAL="10s"  # options: s, m, h suffixes

# Set repository if not in git workspace
export GITHUB_REPO="owner/name"
```

### Interval Units

| Suffix | Meaning | Example | Seconds |
|--------|---------|---------|---------|
| s | seconds | `5s` | 5 |
| m | minutes | `2m` | 120 |
| h | hours | `1h` | 3600 |

## Use Cases

### Developer Waiting for Tests

```bash
# Start watching while doing other work
scripts/gh-pr-status-watch.sh 123 &
WATCH_PID=$!

# Do other development work...
git checkout feature-x

# Stop watching when convenient
kill $WATCH_PID
```

### CI Integration Check

```yaml
# .github/workflows/pr-dashboard.yml
name: PR Status Dashboard
on:
  schedule:
    - cron: '*/15 * * * *'  # Every 15 minutes

jobs:
  dashboard:
    runs-on: ubuntu-latest
    steps:
      - name: Get PR statuses
        run: |
          for PR in 123 456 789; do
            bash scripts/gh-pr-status-watch.sh --json "$PR" >> pr-status.json
          done
      
      - name: Generate report
        run: |
          python generate-dashboard-report.py pr-status.json
```

### Blocked PR Detection

```bash
#!/bin/bash
# find-blocked-prs.sh

echo "=== Finding Blocked PRs ==="

for PR in $(gh pr list --state open --json number --jq '.[].number'); do
  CHECKS=$(scripts/gh-pr-status-watch.sh --json "$PR" | jq '.checks[]')
  
  FAILING=$(echo "$CHECKS" | jq '[.[] | select(.conclusion == "failure")] | length')
  
  if [[ $FAILING -gt 0 ]]; then
    echo "🚫 PR #$PR has $FAILING failing checks"
  fi
done
```

## Safety Notes

- **Read-only operations**: Скрипт только читает статус, не меняет ничего
- **Rate limit aware**: Wait_for_rate_limit встроен автоматически
- **Graceful exit**: Ctrl+C stops clean watch loop
- **No side effects**: Даже в watch mode нет изменений в репозитории

## Comparison with `gh pr checks --watch`

GitHub CLI自带命令:
```sh
gh pr checks 123 --watch  # Fixed timing, only one PR, no customization
```

Наши улучшения:
```sh
scripts/gh-pr-status-watch.sh --interval=5s 123  # Customizable interval
scripts/gh-pr-status-watch.sh --json 123         # Machine-parseable output
scripts/gh-pr-status-watch.sh 123 456 789        # Multiple PRs simultaneously
```

Преимущества:
- ✅ Настраиваемый интервал опроса
- ✅ JSON вывод для automation
- ✅ Поддержка batch режимов
- ✅ Кастомная форм