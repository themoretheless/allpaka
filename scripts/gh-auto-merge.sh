#!/usr/bin/env bash
# scripts/gh-auto-merge.sh — автоматическое слияние PR при выполнении условий
#
# Использование:
#   scripts/gh-auto-merge.sh [OPTIONS] <PR_NUMBER>
#
# Примеры:
#   # Проверить условия без слияния (dry-run)
#   scripts/gh-auto-merge.sh --dry-run 123
#
#   # Слить сsquash strategy (default)
#   scripts/gh-auto-merge.sh --strategy=squash 123
#
#   # Слить с таймаутом (ждать checks до 30 минут)
#   scripts/gh-auto-merge.sh --timeout=30m --strategy merge 123
#
# Выходные коды:
#   0 = успешно слит
#   1 = не выполнены предварительные условия
#   2 = таймаут ожидания checks
#   3 = не авторизован / нет прав на слияние

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.qoder/scripts-common.sh"

# --- Configuration ---
DEFAULT_TIMEOUT="30m"
DEFAULT_STRATEGY="squash"
CHECK_INTERVAL="5s"

# --- Parse arguments ---
DRY_RUN=false
TIMEOUT="$DEFAULT_TIMEOUT"
STRATEGY="$DEFAULT_STRATEGY"
PR_NUM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --strategy=*) STRATEGY="${1#*=}"; shift ;;
    --timeout=*) TIMEOUT="${1#*=}"; shift ;;
    -h|--help)
      cat <<EOF
Использование: $0 [OPTIONS] <PR_NUMBER>

Автоматическое слияние PR при выполнении всех условий.

Опции:
  --strategy=squash|merge|rebase   Стратегия слияния (default: squash)
  --timeout=DURATION               Таймаут ожидания passing checks (default: $DEFAULT_TIMEOUT)
  --dry-run                        Показать что будет сделано без выполнения

Примеры:
  $0 --dry-run 123                # Проверка условий без слияния
  $0 --strategy squash 123         # Слияние с squash
  $0 --timeout=60m 123             # Ждать checks до часа

Выходные коды:
  0 = успешно слит
  1 = не выполнены предварительные условия
  2 = таймаут ожидания checks
  3 = не авторизован / нет прав
EOF
      exit 0
      ;;
    *)
      if [[ -z "$PR_NUM" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
        PR_NUM="$1"
      else
        echo "Ошибка: неизвестный аргумент '$1'" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$PR_NUM" ]]; then
  echo "Ошибка: нужен номер PR" >&2
  exit 2
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
  # Try to get repo from git remote
  GITHUB_REPO=$(git remote get-url origin 2>/dev/null | sed 's/.git$//' | sed 's|.*github.com/||' || echo "")
fi

if [[ -z "$GITHUB_REPO" ]]; then
  read -rp "Введите репозиторий в формате owner/name: " GITHUB_REPO
fi

log_action "AUTO-MERGE" "Starting for PR #$PR_NUM on repo $GITHUB_REPO"

# --- Pre-flight checks ---

echo "=== Checking prerequisites ==="

# 1. Check authentication
if ! gh auth status &>/dev/null; then
  log_error "Not authenticated with GitHub CLI"
  log_info "Run: gh auth login"
  exit 3
fi

# 2. Get PR details
log_info "Fetching PR #$PR_NUM..."
PR_JSON=$(gh pr view "$PR_NUM" --repo "$GITHUB_REPO" --json number,title,state,mergeable,statusCheckRollup,reviewDecision,headRefName,baseRefName --jq '.')

MERGEABLE=$(json_get "$PR_JSON" '.mergeable')
STATE=$(json_get "$PR_JSON" '.state')
TITLE=$(json_get "$PR_JSON" '.title')
CONCLUSION=$(json_get "$PR_JSON" '.statusCheckRollup.conclusion // "none"')
REVIEW_DECISION=$(json_get "$PR_JSON" '.reviewDecision // "unknown"')
HEAD_BRANCH=$(json_get "$PR_JSON" '.headRefName')
BASE_BRANCH=$(json_get "$PR_JSON' .baseRefName)

# Validate basic conditions
echo ""
echo "=== PR Details ==="
printf "  Number:   #%s\n" "$PR_NUM"
printf "  Title:    %s\n" "$TITLE"
printf "  State:    %s\n" "$STATE"
printf "  From:     %s\n" "$HEAD_BRANCH"
printf "  Base:     %s\n" "$BASE_BRANCH"
printf "  Mergeable: %s\n" "$MERGEABLE"
printf "  CI Status: %s\n" "$CONCLUSION"
printf "  Reviews:   %s\n" "$REVIEW_DECISION"

if [[ "$STATE" != "open" ]]; then
  log_error "PR is not open (state: $STATE)"
  exit 1
fi

if [[ "$MERGEABLE" != "true" ]]; then
  log_error "PR is not mergeable"
  log_info "Possible reasons: conflicts, missing checks, no approvals"
  exit 1
fi

# 3. Verify strategy support
case "$STRATEGY" in
  squash|merge|rebase)
    log_info "Using merge strategy: $STRATEGY"
    ;;
  *)
    log_error "Invalid strategy: $STRATEGY (must be: squash|merge|rebase)"
    exit 1
    ;;
esac

# --- Wait for passing checks if needed ---

if [[ "$CONCLUSION" == "waiting" ]] || [[ "$CONCLUSION" == "pending" ]]; then
  log_info "CI checks are still running, waiting..."
  
  local elapsed=0
  local max_wait_seconds
  max_wait_seconds=$(parse_duration "$TIMEOUT")
  
  while [[ $elapsed -lt $max_wait_seconds ]]; do
    sleep 10
    
    local current_pr_json
    current_pr_json=$(gh pr view "$PR_NUM" --repo "$GITHUB_REPO" --json statusCheckRollup --jq '.')
    CONCLUSION=$(json_get "$current_pr_json" '.statusCheckRollup.conclusion // "unknown"')
    
    if [[ "$CONCLUSION" == "success" ]]; then
      log_info "All checks passed!"
      break
    elif [[ "$CONCLUSION" == "failure" ]] || [[ "$CONCLUSION" == "cancelled" ]] || [[ "$CONCLUSION" == "timed_out" ]]; then
      log_error "Checks failed: $CONCLUSION"
      exit 1
    fi
    
    elapsed=$((elapsed + 10))
    printf "  Waiting... (%ds / %ds)\n" "$elapsed" "$max_wait_seconds"
  done
  
  if [[ $elapsed -ge $max_wait_seconds ]]; then
    log_error "Timeout waiting for checks ($TIMEOUT)"
    exit 2
  fi
elif [[ "$CONCLUSION" == "failure" ]]; then
  log_error "Some checks have failed"
  exit 1
elif [[ "$CONCLUSION" == "none" ]] || [[ "$CONCLUSION" == "null" ]]; then
  log_warn "No CI checks found for this PR"
  log_info "Proceeding anyway if mergeable=true"
fi

# --- Final confirmation before merge ---

if [[ "$DRY_RUN" == true ]]; then
  echo ""
  echo "=== Dry Run Mode ==="
  echo "Would perform the following action:"
  printf "  gh pr merge %s --repo %s --strategy %s\n" "$PR_NUM" "$GITHUB_REPO" "$STRATEGY"
  echo ""
  echo "No changes made."
  exit 0
fi

# Show what will happen
echo ""
echo "=== Executing Merge ==="
show_preview "Merge PR #$PR_NUM" \
  "Strategy: $STRATEGY
  Repo: $GITHUB_REPO
  Branch: $HEAD_BRANCH -> $BASE_BRANCH"

# Perform merge
log_info "Merging PR #$PR_NUM..."
if gh pr merge "$PR_NUM" --repo "$GITHUB_REPO" --strategy "$STRATEGY"; then
  log_action "SUCCESS" "PR #$PR_NUM merged successfully with $STRATEGY strategy"
  echo ""
  echo "✓ Successfully merged PR #$PR_NUM"
  exit 0
else
  log_error "Failed to merge PR #$PR_NUM"
  exit 1
fi
