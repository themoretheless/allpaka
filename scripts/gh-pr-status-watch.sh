#!/usr/bin/env bash
# scripts/gh-pr-status-watch.sh — непрерывное мониторинг проверок PR
#
# Использование:
#   scripts/gh-pr-status-watch.sh [OPTIONS] <PR_NUMBER...>
#
# Примеры:
#   # Мониторить один PR до прохода всех checks
#   scripts/gh-pr-status-watch.sh 123
#
#   # Мониторить несколько PR в параллель (batch mode)
#   cat pr-list.txt | xargs -I{} scripts/gh-pr-status-watch.sh {}
#
#   # JSON output для CI integration
#   scripts/gh-pr-status-watch.sh --json 123 | jq '.checks[]'
#
# Выход: Markdown table по умолчанию, можно переключить на JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.qoder/scripts-common.sh"

# --- Configuration ---
DEFAULT_INTERVAL="10s"
OUTPUT_FORMAT="markdown"  # markdown or json

# --- Parse arguments ---
declare -a PR_NUMBERS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) OUTPUT_FORMAT="json"; shift ;;
    --markdown) OUTPUT_FORMAT="markdown"; shift ;;
    --interval=*) INTERVAL="${1#*=}"; shift ;;
    -h|--help)
      cat <<EOF
Использование: $0 [OPTIONS] <PR_NUMBER...>

Непрерывный мониторинг проверок статусов PR.

Опции:
  --json              Вывод в формате JSON (для machine consumption)
  --markdown          Вывод в формате Markdown (по умолчанию)
  --interval=DURATION Интервал опроса (default: $DEFAULT_INTERVAL)

Примеры:
  $0 123                       # Monitor PR #123 until checks pass
  $0 --json 123                # Single poll as JSON
  $0 123 124 125               # Monitor multiple PRs simultaneously
EOF
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        PR_NUMBERS+=("$1")
      else
        echo "Ошибка: неизвестный аргумент '$1'" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ ${#PR_NUMBERS[@]} -eq 0 ]]; then
  echo "Ошибка: нужен хотя бы один номер PR" >&2
  exit 2
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
  GITHUB_REPO=$(git remote get-url origin 2>/dev/null | sed 's/.git$//' | sed 's|.*github.com/||' || echo "")
  if [[ -z "$GITHUB_REPO" ]]; then
    read -rp "Введите репозиторий в формате owner/name: " GITHUB_REPO
  fi
fi

log_action "PR-WATCH" "Starting monitoring for PRs: ${PR_NUMBERS[*]} on repo $GITHUB_REPO"

# Convert interval string to seconds
parse_duration() {
  local dur="$1"
  local num=${dur%[smh]}
  local unit=${dur: -1}
  
  case "$unit" in
    s) echo "$num" ;;
    m) echo "$((num * 60))" ;;
    h) echo "$((num * 3600))" ;;
    *) echo "$num" ;;
  esac
}

INTERVAL_SECONDS=$(parse_duration "${INTERACTIVE_INTERVAL:-$DEFAULT_INTERVAL}")

# --- Monitoring functions ---

# Get PR check status
get_pr_checks() {
  local pr_num="$1"
  gh pr checks "$pr_num" --repo "$GITHUB_REPO" --json name,workflowName,status,conclusion,detail,url --jq '.' 2>/dev/null || echo '{}'
}

# Output as Markdown table
output_markdown_table() {
  local pr_num="$1"
  local checks_json="$2"
  
  printf '\n╔═══════════╤═══════════════════╤═══════════╤═══════════╤════════════════════════╗\n'
  printf '║ PR №     ║ Check Name        ║ Status    ║ Result    ║ Detail                 ║\n'
  printf '╠═══════════╪═══════════════════╪═══════════╪═══════════╪════════════════════════╣\n'
  
  echo "$checks_json" | jq -r '.[] | 
    "║ " + (.[\"name\"] // \"?\") | ljust(20) + 
    " │ " + (.[\"status\"] // \"?\") | ljust(11) + 
    " │ " + (.[\"conclusion\"] // \"?\") | ljust(9) + 
    " │ " + (.[\"detail\"] // \"\")[:30] + " │"'
  
  printf '╚═══════════╧═══════════════════╧═══════════╧═══════════╧════════════════════════╝\n'
}

# Output as JSON
output_json() {
  local pr_num="$1"
  local checks_json="$2"
  
  printf '{"pr": %s, "checks": ' "$pr_num"
  echo "$checks_json"
  printf '}\n'
}

# Parse duration string to seconds
parse_duration() {
  local dur="$1"
  local num=${dur%[smh]}
  local unit=${dur: -1}
  
  case "$unit" in
    s) echo "$num" ;;
    m) echo "$((num * 60))" ;;
    h) echo "$((num * 3600))" ;;
    *) echo "$num" ;;
  esac
}

# Poll all PRs and display results
poll_all_prs() {
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  
  # Clear screen only if not JSON mode (preserve JSON output for piping)
  if [[ "$OUTPUT_FORMAT" == "markdown" ]]; then
    clear
  fi
  
  printf '\n%s\n' "=================================================="
  printf '%s | Monitoring PRs on %s\n' "$timestamp" "$GITHUB_REPO"
  printf 'PRs: %s\n' "${PR_NUMBERS[*]}"
  printf 'Interval: %ds\n' "$INTERVAL_SECONDS"
  printf '==================================================\n'
  
  for pr_num in "${PR_NUMBERS[@]}"; do
    local checks_json
    checks_json=$(get_pr_checks "$pr_num")
    
    # Skip empty results
    if [[ "$checks_json" == "{}" ]] || [[ -z "$checks_json" ]]; then
      continue
    fi
    
    case "$OUTPUT_FORMAT" in
      json)
        output_json "$pr_num" "$checks_json"
        ;;
      markdown)
        printf '\n── PR #%s ──\n' "$pr_num"
        output_markdown_table "$pr_num" "$checks_json"
        
        # Summary line
        local passing failing pending
        passing=$(echo "$checks_json" | jq '[.[] | select(.status == "completed" and .conclusion == "success")] | length')
        failing=$(echo "$checks_json" | jq '[.[] | select(.status == "completed" and .conclusion == "failure")] | length')
        pending=$(echo "$checks_json" | jq '[.[] | select(.status != "completed")] | length')
        
        printf '\n📊 Summary: %s passing, %s failing, %s pending\n' "$passing" "$failing" "$pending"
        
        if [[ $failing -gt 0 ]]; then
          printf '⚠️  WARNING: Some checks are failing!\n'
        elif [[ $passing -gt 0 ]] && [[ $pending -eq 0 ]]; then
          printf '✓ All completed checks passed\n'
        fi
        ;;
    esac
  done
  
  # If watching mode (more than one poll), wait before refreshing
  if [[ "${WATCH_MODE:-false}" == true ]]; then
    sleep "$INTERVAL_SECONDS"
  fi
}

# --- Main execution ---

if [[ ${#PR_NUMBERS[@]} -eq 1 ]] && [[ "${INTERACTIVE_INTERVAL:-$DEFAULT_INTERVAL}" == "$DEFAULT_INTERVAL" ]]; then
  # Watch mode for single PR until all checks complete
  WATCH_MODE=true
  log_info "Watch mode: monitoring PR #${PR_NUMBERS[0]} until all checks complete"
  
  trap 'echo ""; log_info "Stopped by user"; exit 0' INT TERM
  
  while true; do
    poll_all_prs
  done
else
  # Batch mode: just poll once
  log_info "Batch mode: single poll of PRs ${PR_NUMBERS[*]}"
  poll_all_prs
fi
