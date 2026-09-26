#!/usr/bin/env bash
# scripts-common.sh — общие функции для всех скриптов GitHub workflow automation
# Подключать через: source "$0"

set -euo pipefail

# Directory structure for logs and reports
readonly LOG_DIR="${LOG_DIR:-.qoder/logs}"
readonly REPORTS_DIR="${REPORTS_DIR:-.qoder/reports}"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$REPORTS_DIR"

# Logging functions (color-coded)
log_debug() { printf '[DEBUG] %s\n' "$*" >&2; }
log_info()  { printf '[INFO]  %s\n' "$*" >&2; }
log_warn()  { printf '[WARN]  %s\n' "$*" >&2; }
log_error() { printf '[ERROR] %s\n' "$*" >&2; }

# Timestamp for logs
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Log helper that writes to both stdout and log file
log_action() {
  local action="$1"; shift
  local msg="$*"
  local log_file="$LOG_DIR/$(date +%Y%m%d).log"
  printf '[%s] %s: %s\n' "$(timestamp)" "$action" "$msg" | tee -a "$log_file"
}

# Rate limit backoff for gh API
# Usage: wait_for_rate_limit [max_wait_seconds]
wait_for_rate_limit() {
  local max_wait="${1:-60}"
  local waited=0
  
  while true; do
    local rate_info
    if ! rate_info=$(gh api rate_limit --jq '.' 2>/dev/null); then
      # If we can't check rate limit, assume we're good
      return 0
    fi
    
    local remaining
    remaining=$(echo "$rate_info" | jq -r '.resource.remaining // 0')
    
    if [[ "$remaining" -gt 5 ]]; then
      return 0
    fi
    
    local reset_ts
    reset_ts=$(echo "$rate_info" | jq -r '.resource.reset | fromdateiso8601 // empty')
    
    if [[ -n "$reset_ts" ]] && [[ $reset_ts -le $(date +%s) ]]; then
      return 0
    fi
    
    if [[ $waited -ge $max_wait ]]; then
      log_error "Exceeded max wait time ($max_wait s) for rate limit"
      return 1
    fi
    
    sleep 5
    waited=$((waited + 5))
  done
}

# JSON parsing helper with error handling
# Usage: json_get <json_string> <jq_filter> [default_value]
json_get() {
  local json="$1"
  local filter="$2"
  local default="${3:-}"
  
  echo "$json" | jq -r "$filter // \"$default\"" 2>/dev/null || echo "$default"
}

# Check if PR is mergeable
# Usage: check_pr_mergeable <repo_owner/repo> <pr_number>
check_pr_mergeable() {
  local repo="$1"
  local pr_num="$2"
  
  local pr_data
  pr_data=$(gh pr view "$pr_num" --repo "$repo" --json mergeable,statusCheckRollup,reviewDecision --jq '.') 2>/dev/null
  
  [[ "$(json_get "$pr_data" '.mergeable == "TRUE"')" == "true" ]] || return 1
}

# Check CI status for PR
# Usage: check_pr_ci_status <repo_owner/repo> <pr_number>
check_pr_ci_status() {
  local repo="$1"
  local pr_num="$2"
  
  local pr_data
  pr_data=$(gh pr view "$pr_num" --repo "$repo" --json statusCheckRollup --jq '.') 2>/dev/null
  
  local conclusion
  conclusion=$(json_get "$pr_data" '.statusCheckRollup.conclusion // ""')
  
  [[ "$conclusion" == "success" ]]
}

# Show preview of what will be changed
# Usage: show_preview <action_type> <details...>
show_preview() {
  local action="$1"; shift
  local details="$*"
  
  printf '\n%s\n' "=================================================="
  printf 'ACTION: %s\n' "$action"
  printf '%s\n' "$details"
  printf '==================================================\n\n'
}

# Dry-run mode support
DRY_RUN=false
usage_with_dryrun() {
  cat <<EOF
Usage: $0 [OPTIONS] <ARGUMENTS>

Common options:
  --dry-run       Show what would be done without making changes
  --verbose       Enable debug output
  -h, --help      Show this help message

Arguments:
  <ARGUMENTS>     Script-specific arguments
EOF
}

parse_dryrun_flag() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) DRY_RUN=true; shift ;;
      *) break ;;
    esac
  done
}
