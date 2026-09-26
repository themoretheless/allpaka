#!/usr/bin/env bash
# scripts/ci-self-heal-analyzer.sh — анализ неудач CI и предложение исправлений
#
# Использование:
#   scripts/ci-self-heal-analyzer.sh [OPTIONS] <RUN_ID>
#
# Примеры:
#   # Анализ с выводом таблицы исправлений
#   scripts/ci-self-heal-analyzer.sh --output=table 123456
#
#   # Вывод diff сниппетов для ручного применения
#   scripts/ci-self-heal-analyzer.sh --output=diff 123456
#
#   # Создание черновика issue с анализом
#   scripts/ci-self-heal-analyzer.sh --output=issue 123456
#
# ВНИМАНИЕ: этот скрипт только анализирует и предлагает исправления, 
# но не применяет их автоматически!

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.qoder/scripts-common.sh"

# --- Configuration ---
OUTPUT_FORMAT="table"  # table, diff, or issue
COMMIT_FIXES=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output=*) OUTPUT_FORMAT="${1#*=}"; shift ;;
    --commit) COMMIT_FIXES=true; shift ;;
    -h|--help)
      cat <<EOF
Использование: $0 [OPTIONS] <RUN_ID>

Анализ неудач CI и предложение консервативных исправлений.

Опции:
  --output=table|diff|issue   Формат вывода (default: table)
                              table - Markdown таблица с командой исправления
                              diff  - Diff сниппеты для применения
                              issue - Черновик issue/PR с описанием
  
  --commit                    Применить исправления автоматически
                              (ВНИМАНИЕ: все равно требует подтверждения!)

Примеры:
  $0 --output=table 123456        # Генерировать таблицу исправлений
  $0 --output=diff 123456         # Показать diff сниппеты
  $0 --output=issue 123456        # Создать черновик issue
  $0 --commit 123456              # Применить исправления (требует подтверждения)

ВЫХОДНЫЕ КОДЫ:
  0 = успешно проанализировал
  1 = ошибка анализа или нет данных для анализа
  2 = ошибка авторизации / доступа

ВАЖНО: Этот скрипт НЕ ПРИМЕНЯЕТ исправления автоматически!
Он только анализирует и показывает что можно исправить.
EOF
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        RUN_ID="$1"
      else
        echo "Ошибка: неизвестный аргумент '$1'" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  echo "Ошибка: нужен ID запуска CI" >&2
  exit 2
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
  GITHUB_REPO=$(git remote get-url origin 2>/dev/null | sed 's/.git$//' | sed 's|.*github.com/||' || echo "")
fi

if [[ -z "$GITHUB_REPO" ]]; then
  read -rp "Введите репозиторий в формате owner/name: " GITHUB_REPO
fi

log_action "CI-ANALYZER" "Analyzing CI run #$RUN_ID on repo $GITHUB_REPO"

# --- Analysis functions ---

# Get run details
get_run_info() {
  local run_id="$1"
  gh run view "$run_id" --repo "$GITHUB_REPO" --json status,conclusion,workflowName,headBranch --jq '.' 2>/dev/null
}

# Get failed jobs for a run
get_failed_jobs() {
  local run_id="$1"
  gh run view "$run_id" --repo "$GITHUB_REPO" --job --json name,status,conclusion,outcome --jq '.jobs[] | select(.status != "completed")' 2>/dev/null || true
}

# Analyze common failure patterns
analyze_cargo_format() {
  log_info "Checking for formatting issues..."
  
  # Run cargo fmt check in dry-run mode
  local format_output
  if format_output=$(cargo fmt --check --all 2>&1); then
    return 0
  fi
  
  # Extract formatted files
  local affected_files
  affected_files=$(echo "$format_output" | grep "^diff" | awk '{print $NF}' | sort -u)
  
  if [[ -n "$affected_files" ]]; then
    echo "formatting"
    echo "$affected_files"
    return 0
  fi
  
  return 1
}

analyze_cargo_clippy() {
  log_info "Checking for clippy warnings..."
  
  # Run clippy check
  local clippy_output
  if clippy_output=$(cargo clippy --all-targets 2>&1); then
    return 0
  fi
  
  # Extract warning-prone areas (not failures)
  local issues
  issues=$(echo "$clippy_output" | grep -E "(warning:|help:)" | head -20)
  
  if [[ -n "$issues" ]]; then
    echo "clippy"
    echo "$issues"
    return 0
  fi
  
  return 1
}

analyze_test_failures() {
  log_info "Analyzing test failures..."
  
  # Check test output from logs
  local test_log
  test_log=$(gh run download "$RUN_ID" --name test-log --repo "$GITHUB_REPO" 2>/dev/null || true)
  
  if [[ -n "$test_log" ]]; then
    local failing_tests
    failing_tests=$(echo "$test_log" | grep -A5 "thread.*panicked\|test result: FAILED" | head -30)
    
    if [[ -n "$failing_tests" ]]; then
      echo "test_expectation"
      echo "$failing_tests"
      return 0
    fi
  fi
  
  return 1
}

# Generate markdown table output
generate_table_report() {
  echo ""
  echo "## CI Failure Analysis Report"
  echo ""
  printf '| Issue Type | Files Affected | Fix Command | Confidence |\n'
  printf '|------------|----------------|-------------|------------|\n'
  
  local found_issues=false
  
  # Check formatting
  if analyze_cargo_format > /tmp/format_check.txt 2>&1; then
    local issue_type
    issue_type=$(sed -n '1p' /tmp/format_check.txt)
    local files
    files=$(tail -n +2 /tmp/format_check.txt | tr '\n' ', ' | sed 's/,$//')
    
    if [[ "$issue_type" == "formatting" ]] && [[ -n "$files" ]]; then
      printf '| %s | %s | \`cargo fmt\` | 100%% |\n' "$issue_type" "$files"
      found_issues=true
    fi
    rm -f /tmp/format_check.txt
  fi
  
  # Check clippy
  if analyze_cargo_clippy > /tmp/clippy_check.txt 2>&1; then
    local issue_type
    issue_type=$(sed -n '1p' /tmp/clippy_check.txt)
    
    if [[ "$issue_type" == "clippy" ]]; then
      printf '| %s | multiple files | \`cargo clippy --fix --allow-dirty\` | 95%% |\n' "$issue_type"
      found_issues=true
    fi
    rm -f /tmp/clippy_check.txt
  fi
  
  # Check tests
  if analyze_test_failures > /tmp/test_check.txt 2>&1; then
    local issue_type
    issue_type=$(sed -n '1p' /tmp/test_check.txt)
    
    if [[ "$issue_type" == "test_expectation" ]]; then
      printf '| %s | see below | Review test expectations | 80%% |\n' "$issue_type"
      found_issues=true
    fi
    rm -f /tmp/test_check.txt
  fi
  
  if [[ "$found_issues" == false ]]; then
    echo "| No automatic fixes detected | - | Manual inspection required | - |"
  fi
  
  echo ""
  echo "### Recommended Actions"
  echo ""
  
  # Always suggest these basic fixes
  echo "```bash"
  echo "# 1. Apply formatting fixes"
  echo "cargo fmt"
  echo ""
  echo "# 2. Apply clippy fixes (review carefully!)"
  echo "cargo clippy --fix --allow-dirty --allow-staged"
  echo ""
  echo "# 3. Re-run tests"
  echo "cargo test --workspace"
  echo "```"
  echo ""
  echo "After applying fixes:"
  echo "  \$ git commit -m \"chore: self-heal CI fixes\""
  echo "  \$ git push"
  echo ""
}

# Generate diff snippets
generate_diff_report() {
  echo ""
  echo "## Potential Auto-Fixes (Diff Output)"
  echo ""
  
  # Formatting diff
  if analyze_cargo_format > /tmp/format_check.txt 2>&1; then
    echo "### Formatting corrections"
    echo "```diff"
    cargo fmt --check --all 2>/dev/null || true
    echo "```"
    echo ""
    rm -f /tmp/format_check.txt
  fi
  
  # Clippy suggestions
  if analyze_cargo_clippy > /tmp/clippy_check.txt 2>&1; then
    echo "### Clippy suggestions"
    echo "Check for 'help:' lines in clippy output showing suggested fixes."
    echo ""
    rm -f /tmp/clippy_check.txt
  fi
}

# Generate issue template
generate_issue_report() {
  local run_info
  run_info=$(get_run_info "$RUN_ID")
  
  local workflow_name
  workflow_name=$(json_get "$run_info" '.workflowName // "Unknown"')
  local conclusion
  conclusion=$(json_get "$run_info" '.conclusion // "unknown"')
  local branch
  branch=$(json_get "$run_info" '.headBranch // "unknown"')
  
  echo "```markdown"
  echo "# Automated CI Self-Heal Suggestion"
  echo ""
  echo "## CI Run Details"
  echo "- **Workflow**: $workflow_name"
  echo "- **Status**: $conclusion"
  echo "- **Branch**: $branch"
  echo "- **Run ID**: $RUN_ID"
  echo ""
  echo "## Analysis Results"
  echo ""
  
  generate_table_report
  
  echo ""
  echo "---"
  echo ""
  echo "**Generated automatically by `ci-self-heal-analyzer.sh`**"
  echo "Please review suggestions before committing."
  echo "```"
}

# Apply fixes (requires explicit confirmation)
apply_fixes() {
  echo ""
  echo "=== Applying Fixes ==="
  show_preview "Auto-fix application" \
    "This will apply all detected fixes and commit them."
  
  if [[ "${DRY_RUN:-false}" != true ]]; then
    read -rp "Confirm application of all fixes? [y/N]: " confirm
    if [[ "${confirm,,}" != "y" ]]; then
      log_info "Aborted by user"
      return 1
    fi
  fi
  
  log_info "Applying fixes..."
  
  # Apply formatting
  log_info "Applying formatting fixes..."
  cargo fmt || {
    log_error "Formatting fix failed"
    return 1
  }
  
  # Apply clippy fixes
  log_info "Applying clippy fixes..."
  cargo clippy --fix --allow-dirty --allow-staged || {
    log_warn "Clippy fix had some errors, but continuing..."
  }
  
  # Stage changes
  git add -A
  
  # Commit
  git commit -m "chore: self-heal CI fixes" || {
    log_warn "No changes to commit (or commit failed)"
    return 1
  }
  
  log_action "SUCCESS" "Applied fixes and committed"
  echo ""
  echo "✓ Fixes applied successfully!"
  echo ""
  echo "Next steps:"
  echo "  - Review changes: git diff HEAD~1"
  echo "  - Push to trigger re-run: git push"
  
  return 0
}

# --- Main execution ---

# Verify CI status first
log_info "Fetching CI run info..."
if ! run_info=$(get_run_info "$RUN_ID"); then
  log_error "Failed to fetch CI run #$RUN_ID"
  log_info "Make sure you have access to this repository"
  exit 1
fi

local conclusion
conclusion=$(json_get "$run_info" '.conclusion')

if [[ "$conclusion" != "failure" ]]; then
  log_warn "CI run #$RUN_ID did not fail (conclusion: $conclusion)"
  log_info "Analysis is most useful for failed runs"
fi

log_info "Analyzing failure patterns..."

# Generate report based on format
case "$OUTPUT_FORMAT" in
  table)
    generate_table_report
    ;;
  diff)
    generate_diff_report
    ;;
  issue)
    generate_issue_report
    ;;
  *)
    log_error "Invalid output format: $OUTPUT_FORMAT"
    exit 1
    ;;
esac

# If --commit was requested, apply fixes
if [[ "$COMMIT_FIXES" == true ]]; then
  if apply_fixes; then
    exit 0
  else
    exit 1
  fi
fi

exit 0
