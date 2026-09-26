#!/usr/bin/env bash
# scripts/git-worktree-manager.sh — автоматизация управления worktrees
#
# Использование:
#   scripts/git-worktree-manager.sh [command] [options] [arguments]
#
# Команды:
#   add <branch> <path>         Создать новый worktree для ветки
#   remove <path>|<branch>      Удалить worktree и опционально ветку
#   list [--verbose|--json]     Показать список всех worktrees
#   prune                       Удалить орфанованные записи
#   sync <source> <target>      Синхронизировать изменения между worktrees (показать preview)
#
# Примеры:
#   # Создать изолированное рабочее дерево для feature-ветки
#   scripts/git-worktree-manager.sh add feature-x ./worktrees/feature-x
#
#   # Перечислить все worktrees
#   scripts/git-worktree-manager.sh list
#
#   # Удалить worktree с веткой
#   scripts/git-worktree-manager.sh remove feature-x
#
#   # Очистить старые записи
#   scripts/git-worktree-manager.sh prune

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.qoder/scripts-common.sh"

WORKTREE_BASE="${WORKTREE_BASE:-./worktrees}"

# --- Commands ---

cmd_add() {
  local branch="$1"
  local path="$2"
  
  log_action "WORKTREE-ADD" "Creating worktree for branch '$branch' at '$path'"
  
  # Check if path already exists
  if [[ -e "$path" ]]; then
    log_error "Path already exists: $path"
    exit 1
  fi
  
  # Ensure base directory exists
  mkdir -p "$(dirname "$path")"
  
  # Check disk space (at least 100MB free required)
  local available
  available=$(df -m "$(dirname "$path")" | awk 'NR==2 {print $4}')
  if [[ $available -lt 100 ]]; then
    log_error "Insufficient disk space (need at least 100MB, have ${available}MB)"
    exit 1
  fi
  
  # Show preview
  echo ""
  echo "=== Creating Worktree ==="
  printf "Branch: %s\n" "$branch"
  printf "Path:   %s\n" "$path"
  printf "Command: git worktree add -b %s %s\n" "$branch" "$path"
  
  if [[ "${DRY_RUN:-false}" == true ]]; then
    log_info "Dry run mode: no changes made"
    return 0
  fi
  
  # Create worktree
  if git worktree add -b "$branch" "$path"; then
    log_info "Successfully created worktree at $path"
    echo ""
    printf "Switch to new worktree: cd %s\n" "$path"
  else
    log_error "Failed to create worktree"
    exit 1
  fi
}

cmd_remove() {
  local target="$1"
  local branch="${2:-}"
  
  log_action "WORKTREE-REMOVE" "Removing worktree targeting '$target'"
  
  # Determine if target is path or branch name
  local worktree_path=""
  local actual_branch=""
  
  if [[ -d "$target" ]] && git -C "$target" rev-parse --is-inside-work-tree &>/dev/null; then
    worktree_path="$target"
    actual_branch=$(git -C "$target" rev-parse --abbrev-ref HEAD)
  elif [[ -n "$branch" ]]; then
    # User provided explicit branch name
    # Find the worktree for this branch
    while IFS= read -r line; do
      local wb wt
      wb=$(echo "$line" | cut -d' ' -f1)
      wt=$(echo "$line" | cut -d' ' -f3-)
      if [[ "$wt" == "$branch" ]] || [[ "$wb" == *"branch/$branch"* ]]; then
        worktree_path="$wb"
        actual_branch="$branch"
        break
      fi
    done < <(git worktree list)
    
    if [[ -z "$worktree_path" ]]; then
      log_error "Worktree for branch '$branch' not found"
      exit 1
    fi
  else
    # Try to infer from target
    if git -C "$target" rev-parse --is-inside-work-tree &>/dev/null; then
      worktree_path="$target"
      actual_branch=$(git -C "$target" rev-parse --abbrev-ref HEAD)
    else
      log_error "Unknown target: $target (path or branch name expected)"
      exit 1
    fi
  fi
  
  if [[ -z "$worktree_path" ]]; then
    log_error "Could not determine worktree path"
    exit 1
  fi
  
  # Check for uncommitted changes
  local has_changes
  has_changes=$(git -C "$worktree_path" status --porcelain 2>/dev/null | wc -l)
  
  if [[ $has_changes -gt 0 ]]; then
    log_warn "Worktree has uncommitted changes:"
    git -C "$worktree_path" status --short
    echo ""
    if [[ "${FORCE:-false}" != true ]]; then
      read -rp "Remove anyway? (add --force to skip check) [y/N]: " confirm
      if [[ "${confirm,,}" != "y" ]]; then
        log_info "Aborted"
        exit 0
      fi
    fi
  fi
  
  # Show preview
  echo ""
  echo "=== Removing Worktree ==="
  printf "Path:   %s\n" "$worktree_path"
  printf "Branch: %s\n" "$actual_branch"
  printf "Commands:\n"
  printf "  git worktree remove %s\n" "$worktree_path"
  printf "  git branch -D %s\n" "$actual_branch"
  
  if [[ "${DRY_RUN:-false}" == true ]]; then
    log_info "Dry run mode: no changes made"
    return 0
  fi
  
  # Remove worktree
  if git worktree remove "$worktree_path"; then
    log_info "Successfully removed worktree at $worktree_path"
    
    # Optionally delete branch (local only)
    if git branch -D "$actual_branch" 2>/dev/null; then
      log_info "Deleted local branch: $actual_branch"
    else
      log_info "Branch not deleted or doesn't exist locally: $actual_branch"
    fi
  else
    log_error "Failed to remove worktree"
    exit 1
  fi
}

cmd_list() {
  local verbose=false
  local json_output=false
  
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --verbose) verbose=true; shift ;;
      --json) json_output=true; shift ;;
      *) break ;;
    esac
  done
  
  log_action "WORKTREE-LIST" "Listing all worktrees"
  
  if [[ "$json_output" == true ]]; then
    # JSON output for machine consumption
    if [[ "$verbose" == true ]]; then
      git worktree list --json
    else
      git worktree list --json | jq '[.[] | {path, branch: .[\"head\"].branch // \"HEAD\"}]'
    fi
  else
    # Human-readable table
    echo ""
    echo "=== Git Worktrees ==="
    echo ""
    
    if [[ "$verbose" == true ]]; then
      printf '| Path              | Branch         | Commit SHA          |\n'
      printf '|-------------------|----------------|---------------------|\n'
      
      while IFS= read -r line; do
        local path branch sha
        path=$(echo "$line" | cut -d' ' -f1)
        branch=$(echo "$line" | sed 's/^.*branch: \(.*\) (.*/\1/')
        sha=$(echo "$line" | grep -oE '[a-f0-9]{7,40}' | head -1)
        
        printf '| %-17s | %-14s | %-19s |\n' "$path" "$branch" "$sha"
      done < <(git worktree list)
    else
      git worktree list
    fi
    
    # Summary count
    local count
    count=$(git worktree list | wc -l)
    echo ""
    printf "Total worktrees: %s\n" "$count"
  fi
}

cmd_prune() {
  log_action "WORKTREE-PRUNE" "Pruning stale worktree entries"
  
  # Check dry-run flag
  if [[ "${DRY_RUN:-false}" == true ]]; then
    echo "=== Prune Dry Run ==="
    local stale_entries
    stale_entries=$(git worktree prune --dry-run 2>&1 || true)
    
    if [[ -n "$stale_entries" ]]; then
      echo "Would remove the following stale entries:"
      echo "$stale_entries"
    else
      echo "No stale entries found"
    fi
    return 0
  fi
  
  # Execute prune
  local output
  output=$(git worktree prune 2>&1) || {
    log_error "Prune failed: $output"
    exit 1
  }
  
  if [[ -n "$output" ]]; then
    log_info "Removed stale entries:"
    echo "$output"
  else
    log_info "No stale entries found"
  fi
}

cmd_sync() {
  local source_path="$1"
  local target_branch="$2"
  
  log_action "WORKTREE-SYNC" "Syncing between worktrees"
  
  # Validate paths
  if ! git -C "$source_path" rev-parse --is-inside-work-tree &>/dev/null; then
    log_error "Invalid source worktree: $source_path"
    exit 1
  fi
  
  # Show what would be synced
  echo ""
  echo "=== Sync Preview ==="
  printf "Source: %s\n" "$source_path"
  printf "Target branch: %s\n" "$target_branch"
  
  # Get diff between current branch in source and target branch
  local source_branch
  source_branch=$(git -C "$source_path" rev-parse --abbrev-ref HEAD)
  
  echo ""
  echo "Changes in $source_branch that would be applied to $target_branch:"
  echo ""
  
  # Fetch latest from remote
  git -C "$source_path" fetch origin 2>/dev/null || true
  
  # Show commits
  git -C "$source_path" log "origin/$target_branch".."$source_branch" --oneline
  
  if [[ "${DRY_RUN:-false}" == true ]]; then
    echo ""
    log_info "Dry run mode: no changes made"
  else
    read -rp "Apply these changes to $target_branch? [y/N]: " confirm
    if [[ "${confirm,,}" == "y" ]]; then
      git checkout "$target_branch"
      git merge "$source_branch" --no-edit
      log_info "Successfully synced changes"
    else
      log_info "Cancelled"
    fi
  fi
}

# --- Main argument parsing ---

CMD="${1:-help}"
shift || true

case "$CMD" in
  add)
    [[ $# -ge 2 ]] || { echo "Ошибка: команда add требует <branch> <path>" >&2; exit 1; }
    cmd_add "$1" "$2"
    ;;
  remove)
    [[ $# -ge 1 ]] || { echo "Ошибка: команда remove требует <path|branch>" >&2; exit 1; }
    cmd_remove "$@"
    ;;
  list)
    cmd_list "$@"
    ;;
  prune)
    cmd_prune "$@"
    ;;
  sync)
    [[ $# -ge 2 ]] || { echo "Ошибка: команда sync требует <source-path> <target-branch>" >&2; exit 1; }
    cmd_sync "$1" "$2"
    ;;
  help|--help|-h)
    cat <<EOF
Использование: $0 [command] [options]

Команды:
  add <branch> <path>                 Создать новый worktree для ветки
  remove <path|branch>                Удалить worktree и опционально ветку
  list [--verbose|--json]             Показать список всех worktrees
  prune                               Удалить орфанованные записи
  sync <source> <target-branch>       Синхронизировать изменения между worktrees

Опции:
  --dry-run                           Показать что будет сделано без изменений
  --force                             Пропустить проверку изменений при удалении
  -h, --help                          Эта справка

Примеры:
  $0 add feature-x ./worktrees/feature-x
  $0 list --verbose
  $0 remove feature-x
  $0 prune
EOF
    ;;
  *)
    echo "Ошибка: неизвестная команда '$CMD'" >&2
    echo "Доступные команды: add, remove, list, prune, sync" >&2
    exit 1
    ;;
esac
