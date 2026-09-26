#!/usr/bin/env bash
# scripts/git-worktree-manager.sh — жизнь worktree: создать, посмотреть, удалить, прибрать.
#
# Использование:
#   scripts/git-worktree-manager.sh <КОМАНДА> [АРГУМЕНТЫ]
#
#   add <ветка> <путь> [--detach]     создать worktree (ветку заведёт, если её нет)
#   list [--json|--verbose]           показать все worktree этого репозитория
#   remove <путь|ветка> [--force]     удалить worktree; ветку НЕ трогает
#   prune [--dry-run]                 убрать записи о несуществующих путях
#   report [--path ГЛОБ]              сводка по незакоммиченному в каждом worktree
#
# Примеры:
#   scripts/git-worktree-manager.sh add feature-x .qoder-worktrees/feature-x
#   scripts/git-worktree-manager.sh list
#   scripts/git-worktree-manager.sh remove .qoder-worktrees/feature-x
#
# Выходные коды: 0 — ok, 1 — отказ по причине безопасности, 2 — неверные аргументы.
#
# Только remove/prune меняют состояние, и оба показывают что делают. Ни reset --hard,
# ни clean, ни удаления веток здесь нет по построению.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON=""
for _cand in "$SCRIPT_DIR/scripts-common.sh" "$SCRIPT_DIR/../scripts/scripts-common.sh"; do
  [ -f "$_cand" ] && { COMMON="$_cand"; break; }
done
[ -n "$COMMON" ] || { printf 'не найден scripts-common.sh рядом с %s\n' "$0" >&2; exit 2; }
# shellcheck source=/dev/null
source "$COMMON"

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

git_in_repo() { git "$@"; }

# Канонический путь без симлинков: git worktree list отдаёт /private/tmp/..., а
# pwd — /tmp/..., и без этого сверка путей всегда расходится.
phys_path() {
  local t=$1
  if [ -d "$t" ]; then (cd -- "$t" && pwd -P); return 0; fi
  local d=${t%/*} b=${t##*/}
  [ "$d" = "$t" ] && d=.
  if [ -d "$d" ]; then printf '%s/%s\n' "$(cd -- "$d" && pwd -P)" "$b"; return 0; fi
  printf '%s\n' "$t"
}

top_level() {
  local t
  t=$(git_in_repo rev-parse --show-toplevel 2>/dev/null) || return 1
  phys_path "$t"
}

# Вет already checked out в другом worktree? Печатает путь занявшего.
branch_in_use_by() { # <ветка> -> путь|пусто
  local want=$1 p s b
  while IFS=$'\t' read -r p s b; do
    [ "$b" = "$want" ] && { printf '%s\n' "$p"; return 0; }
  done < <(worktree_tsv)
  return 0
}

require_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    printf 'это не git-репозиторий: %s\n' "$(pwd)" >&2; exit 2; }
}

# ---------- add ----------
cmd_add() {
  local branch='' path='' detach=false a
  local -a rest=()
  for a in "$@"; do
    case "$a" in
      --detach) detach=true ;;
      -*) printf 'неизвестная опция add: %s\n' "$a" >&2; return 2 ;;
      *) rest+=("$a") ;;
    esac
  done
  [ ${#rest[@]} -ge 2 ] || { printf 'нужны <ветка> и <путь>\n' >&2; return 2; }
  branch=${rest[0]}; path=${rest[1]}
  [ ${#rest[@]} -eq 2 ] || { printf 'лишний аргумент: %s\n' "${rest[2]}" >&2; return 2; }

  require_repo
  if [ -e "$path" ]; then
    log_error "путь уже существует: $path"
    log_info  "выберите другой путь или удалите worktree командой remove"
    return 1
  fi

  local exists=false
  git show-ref --verify --quiet "refs/heads/$branch" && exists=true

  if [ "$exists" = true ] && [ "$detach" != true ]; then
    local holder
    holder=$(branch_in_use_by "$branch")
    if [ -n "$holder" ]; then
      log_error "ветка $branch уже занята worktree: $holder"
      log_info  "git не позволяет checkout одной ветки в двух местах; возьмите другую ветку или --detach"
      return 1
    fi
  fi

  printf '=== Создание worktree ===\n'
  printf '  ветка   : %s%s\n' "$branch" "$([ "$exists" = true ] && printf ' (существует)' || printf ' (будет создана)')"
  printf '  путь    : %s\n' "$path"
  if [ "$detach" = true ]; then
    printf '  команда : git worktree add --detach %q %q\n' "$path" "$branch"
  elif [ "$exists" = true ]; then
    printf '  команда : git worktree add %q %q\n' "$path" "$branch"
  else
    printf '  команда : git worktree add -b %q %q [текущий HEAD]\n' "$branch" "$path"
  fi

  mkdir -p "$(dirname "$path")"
  local rc=0
  if [ "$detach" = true ]; then
    git_in_repo worktree add --detach "$path" "$branch" || rc=$?
  elif [ "$exists" = true ]; then
    git_in_repo worktree add "$path" "$branch" || rc=$?
  else
    git_in_repo worktree add -b "$branch" "$path" || rc=$?
  fi
  if [ "$rc" != 0 ]; then
    log_error "git worktree add завершился с кодом $rc — worktree не создан"
    rmdir "$(dirname "$path")" 2>/dev/null || true
    return 1
  fi
  log_action WORKTREE-ADD "создан $path (ветка $branch)"
}

# Разбирает `git worktree list --porcelain` в TSV: путь <TAB> sha <TAB> ветка.
# Пустая ветка означает detached; для bare-записи — "(bare)".
worktree_tsv() {
  git_in_repo worktree list --porcelain | awk '
    BEGIN { p = ""; s = ""; b = "" }
    function flush() { if (p != "") printf "%s\t%s\t%s\n", p, s, b }
    /^worktree[ ]/ { flush(); p = substr($0, 10); s = ""; b = "" }
    /^HEAD[ ]/     { s = substr($0, 6, 40) }
    /^branch[ ]/   { b = substr($0, 8); sub(/^refs\/heads\//, "", b) }
    /^detached/    { b = "" }
    /^bare/        { b = "(bare)" }
    END { flush() }'
}

# ---------- list ----------
cmd_list() {
  require_repo
  local mode=${1:-}
  local tsv
  tsv=$(worktree_tsv)

  case "$mode" in
    --json)
      if [ -z "$tsv" ]; then printf '[]\n'; return 0; fi
      local p s b
      while IFS=$'\t' read -r p s b; do
        [ -n "$p" ] || continue
        jq -cn --arg path "$p" --arg sha "$s" --arg branch "$b" \
          '{path: $path, sha: $sha} + (if $branch == "" then {detached: true} else {branch: $branch} end)'
      done <<<"$tsv" | jq -s '.'
      ;;
    --verbose|"")
      printf '=== Worktree (%s) ===\n' "$(basename "$(git_in_repo rev-parse --show-toplevel)")"
      printf '  %-44s  %-18s  %s\n' 'ПУТЬ' 'ВЕТКА' 'SHA'
      local p s b shown=0
      while IFS=$'\t' read -r p s b; do
        [ -n "$p" ] || continue
        shown=$((shown + 1))
        printf '  %-44s  %-18s  %s\n' "$p" "${b:-(detached)}" "${s:0:10}"
      done <<<"$tsv"
      printf '\n  итого: %s\n' "$shown"
      ;;
    *) printf 'неизвестная опция list: %s\n' "$mode" >&2; return 2 ;;
  esac
}

# ---------- remove ----------
cmd_remove() {
  require_repo
  local target='' force=false a
  local -a rest=()
  for a in "$@"; do
    case "$a" in
      --force) force=true ;;
      -*) printf 'неизвестная опция remove: %s\n' "$a" >&2; return 2 ;;
      *) rest+=("$a") ;;
    esac
  done
  [ ${#rest[@]} -eq 1 ] || { printf 'нужен ровно один <путь>\n' >&2; return 2; }
  target=${rest[0]}

  # Путь должен быть именно worktree из списка этого репозитория.
  local list abs
  list=$(worktree_tsv | cut -f1)
  abs=$(phys_path "$target")
  if ! printf '%s\n' "$list" | grep -qxF "$abs"; then
    log_error "'$target' нет в списке worktree — удалять нечего"
    printf 'известные пути:\n' >&2
    printf '  %s\n' $list >&2
    return 1
  fi

  # Основной worktree не удаляем.
  if [ "$abs" = "$(top_level)" ]; then
    log_error 'нельзя удалить основной worktree репозитория'
    return 1
  fi

  local dirty
  dirty=$(git_in_repo -C "$abs" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  if [ "$dirty" != 0 ] && [ "$force" != true ]; then
    log_error "в $abs есть незакоммиченные изменения ($dirty записей) — удаление отменено"
    git_in_repo -C "$abs" status --short | sed 's/^/    /' >&2
    log_info 'сохраните изменения (commit) или передайте --force, если они не нужны'
    return 1
  fi

  printf '=== Удаление worktree ===\n'
  printf '  путь    : %s\n' "$abs"
  printf '  ветка   : %s (остаётся в репозитории)\n' "$(git_in_repo -C "$abs" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  printf '  команда : git worktree remove%s %q\n' "$([ "$force" = true ] && printf ' --force' || true)" "$abs"

  if [ "$force" = true ]; then
    git_in_repo worktree remove --force "$abs"
  else
    git_in_repo worktree remove "$abs"
  fi
  log_action WORKTREE-REMOVE "удалён $abs"
  log_info 'локальная ветка сохранена; удалите сами, если она больше не нужна'
}

# ---------- prune ----------
cmd_prune() {
  require_repo
  local dry=false
  case "${1:-}" in --dry-run) dry=true ;; '') ;; *) printf 'неизвестная опция prune: %s\n' "$1" >&2; return 2 ;; esac

  if [ "$dry" = true ]; then
    printf '=== prune (только показать) ===\n'
    local out
    out=$(git_in_repo worktree prune -v --dry-run 2>&1 || true)
    if [ -z "$out" ]; then printf '  устаревших записей нет\n'; else printf '%s\n' "$out" | sed 's/^/  /'; fi
    return 0
  fi

  printf '=== prune ===\n'
  git_in_repo worktree prune -v | sed 's/^/  /' || true
  log_action WORKTREE-PRUNE "устаревшие записи убраны"
}

# ---------- report ----------
cmd_report() {
  require_repo
  local globs=() a
  while [ $# -gt 0 ]; do
    case "$1" in
      --path) shift; globs+=("${1:-}"); shift ;;
      *) printf 'неизвестная опция report: %s\n' "$1" >&2; return 2 ;;
    esac
  done

  local script=''
  for cand in "$SCRIPT_DIR/git-uncommitted-report.sh" "$SCRIPT_DIR/../scripts/git-uncommitted-report.sh"; do
    [ -f "$cand" ] && { script=$cand; break; }
  done
  [ -n "$script" ] || { log_error 'git-uncommitted-report.sh не найден — report недоступен'; return 1; }

  local p
  for p in $(worktree_tsv | cut -f1); do
    printf '\n########## %s\n' "$p"
    bash "$script" -C "$p" || printf '  (не удалось получить отчёт)\n'
  done
}

# ---------- main ----------
CMD=${1:-help}
[ $# -gt 0 ] && shift

case "$CMD" in
  add)    cmd_add "$@" ;;
  list)   cmd_list "$@" ;;
  remove) cmd_remove "$@" ;;
  prune)  cmd_prune "$@" ;;
  report) cmd_report "$@" ;;
  help|--help|-h) usage ;;
  *) printf 'неизвестная команда: %s\n' "$CMD" >&2
     printf 'доступны: add, list, remove, prune, report\n' >&2; exit 2 ;;
esac
