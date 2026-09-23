#!/usr/bin/env bash
# git-uncommitted-report.sh — отчёт по незакоммиченным файлам: сколько строк (+/-) в каких файлах.
#
# Использование:
#   scripts/git-uncommitted-report.sh [-C <путь-к-репо>] [-o <файл-отчёта.md>]
#
# Источники данных (только чтение, рабочее дерево и индекс не меняются):
#   git status --porcelain=v1 --untracked-files=all  — список файлов и коды статуса
#   git diff --cached --numstat                      — застейдженные строки (+/-)
#   git diff --numstat                               — незастейдженные строки (+/-)
#   wc -l                                            — число строк в новых (untracked) файлах
#
# Особенности:
#   * --no-renames: переименование считается как удаление старого + добавление нового;
#   * бинарные файлы не считаются построчно и помечаются кодом из status;
#   * числа в отчёте — это numstat-дельты, а не размер файла на диске;
#   * для untracked-файлов «+» = число строк в файле, «-» = 0.

set -euo pipefail

repo="."
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -C) repo="${2:?нужен путь к репозиторию}"; shift 2 ;;
    -o) out="${2:?нужен путь к файлу отчёта}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'неизвестный аргумент: %s\n' "$1" >&2; exit 2 ;;
  esac
done

git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  printf 'не git-репозиторий: %s\n' "$repo" >&2
  exit 1
}
root=$(git -C "$repo" rev-parse --show-toplevel)
cd "$root"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

branch=$(git rev-parse --abbrev-ref HEAD)
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
  ab=$(git rev-list --left-right --count "$upstream...HEAD")
  position="upstream: \`$upstream\` (behind ${ab%%$'\t'*}, ahead ${ab##*$'\t'})"
else
  position="upstream: нет (локальная ветка без tracking)"
fi
stashes=$(git stash list | wc -l | tr -d ' ')

git status --porcelain=v1 --untracked-files=all > "$tmp/status"
file_count=$(wc -l < "$tmp/status" | tr -d ' ')
: > "$tmp/rows.tsv"

while IFS= read -r line; do
  [ -n "$line" ] || continue
  code=${line:0:2}
  path=${line:3}
  case "$path" in *" -> "*) path=${path##* -> } ;; esac

  st_add=0; st_del=0; un_add=0; un_del=0

  if [ "$code" = "??" ]; then
    if [ ! -e "$path" ]; then
      un_add=0
    elif [ ! -s "$path" ]; then
      un_add=0
    elif LC_ALL=C grep -Iq . "$path" 2>/dev/null; then
      un_add=$(wc -l < "$path" | tr -d ' ')
    else
      un_add=0   # бинарный новый файл: построчно не считаем
    fi
  else
    while IFS=$'\t' read -r a d; do
      [ -n "${a:-}" ] || continue
      if [ "$a" = "-" ]; then continue; fi
      st_add=$((st_add + a)); st_del=$((st_del + d))
    done < <(git diff --cached --no-renames --numstat -- "$path")
    while IFS=$'\t' read -r a d; do
      [ -n "${a:-}" ] || continue
      if [ "$a" = "-" ]; then continue; fi
      un_add=$((un_add + a)); un_del=$((un_del + d))
    done < <(git diff --no-renames --numstat -- "$path")
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$((st_add + un_add))" "$((st_del + un_del))" \
    "$st_add" "$st_del" "$un_add" "$un_del" "$code" "$path" >> "$tmp/rows.tsv"
done < "$tmp/status"

format() {
  printf '# Незакоммиченные файлы — %s\n\n' "$root"
  printf -- '- ветка: `%s`\n- %s\n- stash-записей: %s\n- файлов с изменениями: %s\n\n' \
    "$branch" "$position" "$stashes" "$file_count"

  if [ ! -s "$tmp/rows.tsv" ]; then
    printf 'Незакоммиченных файлов нет: рабочее дерево и индекс чистые.\n'
    return
  fi

  printf '| файл | статус | index + | index − | worktree + | worktree − | всего + | всего − |\n'
  printf '|---|:--:|---:|---:|---:|---:|---:|---:|\n'
  sort -t$'\t' -k1,1nr "$tmp/rows.tsv" | while IFS=$'\t' read -r ta td sa sd ua ud code path; do
    printf '| `%s` | %s | %s | %s | %s | %s | **%s** | **%s** |\n' \
      "$path" "$code" "$sa" "$sd" "$ua" "$ud" "$ta" "$td"
  done
  printf '\n'

  awk -F'\t' '{ta+=$1; td+=$2; sa+=$3; sd+=$4; ua+=$5; ud+=$6}
    END {printf "**Итого: +%d / −%d** (index: +%d/−%d, worktree: +%d/−%d)\n", ta, td, sa, sd, ua, ud}' \
    "$tmp/rows.tsv"
  printf '\nЛегенда статуса: `M` изменён, `A` добавлен, `D` удалён, `R` переименован, `??` новый, `UU`/`AA`/`DD` конфликт.\n'
  printf 'Числа — дельты строк из `git diff --numstat` (`--no-renames`); для `??` «всего +» = число строк в файле.\n'
}

if [ -n "$out" ]; then
  format | tee "$out"
else
  format
fi
