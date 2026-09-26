#!/usr/bin/env bash
# scripts/scripts-common.sh — общие функции для скриптов автоматизации GitHub-потока.
#
# Подключать так (работает и из scripts/, и из plugins/<name>/bin/ после materialize):
#
#   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
#   for _cand in "$SCRIPT_DIR/scripts-common.sh" "$SCRIPT_DIR/../scripts/scripts-common.sh"; do
#     [ -f "$_cand" ] && { source "$_cand"; break; }
#   done
#
# Все функции только читают состояние репозитория и GitHub; ничего не меняют.
#
# Библиотека намеренно НЕ задаёт set -e/-u: она подключается через source и не должна
# менять опции вызывающего шелла. Каждый скрипт ставит `set -euo pipefail` сам.

# Каталоги логов и отчётов. Путь относительный — от каталога запуска.
LOG_DIR="${LOG_DIR:-.qoder/logs}"
REPORTS_DIR="${REPORTS_DIR:-.qoder/reports}"

log_debug() { printf '[DEBUG] %s\n' "$*" >&2; }
log_info()  { printf '[INFO]  %s\n' "$*" >&2; }
log_warn()  { printf '[WARN]  %s\n' "$*" >&2; }
log_error() { printf '[ERROR] %s\n' "$*" >&2; }

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Пишет действие и в stdout, и в суточный лог. Каталог создаёт по мере надобности.
log_action() {
  local action="$1"; shift
  local line
  line=$(printf '[%s] %s: %s' "$(timestamp)" "$action" "$*")
  { mkdir -p "$LOG_DIR" && printf '%s\n' "$line" >>"$LOG_DIR/$(date +%Y%m%d).log"; } || true
  printf '%s\n' "$line"
}

# Разбирает длительность вида 30s / 5m / 1h в секунды. Без суффикса — секунды.
parse_duration() {
  local dur="$1" num unit

  if [[ "$dur" =~ ^([0-9]+)([smh]?)$ ]]; then
    num="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
  else
    printf 'непонятная длительность: %s\n' "$dur" >&2
    return 1
  fi

  case "$unit" in
    s) printf '%s\n' "$num" ;;
    m) printf '%s\n' "$((num * 60))" ;;
    h) printf '%s\n' "$((num * 3600))" ;;
    *) printf '%s\n' "$num" ;;
  esac
}

# Достаёт поле из JSON по jq-фильтру; при пустом или невозможном значении — default.
# json_get <json> <jq-фильтр> [default]
json_get() {
  local json="$1" filter="$2" default="${3:-}"
  local out
  out=$(printf '%s' "$json" | jq -r "$filter // empty" 2>/dev/null) || out=""
  if [ -z "$out" ]; then printf '%s\n' "$default"; else printf '%s\n' "$out"; fi
}

# Сводит массив statusCheckRollup PR к одному состоянию: none|pending|failure|success.
# У записи два уровня — status (QUEUED/IN_PROGRESS/COMPLETED) и conclusion (SUCCESS/
# FAILURE, пустой у незавершённых), — поэтому нужна свёртка по массиву, а не чтение
# .statusCheckRollup.conclusion: у массива такого поля нет.
pr_checks_rollup() {
  local pr_json="$1"
  printf '%s' "$pr_json" | jq -r '
    (.statusCheckRollup // []) as $c
    | if ($c | length) == 0 then "none"
      else
        (($c | map(((.conclusion // "") + "|" + (.status // "")) | ascii_downcase)) as $s
         | if   ($s | any(test("failure|timed_out|cancelled|action_required"))) then "failure"
           elif ($s | any(test("queued|in_progress|pending|waiting")))          then "pending"
           elif ($s | all(test("success|skipped|neutral")))                     then "success"
           else "pending"
           end)
      end' 2>/dev/null || printf 'none\n'
}

# true, если PR mergeable. gh отдаёт булево, а не строку "TRUE".
pr_is_mergeable() {
  local pr_json="$1"
  printf '%s' "$pr_json" | jq -r '.mergeable // false' 2>/dev/null | grep -qx true
}

# Показывает, что именно собирается сделать скрипт, перед изменяющей операцией.
show_preview() {
  local action="$1"; shift
  printf '\n%s\nACTION: %s\n%s\n%s\n\n' \
    '==================================================' \
    "$action" \
    "$*" \
    '=================================================='
}

# Пауза при приближении исчерпанного лимита GitHub API. Молча пропускает, если
# статус недоступен (нет авторизации — тогда и команда упадёт сама собой).
wait_for_rate_limit() {
  local max_wait="${1:-60}" waited=0 remaining

  while true; do
    remaining=$(gh api rate_limit --jq '.resources.core.remaining' 2>/dev/null) || return 0
    [[ "$remaining" =~ ^[0-9]+$ ]] || return 0
    [ "$remaining" -gt 5 ] && return 0
    if [ "$waited" -ge "$max_wait" ]; then
      log_error "Лимит GitHub API не восстановился за ${max_wait}s"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# Проверяет форму owner/name. gh принимает только её; значение без владельца или с
# лишними сегментами уходит в API и даёт непонятную ошибку вместо валидации.
is_owner_name() {
  case "${1-}" in
    */*/*) return 1 ;;
    */*) [ -n "${1%%/*}" ] && [ -n "${1##*/}" ] ;;
    *) return 1 ;;
  esac
}

# Репозиторий owner/name из URL remote'а (по умолчанию — origin этого дерева).
# Пустая строка, если remote не GitHub: угадывать owner/name нельзя.
detect_repo() {
  local url owner tail base
  url="${1-$(git remote get-url origin 2>/dev/null || true)}"
  [ -n "$url" ] || return 0
  case "$url" in *github.com*) ;; *) return 0 ;; esac
  url=${url%.git}
  url=${url%/}
  tail=${url##*[/:]}            # имя репозитория — последний сегмент
  base=${url%/[!/]*}            # всё до последнего '/name'
  base=${base%/}
  owner=${base##*[/:]}          # владелец — последний из оставшихся
  [ -n "$owner" ] && [ -n "$tail" ] && printf '%s/%s\n' "$owner" "$tail"
}
