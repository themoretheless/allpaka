#!/usr/bin/env bash
# scripts/gh-pr-status-watch.sh — показать check-и PR; по одному PR — следить до завершения.
#
# Использование:
#   scripts/gh-pr-status-watch.sh [ОПЦИИ] <НОМЕР_PR>...
#
#   scripts/gh-pr-status-watch.sh 123              # следить, пока check-и не завершатся
#   scripts/gh-pr-status-watch.sh --once 123 124   # один опрос нескольких PR
#   scripts/gh-pr-status-watch.sh --json 123       # машиночитаемый вывод
#
# Выходные коды:
#   0 — все завершённые check-и успешны (или check-ов нет)
#   1 — есть неуспешный check
#   2 — таймаут ожидания
#   3 — нет авторизации gh
#   4 — неверные аргументы

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON=""
for _cand in "$SCRIPT_DIR/scripts-common.sh" "$SCRIPT_DIR/../scripts/scripts-common.sh"; do
  [ -f "$_cand" ] && { COMMON="$_cand"; break; }
done
[ -n "$COMMON" ] || { printf 'не найден scripts-common.sh рядом с %s\n' "$0" >&2; exit 4; }
# shellcheck source=/dev/null
source "$COMMON"

DEFAULT_INTERVAL=10s
MODE=watch          # watch | once
FORMAT=table        # table | json
TIMEOUT=0           # 0 = без ограничения
REPO_OVERRIDE=""
PRS=()

usage() {
  cat <<EOF
Использование: $0 [ОПЦИИ] <НОМЕР_PR>...

Опрос check-ов pull request'ов.

Опции:
  --once               один опрос и выход (без слежения)
  --json               вывод JSON вместо таблицы (по объекту на PR, NDJSON;
                       для массива: ... --json | jq -s .)
  --interval=DURATION  период опроса при слежении (default: $DEFAULT_INTERVAL)
  --timeout=DURATION   прекратить слежение после этого времени (default: без лимита)
  --repo=owner/name    репозиторий вместо определённого из origin
  -h, --help           эта справка

Выходные коды: 0 — зелёный, 1 — есть упавший check, 2 — таймаут, 3 — нет gh auth, 4 — аргументы

Только чтение: скрипт ничего не меняет в PR и в репозитории.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --once)        MODE=once; shift ;;
    --json)        FORMAT=json; shift ;;
    --table)       FORMAT=table; shift ;;
    --interval=*)  DEFAULT_INTERVAL=${1#*=}; shift ;;
    --timeout=*)   TIMEOUT=$(parse_duration "${1#*=}") || exit 4; shift ;;
    --repo=*)      REPO_OVERRIDE=${1#*=}; shift ;;
    -h|--help)     usage; exit 0 ;;
    --*)           printf 'неизвестная опция: %s\n' "$1" >&2; usage >&2; exit 4 ;;
    *)
      case "$1" in (*[!0-9]*|"") printf 'номер PR должен быть целым: %s\n' "$1" >&2; exit 4 ;; esac
      PRS+=("$1"); shift ;;
  esac
done

[ ${#PRS[@]} -gt 0 ] || { printf 'нужен хотя бы один номер PR\n' >&2; usage >&2; exit 4; }

INTERVAL=$(parse_duration "$DEFAULT_INTERVAL") || exit 4
[ "$INTERVAL" -gt 0 ] 2>/dev/null || { printf '--interval должен быть > 0\n' >&2; exit 4; }

if [ -n "$REPO_OVERRIDE" ]; then REPO=$REPO_OVERRIDE; else REPO=$(detect_repo); fi
[ -n "$REPO" ] || { printf 'не удалось определить owner/name; передайте --repo\n' >&2; exit 4; }
is_owner_name "$REPO" || { printf 'репозиторий должен быть вида owner/name, получено: %s\n' "$REPO" >&2; exit 4; }

gh auth status >/dev/null 2>&1 || {
  log_error "gh не авторизован — выполните gh auth login или задайте GH_TOKEN"
  exit 3
}

fetch_pr() {
  gh pr view "$1" --repo "$REPO" \
    --json number,title,state,isDraft,mergeable,reviewDecision,headRefName,statusCheckRollup
}

# Одна строка на check: name <TAB> status <TAB> conclusion
checks_rows() {
  printf '%s' "$1" | jq -r '
    (.statusCheckRollup // [])[]
    | [ (.name // .context // "?"),
        ((.status // "?") | ascii_downcase),
        ((.conclusion // "-") | ascii_downcase) ]
    | @tsv' 2>/dev/null || true
}

summary_of() {  # <pr_json> -> "success|failure|pending|none"
  pr_checks_rollup "$1"
}

render_table() {
  local pr_json rows s
  pr_json=$1
  printf '\n── PR #%s  %s  [%s] ──\n' \
    "$(json_get "$pr_json" '.number')" \
    "$(json_get "$pr_json" '.title' '')" \
    "$(summary_of "$pr_json")"

  rows=$(checks_rows "$pr_json")
  if [ -z "$rows" ]; then
    printf '  check-ов нет\n'
    return 0
  fi
  printf '  %-38s  %-12s  %-10s\n' 'CHECK' 'STATUS' 'RESULT'
  while IFS=$'\t' read -r name status conclusion; do
    [ -n "$name" ] || continue
    [ "$conclusion" = "-" ] && conclusion=""
    res=$conclusion
    [ -n "$res" ] || res='…'
    mark=' '
    case "$conclusion" in
      failure|timed_out|cancelled) mark='✗' ;;
      success|skipped|neutral)     mark='✓' ;;
    esac
    printf '  %s %-36s  %-12s  %-10s\n' "$mark" "${name:0:36}" "$status" "$res"
  done <<<"$rows"

  s=$(summary_of "$pr_json")
  [ "$s" = failure ] && printf '  ⚠ есть неуспешные check-и\n'
  return 0
}

poll_all() {
  local pr pr_json rc=0
  for pr in "${PRS[@]}"; do
    if ! pr_json=$(fetch_pr "$pr"); then
      log_error "не удалось получить PR #$pr"
      printf '\n── PR #%s — недоступен ──\n' "$pr"
      rc=1
      continue
    fi
    case "$FORMAT" in
      json) printf '%s\n' "$pr_json" ;;
      *)
        printf '\n%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$REPO"
        render_table "$pr_json" ;;
    esac
    [ "$(summary_of "$pr_json")" = failure ] && rc=1
  done
  return "$rc"
}

# ---------- main ----------

run_once() {  # опросить и вернуть код без срабатывания set -e
  local rc=0
  poll_all || rc=$?
  return "$rc"
}

if [ "$FORMAT" = json ]; then
  run_once
  exit $?
fi

if [ ${#PRS[@]} -eq 1 ] && [ "$MODE" = watch ]; then
  log_info "слежу за PR #${PRS[0]} (интервал ${INTERVAL}s); Ctrl-C — выход"
  trap 'printf "\n"; log_info "остановлено пользователем"; exit 0' INT TERM
  waited=0
  while true; do
    clear 2>/dev/null || printf '\n'
    run_once; rc=$?
    if ! pr_json=$(fetch_pr "${PRS[0]}"); then
      log_error "запрос статуса PR упал"; exit 1
    fi
    case "$(summary_of "$pr_json")" in
      success|none|failure)
        [ "$rc" -ne 0 ] && exit 1
        printf '\n✓ check-и завершены\n'; exit 0 ;;
    esac
    if [ "$TIMEOUT" -gt 0 ] && [ "$waited" -ge "$TIMEOUT" ]; then
      log_error "таймаут ${TIMEOUT}s — check-и не завершились"
      exit 2
    fi
    printf '\n(следующий опрос через %ss; пройдено %ss)\n' "$INTERVAL" "$waited"
    sleep "$INTERVAL"
    waited=$((waited + INTERVAL))
  done
else
  run_once
  exit $?
fi
