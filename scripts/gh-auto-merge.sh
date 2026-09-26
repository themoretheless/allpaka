#!/usr/bin/env bash
# scripts/gh-auto-merge.sh — слить PR, только когда выполнены все условия.
#
# Использование:
#   scripts/gh-auto-merge.sh [ОПЦИИ] <НОМЕР_PR>
#
#   # Проверить условия, ничего не меняя
#   scripts/gh-auto-merge.sh --dry-run 123
#
#   # Слить squash'ем и удалить ветку
#   scripts/gh-auto-merge.sh --squash --delete-branch 123
#
#   # Ждать прохождения check'ов до 45 минут
#   scripts/gh-auto-merge.sh --squash --timeout=45m 123
#
# Выходные коды:
#   0 — слит (или dry-run прошёл успешно)
#   1 — условие не выполнено (конфликт, упавшие check'и, PR закрыт)
#   2 — таймаут ожидания check'ов / неверные аргументы
#   3 — нет авторизации gh
#
# Скрипт не обходит защиту веток и не использует --admin.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON=""
for _cand in "$SCRIPT_DIR/scripts-common.sh" "$SCRIPT_DIR/../scripts/scripts-common.sh"; do
  [ -f "$_cand" ] && { COMMON="$_cand"; break; }
done
[ -n "$COMMON" ] || { printf 'не найден scripts-common.sh рядом с %s\n' "$0" >&2; exit 2; }
# shellcheck source=/dev/null
source "$COMMON"

DEFAULT_TIMEOUT=30m
POLL_SECONDS=10

usage() {
  cat <<EOF
Использование: $0 [ОПЦИИ] <НОМЕР_PR>

Сливает PR после проверки: открыт, mergeable, check'и зелёные.

Опции слияния (по умолчанию squash):
  --squash | --merge | --rebase     стратегия слияния
  --delete-branch                   удалить head-ветку после слияния
  --auto                            включить авто-сливание GitHub (не сливать сейчас)

Прочее:
  --timeout=DURATION   сколько ждать зелёных check'ов (default: $DEFAULT_TIMEOUT)
  --interval=SECONDS   период опроса check'ов (default: $POLL_SECONDS)
  --repo=owner/name    репозиторий вместо определённого из origin
  --dry-run            показать решение и команду gh, ничего не меняя
  -h, --help           эта справка

Выходные коды: 0 — слит/ok, 1 — условие не выполнено, 2 — таймаут/аргументы, 3 — нет gh auth

Границы: только чтение до момента слияния; --admin не используется; защита веток не обходится.
EOF
}

DRY_RUN=false
MERGE_FLAG=--squash
EXTRA_FLAGS=()
TIMEOUT=$DEFAULT_TIMEOUT
REPO_OVERRIDE=""
PR_NUM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)       DRY_RUN=true; shift ;;
    --squash)        MERGE_FLAG=--squash; shift ;;
    --merge)         MERGE_FLAG=--merge; shift ;;
    --rebase)        MERGE_FLAG=--rebase; shift ;;
    --delete-branch) EXTRA_FLAGS+=(--delete-branch); shift ;;
    --auto)          EXTRA_FLAGS+=(--auto); shift ;;
    --timeout=*)     TIMEOUT=${1#*=}; shift ;;
    --interval=*)    POLL_SECONDS=${1#*=}; shift ;;
    --repo=*)        REPO_OVERRIDE=${1#*=}; shift ;;
    -h|--help)       usage; exit 0 ;;
    --*)             printf 'неизвестная опция: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$PR_NUM" ]; then
        printf 'лишний аргумент: %s\n' "$1" >&2; exit 2
      fi
      case "$1" in (*[!0-9]*|"") printf 'номер PR должен быть целым: %s\n' "$1" >&2; exit 2 ;; esac
      PR_NUM=$1; shift ;;
  esac
done

[ -n "$PR_NUM" ] || { printf 'нужен номер PR\n' >&2; usage >&2; exit 2; }

MAX_WAIT=$(parse_duration "$TIMEOUT") || exit 2
[ "$POLL_SECONDS" -gt 0 ] 2>/dev/null || { printf '--interval должен быть числом > 0\n' >&2; exit 2; }

if [ -n "$REPO_OVERRIDE" ]; then
  REPO=$REPO_OVERRIDE
else
  REPO=$(detect_repo)
fi
[ -n "$REPO" ] || { printf 'не удалось определить owner/name из origin; передайте --repo\n' >&2; exit 2; }
is_owner_name "$REPO" || { printf 'репозиторий должен быть вида owner/name, получено: %s\n' "$REPO" >&2; exit 2; }

gh auth status >/dev/null 2>&1 || {
  log_error "gh не авторизован"
  log_info  "выполните: gh auth login  (или задайте GH_TOKEN)"
  exit 3
}

fetch_pr() {
  gh pr view "$PR_NUM" --repo "$REPO" \
    --json number,title,state,mergeable,mergeStateStatus,isDraft,reviewDecision,headRefName,baseRefName,statusCheckRollup
}

log_action AUTO-MERGE "PR #$PR_NUM в $REPO"

PR_JSON=$(fetch_pr) || { log_error "не удалось получить PR #$PR_NUM"; exit 1; }

STATE=$(json_get "$PR_JSON" '.state' 'unknown')
TITLE=$(json_get "$PR_JSON" '.title' '(без названия)')
DRAFT=$(json_get "$PR_JSON" '.isDraft' 'false')
REVIEW=$(json_get "$PR_JSON" '.reviewDecision' 'REVIEW_NOT_REQUESTED')
HEAD_REF=$(json_get "$PR_JSON" '.headRefName' '?')
BASE_REF=$(json_get "$PR_JSON" '.baseRefName' '?')
MERGE_STATE=$(json_get "$PR_JSON" '.mergeStateStatus' 'UNKNOWN')

printf '\n=== PR #%s ===\n' "$PR_NUM"
printf '  заголовок : %s\n' "$TITLE"
printf '  состояние : %s%s\n' "$STATE" "$([ "$DRAFT" = true ] && printf ' (черновик)')"
printf '  ветка     : %s -> %s\n' "$HEAD_REF" "$BASE_REF"
printf '  mergeState: %s\n' "$MERGE_STATE"
printf '  reviews   : %s\n' "$REVIEW"

# mergeable у gh — булево; false означает конфликт или неприменимость.
if [ "$STATE" != OPEN ]; then
  log_error "PR не открыт (state=$STATE) — сливать нечего"
  exit 1
fi

if [ "$DRAFT" = true ]; then
  log_error "PR в черновике — сначала переведите его в готовый"
  exit 1
fi

if ! pr_is_mergeable "$PR_JSON"; then
  log_error "PR не mergeable (mergeStateStatus=$MERGE_STATE) — вероятны конфликты"
  log_info  "не решайте это слиянием: обновите ветку и разрешите конфликты вручную"
  exit 1
fi

rollup=$(pr_checks_rollup "$PR_JSON")
printf '  check-и     : %s\n' "$rollup"

if [ "$rollup" = failure ]; then
  printf '\n=== Неуспешные check-и ===\n'
  printf '%s' "$PR_JSON" | jq -r '
    .statusCheckRollup[]?
    | select(((.conclusion // "") | ascii_downcase) as $c
             | $c == "failure" or $c == "timed_out" or $c == "cancelled" or $c == "action_required")
    | "  ✗ \(.name // .context // "?")  (\(.detailsUrl // "без ссылки"))"'
  log_error "есть неуспешные check'и — слияние отменено"
  exit 1
fi

if [ "$rollup" = pending ]; then
  log_info "check'и ещё идут; ожидание до ${TIMEOUT}"
  waited=0
  while [ "$waited" -lt "$MAX_WAIT" ]; do
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
    PR_JSON=$(fetch_pr) || { log_error "запрос статуса PR упал во время ожидания"; exit 1; }
    rollup=$(pr_checks_rollup "$PR_JSON")
    printf '  [%4ss / %ss] check-и: %s\n' "$waited" "$MAX_WAIT" "$rollup"
    case "$rollup" in
      success|none) break ;;
      failure)
        log_error "check-и упали за время ожидания"
        exit 1
        ;;
    esac
  done
  if [ "$rollup" = pending ]; then
    log_error "таймаут ${TIMEOUT}: check'и так и не завершились"
    exit 2
  fi
fi

[ "$rollup" = none ] && log_warn "у PR нет check'ов — суждение только по mergeable и reviews"

MERGE_CMD=(gh pr merge "$PR_NUM" --repo "$REPO" "$MERGE_FLAG")
[ ${#EXTRA_FLAGS[@]} -gt 0 ] && MERGE_CMD+=("${EXTRA_FLAGS[@]}")

printf '\n=== Решение ===\n'
printf '  условия выполнены, стратегия: %s\n' "${MERGE_FLAG#--}"
printf '  команда: %s\n' "${MERGE_CMD[*]}"

if [ "$DRY_RUN" = true ]; then
  log_info "dry-run: ничего не изменено"
  exit 0
fi

log_action AUTO-MERGE "выполняется ${MERGE_CMD[*]}"
if "${MERGE_CMD[@]}"; then
  log_action AUTO-MERGE "PR #$PR_NUM слит (${MERGE_FLAG#--})"
  printf '\n✓ PR #%s слит\n' "$PR_NUM"
  exit 0
else
  log_error "gh pr merge отказал — состояние PR не изменено скриптом"
  exit 1
fi
