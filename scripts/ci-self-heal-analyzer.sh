#!/usr/bin/env bash
# scripts/ci-self-heal-analyzer.sh — разбор упавшего запуска CI и подсказки, что чинить.
#
# Использование:
#   scripts/ci-self-heal-analyzer.sh [ОПЦИИ] <RUN_ID>
#
#   # Таблица находок по команде-ремоуту из origin
#   scripts/ci-self-heal-analyzer.sh 1234567890
#
#   # Markdown-отчёт в файл, без локальных проверок
#   scripts/ci-self-heal-analyzer.sh --output=md --save 1234567890
#
#   # Черновик issue (тело пишется в файл; gh issue create запускает человек)
#   scripts/ci-self-heal-analyzer.sh --output=issue 1234567890
#
# Выходные коды:
#   0 — анализ завершён, находки есть
#   1 — анализ завершён, паттерны не распознаны (нужна ручная проверка)
#   2 — неверные аргументы
#   3 — нет авторизации gh или запуск недоступен
#
# Границы: скрипт ничего не чинит сам. Он читает JSON запуска, шаги и логи gh,
# классифицирует отказ и печатает команды. Изменяющие команды помечены как
# mutating и выполняются только человеком.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON=""
for _cand in "$SCRIPT_DIR/scripts-common.sh" "$SCRIPT_DIR/../scripts/scripts-common.sh"; do
  [ -f "$_cand" ] && { COMMON="$_cand"; break; }
done
[ -n "$COMMON" ] || { printf 'не найден scripts-common.sh рядом с %s\n' "$0" >&2; exit 2; }
# shellcheck source=/dev/null
source "$COMMON"

MAX_LOG_BYTES=400000
OUTPUT=table
REPO_OVERRIDE=""
FETCH_LOG=true
RUN_LOCAL=false
SAVE=false
RUN_ID=""

usage() {
  cat <<EOF
Использование: $0 [ОПЦИИ] <RUN_ID>

Классифицирует отказ запуска GitHub Actions и предлагает команды. сам ничего не применяет.

Вывод:
  --output=table|md|json|issue   формат отчёта (default: table)
  --save                         дополнительно сохранить отчёт в $REPORTS_DIR
  --repo=owner/name              репозиторий вместо определённого из origin

Источники данных:
  --no-log             не тянуть логи упавших шагов (только имена шагов из JSON)
  --max-log=BYTES      ограничить объём логов (default: $MAX_LOG_BYTES)
  --local              дополнить разбор локальными read-only проверками (rustfmt --check,
                       cargo test того же набора); по умолчанию выключено

Прочее:
  -h, --help           эта справка

Выходные коды: 0 — находки есть, 1 — паттерны не распознаны, 2 — аргументы, 3 — нет доступа к gh

Границы: только чтение. Подсказки по форматированию дают rustfmt по конкретным файлам, а не
cargo fmt по всему workspace: в этом репозитории cargo fmt переформатирует чужие файлы.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --output=*)   OUTPUT="${1#*=}"; shift ;;
    --repo=*)     REPO_OVERRIDE="${1#*=}"; shift ;;
    --max-log=*)  MAX_LOG_BYTES="${1#*=}"; shift ;;
    --no-log)     FETCH_LOG=false; shift ;;
    --local)      RUN_LOCAL=true; shift ;;
    --save)       SAVE=true; shift ;;
    -h|--help)    usage; exit 0 ;;
    --*)          printf 'неизвестная опция: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "$RUN_ID" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        RUN_ID="$1"
      else
        printf 'Ошибка: лишний аргумент: %s (нужен числовой ID запуска)\n' "$1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

[ -n "$RUN_ID" ] || { printf 'Ошибка: нужен ID запуска CI (gh run list --limit 5)\n' >&2; exit 2; }
[[ "$MAX_LOG_BYTES" =~ ^[0-9]+$ ]] || { printf 'Ошибка: --max-log должен быть числом\n' >&2; exit 2; }
case "$OUTPUT" in table|md|json|issue) ;; *) printf 'Ошибка: --output=table|md|json|issue\n' >&2; exit 2 ;; esac

REPO=$REPO_OVERRIDE
[ -n "$REPO" ] || REPO=$(detect_repo)
if [ -z "$REPO" ]; then
  printf 'Не удалось определить owner/name из origin. Укажите --repo=owner/name\n' >&2
  exit 2
fi
case "$REPO" in
  */*) ;;
  *) printf 'Репозиторий должен быть вида owner/name, получено: %s\n' "$REPO" >&2; exit 2 ;;
esac

gh auth status >/dev/null 2>&1 || { log_error 'gh не авторизован — выполните gh auth login'; exit 3; }

log_action "CI-HEAL" "RUN=$RUN_ID REPO=$REPO OUTPUT=$OUTPUT" >&2

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ci-heal.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

RUN_JSON="$TMP_DIR/run.json"
FAILED_STEPS="$TMP_DIR/steps.tsv"
LOG_TXT="$TMP_DIR/log.txt"
FINDINGS="$TMP_DIR/findings.tsv"

: >"$LOG_TXT"
: >"$FINDINGS"

# --- Данные о запуске ---------------------------------------------------------

if ! gh run view "$RUN_ID" --repo "$REPO" \
      --json databaseId,displayTitle,workflowName,headBranch,headSha,event,conclusion,status,url \
      >"$RUN_JSON" 2>"$TMP_DIR/run.err"; then
  log_error "Не удалось получить запуск $RUN_ID: $(tr '\n' ' ' <"$TMP_DIR/run.err")"
  log_info "Проверьте доступ к репозиторию и что ID существует: gh run list --repo $REPO --limit 10"
  exit 3
fi

run_field() { json_get "$(cat "$RUN_JSON")" "$1" "$2"; }

WF_NAME=$(run_field '.workflowName' 'неизвестный workflow')
RUN_TITLE=$(run_field '.displayTitle' '(без заголовка)')
RUN_BRANCH=$(run_field '.headBranch' '?')
RUN_SHA=$(run_field '.headSha' '?')
RUN_STATE=$(run_field '.conclusion' "$(run_field '.status' 'unknown')")
RUN_URL=$(run_field '.url' '')

if [ "$RUN_STATE" != "failure" ]; then
  log_warn "Запуск $RUN_ID завершён как '$RUN_STATE', а не failure — разбор всё равно сделан по упавшим шагам"
fi

# name<TAB>step<TAB>stepConclusion — декартовы строки для каждого упавшего шага.
# Строка на каждый упавший шаг: job<TAB>step<TAB>conclusion.
gh run view "$RUN_ID" --repo "$REPO" --json jobs --jq '
  (.jobs // [])[]
  | select(((.conclusion // "") | ascii_downcase) == "failure")
  | . as $j
  | ($j.steps // [])[]
  | select(((.conclusion // "") | ascii_downcase) | test("failure|timed_out|cancelled"))
  | [$j.name, .name, .conclusion] | @tsv' >"$FAILED_STEPS" 2>/dev/null || : >"$FAILED_STEPS"

if [ ! -s "$FAILED_STEPS" ]; then
  # У завершённого успехом запуска упавших шагов нет — это валидный пустой результат.
  log_info "Упавших шагов не найдено (состояние: $RUN_STATE)"
fi

add_finding() {
  # category<TAB>что найдено<TAB>команда<TAB>уверенность<TAB>побочный эффект
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >>"$FINDINGS"
}

# --- Логи ---------------------------------------------------------------------

if [ "$FETCH_LOG" = true ] && [ -s "$FAILED_STEPS" ]; then
  log_info "Читаю логи упавших шагов..."
  gh run view "$RUN_ID" --repo "$REPO" --log-failed 2>/dev/null \
    | head -c "$MAX_LOG_BYTES" >"$LOG_TXT" || : >"$LOG_TXT"
  if [ ! -s "$LOG_TXT" ]; then
    log_warn "Логи недоступны (возможно, артефакты удалены) — классификация по именам шагов"
  fi
fi

# Действия в Actions печатают строки с префиксом "<job>\t<step>\t| ". Снимаем префикс,
# чтобы регулярки работали по содержимому, а не по его хвосту. awk, а не sed: BSD sed
# не понимает \t, и префикс оставался бы в строке — якоря ^error тогда не срабатывали.
LOG_BODY="$TMP_DIR/log.body"
awk -F'\t' -v OFS='\t' '
  { line = $0
    if (NF > 1) {
      for (i = NF; i >= 1; i--) {
        if (substr($i, 1, 1) == "|") { line = substr($i, 2); sub(/^ /, "", line); break }
      }
    }
    print line }' "$LOG_TXT" >"$LOG_BODY" 2>/dev/null || cp "$LOG_TXT" "$LOG_BODY"

# --- Классификация ------------------------------------------------------------

edition_of() {
  local e
  e=$(grep -m1 -oE 'edition *= *"[0-9]+"' Cargo.toml 2>/dev/null | grep -oE '[0-9]+' || true)
  printf '%s\n' "${e:-2021}"
}

classify_log() {
  local log="$1"

  # rustfmt в CI печатает "Diff in <path> at line N:". Перечисляем ровно эти файлы:
  # cargo fmt по workspace заодно переформатирует файлы вне задачи.
  if grep -qE '^Diff in .* at line ' "$log"; then
    local files e cmd
    files=$(grep -oE '^Diff in [^ ]+ at line ' "$log" | awk '{print $3}' | sort -u | tr '\n' ' ')
    e=$(edition_of)
    cmd="rustfmt --edition $e ${files}"
    add_finding "formatting" "rustfmt хочет переформатировать: $(printf '%s' "$files" | tr -d '\n')" "$cmd" "высокая" "локальные правки"
  fi

  # Известная отказоустойчивость этого репозитория: target-cpu=native из
  # .cargo/config.toml и статический минимум ring.
  if grep -qE 'CAPS_STATIC|MIN_STATIC_FEATURES' "$log"; then
    add_finding "toolchain-cpu-flags" \
      "ring проверяет фичи CPU: build идёт с -C target-cpu=native из .cargo/config.toml" \
      'RUSTFLAGS="" cargo build --workspace' "высокая" "локальная сборка"
  fi

  # Тесты: имена из блоков "---- <name> stdout ----".
  if grep -qE 'test result: FAILED|error: test failed' "$log"; then
    local names
    names=$(grep -oE '^---- [^ ]+ (stdout|stderr) ----' "$log" | awk '{print $2}' | sort -u | head -20 | tr '\n' ' ')
    [ -n "$names" ] || names=$(grep -oE 'test [A-Za-z0-9_:]+ \.\.\. FAILED' "$log" | awk '{print $2}' | sort -u | head -20 | tr '\n' ' ')
    if [ -n "$names" ]; then
      add_finding "test-failure" "упавшие тесты: $names" "cargo test --workspace -- --nocapture" "средняя" "локальный прогон"
    else
      add_finding "test-failure" "тесты упали, имена в логе не найдены" "cargo test --workspace" "низкая" "локальный прогон"
    fi
  fi

  # Компиляция.
  if grep -qE '^error\[E[0-9A-Z]+\]|^error: could not compile' "$log"; then
    local crates codes
    crates=$(grep -oE '^\s*(Compiling|Checking) [A-Za-z0-9_.-]+' "$log" | awk '{print $2}' | sort -u | tail -5 | tr '\n' ' ')
    codes=$(grep -oE '^error\[[A-Z][0-9]+\]' "$log" | sort -u | tr '\n' ' ')
    add_finding "compile-error" "коды ошибок: ${codes:-нет}; последние компилируемые crates: ${crates:-неясно}" \
      "cargo check --workspace --all-targets" "средняя" "локальная проверка"
  fi

  if grep -qE "error: .*clippy|warning: .*declared as .#\[deny|clippy::" "$log"; then
    add_finding "clippy" "clippy нашёл нарушения" "cargo clippy --workspace --all-targets" "средняя" "локальная проверка"
  fi

  # Внешние причины: сеть, реестр, лимиты.
  if grep -qiE 'failed to (fetch|load source|download)|network failure|timed out (during|while)|error sending request|Could not resolve host|403 Forbidden|429 ' "$log"; then
    add_finding "transient-network" "похоже на сетевую/реестровую проблему, а не на изменение в коде" \
      "gh run rerun $RUN_ID --failed" "средняя" "МУТАЦИЯ: перезапуск CI"
  fi

  if grep -qE 'out of memory|Cannot allocate|signal: 9 \(SIGKILL\)|Killed|exited with signal: 9' "$log"; then
    add_finding "resource" "процесс убит по памяти или ресурсам" \
      "cargo test --workspace --jobs 1" "низкая" "локальный прогон"
  fi

  # Отсутствующий локальный ресурс (модели, бинарь RAG) — то, что graceful-skip в тестах.
  if grep -qiE 'No such file or directory.*(\.gguf|models/|rag-mcp|duckdb)|missing (model|binary)' "$log"; then
    add_finding "missing-local-resource" "нужен файл или бинарь, которого нет в раннере" \
      "проверить условный skip теста; для покрытия нужен self-hosted раннер" "средняя" "только чтение"
  fi
}

classify_steps() {
  while IFS=$'\t' read -r job step _; do
    [ -n "$step" ] || continue
    case "$step" in
      *fmt*|*format*)
        grep -q 'formatting' "$FINDINGS" || add_finding "formatting" "упал шаг '$step'" 'rustfmt --check на файлах из лога' "средняя" "локальные правки" ;;
      *clippy*)
        grep -q 'clippy' "$FINDINGS" || add_finding "clippy" "упал шаг '$step'" "cargo clippy --workspace --all-targets" "средняя" "локальная проверка" ;;
      *test*)
        grep -q 'test-failure' "$FINDINGS" || add_finding "test-failure" "упал шаг '$step'" "cargo test --workspace" "низкая" "локальный прогон" ;;
      *"Set up"*|*toolchain*|*cache*)
        add_finding "infra" "упал служебный шаг '$step' (job: $job)" "gh run view $RUN_ID --repo $REPO --log-failed" "низкая" "только чтение" ;;
    esac
  done <"$FAILED_STEPS"
}

classify_log "$LOG_BODY"
classify_steps

if [ ! -s "$FINDINGS" ]; then
  if [ -s "$FAILED_STEPS" ]; then
    log_warn "Паттерны не распознаны — читать лог руками"
  else
    log_info "Упавших шагов нет — классифицировать нечего"
  fi
fi

# --- Локальные read-only дополнения ------------------------------------------

local_checks() {
  command -v cargo >/dev/null 2>&1 || { log_warn "cargo не найден — локальные проверки пропущены"; return; }
  local e out
  e=$(edition_of)
  if command -v rustfmt >/dev/null 2>&1; then
    out=$(git diff --name-only HEAD 2>/dev/null | grep -E '\.rs$' || true)
    if [ -n "$out" ]; then
      if printf '%s\n' "$out" | xargs rustfmt --edition "$e" --check >/dev/null 2>&1; then
        log_info "Локально: изменённые .rs файлы уже отформатированы"
      else
        log_warn "Локально: rustfmt хочет переформатировать изменённые файлы — см. rustfmt --edition $e <файлы>"
      fi
    fi
  fi
}

if [ "$RUN_LOCAL" = true ]; then local_checks; fi

# --- Отчёты -------------------------------------------------------------------

COUNT=$(wc -l <"$FINDINGS" | tr -d ' ')

emit_json() {
  local findings steps
  findings=$(jq -Rn '[inputs | select(length > 0) | split("\t")
    | {category: .[0], detail: .[1], command: .[2], confidence: .[3], side_effects: .[4]}]' "$FINDINGS")
  steps=$(jq -Rn '[inputs | select(length > 0) | split("\t")
    | {job: .[0], step: .[1]}]' "$FAILED_STEPS")
  jq -n --argjson count "$COUNT" --argjson findings "$findings" --argjson steps "$steps" \
    --slurpfile run "$RUN_JSON" \
    '{run: $run[0], failed_steps: $steps, findings: $findings, finding_count: $count}'
}

emit_table() {
  printf '\n## Разбор CI: %s (#%s)\n\n' "$WF_NAME" "$RUN_ID"
  printf -- '- **Запуск**: %s\n' "$RUN_TITLE"
  printf -- '- **Ветка / SHA**: %s / %s\n' "$RUN_BRANCH" "$(printf '%.10s' "$RUN_SHA")"
  printf -- '- **Итог**: %s\n' "$RUN_STATE"
  printf -- '- **Репозиторий**: %s\n' "$REPO"
  if [ -n "$RUN_URL" ]; then printf -- '- **Ссылка**: %s\n' "$RUN_URL"; fi
  printf -- '- **Упавших шагов**: %s\n' "$(wc -l <"$FAILED_STEPS" | tr -d ' ')"

  if [ ! -s "$FINDINGS" ]; then
    printf '\nРаспознанных паттернов нет. Читать лог: `gh run view %s --repo %s --log-failed`\n' "$RUN_ID" "$REPO"
    return
  fi

  printf '\n| Категория | Что найдено | Команда | Уверенность | Побочный эффект |\n'
  printf '|---|---|---|---|---|\n'
  while IFS=$'\t' read -r category detail cmd conf side; do
    [ -n "$category" ] || continue
    printf '| %s | %s | `%s` | %s | %s |\n' "$category" "$detail" "$cmd" "$conf" "$side"
  done <"$FINDINGS"

  printf '\nСкрипт ничего не применяет. Мутирующие команды (перезапуск CI, push, правки файлов) выполняйте осознанно.\n'
}

emit_md() {
  {
    emit_table
    printf '\n### Упавшие шаги\n\n'
    if [ -s "$FAILED_STEPS" ]; then
      while IFS=$'\t' read -r job step _; do printf -- '- %s → %s\n' "$job" "$step"; done <"$FAILED_STEPS"
    else
      printf -- '- нет данных\n'
    fi
    if [ -s "$LOG_TXT" ]; then
      printf '\n### Фрагменты лога\n\n```text\n'
      grep -nE '^(error|Diff in|failures:|---- |test result:|warning: )' "$LOG_BODY" 2>/dev/null | head -40 || true
      printf '```\n'
    fi
  }
}

emit_issue() {
  {
    printf '## Отказ CI: %s\n\n' "$WF_NAME"
    printf 'Запуск: <%s> (ID %s, `%s`)\n\n' "${RUN_URL:-https://github.com/$REPO/actions/runs/$RUN_ID}" "$RUN_ID" "$RUN_BRANCH"
    printf 'Итог: `%s`\n\n' "$RUN_STATE"
    emit_table
    printf '\n---\n\nСгенерировано `scripts/ci-self-heal-analyzer.sh`. Исправления не применялись.\n'
  }
}

REPORT=""
case "$OUTPUT" in
  table) emit_table ;;
  md)    emit_md ;;
  json)  emit_json ;;
  issue)
    emit_issue
    REPORT="$REPORTS_DIR/ci-issue-body-$RUN_ID.md"
    mkdir -p "$REPORTS_DIR"
    emit_issue >"$REPORT"
    printf '\nТело issue сохранено: %s\n' "$REPORT"
    printf 'Создать issue (публичное действие, решает человек):\n'
    printf '  gh issue create --repo %s --title "CI: %s (#%s)" --body-file "%s"\n' \
      "$REPO" "$WF_NAME" "$RUN_ID" "$REPORT"
    ;;
esac

if [ "$SAVE" = true ] && [ "$OUTPUT" != issue ]; then
  mkdir -p "$REPORTS_DIR"
  REPORT="$REPORTS_DIR/ci-analysis-$RUN_ID-$(date +%Y%m%d-%H%M%S).md"
  emit_md >"$REPORT"
  printf '\nОтчёт сохранён: %s\n' "$REPORT"
fi

log_action "CI-HEAL" "RUN=$RUN_ID findings=$COUNT" >&2

[ "$COUNT" -gt 0 ] && exit 0
exit 1
