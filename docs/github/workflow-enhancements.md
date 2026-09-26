# Автоматизация GitHub-потока

Четыре скрипта в `scripts/` и четыре плагина к ним. Источник плагина лежит плоско
(`plugins/<имя>.kimi.plugin.json` + `plugins/<имя>.SKILL.md`), а канонический пакет
`kimi.plugin.json` + `skills/<имя>/SKILL.md` + `bin/` из него раскладывает
`plugins/materialize-git-gh-plugins.sh` (см. `plugins/README.md`). Общий код —
`scripts/scripts-common.sh`, он попадает в `bin/` всех четырёх пакетов, потому что
скрипты ищут библиотеку рядом с собой.

Философия та же, что у скиллов `git` и `gh`: по умолчанию только чтение, изменяющее —
только по явной просьбе и с показом того, что уйдёт наружу.

## Что где лежит

| Инструмент | Скрипт | Источник плагина | Назначение |
|---|---|---|---|
| auto-merge | `scripts/gh-auto-merge.sh` | `plugins/gh-auto-merge.{kimi.plugin.json,SKILL.md}` | слить PR, когда открыт, не draft, mergeable и check-и зелёные |
| status watch | `scripts/gh-pr-status-watch.sh` | `plugins/gh-pr-status-watch.{…}` | дождаться check-ов, посмотреть таблицу или NDJSON |
| worktree | `scripts/git-worktree-manager.sh` | `plugins/git-worktree-manager.{…}` | add/list/remove/prune/report для git worktree |
| CI-разбор | `scripts/ci-self-heal-analyzer.sh` | `plugins/ci-self-heal-analyzer.{…}` | классифицировать отказ Actions и предложить команды |

Каталоги вида `plugins/gh-auto-merge/skills/…` в дереве не лежат: их создаёт
`bash plugins/materialize-git-gh-plugins.sh --dest …` из плоских файлов (то же самое
репозиторий делает для `git` и `gh`). В клиент ставится результат раскладки.

Точные флаги и коды выхода — в `--help` каждого скрипта и в соответствующем `SKILL.md`;
здесь описано то, что не видно из справки.

## Свёртка check-ов

`gh pr view --json statusCheckRollup` отдаёт массив, у массива поля `conclusion` нет,
поэтому `.statusCheckRollup.conclusion` в jq — пустота, и наивная проверка считает
любой PR зелёным. `pr_checks_rollup` в `scripts-common.sh` сворачивает массив по
приоритету:

1. любой `failure|timed_out|cancelled|action_required` → `failure`;
2. любой незавершённый (`queued|in_progress|pending|waiting`) → `pending`;
3. все `success|skipped|neutral` → `success`;
4. пустой массив → `none`.

`mergeable` у gh — булево, не строка `"TRUE"`; сравнение со строкой всегда ложно.

## Границы, которые не надо сдвигать

- **Слияние.** `gh-auto-merge.sh` не знает флага `--admin`, не обходит защиту веток и
  не имеет режима «слить любой ценой». `--dry-run` печатает планируемую команду `gh` и
  не обращается к mutating API. Настоящие флаги слияния — `--squash|merge|rebase`,
  `--delete-branch`, `--auto`; `--strategy` не существует.
- **Worktree.** `remove` удаляет только worktree и никогда не трогает ветку: удаление
  дерева обратимо, потеря коммитов — нет. Грязное дерево не удаляется без `--force`,
  основной worktree не удаляется вовсе, ветка уже занятого другого дерева не берётся
  (подсказка — `--detach`).
- **Разбор CI.** Analyzer только читает: не правит файлы, не делает `git add`, не
  коммитит, не перезапускает раннеры. `gh run rerun` появляется в отчёте с пометкой
  МУТАЦИЯ и выполняется человеком. Логи берутся `gh run view --log-failed`, а не
  `gh run download --name test-log` — артефакта с таким именем workflow не создаёт, и
  анализ остался бы без данных.
- **Форматирование.** Подсказки дают `rustfmt --edition <ed> <файлы из лога>`, а не
  `cargo fmt` по workspace: репозиторий не считается fmt-чистым, и `cargo fmt`
  переформатирует файлы вне задачи.
- **Токены.** `gh auth status` достаточно; `gh auth token` не выводить, токен не
  подставлять в команды текстом и не печатать в отчётах.

## Проверено

Прогоны сделаны на заглушках `gh` и на реальном scratch-репозитории (вне дерева allpaka).
Тестовые наборы — временные скрипты в `/tmp`, в репозиторий они не входят; таблица описывает,
что именно проверялось, а не то, что можно прогнать из клона.

| Набор | Что покрывает | Результат |
|---|---|---|
| библиотека | `parse_duration`, `json_get`, `pr_checks_rollup`, `pr_is_mergeable`, `is_owner_name`, `detect_repo` (https, `.git`, scp-ssh, ssh-url, trailing slash, не-GitHub) | 33 / 0 |
| auto-merge + watch | dry-run, `--squash/--rebase --delete-branch/--auto` в argv, failure/draft/closed/mergeable=false, таймаут, `gh auth` отказ, коды 1/2/3/4, NDJSON | 38 / 0 |
| worktree | реальный git-репозиторий: add (новая/существующая/занятая ветка, `--detach`, занятый путь), list/`--json`/`--verbose`, remove (грязное, `--force`, чистое, неизвестный путь, основной worktree), prune (`--dry-run` и реальный прогон), report, коды аргументов | 38 / 0 |
| CI-разбор | сценарии rustfmt/ring/test/compile/network/missing/oom/green, `--no-log`, `--output=json|md|issue`, `--save`, тело issue через `--body-file`, отказ gh → rc 3, аргументы → rc 2, отсутствие мутирующих вызовов. Две регрессии найдены прогоном на реальном красном запуске: лог с упавшими тестами без имён (grep без совпадения под `pipefail` ронял скрипт молча, rc 1 и пустой stdout) и лог длиннее `--max-log` (конвейер `gh … | head -c` терял весь уже принятый кусок) | 61 / 0 |

Все скрипты проходят `bash -n` и печатают `--help` с кодом 0; четыре пакета раскладываются
`bash plugins/materialize-git-gh-plugins.sh --dest …`, манифесты валидируются как JSON, и
материализованные копии из `bin/` запускаются оттуда же (проверено на dry-run слияния,
разборе и `report`).

## Логи и отчёты

`log_action` пишет в `.qoder/logs/YYYYMMDD.log` (переменная `LOG_DIR`), `--save` и
`--output=issue` — в `.qoder/reports/` (`REPORTS_DIR`). Обе директории относительные от
каталога запуска; каталоги создаются по мере надобности.

## CI-хук

В `.github/workflows/ci.yml` оставлен закомментированным job `self-heal-analysis`:
`if: failure()`, `permissions: actions: read`, `GH_TOKEN` из `github.token`, разбор с
`--output=md --save` и загрузка отчёта артефактом. Включать осознанно: он добавляет
ещё один раннер на каждый красный прогон.
