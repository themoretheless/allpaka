---
name: gh-auto-merge
description: Слияние PR через gh только после проверки условий — открыт, не черновик, mergeable, check-и зелёные. Использовать, когда нужно слить PR без ручной сверки statusCheckRollup и ревью, или прогнать очередь PR подряд.
---

# Auto-merge pull request

Скрипт `scripts/gh-auto-merge.sh` (в составе плагина — `bin/gh-auto-merge.sh`, рядом
обязательная `bin/scripts-common.sh`). Все действия — настоящие команды `gh`.

## Как решить, можно ли сливать

Четыре условия, все обязательные:

```sh
gh pr view <n> --json state,isDraft,mergeable,statusCheckRollup
```

- `state == OPEN` и `isDraft == false`;
- `mergeable == true` (у gh это булево, а не строка);
- свёртка `statusCheckRollup` равна `success`.

Последнее — причина, по которой нельзя читать `.statusCheckRollup.conclusion`:
`statusCheckRollup` это массив, у массива такого поля нет. Каждый элемент имеет
`status` (`QUEUED`/`IN_PROGRESS`/`COMPLETED`) и `conclusion` (`SUCCESS`/`FAILURE`,
пустой у незавершённых). Сводка: хотя бы один `failure|timed_out|cancelled|action_required`
→ failure; иначе есть незавершённый → pending; иначе все `success|skipped|neutral` → success.

## Использование

```sh
scripts/gh-auto-merge.sh --dry-run 123              # только проверка условий + команда
scripts/gh-auto-merge.sh --squash 123               # слить squash-ем (default)
scripts/gh-auto-merge.sh --rebase --delete-branch 123
scripts/gh-auto-merge.sh --timeout=45m 123          # ждать зелёных check-ов до 45 минут
scripts/gh-auto-merge.sh --auto 123                 # включить auto-merge GitHub, не сливать сейчас
```

Флаги слияния настоящие: `--squash`, `--merge`, `--rebase`, `--delete-branch`, `--auto`.
`--strategy` у `gh pr merge` нет. Прочее: `--timeout=DURATION` (default `30m`),
`--interval=SECONDS` (период опроса, default 10), `--repo=owner/name`, `--dry-run`, `-h`.

Пока check-и в процессе, скрипт опрашивает статус раз в `--interval` до `--timeout`
(код 2 при таймауте). При `failure` или `action_required` сливать не станет сразу (код 1).

Коды выхода: `0` — слит (или dry-run успешен), `1` — условие не выполнено,
`2` — таймаут или неверные аргументы, `3` — нет `gh auth`.

## Очередь PR

Сначала dry-run по всем кандидатам, потом слияние утверждённых:

```sh
gh pr list --state open --json number,statusCheckRollup,mergeable \
  --jq '.[] | select(.mergeable == true) | .number'
```

Не сливать молча: `gh pr merge` — действие, которое видят другие люди, и его не отменить
штатной командой. Показывать номер, заголовок и стратегию до выполнения.

## Границы

- Только чтение до момента слияния; `--dry-run` не обращается к mutating API.
- `--admin` не используется, защита веток не обходится, `--delete-branch` только вместе
  с уже состоявшимся слиянием.
- `reviewDecision` выводится как факт, но сам скрипт ревью не требует: если в репозитории
  есть required reviews, решение остаётся за `mergeable` и защитой веток.
- Ошибки gh показывать как есть (HTTP 404, `not authorized`, `pull request is locked`) —
  не додумывать состояние PR.
- Токен не печатать: `gh auth status` достаточно, `gh auth token` без явной просьбы не запускать.

## Окружение

Обычный shell, `gh` в PATH и авторизован (`gh auth login` или `GH_TOKEN`/`GITHUB_TOKEN`),
`jq`, и git-репозиторий с GitHub remote — из него берётся `owner/name`, если не задан
`--repo`. Никакой изоляции: ни MCP-сервера, ни контейнера.
