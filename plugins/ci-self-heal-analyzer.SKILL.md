---
name: ci-self-heal-analyzer
description: Разбор упавшего запуска GitHub Actions по JSON и логам gh с предложениями команд. Использовать, когда CI покраснел и нужно определить причину — форматирование, компиляция, тесты, clippy, сеть, память, — не листая раннер руками. Исправления не применяет.
---

# Анализ отказа CI

Скрипт `scripts/ci-self-heal-analyzer.sh` (в составе плагина — `bin/ci-self-heal-analyzer.sh`,
рядом обязательная `bin/scripts-common.sh`). Только чтение: читает `gh run view` и его
логи, печатает находки и команды. Файлы не правит, не коммитит, не пушит, CI не перезапускает.

## Использование

```sh
scripts/ci-self-heal-analyzer.sh 1234567890              # таблица находок
scripts/ci-self-heal-analyzer.sh --output=md --save 1234567890
scripts/ci-self-heal-analyzer.sh --output=json 1234567890
scripts/ci-self-heal-analyzer.sh --output=issue 1234567890
scripts/ci-self-heal-analyzer.sh --no-log 1234567890     # только по именам шагов
```

ID запуска брать из `gh run list --limit 5` (поле `databaseId`).

Флаги: `--output=table|md|json|issue`, `--save` (отчёт в `$REPORTS_DIR`),
`--repo=owner/name`, `--no-log`, `--max-log=BYTES` (default 400000),
`--local` (дополнить read-only `rustfmt --check` по изменённым `.rs`), `-h`.

Коды выхода: `0` — находки есть, `1` — паттерны не распознаны, `2` — аргументы,
`3` — нет `gh auth` или запуск недоступен.

## Что распознаётся

| Категория | По чему | Что предлагает |
|---|---|---|
| `formatting` | `Diff in <file> at line N` | `rustfmt --edition <ed> <файлы из лога>` |
| `toolchain-cpu-flags` | `CAPS_STATIC`/`MIN_STATIC_FEATURES` | `RUSTFLAGS="" cargo build --workspace` |
| `test-failure` | `test result: FAILED`, имена из `---- <test> stdout ----` | `cargo test --workspace -- --nocapture` |
| `compile-error` | `error[EXXXX]`, `could not compile` | `cargo check --workspace --all-targets` |
| `clippy` | `clippy::`, `error: ... clippy` | `cargo clippy --workspace --all-targets` |
| `transient-network` | `failed to load source`, `network failure`, `403`, `429` | `gh run rerun <id> --failed` — помечено как МУТАЦИЯ |
| `resource` | `SIGKILL`, `out of memory` | `cargo test --workspace --jobs 1` |
| `missing-local-resource` | отсутствующий `.gguf`, `models/`, rag-mcp | проверить условный skip теста |
| `infra` | упал служебный шаг (`Set up toolchain`, cache) | смотреть лог шага |

Уверенность проставляется по строке, а низкая уверенность — повод читать лог, а не
повторять команду вслепую.

## Почему форматирование чинят rustfmt по файлам

`cargo fmt` без аргументов проходит по всему workspace и переформатирует файлы вне
задачи — дифф пачкается чужими правками. Поэтому подсказка перечисляет ровно те файлы,
о которых написал rustfmt. Пути в логе могут быть абсолютными внутри раннера: оставляйте
часть от корня репозитория.

## Про ring и target-cpu

В этом репозитории `.cargo/config.toml` задаёт `-C target-cpu=native` для
aarch64-apple-darwin, и эти rustflags доходят до обычных сборок; на раннере ring 0.17
падает на проверке `CAPS_STATIC & MIN_STATIC_FEATURES`. Лечится `RUSTFLAGS=""` (см.
комментарий в `.github/workflows/ci.yml`), а не понижением версии ring.

## Черновик issue

`--output=issue` пишет тело в `$REPORTS_DIR/ci-issue-body-<id>.md` и печатает готовую
команду. Сам issue создаёт человек: публикация видна другим и не отменяется одной кнопкой.

```sh
gh issue create --repo owner/name --title "CI: <workflow> (#<run>)" --body-file <файл>
```

Тело идёт через `--body-file`, а не через `-`: `gh issue create --body -` читает stdin,
а `--body-file` надёжен для многострочного markdown.

## Границы

- Никаких правок файлов, `git add`, `git commit`, `git push` со стороны скрипта.
- `gh run rerun`, `gh run cancel`, `gh workflow run` — только по явной просьбе: это меняет
  состояние CI и видно всем.
- Логи берутся `gh run view --log-failed` (текст упавших шагов), а не
  `gh run download --name test-log`: артефакт с таким именем workflow не создаёт, и
  анализ молча остался бы без данных.
- Объём логов ограничен `--max-log`, потому что полный лог сборки бывает на десятки
  мегабайт и не влезает в разбор.
- Ошибки gh показывать как есть (`HTTP 404`, `not logged in`), не додумывая причину отказа.

## Окружение

`gh` в PATH и авторизован (`gh auth login` или `GH_TOKEN`/`GITHUB_TOKEN`), `jq`, git-репозиторий
с GitHub remote для `owner/name`, `Cargo.toml` рядом для определения edition. `--local`
требует `cargo` и `rustfmt`. Токен не печатать. Ни MCP-сервера, ни контейнера: обычный shell.
