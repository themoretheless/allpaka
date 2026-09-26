# plugins/

Kimi-плагины (личный маркет): всё, что агент получает как скилл и/или MCP-сервер.
Установка — вкладка «个人» → ＋.

## Состав

| Плагин | Вид | Что даёт |
|---|---|---|
| `allpaka/` | скилл | движок allpaka: `serve`, `status`, `chat`, `rag-test` |
| `rag-mcp/` | MCP-сервер | локальная база знаний rag-mcp (DuckDB) |
| `git/` | скилл | git: статус, диффы, отчёт по незакоммиченным файлам (+/− строк по файлам), ветки, stash, worktree |
| `gh/` | скилл | GitHub CLI: PR, issues, Actions и их логи, релизы, `gh api` |
| `gh-auto-merge/` | скилл + `bin/` | слить PR, только когда открыт, не draft, mergeable и check-и зелёные |
| `gh-pr-status-watch/` | скилл + `bin/` | опрос check-ов PR: таблица, NDJSON, слежение за одним PR до завершения |
| `git-worktree-manager/` | скилл + `bin/` | add/list/remove/prune/report для git worktree с проверками |
| `ci-self-heal-analyzer/` | скилл + `bin/` | разбор упавшего запуска GitHub Actions и подсказки команд (не применяет) |

## Конвенция пакета

```
<name>/kimi.plugin.json        # манифест: name, version, interface{}, skillInstructions, skills
<name>/skills/<skill>/SKILL.md # скилл: YAML-frontmatter name/description + тело
<name>/bin/…                   # необязательно: вспомогательные скрипты
```

Манифест подключает MCP-сервер блоком `mcpServers` (как в `rag-mcp/`), либо
остаётся чисто скилловым (как `allpaka/`). Скилловый вариант не требует ни
MCP-хоста, ни внешних зависимостей — он просто описывает агенту, какие команды
запускать.

## Плоские файлы: почему и какие

Авторы этих плагинов — агент без shell: он умеет писать файлы, но не умеет создавать
каталоги. Поэтому источник лежит плоско:

```
git.kimi.plugin.json    git.SKILL.md    gh.kimi.plugin.json    gh.SKILL.md
gh-auto-merge.kimi.plugin.json          gh-auto-merge.SKILL.md
gh-pr-status-watch.kimi.plugin.json     gh-pr-status-watch.SKILL.md
git-worktree-manager.kimi.plugin.json   git-worktree-manager.SKILL.md
ci-self-heal-analyzer.kimi.plugin.json  ci-self-heal-analyzer.SKILL.md
```

Разложить в каноническую структуру (идемпотентно, с проверкой JSON):

```sh
bash plugins/materialize-git-gh-plugins.sh
# или сразу в каталог личного маркета, не трогая дерево репо:
bash plugins/materialize-git-gh-plugins.sh --dest /path/to/personal-plugins
# переразложить один пакет после правки:
bash plugins/materialize-git-gh-plugins.sh --only gh-auto-merge
```

Скрипт создаёт `<пакет>/kimi.plugin.json`, `<пакет>/skills/<пакет>/SKILL.md`, а у
скиллов со скриптами — ещё `<пакет>/bin/`. Канонический источник скриптов — `scripts/`;
в `bin/` попадает их копия на момент раскладки.

Четыре скрипта автоматизации (`gh-auto-merge.sh`, `gh-pr-status-watch.sh`,
`git-worktree-manager.sh`, `ci-self-heal-analyzer.sh`) ищут `scripts-common.sh` рядом с
собой, поэтому в их `bin/` раскладывается и библиотека — без неё скрипт не стартует.
`git-worktree-manager` несёт дополнительно `git-uncommitted-report.sh`: без него команда
`report` недоступна.

## Никакой изоляции

git и gh — обычные CLI-скиллы: они не поднимают MCP-сервер, не используют Docker и
не создают песочницу. Агент просто запускает команды на той же машине, где лежит
репозиторий.

Что должно быть на машине:

```sh
command -v git gh && git --version && gh --version && gh auth status
```

Проверка до установки — этот же однострочник. Единственное требование к клиенту:
возможность агента запускать команды (shell/exec). Если её нет, скиллы ничего не
сделают — и это ограничение клиента, а не плагинов.

## Оговорки

- MCP-варианты (`uvx mcp-server-git`, Docker-образ GitHub MCP server) сознательно
  не используются: доступ к CLI идёт напрямую, без посредника.
- Перед вкладкой «个人» → ＋ раскладку надо выполнить: клиент ожидает каталог с
  `kimi.plugin.json`, а не плоский файл.
