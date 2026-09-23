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

## git и gh: почему плоские файлы

Автор этих двух плагинов — агент без shell: он умеет писать файлы, но не умеет
создавать каталоги. Поэтому источник лежит плоско:

```
git.kimi.plugin.json   git.SKILL.md   gh.kimi.plugin.json   gh.SKILL.md
```

Разложить в каноническую структуру (идемпотентно, с проверкой JSON):

```sh
bash plugins/materialize-git-gh-plugins.sh
# или сразу в каталог личного маркета, не трогая дерево репо:
bash plugins/materialize-git-gh-plugins.sh --dest /path/to/personal-plugins
```

Скрипт создаёт `git/skills/git/SKILL.md`, `gh/skills/gh/SKILL.md`,
`*/kimi.plugin.json` и копирует `scripts/git-uncommitted-report.sh` в
`git/bin/git-uncommitted-report.sh` (с `chmod 0755`). Канонический источник
скрипта отчёта — `scripts/git-uncommitted-report.sh`; в `git/bin/` попадает его
копия на момент раскладки.

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
