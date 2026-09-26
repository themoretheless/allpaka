---
name: git
description: Git в рабочем дереве — статус, отчёт по незакоммиченным файлам (+/− строк по файлам), диффы, история, ветки, stash, worktree. Использовать, когда пользователь спрашивает «что не закоммичено», просит диff/статус, хочет понять состояние репозитория или собирается коммитить.
---

# Git

Плагин даёт агенту git без самодельных обёрток: все действия — реальные команды
git в репозитории пользователя.

## Когда использовать

- «что не закоммичено», «сколько строк изменено и в каких файлах»
- «покажи диff / статус / историю», «на какой я ветке»
- перед правками — понять, чистое ли дерево и не затрутся ли чужие изменения
- подготовка коммита (показать, что войдёт), stash, worktree

## Отчёт по незакоммиченным файлам

Основной сценарий — `bin/git-uncommitted-report.sh` из этого плагина:

```sh
bash bin/git-uncommitted-report.sh -C /path/to/repo        # markdown в stdout
bash bin/git-uncommitted-report.sh -C . -o uncommitted.md  # ещё и в файл
```

Печатает ветку, upstream с ahead/behind, число stash-записей и таблицу
`файл | статус | index +/− | worktree +/− | всего +/−` с итоговой суммой.

Ручной эквивалент, если скрипт недоступен:

```sh
git status --porcelain=v1 --untracked-files=all
git diff --cached --no-renames --numstat    # index:    +/− по файлам
git diff --no-renames --numstat             # worktree: +/− по файлам
git ls-files --others --exclude-standard    # новые файлы (строки — wc -l)
```

Оговорки, которые обязательно озвучивать вместе с отчётом:

- числа — это **дельты строк** из `numstat`, а не размер файла на диске;
- `--no-renames`: переименование считается как удаление старого + добавление нового;
- бинарные файлы (`-` в numstat) построчно не считаются;
- для `??`-файлов «+» = число строк в файле, «−» = 0;
- в больших репах `target/`, `vendor/`, `node_modules/`, `models/`, `*.log`,
  `*.zip` — обычно untracked-мусор; показывать его отдельной строкой, а не
  мешать с кодом.

## Диагностика состояния дерева

```sh
git rev-parse --abbrev-ref HEAD                            # текущая ветка
git rev-parse --abbrev-ref --symbolic-full-name '@{u}'     # upstream (падает, если нет)
git rev-list --left-right --count '@{u}...HEAD'            # behind<TAB>ahead
git stash list
git worktree list
git status --short --branch
git diff --shortstat                                       # N files changed, X insertions(+), Y deletions(−)
git log --oneline -20
git diff --stat                                            # сводка по файлам
```

## Worktree Automation

### Lifecycle Management

**Create isolated worktree for parallel branch development:**
```sh
# Create worktree at ./worktrees/feature-x with new branch feature-x
git worktree add -b feature-x ./worktrees/feature-x

# Or reuse existing branch in separate worktree path
git worktree add ./worktrees/feature-y feature-y
```

**List all worktrees with details:**
```sh
git worktree list                          # basic: path + branch
git worktree list --verbose                # shows head commit SHA
git worktree list --json                   # machine-readable output
```

**Remove worktree and cleanup branch:**
```sh
# First ensure no uncommitted changes, then:
git worktree remove ./worktrees/feature-x  # deletes worktree directory
git branch -D feature-x                    # delete branch if local
```

**Prune stale worktree entries (orphaned paths):**
```sh
git worktree prune                         # removes broken references
git worktree prune --expire 2 weeks        # also expire old entries
```

**Sync changes between worktrees (read-only preview):**
```sh
# See what would be merged if switching to target branch
git -C ./worktrees/feature-x fetch origin
git -C ./worktrees/feature-x log HEAD..main --oneline
```

### Integration with Reporting

The `scripts/git-uncommitted-report.sh` works across all worktrees:

```sh
# Report for main repo
bash scripts/git-uncommitted-report.sh -C .

# Report for specific worktree
bash scripts/git-uncommitted-report.sh -C ./worktrees/feature-x

# Batch across all worktrees (from plan)
for wt in ./worktrees/*/; do
  echo "=== $wt ==="
  bash scripts/git-uncommitted-report.sh -C "$wt" || true
done
```

### Safety Rules for Worktrees

- ✅ Always check disk space before creating new worktrees (`du -sh .`)
- ✅ Validate target path is empty or doesn't exist first
- ⚠️ Never remove worktree with uncommitted changes without warning
- ⚠️ Local modifications in worktree don't affect main repo directly
- ⚠️ Switching branches requires clean working directory or stash first

## Правила безопасности

- **По умолчанию только чтение.** `add`, `commit`, `switch`, `merge`, `rebase`,
  `reset`, `stash pop/drop`, `clean` — только по явной просьбе.
- Никогда без прямого указания: `reset --hard`, `checkout -- .`, `restore .`,
  `clean -fdx`, `push --force`, удаление веток и worktree.
- Перед изменяющей командой показать, что именно будет затронуто
  (`git status --short`, `git diff --cached --stat`).
- Незакоммиченные изменения — это чужая работа: не прятать их в stash и не
  переключать ветки «чтобы получилось», не спросив.
- Коммит от имени пользователя — только с подтверждением и без `--no-verify`,
  если не попросили обратного.
- Не выдумывать вывод git: показывать то, что вернула команда, включая ошибки
  («not a git repository» — это ответ, а не повод угадывать).

## Никакой изоляции: обычный shell

Плагин **не поднимает MCP-сервер**, не использует контейнер и не создаёт
песочницу — он работает прямо в командной строке пользователя, там же, где лежит
репозиторий. Никакой посредник между агентом и git не нужен.

Что должно быть на машине:

```sh
command -v git && git --version        # git в PATH
```

Если клиент даёт агенту только MCP-инструменты и не даёт запускать команды,
скилл работать не сможет — это ограничение клиента, а не плагина.
