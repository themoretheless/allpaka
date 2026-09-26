---
name: git-worktree-manager
description: Автоматизация lifecycle worktrees — создание изолированных рабочих деревьев для параллельной разработки multiple branches без конфликтов. Использовать когда нужно быстро переключаться между ветками или работать над несколькими features одновременно.
---

# Git Worktree Management

Управляемый через CLI автоматический workflow worktrees для изолированной работы на multiple branches.

## Когда использовать

- Параллельная разработка нескольких features без context switching
- Исправление hotfix в основной ветке пока другие разработки продолжаются
- Тестирование разных версий кода одновременно
- Изолированная среда для code reviews / bisect
- Эксперименты с безопасным rollback

## Установка требований

```sh
# Проверить наличие git worktree support
git worktree list  # Должно показать текущие worktrees (или пустую таблицу)
```

## Основные команды

### Создание нового worktree

```sh
scripts/git-worktree-manager.sh add <branch_name> <path_to_worktree>
```

**Опции:**
- `--dry-run` — показать preview без создания
- `--force` — skip checks (не рекомендуется)

**Примеры:**
```bash
# Создать worktree для feature ветки
scripts/git-worktree-manager.sh add feature-api ./worktrees/feature-api

# Добавить существующую ветку в новый путь
scripts/git-worktree-manager.sh add release-v2.0 ./worktrees/release-v2

# Preview before creation
scripts/git-worktree-manager.sh --dry-run add experimental ./worktrees/experiment
```

### Просмотр всех worktrees

```sh
scripts/git-worktree-manager.sh list [--verbose|--json]
```

**Output examples:**

Basic listing:
```
/path/to/repo    branch: main   (head: abc1234)
/path/to/repo/worktrees/dev   branch: dev   (head: def5678)
/path/to/repo/worktrees/feat-x   branch: feat-x   (head: ghi9012)
```

With verbose:
```
=== Git Worktrees ===

Path              Branch         Commit SHA         
----------------- -------------- -------------------
./worktrees/dev   dev            a1b2c3d            
./worktrees/feat  feat-api       e4f5g6h            
./worktrees/test  test-release   i7j8k9l            

Total worktrees: 3
```

JSON format (machine-parseable):
```json
[
  {
    "path": "/repo",
    "head": {
      "ref": "refs/heads/main",
      "branch": "main",
      "sha": "abc1234"
    }
  },
  {
    "path": "/repo/worktrees/dev",
    "head": {
      "ref": "refs/heads/dev",
      "branch": "dev",
      "sha": "def5678"
    }
  }
]
```

### Удаление worktree

```sh
scripts/git-worktree-manager.sh remove <path>|<branch_name> [--force]
```

**Safety features:**
- Проверяет uncommitted changes перед удалением
- Warn если есть локальные изменения
- Опциональное удаление local branch после

**Примеры:**
```bash
# Remove по path
scripts/git-worktree-manager.sh remove ./worktrees/old-feature

# Remove по имени ветки
scripts/git-worktree-manager.sh remove feature-old

# Force skip warning
scripts/git-worktree-manager.sh remove feature-temp --force
```

### Cleanup orphaned entries

```sh
scripts/git-worktree-manager.sh prune [--dry-run]
```

Удаляет stale worktree entries где filesystem path больше не существует:

```bash
# Actual cleanup
scripts/git-worktree-manager.sh prune

# Preview what will be removed
scripts/git-worktree-manager.sh prune --dry-run
```

Sync изменения между worktrees

```sh
scripts/git-worktree-manager.sh sync <source_path> <target_branch> [--dry-run]
```

Показывает commits которые можно применить, но не выполняет merge автоматически:

```bash
# Preview sync
scripts/git-worktree-manager.sh --dry-run sync ./worktrees/feature main
```

## Typical Workflows

### Parallel Development

```bash
# Main repo работает на main ветке
cd /repo
git checkout main

# Создать worktree для feature-A
scripts/git-worktree-manager.sh add feature-a ./worktrees/feature-a
cd ./worktrees/feature-a
# ... разрабатывать feature-A здесь ...

# Создать worktree для feature-B
scripts/git-worktree-manager.sh add feature-b ./worktrees/feature-b
cd ../..  # Вернуться в main repo
cd ./worktrees/feature-b
# ... разрабатывать feature-B здесь ...

# Переключаться между контекстами легко:
cd ./worktrees/feature-a && git status
cd ./worktrees/feature-b && git status
cd .. && git status  # Каждый worktree независим
```

### Hotfix Workflow

```bash
# Основной разработчик продолжает work на main
cd /repo
git checkout main
# ... работа ...

# Нужно срочно зафиксить баг?
scripts/git-worktree-manager.sh add hotfix-fix ./worktrees/hotfix
cd ./worktrees/hotfix
git checkout -b fix-critical-issue

# Fix критическую проблему
# commit, push, create PR

# Основная разработка продолжается
cd ../..  # Возврат к main
git status  # Untracked changes в worktrees/ скрыты
```

### Testing Multiple Versions

```bash
# Test different releases simultaneously
scripts/git-worktree-manager.sh add v1.0 ./worktrees/v1.0
scripts/git-worktree-manager.sh add v2.0 ./worktrees/v2.0
scripts/git-worktree-manager.sh add main ./worktrees/main-latest

# Run tests on each version
for wt in ./worktrees/*; do
  echo "=== Testing $wt ==="
  bash "$wt/run-tests.sh" || true
done
```

### Safe Experimentation

```bash
# Экспериментальная ветка с risk of breakage
scripts/git-worktree-manager.sh add experiment-weird-hack ./worktrees/experiment

cd ./worktrees/experiment
# ломать код как угодно...
git commit -m "breaking change"

# Если сломалось — просто удалить worktree:
scripts/git-worktree-manager.sh remove experiment-weird-hack

# Main repo не затронут
cd ../..
git status  # Чистый worktree
```

## Safety Rules

| Rule | Description | Example |
|------|-------------|---------|
| Disk space check | Requires minimum 100MB free | Aborts if insufficient |
| Path validation | Never overwrites existing dirs | Error if path exists |
| Change warning | Shows uncommitted before remove | Needs confirmation |
| Branch safety | Only deletes local branches | Remote preserved |
| Orphan detection | Auto-prunes broken refs | Maintenance mode |

## Integration with Reporting

Работает seamlessly с `git-uncommitted-report.sh`:

```bash
# Report for all worktrees
for wt in ./worktrees/*/; do
  echo "=== $wt ==="
  bash scripts/git-uncommitted-report.sh -C "$wt" || true
done
```

Вывод:
```
=== ./worktrees/feature-a ===
# Незакоммиченные файлы — /repo/worktrees/feature-a

| файл | статус | index + | index − | worktree + | worktree − | всего + | всего − |
|---|:--:|---:|---:|---:|---:|---:|---:|
| src/api.rs | M | 15 | 2 | 45 | 0 | **60** | **2** |

**Итого: +60 / −2** (index: +15/−2, worktree: +45/−0)

=== ./worktrees/feature-b ===
Незакоммиченных файлов нет: рабочее дерево и индекс чистые.
```

## Use Cases

### Context Switching Without Commit

```bash
# Начали feature-A
cd /repo
git checkout main
scripts/git-worktree-manager.sh add feature-a ./worktrees/feature-a
cd ./worktrees/feature-a
echo "change" > file.txt  # Изменения

# Переходим к другим задачам без commit'а
cd ../..
git checkout bug-fix
# ... fix bug ...

# Возвращаемся к feature-A — все изменения сохранены
cd ./worktrees/feature-a
git status  # Наши изменения на месте
```

### Bisect & Debugging

```bash
# Создаем worktrees для каждой версии при bisect
scripts/git-worktree-manager.sh add HEAD^@{1} ./worktrees/bisect-parent
scripts/git-worktree-manager.sh add HEAD^ @{2} ./worktrees/bisect-grandparent

# Test each independently
bash ./worktrees/bisect-parent/test.sh
bash ./worktrees/bisect-grandparent/test.sh
```

### Code Review Isolation

```bash
# Reviewer получает isolated copy для экспериментов
scripts/git-worktree-manager.sh clone-repo owner/review-needed ./review-worktree

# Reviewer может делать что угодно без риска сломать main
```

## Configuration

### Environment Variables

```bash
# Set default worktrees base directory
export WORKTREE_BASE="./worktrees"  # Default

# Custom log directory
export LOG_DIR=".qoder/logs"
```

### Recommended Directory Structure

```
repo/
├── .git/
├── src/
├── docs/
└── worktrees/
    ├── feature-a/          # Isolated feature development
    ├── feature-b/          # Another parallel feature
    ├── hotfix-123/         # Quick bug fixes
    ├── release-v1.0/       # Release preparation
    └── experiment/         # Risky experiments
```

## Comparison with Manual Commands

Manual approach:
```sh
git worktree add -b feature-x ./worktrees/feature-x  # No validation
# ... work ...
git worktree remove ./worktrees/feature-x  # Danger: may leave orphaned
```

Managed approach:
```sh
scripts/git-worktree-manager.sh add feature-x ./worktrees/feature-x  # Validated
# ... work ...
scripts/git-worktree-manager.sh remove feature-x  # Safe: warnings + cleanup
```

Преимущества:
- ✅ Validation disk space
- ✅ Path existence checks
- ✅ Uncommitted changes warnings
- ✅ Automatic orphan detection via `prune`
- ✅ Better error messages
- ✅ Logging to `.qoder/logs/`

## Troubleshooting

### Path already exists

```bash
$ scripts/git-worktree-manager.sh add feature-x ./worktrees/feature-x
Error: Path already exists: ./worktrees/feature-x
```

**Solution:**
```bash
# Remove existing first
scripts/git-worktree-manager.sh remove ./worktrees/feature-x

# Then create new one
scripts/git-worktree-manager.sh add feature-x ./worktrees/feature-x
```

### Insufficient disk space

```bash
$ scripts/git-worktree-manager.sh add feature-x ./worktrees/x
Error: Insufficient disk space (need at least 100MB, have 50MB)
```

**Solution:**
```bash
# Check available space
df -h ./worktrees

# Clean up old worktrees
scripts/git-worktree-manager.sh prune
```

### Orphaned entries after force-delete

```bash
# Stale worktree references
git worktree list  # Shows paths that don't exist

# Clean them
scripts/git-worktree-manager.sh prune
```

## Best Practices

1. **Use descriptive paths**: `./worktrees/feature-login-flow` vs `./wt-1`
2. **Delete old worktrees promptly**: Don't accumulate unused ones
3. **Use `--dry-run` first**: Always preview before destructive operations
4. **Commit frequently**: Even in isolated worktrees, avoid large uncommitted changesets
5. **Document purpose**: Add README in each worktree explaining its goal

## Related Skills

- **git basic**: Standard git commands and workflows
- **git-uncommitted-report**: Detailed change reporting per worktree
- **gh-auto-merge**: Merge finished worktrees back to main
