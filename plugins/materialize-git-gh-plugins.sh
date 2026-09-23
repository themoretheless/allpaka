#!/usr/bin/env bash
# materialize-git-gh-plugins.sh — разворачивает пакеты плагинов git и gh
# из плоских файлов в каноническую структуру каталогов.
#
# Почему скрипт: автор плагинов (агент без shell) не может создавать каталоги,
# поэтому манифест и SKILL.md лежат плоско, а раскладку делает этот скрипт.
#
# Было (плоско в plugins/):                   Стало:
#   plugins/git.kimi.plugin.json        ->    plugins/git/kimi.plugin.json
#   plugins/git.SKILL.md                ->    plugins/git/skills/git/SKILL.md
#   plugins/gh.kimi.plugin.json         ->    plugins/gh/kimi.plugin.json
#   plugins/gh.SKILL.md                 ->    plugins/gh/skills/gh/SKILL.md
#   ../scripts/git-uncommitted-report.sh ->   plugins/git/bin/git-uncommitted-report.sh
#
# Плоские файлы остаются источником: повторный запуск перезаписывает структуру.
# После запуска манифесты проверяются на валидный JSON (если есть python3).
#
# Использование:
#   bash plugins/materialize-git-gh-plugins.sh [--dest <каталог-для-плагинов>]
#   # --dest полезен, чтобы разложить пакеты сразу в каталог личного маркета,
#   # не трогая дерево репозитория.

set -euo pipefail

src=$(cd -- "$(dirname -- "$0")" && pwd)
dest=$src

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) dest=${2:?нужен путь}; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'неизвестный аргумент: %s\n' "$1" >&2; exit 2 ;;
  esac
done

need() {
  [ -f "$1" ] || { printf 'нет файла: %s\n' "$1" >&2; exit 1; }
}

need "$src/git.kimi.plugin.json"
need "$src/git.SKILL.md"
need "$src/gh.kimi.plugin.json"
need "$src/gh.SKILL.md"

script=""
for cand in "$src/../scripts/git-uncommitted-report.sh" "$src/git-uncommitted-report.sh"; do
  if [ -f "$cand" ]; then script=$cand; break; fi
done
[ -n "$script" ] || printf 'предупреждение: git-uncommitted-report.sh не найден, bin/ будет пустым\n' >&2

mkdir -p "$dest/git/skills/git" "$dest/git/bin" "$dest/gh/skills/gh"

install -m 0644 "$src/git.kimi.plugin.json" "$dest/git/kimi.plugin.json"
install -m 0644 "$src/git.SKILL.md"         "$dest/git/skills/git/SKILL.md"
install -m 0644 "$src/gh.kimi.plugin.json"  "$dest/gh/kimi.plugin.json"
install -m 0644 "$src/gh.SKILL.md"          "$dest/gh/skills/gh/SKILL.md"
if [ -n "$script" ]; then
  install -m 0755 "$script" "$dest/git/bin/git-uncommitted-report.sh"
fi

if command -v python3 >/dev/null 2>&1; then
  for m in "$dest/git/kimi.plugin.json" "$dest/gh/kimi.plugin.json"; do
    python3 -c 'import json,sys; json.load(open(sys.argv[1])); print("json ok:", sys.argv[1])' "$m"
  done
else
  printf 'python3 не найден — проверку JSON пропускаю\n'
fi

printf '\nготово. структура:\n'
find "$dest/git" "$dest/gh" -type f | sort
printf '\nосталось: указать каталог %s в клиенте (вкладка «个人» → ＋) или скопировать\n' "$dest"
printf 'каталоги git/ и gh/ туда, где клиент читает личные плагины.\n'
