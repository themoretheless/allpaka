#!/usr/bin/env bash
# materialize-git-gh-plugins.sh — разворачивает пакеты плагинов из плоских файлов
# в каноническую структуру каталогов.
#
# Почему скрипт: автор плагинов (агент без shell) не может создавать каталоги,
# поэтому манифест и SKILL.md лежат плоско, а раскладку делает этот скрипт.
#
# Было (плоско в plugins/ и scripts/):          Стало:
#   plugins/git.kimi.plugin.json          ->    plugins/git/kimi.plugin.json
#   plugins/git.SKILL.md                  ->    plugins/git/skills/git/SKILL.md
#   plugins/gh.kimi.plugin.json           ->    plugins/gh/kimi.plugin.json
#   plugins/gh.SKILL.md                   ->    plugins/gh/skills/gh/SKILL.md
#   plugins/<инструмент>.kimi.plugin.json ->    plugins/<инструмент>/kimi.plugin.json
#   plugins/<инструмент>.SKILL.md         ->    plugins/<инструмент>/skills/<инструмент>/SKILL.md
#   scripts/<инструмент>.sh               ->    plugins/<инструмент>/bin/<инструмент>.sh
#   scripts/scripts-common.sh             ->    plugins/<инструмент>/bin/scripts-common.sh
#   ../scripts/git-uncommitted-report.sh  ->    plugins/git/bin/git-uncommitted-report.sh
#
# Плоские файлы остаются источником: повторный запуск перезаписывает структуру.
# Bin-пакеты (у которых есть scripts/<имя>.sh) получают каталог bin/ вместе с
# библиотекой scripts-common.sh: скрипты ищут её рядом с собой, без неё bin/
# не работает.
#
# Использование:
#   bash plugins/materialize-git-gh-plugins.sh [--dest <каталог-для-плагинов>] [--only <имя>]
#   # --dest полезен, чтобы разложить пакеты сразу в каталог личного маркета,
#   # не трогая дерево репозитория.
#   # --only повторяет раскладку одного пакета (можно несколько раз).

set -euo pipefail

src=$(cd -- "$(dirname -- "$0")" && pwd)
dest=$src
repo_scripts=$src/../scripts

packages="git gh gh-auto-merge gh-pr-status-watch git-worktree-manager ci-self-heal-analyzer"
filter=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) dest=${2:?нужен путь}; shift 2 ;;
    --only) filter="$filter ${2:?нужно имя пакета}"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'неизвестный аргумент: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[ -d "$dest" ] || mkdir -p "$dest"

need() {
  [ -f "$1" ] || { printf 'нет файла: %s\n' "$1" >&2; exit 1; }
}

# какие bin-файлы тащит пакет (по имени в scripts/), построчно; пусто — без bin/
bin_for() {
  case "$1" in
    git) printf 'git-uncommitted-report.sh\n' ;;
    gh) printf '' ;;
    gh-auto-merge|gh-pr-status-watch|ci-self-heal-analyzer)
      printf '%s.sh\n' "$1" ;;
    git-worktree-manager)
      # у команды report свой помощник — без него report недоступен
      printf 'git-worktree-manager.sh\ngit-uncommitted-report.sh\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

script_path() {  # кандидат-источник для bin-файла пакета
  local name=$1 cand
  for cand in "$repo_scripts/$name" "$src/$name"; do
    if [ -f "$cand" ]; then printf '%s\n' "$cand"; return 0; fi
  done
  return 1
}

materialize() {
  local p=$1 pdir=$dest/$p bins name src_file need_lib=0
  need "$src/$p.kimi.plugin.json"
  need "$src/$p.SKILL.md"

  mkdir -p "$pdir/skills/$p"
  install -m 0644 "$src/$p.kimi.plugin.json" "$pdir/kimi.plugin.json"
  install -m 0644 "$src/$p.SKILL.md" "$pdir/skills/$p/SKILL.md"

  bins=$(bin_for "$p")
  [ -n "$bins" ] || return 0

  case "$p" in
    gh-auto-merge|gh-pr-status-watch|git-worktree-manager|ci-self-heal-analyzer) need_lib=1 ;;
  esac

  mkdir -p "$pdir/bin"
  for name in $bins; do
    if src_file=$(script_path "$name"); then
      install -m 0755 "$src_file" "$pdir/bin/$name"
    else
      printf 'предупреждение: %s не найден в scripts/ — bin/%s пропущен\n' "$name" "$name" >&2
    fi
  done

  if [ "$need_lib" = 1 ]; then
    # скрипты ищут библиотеку рядом с собой, без неё bin/ не работает
    if src_file=$(script_path scripts-common.sh); then
      install -m 0644 "$src_file" "$pdir/bin/scripts-common.sh"
    else
      printf 'предупреждение: scripts-common.sh не найден — скрипты в %s/bin не запустятся\n' "$pdir" >&2
    fi
  fi
}

selected() {
  if [ -z "$filter" ]; then return 0; fi
  case " $filter " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

done_pkgs=""
for p in $packages; do
  selected "$p" || continue
  materialize "$p"
  done_pkgs="$done_pkgs $p"
done

for want in $filter; do
  case " $packages " in
    *" $want "*) ;;
    *) printf 'пакета %s нет среди известных: %s\n' "$want" "$packages" >&2; exit 2 ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then
  for p in $done_pkgs; do
    python3 -c 'import json,sys; json.load(open(sys.argv[1])); print("json ok:", sys.argv[1])' \
      "$dest/$p/kimi.plugin.json"
  done
else
  printf 'python3 не найден — проверку JSON пропускаю\n'
fi

printf '\nготово. пакеты:%s\n' "$done_pkgs"
for p in $done_pkgs; do
  printf '\n%s:\n' "$p"
  find "$dest/$p" -type f | sed "s|^$dest/||" | sort
done
printf '\nосталось: указать каталог %s в клиенте (вкладка «个人» → ＋) или скопировать\n' "$dest"
printf 'каталоги пакетов туда, где клиент читает личные плагины.\n'
