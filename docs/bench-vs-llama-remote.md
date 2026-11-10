# Сравнение allpaka с llama.cpp на отдельной машине

Цель: честные числа «мы против llama.cpp» на тихой машине. Основной
машине мешают параллельные сессии, которые грузят GPU.

## 1. Что нужно

- Apple Silicon, лучше той же конфигурации (M4 Max, 128 ГБ): 235B требует
  около 90 ГБ памяти под веса.
- Свободно 120+ ГБ на диске.
- Rust (nightly ставится сам через `rust-toolchain.toml`), `jq`, llama.cpp:

```bash
brew install jq llama.cpp
```

## 2. Сборка

```bash
git clone https://github.com/themoretheless/allpaka.git
```

```bash
cd allpaka && cargo build --release
```

## 3. Модели

Класть в `~/.cache/allpaka/models/`:

```bash
mkdir -p ~/.cache/allpaka/models
```

30B, 17 ГБ:

```bash
curl -L -C - -o ~/.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

235B, две части, 46.6 + 36.1 ГиБ:

```bash
curl -L -C - -o ~/.cache/allpaka/models/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/resolve/main/UD-Q2_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
```

```bash
curl -L -C - -o ~/.cache/allpaka/models/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00002-of-00002.gguf https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/resolve/main/UD-Q2_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00002-of-00002.gguf
```

Бенчи указывают на часть `00001`, вторую загрузчик подхватит сам.

## 4. Перед замером: машина должна быть тихой

Закрыть всё тяжёлое (IDE, браузеры, другие агенты). Проверить, что GPU в
простое около 0%:

```bash
ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'
```

Сразу после скачивания Spotlight индексирует новые файлы минут 10-20.
Подождать, пока `mds_stores` в `top -o cpu` не успокоится.

## 5. Запуск

Скрипт чередует allpaka и llama-bench (5 повторов, pp480 + tg32), так что
фоновый шум бьёт по обоим одинаково:

```bash
BENCH_OUTPUT_DIR=bench-30b scripts/bench-compare.sh ~/.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

```bash
BENCH_OUTPUT_DIR=bench-235b scripts/bench-compare.sh ~/.cache/allpaka/models/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
```

235B идёт долго: прогрев 83 ГиБ плюс 5 повторов на каждый движок.

## 6. Что прислать обратно

Итоговую таблицу, которую печатает скрипт, и архив папок `bench-30b`,
`bench-235b` (там сырые логи и JSON обоих движков). Плюс одной строкой
модель машины:

```bash
sysctl -n machdep.cpu.brand_string hw.memsize
```

## Ориентиры с основной машины (M4 Max 128 ГБ)

| модель | метрика | allpaka | llama.cpp |
|---|---|---|---|
| 30B | prefill pp480 | 1255-1333 | ~1404 |
| 30B | decode tg32 | 124-125 | ~109 |
| 235B | prefill pp480 | 194 (тихая машина, август) | 204.7 |
| 235B | decode tg32 | 26-27 (тихая машина) | 28.07 |

Если 235B на тихой машине выдаст сильно ниже 194 / 26, это настоящая
регрессия, а не шум.
