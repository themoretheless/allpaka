# MTP (nextn) спекулятивное декодирование для qwen35moe

Родная MTP-спекуляция: драфт предлагает не внешняя модель, а собственный
nextn-блок транка (`blk.N` за последним слоем, `nextn_predict_layers=1`).
Реализовано и проверено на `Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf` (unsloth,
21.1 ГБ); обычный `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` MTP-блока не несёт.

## Механика

MTP-блок — полный слой (attention со своей KV в слоте `n_layers` + MoE
256 экспертов top-8 + shared), плюс тензоры `nextn.eh_proj` [4096,2048],
`enorm`, `hnorm`, `shared_head_norm`. Шаг драфта (как в llama.cpp
`graph_mtp`):

```
x = eh_proj([ enorm(emb(token)) | hnorm(h_prev) ])   // 4096 -> 2048
h_next = shared_head_norm(layer(x))                  // сид следующего шага
logits = lm_head(h_next)                             // head транка
```

Сид первого шага раунда — `output_norm(hidden)` транка на committed-границе;
шаги 2..k цепляют собственный `h_next`. `embed_tokens`/`lm_head` в GGUF под
MTP нет — берутся транковые.

Раунд (`MtpSpeculator::round`, crates/allpaka-model/src/speculate.rs):
1. `arm_slots(k+1)` — GDN rollback-слоты (conv-окно + deltanet-состояние на
   каждую строку verify, ~66 МБ на слот).
2. k драфт-шагов `Model::mtp_step` (позиции пишутся в KV MTP-слоя).
3. `Model::verify_tokens` — один GPU command buffer на [next + k драфтов]:
   per-row argmax, бит-совместимый с plain decode, и hidden-строки.
4. Acceptance: сколько драфтов подтвердил транк. Полный accept — extra
   `mtp_step` для committed-позиции и disarm; частичный —
   `truncate(committed)` + `restore_slot(accepted)` (без replay), сид цепочки
   = hidden-строка verify на committed-строке.

Откатимость: KV транка режется truncate, GDN (необратимый) восстанавливается
из слота одной копией. `gdn_conv_batch`/`gdn_step_batch` пишут слоты во время
verify — окно conv сидится с поправкой на границу чанка (off-by-one там был
корнем дивергенции k=4).

## Verify (crates/allpaka-backend/src/gpu.rs, encode_verify_tokens)

Один буфер на m = k+1 строк. Проекции — TILE=m matvec (веса читаются один
раз на раунд), по строкам батчены: `norm_rows`, `qk_prep_batch256`,
`attend_rows256`, `resnorm_router_rows` (новое: resnorm+router+top-k на все
строки), `moe_combine_rows` (новое: combine на все строки, shared-эксперт в
отдельных TILE-регионах), GDN-ядра батчевые изначально. Эксперты идут
per-row indexed matvec (3m диспатчей/слой) либо ROWS-вариантом
(`matvec_q4_k_mv`/`matvec_q5_k`/`matvec_q6_k` с function constant 6) — одна
dispatch на матрицу для всех строк.

NB: невыставленный bool function constant в Metal на M4 Max читается как
**true** (измерено). Все bool fc теперь пинуются явно в pipeline_wait/
pipeline_dual — иначе специализации молча уезжают на вариантный путь.

## Цифры (M4 Max, 32 токена decode, машина дрейфует ±15%)

| k | acceptance | MTP tok/s | plain tok/s |
|---|-----------|-----------|-------------|
| 1 | 88%       | 72–87     | 89–115      |
| 2 | 64–77%    | 70–80     |             |
| 3 | 58%       | 69–81     |             |
| 4 | 44%       | 48–71     |             |

Стрим бит-экзактен plain greedy при всех k (bench `PASS`). Скорости ниже
plain: движок latency/bandwidth-bound (~110 ГБ/с эффективных на decode
matvec), verify m=3 ≈ 19.5 мс (MoE ~11.6 мс, GDN ~4.5 мс), драфт-шаг ~2 мс
(lm_head по vocab + GPU round-trips). Экспертный трафик растёт линейно по m,
поэтому «бесплатного» verify, как у dense-моделей, здесь нет.

## Флаги

- `ALLPAKA_DRAFT_K` (1–4, default 4) — драфтов на раунд в bench.
- `ALLPAKA_MTP_DEBUG=1` — тайминги раунда, драфты vs target, сиды.
- `ALLPAKA_VERIFY_PARITY=1` — verify через batch-путь вместо one-buffer GPU.
- `ALLPAKA_ROWS_DEBUG=1` — печать engagement ROWS-пайплайнов экспертов.

MTP в `serve` НЕ подключён: при текущем соотношении скоростей (~0.8× plain)
спекуляция не окупается — default off. Путь живёт в `bench --engine` для
моделей с `nextn_predict_layers > 0`.

## Тесты

- `crates/allpaka-model/tests/verify_tokens_parity.rs` — verify vs batch-путь
  (smoke) и teacher-forced verify vs GPU decode (жёсткий, m=5/3/2).
- `gpu_parity` (allpaka-backend) — страж численности decode-ядер.

## Если возвращаться к скорости

Замерено: verify ≈ 12 мс фикс + ~2.65 мс/строка; ROWS-батчинг экспертов
диспатчи режет, но время не двигает — упор в полезную работу ядер
(~110-150 ГБ/с). Реальные рычаги: ускорение matvec-ядер глобально (вне
объёма MTP) либо one-buffer драфт-шаг (~2 мс → ~1 мс). До ~145 tok/s этого
недостаточно.
