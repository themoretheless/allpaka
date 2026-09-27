# Per-decode-token byte census — GLM, 235B, 30B (2026-09-24)

## The question, and why it needed no machine

Every decode prediction in this repo has the same shape:
`gain = recoverable% x per-token byte share`. The shares plugged into it were
derived from kernel *rates* - the same numbers the prediction is used to judge -
so each share was a consistency check, not a measurement. `q5_0`'s write-up
says this out loud ("the 28% time share was itself derived from rates",
[../2026-09-24-q5-0-mv-e2e/paired-32.txt](../2026-09-24-q5-0-mv-e2e/paired-32.txt)).

This closes that by reading the byte budget off the GGUF instead: tensor names,
dims, quant type and block size give bytes/token for the routing the decode
path actually follows (routed experts at `n_used/n_expert`, everything else
whole). It is a count, not a rate, so like the
[dispatch census](../2026-09-24-dispatch-census/README.md) it is valid on a
3x-contended machine and needs no cooldown, band or thermometer gate.

DoD from [../../roadmap.md](../../roadmap.md): independent per-token byte
shares for the three decode targets, with the shares the existing predictions
used either confirmed or corrected.

## Instrument

[../../../scripts/gguf-bytes-per-token.py](../../../scripts/gguf-bytes-per-token.py)
(`python3 scripts/gguf-bytes-per-token.py models/<file> [--rate TYPE=GB/s]
[--tok-s N]`). Three self-checks, all passing on all three models:

- sum of tensor extents == bytes on disk, exactly (GLM 67.96 GiB / 67.96 GiB,
  235B 82.67 / 82.67, 30B 17.28 / 17.28) - so no tensor is missing or double-counted;
- every tensor's measured bytes-per-element is compared against the quant block
  table and mismatches are printed; zero printed, which is what pins the type
  ids (this file's ids are +1 shifted against the ggml enum as first assumed);
- per-tensor counts are cross-read against the model metadata (`expert_count`,
  `expert_used_count`, `layer_count`).

A split GGUF keeps its tensor table **per part**, with part-relative offsets -
GLM's part 1 lists 568 tensors summing to its own 46.56 GiB, part 2 lists 235
summing to 21.40 GiB. The assumption that part 1 carries the whole model was
wrong and is what made the first version of this script report 46.56 GiB for a
67.96 GiB file. It is also the same wrong assumption behind `allpaka inspect`
reporting 46.6 GiB for GLM; a split model's reported residency is shard 1 only.

## Result: the byte budget each decode token pays

Raw: [glm-byte-census.txt](glm-byte-census.txt),
[qwen3-235b-byte-census.txt](qwen3-235b-byte-census.txt),
[qwen3-30b-byte-census.txt](qwen3-30b-byte-census.txt).

| Model | streamed/token | attention | routed gate/up | routed down | shared expert | head | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4.5-Air | 8.57 GiB | 31.6% | 25.9% | 20.0% | 7.0% | 5.5% | 10.0% |
| qwen3-235b | 9.17 GiB | 39.6% | 31.5% | 21.7% | 0 | 5.2% | 2.0% |
| qwen3-30b | 1.79 GiB | 26.9% | 35.4% | 21.8% | 0 | 13.3% | 2.6% |

GLM's split by quant type, which is what the per-format predictions consume:
`q4_k` 65.0%, `q8_0` 15.3%, `q5_0` 8.3%, `q6_k` 6.4%, `q5_k` 4.0%, `f32` 1.1%.
The largest single cells: `q4_k` attention 30.7% (165 tensors), `q4_k` routed
gate/up 25.9% (92), `q8_0` routed down 11.7% (22), `q5_0` routed down 8.3% (24).

## Five things this changes

**1. The `q5_0` prediction had the right numerator and the wrong denominator.**
The port log's arithmetic was "0.71 GiB of 6.1 per token". Measured: the
numerator is exactly right (`q5_0` routed down = 726 MiB = 0.709 GiB, to three
digits), the denominator is 8.57 GiB, so the byte share is **8.3%, not 11.6%**.
This does not touch the shipped +16.05% - that is a measurement - but it does
cap what the same reasoning can forecast for the *next* format: at 8.3% of the
bytes, a `q5_0` kernel that were infinitely fast buys +9.1% only under a
fully-saturated-pipe assumption, and the token is not that: 423 GB/s achieved
across the whole budget, against 496-497 for `q8_0` in the same machine's one
clean window - 85% of the wall, with ~1.7 ms/token of slack.

**2. `n=24` for `q5_0` routed down, and the dispatch census drifted by 24.**
The file has exactly 24 stacked `ffn_down_exps` tensors in `q5_0` (the other 22
MoE layers' downs are `q8_0`), and flipping `ALLPAKA_Q5_0_MV` moves the decode
dispatch count by exactly 803 -> 778. One tensor, one dispatch, 1:1, from two
instruments that share no input. That converts the census's "24 of the drift is
the `q5_0_mv` port" from an inference into a counted fact, and it says the
`_mv` port's dispatch saving is one per *layer tensor*, i.e. capped by how many
tensors of that format a model has - on GLM, 24 out of 778, 3%.

**3. The models are in different regimes, and only two of them are byte-limited.**
Time ledger at streamed rates measured in one window, each format taken at the
shape the model that owns it dispatches
([time-ledger.txt](time-ledger.txt) section 3, raw rows in
[q6k-q5o-streamed-rates.txt](q6k-q5o-streamed-rates.txt)):

| Model | decode tok/s | ms/token | sum of its kernels' own rates | wall / kernel sum |
| --- | ---: | ---: | ---: | --- |
| GLM-4.5-Air | 45.95 | 21.8 | 20.0 ms | **0.92 - accounted** |
| qwen3-235b | **23.43 (paired A/B)** / 20.0 (published single-shot) | 42.7 / 50.0 | 30.1 ms | **1.42x / 1.66x slower than its kernels** |
| qwen3-30b | 137.5 | 7.3 | 4.9 ms | 0.68 - closest to the wall of the three |

This supersedes the version of the table built on
[../2026-09-24-indexed-matvec/README.md](../2026-09-24-indexed-matvec/README.md)
rates, whose band (GLM 19.7-23.8, 235B 34.2-36.8, 30B 2.9-3.8 ms) came from the
repo's two disagreeing streamed `q4_k` numbers, 354.6 and 469 GB/s. That
disagreement is resolved: `q4_k` runs 472 GB/s at the 235B's own shape in a
window whose `q8_0` meter reads 496, and the old 354.6 came from a window whose
`q8_0` meter read 400 - a contended afternoon, not a different kernel. The
30B's ledger, which previously could not be completed because 27.3% of its
bytes are `q6_k` with no published rate, is now closed at 363.9 GB/s.

**4. Priority consequence for the 235B, against the goal's leading item.**
The 235B's `q3_k` down projection is 17.4% of its bytes. Making that kernel
perfect (`q4_k`-parity per *weight*, i.e. 1.22 against its measured 2.43 ps per
weight) removes 4.8 ms of its 50 ms token = **+10.7%** end to end. But its
published e2E was 1.36-1.46x slower than the sum of its own measured kernels,
which is 13-16 ms/token of something the kernel model does not see - a block
bigger than the entire `q3_k` prize, resting on the one row of the status table
flagged "one single-shot arm per engine, never paired".

That row has now been paired, and the fork resolves against `q3_k`. Six
alternating allpaka/llama pairs at pp480/tg64 on this file; three cleared the
band rule and put allpaka decode at **23.43, 23.34, 23.64** tok/s (llama
29.42-29.70, ratio 0.79x in every pair; median **0.793x**). The other three were
rejected as a degraded window - allpaka's own prefill thermometer fell to
110-128 tok/s against 197-202 in the in-band pairs, and they are recorded, not
averaged in, including the loose end that llama's decode did not move in those
same windows. So the 20.0 was ~15% of contention rather than a kernel deficit,
and 27 tok/s
(the micro sum, = 37 ms/token) is NOT where the machine lands: 12.6 ms/token
still sit outside the kernel sum, against a `q3_k` prize of 4.8 ms. The gap to
llama is 6.2 ms/token and the unexplained block is twice that, so the lever on
this model is the 12 ms, not the unpack chain. Paired artifact and per-pair
table:
[../2026-09-24-235b-e2e/pair-analysis.txt](../2026-09-24-235b-e2e/pair-analysis.txt).

**5. Byte rates are shape-bound, in one direction only.** Re-measuring each
format at the geometry its owner dispatches moved almost nothing for the wide
models and a lot for the narrow one: swapping `n_out` and `n_in` costs ~1%
(`q4_k` 472.2 vs 468.6 GB/s), narrowing a 4096 row to 1408 costs 3% (`q5_0`
396.6 vs 383.6), but a dimension at 768 costs 18-23% - the 30B's expert set
runs `q4_k` [768,2048] at 386.7 and `q6_k` [2048,768] at 363.9, the narrowest
byte rates recorded anywhere in the repo. The mechanism is not identified here
(3 blocks per row at `n_in` 768 against 6-16 elsewhere is the obvious candidate
and is untested). The consequence is a correction to point 3's own predecessor:
at the wide-table rates the 30B's kernels explained 54% of its token and it read
as "not byte-limited at all"; at its own rates they explain 68%, so the
fixed-per-token-cost story for the 30B is weaker than the last section said.
Practical rule for predictions: take the rate at the dispatched shape, or the
error is up to 20% and always in the flattering direction.

> **Annotation, 2026-09-25 - the candidate this point left untested is now
> tested, and it covers half the claim.** The "3 blocks per row at `n_in` 768"
> hypothesis was priced at equal bytes per dispatch by the isomer ladders
> ([../2026-09-25-narrow-shape-isomer-ladders/results.txt](../2026-09-25-narrow-shape-isomer-ladders/results.txt),
> §10) and it is real but narrow: at the 30B's own dispatch size a `nb=3` row
> costs 1.224x per block what a `nb=4` row costs (12 windows, 1.188-1.239), while
> `nb=6` costs 1.059/0.997 and `nb=8` 0.994/0.999, and `q8_0` at the identical
> 768-wide shape costs 0.998. So of this point's two 768 rows, `q6_k` [2048,768]
> is a block-count victim and `q4_k` [768,2048] is not - its gap against
> [4096,1536] is a blocks-per-dispatch difference (49152 against 196608), an axis
> that paired within-ladder statistics cannot measure. Full write-up and the
> revised price of the whole effect (~4% of the 30B's decode wall, not the 14%
> the shape looked like worth) in correction #11 of
> [../2026-09-24-indexed-matvec/README.md](../2026-09-24-indexed-matvec/README.md).
> The practical rule is unchanged and survives with its numbers intact: the rates
> quoted here are measured at the dispatched shape, which is exactly why the
> ledger built from them holds whichever mechanism owns each gap.

## What this does not say

- The byte budgets themselves measure no rate. Every ms in the ledger is a
  kernel rate from another artifact applied to a byte count from this one, and
  inherits that rate's geometry - which is why point 5 exists.
- The 8.57 GiB/token is weights only. KV-cache traffic at pp480+tg32 is ~0.1
  GiB/token on GLM (1.2%) and grows with context, so the shares drift slowly
  with sequence length - irrelevant at decode-benchmark context, not at 100k.
- The per-token rule assumes the indexed expert path: routed tensors cost
  `n_used/n_expert` of their bytes. A model dispatched through a non-indexed
  fallback pays up to 16x those bytes, which would change the ledger
  completely - and is one candidate explanation for point 4 that this
  instrument cannot see, because it reads the file, not the dispatch.
- `q5_0` is now measured (396.6 GB/s at [4096,1536], 383.6 at GLM's
  [4096,1408]), not assumed - but it is measured against *this* repo's kernels.
  A llama.cpp `q5_0` rate would not transfer, and no parity claim here rests on
  one.
- None of the rates above are dense `lm_head` rates. The head is 4-13% of each
  model's token bytes and goes through the un-indexed matvec path, which
  `indexed_matvecs_report_effective_bandwidth` does not exercise; the ledger
  applies an indexed rate to it, and that is an approximation with an
  unmeasured sign.
