# Decode matvec bandwidth, by geometry — 2026-09-24

Apple M4 Max, allpaka `3ef9d9c` + this session's working tree. Every number is
GPU-clock effective bandwidth of weight bytes, from
`cargo test -q -p allpaka-backend --test gpu_ffnbench -- --ignored --nocapture --test-threads=1`
(the counters are process-wide, so the two tests in that binary must not run at
once — see [Caveats](#caveats-on-the-harness)).

The point of the file is eleven corrections and one consequence. The second
correction undoes most of the consequence and the third undoes part of what the
consequence blamed; the fourth takes back one sentence of the third, so read them
in that order. Corrections 5-9 bound `q3_k`'s residue from four directions and
none of them shipped; #10 rebuilds the streamed table in one window; #11 splits
the shape claim that #10 left open.

## Correction: the flat shape is not the decode shape

`ffn_shaped_matvecs` measured single-column, non-indexed matvecs. MoE decode
does not issue those: a top-k layer sends `k` experts through **one** indexed
dispatch. Measured both ways, the same kernels:

| Kernel | flat single column | indexed ×8 (decode) |
| --- | ---: | ---: |
| `q8_0` | 198–206 GB/s | **454–476** GB/s |
| `q5_k` | ~124 GB/s | 162–202 GB/s |
| `q2_k` (gate/up) | 118–138 GB/s | 136–160 GB/s |
| `q3_k` (down) | 84 GB/s | 121–141 GB/s |

Two decisions this reversed, both as interleaved A/B/A pairs:

1. **`matvec_q3_k` looked 10–15% faster than the port on the flat shape**
   (93–99 against 85 GB/s best). Indexed, the port wins every pair
   (165.7/145.0/144.1 against 133.4/140.7/131.2), and the 235B end-to-end A/B
   ([q3-235b-ab.txt](q3-235b-ab.txt)) agrees: no gain from the flip, so the
   default stays on `matvec_q3_k_mv`.
2. **Vectorising `matvec_q5_k_mv`'s activation loads looked +5–9% flat**
   (113.6 → 124.2/121.3 best). Indexed it is ~13% *slower*
   (162.1/185.6 against the incumbent's 194.6/202.8), so the change was
   reverted and `matvec_q5_k_mv` is back at its `3ef9d9c` form. Register
   pressure is the likely mechanism; not pursued further.

The one kernel change that survived both geometries is the activation-load
vectorisation in `matvec_q3_k_mv` (32 scalar loads → 8 `float4`): flat 85
against 52–82, indexed 145/166/144 against 118/141/129. It is in the tree, and
`gpu_parity`'s `q3_k_matches_cpu_reference` covers it.

## Correction #2 (same day): those indexed numbers were cache-resident

The table above re-read **one** set of 8 experts forever. That set is 16–53 MB,
which fits the M4 Max's last-level cache, so the indexed column measured cache
bandwidth and ranked formats by *footprint* rather than by cost – and part of
the Q8-vs-K-quant gap the section below builds on evaporates once the weights
actually stream. `indexed_matvec_bandwidth` now takes a `sets` argument and
rotates the routed experts per round (`ids[s] = (r % sets) * slots + s`) so
~384 MB passes between two visits to one set, and the formats are measured
interleaved inside one process, because this machine's own share of its GPU
drifts by 3× over tens of seconds.

Streaming, clean run (`rep 2` of
[indexed-controls.txt](indexed-controls.txt); `rep 1` and `rep 3`, run a few
seconds either side of it, are 30-50% slower on *every* format at once, so a run
whose best/median spread is not ~1.00x is unreadable and `q8_0` doubles as the
contention meter):

| Kernel | shape | cached (table above) | streamed GB/s | Gweight/s | ps per weight |
| --- | --- | ---: | ---: | ---: | ---: |
| `q8_0` | [4096,1536] | 454-476 | 400.2 | 377 | 2.65 |
| `q4_k` | [4096,1536] | not measured | 354.6 | 630 | **1.59** |
| `q5_k` | [4096,1536] | not measured | 248.9 | 362 | 2.76 |
| `q3_k` | [4096,1536] | 121-236 | 177.1 | 412 | 2.43 |
| `q2_k` | [4096,1536] control | - | 114.7 | 350 | 2.86 |
| `q2_k` | [1536,4096] (real gate/up) | 136-276 | 199.7 | 609 | **1.64** |
| `q4_k` | [1536,4096] control | - | 350.0 | 622 | 1.61 |

`q4_k`/`q5_k` join the table at 1536-wide `n_in`: the earlier 1408 is not a
multiple of the 256-element block, so that row walked 1280-wide rows and
labelled them 1408. The two `control` rows are not shapes the 235B has; they are
each format at the *other* format's shape, added because `q2_k`'s real row is
4096 wide and the rest are 1536 wide, and without them the table below silently
compares a format difference and a shape difference.

*(Correction #10, same day, re-measures every row of this table in one
uncontended process. Read the `streamed GB/s` and `ps per weight` columns as
19% low and the `q3_k`/`q4_k` ratio as 6% flattering; the conclusions below
stand, and `q4_k`'s comes out stronger.)*

## Consequence (revised): the lever is `q3_k`'s down path, and `q4_k` has none

Per weight - the only unit in which a quant choice can be judged, since the
formats carry different bytes per weight - the decode kernels cost:

```
1.59-1.61 ps  q4_k, either shape
1.64          q2_k at [1536,4096]   <- the 235B's gate/up
2.43          q3_k at [4096,1536]   <- the 235B's down
2.65          q8_0
2.76          q5_k
2.86          q2_k at [4096,1536]
```

Three things follow, and two of them undo what this file said an hour earlier.

1. **`q4_k` is at the memory limit, so it is off the suspect list.** It reads
   89% of `q8_0`'s bytes on the same streaming path, and its per-weight cost is
   the *best* in the table despite doing the most bit work per weight. No kernel
   change to `matvec_q4_k_mv` recovers decode time, and GLM's 0.86x cannot be
   inside that kernel - which is what the status doc's "in-graph Q4/Q8 matvec BW
   is the remaining lever" said. It is not. What is left for GLM is the time
   *between* kernels (`WAIT_NS` - `GPU_BUSY_NS` under `ALLPAKA_PROFILE`), the
   shared-expert barrier, and `q5_k`'s down path.
2. **The 235B's `q2_k` gate/up is already at `q4_k` parity** (1.64 against
   1.61 ps/weight), so re-quantizing those 46.27 GiB of experts to Q4_K buys
   nothing and costs ~1.7x the bytes. The earlier "1.9x of FFN headroom" in this
   file came from comparing `q2_k`'s wide shape against `q8_0`'s narrow one.
3. **What is left is `q3_k` (and `q5_k`) at the down shape** - as a layout
   question, not a kernel one; see Correction #5 for what that turned out to
   mean. `q3_k` costs 2.43
   ps/weight against `q4_k`'s 1.59 - a 1.53x penalty on the 25.46 GiB of Q3_K
   down projections, and `q5_k` is worse still at 2.76. Applied to the 235B's FFN
   as a whole (two gate/up + one down per expert) the gap between its
   Q2_K/Q3_K mix and an all-Q4_K mix is 5.71 against 4.80 ps/weight = **1.19x**
   - a model-side option worth knowing about, not worth +32 GiB of weights
   unless the kernels stay where they are.

`q2_k`'s 1.7x shape sensitivity (199.7 GB/s at [1536,4096] against 114.7 at
[4096,1536]) next to `q4_k`'s flatness (354.6 against 350.0) says the Q2/Q3
kernels pay a large fixed cost per output row - the 6-block row gives each lane
one and a half `ib` iterations against four on the wide row, so per-row
addressing, activation staging and the tail reduction dominate the short row.
That is a *structural* account, and it is also the reason "fewer ops per weight
byte" was the wrong diagnosis: at the narrow shape the K-quants do not even
approach the byte rate they reach at the wide one, on the same memory.

**Was still open, and was the reason to stop tuning geometry:** whether the
narrow-shape deficit is ALU issue or lane load-latency. Correction #9 closed it
- see the note at the end of this section. The one clean A/B says thread count
is not the lever - `ALLPAKA_Q2_NR0=1` (2x the SIMD groups, half the serial
blocks per lane) measured 257.1 against the default's 256.6 GB/s, flat to 0.2%,
and `=4` only loses. The discriminator is a measurement-only function constant
that keeps every weight load and every FMA and replaces just the bit-extraction
chain with a cheap derived value: same bytes, far fewer ALU ops. If the byte
rate rises toward `q4_k`'s, the lanes are issue-bound and a Q4-style inner loop
ported to `q3_k`/`q5_k` pays; if it does not move, it is latency and only more
bytes in flight per lane helps. Do not write that constant on a busy machine -
the same config on this one read 44.8 against 256.6 GB/s ten minutes apart, with
every format falling in step, and the *ratios* moved too.

*(Update, same day: the curve fit below settled the per-row half of that
question without needing the constant - there is no per-row cost to find. The
issue/latency half was settled the same way the other four mechanisms were, by a
whole second kernel rather than a function constant: `matvec_q3_k_mlp` hoists
every weight load of a block-row ahead of its consumers and moves `q3_k` by
-0.26%, so the answer is "neither" - the schedule is not the residue either. See
[correction #9](#correction-9-the-fifth-mechanism---bytes-in-flight-per-lane---moves-nothing).)*

## Correction #3: the short-row deficit is per-block arithmetic, not per-row setup

`indexed_matvec_curve_separates_row_and_block_cost` walks `n_in` at fixed
`n_out = 4096` and x8 experts, so a row's dispatch time decomposes as the kernel
actually spends it:

```
T(nb) = A + B * nb      A = per-row cost, B = per-block cost
```

25 cells (5 formats x `n_in` in 1024/1536/2048/3072/4096), every cell streamed
past the last-level cache, all of them burned, then measured interleaved in 6
passes keeping the per-cell best, and least-squares fitted per format - **inside
one process**, because cross-process A/B on this machine is what produced the
354.6 against 405.2 GB/s disagreement that made the previous section's shape
claims unreadable. A fit reads ratios from a single run, so the drift is
common-mode. [curve-fit-rep1.txt](curve-fit-rep1.txt) and
[curve-fit-rep2.txt](curve-fit-rep2.txt) are two processes, ~2 minutes apart:

| Format | `A` ns/row | `B` ps/row/block | `B` per weight | residual | GB/s at nb=16 | % of `q8_0`'s byte rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q2_k` | 1.8 | 205 | 0.80 | +26/-17/+8/-12/+9% | 288 | 57% |
| `q3_k` | 0.1 | 465 | 1.82 | ±4% | 234 | 47% |
| `q4_k` | 0.2 | 279 | 1.09 | ±2% | 498 | 99% |
| `q5_k` | 0.2 | 493 | 1.93 | ±7% | 349 | 70% |
| `q8_0` | 0.0 | 68 | 2.12 | ±1% | 501 | 100% |

(`q8_0`'s `B` is per 32-weight block; the last two columns of `q2_k`'s residuals
are per-cell, because its fit is meaningless - see 4. Rep 2's `B`s came out
206/467/297/490/70, so every ratio above repeats to within 7% across processes.)

The third rep, [curve-fit-rep3-release.txt](curve-fit-rep3-release.txt), is the
same test built `--release` on a quiet machine: `B` came out **207/468/283/492/68** -
within 1.5% of both debug runs. So the fit is profile-independent, which is the
strongest evidence in this file that the *ratios* are a property of the kernels
and not of the harness. (Absolute GB/s is not: the same release build under
load read 1.6-1.75x slower on every K-quant while `q8_0` barely moved - see
[Caveats](#caveats-on-the-harness).)

1. **There is no per-row cost.** `A` is 0.0-0.3 ns/row for every format except
   `q2_k`, against `B * nb` of 2.8-7.9 ns/row over nb=4-16 - the intercept is
   under 4% of a down-shape row. That deletes the structural account the section
   above offered: per-row addressing, activation staging and the tail reduction
   cannot add up to the `q3_k` deficit, and neither can lane geometry - which is
   exactly what the `ALLPAKA_Q2_NR0=1` flat result said. **Geometry tuning is
   finished as a lever, on measurement rather than on taste.** — that last
   sentence was wrong the hour it was written. `A` is flat, and that is all the
   fit says; `B` turned out to move 13-25% on `q5_k` from the same knob, and that
   change now ships. See
   [Correction #4](#correction-4-geometry-moves-the-per-block-cost-itself).
2. **`q3_k` and `q5_k` pay per block, and not for their bytes.** `q3_k` costs
   1.67x `q4_k`'s per-block time (465 against 279 ps) while carrying 24% *fewer*
   bytes per block (110 against 144); `q5_k` is 1.77x for 1.22x the bytes. The
   deficit is the unpack chain, and the fix is the inner loop, not the launch.
3. **`q4_k` reaches 99% of the machine's byte rate at nb=16 and 94% at nb=6**,
   from a completely different measurement than the one that first put it at the
   memory limit - and it also has no intercept, so nothing is left to recover in
   `matvec_q4_k_mv`. Same verdict, arrived at twice.
4. **`q2_k` is a different bug, and it is a quantization of passes not of rows.**
   Its per-weight cost walks 2.03 → 2.37 → 1.55 → 1.58 → 1.14 ps/w at nb =
   4, 6, 8, 12, 16, and the *time* at nb=6 is larger than at nb=8 (118 against
   104 µs per dispatch) despite 25% fewer bytes: the lanes address four
   super-blocks per pass, so a 6-block row pays a second pass for two blocks of
   work. This is the mechanism behind the old `q2_k [4096,1536]` control row's
   114.7 GB/s. It is **not** the 235B's problem - its `q2_k` tensors are the
   wide gate/up rows (nb=16, where `q2_k` is the cheapest format in the table at
   0.80 ps/weight of block cost) - so it only matters for models that keep
   `q2_k` on short rows.

Actionable, in order of measured size: port `q4_k`'s per-block structure into
`matvec_q3_k_mv` (25.46 GiB of the 235B is `q3_k` down projections, 1.82 against
1.09 ps/weight, and 1.94 `q4_k`-equivalents at the dispatch geometry its models
actually use); then the same for `q5_k`, which is still 1.54 `q4_k`-equivalents
*after* the geometry lever below; the `q2_k` partial-pass fix for short rows;
nothing at all for `q4_k`, `q8_0`, or any intercept knob. The "no lane knob helps"
clause did not survive - `q5_k`'s moved 13-25% - see Correction #4. **The first
item on this list is dead**: the probe written to size it found that arithmetic
is only 23% of `q3_k`'s deficit, which is Correction #5.


## Correction #4: geometry moves the per-block cost itself

**Verdict first, because it reversed while this section was being written: `q5_k`
ships at `ALLPAKA_Q5_NR0=4`.** 13% cheaper per quant block at the shape
Qwen3.6-35B-A3B's `q5_k` expert downs actually dispatch, 23-25% at a wide one, no
arm measured worse than the geometry it replaced, parity-clean at 1/2/4. What is
*not* claimed is an end-to-end win: the share arithmetic below predicts +1.8%
tok/s on that model, this machine's e2e repeatability is ±2-4%, and five
interleaved AB/BA pairs came back flat - which is what a true +1.8% looks like
here, not what a null looks like. The earlier revision of this section said
"reproduced, and it buys nothing"; at the one geometry that decided it, that
sentence was the harness talking. [feed-depth-real-shape.txt](feed-depth-real-shape.txt),
[q5-nr0-real-shape-fed32.txt](q5-nr0-real-shape-fed32.txt).

The knob is `ALLPAKA_Q{4,5}_NR0` - rows per SIMD group, so its inverse is lanes
per row. Two claims had said it was spent: `ALLPAKA_Q2_NR0=1` measured flat, and
correction 3's `A ≈ 0` (a lever that only moves `A` has nothing left to move on
these shapes, because `A` is 0.1-0.3 ns against `B * nb` of 2.8-7.9 ns). Both are
about `A`. `B` is not a hardware constant - it is issue slots per block per lane,
and halving the lanes per row doubles the blocks each lane walks back to back.

**Method, because this is the part that lied earlier.** `B`'s absolute value
drifts up to 70% with machine contention on the K-quants while `q8_0`'s holds
flat (see [Caveats](#caveats-on-the-harness)), so no number below is read across
processes. Every one is a *ratio to `q4_k` inside the same process*, from
`indexed_matvec_curve_separates_row_and_block_cost` at the fitted band and
`q5_shape_ratio_against_q4` at two other shapes. Arms alternate within and across
runs, so drift is common-mode and a one-arm pattern cannot survive it.

A second control turned out to be load-bearing. The harness commits one command
buffer per dispatch and, historically, waited on every one of them; the new
`ALLPAKA_Q5AB_PER_WAIT` lets a batch of `per_wait` commits share one
`wait_until_completed`, which keeps the queue non-empty and the SM clock up. At
the 32k-41k-row grids of the curve fit and the wide shape that changes nothing
(both repeat as well as before); at 4096 rows it is the difference between noise
and a measurement. The fourth column below uses `per_wait=32`.

`q5_k` per-block cost against `q4_k`'s, short-row band (`n_in` 1024-4096):

| `q5_k` NR0 | ratios, one per rep, oldest first | `B` ps/row/block | at the model's own dispatch (`[512,2048]` x8, feed=32) |
| --- | --- | ---: | --- |
| **4 (ships)** | 1.85, 1.75, 1.37, 1.86, 1.86 | 401 (against 492 at NR0=2) | 1.539, 1.537, 1.544 |
| 2 (was the default) | 2.37, 2.52, 2.56, 1.66, 2.46, 2.66 | 492 | 1.764, 1.770 |
| 1 | 2.27 | - | not tried |
| 8 | - | - | fails `q5_k_matches_cpu_reference`; removed from the sweep filter |

The first two reps in each row come from the sweep taken while `q3_k` was still
parameterised, which inflated every K-quant's `B` and is exactly why the ratio,
not `B`, is the number to quote. [q5-nr0-sweep.txt](q5-nr0-sweep.txt),
[q5-nr0-curvefit-pair.txt](q5-nr0-curvefit-pair.txt),
[q5-shape-ratio.txt](q5-shape-ratio.txt).

The fourth column is the one that reversed the verdict, and the reason the
previous revision of this section dismissed it is worth more than the column. At
`[512,2048]` x8 - 4096 rows, the shape those downs really issue - measured with
`per_wait=1`, the `q4_k` meter read 1579 then 652 ps/row/block across two runs of
one arm, and the `q5_k`/`q4_k` ratio came out 1.737 then 1.528 for the *same
default geometry* (NR0=2). That is not a kernel reading, it is the CPU failing to
stay ahead of the GPU: the dispatch is short enough that the queue empties between
buffers and the clock drops. With `per_wait` >= 8 the meter sits at 394-464 and
repeats to ±1.3% across eight arms, and the reading it gives is decisive: NR0=4
1.537-1.544, NR0=2 1.764-1.770. [feed-depth-real-shape.txt](feed-depth-real-shape.txt)
is the 1/8/32/128/128/32/8/1 interleaved sweep;
[q5-nr0-real-shape-fed32.txt](q5-nr0-real-shape-fed32.txt) is the 4/2/4/2/4
confirmation on top of it. The shipped default has since been re-run there: with
`q4_k`'s meter back in its 394-405 band it reads 1.543, and the four out-of-band
re-runs on a loaded machine (average 11.9-17.0) read 1.571-1.699 - see
[q5-nr0-default-recheck.txt](q5-nr0-default-recheck.txt), which is also the
evidence for the last Caveats bullet.

The same knob at a 14080-wide expert down (`[5120,14080]`, nb=55, the stablest
instrument in this file - the meter repeats to ±0.5%) is unambiguous: NR0=4
gives 1.482 / 1.478 against NR0=2's 1.937 / 1.922, i.e. `q5_k`'s `B` drops from
584 to 445 ps/row/block. **23-25%, reproduced.** The two shapes disagree about size, not direction: 13% at
4096 rows, 23-25% at 41k. No mechanism is claimed for the gap - the two shapes
have different `nb` (8 against 55), different grids, and one of them is a
synthetic width no local model has. An earlier, non-interleaved version of the
same comparison ([q5-wide-ratio.txt](q5-wide-ratio.txt)) gave 1.24-1.60 against
1.67-1.97.

The model to judge it on is Qwen3.6-35B-A3B-UD-Q4_K_M - 37 `q5_k` tensors, the
MoE expert downs, 6.36 of its 20.6 GiB. Five interleaved AB/BA pairs of
`allpaka bench --engine` at PP=480 TG=64 give 102.0 against 103.7 tok/s on the
means and 106.9 against 104.2 on the bests
([q5-e2e-ab.txt](q5-e2e-ab.txt)) - no sign at either statistic, which is exactly
what the prediction below says to expect. GLM-4.5-Air cannot speak to this lever:
its expert downs are `Q5_0` and it holds 0.34 GiB of `q5_k`, and
[the 235B](../../roadmap.md) holds 0.18 GiB across 16 tensors, so nothing this
roadmap is being closed on can regress from it. GLM being the wrong instrument is
this section's own error - the residency dump said so after two hours had gone
into a hand-rolled GGUF parse that answered the same question wrongly. Read type
shares out of the engine, not out of a script.

The prediction, stated rather than measured, is arithmetic on that same residency
dump. The model reads ~3.0 GiB of weights per token: 17.6 GiB of expert tensors
(`q4_k` gate/up + `q5_k` down) at top-k 8 of 128 = one sixteenth, so ~1.1 GiB, of
which `q5_k`'s share is 6.36/16 = 0.40 GiB; plus 1.9 GiB of `q8_0` attention, the
`q6_k` head and the one dense layer. So `q5_k` is ~13% of decode bytes, and 13%
off 13% is **+1.8% tok/s** - inside the ±2-4% this harness repeats at. Two
things keep the effect small, and only one of them is the share: `q4_k`'s meter at
that exact dispatch (nb=8, 394-405 ps/row/block) is ~360 GB/s against the ~330
decode achieves, i.e. the real loop is running within 10% of what the cheapest
big-format kernel manages on the same shape, so most of `q5_k`'s remaining issue
slots are already hidden behind bytes. Same wall correction 3 found from the
other side.

Default is 4; the knob stays so the arm can be re-run. Shipping it required
widening the indexed fallback: a SIMD group must not straddle two expert slots,
so `matvec_q5_k_mv` now drops to the block-per-lane kernel when
`n_out % NR0 != 0` rather than `% 2`. The two fallback maps became one function,
`mv_indexed_kernel`, which takes NR0 from `lanes_per_row` - the same value the
launcher passes as the LPR function constant - so no `ALLPAKA_Q*_NR0` sweep can
out-run the check. Reachability, since it decides how much this matters: down
projections cannot hit it (`decode_token` requires `hidden % 4 == 0` and
`ffn_batch` requires both widths), and every shipped model's `expert_ffn` is a
multiple of 256, so gate/up cannot hit it either - `QuantMat` only constrains
`n_in`, so a hand-made mixed-quant layout with `expert_ffn % 4 == 2` is the only
way in. Defensive, not urgent. What made it worth closing properly is that the
shape is now tested twice: `gpu::metal::tests::indexed_mv_routing_respects_rows_per_simd_group`
on the decision, and `gpu_parity::indexed_matvec_matches_cpu_reference` on the
resulting numbers through `gpu::indexed_matvec_rows`. The routing test passes at every
`ALLPAKA_Q{4,5,8}_NR0` arm, the parity test at those plus `ALLPAKA_Q2_MV=1`
(NR0 2 and 4) and with the `_mv` kernels switched off. Deleting the guard makes
the parity test red, which is the only reason to believe it: `q3_k n_out=5`
returns 2.406 against the CPU's 0.963, and at `ALLPAKA_Q2_NR0=4` so does
`q2_k n_out=6` (0.260 against -2.003). Note what the first version of that test
did instead - pass, with the guard deleted - because it is easy to write an
indexed test that cannot fail: with contiguous expert ids and one shared `x`,
slot A's rows past `n_out` are exactly slot B's first rows in memory, so the
straddle computes the right answer by accident. Non-adjacent, out-of-order ids
and a NaN pre-filled output are what make it a check. `q4_k` stays at 2 - the
same sweep made `q4_k` *slower* at 4 on the wide shapes, which is the remaining
proof that this is per-format and not a global geometry truth. And note which
number this default was chosen on: the fourth column - the model's own dispatch
shape, with the queue fed the way a decode loop feeds it - not the fitted band.

`q3_k`, negative, published because it is the other half of the same experiment.
Making `q3_k`'s NR0 sweepable did not just fail to help - the parameterised
kernel is slower than the one that was in the tree at *every* arm, including
arm 2, which is the geometry it already had:

| arm | `q3_k`/`q4_k` per-block, reps |
| --- | --- |
| NR0=2, parameterised code | 1.84, 1.89, 1.88, 1.91 |
| NR0=4, parameterised | 2.81 |
| NR0=8, parameterised | 2.11, 2.29, 2.06 |
| unparameterised (`3ef9d9c` form) | 1.59, 1.62, 1.65, 1.68 |

[q3-nr0-pair-negative.txt](q3-nr0-pair-negative.txt) is the interleaved
2/8/2/8/2/8/4/2 series. The file this sentence used to link - a quiet-machine
fit on the reverted tree with `q3_k`'s `B` at 463-464 against `q4_k`'s 293 -
never existed under that name; [curve-fit-post-revert.txt](curve-fit-post-revert.txt)
is that fit redone on the same (unparameterised) tree at load ~10-14, where the
meter is degraded and `B` reads 730 against 369. Same verdict, different regime,
and the file says so.
The parameterisation is gone from `matvec_q3_k_mv` and `ALLPAKA_Q3_NR0` no longer
exists. Do not retry it as a geometry question - `q3_k`'s deficit is the unpack
chain itself, and the fixed instrument makes that job bigger, not smaller: at the
model's own `[512,2048]` x8 dispatch with feed=32 `q3_k`/`q4_k` reads 1.86-2.01
([feed-depth-real-shape.txt](feed-depth-real-shape.txt)) against 1.59-1.68 from
the fitted band. Still an inner-loop port, still 25.46 GiB of the 235B.

One more thing this section is evidence against: the ALU-removal probe that ran
before it concluded "deleting the unpack arithmetic changes nothing" on `q3_k`
and `q5_k`. That was measured with an MSL function constant left unset, and an
unset constant does not reliably read as 0 - the control arm ran the probe body.
The conclusion is void and the probe is deleted; the mechanism question was
answered later the same day by a working one - see Correction #5.



## Correction #5: the q3_k deficit is not the unpack chain

Correction #3 ended with "the deficit is the unpack chain, and the fix is the
inner loop, not the launch", and the actionable list inherited that as `q3_k`'s
top item - port `q4_k`'s per-block structure into `matvec_q3_k_mv`. The probe
written to size that port says the port cannot work. [q3-nomath-probe.txt](q3-nomath-probe.txt)
has the runs; `matvec_q3_k_nm` is the incumbent's geometry, addresses,
activation loads and 16 weight words per block-row with the dequant arithmetic
collapsed to one add per load, selected by `ALLPAKA_Q3_PROBE=nomath` (a ceiling,
not a candidate - it does not compute a dot product).

At the shape Qwen3-235B's `q3_k` downs really use, `[4096,1536]` nb=6 x8,
12 streamed sets, feed=32, all formats in one process with `q4_k` as the meter:

| arm | `q5_k` | `q3_k` | `q6_k` |
| --- | --- | --- | --- |
| incumbent | 1.788, 1.813 | **1.809, 1.809** | 1.470, 1.442 |
| `q3_k` arithmetic removed | 1.797, 1.764 | **1.425, 1.359** | 1.387, 1.378 |

1. Removing essentially all of the arithmetic recovers 23%, not the 1.81x. The
   untouched controls repeat to +-1.5% across the arms, and the incumbent's 1.809
   agrees with the quiet-machine 1.75-1.82, so the ratio shift is the knob. Three
   quarters of the cost survives without any dequant at all.
2. It is not bytes: per byte of block, `q4_k` 5.8 ps/B and `q6_k` 5.8 - the
   format with 46% more bytes per block is at the same byte rate - while `q5_k`
   is 8.5 and `q3_k` 13.7, falling only to 10.1 with no arithmetic.
3. It is not block alignment either: the 2-aligned `q6_k` (210 B) is the
   cheapest format after `q4_k`; the 16-aligned `q5_k` (176 B) is 1.5x worse.
4. What appeared to be left was the load instruction stream: a 110-byte block
   leaves every second block 2-aligned, MSL will not do a 4-byte vector load
   from a 2-aligned address, so `q3_k` reads its 32 weight bytes per block-row as
   16 ushorts where `q4_k` reads the same span in ~4 uints. **That attribution
   was measured and did not survive - see Correction #6.**

**So the port is not pursued.** Its ceiling is the 23% the probe gave back, and a
kernel that keeps the arithmetic gets a fraction of that; against `q3_k`'s ~20%
share of 235B decode time that is 2-4% end to end, inside this machine's noise.
The layout change item 4 implied - repacking to 112-byte blocks so the quants
read as words - was DoD step (a) of this section, and it came back null the same
day: [q3-load-width-control.txt](q3-load-width-control.txt), Correction #6.

## Correction #6: the residual is not load width either - what #5 named as the cause is worth <= ~6%

Correction #5 ended by naming a mechanism - 16 ushort weight reads per block-row,
forced by the 110-byte block's 2-byte alignment, against `q4_k`'s ~4 uints. That
was inference from what the probe left in place, not a measurement. It is now a
measurement, and it says the inference was wrong:
[q3-load-width-control.txt](q3-load-width-control.txt).

`matvec_q3_k_nw` (`ALLPAKA_Q3_PROBE=nomath_word`) is the no-math probe with the
block padded to 112 bytes, which makes every block 16-aligned and lets the same
32 quant bytes and 32 high-bit bytes arrive as 8 uints instead of 16 ushorts.
Geometry, in-block addresses, activation traffic and arithmetic are otherwise
identical, so the pair isolates load width - half the weight load instructions,
at the same bytes.

| arm | `q3_k` ratio to the in-run `q4_k` |
| --- | --- |
| incumbent | 1.72, 1.76 (and 1.809, 1.809 from the quieter run) |
| no arithmetic, ushort loads | mean 1.401 over 7 runs |
| no arithmetic, uint loads on 112-byte blocks | mean 1.311 over 6 runs |

6.4% between the probe arms, against +-10% run-to-run spread within each and a
~7% standard error on the difference - and one further arm discarded because its
`q5_k` control read 2.124 outside the 1.72-1.84 band every other run holds (an
unrelated build at 412% CPU was in `ps` across it). So the width effect is small,
unresolved, and bounded above by about a tenth, which is enough to kill the
attribution: at least ~1.25x of the residual is not load width. Therefore:

- **The 112-byte repack is withdrawn on economics, not on a null.** It existed
  to buy narrower-to-wider loads; read most optimistically that is worth ~6% of
  `q3_k`'s per-block cost, against ~26 GiB of private GPU memory that cannot
  share the model's mmap - the one property the loader is built around. It
  cannot explain the residual, so it needs a mechanism, not a decision.
- **What survives from Correction #5 is the ceiling, because it does not depend
  on the attribution.** Deleting nearly all of the arithmetic returns 23% at
  *either* load width, and a correct kernel cannot do better than the bound a
  probe of itself establishes. The inner-loop port stays unpursued.
- **What `q3_k` actually pays for is back to unexplained**, with three
  exclusions now on the record instead of one: not bytes per block (it carries
  24% fewer than `q4_k` and costs 1.4-1.8x), not block alignment or load width
  (this correction, <= ~6%), not the unpack arithmetic beyond a 23% bound. Two
  candidates remain: the probes' own retained work - each no-math arm still does
  ~2 ops per element, which is comparable to a correct vectorized kernel, so
  "arithmetic" and "residual" may never separate this way - and the address
  *pattern* of a 110-byte stride across a SIMD group's four concurrent blocks.
  DoD, cheapest first: a quiet 5-rep re-run of the three arms to settle the 6.4%,
  then a probe that keeps the incumbent's addresses but issues its weight loads
  once for both rows, hoisting only the row step. Full data and both experiments
  in [q3-load-width-control.txt](q3-load-width-control.txt).

## Correction #7: three bounds on `q3_k`, and the residue is byte layout

Corrections #5 and #6 each removed one mechanism and measured what was left. The
series is now complete enough to state the arithmetic, on the 235B's own
`[4096,1536]` nb=6 x8 dispatch, streamed, feed=32, `q4_k` as the in-run meter,
controls (`q5_k` 1.81-1.83, `q6_k` 1.56-1.62) tight across six runs:

| arm | `q3_k` / `q4_k` | what the arm removes | reachable by a correct kernel? |
| --- | --- | --- | --- |
| incumbent | 1.757 | - | - |
| `matvec_q3_k_nm` | 1.315 | the dequant arithmetic, every load kept | **yes** - 25%, the ceiling for any inner-loop rewrite |
| `matvec_q3_k_nw` | 1.311 (vs 1.401 unpadded) | half the weight *load instructions*, via a 112-byte aligned block | partly - worth <= ~6%, unresolved |
| `matvec_q3_k_ns` | 1.148 | half the weight *bytes* per block, row 1 reusing row 0's words | **no** - row 1 genuinely has its own bytes |

Read as a budget: of `q3_k`'s 0.757 excess over `q4_k`, arithmetic is 0.44
(25% of the incumbent cost), load width is <= ~0.09, and the floor a correct
kernel cannot pass is ~1.32. Which is the number that matters for the decision
this file has been circling: **the unpack-chain port was never going to close
`q3_k`'s gap**, because a quarter of it is arithmetic and three quarters of the
rest is not arithmetic.

So what is the residue? Not bytes per element - `q3_k` carries 0.430 B/weight
against `q4_k`'s 0.563. Not alignment, since the 2-aligned `q6_k` is the
cheapest format after `q4_k` per byte. The `ns` arm is the hint: it improves on
`nm` by *touching half as many weight bytes*, so the cost tracks how many
distinct bytes a lane pulls per block, and `q3_k`'s 110 bytes live in three
separate planes (32 B of 1-bit masks at offset 0, 64 B of 2-bit quants at 32,
12 B of scales at 96) that a lane reaches with three address streams and a
16-byte-sector footprint wider than its information content. `q4_k` has two
planes and `q6_k` three but 210 bytes of them.

That turns the repack question from "pad the block" (measured dead) into a
specific, testable-before-paying claim: interleave the mask plane into the quant
plane (e.g. per 16 quants, 4 B of mask adjacent to the 16 bytes that hold their
2 bits and the scale nibble) so a lane's per-block read is one contiguous run.
The cheap version of that experiment is bench-side: repack a copy of the region
in the harness, run a probe kernel addressed for the new layout, and compare
against `nm` at the same shape. `q3_k` is 25.46 GiB of the 235B and ~20% of its
decode time, so 1.32 -> 1.15 would be worth ~4-5% on that model - which is only
worth the loader's memory cost if the bench-side experiment says it is. DoD:
(a) the interleaved bench probe, 5 reps, controls in band; (b) if `q3_k`'s ratio
does not fall below ~1.20, the layout idea dies with this table as its proof;
(c) only on a pass, cost the repack against the residency dump.

## Correction #8: run locality measured - it costs 2-4%, so the repack dies with its memory question unasked

Correction #7's DoD was (a) a bench-side interleaved probe at 5 reps with
controls in band, (b) a kill at ratio >= ~1.20, (c) only on a pass, the residency
cost. It ran, and it kills. Artifact with the full design and raw log:
[q3-locality-arms.txt](q3-locality-arms.txt).

Two new arms, each differing from an existing arm in exactly one property - the
same 32 weight bytes per lane-row, the same load count and width, from ONE
contiguous 32-byte run instead of two runs 32 bytes apart:

| pair | ratio (mean of in-band runs, sd in brackets) | locality effect |
| --- | --- | --- |
| `matvec_q3_k_nm` | 1.383 [0.010] | - |
| `matvec_q3_k_ic` (`contig`) | 1.410 [0.016] | **+2.0%** |
| `matvec_q3_k_nw` | 1.319 [0.010] | - |
| `matvec_q3_k_icw` (`contig_word`) | 1.372 [0.024] | **+4.0%** |

Incumbent 1.808, `matvec_q3_k_ns` 1.198. 24 of 30 runs survived the control gate
(`q5_k` 1.72-1.84, `q6_k` 1.30-1.53 against the in-run `q4_k` meter), the arm
order was rotated by two slots per rep, and the surviving within-arm spread is
0.010-0.024 against effects of 0.027-0.053 - small but consistently signed, which
is what settles the `nomath_word` number correction #6 left unresolved: load
width is worth **-4.6%** (`nm` 1.383 -> `nw` 1.319), not the 6.4% of one noisy
pair.

Contiguity goes the *wrong* way, by about as much as width buys. So the residue
correction #7 attributed to "how the three planes are laid out" is not layout
either: a lane's address stream is already about as good as a permutation can
make it, and concentrating it costs (the likely mechanism is that the four lane
classes then overlap in fewer 16-byte sectors, so the same instruction issues
more conflict work per warp). The repack's price - ~29% more weight bytes, and a
private copy the loader's zero-copy mmap cannot give - no longer has any gain to
be weighed against. DoD item (c) is therefore not owed at all.

What the series leaves on `q3_k`, stated once so it stops being reopened:

| mechanism | measured effect on `q3_k`'s per-block cost | a correct kernel can use it? |
| --- | --- | --- |
| dequant arithmetic | -23% (`nm`) | yes, and it is the incumbent's own unpack |
| load width (ushort -> uint) | -4.6% (`nw`) | only with a padded block: +2 bytes/block and a private repacked copy |
| run locality (2 runs -> 1) | +2 to +4% (`ic`, `icw`) | no - it is a loss |
| weight bytes per block (row sharing) | -13% (`ns`) | no - row 1 has its own weights |
| load schedule (16 loads hoisted ahead of consumers) | -0.26% (`mlp`, correction #9) | no - there is no effect to ship |

All five are bounded, and the one still worth anything (padding, -4.6%) costs
a model-memory decision to buy a quarter of what `q3_k`'s gap is - against 25.46
GiB of `q3_k` in the 235B, the prediction is well inside the noise band that has
already swallowed three candidate levers on this model. It is recorded as
available, not as pending.

## Correction #9: the fifth mechanism - bytes in flight per lane - moves nothing

Four mechanisms bounded, one left named and untested. The paragraph under the
shape table offered the fork explicitly: "if it does not move, it is latency and
only more bytes in flight per lane helps". `nm` moved the arithmetic, `nw` the
width, `ic` the locality, `ns` the count - none of them touched *when* the loads
are issued. `matvec_q3_k_mv` reads its sixteen ushort weight words per block-row
interleaved with the FMAs that consume them, with `yl[32]` already holding
thirty-two activation registers, so the incumbent plausibly has only a couple of
weight loads outstanding at a time. [`q3-mlp-arms.txt`](q3-mlp-arms.txt) adds
`matvec_q3_k_mlp` (`ALLPAKA_Q3_PROBE=mlp`): the incumbent's bytes, addresses,
widths, arithmetic, geometry and reduction, with all sixteen hoisted into locals
ahead of both inner loops. One property changes, and unlike the other four arms
this one is a real dot product - a schedule win would have shipped as-is, with no
repack and no memory cost.

6 reps x 4 arms on the model's own `[4096,1536]` x8 dispatch, 12 rotating sets,
feed=32, order rotated one slot per rep, all 24 runs kept:

| arm | n | mean ratio to in-run `q4_k` | sd | range | vs incumbent |
| --- | --- | ---: | ---: | --- | --- |
| incumbent | 12 | 1.767 | 0.055 | 1.683-1.882 | - |
| `mlp` | 6 | 1.763 | 0.045 | 1.684-1.808 | **-0.26%** |
| `nm` | 6 | 1.358 | 0.019 | 1.332-1.384 | -23.1% |

Paired per rep: -2.9%, +0.6%, +5.6%, +4.4%, -10.5%, -0.3% - four signs in six,
and a spread five times the effect. The `nm` column is why the null means
something: in the same processes at the same loads the harness resolves a real
`q3_k` difference to 23.1% with sd 0.019. The load rose 2.4x across the sequence
(the `q4_k` meter went 310 -> 747 ps/row/block as the desktop filled up) and the
ratios did not care - rep 6's `mlp` run is an absolute 1269 ps and a 1.779,
indistinguishable from rep 1's 593 ps and 1.739.

So the schedule is not it either: either the compiler already issues those loads
ahead of their consumers, or outstanding-load count is not what the block-row
waits on. Both readings close the same lever - there is no schedule inside this
kernel to win. `q3_k`'s cost is 0.44 of arithmetic plus a per-block residue that
survives reordering, widening, concentrating and pipelining the identical loads,
and the only arm that ever moved it materially was the one that reads *fewer
weight bytes*, which the format's row ownership forbids for real rows.

That is five of five mechanisms bounded, and it is the end of in-kernel `q3_k`
work: the table in correction #8 gains a row, and nothing further on this kernel
is pending. `mlp` was never promoted, so it was never gated on numerical
correctness - a -0.26% candidate does not need to be proven right, only recorded
as not worth shipping; the gate if it returns is
`gpu_parity::indexed_matvec_matches_cpu_reference`.

## Correction #10: the streamed table's absolute column was a contended window, and the two missing formats are now measured

The 7-row streamed table under "Consequence (revised)" came from `rep 2` of
[indexed-controls.txt](indexed-controls.txt), chosen because its best/median
spread was 1.00x. That check passed and the window was still contended: its
`q8_0` meter read 400.2 GB/s against 495.6-497.0 measured twice in one process
on 2026-09-24 18:20 UTC
([../2026-09-24-per-token-byte-census/q6k-q5o-streamed-rates.txt](../2026-09-24-per-token-byte-census/q6k-q5o-streamed-rates.txt)),
which is 19% of the machine, not 1% of noise. So a 1.00x spread proves a run is
self-consistent, not that it is unshared; the meter also has to sit near the
unthrottled ceiling, and from now on that is the second gate.

The same arms, one process, one window, spread 1.00-1.01x, two `q8_0` arms
agreeing to 0.3%:

| Format | shape | streamed GB/s | Gweight/s | ps per weight | GB/s / in-run `q8_0` |
| --- | --- | ---: | ---: | ---: | ---: |
| `q2_k` | [1536,4096] (real gate/up) | 257.7 | 785 | 1.27 | 0.520 |
| `q3_k` | [4096,1536] (real down) | 222.1 | 517 | 1.93 | 0.448 |
| `q4_k` | [4096,1536] | 472.2 | 840 | 1.19 | 0.953 |
| `q5_k` | [4096,1536] | 345.2 | 502 | 1.99 | 0.697 |
| `q8_0` | [4096,1536] (meter) | 495.6 | 466 | 2.15 | 1.000 |
| `q2_k control` | [4096,1536] | 138.7 | 423 | 2.36 | 0.280 |
| `q4_k control` | [1536,4096] | 468.6 | 833 | 1.20 | 0.946 |
| `q6_k` | [4096,1536] | 471.7 | 575 | 1.74 | 0.952 |
| `q6_k` | [1536,4096] | 459.7 | 560 | 1.79 | 0.928 |
| `q5_0` | [4096,1536] | 396.6 | 577 | 1.73 | 0.800 |
| `q5_0 (GLM's own)` | [4096,1408] | 383.6 | 558 | 1.79 | 0.774 |
| `q4_k (30B's own)` | [768,2048] | 386.7 | 688 | 1.45 | 0.780 |
| `q6_k (30B's own)` | [2048,768] | 363.9 | 444 | 2.25 | 0.734 |
| `q4_k (GLM's own)` | [1408,4096] | 464.6 | 826 | 1.21 | 0.938 |
| `q8_0 (GLM's own)` | [4096,1408] | 497.0 | 468 | 2.14 | 1.003 |

Rows 8-15 are new: `q6_k` and `q5_0` had no streamed rate anywhere in the repo
(`q6_k` carries 27.3% of the 30B's decode weight bytes), and rows 11-15 put each
format at the geometry the model that owns it dispatches instead of the 235B's.
What follows:

1. **`q4_k` is closer to the wall than the old table said.** It reads 0.953 of
   `q8_0`'s byte rate here against 0.886 there, which strengthens the existing
   "no kernel change recovers decode time" conclusion rather than weakening it.
2. **Cross-format ratios are contention-biased in a knowable direction.** `q8_0`
   uses the most bandwidth, so it loses the most to a co-runner, so a contended
   window *flatters the slow formats*: the `q3_k` per-weight penalty against
   `q4_k` was 2.43/1.59 = 1.53 in the old window and is 1.93/1.19 = **1.63**
   here. Nothing in this file's chain of corrections turns on that (the residue
   is per-block and layout-shaped, not arithmetic), but a per-weight cost quoted
   from one window carries ~6% on the ratio and ~20% on the absolute.
3. **`q6_k` is a full-rate format** - 471.7/459.7 GB/s at the two expert shapes,
   statistically tied with `q4_k` - so its 1.46x-bytes-per-weight cost over
   `q4_k` is entirely a byte cost, and its 1.74 ps/weight sits between `q4_k`
   (1.19) and `q3_k` (1.93). Pushing more tensors toward `q6_k` buys
   per-weight *speed* against `q3_k`/`q5_k` and pays only in GiB.
4. **`q5_0` is not `q8_0`.** 396.6 against 495.6 GB/s, 0.800x, so the
   `q5_0_mv` port's ledger input (which assumed 400) survives - by luck, and now
   as a measurement. At 32-element blocks it is a wide-row format and still pays
   1.73 ps/weight against `q8_0`'s 2.15: a good format per weight, a mediocre
   one per byte.
5. **A dimension at 768 costs 18-23% of the byte rate.** `q4_k` at the 30B's
   [768,2048] runs 386.7 against 472.2 at [4096,1536], and `q6_k` at [2048,768]
   runs 363.9 against 471.7. Swapping `n_out`/`n_in` costs ~1% and
   4096->1408 costs 3%, so this is not a smooth size effect and the mechanism is
   unidentified here (3 blocks per row at `n_in` 768 against 6-16 elsewhere is
   the candidate, untested). It is the largest shape sensitivity in the table
   after `q2_k`'s, and it is why the byte ledger in
   [../2026-09-24-per-token-byte-census/README.md](../2026-09-24-per-token-byte-census/README.md)
   now builds each model's ms/token from its own shapes.
   **Superseded in part by
   [correction #11](#correction-11-the-768-claim-splits---it-is-three-blocks-per-row-and-it-is-worth-4-of-the-30bs-wall)** -
   the "3 blocks per row" candidate is now tested and true, but it covers only
   the `q6_k` row of this item, not the `q4_k` one, and the "18-23%" headline is
   two effects that the table above cannot separate.

## Correction #11: the 768 claim splits - it is three blocks per row, and it is worth ~4% of the 30B's wall

Item 5 above named one candidate ("3 blocks per row at `n_in` 768") and left it
untested because the table has no fixed-bytes comparison in it: its narrow rows
are 6144 blocks per matrix (49152 across the eight routed slots) and its
reference row 24576 (196608), so every absolute
difference there is shape and dispatch size at once. The instrument to separate
them is [../2026-09-25-narrow-shape-isomer-ladders/results.txt](../2026-09-25-narrow-shape-isomer-ladders/results.txt)
(`gpu_ffnbench.rs::narrow_dim_isomer_ladders_separate_rows_from_blocks`): five
ladders whose members all dispatch the same block count - so the same bytes per
dispatch, the same footprint, the same `n_out * (n_in/256)` - while `n_out`
varies 4-5x and `nb = n_in/256` runs 2-16. Ladder 1 (49152 blocks per dispatch)
is the 30B's own size: it contains both of its rows, `q4_k` [768,2048] at nb=8
and `q6_k` [2048,768] at nb=3, plus the nb=4 and nb=6 references. Every cell
below is a per-pass paired ratio against the nb=4 member of the *same* ladder,
median over passes, because the pass-to-pass rate decay inside one run reaches
0.41 and cross-ladder ratios disagree 22-58% between two windows that agree to
1.002 inside a ladder (that file's section 8).

1. **The candidate is real, and it is the only shape cost in the family.** At
   equal bytes, nb=3 rows burn more time per block than nb=4 rows, and at the
   30B's own dispatch size the number is tight across every window in the folder
   (§10, 12 window x feed cells each): `q4_k` [2048,768] 1.265 (the 12 cells
   span 1.210-1.287 outside the one deliberately-slowed NR0=1 arm), `q6_k`
   [2048,768] **1.224 [1.188,1.239]**. nb=6 falls to 1.059/0.997 and nb=8 to
   0.994/0.999, so the deficit belongs to *three blocks per row*, not to a 768
   dimension: `q8_0` at the identical 768-wide rows reads 0.998 [0.983,1.030]
   (its rows hold 24 of its 32-element blocks, which is why it is a control for
   the memory shape and not for the block count). At 2x and 4x the bytes per
   dispatch the same pair reads 1.29-1.32, so the hole does not close as the
   dispatch grows.
2. **The mechanism that would have made it useful is wrong.** The block walk in
   `matvec_q4_k_mv` steps in four 8-lane slots, so a 3-block row idles a quarter
   of the group and a period-4 phase model predicts 4·ceil(nb/4)/nb = 1.333 at
   nb=3 *and the same 1.333 at nb=6*. Measured: 1.27 at nb=3, 1.05 at nb=6. The
   model explains roughly a quarter of the nb=3 hole and nothing else, so there
   is no sawtooth to tune against, and the sweep confirms the rest of the
   candidate list is empty: rows per group `ALLPAKA_Q4_NR0` 1/2/4 → 1.495/1.253/1.220
   (NR0=1 is the slowest arm, and NR0=4's -2.6% is inside this instrument's
   noise band, so the shipped 2 stands), `ALLPAKA_MV_TG`
   64/256 → 1.274/1.234, and visit position forward/reversed → 1.210/1.210.
   No knob moves nb=3 below ~1.2, which is why this is a negative result.
3. **It prices at ~4%, not 14%.** Of the two rows item 5 called narrow, only
   `q6_k` [2048,768] has nb=3; `q4_k` [768,2048] has nb=8 and is inside a few
   percent of its own ladder's reference. So the recoverable share is the down
   projection's 27.3% of the 30B's decode bytes, which the rebaselined ledger
   ([../2026-09-24-per-token-byte-census/time-ledger.txt](../2026-09-24-per-token-byte-census/time-ledger.txt)
   §4) puts at 1.44 ms of 6.70 ms/token: removing a 1.22x cost there is
   1.44·(1 − 1/1.22) ≈ 0.26 ms/token ≈ **~4% tok/s**, which sits at or under
   this machine's end-to-end noise band. The 0.95 ms / +14% ceiling quoted in
   [../2026-09-25-narrow-shape-axes/results.txt](../2026-09-25-narrow-shape-axes/results.txt)
   assumes both rows are shape victims; the paired data says only one is.
4. **What is left of the 18-23% is the axis pairing cannot touch.** `q4_k`
   [768,2048] really does read 386.7 against 472.2 at [4096,1536] in one window,
   and really is ~1.0 against its own ladder's nb=4 - so that residual is
   ladder-1-against-ladder-3, i.e. blocks per dispatch (49152 against 196608),
   and section 8 shows cross-ladder ratios are not a statistic on this machine.
   It is not nothing: nb=3 paired cost goes 0.96 → 1.27 → 1.30 → 1.32 across
   24576 → 49152 → 98304 → 196608 blocks per dispatch, and it needs two
   processes whose absolute `q8_0` meters match within a few percent - the gate
   the axes file already sets and this machine has not yet met.

Reopen DoD, either arm: (a) a `matvec_q4_k_mv` / `matvec_q6_k_mv` arm that
changes the fixed four-slot block step for 3-block rows - a two-phase walk with
a different lane split, not a knob value - and must bring the ladder-1 nb=3
paired cost from ~1.21 to below 1.05 with
`gpu_parity::indexed_matvec_matches_cpu_reference` clean; (b) one window where
two processes' `q8_0` meters agree within 3%, which is the only admissible way
to price the blocks-per-dispatch axis. Arm (b) is a ~16 s run, not a design
task; arm (a) is the one that would have to beat ~4% end to end before shipping.

## Superseded consequence: MoE decode is dequant-bound, not DRAM-bound

At the decode geometry `q8_0` moves 454–476 GB/s — most of the ~546 GB/s the
M4 Max can read — while the K-quants sit at 120–200 GB/s on the same memory
path. Q8_0's dequant is one multiply per weight; Q2/Q3/Q5's is a chain of
shifts, masks and selects. So the remaining 235B gap (0.72× llama.cpp,
`q2_k`/`q3_k` experts) is not reachable by dispatch folding, lane counts or
launch tricks, and it is not a bandwidth ceiling. It is arithmetic per weight
byte, which is where any next attempt has to start.

*(Kept because it is the negative result of the cached-geometry run; the
"Correction #2" section above is what it measured. Q4_K is the part that did not
survive.)*

Earlier numbers in this file's own history were taken in the boosted-burst
regime (a ~19 ms window, no warm-up) and read 1.5–2× higher than sustained;
`timed()` now burns ~1 s and takes best-of-5. Absolute end-to-end tok/s in
`q3-235b-ab.txt` (9.3–10.5 decode) is 2× under the recorded 20.0 for the same
model because the desktop was busy — the pairs are only readable against each
other, not against `../matrix-2026-09-12.md`.

## Caveats on the harness

- `mm_shaped_matmuls_report_effective_bandwidth` in the same binary reported
  0.3 GB/s when the two tests ran concurrently and 28.8 GB/s serially: the GPU
  clock window is global, so always pass `--test-threads=1`.
- The `q2_k one dispatch` probe (287 MB in a single call) tracks the 139-matrix
  round closely, which is what ruled out per-dispatch launch cost as the flat
  shape's limiter.
- The tile kernel is not a decode alternative at these shapes: `mm q2_k` at
  m = 32 over 63 streamed matrices reads 33.5–34.1 GB/s (3.27 Tel/s touched
  weights) against the matvec's 256 GB/s. It buys arithmetic intensity it has
  no use for at m = 1 and gives up byte rate.
- The streamed table is a debug-profile run (`cargo test -q`); so is every
  number in the sweep log. The metric is the GPU's own clock, so the CPU-side
  encode cost shows up as gaps between buffers rather than in the number - but
  **that immunity does not hold on a contended machine**: the `--release` re-run
  of the curve test ([curve-fit-release-contended.txt](curve-fit-release-contended.txt),
  taken while a stray `rustc` had load average at 47) read *1.6-1.75x slower on
  every K-quant* than the debug runs while `q8_0` held 467 GB/s. Sparse
  submissions let the SM clock fall between buffers, and only the issue-bound
  kernels feel that; `q8_0`, being DRAM-bound, does not. Treat that log as void
  and do not compare the two profiles on this machine at all.
- **`B` is a property of the dispatch, not only of the kernel.** The curve test
  fixes `n_out = 4096` at x8 experts, so it measures 32768 rows per dispatch; a
  real Qwen3.6-35B-A3B expert-down dispatch is `[512,2048]` x8 = 4096 rows. At
  the fitted band `q4_k`'s `B` is 293-297 ps/row/block, and at the real one 394-464
  with the queue fed (652-1579 unfed, where the reading is the harness's) -
  1.3-1.6x higher for the same kernel because the grid is a fifth the size. Read
  the fit's `B` values as *within-shape* comparisons, which is all any conclusion
  in this file uses them for, and re-measure before spending one.
- **An unset MSL function constant does not reliably read as 0.** A probe arm
  controlled by a `constant int` that was never given a value ran the probe body
  as its own control and reported a clean, wrong result. Pin every arm constant,
  even the zero arm. (Correction #4's last paragraph.)
- **Feed depth, not grid size, is what makes a small dispatch unreadable.**
  Every number published here before 2026-09-24 waited on every commit
  (`per_wait=1`). At 32k-41k-row dispatches that is invisible - the curve fit
  repeats across processes and the wide-shape ratio to ±0.5%. At `[512,2048]` x8
  (4096 rows) the CPU cannot stay ahead of the GPU, the queue drains between
  buffers, and one arm's `q4_k` meter reads 1579 then 652 ps/row/block. With
  `ALLPAKA_Q5AB_PER_WAIT` >= 8 it reads 394-464 and repeats to ±1.3% - and the
  reading it gives reversed Correction #4's verdict. The rule was right (check the
  meter's own repeat before believing its ratio); the diagnosis attached to it
  ("small grids cannot be measured") was wrong, and cost a 13% win a day in
  limbo.
- **An in-run ratio is only as good as the meter's absolute reading.** Five
  re-runs of the shipped `q5_k` default at one shape spanned 1.543-1.699 while
  `q4_k`'s own `B` spanned 397-544 ps/row/block; the single run whose meter landed
  in the recorded band (394-405) is the one that reproduces the quiet-window
  number, and at load 15 `q3_k`'s ratio moved 27% against `q5_k`'s 10%.
  Normalizing against `q4_k` in one process removes the common-mode part of drift
  only, so the gate is the meter's *absolute* value against its recorded band -
  not merely its within-run repeat.
  [q5-nr0-default-recheck.txt](q5-nr0-default-recheck.txt).
- The geometry sweep driver is [disc-sweep.sh](disc-sweep.sh) next to this file
  (12 configs x 2 passes over the streamed shapes); its output is
  [streamed-sweep.txt](streamed-sweep.txt). Only the first two rows of its pass
  A were taken before the machine stopped being ours.

## Reproducing the two reversals

```sh
# Correction #3: the A/B fit. ~25 s, and readable on a busy machine because
# every ratio comes out of one process. curve-fit-probe-run0.txt is the first
# 4-point version of the same test, before 2048/3072 were added.
cargo test -q -p allpaka-backend --test gpu_ffnbench -- --ignored --nocapture \
  --test-threads=1 indexed_matvec_curve

# Correction #4: one shape, three formats, q4_k as the in-run meter. Shape,
# rotation and feed have to be chosen together - raise sets until the working set
# streams and per_wait until the meter repeats (see Caveats). q5_k ships at
# NR0=4, so ask for the old default explicitly.
# The model's own expert-down shape (this is the column that decided the default):
ALLPAKA_Q5AB_N_OUT=512 ALLPAKA_Q5AB_N_IN=2048 ALLPAKA_Q5AB_SETS=60 \
  ALLPAKA_Q5AB_PER_WAIT=32 ALLPAKA_Q5_NR0=2 \
  cargo test -q -p allpaka-backend --test gpu_ffnbench \
  -- --ignored --nocapture --test-threads=1 q5_shape_ratio
# and the wide, stablest version:
ALLPAKA_Q5AB_N_OUT=5120 ALLPAKA_Q5AB_N_IN=14080 ALLPAKA_Q5AB_SETS=2 \
  ALLPAKA_Q5_NR0=2 cargo test -q -p allpaka-backend --test gpu_ffnbench \
  -- --ignored --nocapture --test-threads=1 q5_shape_ratio
# and the end-to-end half. `allpaka bench` prints the model's per-type weight
# residency on every run - read the q5_k share there before picking a model.
for pair in "4 2" "2 4" "4 2" "2 4" "4 2"; do for n in $pair; do
  ALLPAKA_Q5_NR0=$n ALLPAKA_BENCH_SKIP_MTP=1 ALLPAKA_BENCH_PP=480 \
  ALLPAKA_BENCH_TG=64 ALLPAKA_PROFILE=max-performance ./target/release/allpaka \
    bench --engine models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | grep -E 'decode +64|Q5K'
  sleep 2
done; done
# q3_k: the port is the default and wins indexed; =0 picks the word-load kernel.
for i in 1 2 3; do for v in 1 0; do printf "Q3_MV=%s | " $v
  ALLPAKA_Q3_MV=$v cargo test -p allpaka-backend --test gpu_ffnbench \
    -- --ignored --nocapture --test-threads=1 indexed_matvecs | grep 'q3_k'
done; done

# q5_k: needs two builds - the incumbent body against the vectorised one, then
# the same command grepping 'q5_k'. The incumbent won both samples. Note that
# both were measured in the cached geometry; re-check before acting on them.
```

The end-to-end half of the q3 decision is `q3-235b-ab.txt` (arm A is the port,
arm B the word-load kernel), produced by running `allpaka bench --engine
models/qwen3-235b-a22b-instruct-2507-Q2_K_XL.gguf` with `ALLPAKA_BENCH_PP=480
ALLPAKA_BENCH_TG=32` alternating `ALLPAKA_Q3_MV=1` and `=0`;
`235b-run-A-full.txt` is one run's full stdout, including the per-format weight
residency dump that shows why q3_k matters there (79 tensors, 25.46 GiB).

