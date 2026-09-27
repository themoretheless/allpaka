# GLM decode knobs (MV_ID, SHARED_STAGGER, Q5_0_MV): not measurable on this machine today

Three decode-path knobs on GLM-4.5-Air were pending an end-to-end verdict:
`ALLPAKA_MV_ID` (shipped ON under an unmet ship rule), `ALLPAKA_SHARED_STAGGER`
(shipped OFF, never measured), and `ALLPAKA_Q5_0_MV` (shipped ON today on micro
evidence). This directory holds the attempt, the raw numbers, and the reason the
verdict did not happen.

## Regime

`scripts/glm-ab-stagger.sh` first (single-shot arms, `PP=480 TG=32`), then a
hand-rolled interleaved loop over all five arms, `PP=480 TG=64`,
`SKIP_MTP=1`, `max-performance`, 24 runs across four rounds. Load average 10-13
with another agent session compiling throughout, which is the whole problem.

## Reference point from the script run

    llama.cpp (llama-bench -r 1, pp480/tg32): pp 381.6, tg 42.89 tok/s
    allpaka mvid-on  tg32: 38.8        mvid-off: 38.8        stagger-on: 39.7

Consistent with the recorded 2026-09-12 cool baseline (39.0 / 45.5 = 0.86x):
GLM decode is ~0.86-0.90x llama on a mid-load machine, and MV_ID is not what
separates the two (on and off read the identical 38.8).

## The interleaved loop, decode tok/s (round-robin over arms, 4 rounds)

    arm          r1     r2    r3    r4
    MV_ID on    32.9   20.0    8.0  16.5
    MV_ID off   33.6   20.6   15.4  22.0
    STAGGER on  34.2   18.0   17.8  19.6
    Q5_0 MV=0   29.9   12.2   16.7  17.1
    Q5_0 MV=1   25.0   15.9   16.2  18.3

Absolute rates move 4x inside the sequence (32.9 -> 8.0 -> 22.0 on adjacent
runs of near-identical configurations), and that alone would be enough to
discard the table. What makes it worth writing down is the *second* flaw, which
is fixable and which the loop walked straight into: every round visited the arms
in the same order, so any monotone drift across a round - page cache warming,
another session finishing a compile, thermal ramp - lands entirely on the later
arm. MV_ID-off beat MV_ID-on in 4/4 pairs (+2.1%, +3.0%, +92.5%, +33.3%) and
off is always second in the sequence. That is not a threadgroup layout halving a
token's time; it is position. `q5_0`'s pair, also off-then-on, went -16%, +30%,
-3%, +7% - no order signal there, which is what a real ~0-10% effect looks like
buried in this much noise.

So the design requirement for the re-run is not just "fewer interruptions": arms
must alternate order round to round (AB, BA, AB, ...), which is exactly what
`allpaka airbug` already does and what a hand-rolled env loop does not.

## Verdict

No conclusion is drawn from these 24 runs, and none is recorded as a result:
the repo invariant is that a conclusion needs a paired design whose noise floor
sits below the effect, and here the noise floor is ~4x the effect being sought.
What survives is the script's single-shot triple (38.8 / 38.8 / 39.7 against
llama's 42.89), which says the GLM gap is not MV_ID's and is not the shared
expert's schedule either - both knobs are worth ~0-2% at the resolution
available, against an 11% gap.

To close the three gates properly, in this order:

1. Use `allpaka airbug` (`--pp 480 --tg 64 --repeats 5`), which does AB/BA
   process pairs with a discarded warmup and inter-engine cooldown, rather than
   hand-rolled env loops - the loop above is the instrument that failed, not the
   knobs.
2. Require sustained load < 5 for the whole sequence; on this shared worktree
   that is a scheduling decision, not a waiting one.
2b. Alternate the arm order between rounds. A fixed order turns any drift into a
   fake effect, and the drift here was up to 92% - larger than anything the
   knobs under test could plausibly produce.
3. Then, and only then: MV_ID's ship rule as written ("keep ON if cool GLM
   decode >= llama") should be restated as on-vs-off delta plus the indexed
   micro rate, because it currently asks a knob to explain a gap between two
   engines that the knob does not cause.

## Files

`full.txt` is the whole single-shot script run in one file (mvbench, the three
allpaka arms, llama), and `allpaka-mvid-{on,off}.txt`, `allpaka-stagger-on.txt`,
`llama.txt`, `mvbench-on.txt`, `metal.txt` are its parts, copied here because
`.airbug-bench/` is gitignored. `.txt`, not `.log`: `*.log` is gitignored too, so <!-- audit:ghost -->
a cited "artifact" with that extension would exist only on this machine (see the
invariant in [../../roadmap.md](../../roadmap.md)). The 20 interleaved runs of
the five-arm loop are the table above; the loop's own output is not saved - it is
the flawed instrument (fixed arm order, see the diagnosis below), and `airbug`
supersedes it.

## Combined reading that does survive

Pooling the `q5_0` arm with the seven-pair run in
[../2026-09-24-indexed-matvec/q5_0-mv-port.txt](../2026-09-24-indexed-matvec/q5_0-mv-port.txt):
11 interleaved pairs, mean paired delta **+7.2%**, median **+7.0%**, per-pair
spread -16% to +35%. Direction consistent with the +6-7% prediction and with
the 2-3x single-process micro measurement; still not 2-sigma, and now the
reason is documented twice over - contention and, worse, a fixed arm order.
