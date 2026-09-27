# The RFUSE / RMT / RTOPK verdicts describe arms the default policy never encodes

Captured 2026-09-25 21:31 UTC. **No GPU was used and nothing was built for
this file** - every claim below is a `grep` plus one line of policy resolution,
so it is checkable on a busy machine and while another benchmark holds the GPU.

## The claim being audited

`docs/decode-opts.md`, "Measured dead ends (2026-09-12, cool machine)":

> **`ALLPAKA_RFUSE=1` under default `normflag`:** −20–27% decode (GLM ~31 vs
> ~39; 30B ~79 vs ~109) despite fewer dispatches.

That sentence asserts a cost for an env variable *while `normflag` is at its
default*. The question is therefore not "was the number noisy" (the 09-24
section of the same file already disowns that era's meter) but "did the two
arms run different code at all".

## What the greps show

Three reads, three use sites, no more (`crates/allpaka-backend/src/gpu/metal.rs`
at `dcdc131`; `8a16214` has the identical structure ~1100 lines earlier, so the
line numbers below are the current tree's and the check is the pattern, not the
number):

```
16011:    *R.get_or_init(|| std::env::var("ALLPAKA_RFUSE").is_ok_and(|v| v == "1"))   # fn rfuse()
16001:    *R.get_or_init(|| std::env::var("ALLPAKA_RTOPK").is_ok_and(|v| v == "1"))   # fn rtopk_fused()
16023:    *R.get_or_init(|| std::env::var("ALLPAKA_RMT").is_ok_and(|v| v == "1"))     # fn rmt()

11910:                !refs.normflag && (rmt() || rfuse()) && matches!(&refs.ffn, FfnRefs::Moe { .. });
11993:                            let mt = rmt();          # inside `if moe_plain`
12042:                        } else if !refs.normflag && rtopk_fused() {
```

Reproduce with:

```
grep -rn --include='*.rs' 'ALLPAKA_RFUSE\|ALLPAKA_RMT\|ALLPAKA_RTOPK' crates/
grep -rn --include='*.rs' 'rfuse()\|rmt()\|rtopk_fused()' crates/allpaka-backend/src/gpu/metal.rs
```

`rfuse()` is called exactly once, in the initializer of

```rust
let moe_plain =
    !refs.normflag && (rmt() || rfuse()) && matches!(&refs.ffn, FfnRefs::Moe { .. });
```

so with `refs.normflag == true` the whole `&&` chain is false regardless of the
env, `moe_plain` is false, and neither the `resnorm_router` nor the
`resnorm_router_mt` dispatch is ever encoded. `rmt()`'s second read at 11993 is
inside that same `if moe_plain` arm. `rtopk_fused()` is gated on
`!refs.normflag` directly. All three envs are read once into a `OnceLock`, so
there is no path where the value reaches the encoder some other way.

## What `normflag` is under each profile

`crates/allpaka-backend/src/profile.rs`, `RuntimeProfile::resolve` (plus the
`ALLPAKA_NORMFLAG` override applied after it, and `RuntimePolicy::default` in
`runtime.rs`):

| profile | `normflag` | are RFUSE/RMT/RTOPK reachable? |
| --- | --- | --- |
| `auto`, `balanced`, unset (default policy) | **true** | no |
| `max-performance` (what airbug exports if unset) | **true** | no |
| `safe` | false | yes |
| `deterministic` | false | yes |
| any profile + `ALLPAKA_NORMFLAG=0` | false | yes |

## Consequence

Under every default configuration - which is what "under default `normflag`"
says - `ALLPAKA_RFUSE=1` selects nothing. The two arms encode the same command
buffers, so the −20–27% recorded on 2026-09-12 is not that knob's cost. It is
one arm measured twice by the meter this file later describes as reading "178
against 306 GB/s minutes apart".

The two neighbours come out differently, and the difference is the useful part:

- **RTOPK's number is structurally sound in a way RFUSE's is not.** Its verdict
  ("113.8 vs 79.2 tok/s", with GPU executing 265 vs 388 ms over 32 tokens)
  records two arms that *differed*, because the fused kernel is only reachable
  under `!normflag` - a `normflag`-ON pair would have read the same rate twice.
  So RTOPK was measured with at least its fused side under `!normflag`; the
  comment never says so, and if the 113.8 side came from a `normflag`-ON run
  the pair is confounded with the policy switch itself - which is B3 again.
- **RMT's "neutral" cannot be rescued that way.** Like RFUSE it is read only
  inside `moe_plain`, so a normflag-ON RMT test compares identical encodes; its
  comment's own explanation (saved drains paid back by the device-scope fence)
  is a mechanism reading, not a measurement.

The row also contains the correction that makes this sharper: "earlier advice
to default-ON RFUSE + drop the `!normflag` gate is wrong on this machine". The
gate is exactly what makes the knob a no-op, so the advice it rejects and the
verdict that replaced it cannot both be about a live arm.

## Positive check, to run when the GPU frees (not yet run)

Implemented as `census.py --run` in this directory; the shell form below is what
that script executes, kept here because it is the part worth reading.

The claim above is structural; a passing check must be able to fail. Correctness
observables are load-independent, so this is valid on a contended machine and
costs two runs, no pairs:

```
# identical env except the one knob; the shape flags are env, not argv
BASE=(ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=320 ALLPAKA_BENCH_SKIP_MTP=1
      ALLPAKA_PROFILE=max-performance)
env "${BASE[@]}" ALLPAKA_BENCH_REPORT=$D/default.json \
  allpaka bench --engine models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf
env "${BASE[@]}" ALLPAKA_RFUSE=1 ALLPAKA_BENCH_REPORT=$D/rfuse.json \
  allpaka bench --engine models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf
```

Compare, from the reports and the log's census lines: greedy continuation
tokens, `decode dispatches`, `windows=`/`set=`. **All three identical ⇒ the arm
encodes nothing and the −20–27% is retired.** Any difference ⇒ the guard read
above is wrong, the knob is live, and its cost is back on the table.

## The real question this uncovers, with a DoD

These three kernels are not thereby proven fast or slow - they are proven
*untested* under the policy that ships, and only reachable behind
`ALLPAKA_NORMFLAG=0` (or `ALLPAKA_PROFILE=safe|deterministic`). Two things
follow, in order:

1. `NORMFLAG` itself is the gate, and its own verdict ("30B decode 110 vs 116
   tok/s") is a 09-12 number too - see B3 of the era audit in
   `docs/decode-opts.md`. It is also not one knob: OFF changes the combine/norm
   fold, the norm's spin-flag ordering, and eligibility for this whole family.
2. **DoD for the family, after 1:** paired A/B *within* `NORMFLAG=0` -
   `{neither, RTOPK=1, RMT=1, RFUSE=1}` on qwen3-30b, AB/BA order, the null and
   drift vetoes from `docs/benchmarks/2026-09-25-glm-shared-stagger/drive.py`.
   A four-arm family that only exists under one policy flag has to be measured
   under that flag or not at all; measuring it under the default, which is what
   2026-09-12 did, cannot produce a result either way.

---

## Amendment 2026-09-25T21:55Z - the check above is now a script, and its
## token observable is real but one token short

`census.py` (same directory) implements the four-observable comparison, runs the
two arms when given `--run`, and refuses to say "identical" without seeing every
field (`--selfcheck` mutates each one and requires the comparator to notice). No
GPU was used to write or validate it: it was exercised on the two already-captured
identical-code arms from `docs/benchmarks/2026-09-25-glm-shared-stagger/` run 3,
where it reports 0 of 7 differing,

```
= decode token stream    [565, 220, 16, 13, 220, 150096, 60477, ...
= decode dispatches      256640
= decode waits           320
= decode fast path       {"attempts": 320, "successes": 320, "declines": 0}
= residency              residency: windows=3 set=true
= model fingerprint      8b1893cf1b9d2bb1
= git commit             8a162146e788
```

and on the mutations, which it sees. Two things this pins down that the DoD text
above left open:

* "Greedy continuation tokens" **is** in the report JSON, but not under a name
  that says so: `measurements[name=decode].input_tokens`. Reading
  `crates/allpaka-cli/src/bench.rs:445-462` - `decode_inputs.push(next)` then
  `extend_from_slice(&outs[..decode_tokens - 1])` - it is the seed token plus the
  generated stream, not the prompt (the prompt is `prefill.input_tokens`, and it
  is a synthetic arithmetic ramp, so the two are easy to tell apart). A check
  that grabbed `input_tokens` believing it was the prompt would have been
  vacuously true for every pair ever run; this one is not, and the `null` vs
  `plain` control above is the demonstration - same 320-token stream from two
  processes, and a mid-stream mutation makes the comparator fire.
* It is still **319 of 320 generated tokens**: the last generated token is
  dropped by that `[..decode_tokens - 1]` slice, so an arm that diverged only on
  the final step would read as identical. The stagger series' clean-pair
  criterion has the same blind spot, and nothing in the report covers it (there
  is no checksum field). Stated rather than fixed, because fixing it means
  touching `bench.rs` while another session is editing the tree.

So the RFUSE check needs no new instrument: on the next free GPU window,
`python3 census.py --run` in this directory is the whole positive check, at two
processes and no pairs, valid under load.

## Amendment 2026-09-26T00:03Z - the positive check ran, on another model, and the
## conclusion above does not hold there

The deferred check finally happened, as part of designing
[`../2026-09-25-normflag-policy/`](../2026-09-25-normflag-policy/preregistration.txt),
whose
[drive.py --probe](../2026-09-25-normflag-policy/drive.py) compares the same kind of
load-independent observables. It ran
on **qwen3-30b-a3b-Q4_K_M** at PP=3072/TG=320 (`ALLPAKA_PROFILE=max-performance`, binary
`8a16214`, sha256 `6a4418fa1f0592977b17f0940d4138d0153bed9e6e5c2d68863fc1ce7f622988`),
fourteen single uncounted processes at 23:53:58Z and 23:56:03Z, captured in
[probe 1](../2026-09-25-normflag-policy/probe-1-235358z.txt) and
[probe 2](../2026-09-25-normflag-policy/probe-2-235603z.txt) (every rate below is that
file's full-float `decode_tok_s`, rounded here to two decimals):

| arm | env | probe | decode tok/s | dispatches | vs default |
|---|---|---|---|---|---|
| `norm` | (none) | 1 | 132.44 | 215680 | - |
| `null` | (none) | 1 | 130.83 | 215680 | identical observables |
| `off` | `NORMFLAG=0` | 1 | 131.11 | 215680 | identical observables |
| `off_rf` | `NORMFLAG=0` + `RFUSE=1` | 1 | 90.93 | 200320 | **-15360** |
| `norm` | (none) | 2 | 129.93 | 215680 | - |
| `rf` | `RFUSE=1` | 2 | 106.86 | 208000 | **-7680** |
| `rmt` | `RMT=1` | 2 | 129.53 | 208000 | **-7680** |
| `k` | `RTOPK=1` | 2 | 105.96 | 208000 | **-7680** |

`norm` is the same arm in both probes and reads 132.44 against 129.93, so single
uncounted processes on this instrument carry about 1.9 % of spread; the three -7680
dispatch deltas are exact and identical across all three knobs, which is what makes them
load-independent evidence, and only `rf`/`k` clear that rate spread by more than it.

**The headline claim of this file - "`ALLPAKA_RFUSE=1` selects nothing under every
default configuration" - is false on this model.** All three knobs change the dispatch
count by exactly -7680 (= -24/token) with the policy bit at its shipped value, so on
qwen3-30b they are live under `max-performance`, and the 2026-09-12 rates the section
above retired as "one arm measured twice" were measured on arms that genuinely differed.

What the reasoning did wrong is a general trap, and it is the reason this file's own
"positive check" section existed: the table above resolves `normflag` from
`RuntimePolicy`/`RuntimeProfile`, i.e. from the **policy value** of the bit. The encoder
does not read that. It reads `LayerRefs.normflag`, which is a per-layer conjunction
(`metal.rs`: `let normflag = runtime.normflag && <q,k,v all matvec_q4_k_mv> &&
matches!(l.ffn, Moe)`), so `!refs.normflag` - the guard every one of the three knobs sits
behind - can hold while the policy says ON. A reachability claim about an env variable
therefore needs the conjunction evaluated for the model in question, not the profile table
quoted.

What survives this file intact:

* The structural read (one call site each, all under `!refs.normflag`) is correct as far
  as it goes, and is exactly what made the claim falsifiable.
* The instrument rule - *show that two arms encode different things before reading a rate
  off them* - is what caught this in ten minutes of one-process runs, and what would have
  caught the original 09-12 verdicts had it been applied then. It is now a script
  (`../2026-09-25-normflag-policy/drive.py --probe`) rather than advice.
* The GLM numbers in the section above are **not** re-tested here and are not claimed
  either way: on GLM the qkv triple may well be all-`matvec_q4_k_mv`, in which case
  `refs.normflag` is genuinely true there and the knobs genuinely inert. The claim is now
  model-scoped rather than universal, and the DoD below prices it.

Two things this amendment does **not** explain, stated so the next reader does not have to
rediscover them. `NORMFLAG=0` alone changes no observable at all on this model, yet
`NORMFLAG=0` *together with* `RFUSE=1` removes a second, equal 7680 - which no reading of
the two `!refs.normflag` sites (the `moe_plain` selection and the deferred-combine fold at
`combine_resnorm`) predicts. Either `refs.normflag` is not the only consumer of that bit,
or the two -7680s are different structural changes that happen to be the same size, or the
`dispatches` counter is not the pure function of the arm that the DCOMB series assumed
(its stability across 20 arms there argues against that). **DoD:** one
`ALLPAKA_DECODE_SPLIT=1` stage table each for `norm`, `off`, `rf`, `off_rf` on this model
names which stage loses each 7680, at four processes and no pairs; the paired series in
`../2026-09-25-normflag-policy/` prices the wall effect, and will say whether the dispatch
deltas reproduce pair after pair.

## Amendment 2026-09-26T00:22Z - both open questions above closed without the GPU

The paragraph before this one is left as written; this section supersedes its DoD. The
two unexplained things - the knobs being live under a policy that says otherwise, and the
`NORMFLAG=0`-alone no-op that halves into `NORMFLAG=0 + RFUSE=1` - have one cause, and
reading it needed a GGUF header parse rather than a stage table.

**The 30B's layers are not uniform.** In `models/qwen3-30b-a3b-Q4_K_M.gguf`, `attn_v` and
`ffn_down_exps` are Q6_K in 24 of the 48 layers - layers
0,1,2,3,4,5,8,11,14,17,20,23,26,29,32,35,38,41,42,43,44,45,46,47, the same set for both
tensors - and Q4_K in the other 24. That is llama.cpp's own Q4_K_M rule, and it was
already printed by the engine in every run log of every series in this repository as
`Q4K: tensors=289 / Q6K: tensors=49`: 240 attention-and-gate/up tensors plus 48 Q4_K
`attn_v`/`ffn_down` = 289, and those 48 plus `output.weight` = the 49 Q6_K. Two
independent fold gates then sit on exactly those two tensors:

* `refs.normflag` requires q, k **and v** to be `matvec_q4_k_mv`. In the 24 Q6_K-`v`
  layers it is therefore false *with the policy bit at its shipped value*, and
  `moe_plain = !refs.normflag && (rmt() || rfuse()) && Moe` is reachable there.
* `sw_capable` (the kernels that carry the fused-swiglu down variant) lists
  `matvec_q2_k`, `q3_k_mv`, `q4_k_mv`, `q5_k_mv`, `q8_0`, `q8_0_mv` - and **not**
  `matvec_q6_k_mv`. So the standalone swiglu dispatch survives in precisely the 24
  Q6_K-`down` layers and drops out in the other 24.

With those two facts the arithmetic closes on four numbers, none of them measured here:

| prediction from the header | measured |
|---|---|
| `RFUSE` / `RMT` / `RTOPK` fold in the 24 non-uniform layers: -24 dispatch/token, -7680 over TG=320 | -7680 for each of the three |
| `NORMFLAG=0` alone changes no *dispatch count* (the spin-flag path and the barrier path emit the same number; the flag replaces a `bar_c`, not a kernel) | `off` read 215680, identical to `norm` |
| `NORMFLAG=0` + `RFUSE=1` folds in all 48 layers: -15360 | `off_rf` read 200320 = 215680 - 15360 |
| the `DCOMB` fold removes one dispatch only where swiglu is standalone: -24/token | the 2026-09-25 DCOMB series measured +7680 for `unfold` in 14/14 real pairs |

The capture and the check are
[`../2026-09-25-normflag-policy/layer-types-qwen3-30b.txt`](../2026-09-25-normflag-policy/layer-types-qwen3-30b.txt),
produced by
[layer-types.py](../2026-09-25-normflag-policy/layer-types.py) in that directory; it
prints the per-suffix type census, cross-checks its own dtype table against the engine's
`tensor-types:` census, and emits the predicted dispatch deltas. Run it on a model before
quoting any per-layer dispatch arithmetic against its layer count.

Consequences:

* This file's original structural claim is wrong in its conclusion but right in its
  method, and the method is what generalises: the gate *is* `!refs.normflag`, and the
  mistake was resolving that from the policy table instead of the conjunction. The
  conjunction is computable offline; that is now a script rather than advice.
* The `ALLPAKA_NORMFLAG` row is **not** a policy comparison on this model, and cannot be
  measured as one: with the shipped policy the bit is already false in half the layers,
  so `NORMFLAG=0` extends an existing state rather than flipping it. What it will do is
  move the other 24 layers onto the barrier path, which is the thing worth pricing.
* On GLM the same census decides everything, and it is cheap: if `attn_v` is uniform
  there, the 2026-09-12 GLM rates were indeed one arm measured twice, and if it is split
  they were not. That is a header read on `models/`, not a benchmark.
* A lever falls out of the `sw_capable` line, which is the only reason this repository
  has two series' worth of 7680s instead of 15360s: the fused down+combine currently
  pays a standalone swiglu pass in every layer, so it saves a dispatch in only the 24
  Q6_K-down ones. Teaching that kernel to apply swiglu once per slot row (stage it in
  threadgroup memory instead of re-evaluating `exp` per row pair, which is what
  `ALLPAKA_DCOMB_SW=1` does and what it measured at: -0.8 ms/token) would double the
  fold's reach on this model. Priced against the measured DCOMB quantum of +0.868
  ms/token for the first 24 layers, the ceiling is another ~0.9 ms/token (~11 % decode)
  and the floor is the `DCOMB_SW` penalty; neither number is small enough to decide on
  the spot.


## Amendment 2026-09-26T00:43Z - the census run on the other three models, and the bug it found in the census tool

The GLM question above was a header read, so it was done: 47 layers, 46 of them MoE,
`attn_v` Q6_K in 23 and Q4_K in 24. Nothing was measured and no GPU was used.
[layer-types.py](../2026-09-25-normflag-policy/layer-types.py) was then run on every local
MoE model and the output is
[layer-types-multi-2026-09-26-004214z.txt](../2026-09-25-normflag-policy/layer-types-multi-2026-09-26-004214z.txt):

| model | MoE layers | `refs.normflag` can be TRUE | router folds reachable | down sw_capable | DCOMB dispatch saving |
|---|---|---|---|---|---|
| qwen3-30b-a3b Q4_K_M | 48 | 24 | 24 (7680 over TG=320) | 24 | 24 layers, 7680 |
| GLM-4.5-Air Q4_K_M | 46 of 47 | 24 | 22 (7040) | 22 | shared expert - the fold is not built at all |
| q235 (235B-a22b Q2_K_XL) | 94 | 90 | 4 (1280) | 94 | 0 layers |
| Qwen3.6-35B-A3B UD Q4_K_M | 40 | 0 | 40 (12800) | 37 | shared expert - not built |

Four things follow, and one of them is a bug in the tool that produced the other three.

**(1) A split GGUF carries only its own tensor infos.** Part 1 of GLM lists 568 tensors
and `blk.0..blk.33`; part 2 lists the remaining 235 and carries only `split.*` metadata.
The first version of the script read one part and used `block_count` as the denominator,
so GLM came out as "reachable in 27 of 47" - a prefix of the layers counted against the
whole model, and a number that looked perfectly reasonable. The tool now unions every
part, requires every `blk.0..blk.{block_count-1}` to be present, and refuses a part set
whose `split.count` disagrees. That is the same failure this file was written about
(a model-level property read as a layer-level one), reached from the other side, and it
is why the check is code and not advice.

**(2) On the 235B the router folds are nearly dead, and the DCOMB fold saves no dispatch
at all.** 90 of its 94 layers have q,k,v all Q4_K, so `RFUSE`/`RMT`/`RTOPK` would change
4 dispatches per token there (1280 over TG=320) - 4 % of the layers, and no measured
235B dispatch-per-token census exists to turn that into a share of the stream
(`../2026-09-24-dispatch-census/README.md` counts GLM at 778/tok and 30B at 674/tok and
does not cover the 235B). And because every `ffn_down_exps` is Q3_K or Q4_K, both of which are
`sw_capable`, the unfolded arm never pays a standalone swiglu, so `DCOMB` removes zero
dispatches per layer on that model. Whatever the DCOMB A/B measures on the 235B is the
fused kernel and the dropped `bar_c`, not a launch count; the qwen3-30b +0.868 ms/token
is therefore not a transferable number, and the roadmap already says the fold is not
built on shared-expert models for a third reason.

**(3) Qwen3.6-35B-A3B is a hybrid: 30 of its 40 MoE layers are GDN.** They carry
`attn_qkv`/`attn_gate` and no q,k/v at all (`crates/allpaka-model/src/model.rs:815-839`),
so `refs.normflag` cannot be TRUE in them for an architectural reason rather than a quant
one, and the router folds are reachable in 40 of 40 layers there. Any 35B number for this
family is a whole-model number, which no other local model is.

**(4) The 2026-09-12 GLM rates were measured on arms that genuinely differed** - 22 layers
of encoder difference - so the "one arm measured twice" retirement is wrong for GLM in
exactly the way it was wrong for qwen3-30b, and for the same reason. What is still unknown
on GLM is the sign and size, which a header read cannot give; the prediction to check
against a probe is 7040 dispatches, not 15040.
