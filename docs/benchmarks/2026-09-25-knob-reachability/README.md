# Every env knob the Metal decoder reads, and whether the shipped policy can reach it

Captured 2026-09-25 22:08 UTC. **No GPU and no build**: the scan reads source.
It exists because the RFUSE audit
([`../2026-09-25-rfuse-unreachable-arm/README.md`](../2026-09-25-rfuse-unreachable-arm/README.md))
found a recorded dead end that described an arm the default policy never
encodes, and that class of error is only dismissible once it is counted - one
knob at a time by hand is how RFUSE was found, and hand-checking 28 more is how
the next one stays hidden.

```sh
python3 reachability.py        # output pasted into results.txt
```

Scope as written: accessor functions in `crates/allpaka-backend/src/gpu/metal.rs`
that read an `ALLPAKA_*` env var - 28 of them at `dcdc131`, 58 names counted
across the file. Policy fields read in `runtime.rs`/`profile.rs` are the *other*
direction (they are covered by the polarity census,
[`../2026-09-25-knob-polarity-census/README.md`](../2026-09-25-knob-polarity-census/README.md));
the CLI's `ALLPAKA_BENCH_*` and the CUDA backend's 38 names are out of scope by
design and are reported only as counts.

## Finding 1 - policy-gated knobs are exactly four, and they are one family

For each accessor the scan walks outward from every call site through its
enclosing blocks and then resolves one level of indirection, because the gate
word is usually not on the line that reads the env: `moe_plain` is what the `if`
tests, and `rfuse()` is read inside the `let moe_plain = ...` that defines it.
Without that second level the scan reports RFUSE as ungated, which is the wrong
answer for the knob that started this.

| accessor | gate on its call path | reachable under `normflag` ON (every default profile)? |
| --- | --- | --- |
| `rfuse()`, `rmt()` | `if moe_plain { ... }`, `moe_plain = !refs.normflag && (rmt() \|\| rfuse()) && Moe` | no |
| `rtopk_fused()` | `else if !refs.normflag && rtopk_fused()` | no |
| `mega_debug()` | `if any_mega && ...`, `any_mega` needs `FfnRefs::Moe { mega: Some(_) }`, and `mega_enabled()` returns `false` unconditionally | never, under any policy |
| the other 24 | none found | yes |

*(The "no" rows were withdrawn on 2026-09-26T01:04Z, two sections down: `refs.normflag`
is a per-layer conjunction over the model's tensor types, so the family is live under the
shipped policy on qwen3-30b. The table is kept as written because the mistake - reading a
conjunctive field as its policy bit - is the finding.)*

Two consequences, in opposite directions:

* The RFUSE/RMT/RTOPK verdicts stay retired-as-measured (the arm-encodes-nothing
  argument), and they are the *whole* extent of the class - no other recorded
  cost in the registry can be dismissed the same way, so the remaining dead ends
  stand or fall on their instrument, which is the era audit's problem, not this
  file's.
* MEGA's debug print is dead under every policy, which is the intended state
  (`mega_enabled()` is a hard `false` after the device-atomic freeze). The
  registry already classifies `ALLPAKA_MEGA*` as instrument-not-lever, so this
  confirms the docs rather than correcting them.

## Finding 2 - one knob the code reads and no doc names: `ALLPAKA_DCOMB`

The scan's second half compares name sets in both directions, exactly (a name
counts only next to a read idiom - `env::var`, `env::var_os`, or the bench
harness's own `get("NAME", default)` - because a name that appears in code only
inside a comment is the `ALLPAKA_Q` phantom the census already met).

* Names read in `metal.rs`, absent from the registry: **`ALLPAKA_DCOMB`**, one
  of 58. It is the down+combine fold, **default ON**, so the shipped decoder has
  been running an unregistered knob the whole time.
* Why the census missed it, twice over: the registry's coverage was checked with
  a bare `grep "$name"`, and `ALLPAKA_DCOMB` is a prefix of the documented
  `ALLPAKA_DCOMB_SW`, so the short name appeared covered. And the loop that ran
  that grep was itself vacuous - this shell is zsh, where `for n in $names` does
  not split a multiline scalar, so it executed once against a 65-line pattern.
  Both flaws are corrected in
  [`docs/decode-opts.md`](../../decode-opts.md) (the dated note under the
  enumeration command, and a `DCOMB` registry row).
* Why no comment saved it either: the paragraph describing the fold sat on
  `fn dcomb_sw()` and `fn dcomb()` had no doc comment at all. Split, comment-only.
* Its status after registration: **listed, still unmeasured.** Its only number
  is the code comment's `~0.3 ms/token on Qwen3-30B-A3B`; no A/B against `=0`
  exists anywhere in `docs/benchmarks/`. DoD is in the registry row. Reading the
  gate by hand also bounds it: the fold needs `shared.is_none()`, so it never
  builds on GLM-4.5-Air or the 235B - a 30B-class lever, not a whole-fleet one.

## Finding 3 - names the docs carry that no code reads

Six at first capture (22:08 UTC), and the split matters more than the list. The
amendment below moves one of them out entirely, so read this list together with
it:

* `ALLPAKA_MEGA` - absent from code entirely. The registry's "do NOT set
  `ALLPAKA_MEGA`" warning is about the megakernel path, not a variable; the name
  is inert. The warning should stay (the hazard is real and the sentence explains
  it), but it must not be read as "there is an env var to avoid".
* `ALLPAKA_MEGA_TG` - code mentions it in prose only, and that prose says
  `ALLPAKA_MEGA_TG` is ignored (`mega_tg()` returns a literal 1). Same class:
  documented name, no read site.
* `ALLPAKA_Q3_NR0` - absent from code entirely, and appears only in
  [`../2026-09-24-indexed-matvec/README.md`](../2026-09-24-indexed-matvec/README.md),
  i.e. a dated record of a probe arm that has since been removed with its code.
  A record is allowed to name a dead knob; the registry is not, and the registry
  does not.
* `ALLPAKA_Q` - the census's own phantom, matched out of `ALLPAKA_Q{2,4,5,8}_NR0`
  by an unquoted `grep -o`. Listed here so the set difference stays reproducible.
* `ALLPAKA_LOCAL_API_KEY` - a prose-only mention in the studio/chat docs, not
  decode path, and no read site.
* `ALLPAKA_TEST_NATIVE_VAULT` - listed above as unread, and **wrong**: the
  22:21Z amendment below is about exactly this entry.
* `ALLPAKA_Q` and the two wildcard stems the printout also yields
  (`ALLPAKA_BENCH_`, and one the README itself mints) are prose shorthand rather
  than stale rows; `results.txt`'s header names all three by hand.

## Amendment 2026-09-25T22:21Z - the scan had a quote-shaped blind spot of its own

Re-running the name-set comparison after writing this - one fix to
`reachability.py` and one feature added then dropped - changed its own findings:
the same failure mode Finding 2 describes, met again in the tool that finds it.

* **`READ_IDIOM` matched double quotes only**, so `scripts/test-studio.py`'s
  `os.environ.get('ALLPAKA_TEST_NATIVE_VAULT')` was invisible and the name was
  filed above as "docs mention it, code does not read it". It is read, in the
  studio test harness, single-quoted. Both quote styles and `os.environ.get` /
  `getenv` now count: read sites go 142 -> 144 (`ALLPAKA_TEST_NATIVE_VAULT`,
  `ALLPAKA_TEST_BINARY`), and the docs-only set loses that bullet - which was
  already half-wrong as written, since neither name is decode path.
* **An automatic "glob stem" classifier was written, run and dropped.** Prose
  like `ALLPAKA_Q` + `{2,4,5,8}_NR0` and the bench-variable family written with
  a trailing wildcard yield stems no code can ever read, and a first pass
  labelled them inside the tool. It does not survive contact with its own corpus:
  the scan reads `docs/benchmarks/**/*.md`, which contains this README, so the
  labelling depends on how these sentences happen to be worded - and one of them
  mints a third entry, the MEGA family stem, on its own account. A rule whose
  output moves when the documentation about it is rephrased is not a check. The
  three stems are named by hand in `results.txt`'s header and left visible in the
  printed set instead of hidden by classification.
* **Finding 2's headline set is now empty by design**: "names read in `metal.rs`,
  absent from the registry" prints `none`, because the `ALLPAKA_DCOMB` row went
  into `docs/decode-opts.md` at 22:09 UTC. That is registration, not measurement;
  the knob's status in that row is still "listed, still unmeasured", and task
  DoD unchanged.
* Line numbers in `results.txt` past ~16000 shifted +5 between the two captures -
  the `dcomb`/`dcomb_sw` comment split itself, i.e. this session's own edits to
  the dirty file. Another count that moved for a reason worth stating: 1547 ->
  1558 insertions of the other session's WIP plus mine.

What the amendment does *not* change: the four policy-gated accessors, the DCOMB
verdict, or the `NORMFLAG=0` DoD. What it adds is the discipline. A set
difference inherits the shape of its matcher, so a naming check has to be
mutation-tested the way a timing check is:

```sh
python3 reachability.py --selfcheck    # 8 cases, exit 1 if the matcher misses any
```

The cases cover both quote styles, `os.environ.get`, `getenv`, the harness's
`env(NAME, default)` helper, and two negatives - a name inside a comment and a
name in bare prose, neither of which counts as a read. Current matcher: 8 of 8.
The 22:04 matcher, run against the same cases: fails exactly the two
single-quote ones, which is the same pair the real tree moved (142 -> 144 read
sites). That is the check doing its job - an instrument that cannot fail on the
bug it was written to catch is not catching anything. The census comparator in
[`../2026-09-25-rfuse-unreachable-arm/census.py`](../2026-09-25-rfuse-unreachable-arm/census.py)
got the same treatment at 21:55Z for the same reason.


## What the scan cannot see

Three limits, all of which bit during the work and none of which the tool fixes:

* **Structural gates.** `shared.is_none()`, `hidden % 2`, `n_used <= 32` and the
  per-model kernel names bound `DCOMB` far more tightly than `normflag` bounds
  RFUSE, and the scan does not look for them because they are not policy fields.
  Reading them is why the `DCOMB` row says "30B-class" instead of "MoE decode".
* **Model-shape reachability.** A knob can be live in code and unmeasurable on
  this machine because no local model has the shape it gates (the 235B has no
  `nextn` block, so its experts cannot be put through the MTP rows path). The
  scan is silent on that.
* **A dirty tree.** `metal.rs` at capture time carried another session's
  uncommitted WIP (1547 insertions over `dcdc131`), so line numbers are the
  working tree's and the gate text is quoted verbatim to make the reading
  checkable against any revision.

## What this changes about the goal

The goal's residue list is "every measurable lever closed with an artifact".
This scan adds one entry to it (`ALLPAKA_DCOMB`, registered and unmeasured, with
a DoD) and closes a worry rather than a lever: the policy-gate class that
retired three 2026-09-12 numbers contains exactly those three, so the other
recorded dead ends have to be argued with instruments, not with reachability.


## Amendment 2026-09-26T01:04Z - the scan can now resolve a field, and Finding 1's "no" was the wrong answer

The third column of Finding 1's table reads "reachable under `normflag` ON (every default
profile)? **no**". That column is withdrawn. `../2026-09-25-normflag-policy/` measured the
`rf` arm 18 % slower than the default with exactly -7680 dispatches over TG320 while
`ALLPAKA_NORMFLAG` was left unset, so the family is live under the shipped policy on
qwen3-30b. The table's own quoted gate text was the evidence and I read it as a policy
value instead of as a conjunction; the tool made the same mistake, because it stopped at
"this accessor sits behind a `RuntimePolicy` field".

`reachability.py` now resolves the field rather than naming it. For each policy field it
finds every read site - a local named after the field, or an inline
`crate::runtime::get().<field>` / `policy.<field>` read, including one that lands in a
differently named local - joins the enclosing statement across lines with comments
stripped, splits it on top-level `&&` and `||` only, and classifies each clause as a
policy read or a structural one. A field with a structural clause whose value is carried
per layer (`LayerRefs`, `refs.`) is reported as `PER-LAYER CONJUNCTION`, and the tool
refuses the reachability conclusion for it. Current output, in
`resolutions-2026-09-26-010300z.txt`:

| field | verdict | why |
| --- | --- | --- |
| `normflag` | PER-LAYER CONJUNCTION | 3 clauses, 2 structural (`m.kernel == "matvec_q4_k_mv"` over the first three mats, `matches!(l.ffn, TokenFfn::Moe)`); stored at metal.rs:10757 and 11009, read back at 5 `refs.normflag` sites |
| `decode_serial` | CONJUNCTION, not carried per layer | `serial = runtime.decode_serial \|\| any_mega` - the second term is structural, but the value never reaches a per-layer struct |
| `prefill_defer`, `prefill_one_buffer`, `gpu_route`, `mm_pipeline`, `attention_split` | POLICY VALUE | each read is the bare field, so for these the policy bit really does decide |

So of the seven fields the decoder consults, exactly one makes reachability a question
about the model file, and it is the one that gates this family. The first bullet of "What
the scan cannot see" is therefore narrowed rather than deleted: the scan still does not
evaluate `shared.is_none()`, `hidden % 2` or `n_used <= 32` in a call path it was not
asked about, but structural terms adjacent to a policy read are now found, counted, and
printed - which is what turns "not a policy field" from an excuse into a check.

`python3 reachability.py --selfcheck` is 17 cases: 8 on the read idiom (unchanged) and 9
on the resolver, whose fixture is the real `normflag` shape. The resolver cases were
mutation-checked - a naive `&&` split (4 clauses instead of 3), unstripped comments (a
comment clause leaking in as a term), treating every clause as a policy read (`normflag`
downgraded to a bit), and reading `normflag: bool,` as a store (type declarations counted
as carries) each fail at least one case. The first version of these cases passed under
three of those four mutations, which is why they were rewritten: a case that cannot fail
is not a regression check.

## Amendment 2026-09-26T06:56Z - the enumeration was a signature test, and it was blind to half the file

The sentence just above ("`python3 reachability.py --selfcheck` is 17 cases") is
superseded: it is 25 cases, and the reason is a defect in what the scan looks at.

`accessors()` matched `fn (\w+)\(\) -> (?:bool|u32|usize|f32|i32) \{`. That is a test
of the *signature*, not of the knob, and the Metal encoder does not have one dialect for
"this env var selects work". Against the current file the scan enumerated **30 of the 59
names `metal.rs` reads**; the run now covers 59 of 59 (35 zero-arg readers + 24 inline
reads, `outside both sections: 0`). Three shapes were missing:

- **a return type not in the list.** `fn mv_tg() -> u64` reads `ALLPAKA_MV_TG`, and `u64`
  was not one of the five allowed types. So was `fn serial_dispatch() ->
  metal::MTLDispatchType` (`ALLPAKA_SERIAL`, six call sites).
- **a kernel SELECTOR.** `fn attend_kernel() -> &'static str` reads `ALLPAKA_ATTN_MV` and
  `ALLPAKA_ATTN_S8`; the decode attention ladder, `attend_rows_kernel` (prefill),
  `q2_kernel` and `q3_probe` are written the same way. This is the one that bit the goal
  directly: the two knobs the registry still lists as unpriced are read *only* inside
  `attend_kernel`, so the file whose job is to say whether an arm can be reached at all
  could not see them, and every sentence about them rested on hand-reading one function.
- **a read inside a function that takes arguments**, whose gating is not a call path.
  `decode_token` (SWFUSE, DECODE_SPLIT, TOKENBUF_DEBUG), `encode_verify_tokens` (the six
  `*_ZERO`/`*_DEBUG`/NO_EXPERT_ROWS reads), `mm_kernel_for` (MM64, MM_BN32, MM_BN64,
  MM_LL_F32), `lanes_per_row` (LPR_DIV, Q2_ILV, Q2_NR0), `ffn_batch_grouped` (DUAL,
  FFN_SPLIT, MMID_RTILES), the prefill blocks (FFN_TIME, PF_SPLIT), `add_mapping`
  (GPU_WINDOW_GIB), `attach` (NO_GPU), `probe_skip` (SKIP). Several of these have
  multi-line signatures, so even a one-line regex for `fn name(args)` missed them.

`inline_reads()` now reports that last class with the brace nesting at the read line,
which is the only gating it has.

**The second fix is the one that changes an answer.** A selector written

```rust
if *MV.get_or_init(|| env::var("ALLPAKA_ATTN_MV") …) { return "attend_mv"; }
… env::var("ALLPAKA_ATTN_S8") …
```

gates S8's arm on MV without S8's read sitting inside any block that mentions MV, so a
walk over *nesting* cannot see the guard at any point. `sequential_gates()` looks for an
earlier `if`-block in the same body that ends before the read and contains a `return`, and
the run now prints it:

```
attend_kernel  env=ALLPAKA_ATTN_MV,ALLPAKA_ATTN_S8  calls=4 at [10079, 11017, 12631, 15935]
    in-body: ALLPAKA_ATTN_S8 (metal.rs:15921) only reached if: *MV.get_or_init(|| std::env::var("ALLPAKA_ATTN_MV").map_or(true, |v| v != "0"))
```

That is the instrumented version of a claim the attention campaign's preregistration was
making from memory: the four `ATTN_S8` arms (`attend_s32`/`s16`/`s8`/`attend`) are
encodable only with `ATTN_MV=0`, so an `ATTN_S8` sweep is a sweep of the *fallback family*,
not of the shipped decode attention.

**What did NOT change is the more useful result.** `metal.rs` reads 59 knob names; the run
at 01:03Z enumerated 29 of them through 28 accessors, and this run enumerates all 59
through 35 zero-arg readers plus 24 inline reads, with `outside both sections: 0`. Finding 1
survives the widening untouched: still exactly four policy-gated readers - `mega_debug`,
`rfuse`, `rmt`, `rtopk_fused` - and `attend_kernel` joins the *ungated* list, which is what
licenses measuring the attention knobs under the shipped default policy at all. So the
retirement of the 2026-09-12 numbers this file produced does not need re-examining. A blind
spot in an instrument does not automatically mean the conclusions it produced were wrong;
it means nothing checked them.

The 30 newly visible names, and what each one is worth:

| names | status |
|---|---|
| `ATTN_MV`, `ATTN_S8` | the two live decode gaps; the attention campaign is armed for them, and the registry already states that "no measurement of s8/s16 against s32 exists in the repo" |
| `MV_TG`, `LPR_DIV`, `Q2_ILV`, `Q2_NR0` | `docs/decode-opts.md` records the first three in the geometry group "all re-measured now regress or are neutral"; `Q2_NR0` has its own streamed sweep in [../2026-09-24-indexed-matvec/streamed-sweep.txt](../2026-09-24-indexed-matvec/streamed-sweep.txt) |
| `Q3_PROBE`, `Q5_0_NR0` | probe/arm selectors - the q3_k probe arms and the `q5_0` NR0 sweep the registry prices in its own row |
| `SWFUSE`, `DUAL` | measured; `DUAL` in its own row (30B 105.0 against 113.7 tok/s, "measured slower both ways") |
| `GPU_WINDOW_GIB` | gap 3, closed the same evening it was named |
| `MM_LL_F32` | gap 2, still open |
| `ATTN_ZERO`, `MOE_ZERO`, `GDN_ZERO`, `ROWS_DEBUG`, `VERIFY_DEBUG`, `TOKENBUF_DEBUG`, `NO_EXPERT_ROWS`, `NO_GPU`, `SKIP`, `DECODE_SPLIT`, `FFN_SPLIT`, `FFN_TIME`, `PF_SPLIT` | the instrument family - one registry row classifies all of them "instrument, not lever", which is the right call and the only place the registry carries a per-name verdict inside a group row |
| `MM64` | priced and negative, but by a pointer rather than a number: that row's verdict is "`MM64` measured worse (comment at `metal.rs:9528`)" |
| `ATTN_T4`, `MMID_RTILES`, `MM_BN32`, `MM_BN64` | **the finding:** named in a registry row that delegates their numbers to `docs/moe-prefill.md` ("Names resolve there, not here"), where they appear only in the switch table (`ATTN_T4=n \| attend_rows tile: 8 (default), 4, 0=row-per-tg`, `ALLPAKA_MMID_RTILES=n \| pin mmid grid y`) with no rate. `MM64` is not in this row: its registry verdict is "measured worse". |

Those four are the prefill analogue of the RFUSE pattern - a row that names a knob and
carries no number - and no benchmark artifact in the repo mentions them (the only hits are
the two census files and strings embedded in pinned binaries). They are prefill, so they
are outside this goal's decode-path scope, but they do bound the registry's claim: "the
open set is six knobs" is a statement about the decode surface, and on the prefill side
four more are reachable, selectable and unpriced. Fixing the instrument moved no decode
verdict and enlarged the honest map, which is the whole point of having one.

**No prior artifact was blind in the same way.** The sibling scan
[../2026-09-25-knob-polarity-census/results.txt](../2026-09-25-knob-polarity-census/results.txt)
enumerates 65 names by walking read sites directly, so the repo already *knew* the
surface; what this file lacked was the gating walk over the parts of it that are not
`fn name() -> bool`. The two tools answer different questions and should be read together.

**Re-derivation and mutation checks.** Full run
[resolutions-2026-09-26-065600z.txt](resolutions-2026-09-26-065600z.txt) (tree `dcdc131`
plus the other session's uncommitted WIP - note the re-run also picked up their accessor
extraction of `q5_0_nr0`, which shifted line numbers ~19-21 after `metal.rs:16180` and
moved the accessor count 28 -> 29 on its own). `--selfcheck` is 25 cases: the 17 above
plus 8 on a synthetic file covering the selector shape, the `-> u64` shape, the in-body
sequential gate in both directions (present for the later read, not invented for the
earlier one), the multi-line signature, and the negative case that a selector's knobs are
not also reported as inline reads. Mutation-checked by reverting one constant: setting
`ZERO_ARG_RE` back to the old five-type signature test fails exactly the two new-shape
cases and nothing else (mutated copy `2ae1a41b265bbf879fc3911ef0037457c39ab8eb10e25209672e1cfb1232d37b`,
run in place at the correct repo depth, deleted afterwards; the live file is
`4ed708d7130730b1f0e3730f01fd692622d53ea33deb8157e5da83159af766c8`).
