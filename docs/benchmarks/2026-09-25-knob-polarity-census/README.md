# How each Metal knob's env is actually parsed - and the one grammar trap that makes arms empty

Captured 2026-09-25 21:43 UTC at tree `dcdc131`. **No GPU, no build** - this is
a reading of `env::var` idioms, and the whole output is reproducible with one
command:

```
python3 docs/benchmarks/2026-09-25-knob-polarity-census/census.py
```

[`results.txt`](results.txt) is its output at that revision: 65 names with an
env read inside `gpu/metal.rs`, `gpu/msl/`, `runtime.rs`, `profile.rs`,
classified by idiom, one row per read site, names whose sites disagree marked
MIXED. Scope note: `crates/allpaka-backend/src/gpu/cuda/` is deliberately
excluded - it has its own dialect and its own docs.

## Finding 1: twenty knobs read *presence*, so `=0` turns them ON

The dominant idiom in this backend is `is_ok_and(|v| v == "1")` - opt-in, and
`=0` means off. Nineteen other names are read as `env::var_os(NAME).is_some()`,
where **any value, including `0`, `false` and the empty string, turns the
feature on**: `ATTN_ZERO`, `DECODE_SPLIT`, `DUAL`, `FFN_SPLIT`, `FFN_TIME`,
`GDN_ZERO`, `MM64`, `MM_BN32`, `MM_BN64`, `MOE_ZERO`, `NO_BARRIER`,
`NO_EXPERT_ROWS`, `NO_GPU`, `NO_STAGE_BARRIER`, `PF_SPLIT`, `Q2_ILV`,
`ROWS_DEBUG`, `SERIAL`, `TOKENBUF_DEBUG`, `VERIFY_DEBUG`.

That is not a bug in itself - most of them are instruments, and "just set it"
is the right ergonomics for a probe. It is a trap for a **sweep driver**: a loop
that emits `NAME=0` and `NAME=1` as its two arms reports those twenty knobs with
both arms on and a flat difference, which is precisely the shape of a "measured
neutral". `Q2_ILV` is the one of these that is a real kernel-choice lever, and
its registry row in `docs/decode-opts.md` now says so.

## Finding 2: the seven policy knobs have TWO grammars, and which one applies depends on a third variable

`crates/allpaka-cli/src/main.rs:327-331` (same two lines in `autotune.rs:218`):

```rust
if let Ok(profile) = std::env::var("ALLPAKA_PROFILE") {
    let profile: RuntimeProfile = profile.parse()?;
    let resolved = profile.resolve_with_env();
    let _ = allpaka_backend::runtime::install(resolved.policy);
}
```

- **`ALLPAKA_PROFILE` unset** → `runtime::get()` lazily calls
  `RuntimePolicy::from_env()` (`runtime.rs:35-48`), whose reads are narrow:
  `NORMFLAG` reverts only on the exact string `0`, `DECODE_SERIAL` turns on only
  on the exact `1`, `PF_DEFER`/`PF_ONEBUF`/`GPU_ROUTE`/`MM_PIPE` turn off only on
  the exact `0`.
- **`ALLPAKA_PROFILE` set to anything valid** → `install()` replaces the policy
  with one built from the *profile table* plus `resolve_with_env()`, whose
  overrides go through `parse_bool` (`profile.rs:88-94`), accepting
  `1/true/yes/on` and `0/false/no/off`. `from_env()` is never consulted again.

So `ALLPAKA_NORMFLAG=off` means **default (ON)** to a bare `allpaka bench` and
**OFF** to the same command with `ALLPAKA_PROFILE=max-performance` - which is
what airbug exports by default (`docs/decode-opts.md`, airbug defaults). Same
for `DECODE_SERIAL=true`, `PF_DEFER=false`, `GPU_ROUTE=no`, and the rest. An
arm written in the wrong grammar is an *empty* arm: identical encodes, and a
flat ratio that reads as a measurement.

This is the sibling failure to
[`../2026-09-25-rfuse-unreachable-arm/README.md`](../2026-09-25-rfuse-unreachable-arm/README.md)
- there the guard made an arm empty, here the parser does - and it is cheap to
avoid: **quote the literal in the driver and say which reader you are talking
to.** Every arm string in the drivers under `docs/benchmarks/` was checked
against both grammars and none of them diverges: the stagger series uses
`ALLPAKA_SHARED_STAGGER` ∈ {`1`,`0`} (both grammars agree), the attention series
uses numeric `ATTN_S8` arms and `ATTN_MV=0`, the NR0 preregistration uses
`{1,2,4}`. The `SHARED_STAGGER=0` null arm is safe for a specific reason worth
keeping in mind: it is the *only* off-spelling both readers agree on.

**DoD (a test, not a measurement):** `runtime.rs` has
`production_defaults_are_explicit` for the unset case and nothing for the
override path. Add a test per policy field asserting that `NAME=off`,
`NAME=0` and unset produce the *documented* value under both entry points
(`from_env()` and `RuntimeProfile::MaxPerformance.resolve_with_env()`), with
`env::set_var` in a serial-test guard. It fails today for the four spellings in
the table above - which is the finding, pinned. Not written yet because writing
it means compiling, and the GLM stagger series holds the machine.

## The three names the regexes did not classify, read by hand

| name | idiom | consequence |
| --- | --- | --- |
| `ALLPAKA_SWFUSE` (`metal.rs:10709`) | `is_ok_and(\|v\| v != "0")` | opt-in **and** `=0`-off: unset and `=0` are the same arm, any other non-`0` string is on. Not a "default ON" knob despite the `!= "0"` shape - the `Ok` must exist first |
| `ALLPAKA_Q3_PROBE` (`metal.rs:15743`) | string must equal one of `Q3_PROBE_ARMS`, else `""` | selector; an unrecognised value silently means "no probe", so a typo'd arm name reads as the incumbent |
| `ALLPAKA_MMID_RTILES` (`metal.rs:16391`) | numeric, `map_or(auto, \|n\| n.max(1))` | its default is *computed* from `m_fused`, not a constant, so "default" in a verdict table is a shape, not a number |

## Did this already contaminate the archive? No - and that check is one command

```
grep -rInoE "ALLPAKA_[A-Z0-9_]+=(true|false|on|off|yes|no)\b" docs scripts
grep -rInoE "ALLPAKA_(ATTN_ZERO|DECODE_SPLIT|DUAL|FFN_SPLIT|FFN_TIME|GDN_ZERO|MM64|\
MM_BN32|MM_BN64|MOE_ZERO|NO_BARRIER|NO_EXPERT_ROWS|NO_GPU|NO_STAGE_BARRIER|PF_SPLIT|\
Q2_ILV|ROWS_DEBUG|SERIAL|TOKENBUF_DEBUG|VERIFY_DEBUG)=0" docs scripts
```

Run 2026-09-25 21:45 UTC over `docs/` (including every driver under
`docs/benchmarks/`) and `scripts/`: the only hits were the two strings in this
file and its neighbour, written minutes earlier. No published verdict in the
repo rests on a word-valued policy arm or a presence knob swept with `=0`. The
trap is live for *future* sweeps, which is why it is documented next to the
registry rather than as a correction.

## What this file does *not* claim

It classifies how a value is parsed. It says nothing about whether a knob is
good, measured, or reachable - reachability is a guard-condition question, and
for three decode knobs the answer there turned out to be "not under the shipped
policy", which is the neighbouring artifact's subject.
