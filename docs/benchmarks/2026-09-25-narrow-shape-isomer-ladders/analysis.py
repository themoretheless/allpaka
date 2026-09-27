#!/usr/bin/env python3
"""Read the isomer-ladder logs and print the shape-law tables.

The logs come from `gpu_ffnbench::narrow_dim_isomer_ladders`: one process, every
ladder member interleaved inside each pass, `q8_0` run at every shape as the
in-run meter. Each row line carries the per-pass samples, so every statistic
here is derived from those and printed at full precision - nothing is retyped
from the bench's own summary columns.

Every statistic that crosses shapes is *paired by pass index*: pass p of shape A
divided by pass p of shape B, then summarised over p. This is not a stylistic
choice. Section 0 shows the reason - the rate inside one row's passes decays
monotonically (1.3-4.1x from the first pass to the last in these windows), so a
ratio built from two rows' independent maxima inherits whichever one happened to
peak earlier, and the machine's throttle state rather than the shape.

Run from anywhere: `python3 analysis.py > results.txt`.
"""
import math
import os
import re
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "ladders-run1-feeds8-32.txt",
    "ladders-run2-with-sawtooth.txt",
    "sawtooth-rep-A.txt",
    "sawtooth-rep-B.txt",
]
# The mechanism probe: ladder 1 (the 30B's own dispatch size) re-run per process
# under each dispatch-geometry knob. `q6_k` cannot see `ALLPAKA_Q4_NR0`, so it is
# the run's negative control; `base2` repeats the incumbent at the end, so a
# drift across the six arms shows up as base2 differing from base.
GEOM = [
    ("ladder1-geom-base.txt", "NR0 default (=2), TG default (=128)"),
    ("ladder1-geom-nr0-1.txt", "ALLPAKA_Q4_NR0=1  (32 lanes/row, 1 phase)"),
    ("ladder1-geom-nr0-4.txt", "ALLPAKA_Q4_NR0=4  (8 lanes/row, 4 phases)"),
    ("ladder1-geom-tg64.txt", "ALLPAKA_MV_TG=64"),
    ("ladder1-geom-tg256.txt", "ALLPAKA_MV_TG=256"),
    ("ladder1-geom-base2.txt", "NR0 default, repeated last (drift control)"),
]
# The unthrottled `q8_0` streamed ceiling this machine has ever repeated:
# 495.6 and 497.0 GB/s at [4096,1536] in one process, 2026-09-24 18:20 UTC, kept
# as the window gate by correction #10 of ../2026-09-24-indexed-matvec/README.md.
# It is a *shape-specific* reading, so it gates a window rather than calibrating
# one: a ladder's own meter wanders 373-621 GB/s across its members.
CEIL = 495.6
# Every ladder has exactly one q4_k member with nb=4, so that is the reference
# shape for the per-pass pairing and the tables are comparable across ladders.
REF_NB = 4
ROW = re.compile(
    r"^(\S+)\s+\[\s*(\d+),\s*(\d+)\] nb=(\d+)\s+rows=(\d+)\s+sets=(\d+)(.*)"
)
# The bench prints value-then-label, so `::` follows the `rmed` label with no
# number of its own: `... 198.7 best 171.3 med 0.502 rmax 0.477 rmed :: <samples>`.
CELL = re.compile(
    r"feed=(\d+)\s+([\d.]+) best\s+([\d.]+) med\s+([\d.]+) rmax\s+([\d.]+) "
    r"rmed\s*::((?: [\d.]+)+)"
)


def load(fn):
    """`{(file, ladder, label, n_out, n_in): {feed: [GB/s per pass]}}`."""
    out, lad = {}, None
    for ln in open(os.path.join(D, fn)):
        m = re.match(r"### ladder (\d+): (\d+) blocks", ln)
        if m:
            lad = int(m.group(1))
            continue
        m = ROW.match(ln)
        if not m or lad is None:
            continue
        label, no, ni, nb, _rows, _sets, rest = m.groups()
        feeds = {
            int(fw): [float(t) for t in raw.split()]
            for fw, _b, _m, _rmax, _rmed, raw in CELL.findall(rest)
        }
        assert int(nb) == int(ni) // 256 or label == "q8_0", ln
        out[(fn, lad, label, int(no), int(ni))] = feeds
    return out


def paired(tab, key, ref, feed):
    """Per-pass cost ratio `key`/`ref` (both `1/rate`), so >1 means slower per
    block. Returns the list over the passes both rows have."""
    a, b = tab[key].get(feed), tab[ref].get(feed)
    if not a or not b:
        return []
    n = min(len(a), len(b))
    return [b[i] / a[i] for i in range(n)]


def summarise(v):
    if not v:
        return None
    s = sorted(v)
    return {
        "n": len(v),
        "med": st.median(s),
        "lo": s[0],
        "hi": s[-1],
        "spread": s[-1] / s[0],
    }


def ref_of(tab, fn, lad, label, feed):
    ks = [
        k
        for k in tab
        if k[0] == fn and k[1] == lad and k[2] == label and k[4] // 256 == REF_NB
    ]
    return ks[0] if ks else None


def phase_cost(nb, lanes_per_pass=4):
    """Blocks a 4-phase row actually pays for: whole iterations of 4 slots."""
    return lanes_per_pass * math.ceil(nb / lanes_per_pass)


tab = {}
for f in FILES:
    tab.update(load(f))
lads = {k[1] for k in tab}
feeds = sorted({fw for v in tab.values() for fw in v})
WINS = [(fn, lad) for fn in FILES for lad in sorted(lads) if any(
    k[0] == fn and k[1] == lad for k in tab
)]


def fmt(r, width=7):
    return f"{r:{width}.3f}" if r else " " * width


print("# Narrow-shape rate deficit, measured at fixed dispatch: which axis owns it?\n")
print(
    "Question. The published streamed table (correction #10 item 5 of\n"
    "../2026-09-24-indexed-matvec/README.md) says a dimension at 768 costs 18-23%\n"
    "of the byte rate: `q4_k` [768,2048] at 386.7 against 472.2 at [4096,1536],\n"
    "`q6_k` [2048,768] at 363.9 against 471.7. That table cannot answer *why*, and\n"
    "not because of contention: its narrow rows are 6144 blocks per dispatch and\n"
    "its reference row 24576, so a 'shape' difference there is also a dispatch-size\n"
    "difference. The named candidate was the block count - `n_in` 768 is three\n"
    "256-element K-quant blocks per row against 6-16 elsewhere - and the rival was\n"
    "row count, since a 768-wide row is also a 2x smaller `n_out` in one of the two.\n"
    "\n"
    "DoD. Say which axis the deficit tracks, and price it against the 30B's decode\n"
    "wall, or state that it tracks neither and name what does. Two ladders of\n"
    "knobs are in scope: dispatch geometry (`ALLPAKA_Q4_NR0`, `ALLPAKA_MV_TG`) and\n"
    "the period-4 phase model the kernel's own block step implies. Nothing in\n"
    "between: a number for the deficit without an axis is not an answer.\n"
    "\n"
    "Harness. `cargo test --release -p allpaka-backend --test gpu_ffnbench --\n"
    " narrow_dim_isomer_ladders --ignored --nocapture`, five isomer ladders whose\n"
    "members all dispatch the same number of quant blocks - so identical bytes per\n"
    "dispatch, identical footprint, identical `n_out * (n_in/256)` - while `n_out`\n"
    "varies 4-5x within a ladder and `nb = n_in/256` runs 2-16. Ladder 1\n"
    "(49152 blocks per dispatch) is the 30B's own size and contains both of its\n"
    "rows: `q4_k` [768,2048] at nb=8 and `q6_k` [2048,768] at nb=3. 384 MiB\n"
    "streamed per shape, four interleaved passes, `slots=8`, shared activation.\n"
    "`ALLPAKA_BENCH_LADDER_ONLY`, `ALLPAKA_BENCH_REPS`, `ALLPAKA_BENCH_FEEDS` and\n"
    "`ALLPAKA_BENCH_REVERSE` are the arm knobs; each log file in this directory is\n"
    "one process.\n"
    "\n"
    "Admission. Nothing here is an absolute. The in-run `q8_0` meter of these\n"
    "windows sits anywhere from 0.42 to 1.25 of the 495.6 GB/s ceiling (section 1)\n"
    "and pass 1 to pass 8 of one row decays by up to 2.4x (section 0), so every\n"
    "crossed statistic is a per-pass paired ratio against the nb=4 member of the\n"
    "*same* ladder, and section 8 shows the pairing stops at the ladder boundary.\n"
    "\n"
    "Generated by `analysis.py` in this directory from the logs beside it; the\n"
    "tables below are its stdout, so no number is retyped from the bench.\n"
)

print("\n### 0. Throttle state: rate by pass index inside one row\n")
print("  Each row's passes are printed in order by the bench, so this is the shape\n"
      "  of the window: pass 1 against every later pass, averaged over the run's\n"
      "  `q4_k` rows. A monotone decline is the machine leaving burst mode, not a\n"
      "  shape effect, and it is why nothing below crosses rows unpaired.\n")
print("  run                                 feed  pass1  pass2  pass3  pass4  last  n")
for fn in FILES:
    for fw in feeds:
        ks = [k for k in tab if k[0] == fn and k[2] == "q4_k" and tab[k].get(fw)]
        if not ks:
            continue
        n = min(len(tab[k][fw]) for k in ks)
        col = []
        for p in [0, 1, 2, 3, n - 1]:
            col.append(st.mean(tab[k][fw][p] / tab[k][fw][0] for k in ks))
        print(
            f"  {fn:<35} {fw:>5} " + "".join(f" {c:6.3f}" for c in col) + f"  {n:>3}"
        )

print("\n### 1. Window gate: the in-run `q8_0` meter against the 495.6 GB/s ceiling\n")
print("  Per ladder, because a ladder is one contiguous window and its members are\n"
      "  the paired unit. `meter` is that ladder's `q8_0` best/median over its own\n"
      "  shapes: if `med` is far under the ceiling the absolutes are contended, and\n"
      "  if `best` is over it the window is partly cache- or burst-resident. Either\n"
      "  way only the paired ratios below are quoted.\n")
print("  run                                 feed  ladder  blocks  meter best  meter med  best/CEIL  med/CEIL")
for fn, lad in WINS:
    for fw in feeds:
        ks = [
            k
            for k in tab
            if k[0] == fn and k[1] == lad and k[2] == "q8_0" and tab[k].get(fw)
        ]
        if not ks:
            continue
        bmax = max(max(tab[k][fw]) for k in ks)
        bmed = st.median([st.median(tab[k][fw]) for k in ks])
        blocks = ks[0][3] * (ks[0][4] // 256) * 8
        print(
            f"  {fn:<35} {fw:>5}  {lad:>6}  {blocks:>7}  {bmax:10.1f}  {bmed:9.1f}  "
            f"{bmax/CEIL:9.3f}  {bmed/CEIL:8.3f}"
        )

print("\n### 2. Row count is exonerated: cost per block along a fixed-bytes ladder\n")
print("  Every member of a ladder dispatches the same block count, so the same bytes\n"
      "  and the same footprint; `n_out` (and so the row count) still varies 4-5x.\n"
      "  `cost/blk` is the per-pass paired ratio against the ladder's own `nb=4`\n"
      "  member (>1 = more time per block than `nb=4`), median over its passes.\n")
for fw in feeds:
    print(f"\n  feed={fw}")
    for fn, lad in WINS:
        head = [k for k in tab if k[0] == fn and k[1] == lad and k[2] == "q4_k"]
        ref = ref_of(tab, fn, lad, "q4_k", fw)
        if not head or ref is None:
            continue
        print(f"  ladder {lad} ({len(head)} shapes), {fn[:26]}, ref [{ref[3]},{ref[4]}]")
        print("     nb  shape        rows      cost/blk med [lo,hi]  pass-spread   q6_k med  q8_0 med")
        for k in sorted(head, key=lambda k: k[4]):
            nb = k[4] // 256
            s = summarise(paired(tab, k, ref, fw))
            if not s:
                continue
            k6, k8 = (fn, lad, "q6_k", k[3], k[4]), (fn, lad, "q8_0", k[3], k[4])
            r6 = summarise(paired(tab, k6, ref, fw)) if k6 in tab else None
            r8 = summarise(paired(tab, k8, ref, fw)) if k8 in tab else None
            print(
                f"    {nb:>3}  [{k[3]:>5},{k[4]:>4}] {k[3]*8:<8}  "
                f"{fmt(s['med'])}  [{s['lo']:.3f},{s['hi']:.3f}]     {s['spread']:5.3f}"
                f"     {fmt(r6['med'] if r6 else 0)}      {fmt(r8['med'] if r8 else 0)}"
            )

print("\n### 3. The sawtooth probe: `nb` 3..12 at constant total blocks (55440)\n")
print("  `matvec_q4_k_mv` walks a row's blocks as `for (ib = ix; ib < nb; ib += 4)`,\n"
      "  with `ix = tiisg/8` so a row's block loop runs `ceil(nb/4)` whole iterations\n"
      "  of four slots - four block phases - and `simd_sum` closes the four `ix`\n"
      "  groups. The 4-phase model therefore charges a row `4*ceil(nb/4)` block slots\n"
      "  and its cost per block, against `nb=4`, is `(4*ceil(nb/4)/nb) / 1`.\n"
      "  A deficit at `nb=3` is one idle slot per phase of the one iteration it runs,\n"
      "  and `nb=6` is two idle slots in each phase's last iteration.\n")
print("  Measured column is the paired per-pass ratio from section 2's rule.\n")
for fn in ["ladders-run2-with-sawtooth.txt", "sawtooth-rep-A.txt", "sawtooth-rep-B.txt"]:
    ks = sorted(
        [k for k in tab if k[0] == fn and k[1] == 4 and k[2] == "q4_k"],
        key=lambda k: k[4],
    )
    if len(ks) < 4:
        continue
    print(f"\n  {fn}")
    for fw in feeds:
        ref = ref_of(tab, fn, 4, "q4_k", fw)
        if ref is None:
            continue
        print(f"   feed={fw}   ref nb={REF_NB}  [{ref[3]},{ref[4]}]")
        print("    nb  n_out    rows     measured med [lo,hi]  spread   4-phase   nb%4")
        for k in ks:
            nb = k[4] // 256
            s = summarise(paired(tab, k, ref, fw))
            if not s:
                continue
            print(
                f"    {nb:>3} {k[3]:>6} {k[3]*8:>8}   {fmt(s['med'])} "
                f"[{s['lo']:.3f},{s['hi']:.3f}]  {s['spread']:6.3f}   "
                f"{(phase_cost(nb)/nb)/(phase_cost(REF_NB)/REF_NB):7.3f}   {nb%4:>3}"
            )

print("\n### 4. The `nb` 3-vs-4 hole, window by window, paired\n")
print("  The one shape deficit that appears in every window: at equal bytes, `nb=3`\n"
      "  rows cost more per block than `nb=4` rows in the same ladder. `q4(3)/q4(4)`\n"
      "  is the paired rate ratio (so <1 means the `nb=3` shape is slower per byte);\n"
      "  `cost ratio` is its reciprocal, per pass, median and extremes.\n")
print("  run                                 feed  ladder  blocks  [nb=3]      [nb=4]      rate ratio med [lo,hi]   cost ratio med [lo,hi]  4-phase")
for fw in feeds:
    for fn, lad in WINS:
        k3 = [k for k in tab if k[0] == fn and k[1] == lad and k[2] == "q4_k" and k[4] == 768]
        k4 = [k for k in tab if k[0] == fn and k[1] == lad and k[2] == "q4_k" and k[4] == 1024]
        if not k3 or not k4:
            continue
        c = summarise(paired(tab, k3[0], k4[0], fw))
        if not c:
            continue
        rate = [1 / v for v in paired(tab, k3[0], k4[0], fw)]
        r = summarise(rate)
        print(
            f"  {fn:<35} {fw:>5}  {lad:>6}  {k3[0][3]*3*8:>7}  "
            f"[{k3[0][3]:>5},  768] [{k4[0][3]:>5}, 1024]  "
            f"{r['med']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]   "
            f"{c['med']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]   "
            f"{(phase_cost(3)/3)/(phase_cost(4)/4):.3f}"
        )

print("\n### 5. Does the `nb=3` cost ratio depend on blocks per dispatch?\n")
print("  Inside one process the ladders run in order, so a later ladder is also a\n"
      "  later, more throttled window - which is a competing explanation for any\n"
      "  trend here. The two dedicated sawtooth runs sit at the top of the block\n"
      "  count and start cold, so they break the tie only against run2's ladder 4,\n"
      "  which is the same shapes at the end of a long run.\n")
for fw in feeds:
    print(f"\n  feed={fw}")
    print("   blocks/dispatch  q4_k cost/blk at nb=3 (paired med, all windows)     window")
    for blocks in sorted({k[3] * (k[4] // 256) * 8 for k in tab if k[2] == "q4_k"}):
        vals = []
        for fn, lad in WINS:
            k3 = [k for k in tab if k[0] == fn and k[1] == lad and k[2] == "q4_k" and k[4] == 768]
            if not k3 or k3[0][3] * 3 * 8 != blocks:
                continue
            ref = ref_of(tab, fn, lad, "q4_k", fw)
            if ref is None:
                continue
            s = summarise(paired(tab, k3[0], ref, fw))
            if s:
                vals.append((s["med"], fn, lad))
        if vals:
            med = st.median([v[0] for v in vals])
            print(
                f"    {blocks:>8}      {med:6.3f}  [min {min(v[0] for v in vals):.3f}"
                f" max {max(v[0] for v in vals):.3f}]  n={len(vals)}: "
                + ", ".join(f"{v[1][:14]}:{v[2]}" for v in vals)
            )

print("\n### 6. Resolution floor: how much of a difference this machine will settle\n")
print("  Two numbers. `within-window` is the pass-to-pass spread of the paired ratio\n"
      "  for one shape pair - the error bar on every cell above. `window-to-window`\n"
      "  is the same paired ratio's medians across runs, which is what an absolute\n"
      "  claim has to beat.\n")
for fn in FILES:
    for fw in feeds:
        sp, rat = [], []
        for k in tab:
            if k[0] != fn or k[2] != "q4_k":
                continue
            ref = ref_of(tab, fn, k[1], "q4_k", fw)
            if ref is None or k == ref:
                continue
            s = summarise(paired(tab, k, ref, fw))
            if s and s["n"] > 2:
                sp.append(s["spread"])
                rat.append(s["med"])
        if sp:
            print(
                f"  {fn:<35} {fw:>5}  pairs={len(sp):>3}  spread med {st.median(sp):.3f}"
                f" p90 {sorted(sp)[int(0.9*len(sp))-1]:.3f} max {max(sp):.3f}"
                f" | ratio range {min(rat):.3f}-{max(rat):.3f}"
            )

print("\n### 7. Geometry probes: is the `nb=3` cost an occupancy or row-sharing artifact?\n")
print("  In `matvec_q4_k_mv` a row's blocks are walked by `ix = tiisg/8`, so the\n"
      "  block step is four 8-lane slots per SIMD group whatever the settings, and\n"
      "  `NR0 = 32/LPR` only chooses how many *rows* share those four slots. So\n"
      "  `ALLPAKA_Q4_NR0` sweeps rows-per-group (1, 2, 4) and `ALLPAKA_MV_TG` sweeps\n"
      "  the threadgroup size (64, 128, 256); neither is a phase-count knob, and\n"
      "  what a flat answer here rules out is activation-reuse and occupancy, the\n"
      "  two other reasons a 3-block row could cost more than a 4-block one. The\n"
      "  phase model itself is tested by `nb=6`, which idles the same quarter of the\n"
      "  group and shows nothing (section 3).\n")
print("  `q6_k` has no rows-per-group knob (its `lanes_per_row` is a hard 16), so it\n"
      "  is this probe's negative control: only `ALLPAKA_MV_TG` can move it. Each arm\n"
      "  is its own process and the incumbent is re-run last as the drift check.\n")
print("  Cells are the per-pass paired cost ratio against the *same format's*\n"
      "  `[1536,1024]` row of the same ladder, so no column crosses formats.\n")
gtab = {}
for fn, _desc in GEOM:
    if os.path.exists(os.path.join(D, fn)):
        gtab.update(load(fn))
print("  arm                                       q4_k nb=3   nb=6   nb=8  | q6_k nb=3   nb=6   nb=8  | q8_0 nb=3  pass-spread")
for fn, desc in GEOM:
    if not os.path.exists(os.path.join(D, fn)):
        print(f"  {desc:<39} (missing {fn})")
        continue
    fw = max(f for k in gtab if k[0] == fn for f in gtab[k])
    cells, spreads = [], []
    for label in ["q4_k", "q6_k", "q8_0"]:
        ref = [k for k in gtab if k[0] == fn and k[2] == label and k[4] == 1024 and k[3] == 1536]
        for nb in [3, 6, 8]:
            ni = nb * 256
            k = [k for k in gtab if k[0] == fn and k[2] == label and k[4] == ni]
            s = summarise(paired(gtab, k[0], ref[0], fw)) if k and ref else None
            cells.append(f"{s['med']:5.3f}" if s else "     -")
            if s and label == "q4_k":
                spreads.append(s["spread"])
    print(
        f"  {desc:<39} "
        + " ".join(cells[:3]) + "  | " + " ".join(cells[3:6]) + "  | "
        + cells[6] + f"   {max(spreads) if spreads else 0:5.3f}"
    )

print("\n### 8. Where the pairing stops: cross-ladder ratios are not a statistic\n")
print("  Every claim above compares shapes inside one ladder, where the members are\n"
      "  visited seconds apart. Comparing *ladders* at the same `nb` reuses the pass\n"
      "  index across a gap of minutes, so it is unpaired in exactly the way the\n"
      "  header of this file says not to be. The test is the same pair measured in\n"
      "  two processes that each held every ladder: if the pairing travelled, the two\n"
      "  windows agree; they do not, by 8-40%, while the within-ladder pair of the\n"
      "  same two windows agrees to 0.2%.\n")
fn1, fn2 = "ladders-run1-feeds8-32.txt", "ladders-run2-with-sawtooth.txt"
fw = 32
print("  pair (q4_k, rate ratio)                        run1    run2   window disagreement")
pairs = []
for lad in [0, 2, 3]:
    pairs.append((f"ladder {lad} nb=4 / ladder 1 nb=4", lad, 4))
pairs.append(("ladder 1 nb=3 / ladder 1 nb=4 (inside)", 1, 3))
for desc, lad, nb in pairs:
    vals = []
    for fn in [fn1, fn2]:
        k = [k for k in tab if k[0] == fn and k[1] == lad and k[2] == "q4_k" and k[4] // 256 == nb]
        r = [k for k in tab if k[0] == fn and k[1] == 1 and k[2] == "q4_k" and k[4] // 256 == 4]
        s = summarise(paired(tab, k[0], r[0], fw)) if k and r else None
        vals.append(s["med"] if s else None)
    if None in vals:
        continue
    print(
        f"  {desc:<40} {vals[0]:6.3f} {vals[1]:7.3f}   {max(vals)/min(vals):8.3f}"
    )
print("\n  Consequence, and it is a limit on this file: the dispatch-size axis - the")
print("  other half of what the published table called the 768 deficit, since its")
print("  narrow row is 6144 blocks and its reference row 24576 - cannot be measured")
print("  by pairing here at all. It needs two processes whose absolute meters sit")
print("  within a few percent of each other, which is the gate")
print("  ../2026-09-25-narrow-shape-axes/results.txt")
print("  already sets and this machine has not yet met. What this file does settle is")
print("  the shape axis at fixed dispatch, sections 2-7.")

print("\n### 9. Visit-order control: does the `nb=3` shape own its cost, or the slot?\n")
print("  The ladder arrays are written wide to narrow, so in the forward walk the\n"
      "  `nb=3` member is always the last shape visited in its pass. Reversing the\n"
      "  walk (`ALLPAKA_BENCH_REVERSE=1`, which reverses the row list and so also\n"
      "  the within-shape format order) moves `nb=3` to first. If the deficit is a\n"
      "  shape cost it survives the swap at the same value; if it is a position cost\n"
      "  it moves to whichever shape ends up last. Forward and reversed are adjacent\n"
      "  processes, and the incumbent geometry arm sits between the pair as a third\n"
      "  reading.\n")
ORD = [
    ("order-L0-fwd.txt", 0, "ladder 0, forward"),
    ("order-L0-rev.txt", 0, "ladder 0, reversed"),
    ("order-L1-fwd.txt", 1, "ladder 1, forward"),
    ("order-L1-rev.txt", 1, "ladder 1, reversed"),
    ("order-L4-fwd.txt", 4, "ladder 4, forward"),
    ("order-L4-rev.txt", 4, "ladder 4, reversed"),
]
otab = {}
for fn, _l, _d in ORD:
    if os.path.exists(os.path.join(D, fn)):
        otab.update(load(fn))
print("  window                          q4_k nb=3   nb=6   nb=8  | q6_k nb=3   nb=6   nb=8  | worst pass-spread")
for fn, lad, desc in ORD:
    if not os.path.exists(os.path.join(D, fn)):
        print(f"  {desc:<30} (missing {fn})")
        continue
    fw = max(f for k in otab if k[0] == fn for f in otab[k])
    cells, spread = [], 0.0
    for label in ["q4_k", "q6_k"]:
        ref = [k for k in otab if k[0] == fn and k[2] == label and k[1] == lad and k[4] // 256 == 4]
        for nb in [3, 6, 8]:
            k = [k for k in otab if k[0] == fn and k[2] == label and k[1] == lad and k[4] // 256 == nb]
            s = summarise(paired(otab, k[0], ref[0], fw)) if k and ref else None
            cells.append(f"{s['med']:5.3f}" if s else "    -")
            if s:
                spread = max(spread, s["spread"])
    print(f"  {desc:<30} " + " ".join(cells[:3]) + "  | " + " ".join(cells[3:]) + f"   {spread:8.3f}")

# ---------------------------------------------------------------- 10. verdict
# Every window this directory holds: the four multi-ladder runs, the six
# dispatch-geometry arms (all ladder 1) and the six visit-order arms.
ALLT = {}
WL = [(fn, lad) for fn in FILES for lad in sorted(lads) if any(
    k[0] == fn and k[1] == lad for k in tab
)]
for fn, _desc in GEOM:
    if os.path.exists(os.path.join(D, fn)):
        ALLT.update(load(fn))
        WL.append((fn, 1))
for fn, lad, _desc in ORD:
    if os.path.exists(os.path.join(D, fn)):
        ALLT.update(load(fn))
        WL.append((fn, lad))
ALLT.update(tab)

print("\n### 10. Verdict: three blocks per row, ~4% of the 30B's wall, no knob\n")


def cost(label, nb, lads_ok):
    """Every paired cost ratio for `label` at `nb` blocks/row, over windows."""
    out = []
    for fn, lad in WL:
        if lad not in lads_ok:
            continue
        ref = [k for k in ALLT if k[0] == fn and k[2] == label and k[1] == lad
               and k[4] // 256 == REF_NB]
        ks = [k for k in ALLT if k[0] == fn and k[2] == label and k[1] == lad
              and k[4] // 256 == nb]
        if not ref or not ks:
            continue
        for fw in sorted(ALLT[ks[0]]):
            s = summarise(paired(ALLT, ks[0], ref[0], fw))
            if s:
                out.append((s["med"], f"{fn[:-4]}:L{lad}:f{fw}"))
    return out


print("  Every cell is the per-pass paired cost/blk against the nb=4 member of the\n"
      "  same shape's own ladder, one value per window x feed x ladder. `q8_0` is the\n"
      "  control for the memory shape and not for the block count: at [2048,768] its\n"
      "  rows carry 24 of its own 32-element blocks, and its reference is the\n"
      "  [1536,1024] row at 32 of them.\n")
print("  label   shape          nb  med    min    max    n  | ladder 1 only (the 30B)")
for label, no, ni, nb in [("q4_k", 2048, 768, 3), ("q6_k", 2048, 768, 3),
                          ("q8_0", 2048, 768, 3), ("q4_k", 1024, 1536, 6),
                          ("q6_k", 1024, 1536, 6), ("q4_k", 768, 2048, 8),
                          ("q6_k", 768, 2048, 8)]:
    v = cost(label, nb, {0, 1, 2, 3})
    if not v:
        continue
    ms = [x[0] for x in v]
    l1 = [m for m, w in v if w.endswith((":L1:f8", ":L1:f32"))]
    tail = f"{st.median(l1):5.3f} [{min(l1):.3f},{max(l1):.3f}] n={len(l1)}" if l1 else "-"
    print(f"  {label:<6} [{no:>5},{ni:>5}] {nb:>3}  {st.median(ms):5.3f}"
          f"  {min(ms):5.3f}  {max(ms):5.3f}  {len(ms):>3}  | {tail}")

# The 30B's own dispatch size, window by window, so the spread is visible rather
# than summarised into the table above.
L1 = [x for x in cost("q4_k", 3, {1})], [x for x in cost("q6_k", 3, {1})]
print("\n  At ladder 1, the 30B's size, window by window:")
for lbl, v in zip(["q4_k", "q6_k"], L1):
    print(f"    {lbl} nb=3: " + ", ".join(f"{m:.3f} ({w})" for m, w in sorted(v, key=lambda x: x[1])))

# What the nb=3 hole is worth on the model that dispatches it. Both inputs are
# quoted, not measured here: the 30B's down projection is 27.3% of its decode
# weight bytes and 1.44 ms of its 6.70 ms/token wall at the published 364 GB/s
# (../2026-09-24-per-token-byte-census/time-ledger.txt section 4).
SHARE, MS_DOWN, WALL = 0.273, 1.44, 6.70
r = st.median([m for m, _ in cost("q6_k", 3, {1})])
rec = MS_DOWN * (1 - 1 / r)
print(
    f"\n  Price. The 30B's rows are `q4_k` [768,2048] at nb=8 (1.00 x its own\n"
    f"  reference, i.e. no shape cost) and `q6_k` [2048,768] at nb=3 ({r:.3f} x).\n"
    f"  Only the second is a victim, and it is {SHARE:.1%} of the model's decode\n"
    f"  bytes = {MS_DOWN:.2f} ms of a {WALL:.2f} ms/token wall\n"
    f"  (../2026-09-24-per-token-byte-census/time-ledger.txt section 4). Removing a\n"
    f"  {r:.2f}x cost there recovers {MS_DOWN:.2f}*(1-1/{r:.2f}) = {rec:.2f} ms/token\n"
    f"  = {100 * rec / WALL:.1f}% of the wall, at or under this machine's e2e noise.\n"
    f"  The 0.95 ms / +14% ceiling quoted by\n"
    f"  ../2026-09-25-narrow-shape-axes/results.txt assumed both rows were shape\n"
    f"  victims; the paired data above says one is and one is not."
)

print(
    "\n  Mechanism, and the list of what it is not. The kernel walks a row's blocks\n"
    "  in four 8-lane slots (`ix = tiisg/8` in `matvec_q4_k_mv`), so a period-4\n"
    "  phase model predicts 4*ceil(nb/4)/nb = 1.333 at nb=3 *and the same 1.333 at\n"
    "  nb=6*. Measured: 1.27 and 1.05. Section 3 sweeps nb 3..12 at constant total\n"
    "  blocks and finds no period-4 sawtooth, so quantisation is not the law and\n"
    "  nb=3 is a hole rather than a tooth. Row count is exonerated by section 2\n"
    "  (`n_out` varies 4-5x at fixed bytes with nb in {4,6,8} flat to a few\n"
    "  percent), the byte pipe by `q8_0` at the identical 768-wide shape, occupancy\n"
    "  and row sharing by section 7's six dispatch-geometry arms, visit position by\n"
    "  section 9's forward/reverse pair. No knob in this build reaches the hole.\n"
    "\n"
    "  Verdict, and the one axis this instrument cannot see. The published '768\n"
    "  costs 18-23%' splits in two: its `q6_k` half is this nb=3 hole and is worth\n"
    "  ~4% of the 30B's wall, and its `q4_k` half is not a shape cost at all - it is\n"
    "  ladder 1 against ladder 3, i.e. 49152 blocks per dispatch against 196608,\n"
    "  which is exactly the comparison section 8 shows pairing cannot support. The\n"
    "  nb=3 cost ratio does drift with that axis (0.96 -> 1.27 -> 1.30 -> 1.32 across\n"
    "  the four ladder sizes in section 5), but those are four different windows in\n"
    "  one process, so it is a lead and not a result.\n"
    "\n"
    "  Reopen DoD, either arm. (a) A `matvec_q4_k_mv` / `matvec_q6_k_mv` arm that\n"
    "  changes the fixed four-slot block step for 3-block rows - a different lane\n"
    "  split, not another value of `ALLPAKA_Q4_NR0` or `ALLPAKA_MV_TG`, both of which\n"
    "  are measured flat here - and must bring the ladder-1 nb=3 paired cost from\n"
    "  ~1.22 to below 1.05 with `gpu_parity::indexed_matvec_matches_cpu_reference`\n"
    "  clean, then be gated e2e on the 30B against this file's ~4% ceiling. (b) Two\n"
    "  processes whose absolute `q8_0` meters agree within 3%, which is the only\n"
    "  admissible way to price the blocks-per-dispatch axis; ~16 s per run, and the\n"
    "  gate is the one ../2026-09-25-narrow-shape-axes/results.txt already sets.\n"
)
