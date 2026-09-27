#!/usr/bin/env python3
"""Task #16 analysis, implementing criteria 1-9 of preregistration.txt.

Reads the per-arm JSON that drive.py leaves behind at full float - never the
driver's human-readable lines, which is criterion-0 of task #26 - and applies:

  criterion 2  arm validity against each arm's own history: llama decode
               45.46 +- 0.21 anchor, usable band 35-60 tok/s; allpaka decode
               usable at or above 30 tok/s (39.0 pre-port, ~46 post-port);
  criterion 7  the in-run thermometer that replaced trust in the load gate: a
               pair whose allpaka prefill is below 320 tok/s is disturbed
               regardless of its load reading;
  criterion 5  a pair whose gate load exceeded 5 is counted and named, and no
               "allpaka is faster" claim may rest on those pairs;
  criterion 1  the primary statistic is the median per-pair decode ratio, and a
               parity or win claim requires the AB and BA halves to agree in sign;
  criterion 9  fewer than 3 admitted of the first 12 pairs -> abandon the run.

    python3 analyze.py DIR [more dirs...]

Default DIR is the newest .airbug-bench/glm-parity-* run.
"""

import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LLAMA_TG = (35.0, 60.0)
AP_TG_MIN = 30.0      # criterion 2
AP_PP_MIN = 320.0    # criterion 7
GATE = 5.0               # criterion 5


def usable(r):
    why = []
    if not LLAMA_TG[0] <= r["ll_tg"] <= LLAMA_TG[1]:
        why.append(f"llama decode {r['ll_tg']:.2f} outside its {LLAMA_TG[0]:.0f}-{LLAMA_TG[1]:.0f} band (anchor 45.46)")
    if r["ap_tg"] < AP_TG_MIN:
        why.append(f"allpaka decode {r['ap_tg']:.2f} below {AP_TG_MIN:.0f} (anchors 39.0 pre-port, ~46 post-port)")
    if r["ap_pp"] < AP_PP_MIN:
        why.append(f"allpaka prefill {r['ap_pp']:.1f} below {AP_PP_MIN:.0f} - disturbed window")
    return why


def report(name, rows):
    if not rows:
        print(f"{name:34s} n= 0")
        return
    ratios = [r["ap_tg"] / r["ll_tg"] for r in rows]
    pct = [100.0 * (x - 1.0) for x in ratios]
    n = len(ratios)
    sem = st.stdev(pct) / n**0.5 if n > 1 else float("nan")
    print(f"{name:34s} n={n:2d}  ratio median {st.median(ratios):.3f}  mean {st.mean(pct):+6.2f}%  "
          f"SEM {sem:5.2f}  signs {sum(1 for x in ratios if x > 1)}/{n}  "
          f"ap {st.median([r['ap_tg'] for r in rows]):5.2f}  ll {st.median([r['ll_tg'] for r in rows]):5.2f}")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        dirs = [sorted(glob.glob(os.path.join(ROOT, ".airbug-bench", "glm-parity-*")))[-1]]
    rows = []
    for d in dirs:
        for line in open(os.path.join(d, "pairs.jsonl")):
            r = json.loads(line)
            r["dir"] = d
            rows.append(r)
    rows.sort(key=lambda r: (r["dir"], r["pair"]))
    rows = [r for r in rows if r["pair"] > 1]  # criterion: the warm pair is discarded
    print(f"# GLM-4.5-Air decode parity, allpaka vs llama, tg32")
    print(f"  {len(dirs)} run dir(s), {len(rows)} non-warm pairs recorded\n")
    good = [r for r in rows if not usable(r)]
    hot = [r for r in good if r["loadmax"] >= GATE]
    # drive.py recorded, until 2026-09-25, only the load it read before each arm
    # and only after the launch gate had let it through - so every such reading
    # is under the ceiling, and `hot` is empty by construction rather than empty
    # because the window was quiet. The driver now also reads the load after each
    # arm (`loadpre_max` marks the captures that have it).
    pre_only = all("loadpre_max" not in r for r in rows)
    vacuous = " (vacuous: this capture stores pre-arm load readings only)" if pre_only else ""
    print(f"{'admitted (criteria 2 and 7)':34s} {len(good)} of {len(rows)}")
    print(f"{'  of those, gate load >= 5':34s} {len(hot)}{vacuous}  - criterion 5: an "
          f"'allpaka is faster' claim may not rest on these pairs")
    first12 = rows[:12]
    if len(first12) == 12:
        n = len([r for r in first12 if not usable(r)])
        print(f"{'stop rule, first 12 pairs':34s} {n}/12 usable -> "
              + ("STOP, the window will not carry this run" if n < 3 else "continue"))
    print()
    report("all admitted pairs", good)
    report("  AB (allpaka arm first)", [r for r in good if r["order"] == "AB"])
    report("  BA (llama arm first)", [r for r in good if r["order"] == "BA"])
    if good:
        ratios = sorted(r["ap_tg"] / r["ll_tg"] for r in good)
        print(f"\n  spread of admitted ratios: {ratios[0]:.3f} .. {ratios[-1]:.3f}")
        print(f"  the status row's 0.86x (2026-09-12, pre-q5_0_mv, llama build 10809) becomes {st.median(ratios):.2f}x")
    print("\n  pair  order  ap_tg   ll_tg  ratio   ap_pp   ll_pp  loadmax  verdict")
    for r in rows:
        u = usable(r)
        print(f"  {r['pair']:4d}  {r['order']}  {r['ap_tg']:6.2f} {r['ll_tg']:7.2f} {r['ap_tg'] / r['ll_tg']:6.3f}  "
              f"{r['ap_pp']:6.1f} {r['ll_pp']:7.1f}  {r['loadmax']:7.2f}  "
              + ("admitted" if not u else "; ".join(u)))


if __name__ == "__main__":
    main()
