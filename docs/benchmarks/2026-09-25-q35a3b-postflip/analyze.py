#!/usr/bin/env python3
"""Summarise the 35B post-flip pairs written by run-pairs.sh.

Reads the JSON that `ALLPAKA_BENCH_REPORT` writes and takes ms/token from
`.measurements[decode].summary.median` at full float, never from the stdout
seconds column (preregistration.txt, "Precision": that print steps at 0.3125
ms/token at TG=32, more than the low end of the band under test).

Point the OUT directory at the run: `python3 analyze.py <dir>`.
"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "."
DISPATCH = re.compile(r"gpu during decode:.*?(\d+) dispatches")
PREFILL_BAND = (1412.0, 1449.0)  # tok/s, from ../qwen35-35b-m4-max-pending.md


def read(arm, i):
    j = os.path.join(D, f"allpaka-{arm}-{i}.json")
    if not os.path.exists(j):
        return None
    m = json.load(open(j))["measurements"]
    dec = next(x for x in m if x["name"] == "decode")
    pre = next(x for x in m if x["name"] == "prefill")
    t = os.path.join(D, f"allpaka-{arm}-{i}.txt")
    disp = None
    if os.path.exists(t):
        g = DISPATCH.search(open(t).read())
        disp = int(g.group(1)) if g else None
    return {
        "dec": dec["summary"]["median"],
        "ms": 1000.0 / dec["summary"]["median"],
        "pre": pre["summary"]["median"],
        "disp": disp,
    }


def llama(i):
    """llama-bench `-o json` writes one record per test as a flat array element;
    `avg_ts` is its tok/s and `samples_ts` the per-repeat list (one here)."""
    j = os.path.join(D, f"llama-tg-{i}.json")
    if not os.path.exists(j):
        return None
    r = json.load(open(j))[0]
    assert r["n_gen"] == 32 and r["type_k"] == "f16" and r["type_v"] == "f16", j
    return r["avg_ts"]


# Two eras of capture live in this directory's history: run1 (`q35-postflip-run1`)
# wrote one plain arm per pair, and `run-pairs.sh` as it stands now also writes a
# `plain2` repeater. Reading a repeater-less capture under the current gates
# admitted 0/10 and printed no rows at all, which is indistinguishable from "all
# ten pairs failed their gates" - and it hid the fact that the published "four
# admitted pairs of run1, mean 0.1195, SEM 0.1566"
# (stationarity-vs-level.txt) could not be reproduced by the command that
# documents it. So the repeater gate is applied only when the capture has a
# repeater, and a pair that cannot be read says which arm is missing.
HAS_REP = bool(glob.glob(os.path.join(D, "allpaka-plain2-*.json")))


def plain_pair(i):
    p1 = read("plain", i)
    return p1, (read("plain2", i) if HAS_REP else p1)


pairs = []
for f in sorted(glob.glob(os.path.join(D, "allpaka-plain-*.json"))):
    pairs.append(int(re.search(r"plain-(\d+)\.json", f).group(1)))
print(f"  pairs found: {len(pairs)} {sorted(pairs)}"
      f"{'  (+ plain2 repeater arms)' if HAS_REP else '  (no plain2 repeater in this capture)'}\n")
print("  #   plain ms  plain2 ms  rep  |  fused ms  delta  pre-ratio  llama"
      "   disp p/f     gates")
d, kept, kept_plain, unreadable = [], 0, [], []
for i in pairs:
    p1, p2, f = *plain_pair(i), read("fused", i)
    if not (p1 and p2 and f):
        missing = "/".join(n for n, r in (("plain", p1), ("plain2", p2), ("fused", f))
                           if r is None)
        unreadable.append(i)
        print(f"  {i:>2}  UNREADABLE - no {missing} arm file in {D}")
        continue
    l = llama(i)
    ms = sorted([p1["ms"], p2["ms"]])
    rep = ms[1] / ms[0]
    pres = sorted([p1["pre"], p2["pre"], f["pre"]])
    pr = pres[-1] / pres[0]
    g1 = pr <= 1.01
    g2 = p1["disp"] and f["disp"] and p1["disp"] != f["disp"]
    g3 = PREFILL_BAND[0] <= pres[-1]
    g4 = rep <= 1.02
    pla = (p1["ms"] + p2["ms"]) / 2
    if g1 and g2 and g4:
        kept += 1
        d.append(f["ms"] - pla)
        kept_plain.append(pla)
    print(f"  {i:>2} {p1['ms']:9.4f} {p2['ms']:10.4f} {rep:5.3f}  |"
          f" {f['ms']:8.4f} {f['ms'] - pla:+7.4f} {pr:9.4f}"
          f" {(l or 0):7.2f}   {p1['disp']}/{f['disp']}"
          f"  {'pair OK ' if g1 else 'pair REJ'} {'census OK' if g2 else 'census EQ '}"
          f" {'rep OK ' if g4 else 'rep REJ'} {'band OK  ' if g3 else 'band MISS'}")

gates = "pair + census + repeater" if HAS_REP else "pair + census (no repeater arm)"
print(f"\n  admitted on {gates} gates: {kept}/{len(pairs) - len(unreadable)} pairs read"
      + (f"; {len(unreadable)} unreadable {unreadable}" if unreadable else ""))
if kept == 0 and len(pairs) == len(unreadable):
    print("  nothing could be read from this directory: it is not a capture this tool"
          " understands, which is not the same as every pair failing a gate.")
    sys.exit(1)
if d:
    sem = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    pos = sum(1 for x in d if x > 0)
    pm = st.mean(kept_plain)
    print(f"  delta fused-plain = mean {st.mean(d):+.4f} ms/token, SEM {sem:.4f},"
          f" median {st.median(d):+.4f}, signs {pos}/{len(d)} positive (fused slower)")
    print(f"  admitted plain mean {pm:.4f} ms/token = {1000/pm:.2f} tok/s;"
          f" the fold costs {100*st.mean(d)/pm:.1f}% of per-token time")
    print(f"  pair spread {min(d):+.4f}..{max(d):+.4f}; |mean|/SEM = "
          f"{abs(st.mean(d))/sem:.1f}" if sem == sem else "  n=1, no SEM")

rat = []
for i in pairs:
    p1, p2, l = *plain_pair(i), llama(i)
    if p1 and p2 and l:
        rat.append((1000 / ((p1["ms"] + p2["ms"]) / 2)) / l)
if rat:
    rat.sort()
    print(f"\n  allpaka(shipped)/llama decode, paired in this window: median"
          f" {st.median(rat):.3f} (n={len(rat)}, {min(rat):.3f}-{max(rat):.3f});"
          f" the published 1.35 was the fused regime in a 2026-09-12 window,\n"
          f"  against llama build b29c606e2 (10964) / ggml 0.24.0 here.")
