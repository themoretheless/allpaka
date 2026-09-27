#!/usr/bin/env python3
"""Summarise the NORMFLAG-family pairs written by drive.py.

Two null series (`control` = NORMFLAG=1 vs unset, `policy` = NORMFLAG=0 vs unset) both
compare arms the probe showed to encode identically, so they are pooled into one noise
floor - that is the whole reason the 23:57Z amendment kept the `policy` series. The three
family series are then judged against that floor, never against the 3 % hope, and the
dispatch census is reported per pair because it is the structural claim.

Rates come from `.measurements[decode].summary.median` in the per-arm report JSON, never
from the stdout seconds column. Point the directory at the run: `python3 analyze.py DIR`.

Two admission rules are re-derived here instead of trusting the run's own flags:

* **Drift is per arm.** The run vetoed a pair when its slowest arm fell under
  DRIFT_VETO x the median of every arm seen so far, across every series. An arm that
  genuinely runs 18 % slow then looks exactly like a machine that slowed down, so the
  veto deleted the effect it was built to find - 8 of the first 10 rfuse pairs died on
  it. Comparing each arm against its own history keeps a permanent difference and still
  catches a whole-machine slide.
* **A ratio is challenge over reference**, never first slot over second: in a BA pair
  the first slot IS the reference, and slot order reports the reciprocal of the effect
  with the opposite sign.

`python3 analyze.py --selfcheck` proves both on synthetic rows; each case is written to
fail on the rule it replaces.
"""
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

NULL_SERIES = ["control", "policy"]
FAMILY = ["rfuse", "rmt", "rtopk"]
DECISION_BAND = 3.0  # percent, criterion 5 of preregistration.txt
DRIFT_VETO = 0.85    # the run's own strictness; only the reference changes here

SERIES_REF = {
    "control": ("null", "norm"),
    "policy": ("off", "norm"),
    "rfuse": ("rf", "norm"),
    "rmt": ("rmt", "norm"),
    "rtopk": ("k", "norm"),
}


def arms(r):
    ch = SERIES_REF[r["kind"]][0]
    ref = SERIES_REF[r["kind"]][1]
    d = {r["first"]["arm"]: r["first"], r["second"]["arm"]: r["second"]}
    return d[ch], d[ref]


def ratio(r):
    c, f = arms(r)
    return c["decode_tok_s"] / f["decode_tok_s"]


def annotate_dir(directory):
    """One pass over every series in time order, re-deciding the drift veto per arm."""
    every = []
    for name in NULL_SERIES + FAMILY:
        path = os.path.join(directory, f"{name}-pairs.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            r["series"] = name
            every.append(r)
    every.sort(key=lambda r: min(r[k]["at"] for k in ("first", "second") if k in r))
    hist = {}
    for r in every:
        veto = bool(r.get("load_vetoed"))
        why = ["load"] if veto else []
        for k in ("first", "second"):
            arm, rate = r[k]["arm"], r[k]["decode_tok_s"]
            h = hist.setdefault(arm, [])
            ref = st.median(h) if len(h) >= 3 else None
            if ref and rate < DRIFT_VETO * ref:
                veto = True
                why.append(f"drift:{arm} {rate:.2f} < {DRIFT_VETO:.0%} of its own "
                           f"running median {ref:.2f}")
            h.append(rate)
        r["recorded_veto"] = bool(r.get("load_vetoed") or r.get("drift_vetoed"))
        r["admitted"] = not veto
        r["recovered"] = r["recorded_veto"] and r["admitted"]
        r["veto_why"] = why
    return {name: [r for r in every if r["series"] == name] for name in NULL_SERIES + FAMILY}


def slot(arm, minute, rate, disp):
    return {"arm": arm, "at": f"2026-09-26T00:{minute:02d}:00Z",
            "decode_tok_s": rate, "dispatches": disp,
            "outside_gpu_pct": 1, "census_secs": 2.6}


def fixture_rows():
    """Six null pairs at ~131, then rfuse pairs whose `rf` arm is a steady ~110.

    The run's pooled rule vetoes every one of those rfuse pairs, because the effect *is*
    the gap. A real machine slide (pair 5, both arms down together) and a load veto
    (pair 6) must still be caught.
    """
    rows = []

    def pair(kind, n, order, ch_rate, ref_rate, load=False):
        ch, ref = SERIES_REF[kind]
        ch_slot, ref_slot = (slot(ch, 10 * n, ch_rate, 208000),
                             slot(ref, 10 * n + 1, ref_rate, 215680))
        rows.append({
            "kind": kind, "pair": n, "order": order,
            "first": ch_slot if order == "AB" else ref_slot,
            "second": ref_slot if order == "AB" else ch_slot,
            **({"load_vetoed": True} if load else {}),
            # the flag the run's pooled rule would have written against its ~131 pool
            **({"drift_vetoed": True}
               if min(ch_rate, ref_rate) < DRIFT_VETO * 131 else {}),
        })

    for n in range(1, 7):
        pair("control", n, "AB" if n % 2 else "BA", 131 + n * 0.1, 131.2 - n * 0.1)
    for n in range(1, 5):
        pair("rfuse", n, "AB" if n % 2 else "BA", 107 + n, 131 - n * 0.2)
    pair("rfuse", 5, "AB", 88.0, 92.0)
    pair("rfuse", 6, "BA", 109.0, 130.0, load=True)
    return rows


def selfcheck():
    with tempfile.TemporaryDirectory() as tmp:
        by_series = {}
        for r in fixture_rows():
            by_series.setdefault(r["kind"], []).append(r)
        for name, rs in by_series.items():
            with open(os.path.join(tmp, f"{name}-pairs.jsonl"), "w") as fh:
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
        table = annotate_dir(tmp)
        rfuse = table["rfuse"]
        ab = [r for r in rfuse if r["order"] == "AB" and r["admitted"]]
        ba = [r for r in rfuse if r["order"] == "BA" and r["admitted"]]
        report = subprocess.run([sys.executable, os.path.abspath(__file__), tmp],
                                capture_output=True, text=True)
        cases = [
            ("a steady 18 % treatment arm is not drift (4 pairs recovered)",
             sum(1 for r in rfuse if r["admitted"] and r["recovered"]) == 4),
            ("the run's own rule really did veto those four",
             all(r["recorded_veto"] for r in rfuse[:4])),
            ("a pair where both arms slide together is still vetoed",
             not [r for r in rfuse if r["pair"] == 5 and r["admitted"]]),
            ("a load veto is respected whatever the rates",
             not [r for r in rfuse if r["pair"] == 6 and r["admitted"]]),
            ("BA pairs keep the AB sign",
             bool(ab) and bool(ba)
             and all(ratio(r) < 1 for r in ab) and all(ratio(r) < 1 for r in ba)),
            ("null pairs are all admitted", all(r["admitted"] for r in table["control"])),
            ("the report runs on the fixture and marks the recovered pairs",
             report.returncode == 0 and "RECOVERED" in report.stdout),
        ]
    fails = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if fails:
        print(report.stdout, report.stderr)
    print(f"{len(fails)} failure(s) of {len(cases)}")
    return 1 if fails else 0


if "--selfcheck" in sys.argv:
    sys.exit(selfcheck())

D = next((a for a in sys.argv[1:] if not a.startswith("-")), ".")
TABLE = annotate_dir(D)


def rows(name):
    return TABLE.get(name, [])


def admitted(r):
    return r["admitted"]


print(f"  # analysis of {os.path.basename(os.path.abspath(D))}"
      f"  (decision band {DECISION_BAND}%, drift per arm at {DRIFT_VETO:.0%})")
null_ratios = []
for name in NULL_SERIES + FAMILY:
    rs = rows(name)
    if not rs:
        continue
    ok = [r for r in rs if admitted(r)]
    rec = sum(1 for r in rs if not r["recorded_veto"])
    print(f"\n  {name} ({SERIES_REF[name][0]} vs {SERIES_REF[name][1]}):"
          f" {len(ok)}/{len(rs)} pairs admitted"
          + (f" (the run's own flags admitted {rec})" if rec != len(ok) else ""))
    ds = []
    for r in ok:
        c, f = arms(r)
        q = c["decode_tok_s"] / f["decode_tok_s"]
        dd = c["dispatches"] - f["dispatches"]
        ms = 1000 / c["decode_tok_s"] - 1000 / f["decode_tok_s"]
        ds.append(q)
        flag = "  TOKENS DIFFER" if r.get("tokens_differ") else ""
        flag += "  RECOVERED" if r.get("recovered") else ""
        print(f"    pair {r['pair']:>2} {r['order']}  ratio {q:.6f} ({(q-1)*100:+.2f}%)"
              f"  ms/token {ms:+.4f}  dispatches {dd:+d} ({dd/320:+.1f}/token)"
              f"  outside {f['outside_gpu_pct']}%/{c['outside_gpu_pct']}%"
              f"  census {f['census_secs']}/{c['census_secs']}s{flag}")
    for r in rs:
        if not admitted(r):
            print(f"    pair {r['pair']:>2} dropped: {', '.join(r['veto_why'])}")
    if name in NULL_SERIES:
        null_ratios += ds
        if ds:
            print(f"    null: median {st.median(ds):.6f}, min {min(ds):.6f}"
                  f" max {max(ds):.6f}, {sum(1 for x in ds if x < 1)}/{len(ds)} below 1")
    elif ds:
        print(f"    median {st.median(ds):.6f} ({(st.median(ds)-1)*100:+.2f} %),"
              f" mean {st.mean(ds):.6f} ({(st.mean(ds)-1)*100:+.2f} %),"
              f" SEM {st.stdev(ds)/len(ds)**0.5*100:.2f} pp,"
              f" range {min(ds):.6f}...{max(ds):.6f},"
              f" {sum(1 for x in ds if x < 1)}/{len(ds)} below 1")
        dd = {c["dispatches"] - f["dispatches"] for r in ok for c, f in [arms(r)]}
        print(f"    dispatch census, all admitted pairs: {sorted(dd)}"
              f" = {sorted({v/320 for v in dd})} per token")

if null_ratios:
    band = (max(null_ratios) - min(null_ratios)) * 100
    print(f"\n## noise floor from {len(null_ratios)} null pairs"
          f" ({'+'.join(s for s in NULL_SERIES if rows(s))})")
    print(f"  band {band:.2f} pp, median off centre"
          f" {abs((st.median(null_ratios)-1)*100):.2f} pp")
    print(f"  criterion 2: the instrument resolves {'NO' if band >= DECISION_BAND else 'YES'}"
          f" better than {DECISION_BAND} %; levels below are"
          f" {'interval-only' if band >= DECISION_BAND else 'publishable'}.")
    for name in FAMILY:
        ds = [ratio(r) for r in rows(name) if admitted(r)]
        if not ds:
            continue
        med, m = st.median(ds), st.mean(ds)
        sem = st.stdev(ds) / len(ds) ** 0.5 if len(ds) > 1 else float("nan")
        below = sum(1 for x in ds if x < 1)
        # A sign count is only informative against the null's own below-1 rate.
        null_below = sum(1 for x in null_ratios if x < 1)
        n_null, n_real = len(null_ratios), len(ds)
        pooled = null_below + below
        from math import comb
        p = (comb(n_real, below) * comb(n_null, null_below) / comb(n_null + n_real, pooled)
             if pooled >= max(below, null_below) and n_real + n_null >= pooled else float("nan"))
        print(f"\n  {name}: median {(med-1)*100:+.2f} %, mean {(m-1)*100:+.2f} %,"
              f" SEM {sem*100:.2f} pp, {below}/{n_real} below 1"
              f" (null put {null_below}/{n_null} below 1;")
        print(f"    P(this split | pooled {pooled} below 1 over {n_null+n_real} pairs)"
              f" = {p:.3g}; effect {'BEATS' if abs((med-1)*100) > band else 'INSIDE'}"
              f" the {band:.2f} pp floor)")
else:
    print("\n  no null pairs admitted - no band, so no family level can be read")
    sys.exit(1)
