#!/usr/bin/env python3
"""Paired-by-rotation re-measure of ALLPAKA_Q4_NR0 / Q8_NR0 in the indexed x8 geometry.

Implements ../2026-09-25-nr0-indexed-remeasure/preregistration.txt (written 2026-09-25
21:40Z, before any counted round). One arm = one process of the existing
`gpu_ffnbench::indexed_matvecs_report_effective_bandwidth`; arms rotate across rounds.

    python3 drive-nr0.py --out DIR            # both series (Q4 then Q8)
    python3 drive-nr0.py --series Q4 --rounds 5
    python3 drive-nr0.py --selfcheck          # parser/gates on a fixture, no GPU
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = pathlib.Path("/Users/themoretheless/Documents/Sources/allpaka")
TEST = "indexed_matvecs_report_effective_bandwidth"
LOAD_VETO = 8.0
LAUNCH_LOAD = 5.0
METER_BAND = 0.85          # gate 2: a process whose meter is <85% of the reference
SPREAD_GATE = 1.05         # gate 3, implementable form: best/med on every q4_k row
# A third series joins the same instrument rather than getting a copied driver:
# the arms, rotation, meter, gates and flip rule are the 2026-09-25
# pre-registration's, and only the knob and the shape list differ. `Q50` is the
# `ALLPAKA_Q5_0_NR0` sweep the 09-26 B1/B2 confirmations surfaced.
ARMS = {"Q4": ["1", "2", "4"], "Q8": ["1", "2", "4"], "Q50": ["1", "2", "4"]}
DEFAULT = {"Q4": "2", "Q8": "2", "Q50": "2"}
ENV_OF = {"Q4": "ALLPAKA_Q4_NR0", "Q8": "ALLPAKA_Q8_NR0", "Q50": "ALLPAKA_Q5_0_NR0"}
FEEDS = "1,8"
SWFUSES = "0,1"
METER = "q6_k down"
SHAPES = {
    "Q4": ["q4_k", "q4_k control", "q4_k 30b gate/up", "q4_k glm gate/up"],
    "Q8": ["q8_0", "q8_0 glm down"],
    # GLM-4.5-Air's expert down is [4096,1408]; the 1536-wide row is the shape
    # the harness has carried since the q5_0 port, kept so the two are comparable.
    "Q50": ["q5_0 down", "q5_0 glm down"],
}

ROW_RE = re.compile(
    r"^(?P<label>\S.*?)\s+\[(?P<no>\d+),(?P<ni>\d+)\] sets=(?P<sets>\d+) "
    r"(?P<sc>scatter|contig)\s+(?P<xs>x/slot|x/shared)\s+(?P<fu>fused|plain)\s+"
    r"(?P<cells>.*)$"
)
CELL_RE = re.compile(r"feed=(?P<feed>\d+) (?P<best>[\d.]+) best (?P<med>[\d.]+) med "
                     r"(?P<rel>[\d.]+)x (?P<gw>[\d.]+) GW/s")


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load1():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True,
                         text=True).stdout
    return float(out.split()[1])


def parse_rows(text):
    """{label: {(arm, feed): (best, med, rel)}}"""
    rows = {}
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        arm = f"{m['sc'].strip()} {m['xs']} {m['fu']}"
        for c in CELL_RE.finditer(m["cells"]):
            rows.setdefault(m["label"].strip(), {})[(arm, int(c["feed"]))] = (
                float(c["best"]), float(c["med"]), float(c["rel"]))
    return rows


def wait_quiet(limit_s=25 * 60, step_s=20.0):
    waited = 0.0
    while load1() > LAUNCH_LOAD and waited < limit_s:
        time.sleep(step_s)
        waited += step_s
    return waited


def run_process(series, arm, out_dir, tag):
    env = {**os.environ, ENV_OF[series]: arm,
           "ALLPAKA_BENCH_FEEDS": FEEDS, "ALLPAKA_BENCH_SWFUSES": SWFUSES,
           "ALLPAKA_BENCH_XPER_SLOT": "0", "ALLPAKA_BENCH_SCATTERS": "0"}
    log = out_dir / f"{series}-{arm}-{tag}.log.txt"
    lb = load1()
    t0 = time.monotonic()
    proc = subprocess.run([str(out_dir / "gpu_ffnbench"), TEST, "--ignored",
                           "--exact", "--nocapture"],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    la = load1()
    text = proc.stdout + proc.stderr
    log.write_text(text)
    rows = parse_rows(text)
    rec = {"series": series, "arm": arm, "tag": tag, "started": utc(),
           "elapsed_s": round(time.monotonic() - t0, 1),
           "load_before": lb, "load_after": la, "returncode": proc.returncode,
           "panic": "panicked" in text or "msl" in text.lower() and "compile" in text.lower(),
           "rows": {lab: {f"{a}|{f}": v for (a, f), v in cells.items()}
                    for lab, cells in rows.items()}}
    return rec


def gate_process(rec, series, meter_ref):
    """Return (admitted, reason). Gates 2,3,4 of the pre-registration."""
    if rec["returncode"] != 0 or not rec["rows"]:
        return False, "process failed or printed no rows"
    if max(rec["load_before"], rec["load_after"]) > LOAD_VETO:
        return False, f"load veto ({rec['load_before']}, {rec['load_after']})"
    meter = rec["rows"].get(METER, {})
    key = f"contig x/shared plain|8"
    mv = meter.get(key)
    if mv is None:
        return False, "meter row absent"
    if meter_ref is not None and mv[0] < METER_BAND * meter_ref:
        return False, f"meter veto ({mv[0]:.1f} under {METER_BAND} of {meter_ref:.1f})"
    for lab in SHAPES[series]:
        cell = rec["rows"].get(lab, {}).get(key)
        if cell is None:
            return False, f"shape {lab} absent"
        if cell[0] / cell[1] > SPREAD_GATE:
            return False, f"spread veto on {lab} ({cell[0]:.1f}/{cell[1]:.1f})"
    return True, ""


def ratio(rec, series, shape, feed, fuse):
    arm = f"contig x/shared {'fused' if fuse else 'plain'}"
    a = rec["rows"].get(shape, {}).get(f"{arm}|{feed}")
    b = rec["rows"].get(METER, {}).get(f"{arm}|{feed}")
    if not a or not b or b[0] == 0:
        return None
    return a[0] / b[0]


def median(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# The regime the verdict is read in: feed=8 (a decode step is one command buffer
# with ~1300 dispatches) and plain (swiglu fusion is opt-in since 2026-09-24, so
# the shipped pipeline does not fuse the down). The other three cells are printed
# alongside as secondary evidence but cannot flip a default on their own.
PRIMARY = (8, False)


def decide(kept, series, rounds):
    """Pre-registration decision rule, on the per-round shape/meter ratios.

    An arm wins a cell if its median ratio over the admitted rounds beats the
    default's median by >3% AND the per-round sign is positive in at least 4 of
    the rounds both arms were admitted for. The series flips only if one arm
    wins >=3 of the four shapes (Q8 has two shapes, so it needs both).
    """
    lines, flip, winner = [], False, None
    feed, fuse = PRIMARY
    shapes = SHAPES[series]
    need = 3 if len(shapes) == 4 else len(shapes)
    for arm in [a for a in ARMS[series] if a != DEFAULT[series]]:
        won, cells = 0, []
        for shape in shapes:
            pairs = [(r["arm"], r.get("tag", "solo"), ratio(r, series, shape, feed, fuse))
                     for r in kept if r["series"] == series]
            dvals = [v for a, _, v in pairs if a == DEFAULT[series] and v]
            avals = [v for a, _, v in pairs if a == arm and v]
            base, alt = median(dvals), median(avals)
            if not base or not alt:
                cells.append(f"{shape}: n/a")
                continue
            pct = 100 * (alt / base - 1)
            # per-round sign, over the rounds both arms contributed to
            by_tag = {}
            for a, t, v in pairs:
                if v:
                    by_tag.setdefault(t, {})[a] = v
            shared = [d for d in by_tag.values() if DEFAULT[series] in d and arm in d]
            signs = sum(1 for d in shared if d[arm] > d[DEFAULT[series]])
            ok = pct > 3.0 and signs >= max(4, len(shared) - 1) and len(shared) >= 4
            won += ok
            cells.append(f"{shape}: {pct:+.2f}% ({signs}/{len(shared)} rounds)")
        lines.append((arm, won >= need, cells))
        if won >= need:
            flip, winner = True, arm
    return lines, flip, winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--series", default="Q4,Q8", help="comma list of Q4,Q8,Q50")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--start-round", type=int, default=1,
                    help="first round number to run; with an existing provenance.json "
                         "in --out this continues that series, which requires the "
                         "newly built binary to have the same sha256")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    out = args.out.resolve()  # run_process launches with cwd=REPO, so it must be absolute
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.splitlines()
    src = None
    for cand in (REPO / "target/release/build", ):
        for p in cand.rglob("gpu_ffnbench-*"):
            if p.is_file() and os.access(p, os.X_OK) and not p.name.endswith(".d"):
                src = p
    if src is None:
        print("no built gpu_ffnbench binary; run cargo test --release --no-run first")
        return 2
    pin = out / "gpu_ffnbench"
    prior = out / "provenance.json"
    resume = json.loads(prior.read_text()) if prior.exists() else None
    if resume and resume["binary_sha256"] != hashlib.sha256(src.read_bytes()).hexdigest():
        # The pre-registration forbids mixing two binaries in one series, so a
        # continuation only runs against the byte-identical test binary.
        print("binary changed since the pinned series - refusing to mix two builds")
        return 3
    subprocess.run(["cp", str(src), str(pin)], check=True)
    sha = hashlib.sha256(pin.read_bytes()).hexdigest()
    # One meter reference per series, carried across a continuation so the
    # 15% band means the same thing in round 1 and round 9.
    meter_seed = {}
    if resume:
        store = out / "records.jsonl"
        if store.exists():
            for line in store.read_text().splitlines():
                rec = json.loads(line)
                ref = rec.get("meter_reference")
                if ref and rec["series"] not in meter_seed:
                    meter_seed[rec["series"]] = ref
    prov = resume or {"started": utc(), "head": head, "tree_dirty_files": len(dirty),
            "binary_from": str(src), "binary_sha256": sha,
            "feeds": FEEDS, "swfuses": SWFUSES, "veto_load": LOAD_VETO,
            "meter_band": METER_BAND, "spread_gate": SPREAD_GATE}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"pinned {pin.name} sha256 {sha[:16]}  head {head[:8]}  "
          f"dirty files {len(dirty)}  "
          f"{'continuing from round ' + str(args.start_round) if resume else 'new series'}",
          flush=True)

    kept, records = [], []
    store = out / "records.jsonl"
    if resume and store.exists():
        # The decision rule needs the earlier rounds too, so they are read back
        # from the append-only log rather than re-measured.
        records = [json.loads(line) for line in store.read_text().splitlines()]
        kept = [r for r in records if r.get("admitted")]
    for series in [s.strip() for s in args.series.split(",")]:
        meter_ref = meter_seed.get(series)
        order = ARMS[series]
        for ri in range(args.start_round - 1, args.start_round - 1 + args.rounds):
            r = ri
            arms = order[r % len(order):] + order[: r % len(order)]
            for arm in arms:
                w = wait_quiet()
                rec = run_process(series, arm, out, f"r{r+1}")
                ok, why = gate_process(rec, series, meter_ref)
                rec["admitted"], rec["veto"] = ok, why
                if ok and meter_ref is None:
                    meter_ref = rec["rows"][METER]["contig x/shared plain|8"][0]
                    rec["meter_reference"] = meter_ref
                records.append(rec)
                with store.open("a") as h:
                    h.write(json.dumps(rec) + "\n")
                kept.append(rec) if ok else None
                line = (f"{series} NR0={arm} r{r+1} "
                        f"{'ok ' if ok else 'VETO ' + why} "
                        f"load {rec['load_before']:.1f}/{rec['load_after']:.1f} "
                        f"{rec['elapsed_s']:.0f}s")
                if ok:
                    q = ratio(rec, series, SHAPES[series][0], 8, False)
                    line += f"  ratio[0,plain,feed8]={q:.4f}" if q else "  ratio n/a"
                print(line, flush=True)
    print(f"{sum(1 for r in records if r['admitted'])}/{len(records)} processes admitted")
    decisions = {}
    for series in [s.strip() for s in args.series.split(",")]:
        lines, flip, winner = decide(kept, series, args.rounds)
        decisions[series] = {"flip": flip, "winner": winner,
                             "cells": {a: {"won": w, "per_shape": c}
                                       for a, w, c in lines}}
        for arm, won, cells in lines:
            print(f"  {series} vs NR0={DEFAULT[series]}: "
                  f"{'FLIP' if won else 'hold'}  " + "  ".join(cells), flush=True)
    (out / "report.json").write_text(json.dumps(
        {"provenance": prov, "decisions": decisions, "records": records}, indent=2))
    return 0


def selfcheck():
    fixture = """
--- indexed x8, 384/512 MB streamed per format ---
q4_k           [4096,1536] sets=16 contig  x/shared plain  | feed=1 240.0 best 236.0 med 1.00x 1152 GW/s | feed=8 260.0 best 250.0 med 1.02x 1250 GW/s
q4_k control   [1536,4096] sets=22 contig  x/shared plain  | feed=1 200.0 best 198.0 med 1.00x 960 GW/s | feed=8 210.0 best 205.0 med 1.02x 1010 GW/s
q6_k down      [4096,1536] sets=11 contig  x/shared plain  | feed=1 400.0 best 395.0 med 1.00x 1160 GW/s | feed=8 400.0 best 398.0 med 1.00x 1160 GW/s
q6_k down      [4096,1536] sets=11 contig  x/shared fused   | feed=1 390.0 best 385.0 med 1.00x 1130 GW/s | feed=8 390.0 best 388.0 med 1.00x 1130 GW/s
q4_k           [4096,1536] sets=16 contig  x/shared fused  | feed=1 120.0 best 118.0 med 1.00x 576 GW/s | feed=8 125.0 best 124.0 med 1.02x 600 GW/s
"""
    bad = 0
    cases = [
        ("the label, shape and arm tags are parsed", lambda r: (
            r["q4_k"] and list(r["q4_k"])[0][0] == "contig x/shared plain")),
        ("feed cells are keyed by feed", lambda r: set(
            f for (_, f) in r["q6_k down"]) == {1, 8}),
        ("best is the first number and med the second", lambda r: r["q4_k"][("contig x/shared plain", 8)] == (260.0, 250.0, 1.02)),
        ("a fused arm is a separate cell, not a rename", lambda r: r["q4_k"][("contig x/shared fused", 8)][0] == 125.0),
    ]
    rows = parse_rows(fixture)
    for name, ok in cases:
        res = bool(ok(rows))
        bad += 0 if res else 1
        print(f"  {'ok ' if res else 'FAIL'} {name}")
    rec = {"series": "Q4", "arm": "2", "returncode": 0, "load_before": 4.0,
           "load_after": 5.0, "rows": {k: {f"{a}|{f}": v for (a, f), v in c.items()}
                                       for k, c in rows.items()}}
    ok, why = gate_process(rec, "Q4", meter_ref=400.0)
    res = not ok and "absent" in why
    bad += 0 if res else 1
    print(f"  {'ok ' if res else 'FAIL'} a fixture missing two of the four q4_k rows "
          f"is vetoed, not silently admitted ({why})")
    # shapes the fixture omits must veto, not silently pass
    rec2 = json.loads(json.dumps(rec))
    rec2["rows"]["q4_k 30b gate/up"] = rec2["rows"]["q4_k"]
    rec2["rows"]["q4_k glm gate/up"] = rec2["rows"]["q4_k"]
    ok2, _ = gate_process(rec2, "Q4", meter_ref=400.0)
    bad += 0 if ok2 else 1
    print(f"  {'ok ' if ok2 else 'FAIL'} with all four shapes present a clean process is admitted")
    rec3 = json.loads(json.dumps(rec2)); rec3["load_after"] = 9.4
    ok3, why3 = gate_process(rec3, "Q4", 400.0)
    bad += 0 if (not ok3 and "load" in why3) else 1
    print(f"  {'ok ' if not ok3 and 'load' in why3 else 'FAIL'} a load above 8 vetoes ({why3})")
    rec4 = json.loads(json.dumps(rec2))
    rec4["rows"]["q6_k down"]["contig x/shared plain|8"] = [300.0, 299.0, 1.0]
    ok4, why4 = gate_process(rec4, "Q4", 400.0)
    bad += 0 if (not ok4 and "meter" in why4) else 1
    print(f"  {'ok ' if not ok4 and 'meter' in why4 else 'FAIL'} a meter 25% down vetoes ({why4})")
    rec5 = json.loads(json.dumps(rec2))
    rec5["rows"]["q4_k"]["contig x/shared plain|8"] = [500.0, 400.0, 1.0]
    ok5, why5 = gate_process(rec5, "Q4", 400.0)
    bad += 0 if (not ok5 and "spread" in why5) else 1
    print(f"  {'ok ' if not ok5 and 'spread' in why5 else 'FAIL'} a best/med spread over 5% vetoes ({why5})")
    r = ratio(rec2, "Q4", "q4_k", 8, False)
    bad += 0 if r and abs(r - 260.0 / 400.0) < 1e-9 else 1
    print(f"  {'ok ' if r else 'FAIL'} the ratio is shape/meter best-GBps at feed 8 ({r})")

    def fake(arm, tag, mult):
        """Four q4_k rows at `mult` times half the meter, so ratio = 0.5*mult."""
        cells = {lab: {"contig x/shared plain|8": [200.0 * mult, 200.0 * mult, 1.0],
                       "contig x/shared fused|8": [100.0 * mult, 100.0 * mult, 1.0]}
                 for lab in SHAPES["Q4"]}
        cells["q6_k down"] = {"contig x/shared plain|8": [400.0, 400.0, 1.0],
                              "contig x/shared fused|8": [400.0, 400.0, 1.0]}
        return {"series": "Q4", "arm": arm, "tag": tag, "rows": cells}

    def series_with(arm4):
        kept = []
        for i in range(5):
            kept.append(fake("2", f"r{i+1}", 1.0))
            kept.append(fake("4", f"r{i+1}", arm4[i]))
            kept.append(fake("1", f"r{i+1}", 1.0))
        return kept

    dcases = [
        ("+5% on all four shapes, 5/5 rounds, flips",
         series_with([1.05] * 5), True),
        ("+2% on all four shapes is inside the gate and holds",
         series_with([1.02] * 5), False),
        ("+5% median but 3/5 round sign does not flip",
         series_with([1.05, 1.05, 1.05, 0.945, 0.945]), False),
    ]
    for name, kept_s, want in dcases:
        _, got, _ = decide(kept_s, "Q4", 5)
        bad += 0 if got == want else 1
        print(f"  {'ok ' if got == want else 'FAIL'} {name} (got flip={got})")
    thin = series_with([1.05] * 5)
    for rec in thin:
        if rec["arm"] == "4":
            rec["rows"].pop("q4_k glm gate/up")
    _, got, _ = decide(thin, "Q4", 5)
    vetoed = not gate_process({**[r for r in thin if r["arm"] == "4"][-1],
                               "returncode": 0, "load_before": 1.0,
                               "load_after": 1.0}, "Q4", 400.0)[0]
    res = got and vetoed
    bad += 0 if res else 1
    print(f"  {'ok ' if res else 'FAIL'} a challenger missing one shape still clears the "
          f"literal >=3-of-4 bar (flip={got}), so the shape-presence gate is what "
          f"defends it (process vetoed={vetoed})")

    total = len(cases) + 10
    print(f"selfcheck: {total - bad} of {total} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
