#!/usr/bin/env python3
"""Analyze the ALLPAKA_MV_ID paired campaign against preregistration.txt.

Reads only the files drive-mv-id.py leaves behind, at the full float each arm's
report JSON carries. Gates are remove-only and their counts are published; the
decision rule is the one written before the first counted pair.

    python3 analyze-mv-id.py RUN_DIR [more run dirs...]

Ratio convention everywhere: A/B with A = `on`, so >1 means the shipped default is
faster. `null` is an identical encode, so its spread is this run's noise floor, and
`control` (on vs Q5_0_MV=0) is the positive control whose failure voids a null
verdict (criterion 0).
"""

import json
import os
import statistics as st
import sys

AP_TG_MIN = 30.0      # A4
LOAD_VETO = 8.0       # A5
CO_TENANT_CPU = 25.0  # A6
CO_TENANT = ("remoting_me2me_host", "fvid")
CONTROL_BAR = 1.05    # criterion 0: the run must see a published ~16% knob
SHIP_BAR = 0.03       # +/-3% decision threshold
TG_EXPECT = 32


def expect_env(kind, arm):
    """What metadata.resolved_overrides must echo for this arm (amendment 2)."""
    if arm == "on":
        return {"ALLPAKA_MV_ID": None, "ALLPAKA_Q5_0_MV": None}
    if arm == "off":
        return {"ALLPAKA_MV_ID": "0"}
    if arm == "null":
        return {"ALLPAKA_MV_ID": "1"}
    if arm == "q5off":
        return {"ALLPAKA_MV_ID": None, "ALLPAKA_Q5_0_MV": "0"}
    return {}


def arm_problems(rec, floor):
    why = []
    for side in ("A", "B"):
        r = rec[side]
        tag = r["arm"]
        ov = r.get("overrides") or {}
        for k, want in expect_env(rec["kind"], tag).items():
            have = ov.get(k)
            if want is None:
                if have is not None:
                    why.append(f"A0 {tag}: {k}={have!r} leaked into the default arm")
            elif have != want:
                why.append(f"A0 {tag}: {k}={have!r}, expected {want!r}")
        fp = r.get("fast_path") or {}
        if not (fp.get("attempts") == TG_EXPECT and fp.get("successes") == TG_EXPECT
                and fp.get("declines") == 0):
            why.append(f"A1 {tag}: fast path {fp}")
        if r.get("rc") != 0:
            why.append(f"A1 {tag}: exit {r.get('rc')}")
        if r.get("outside_pct") is None or r.get("exec_ms") is None:
            why.append(f"A2 {tag}: GPU counters unparsed")
        else:
            if r["outside_pct"] > 5:
                why.append(f"A2 {tag}: {r['outside_pct']}% outside the GPU")
            if r["clock_wait_ms"] and r["exec_ms"] < 0.5 * r["clock_wait_ms"]:
                why.append(f"A2 {tag}: executing {r['exec_ms']} of {r['clock_wait_ms']} ms waited")
        if floor and r["pp"] < floor:
            why.append(f"A3 {tag}: prefill {r['pp']:.1f} < {floor:.1f}")
        if r["tg"] < AP_TG_MIN:
            why.append(f"A4 {tag}: decode {r['tg']:.2f} < {AP_TG_MIN:.0f}")
    return why


def max_in(loads, w0, w1):
    v = [l["load"] for l in loads if "load" in l and w0 - 1 <= l["t"] <= w1 + 1]
    return max(v) if v else 0.0


def co_tenant(snaps, w0, w1):
    hits = []
    for s in snaps:
        if not (w0 - 60 <= s["t"] <= w1 + 60):
            continue
        for p in s.get("busy", []):
            if any(k in p["comm"] for k in CO_TENANT) and p["cpu"] >= CO_TENANT_CPU:
                hits.append(f"{p['comm']} {p['cpu']:.0f}%")
    return hits


def swap_used(snaps, w0, w1):
    for s in snaps:
        if w0 - 60 <= s["t"] <= w1 + 60 and "used" in s.get("swapusage", ""):
            try:
                return float(s["swapusage"].split("used =")[1].split()[0].rstrip("M")) / 1024.0
            except (IndexError, ValueError):
                return None
    return None


def block(name, pairs, floor, loads, snaps, lines):
    rows = []
    for rec in pairs:
        why = arm_problems(rec, floor)
        # The report JSON is written when the process closes, so its (birth, mtime)
        # pair is a 0-length instant and A5/A6 would attribute covariates to a
        # timestamp instead of to the arm. The stdout log is created at process
        # start, so that is the window; fall back to the JSON only if it is absent.
        wa = rec["A"].get("win_log") or rec["A"]["win"]
        wb = rec["B"].get("win_log") or rec["B"]["win"]
        w0 = min(wa[0], wb[0])
        w1 = max(wa[1], wb[1])
        lm = max_in(loads, w0, w1)
        ct = co_tenant(snaps, w0, w1)
        if lm >= LOAD_VETO:
            why.append(f"A5 in-window load {lm:.2f}")
        if ct:
            why.append(f"A6 co-tenant {', '.join(ct)}")
        ratio = rec["A"]["tg"] / rec["B"]["tg"]
        rows.append((rec, ratio, why, lm, ct))
        sw = swap_used(snaps, w0, w1)
        sws = f"{sw:.1f}" if sw else " -  "
        lines.append(f"  {rec['round']:12s} {name:8s} {rec['order']}  {rec['A']['arm']:>5}/"
                     f"{rec['B']['arm']:<5} {rec['A']['tg']:6.2f}/{rec['B']['tg']:6.2f}  "
                     f"ratio {ratio:6.4f}  pp {rec['A']['pp']:5.1f}/{rec['B']['pp']:5.1f}  "
                     f"load {lm:4.2f}  out {rec['A']['outside_pct']}%/{rec['B']['outside_pct']}%  "
                     f"disp {'same' if rec['A']['dispatches'] == rec['B']['dispatches'] else 'DIFF'}  "
                     f"swap {sws:>4s}  " + ("admitted" if not why else "removed: " + "; ".join(why)))
    ok = [r for r in rows if not r[2]]
    return ok, rows


def stats(pairs):
    if not pairs:
        return None
    ratios = [p for _, p, _, _, _ in pairs]
    pct = [100.0 * (x - 1.0) for x in ratios]
    sem = st.stdev(pct) / len(ratios) ** 0.5 if len(ratios) > 1 else float("nan")
    return {"n": len(ratios), "median": st.median(ratios), "mean_pct": st.mean(pct),
            "sem": sem, "above1": sum(1 for x in ratios if x > 1), "min": min(ratios),
            "max": max(ratios),
            "spread_pp": 100.0 * (max(ratios) / min(ratios) - 1.0) if min(ratios) > 0 else None}


def report(label, ok):
    s = stats(ok)
    if not s:
        print(f"  {label:22s} n= 0   (nothing admissible)")
        return None
    print(f"  {label:22s} n={s['n']:2d}  median {s['median']:.4f}  mean {s['mean_pct']:+6.2f}%  "
          f"SEM {s['sem']:5.2f}  >1 in {s['above1']}/{s['n']}  range {s['min']:.4f}..{s['max']:.4f}  "
          f"spread {s['spread_pp']:.2f} pp")
    return s


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = []
    loads, snaps = [], []
    floor = None
    for d in sys.argv[1:]:
        pj = os.path.join(d, "pairs.jsonl")
        if os.path.exists(pj):
            rows += [json.loads(l) for l in open(pj) if l.strip()]
        for fn, sink in (("loads.jsonl", loads), ("snaps.jsonl", snaps)):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                sink += [json.loads(l) for l in open(p) if l.strip()]
        c = os.path.join(d, "calibration.json")
        if os.path.exists(c):
            floor = json.load(open(c)).get("band_floor") or floor
    counted = [r for r in rows if not r.get("prime")]
    err = [l for l in loads if "error" in l]
    print("# ALLPAKA_MV_ID on GLM-4.5-Air, paired AB/BA, pp480/tg32")
    print(f"  {len(sys.argv) - 1} dir(s), {len(counted)} counted pairs "
          f"({sum(1 for r in rows if r.get('prime'))} priming discarded)")
    print(f"  A3 floor {floor}")
    if err:
        print(f"  NOTE {len(err)} sampler tick(s) errored: {err[0]['error']}")
    commits = {r["A"].get("git_commit") for r in counted} | {r["B"].get("git_commit") for r in counted}
    print(f"  per-process metadata.git_commit seen: {sorted(c for c in commits if c)}")
    if floor is None:
        print("  no calibration.json -> A3 cannot run; the gates below are incomplete")

    lines = []
    out = {}
    for kind in ("real", "null", "control"):
        pairs = [r for r in counted if r["kind"] == kind]
        out[kind] = block(kind, pairs, floor, loads, snaps, lines)
    print("\n  per-pair ledger (ratio is on/other; >1 = shipped default faster)\n")
    print("\n".join(lines))

    print()
    for kind in ("real", "null", "control"):
        ok, allr = out[kind]
        print(f"  {kind}: admitted {len(ok)} of {len(allr)}")
        for rec, _, why, _, _ in allr:
            for w in why:
                print(f"    removed {rec['round']} {rec['order']} {rec['A']['arm']}/{rec['B']['arm']}: {w}")
    real_ok, _ = out["real"]
    null_ok, _ = out["null"]
    ctrl_ok, _ = out["control"]
    first4 = [r for r in counted if r["kind"] == "real"][:4]
    n4 = len([r for r in first4 if not arm_problems(r, floor)])
    if len(first4) == 4:
        print(f"\n  stop rule: {n4}/4 of the first real pairs admitted -> "
              + ("STOP, abandon the campaign" if n4 < 3 else "continue"))

    print()
    rs = report("REAL on/off", real_ok)
    ns = report("NULL floor on/null", null_ok)
    cs = report("CONTROL on/q5off", ctrl_ok)
    if rs:
        ab = report("  real, AB", [r for r in real_ok if r[0]["order"] == "AB"])
        ba = report("  real, BA", [r for r in real_ok if r[0]["order"] == "BA"])
        if ab and ba:
            agree = (ab["median"] - 1) * (ba["median"] - 1) > 0
            print(f"\n  order split: AB {ab['median']:.4f} vs BA {ba['median']:.4f} -> "
                  f"gap {abs(ab['median'] - ba['median']) * 100:.2f} pp, same sign: {agree}")
        else:
            print("\n  order split: one arm-order has no admissible pair, so the "
                  "position check cannot run and the verdict is not certifiable")
    print()
    null_dev = max(abs(x["median"] - 1) for x in [ns] if x) if ns else None
    if ns:
        dev = [abs(r[1] - 1) for r in null_ok]
        null_dev = max(dev)
        print(f"  measured noise floor (null pairs): median |ratio-1| = "
              f"{st.median([100 * d for d in dev]):.2f}%, max = {100 * null_dev:.2f}%")
    if cs is None:
        print("  criterion 0: NO control pairs admissible -> nothing from this run can be "
              "read as 'no effect', and a near-threshold reading is equally uncertifiable")
    else:
        ok0 = cs["median"] >= CONTROL_BAR
        print(f"  criterion 0: control median on/q5off {cs['median']:.4f} "
              f"(bar {CONTROL_BAR}) -> instrument {'CAN' if ok0 else 'CANNOT'} see a ~16% knob on this model")
        if not ok0:
            print("     verdict must be published as 'not resolvable in this window', not as a null")

    # The stop rule outranks the decision rules: a campaign that was abandoned
    # before it could be read has no verdict, however well its surviving pairs
    # happen to line up. Printing a decision anyway is how an n=2 reading becomes
    # somebody's default flip.
    abandoned = len(first4) == 4 and n4 < 3
    print("\n  decision (preregistration, evaluated in order)")
    if abandoned:
        print(f"    ABANDONED under the stop rule ({n4}/4 of the first real pairs "
              "admitted) - the decision below is what the surviving pairs would "
              "have said, published as the reason for the abandonment and NOT as a "
              "verdict on the knob. It may not be quoted as a ship/flip signal.")
    if not rs:
        print("    no admissible real pairs: publish the abandonment, not a number")
    elif not (null_dev is not None):
        print("    no admissible null pairs: no measured floor, so the +3% rule cannot be applied")
    else:
        m = rs["median"] - 1.0
        ab = [r for r in real_ok if r[0]["order"] == "AB"]
        ba = [r for r in real_ok if r[0]["order"] == "BA"]
        same = bool(ab and ba) and ((stats(ab)["median"] - 1) * (stats(ba)["median"] - 1) > 0)
        clears = abs(m) * 100 > 100 * null_dev
        if m >= SHIP_BAR and same and clears:
            print(f"    CONFIRM ON: median {rs['median']:.4f} is +{100 * m:.2f}% >= +3%, "
                  "AB/BA agree, and it exceeds the null floor")
        elif m <= -SHIP_BAR and same and clears:
            print(f"    FLIP-TO-OFF CANDIDATE: median {rs['median']:.4f} is {100 * m:.2f}%, "
                  "AB/BA agree, exceeds the null floor - but a GLM-only number does not "
                  "change a shared default: re-measure on qwen3-30b before touching metal.rs")
        else:
            print(f"    NEGATIVE ROW: median {rs['median']:.4f} ({100 * m:+.2f}%), "
                  f"AB/BA same sign: {same}, exceeds null floor: {clears} -> ON is not "
                  f"distinguishable from OFF on GLM at +/-{100 * null_dev:.2f}%; the ship "
                  "rule stays unmet on this model and the default stands as shipped")
    if rs:
        print(f"\n  absolutes: on median {st.median([r[0]['A']['tg'] for r in real_ok]):.2f}  "
              f"off median {st.median([r[0]['B']['tg'] for r in real_ok]):.2f}  "
              f"llama-parity cross-check: this binary read 46.04 against llama 43.43 in "
              "../2026-09-26-glm-llama-parity/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
