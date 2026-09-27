#!/usr/bin/env python3
"""GLM-4.5-Air decode parity: gates A1-A6 and the pre-registered statistic.

Reads only the JSON each engine wrote and the per-process GPU counters, never the
driver's human-readable lines. Every gate can only remove a pair; the removals and
their reasons are printed, because a verdict that silently drops pairs is not
auditable. See preregistration.txt beside this file for what each gate is for.

    python3 analyze-glm-parity.py OUT_DIR
"""

import glob
import json
import os
import re
import statistics as st
import sys

NULL_FLOOR_PP = 8.28          # 2026-09-25-dcomb-ab identical-code control series
NULL_FLOOR_PESSIMISTIC = 17.56  # 2026-09-25-normflag-policy
LLAMA_BAND = (35.0, 60.0)     # A4, around the 45.46 anchor of 2026-09-12-matrix
AP_TG_MIN = 30.0              # A4
LOAD_VETO = 8.0               # A5
LOAD_BIAS = 5.0               # criterion 5: no win claim above this
CO_TENANT = ("remoting_me2me_host", "fvid")   # A6
CO_TENANT_CPU = 25.0


def load_series(out):
    pts = []
    p = os.path.join(out, "loads.jsonl")
    if os.path.exists(p):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if "load" in r:
                pts.append((r["t"], r["load"]))
            else:
                print(f"  note: sampler dropped a tick at {r.get('t')}: {r.get('error')}")
    return sorted(pts)


def max_in(series, lo, hi):
    vals = [v for (t, v) in series if lo - 2 <= t <= hi + 2]
    return max(vals) if vals else None


def gpu_counters(path):
    """`% outside the GPU` and the executing/waited ms pair from allpaka's own log."""
    txt = open(path).read()
    m = re.search(r"gpu during decode:.*?\((\d+)% outside the GPU\)", txt)
    c = re.search(r"gpu clock during decode: executing (\d+) ms.*?\(of (\d+) ms waited\)", txt)
    return (int(m.group(1)) if m else None,
            (int(c.group(1)), int(c.group(2))) if c else None)


def co_tenant(snaps, lo, hi):
    hits = []
    for s in snaps:
        if not (lo - 60 <= s["t"] <= hi + 60):
            continue
        for b in s["busy"]:
            name = b["comm"]
            if any(k in name for k in CO_TENANT) and (b["cpu"] >= CO_TENANT_CPU or name.startswith("fvid")):
                hits.append(f"{name}@{b['cpu']:.0f}%")
    return sorted(set(hits))


def swap_used(snaps, lo, hi):
    """A7: the memory state as a recorded covariate, not a gate. The parked 2026-09-24
    attempt failed with swap at 8.86 of 9.2 GB, and this campaign starts with the
    machine at ~9 GB swapped, so the number is printed per pair and the decision is
    left to the absolute arm bands (A3/A4), which are what caught that attempt."""
    out = []
    for s in snaps:
        if lo - 60 <= s["t"] <= hi + 60 and s.get("swapusage"):
            m = re.search(r"used = ([\d.]+)M", s["swapusage"])
            if m:
                out.append(float(m.group(1)) / 1024.0)
    return max(out) if out else None


def main():
    out = sys.argv[1]
    pairs = [json.loads(l) for l in open(os.path.join(out, "pairs.jsonl")) if l.strip()]
    prov = json.load(open(os.path.join(out, "provenance.json")))
    cal = json.load(open(os.path.join(out, "calibration.json")))
    series = load_series(out)
    snaps = [json.loads(l) for l in open(os.path.join(out, "snaps.jsonl"))] if \
        os.path.exists(os.path.join(out, "snaps.jsonl")) else []
    floor = cal.get("band_floor")
    counted = [p for p in pairs if not p.get("prime")]
    prime = [p for p in pairs if p.get("prime")]
    print(f"# GLM-4.5-Air decode parity, allpaka vs llama, pp{prov['pp']}/tg{prov['tg']}")
    print(f"  binary {prov['allpaka_sha256'][:16]}  HEAD {prov['head'][:8]}  "
          f"dirty {prov['tree_dirty_files']}  llama {prov['llama_bench'].splitlines()[-1] if prov.get('llama_bench') else '-'}")
    print(f"  thermometer (A3): calibration prefill medians {[round(c['pp'], 1) for c in cal['calibration']]} "
          f"-> median {cal['prefill_median']} floor {floor}  (2026-09-24 band was a 320 floor on history 336-374)")
    builds = {p["ll_build"] for p in pairs}
    print(f"  llama builds seen: {sorted(builds)}  load samples {len(series)}  snapshots {len(snaps)}")

    verdicts = []
    for p in counted:
        why = []
        lo = min(p["win"]["ap"][0], p["win"]["ll_pp"][0], p["win"]["ll_tg"][0])
        hi = max(p["win"]["ap"][1], p["win"]["ll_pp"][1], p["win"]["ll_tg"][1])
        ap_load = max_in(series, *p["win"]["ap"])
        ll_load = max_in(series, min(p["win"]["ll_pp"][0], p["win"]["ll_tg"][0]),
                         max(p["win"]["ll_pp"][1], p["win"]["ll_tg"][1]))
        reads = [v for v in (ap_load, ll_load) if v is not None]
        p["loadmax"] = max(reads) if reads else 0.0
        p["swap_gib"] = swap_used(snaps, lo, hi)
        fp = p["fast_path"]
        if not (fp["attempts"] == fp["successes"] == prov["tg"] and fp["declines"] == 0):
            why.append(f"A1 fast path {fp['attempts']}/{fp['successes']}/{fp['declines']}")
        log = os.path.join(out, p["round"], "raw", f"allpaka-{p['pair']}.log")
        outside, clock = gpu_counters(log) if os.path.exists(log) else (None, None)
        p["gpu_outside_pct"], p["gpu_clock"] = outside, clock
        if outside is None or outside > 5:
            why.append(f"A2 {outside}% outside the GPU")
        if clock is None or clock[0] < 0.5 * clock[1]:
            why.append(f"A2 executing {clock[0] if clock else '-'} of waited {clock[1] if clock else '-'} ms")
        if floor and p["ap_pp"] < floor:
            why.append(f"A3 prefill thermometer {p['ap_pp']:.1f} < {floor:.1f}")
        if not LLAMA_BAND[0] <= p["ll_tg"] <= LLAMA_BAND[1]:
            why.append(f"A4 llama decode {p['ll_tg']:.2f} outside {LLAMA_BAND}")
        if p["ap_tg"] < AP_TG_MIN:
            why.append(f"A4 allpaka decode {p['ap_tg']:.2f} < {AP_TG_MIN}")
        if p["loadmax"] >= LOAD_VETO:
            why.append(f"A5 load {p['loadmax']:.2f} >= {LOAD_VETO}")
        ct = co_tenant(snaps, lo, hi)
        if ct:
            why.append("A6 co-tenant " + ",".join(ct))
        verdicts.append((p, why))

    good = [p for p, w in verdicts if not w]
    print(f"\n  admitted {len(good)} of {len(counted)} counted pairs"
          + (f", {len(prime)} priming pairs discarded" if prime else ""))
    for p, w in verdicts:
        if w:
            print(f"    removed round {p['round']} pair {p['pair']} {p['order']}: " + "; ".join(w))
    if len(counted) >= 6 and len([p for p, w in verdicts[:6] if not w]) < 3:
        print("  STOP RULE (criterion 9) fired: fewer than 3 of the first 6 pairs admitted -> "
              "the run reports the instrument, not a ratio")
    hot = [p for p in good if p["loadmax"] > LOAD_BIAS]
    print(f"    of the admitted, sampled load above 5 at some point: {len(hot)} "
          f"(criterion 5: no win claim may rest on these)")

    if not good:
        print("\n  no admitted pairs")
        return 1

    def block(rows, name):
        if not rows:
            print(f"  {name:32s} n= 0")
            return None
        ratios = [r["ap_tg"] / r["ll_tg"] for r in rows]
        pct = [100.0 * (x - 1.0) for x in ratios]
        n = len(ratios)
        sem = st.stdev(pct) / n**0.5 if n > 1 else float("nan")
        print(f"  {name:32s} n={n:2d}  ratio median {st.median(ratios):.4f}  mean {st.mean(pct):+6.2f}%  "
              f"SEM {sem:5.2f}  above parity {sum(1 for x in ratios if x > 1)}/{n}  "
              f"ap {st.median([r['ap_tg'] for r in rows]):6.2f}  ll {st.median([r['ll_tg'] for r in rows]):6.2f}  "
              f"min {min(ratios):.4f} max {max(ratios):.4f}")
        return ratios

    print("\n  per-pair ratios, full float")
    print(f"    {'round':<12} {'ord':<3} {'ap_tg':>7} {'ll_tg':>7} {'ratio':>7} {'ap_pp':>7} "
          f"{'ll_pp':>7} {'loadmax':>7} {'gpu_out':>7} {'exec/wait':>10} {'swapGiB':>7}")
    for p, w in verdicts:
        cw = f"{p['gpu_clock'][0]}/{p['gpu_clock'][1]}" if p.get("gpu_clock") else "-"
        sw = "-" if p.get("swap_gib") is None else f"{p['swap_gib']:.1f}"
        print(f"    {p['round'][-6:]:<12} {p['order']:<3} {p['ap_tg']:7.2f} {p['ll_tg']:7.2f} "
              f"{p['ap_tg'] / p['ll_tg']:7.4f} {p['ap_pp']:7.1f} {p['ll_pp']:7.1f} "
              f"{p['loadmax']:7.2f} {str(p.get('gpu_outside_pct')) + '%':>7} {cw:>10} {sw:>7} "
              + ("" if not w else "  vetoed"))
    if prime:
        print("    priming (discarded): " + "  ".join(
            f"{p['order']} {p['ap_tg'] / p['ll_tg']:.4f}" for p in prime))
    print()
    ratios = block(good, "all admitted pairs")
    block([p for p in good if p["order"] == "AB"], "  AB (allpaka arm first)")
    block([p for p in good if p["order"] == "BA"], "  BA (llama arm first)")

    med = st.median(ratios)
    spread_pp = 100.0 * (max(ratios) - min(ratios))
    ab = [r["ap_tg"] / r["ll_tg"] for r in good if r["order"] == "AB"]
    ba = [r["ap_tg"] / r["ll_tg"] for r in good if r["order"] == "BA"]
    half_gap = abs(100.0 * (st.median(ab) - st.median(ba))) if ab and ba else None
    certifiable = spread_pp < NULL_FLOOR_PP and (half_gap is None or half_gap < NULL_FLOOR_PP)
    print(f"\n  order split: AB {st.median(ab):.4f} ({len(ab)}) vs BA {st.median(ba):.4f} ({len(ba)}) "
          f"-> gap {half_gap:.2f} pp, same sign: "
          f"{(st.median(ab) - 1) * (st.median(ba) - 1) > 0 if ab and ba else '-'}")
    print(f"  cross-pair spread {spread_pp:.2f} pp against the published identical-code null floor "
          f"{NULL_FLOOR_PP} pp (pessimistic bookend {NULL_FLOOR_PESSIMISTIC} pp) -> "
          + ("level certifiable to +-3 %" if certifiable else "level NOT certifiable, publish the interval"))
    print(f"  verdict band: median {med:.4f} -> " +
          ("PARITY (0.95-1.05)" if 0.95 <= med <= 1.05 else
           ("WIN for allpaka (>1.05)" if med > 1.05 else "GAP REMAINS (<0.95), re-date the 0.86x row")))
    print(f"  the status row's 0.86x (2026-09-12, pre-q5_0_mv, llama build 10809) becomes {med:.2f}x")
    print(f"  absolutes: allpaka decode median {st.median([p['ap_tg'] for p in good]):.2f} "
          f"(anchors 39.0 pre-port, 45.95-46.5 post-port)  llama decode median "
          f"{st.median([p['ll_tg'] for p in good]):.2f} (anchor 45.46 at build 10809)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
