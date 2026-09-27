#!/usr/bin/env python3
"""Statistics for the decode down+combine fold A/B (preregistration.txt).

    python3 analyze.py .airbug-bench/glm-stagger-20260925-0300

Three drop rules, all of them pre-registered (the cold veto in the original
design, the load and drift vetoes in the 21:15Z amendment), applied here rather
than trusted from the driver's flags: `drive.py` as run through 22:40Z never
appended to its drift reference pool, so its drift veto could not fire, and a
row's `drift_vetoed` flag is therefore absent from the captures rather than
false. This script recomputes that rule from the timestamps and rates in the
files, and prints which rule dropped every pair.
"""

import json
import math
import pathlib
import statistics
import sys

BAND = 0.03
COLD_FACTOR = 2.0
DRIFT_VETO = 0.85


def read(d, kind):
    p = d / f"{kind}-pairs.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    # The 21:06 and contended captures predate the first/second schema.
    kept, skipped = [], 0
    for r in rows:
        if r.get("first") and r.get("second"):
            kept.append(r)
        else:
            skipped += 1
    if skipped:
        print(f"  note: {p.name} - {skipped} row(s) in the older arm-named schema "
              "skipped (null-spread-witness.py reads them)")
    return kept


def median(values):
    return statistics.median(values) if values else None


def drift_reference(pool, pair_key):
    """Median of every arm rate that finished before this pair's first arm,
    excluding this pair's own arms - what `drive.py` means by "the running
    median of every arm rate seen so far in the process"."""
    start = min(at for k, at, _ in pool if k == pair_key)
    earlier = [r for k, at, r in pool if k != pair_key and at < start]
    return median(earlier)


def analyse(d):
    rows_by_kind = {kind: read(d, kind) for kind in ("real", "control")}
    pool = [(f"{kind}/{r['pair']}", arm["at"], arm["decode_tok_s"])
            for kind, rows in rows_by_kind.items() for r in rows
            for arm in (r["first"], r["second"])]
    out = {}
    for kind, rows in rows_by_kind.items():
        censuses = [a["census_secs"] for r in rows for a in (r["first"], r["second"])]
        med = median(censuses)
        pairs = []
        for r in rows:
            first, second = r["first"], r["second"]
            challenge = "unfold" if kind == "real" else "null"
            c = first if first["arm"] == challenge else second
            b = second if c is first else first
            slow = min(first["decode_tok_s"], second["decode_tok_s"])
            ref = drift_reference(pool, f"{kind}/{r['pair']}")
            pairs.append({
                "pair": r["pair"], "order": r["order"],
                "ratio": c["decode_tok_s"] / b["decode_tok_s"],
                "cold": bool(med and max(c["census_secs"], b["census_secs"])
                             > COLD_FACTOR * med),
                "load": bool(r.get("load_vetoed")),
                "drift": bool(r.get("drift_vetoed")) or bool(ref and slow < DRIFT_VETO * ref),
                "drift_ref": ref, "slow": slow,
                "flag_only": bool(r.get("drift_vetoed")),
                "differ": bool(r.get("tokens_differ")),
                "outside": (c["outside_gpu_pct"], b["outside_gpu_pct"]),
                "encode_ms": (c["encode_ms"], b["encode_ms"]),
                "wait_ms": (c["wait_ms"], b["wait_ms"]),
                "dispatches": (c["dispatches"], b["dispatches"]),
                "census_secs": (c["census_secs"], b["census_secs"]),
            })
        out[kind] = pairs
    return out


def dropped(p):
    return p["cold"] or p["load"] or p["drift"] or p["differ"]


def summarise(pairs, label, key="ratio"):
    kept = [p for p in pairs if not dropped(p)]
    vals = [p[key] for p in kept]
    if not vals:
        print(f"{label}: no admissible pairs ({len(pairs)} captured, "
              f"{len(pairs) - len(kept)} dropped)")
        return None
    med = statistics.median(vals)
    sem = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else float("nan")
    below = sum(1 for x in vals if x < 1)
    print(f"{label}: {len(vals)} pairs (dropped {len(pairs) - len(kept)}), median {med:.6f} "
          f"({(med - 1) * 100:+.2f}%), SEM {sem * 100:.2f} pp, min {min(vals):.4f} "
          f"max {max(vals):.4f}, {below}/{len(vals)} below 1")
    return {"n": len(vals), "median": med, "sem": sem,
            "spread": max(vals) / min(vals) if min(vals) > 0 else float("inf")}


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.cwd())
    pairs_by_kind = analyse(d)
    real, ctrl = pairs_by_kind["real"], pairs_by_kind["control"]
    print(f"# analysis of {d.name}  (band {BAND:.0%}, criterion 5)\n")
    print("## per pair  (cold = census over 2x the series median, LOAD/DRIFT = the")
    print("## vetoes of the 21:15Z amendment; DRIFT is recomputed here)\n")
    for kind, pairs in (("real", real), ("control", ctrl)):
        for p in pairs:
            rules = ("cold" if p["cold"] else "", "load" if p["load"] else "",
                     "drift" if p["drift"] else "",
                     "tokens" if p["differ"] else "")
            print(f"  {kind:8} pair {p['pair']:>2} {p['order']}  ratio {p['ratio']:.4f}"
                  f"  outside-GPU % {p['outside']}  encode ms {p['encode_ms']}"
                  f"  wait ms {p['wait_ms']}  dispatches {p['dispatches']}"
                  f"  census {p['census_secs']}"
                  f"{'  DROPPED: ' + ','.join(r for r in rules if r) if any(rules) else ''}")
    missed = [p for kind in pairs_by_kind.values() for p in kind
              if p["drift"] and not p["flag_only"]]
    print(f"\n  drift rule fired on {len(missed)} pair(s) whose row carries no "
          "drift_vetoed flag:")
    for p in missed:
        print(f"    pair {p['pair']} {p['order']}: slowest arm {p['slow']:.2f} under "
              f"{DRIFT_VETO:.0%} of running median {p['drift_ref']:.2f}")
    differing = [p for kind in pairs_by_kind.values() for p in kind if p["differ"]]
    s_ctrl = summarise(ctrl, "  control (null vs plain, identical code)")
    s_real = summarise(real, "  real    (unfold vs plain, decode tok/s)")
    print("\n## criterion 4 secondary (GPU-idle share, percentage points)")
    for kind, pairs in (("real", real), ("control", ctrl)):
        diffs = [c - b for c, b in (p["outside"] for p in pairs if not dropped(p))]
        if diffs:
            print(f"  {kind}: median {statistics.median(diffs):+.1f} pp, "
                  f"range {min(diffs):+.1f}...{max(diffs):+.1f}")
    if not s_real or not s_ctrl:
        print("\n  criterion 2: the control series admits no pair, so no band exists "
              "to compare against and this run decides nothing.")
        return 1
    band = max(abs(s_ctrl["median"] - 1), (s_ctrl["spread"] - 1) / 2) * 100
    effect = (s_real["median"] - 1) * 100
    print("\n## decision")
    print(f"  control band today: {band:.2f}% (median off-centre "
          f"{abs(s_ctrl['median'] - 1) * 100:.2f}%, half-spread "
          f"{((s_ctrl['spread'] - 1) / 2) * 100:.2f}%)")
    print(f"  measured effect: {effect:+.2f}% of decode tok/s over {s_real['n']} pairs")
    if differing:
        print(f"  NOTE: {len(differing)} pairs had differing greedy continuations; a "
              "schedule change that changes tokens is not a clean ablation.")
    if band >= 100 * BAND:
        print("  criterion 2 fails: the control band is at or above the decision band, so")
        print("  this run reports only the interval it excludes and the knob stays open.")
    if effect > 100 * BAND:
        print("  criterion 5, merge branch: stagged is above the band; flip the default and")
        print("  keep this directory as the regression reference.")
    elif abs(effect) <= band:
        print("  criterion 5, negative branch: inside the measured band; SHARED_STAGGER is a")
        print("  priced negative and stays OFF.")
    else:
        print("  criterion 5, interval branch: between the band and the decision threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
