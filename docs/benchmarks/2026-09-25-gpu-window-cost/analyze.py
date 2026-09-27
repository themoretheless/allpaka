#!/usr/bin/env python3
"""Statistics for the weight-window A/B; implements preregistration.txt criteria.

    python3 analyze.py .airbug-bench/window-cost-20260925-023600
"""

import json
import math
import pathlib
import statistics
import sys

BAND = 0.03  # criterion 6's decision band, in fraction of tok/s


def pairs(d, kind):
    p = d / f"{kind}-pairs.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load(d, kind, drop_cold):
    """Ratios of the challenge arm over its reference, with the cold veto applied."""
    challenge, reference = ("narrow", "wide") if kind == "real" else ("null", "wide")
    rows = pairs(d, kind)
    censuses = [r[a]["census_secs"] for r in rows for a in (challenge, reference)
                if r[a].get("census_secs")]
    med = statistics.median(censuses) if censuses else None
    out = []
    for r in rows:
        c, b = r[challenge], r[reference]
        cold = bool(drop_cold and med and max(c["census_secs"] or 0, b["census_secs"] or 0) > 2 * med)
        out.append({
            "pair": r["pair"],
            "order": r["order"],
            "ratio": c["decode_tok_s"] / b["decode_tok_s"],
            "cold": cold,
            "same_tokens": c.get("decode_tokens") == b.get("decode_tokens"),
            "windows": (c["windows"], b["windows"]),
            "census_secs": (c["census_secs"], b["census_secs"]),
            "dispatches": (c["dispatches"], b["dispatches"]),
            "wait_ms": (c["wait_ms"], b["wait_ms"]),
            "declines": (c["declines"], b["declines"]),
        })
    return out


def summarise(rows, label):
    kept = [r for r in rows if not r["cold"]]
    ratios = [r["ratio"] for r in kept]
    if not ratios:
        print(f"{label}: no admissible pairs")
        return None
    med = statistics.median(ratios)
    sem = statistics.stdev(ratios) / math.sqrt(len(ratios)) if len(ratios) > 1 else float("nan")
    below = sum(1 for x in ratios if x < 1)
    print(f"{label}: {len(ratios)} pairs (dropped {len(rows) - len(kept)} cold), "
          f"median ratio {med:.6f} ({(med - 1) * 100:+.2f}%), SEM {sem * 100:.2f} pp, "
          f"min {min(ratios):.4f} max {max(ratios):.4f}, {below}/{len(ratios)} below 1")
    return {"n": len(ratios), "median": med, "sem": sem, "below": below,
            "spread": max(ratios) / min(ratios)}


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.cwd())
    print(f"# analysis of {d.name}  (band {BAND:.0%}, criterion 6)\n")
    real_all, ctrl_all = load(d, "real", False), load(d, "control", False)
    real, ctrl = load(d, "real", True), load(d, "control", True)

    print("## per pair (challenge/reference decode tok/s)")
    for kind, rows in (("real", real), ("control", ctrl)):
        for r in rows:
            print(f"  {kind:8} pair {r['pair']:>2} {r['order']}  ratio {r['ratio']:.4f}"
                  f"  windows {r['windows']}  census {r['census_secs']}s"
                  f"  dispatches {r['dispatches']}  same greedy tokens: {r['same_tokens']}"
                  f"{'  COLD' if r['cold'] else ''}")

    token_rows = real_all + ctrl_all
    bad = [r for r in token_rows if not r["same_tokens"]]
    declined = [r for r in token_rows if any(d_ and d_ > 0 for d_ in r["declines"])]
    print(f"\n## criteria")
    print(f"  2 (coverage): {len(declined)} pairs with any CPU decline"
          f"    3 (greedy equality): {len(bad)} of {len(token_rows)} pairs differ")
    s_ctrl = summarise(ctrl, "  control (null vs wide, identical code)")
    s_real = summarise(real, "  real    (narrow 2 GiB vs wide 32 GiB)")
    if not s_real:
        return 1
    effect = (s_real["median"] - 1) * 100
    band = (max(abs(s_ctrl["median"] - 1), (s_ctrl["spread"] - 1) / 2) * 100) if s_ctrl else None
    print(f"\n## decision")
    print(f"  today's null band from {s_ctrl['n'] if s_ctrl else 0} control pairs: "
          f"{band:.2f}% (median off-centre {abs(s_ctrl['median'] - 1) * 100:.2f}%, "
          f"half-spread {((s_ctrl['spread'] - 1) / 2) * 100:.2f}%)")
    print(f"  measured effect: {effect:+.2f}% of decode tok/s")
    if band >= 100 * BAND:
        print("  criterion 3 fails: the null band is at least the decision band, so this run")
        print("  can only report the interval it excludes and the DoD stays open.")
    if abs(effect) < 100 * BAND and abs(effect) <= band:
        print("  criterion 6, first branch: no windowing cost at 1 -> 20 windows, inside a")
        print(f"  band of {band:.2f}% measured on this machine today.")
    elif effect < -100 * BAND:
        print("  criterion 6, second branch: windowing costs more than the band; WINDOW_CAP")
        print("  and CHUNK_OVERLAP become tunables to price, and the 235B gains a non-kernel")
        print("  deficit candidate.")
    else:
        print("  criterion 6, third branch: interval, not a verdict.")


if __name__ == "__main__":
    sys.exit(main())
