#!/usr/bin/env python3
"""Statistics for the attention-knob run (preregistration.txt).

    python3 analyze.py .airbug-bench/attn-e2e-20260926-0100
    python3 analyze.py .airbug-bench/attn-stage-20260926-0040

Auto-detects the mode from provenance.json. The e2E decision band is 3% and is
set by the control (mv vs mv) series, exactly as in the stagger driver.
"""

import json
import math
import pathlib
import statistics
import sys

BAND = 0.03


def ratios(d, kind):
    p = d / f"{kind}-pairs.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        a, b = r["first"], r["second"]
        if kind == "real":
            ch, ref = (a, b) if a["arm"] != "mv" else (b, a)
        else:
            # Both arms are mv: the ratio is first/second, so the alternating
            # order keeps the control series centred instead of folding it.
            ch, ref = a, b
        out.append({"pair": r["pair"], "order": r["order"],
                    "ratio": ch["decode_tok_s"] / ref["decode_tok_s"],
                    "differ": bool(r.get("tokens_differ")),
                    "vetoed": bool(r.get("load_vetoed") or r.get("drift_vetoed")),
                    "attend": {x["arm"]: x["attend_ms_token"] for x in (a, b)},
                    "tok_s": {x["arm"]: x["decode_tok_s"] for x in (a, b)}})
    return out


def summarise(rows, label):
    rows = [r for r in rows if not r["differ"] and not r["vetoed"]]
    v = [r["ratio"] for r in rows]
    if not v:
        print(f"{label}: no admissible pairs")
        return None
    med = statistics.median(v)
    sem = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    below = sum(1 for x in v if x < 1)
    print(f"{label}: {len(v)} pairs, median {med:.6f} ({(med - 1) * 100:+.2f}%), "
          f"SEM {sem * 100:.2f} pp, min {min(v):.4f} max {max(v):.4f}, "
          f"{below}/{len(v)} below 1")
    return {"n": len(v), "median": med, "sem": sem, "spread": max(v) / min(v)}


def stage(d):
    rows = {}
    for f in sorted(d.glob("*-rec.json")):
        r = json.loads(f.read_text())
        if r.get("split") and r["attend_ms_token"] is not None:
            rows.setdefault(r["arm"], []).append(r["attend_ms_token"])
    if "mv" not in rows:
        print("stage: no mv baseline recorded")
        return
    base = statistics.median(rows["mv"])
    others = [a for a in rows if a != "mv" and rows[a]]
    if others and all(statistics.median(rows[a]) == base for a in others):
        print("  WARNING: every arm recorded the same attend ms/token as mv - identical "
              "numbers across arms that select different kernels means the arms did not "
              "encode differently; the table would be a meter reading, not a ladder.")
    print(f"  attend stage, ms/token (one token sampled per process, reps as listed)")
    for arm in ("mv", "s32", "s16", "s8", "attend4"):
        if arm in rows:
            v = rows[arm]
            med = statistics.median(v)
            print(f"    {arm:9} {' '.join(f'{x:6.3f}' for x in v)}   median {med:6.3f}  "
                  f"vs mv {100 * (med / base - 1):+6.1f}%")


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.cwd())
    prov = json.loads((d / "provenance.json").read_text())
    sha = prov.get("binary_sha256", "?")
    print(f"# analysis of {d.name}  (mode {prov['mode']}, git {prov.get('git_head')}, "
          f"pp/tg {prov['pp_tg']})")
    print(f"#   binary {sha[:16]}  source {prov.get('binary_source', '?')}")
    if prov.get("preregistered_sha") and prov["preregistered_sha"] != sha:
        print(f"#   MISMATCH against the preregistered {prov['preregistered_sha'][:16]}")
    print()
    if prov["mode"] == "stage":
        stage(d)
        return 0
    print("## per pair (ratio is challenge/mv, so <1 means the fallback is faster)")
    for kind in ("real", "control"):
        for r in ratios(d, kind):
            print(f"  {kind:8} pair {r['pair']:>2} {r['order']}  ratio {r['ratio']:.4f}  "
                  f"attend ms/token {r['attend']}")
    s_ctrl = summarise(ratios(d, "control"), "  control (mv vs mv)")
    s_real = summarise(ratios(d, "real"), "  real    (s32 vs mv; <1 = fallback wins)")
    if not s_real or not s_ctrl:
        return 1
    # The RFUSE lesson, applied to a null before it is published: two arms that
    # recorded bit-identical work for the stage the knob changes did not encode
    # different code, so "no difference e2E" would be a meter reading, not a result.
    ident = [r for r in ratios(d, "real")
             if not r["differ"] and not r["vetoed"] and len(r["attend"]) > 1
             and min(r["attend"].values()) > 0
             and abs(max(r["attend"].values()) / min(r["attend"].values()) - 1) < 1e-9]
    adm = [r for r in ratios(d, "real") if not r["differ"] and not r["vetoed"]]
    ident_warn = bool(adm) and len(ident) == len(adm)
    if ident_warn:
        print("  WARNING: every admissible pair recorded identical attend ms/token for both "
              "arms - the two arms encoded the same work, so this null says nothing about "
              "the knob (check the arm strings' readers, not the arithmetic).")
    print("\n## decision")
    if ident_warn:
        print("  REFUSED: the arms recorded identical work, so neither the null nor the "
              "interval below is a verdict on the knob. Publish the diagnosis, not the "
              "number.")
    band = max(abs(s_ctrl["median"] - 1), (s_ctrl["spread"] - 1) / 2) * 100
    effect = abs((s_real["median"] - 1) * 100)
    print(f"  control band today: {band:.2f}%")
    print(f"  measured effect: {effect:.2f}% of decode tok/s over {s_real['n']} pairs")
    if band >= 100 * BAND:
        print("  criterion 2 fails: band >= decision threshold; the run reports the interval "
              "it excludes and the knob stays open.")
    elif effect <= band:
        print("  inside the band: at PP 3072 mv and s32 are indistinguishable e2E.")
    else:
        sign = "s32 (the fallback)" if s_real["median"] < 1 else "mv (the shipped default)"
        print(f"  above the band: {sign} is faster by {effect:.2f}% e2E at PP 3072.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
