#!/usr/bin/env python3
"""Fit the prefill scaling ladder and apply the Amendment 2 decision rule.

Reads a ladder-1.txt (rows: `rep N pp P  allpaka A (exec E ms commits C)
llama L  ratio R  load S`), converts each arm to cost per token, and least-
squares fits cost/token = A + B*(L/2) per arm. SEs come from resampling the
repeats with replacement, so machine drift across repeats shows up as width
instead of being averaged into the slope.

Verdicts are exactly the ones pre-registered in ../preregistration.txt
(Amendment 2); nothing here re-tunes the thresholds.
"""
import random
import re
import statistics
import sys

random.seed(0)

ROW = re.compile(
    r"rep\s+(\d+)\s+pp\s+(\d+)\s+allpaka\s+([\d.]+)\s+\(exec\s+(\S+)ms commits (\S+)\)"
    r"\s+llama\s+([\d.]+)\s+ratio\s+([\d.]+)\s+load\s+(\S+)"
)
BOOT = 2000


def load(path):
    rows = {}
    for line in open(path):
        m = ROW.search(line)
        if not m:
            continue
        rep, pp = int(m.group(1)), int(m.group(2))
        rows.setdefault((rep, pp), {})["ap"] = float(m.group(3))
        rows[(rep, pp)]["ll"] = float(m.group(6))
        rows[(rep, pp)]["exec"] = m.group(4)
        rows[(rep, pp)]["ratio"] = float(m.group(7))
    return rows


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main(path, min_reps=3):
    rows = load(path)
    if not rows:
        sys.exit(f"no rows parsed from {path}")
    pps = sorted({pp for (_, pp) in rows})
    reps = sorted({rep for (rep, _) in rows})
    full = [pp for pp in pps if sum(1 for r in reps if (r, pp) in rows) >= min_reps]
    print(f"ladder {pps}  reps {len(reps)}  usable pp points (>= {min_reps} reps): {full}")
    if len(full) < 4:
        print("VERDICT: undetermined - fewer than 4 usable PP points. Do not act.")
        return

    print(f"{'pp':>6} {'ap us/tok':>10} {'ll us/tok':>10} {'ratio':>7} {'ap med exec':>12}")
    per = {}
    for pp in full:
        vals = {arm: [1e6 / rows[(r, pp)][arm] for r in reps if (r, pp) in rows] for arm in ("ap", "ll")}
        ex = [rows[(r, pp)]["exec"] for r in reps if (r, pp) in rows and rows[(r, pp)]["exec"].isdigit()]
        per[pp] = (statistics.median(vals["ap"]), statistics.median(vals["ll"]))
        print(f"{pp:>6} {per[pp][0]:>10.1f} {per[pp][1]:>10.1f} "
              f"{per[pp][1] / per[pp][0]:>7.4f} {statistics.median(ex) if ex else '-':>12}")
    xs = [pp / 2 for pp in full]

    out = {}
    for arm, key in (("allpaka", "ap"), ("llama", "ll")):
        ys = [per[pp][0 if key == "ap" else 1] for pp in full]
        a, b = fit(xs, ys)
        bs = []
        for _ in range(BOOT):
            by_rep = {}
            for r in reps:
                for pp in full:
                    if (r, pp) in rows:
                        by_rep.setdefault(r, []).append(1e6 / rows[(r, pp)][key])
            keys = [k for k in by_rep if len(by_rep[k]) == len(full)]
            if not keys:
                continue
            pick = random.choices(keys, k=len(keys))
            sample = [(xs[j], statistics.median([by_rep[k][j] for k in pick])) for j in range(len(full))]
            f = fit([s[0] for s in sample], [s[1] for s in sample])
            if f[0] is not None:
                bs.append(f)
        if not bs:
            print(f"{arm}: no bootstrap samples - undetermined")
            return
        bs.sort(key=lambda f: f[0]); lo_a, hi_a = bs[int(.025 * len(bs))][0], bs[int(.975 * len(bs))][0]
        bs.sort(key=lambda f: f[1]); lo_b, hi_b = bs[int(.025 * len(bs))][1], bs[int(.975 * len(bs))][1]
        out[arm] = (a, lo_a, hi_a, b, lo_b, hi_b)
        print(f"{arm:>8}: A={a:7.1f} us/tok [{lo_a:.1f}, {hi_a:.1f}]   "
              f"B={b:7.4f} us/tok/key [{lo_b:.4f}, {hi_b:.4f}]")

    A1, AloA, AhiA, B1, BloB, BhiB = out["allpaka"]
    A2, Alo2, Ahi2, B2, Blo2, Bhi2 = out["llama"]
    ra, rb = A1 / A2, B1 / B2
    print(f"\nratios: A(allpaka/llama)={ra:.3f}  B(allpaka/llama)={rb:.3f}")
    print(f"bootstrap separation of B: {'yes' if (BhiB < Blo2 or Bhi2 < BloB) else 'no'}")
    print(f"bootstrap separation of A: {'yes' if (AhiA < Alo2 or Ahi2 < AloA) else 'no'}")

    if rb > 1.2 and (BhiB < Blo2 or Bhi2 < BloB):
        verdict = "QUADRATIC - target is the attend_mm kernel (8 query rows per threadgroup)"
    elif ra > 1.1 and 1 / 1.2 < rb < 1.2:
        verdict = "LINEAR - target is the tile matmul path; long context is not a separate problem"
    else:
        verdict = "UNDETERMINED - do not optimise either term"
    print(f"VERDICT: {verdict}")

    # Falsification of the two-term model itself: fit on the ends, test the middle.
    if len(full) >= 3 and full[1] != full[-1]:
        ends = [full[0], full[-1]]
        ex, eap, ell = [p / 2 for p in ends], [per[p][0] for p in ends], [per[p][1] for p in ends]
        aap, bap = fit(ex, eap); all_, bll = fit(ex, ell)
        for pp in full[1:-1]:
            if all_ is not None and aap is not None:
                pred = (all_ + bll * pp / 2) / (aap + bap * pp / 2)
                meas = per[pp][1] / per[pp][0]
                print(f"  held-out pp{pp}: model predicts ratio {pred:.3f}, measured {meas:.3f}, "
                      f"error {abs(pred - meas) / meas * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
