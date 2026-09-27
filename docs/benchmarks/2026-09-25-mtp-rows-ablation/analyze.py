#!/usr/bin/env python3
"""Admission and statistics for the ROWS ablation pairs (preregistration.txt).

Criteria 1-8 as first written, plus the 2026-09-25T01:41Z amendment (4'-8'):
the plain-control band no longer voids a pair, the per-round verify ms is the
primary statistic, and the e2E rate ratio has to agree with it in sign before
either decides anything.

    python3 analyze.py [.airbug-bench/mtp-rows-<ts>]
"""

import json
import pathlib
import statistics
import sys

M_OF_ROUND = 5  # ALLPAKA_DRAFT_K=4
PLANE_AGREEMENT = 0.02  # criterion 4, now a diagnostic
THRESHOLD = 0.03  # criterion 6 / 6'
STOP_AFTER, STOP_MIN = 14, 8  # criterion 8'


def hard_reasons(row):
    """Criterion 2 and 3: same stream, same acceptance, and an arm census apart."""
    bad = []
    a, b = row["rows"], row["norows"]
    for arm in (a, b):
        if not arm["stream_pass"]:
            bad.append(f"{arm['arm']} stream diverged from plain greedy")
    if (a["accepted"], a["drafted"]) != (b["accepted"], b["drafted"]):
        bad.append(
            f"acceptance differs: {a['accepted']}/{a['drafted']} vs {b['accepted']}/{b['drafted']}"
        )
    if not (a["rows_pipeline_engaged"] > 0 and b["rows_pipeline_engaged"] == 0):
        bad.append(
            f"ROWS engaged {a['rows_pipeline_engaged']}x in rows and "
            f"{b['rows_pipeline_engaged']}x in norows, expected >0 and 0"
        )
    gap = (b["verify_dispatches"] or 0) - (a["verify_dispatches"] or 0)
    if gap <= 0 or gap % (3 * (M_OF_ROUND - 1)):
        bad.append(
            f"verify dispatch gap {gap} is not a positive multiple of {3 * (M_OF_ROUND - 1)}"
        )
    return bad


def soft_note(row):
    """Criterion 4': the plain control is a 0.34 s pass, so this band says how
    far the machine moved between the two processes, not that the pair is void."""
    a, b = row["rows"], row["norows"]
    move = b["plain_tok_s"] / a["plain_tok_s"] - 1.0
    return move, ("" if abs(move) <= PLANE_AGREEMENT else f"plain moved {move:+.1%}")


def stats(values, label, above1):
    percents = [(v - 1.0) * 100.0 for v in values]
    med = statistics.median(values)
    sem = statistics.stdev(percents) / len(values) ** 0.5
    print(
        f"{label}: median {med:.4f}, mean {statistics.mean(percents):+.2f}% SEM {sem:.2f} "
        f"n {len(values)}, {sum(1 for p in percents if p > 0)}/{len(values)} above 1 ({above1})"
    )
    return med


def report(pairs):
    good, dropped = [], []
    for row in pairs:
        bad = hard_reasons(row)
        (dropped if bad else good).append((row, bad))
    print(f"{len(good)} of {len(pairs)} pairs admitted by criteria 2 and 3")
    for row, bad in dropped:
        print(f"  pair {row['pair']:>2} ({row['order']}): " + "; ".join(bad))

    clean = [r for r, _ in good if not soft_note(r)[1]]
    print(
        f"criterion 4 (diagnostic, no longer a veto): {len(clean)} of {len(good)} "
        f"pairs have the plain control inside 2%"
    )

    print("\n  pair ord   verify ms rows  norows  ratio   spec tok/s rows  norows  ratio   plain   load")
    for row, _ in good:
        a, b = row["rows"], row["norows"]
        flag = "*" if soft_note(row)[1] else " "
        print(
            f"  {row['pair']:>4} {row['order']:>3}  {a['verify_ms']:>12.3f} {b['verify_ms']:>8.3f}"
            f"  {b['verify_ms'] / a['verify_ms']:>7.4f}   {a['spec_tok_s']:>9.4f} {b['spec_tok_s']:>9.4f}"
            f"  {b['spec_tok_s'] / a['spec_tok_s']:>7.4f}  {flag} {a['plain_tok_s']:.1f}/{b['plain_tok_s']:.1f}"
            f"  {a['load']:.1f}/{b['load']:.1f}"
        )
    print("  (* = the plain control moved more than 2% inside the pair)")

    if len(good) < 2:
        print("\ntoo few admitted pairs for a series statistic")
        return
    vr = [r["norows"]["verify_ms"] / r["rows"]["verify_ms"] for r, _ in good]
    sr = [r["norows"]["spec_tok_s"] / r["rows"]["spec_tok_s"] for r, _ in good]
    sc = [r["norows"]["spec_tok_s"] / r["rows"]["spec_tok_s"] for r in clean]
    print()
    vmed = stats(
        vr,
        "primary  5' verify-stage ms, norows/rows",
        "the fallback costs more ms, ie ROWS wins",
    )
    smed = stats(
        sr,
        "secondary 5 e2E speculative rate, norows/rows (all admitted)",
        "the fallback is faster",
    )
    if len(sc) >= 2:
        stats(
            sc,
            "secondary 5 e2E speculative rate, norows/rows (criterion-4 clean)",
            "the fallback is faster",
        )
    for label, vals in (("verify", vr), ("e2E", sr)):
        for order in ("AB", "BA"):
            sub = [v for (r, _), v in zip(good, vals) if r["order"] == order]
            if sub:
                print(
                    f"  {label} {order}: n {len(sub)}, median {statistics.median(sub):.4f}, "
                    f"{sum(1 for v in sub if v > 1)} of {len(sub)} above 1"
                )

    # Both statistics are norows over rows, but one is milliseconds and the
    # other a rate, so they need opposite transforms to read as "what ROWS is
    # worth": fewer ms for the rows arm means ROWS won, a higher norows rate
    # means the fallback won.
    vgain, sgain = vmed - 1.0, 1.0 / smed - 1.0
    print(
        f"\nROWS worth {vgain:+.2%} by the verify stage, {sgain:+.2%} by the e2E rate "
        f"(positive favours ROWS)"
    )
    if (vgain >= THRESHOLD) != (sgain >= THRESHOLD) and (vgain <= -THRESHOLD) != (
        sgain <= -THRESHOLD
    ):
        print("  6': the two statistics disagree in direction: the lever stays open, publish both")
    elif vgain <= -THRESHOLD and sgain <= -THRESHOLD:
        print(
            "  6': the ROWS mapping is a loss where it exists. q3_k/q2_k do not get a branch;"
        )
        print("     retire it from the kernels that carry it (see the task this opens).")
    elif vgain >= THRESHOLD and sgain >= THRESHOLD:
        print("  6': ROWS wins by both readings, so roadmap step (b) opens for q3_k/q2_k.")
    else:
        print(
            "  6': inside +/-3% on at least one reading: the mapping is e2E-neutral where it"
        )
        print("     exists, so the q3_k/q2_k branch is not worth writing; publish negative.")
    if len(pairs) >= STOP_AFTER and len(good) < STOP_MIN:
        print(f"stop rule 8': only {len(good)} of the first {STOP_AFTER} pairs admitted")


def main():
    where = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    src = where / "pairs.jsonl" if where.is_dir() else where
    pairs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    report(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
