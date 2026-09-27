#!/usr/bin/env python3
"""Print a drive-nr0.py series at full float: per-cell medians, signs, gate ledger.

The driver's own verdict line only covers the shipped cell; writing an artifact
needs the other three cells, the absolute rates and the veto breakdown, which
until now were recomputed by hand from records.jsonl each time.

    python3 analyze-nr0.py DIR --series Q50
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load_dn(series):
    """Gates, shapes and decide() come from the driver that ran the series, so an
    analysis cannot diverge from the campaign by method. Q8P's arms are two
    binaries, and drive-packed.py is what registers that series."""
    if series == "Q8P":
        path = HERE.parent / "2026-09-26-packed-char4" / "drive-packed.py"
        spec = importlib.util.spec_from_file_location("drive_packed", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_nr0()
    spec = importlib.util.spec_from_file_location("drive_nr0", HERE / "drive-nr0.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CELLS = [(1, False), (8, False), (1, True), (8, True)]


def cell_key(feed, fuse):
    return f"contig x/shared {'fused' if fuse else 'plain'}|{feed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--series", required=True)
    ap.add_argument("--floor", type=float, default=None,
                    help="extra symmetric veto: drop processes whose shipped-cell best on any "
                         "measured row falls under this fraction of the reference. The "
                         "instrument's gates catch a disturbed meter but not a process whose "
                         "rows sat uniformly slow (a swapped-out working set keeps best/med "
                         "tight), so the band is read off the measured rows themselves.")
    ap.add_argument("--clean", default="",
                    help="comma list of round tags whose admitted processes set the --floor "
                         "reference; defaults to every admitted process")
    args = ap.parse_args()
    dn = load_dn(args.series)
    # Env-knob series print their arms as NR0=<n>; a binary-vs-binary series does not.
    pk = "" if args.series == "Q8P" else "NR0="
    d = args.dir.resolve()
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip()]
    prov = json.loads((d / "provenance.json").read_text())
    srecs = [r for r in recs if r["series"] == args.series]
    if any("admitted" not in r for r in srecs):
        # drive-packed.py predates the admitted stamp. The gate is a deterministic
        # function of the record plus the series meter reference, so re-running it
        # here restores the flags exactly rather than approximating them.
        ref = json.loads((d / "meter-ref.json").read_text())["ref"] \
            if (d / "meter-ref.json").exists() else None
        for r in srecs:
            ok, why = dn.gate_process(r, args.series, ref)
            r["admitted"], r["veto"] = ok, why
    kept = [r for r in srecs if r.get("admitted")]
    key = cell_key(*dn.PRIMARY)
    bins = prov.get("binary_sha256") or prov.get("packed_sha256", "")
    print(f"{args.series}: {len(kept)}/{len(srecs)} admitted  binary {bins[:16]}"
          f"  head {prov['head'][:8]}  dirty {prov['tree_dirty_files']}")
    if not kept:
        print("no admitted processes")
        return 1
    extra_veto = []
    if args.floor:
        clean = [r for r in kept if not args.clean or r["tag"] in args.clean.split(",")]
        key = cell_key(*dn.PRIMARY)
        refs = {}
        for shape in dn.SHAPES[args.series]:
            vals = sorted(r["rows"][shape][key][0] for r in clean if shape in r["rows"])
            refs[shape] = dn.median(vals)
        floor = []
        for r in kept:
            bad = [s for s in dn.SHAPES[args.series]
                   if s in r["rows"] and r["rows"][s][key][0] < args.floor * refs[s]]
            (floor if not bad else extra_veto).append((r, bad))
        print("  floor refs (" + ", ".join(f"{s} {v:.1f}" for s, v in refs.items()) + ")")
        print(f"  --floor {args.floor}: dropped {len(extra_veto)} of {len(kept)} admitted"
              + (" - " + ", ".join(f"{r['arm']}/{r['tag']}:{','.join(b)}" for r, b in extra_veto)
                 if extra_veto else ""))
        kept = [r for r, _ in floor]
    vetoes = {}
    for r in srecs:
        if not r.get("admitted"):
            vetoes[r["veto"].split(" (")[0]] = vetoes.get(r["veto"].split(" (")[0], 0) + 1
    print("  vetoes: " + (", ".join(f"{k} x{v}" for k, v in vetoes.items()) or "none"))
    loads = [max(r["load_before"], r["load_after"]) for r in srecs]
    meters = [r["rows"][dn.METER][key][0] for r in kept]
    spread = max(r["rows"][s][key][0] / r["rows"][s][key][1]
                 for r in kept for s in dn.SHAPES[args.series])
    print(f"  load max {max(loads):.2f}   meter best {min(meters):.1f}..{max(meters):.1f}"
          f"   best/med max {spread:.4f}")

    shapes, arms, default = dn.SHAPES[args.series], dn.ARMS[args.series], dn.DEFAULT[args.series]
    print(f"\n  absolute GB/s (best of window, median over admitted processes)")
    for feed, fuse in CELLS:
        k = cell_key(feed, fuse)
        for shape in shapes + [dn.METER]:
            cells = []
            for arm in arms:
                vals = [r["rows"][shape][k][0] for r in kept
                        if r["arm"] == arm and shape in r["rows"] and k in r["rows"][shape]]
                cells.append(f"{pk}{arm} {dn.median(vals):7.1f} (n={len(vals)})" if vals
                             else f"{pk}{arm} -")
            print(f"    feed={feed} {'fused' if fuse else 'plain':<5} {shape:<16} " + "  ".join(cells))
    for feed, fuse in CELLS:
        print(f"\n  ratio to meter, feed={feed} {'fused' if fuse else 'plain'}"
              + ("   <- shipped" if (feed, fuse) == dn.PRIMARY else ""))
        for shape in shapes:
            cells = []
            for arm in arms:
                vals = [v for r in (x for x in kept if x["arm"] == arm)
                        if (v := dn.ratio(r, args.series, shape, feed, fuse))]
                cells.append(f"{pk}{arm} {dn.median(vals):.4f} (n={len(vals)})" if vals
                             else f"{pk}{arm} -")
            print(f"    {shape:<16} " + "  ".join(cells))
    feed, fuse = dn.PRIMARY
    print(f"\n  per-round ratios, shipped cell (plain feed={feed})")
    tags = sorted({r["tag"] for r in kept}, key=lambda t: int(t[1:]))
    for shape in shapes:
        print(f"    {shape}")
        for t in tags:
            row = {r["arm"]: dn.ratio(r, args.series, shape, feed, fuse)
                   for r in kept if r["tag"] == t}
            print(f"      {t:<4} " + "  ".join(
                f"{'-' if row.get(a) is None else f'{a}={row[a]:.4f}'}" for a in arms))
    lines, flip, winner = dn.decide(kept, args.series, len(tags))
    print(f"\n  decide vs default {pk}{default}: flip={flip} winner={winner}")
    for arm, won, cells in lines:
        print(f"    {pk}{arm} {'wins' if won else 'holds'}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
