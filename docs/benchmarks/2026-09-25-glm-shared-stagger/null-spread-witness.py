#!/usr/bin/env python3
"""The identical-code spread of the GLM stagger instrument, and what its own
drop rules admit.

Reads control series - pairs whose two arms run IDENTICAL code (`null`,
ALLPAKA_SHARED_STAGGER=0, against `plain`, unset) - and prints, per pair, the
wall ratio beside three witnesses that live inside the arms: time-to-census, GPU
encode ms, GPU wait ms. No GPU and no build; it only reads the JSONL that
drive.py writes.

    python3 null-spread-witness.py [file-or-dir ...]

With no arguments it reads the three captures this design has produced. A
directory is expanded to `control-pairs.jsonl` and `contended-control-pairs.jsonl`.
analyze.py does the same job for one run directory including the real series;
this file exists to answer the narrower question - what does this instrument do
to a pair where the answer is known to be zero - across every capture at once.

Three caveats the census has to carry:

* The `contended-*` and 21:06 rows are one capture stored twice (same arm
  timestamps); duplicates are dropped, or the median moves.
* Those rows predate drive.py's veto fields, so a missing flag is not a pass.
  Rows are labelled by era, using `load_before` as the marker.
* drive.py's drift veto never fired in any capture: its reference pool was
  initialized and never appended to (see the 22:45Z amendment in
  preregistration.txt). The drift column here recomputes that rule from the
  rates and timestamps in the file, per file rather than per process.
"""

import json
import pathlib
import statistics
import sys

DEFAULTS = [
    "/tmp/allpaka-head/.airbug-bench/glm-stagger-run3",
    "/tmp/allpaka-head/.airbug-bench/glm-stagger-20260925-2106",
    str(pathlib.Path(__file__).parent / "contended-control-pairs.jsonl"),
]
NAMES = ("control-pairs.jsonl", "contended-control-pairs.jsonl")
BAND = 0.03
COLD_FACTOR = 2.0
DRIFT_VETO = 0.85


def ratio(a, b):
    return max(a, b) / min(a, b) if a and b else None


def rows_from(path):
    """Two row schemas exist: drive.py's current one stores the arms under
    first/second with veto flags, the earlier captures store them under the arm
    names. Both are read so the census is every identical-code pair this design
    has produced. Rows whose arms are not the null/plain pair - a real stagger
    pair, say - are refused rather than silently re-labelled."""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ctrl, base = row.get("null"), row.get("plain")
        if not ctrl or not base:
            first, second = row.get("first"), row.get("second")
            if not first or not second:
                continue
            ctrl, base = ((first, second) if first["arm"] == "null"
                          else (second, first))
        if {ctrl["arm"], base["arm"]} != {"null", "plain"}:
            print(f"  skipped pair {row.get('pair')}: arms "
                  f"{ctrl['arm']}/{base['arm']} are not null/plain")
            continue
        yield row, ctrl, base


def sources(args):
    out = []
    for a in args or DEFAULTS:
        p = pathlib.Path(a)
        if p.is_dir():
            out += [p / n for n in NAMES if (p / n).exists()]
        elif p.exists():
            out.append(p)
        else:
            print(f"absent: {p}")
    return out


def main(args):
    kept, seen = [], set()
    for f in sources(args):
        pairs = list(rows_from(f))
        censuses = [x["census_secs"] for _, c, b in pairs for x in (c, b)]
        med_cens = statistics.median(censuses) if censuses else None
        rates = sorted((x["at"], x["decode_tok_s"]) for _, c, b in pairs
                       for x in (c, b))
        print(f"\n== {f}")
        print("pair ord   maxload  era     wall   cens-ratio  encode  wait   "
              "disp= ident=  cold drift")
        for row, ctrl, base in pairs:
            key = tuple(sorted(x["at"] for x in (ctrl, base)))
            if key in seen:
                print(f"  skipped pair {row['pair']}: duplicate of an earlier row "
                      f"(arms ran at {key[0]} / {key[1]})")
                continue
            seen.add(key)
            loads = [x[k] for x in (ctrl, base)
                     for k in ("load_before", "load") if x.get(k) is not None]
            ident = (ctrl.get("git_commit") == base.get("git_commit")
                     and ctrl.get("fingerprint") == base.get("fingerprint"))
            pre = all(x.get("load_before") is None for x in (ctrl, base))
            wall = ratio(ctrl["decode_tok_s"], base["decode_tok_s"])
            cens = ratio(ctrl["census_secs"] or 0, base["census_secs"] or 0)
            enc = ratio(ctrl["encode_ms"], base["encode_ms"])
            wait = ratio(ctrl["wait_ms"], base["wait_ms"])
            cold = med_cens and max(ctrl["census_secs"], base["census_secs"]) \
                > COLD_FACTOR * med_cens
            # the pair's own arms are excluded, as the rule is worded
            before = [r for at, r in rates if at < key[0]]
            ref = statistics.median(before) if before else None
            slow = min(ctrl["decode_tok_s"], base["decode_tok_s"])
            drift = bool(ref and slow < DRIFT_VETO * ref) \
                or bool(row.get("drift_vetoed"))
            print(f"{row['pair']:<4} {row['order']:3s} {max(loads):9.2f} "
                  f"{'pre-veto' if pre else 'vetoed':8}  {wall - 1:6.2%} "
                  f"{cens:9.2f}x {enc - 1:7.2%} {wait - 1:6.2%}  "
                  f"{'Y' if ctrl['dispatches'] == base['dispatches'] else 'N'}"
                  f"      {'Y' if ident else 'N'}"
                  f"   {'Y' if cold else '-'}    {'Y' if drift else '-'}")
            kept.append(dict(pre=pre, wall=wall, cens=cens, enc=enc, ident=ident,
                             drop=bool(cold or drift or row.get("load_vetoed")),
                             flags=bool(row.get("load_vetoed")
                                        or row.get("drift_vetoed")),
                             same=ctrl["dispatches"] == base["dispatches"]))
    if not kept:
        return 1
    same_code = [k for k in kept if k["ident"]]
    print("\n== summary (both arms of every row are the same binary: "
          f"{sum(k['ident'] for k in kept)}/{len(kept)} rows share commit and fingerprint)")
    print(f"  dispatch counts equal in {sum(k['same'] for k in same_code)} of "
          f"{len(same_code)} identical-code pairs,")
    print("  so every wall spread below is the instrument's, not the knob's.")
    for label in ("wall", "cens", "enc"):
        vals = [k[label] for k in same_code if k[label]]
        print(f"  {label:5s} spread: median {statistics.median(vals) - 1:6.2%}   "
              f"max {max(vals) - 1:6.2%}   n={len(vals)}")
    veto_era = [k for k in same_code if not k["pre"]]
    pre_era = [k for k in same_code if k["pre"]]
    print(f"  veto-era rows (run 3, all three rules exist to fire): {len(veto_era)}, of which")
    print(f"    the capture-time flags admit {sum(not k['flags'] for k in veto_era)}, "
          f"cold + load + drift admit {sum(not k['drop'] for k in veto_era)}.")
    print(f"  pre-veto rows (21:06, no load flag existed): {len(pre_era)}, of which")
    print(f"    cold + drift alone admit {sum(not k['drop'] for k in pre_era)} - at wall "
          + ", ".join(f"{k['wall'] - 1:.2%}" for k in pre_era) + ",")
    print("    which is precisely the series that made the load veto necessary (21:15Z "
          "amendment).")
    survivors = [k for k in veto_era if not k["flags"]]
    if survivors:
        print("  the veto-era pair the driver kept, which no rule at capture could see:")
        for k in survivors:
            print(f"    wall {k['wall'] - 1:.2%} on identical code - above the "
                  f"{BAND:.0%} decision band - time-to-census ratio {k['cens']:.2f}x,")
            print("    load under the veto, and a common-mode slowdown the drift rule "
                  "cannot see either.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
