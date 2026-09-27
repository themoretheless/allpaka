#!/usr/bin/env python3
"""Paired re-measure of the in-tree `packed_char4` qs loads (era-audit class B, row B4).

Implements ../2026-09-26-packed-char4/preregistration.txt. B4 is the one class-B row
whose artifact is *in the default build* and is not env-switchable, so its arms are two
binaries of the same source that differ only in those two loads:

    packed  - the shipped kernel (crates/allpaka-backend/src/gpu/metal.rs:566, :677)
    scalar  - a revert copy at /private/tmp/allpaka-b4, byte-addressed uchar loads

Everything else - the instrument, the meter, the gates and the decision rule - is
imported from drive-nr0.py rather than copied, so a verdict here cannot disagree with
the NR0 series by method. One process per arm per round, arms alternating in order.

    python3 drive-packed.py --out DIR --rounds 6
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
NR0_DRIVER = HERE.parent / "2026-09-25-nr0-indexed-remeasure" / "drive-nr0.py"


def load_nr0():
    spec = importlib.util.spec_from_file_location("drive_nr0", NR0_DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # A fourth series on the same instrument: the arm is a binary, not an env knob.
    mod.ARMS["Q8P"] = ["packed", "scalar"]
    mod.DEFAULT["Q8P"] = "packed"
    mod.ENV_OF["Q8P"] = "ALLPAKA_Q8_NR0"   # pinned to the default; see run()
    mod.SHAPES["Q8P"] = ["q8_0", "q8_0 glm down"]
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dn, binary, out_dir, arm, tag):
    env = {**os.environ, "ALLPAKA_Q8_NR0": "2",
           "ALLPAKA_BENCH_FEEDS": dn.FEEDS, "ALLPAKA_BENCH_SWFUSES": dn.SWFUSES,
           "ALLPAKA_BENCH_XPER_SLOT": "0", "ALLPAKA_BENCH_SCATTERS": "0"}
    log = out_dir / f"Q8P-{arm}-{tag}.log.txt"
    lb = dn.load1()
    t0 = time.monotonic()
    proc = subprocess.run([str(binary), dn.TEST, "--ignored", "--exact", "--nocapture"],
                          cwd=dn.REPO, env=env, capture_output=True, text=True, timeout=900)
    la = dn.load1()
    text = proc.stdout + proc.stderr
    log.write_text(text)
    rows = dn.parse_rows(text)
    return {"series": "Q8P", "arm": arm, "tag": tag, "binary": str(binary),
            "binary_sha256": sha(binary), "started": dn.utc(),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "load_before": lb, "load_after": la, "returncode": proc.returncode,
            "rows": {lab: {f"{a}|{f}": v for (a, f), v in cells.items()}
                     for lab, cells in rows.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--packed", type=pathlib.Path, required=True)
    ap.add_argument("--scalar", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    dn = load_nr0()
    for p in (args.packed, args.scalar):
        if not p.exists():
            print(f"missing binary: {p}")
            return 2
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    store = out / "records.jsonl"
    prov_path = out / "provenance.json"
    head = subprocess.run(["git", "-C", str(dn.REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = len(subprocess.run(["git", "-C", str(dn.REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines())
    prov = {"preregistration": str(HERE / "preregistration.txt"),
            "method_from": str(NR0_DRIVER), "head": head, "tree_dirty_files": dirty,
            "packed_sha256": sha(args.packed), "scalar_sha256": sha(args.scalar),
            "shapes": dn.SHAPES["Q8P"], "meter": dn.METER,
            "gates": {"load_veto": dn.LOAD_VETO, "launch_load": dn.LAUNCH_LOAD,
                      "meter_band": dn.METER_BAND, "spread_gate": dn.SPREAD_GATE}}
    records = []
    if store.exists():
        records = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
        if records and prov_path.exists():
            prior = json.loads(prov_path.read_text())
            if prior["packed_sha256"] != prov["packed_sha256"] or \
               prior["scalar_sha256"] != prov["scalar_sha256"]:
                print("a binary changed since the pinned series - refusing to mix builds")
                return 3
            prov = prior  # keep the original provenance across a continuation
    else:
        prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    meter_ref = json.loads((out / "meter-ref.json").read_text())["ref"] \
        if (out / "meter-ref.json").exists() else None

    arms = {"packed": args.packed, "scalar": args.scalar}
    for rnd in range(1 + len({r["tag"] for r in records}), args.rounds + 1):
        order = ["packed", "scalar"] if rnd % 2 else ["scalar", "packed"]
        for arm in order:
            waited = dn.wait_quiet()
            rec = run(dn, arms[arm], out, arm, f"r{rnd}")
            with store.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            ok, why = dn.gate_process(rec, "Q8P", meter_ref)
            key = "contig x/shared plain|8"
            r = rec["rows"].get("q8_0 glm down", {}).get(key, [0])[0]
            m = rec["rows"].get(dn.METER, {}).get(key, [0])[0]
            note = "ok" if ok else f"VETO {why}"
            if ok and meter_ref is None and m:
                meter_ref = m
                (out / "meter-ref.json").write_text(json.dumps({"ref": m}))
            print(f"Q8P {arm} r{rnd} {note}  load {rec['load_before']}/{rec['load_after']}"
                  f" {rec['elapsed_s']}s  q8_0/meter={r / m if m else 0:.4f}"
                  + (f"  waited {waited:.0f}s" if waited else ""), flush=True)

    kept = [r for r in records if r.get("returncode") == 0]
    kept = [r for r in kept if dn.gate_process(r, "Q8P", meter_ref)[0]]
    lines, flip, winner = dn.decide(kept, "Q8P", args.rounds)
    print(f"\n{len(kept)}/{len(records)} processes admitted")
    for arm, won, cells in lines:
        print(f"  {arm}: {'SURVIVES' if not won else 'BEATEN'}  " + "  ".join(cells))
    print(f"meter reference: {meter_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
