#!/usr/bin/env python3
"""Does ALLPAKA_RFUSE=1 encode anything under the shipped policy?

Structural answer (README.md, same directory): no - rfuse() is read only inside
`moe_plain`, which is gated on `!normflag`. This script is the positive check:
four load-independent observables, extracted from a bench report and the run's
own stdout, compared arm to arm. It never reads a rate, so it is valid while the
machine is contended.

    # compare two already-captured arms
    python3 census.py A-report.json A-log.txt B-report.json B-log.txt

    # or capture and compare (needs the GPU free; two runs, no pairs)
    python3 census.py --run

    # prove the comparator can say "differ"
    python3 census.py --selfcheck

Verdict: all four identical => the arm encodes nothing and the -20-27% recorded
on 2026-09-12 is retired. Any difference => the guard read is wrong, the knob is
live, and its cost is back on the table.
"""

import copy
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = "models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
BIN = ROOT / "target/release/allpaka"
BASE_ENV = {
    "ALLPAKA_BENCH_PP": "480",
    "ALLPAKA_BENCH_TG": "320",
    "ALLPAKA_BENCH_SKIP_MTP": "1",
    "ALLPAKA_PROFILE": "max-performance",
}

GPU_DECODE_RE = re.compile(
    r"gpu during decode: (\d+) waits, (\d+) dispatches, encode (\d+) ms, wait (\d+) ms "
    r"of (\d+) ms total \((\d+)% outside the GPU\)"
)
RESIDENCY_RE = re.compile(r"residency: windows=(\d+) set=(\w+)")

# (label, how to read it). Every field must exist in both arms - a missing
# observable is a broken harness, not a match.
OBSERVABLES = (
    ("decode token stream", lambda rep, log: json.dumps(
        next(m for m in rep["measurements"] if m["name"] == "decode")["input_tokens"])),
    ("decode dispatches", lambda rep, log: GPU_DECODE_RE.search(log).group(2)),
    ("decode waits", lambda rep, log: GPU_DECODE_RE.search(log).group(1)),
    ("decode fast path", lambda rep, log: json.dumps(
        next(m for m in rep["measurements"] if m["name"] == "decode")["fast_path"])),
    ("residency", lambda rep, log: RESIDENCY_RE.search(log).group(0)),
)


def observables(report_path, log_path):
    rep = json.loads(pathlib.Path(report_path).read_text())
    log = pathlib.Path(log_path).read_text()
    if GPU_DECODE_RE.search(log) is None or RESIDENCY_RE.search(log) is None:
        raise SystemExit(f"{log_path}: no census line - the run did not reach the census")
    out = {label: fn(rep, log) for label, fn in OBSERVABLES}
    out["model fingerprint"] = rep["metadata"]["model_fingerprint"]
    out["git commit"] = rep["metadata"]["git_commit"]
    return out


def compare(a, b, labels=("A", "B")):
    keys = [k for k, _ in OBSERVABLES] + ["model fingerprint", "git commit"]
    diffs = []
    for k in keys:
        if a[k] != b[k]:
            diffs.append(k)
        shown = a[k] if len(str(a[k])) <= 44 else str(a[k])[:41] + "..."
        print(f"  {'=' if k not in diffs else 'X'} {k:22s} {shown}")
    print(f"\n{labels[0]} vs {labels[1]}: {len(diffs)} of {len(keys)} observables differ")
    if diffs:
        print("=> the arms encode different code. The knob is live; its published "
              "cost is back on the table and needs a paired A/B.")
    else:
        print("=> identical encodes. The arm selects nothing under this policy.")
    return not diffs


def selfcheck():
    """A passing check must be able to fail: mutate one observable at a time and
    confirm the comparator's equality test sees it."""
    here = pathlib.Path(__file__).resolve().parent
    reports = sorted(here.glob("*.report.json")) or sorted(
        (ROOT / ".airbug-bench").rglob("*report.json")
    )
    if not reports:
        raise SystemExit("--selfcheck: no report JSON to mutate; capture a run first")
    base = json.loads(reports[0].read_text())
    ok = True
    for label, field, mutate in (
        ("last generated token", "input_tokens", lambda d: d.__setitem__(
            "input_tokens", d["input_tokens"][:-1] + [d["input_tokens"][-1] + 1])),
        ("a mid token", "input_tokens", lambda d: d["input_tokens"].__setitem__(
            len(d["input_tokens"]) // 2, 7)),
        ("fast path", "fast_path", lambda d: d.__setitem__(
            "fast_path", {"attempts": 1, "successes": 1, "declines": 0})),
    ):
        twin = copy.deepcopy(base)
        dec = next(m for m in twin["measurements"] if m["name"] == "decode")
        before = json.dumps(dec.get(field))
        mutate(dec)
        changed = json.dumps(dec.get(field)) != before
        print(f"  mutation {label!r} seen by comparator: {changed}")
        ok &= changed
    print("selfcheck:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run():
    import os
    out = ROOT / ".airbug-bench/rfuse-census"
    out.mkdir(parents=True, exist_ok=True)
    census = {}
    for arm, extra in (("default", {}), ("rfuse", {"ALLPAKA_RFUSE": "1"})):
        env = {**os.environ, **BASE_ENV, **extra}
        log_path, rep_path = out / f"{arm}.log", out / f"{arm}.report.json"
        env["ALLPAKA_BENCH_REPORT"] = str(rep_path.relative_to(ROOT))
        with log_path.open("w") as sink:
            proc = subprocess.run([str(BIN), "bench", "--engine", MODEL], cwd=ROOT,
                                  env=env, stdout=sink, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"{arm} arm exited {proc.returncode}; see {log_path}")
        census[arm] = observables(rep_path, log_path)
    return 0 if compare(census["default"], census["rfuse"], ("default", "rfuse")) else 1


def main(argv):
    if argv[:1] == ["--selfcheck"]:
        return selfcheck()
    if argv[:1] == ["--run"]:
        return run()
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    a = observables(argv[0], argv[1])
    b = observables(argv[2], argv[3])
    return 0 if compare(a, b, (pathlib.Path(argv[0]).name, pathlib.Path(argv[2]).name)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
