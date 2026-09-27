#!/usr/bin/env python3
"""Paired A/B of the wired weight-window cap (ALLPAKA_GPU_WINDOW_GIB).

Arms differ only by the cap; `null` is a no-op setting of the same variable, so
the control series measures today's noise floor for this statistic. Criteria,
bands and the mechanism inventory are in preregistration.txt.

    python3 drive.py --real 12 --control 6
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BIN = ROOT / "target/release/allpaka"
PP, TG = 64, 400

# window=None means the shipped WINDOW_CAP path (32 GiB ceiling).
ARMS = {
    "wide": {},
    "narrow": {"ALLPAKA_GPU_WINDOW_GIB": "2"},
    "null": {"ALLPAKA_GPU_WINDOW_GIB": "32"},
}
# 20.61 GiB of file, 2 GiB windows stepping 1 GiB -> starts 0..19.
EXPECT_WINDOWS = {"wide": 1, "null": 1, "narrow": 20}

RESIDENCY_RE = re.compile(r"residency: windows=(\d+) set=(\w+)")
MAPPED_RE = re.compile(r"in (\d+) windows \(maxBufferLength ([\d.]+) GiB\)")
GPU_DECODE_RE = re.compile(
    r"gpu during decode: (\d+) waits, (\d+) dispatches, encode (\d+) ms, wait (\d+) ms "
    r"of (\d+) ms total"
)
DECODE_RE = re.compile(r"^\s+decode\s+(\d+) tok in\s+([\d.]+) s\s+([\d.]+) tok/s", re.M)


def tenants(n=8):
    out = subprocess.run(["ps", "-Aceo", "pid,pcpu,comm", "-r"],
                         capture_output=True, text=True).stdout
    return out.splitlines()[: n + 1]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load1():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(out.split()[1])


def median(values):
    v = sorted(values)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def run_arm(arm, tag, out_dir):
    """One bench process; stdout is read live so the load census can be timed."""
    env = {**os.environ, "ALLPAKA_BENCH_PP": str(PP), "ALLPAKA_BENCH_TG": str(TG),
           "ALLPAKA_BENCH_REPORT": str(out_dir / f"{arm}-{tag}-report.json")}
    env.update(ARMS[arm])
    log = out_dir / f"{arm}-{tag}-log.txt"
    started = time.monotonic()
    census_at, windows_mapped, mapped_max_gib = None, None, None
    windows, residency_set = None, None
    lines = []
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [str(out_dir / "allpaka"), "bench", "--engine", str(MODEL)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
        )
        for line in proc.stdout:
            if census_at is None and "residency: windows=" in line:
                census_at = time.monotonic() - started
            m = RESIDENCY_RE.search(line)
            if m:
                windows, residency_set = int(m.group(1)), m.group(2)
            m = MAPPED_RE.search(line)
            if m:
                windows_mapped, mapped_max_gib = int(m.group(1)), float(m.group(2))
            lines.append(line)
            handle.write(line)
        handle.flush()
        rc = proc.wait()
    total_secs = time.monotonic() - started
    text = "".join(lines)
    if rc != 0:
        raise SystemExit(f"{arm} arm exited {rc}; see {log}")
    data = json.loads((out_dir / f"{arm}-{tag}-report.json").read_text())
    dec = next(m for m in data["measurements"] if m["name"] == "decode")
    gpu = GPU_DECODE_RE.search(text)
    return {
        "arm": arm,
        "at": utc(),
        "load": load1(),
        "decode_tok_s": dec["summary"]["median"],
        # The greedy continuation itself is in the report, so token equality is
        # checkable across arms without a second instrument.
        "decode_tokens": dec.get("input_tokens"),
        "declines": dec.get("fast_path", {}).get("declines"),
        "dispatches": int(gpu.group(2)) if gpu else None,
        "encode_ms": float(gpu.group(3)) if gpu else None,
        "wait_ms": float(gpu.group(4)) if gpu else None,
        "decode_ms": float(gpu.group(5)) if gpu else None,
        "windows": windows,
        "windows_in_mapped_line": windows_mapped,
        "max_gib_in_mapped_line": mapped_max_gib,
        "residency_set": residency_set,
        "census_secs": round(census_at, 2) if census_at else None,
        "process_secs": round(total_secs, 2),
        "fingerprint": data["metadata"]["model_fingerprint"],
        "git_commit": data["metadata"]["git_commit"],
    }


def series(kind, pairs, out_dir):
    """kind is 'real' (narrow vs wide) or 'control' (null vs wide)."""
    names = ["narrow", "wide"] if kind == "real" else ["null", "wide"]
    rows_path = out_dir / f"{kind}-pairs.jsonl"
    with rows_path.open("a") as sink:
        for pair in range(1, pairs + 1):
            order = "AB" if pair % 2 else "BA"
            seq = names if order == "AB" else names[::-1]
            row = {"kind": kind, "pair": pair, "order": order}
            for arm in seq:
                time.sleep(20.0)
                rec = run_arm(arm, f"{kind}{pair}", out_dir)
                if rec["windows"] != EXPECT_WINDOWS[arm]:
                    raise SystemExit(
                        f"criterion 1: {arm} reported windows={rec['windows']}, "
                        f"source predicts {EXPECT_WINDOWS[arm]}"
                    )
                row[arm] = rec
                print(f"{kind} pair {pair} {order} {arm}: {rec['decode_tok_s']} tok/s "
                      f"windows={rec['windows']} load {rec['load']} "
                      f"census {rec['census_secs']}s", flush=True)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return rows_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=12)
    ap.add_argument("--control", type=int, default=6)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    out_dir = args.out or ROOT / ".airbug-bench" / (
        "window-cost-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pinned = out_dir / "allpaka"
    subprocess.run(["cp", str(BIN), str(pinned)], check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    provenance = {
        "started": utc(),
        "binary_sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
        "git_head": head,
        "tree": "dirty by design; the ablated variable is ALLPAKA_GPU_WINDOW_GIB",
        "model": str(MODEL.relative_to(ROOT)),
        "pp_tg": [PP, TG],
        "arms": ARMS,
        "expected_windows": EXPECT_WINDOWS,
        "top_cpu_at_launch": tenants(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance), flush=True)

    # One uncounted run to fault the 20.6 GiB mapping into the page cache: the
    # pilot showed a cold first arm (21.75 s to the residency census against
    # 3.42 s warm, and 6.7% off its pair partner's tok/s).
    prime = run_arm("wide", "prime", out_dir)
    print(f"priming run: {prime['decode_tok_s']} tok/s, census at "
          f"{prime['census_secs']}s (not a result)", flush=True)

    # Control first: it decides what the real series can prove, and finding that
    # out after 12 real pairs would waste them.
    series("control", args.control, out_dir)
    series("real", args.real, out_dir)
    print("pairs:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
