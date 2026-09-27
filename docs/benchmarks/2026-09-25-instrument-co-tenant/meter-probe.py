#!/usr/bin/env python3
"""Throwaway probe: what is the streamed byte rate right now, and who is on the machine?

One line per iteration: the harness's own published ratios for the two x-shared
byte meters, next to the 1-minute load, the top CPU tenants and disk throughput.
Written to answer why the task #25 preflight saw 0.74-0.76x of published through
20 consecutive probes, and why a control arm of this machine's own harness just
ran 3.07x slower with an unchanged dispatch count.

    python3 .airbug-bench/meter-probe.py --iterations 8
"""

import argparse
import pathlib
import re
import subprocess
import time

BIN = pathlib.Path("target/release/build/allpaka-backend/402d2f60b4b17b8a/out/"
                   "gpu_ffnbench-402d2f60b4b17b8a")
METER_RE = re.compile(r"(q4_k meter|q5_k meter)\s+\[[\d,]+\]\s+feed=\d+ \| shared ([\d.]+) "
                      r"spread [\d.]+x \| vs published ([\d.]+)x")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def snapshot():
    """1-minute load average, the top CPU tenants, and disk throughput if iostat answers.

    iostat blocked this probe for seven minutes on 2026-09-25, so it now runs
    under a three second deadline; the CPU tenants are the covariate. `vm.loadavg`
    prints `{ 1-min 5-min 15-min }`, and this probe strips the braces, so the
    1-minute value is index 0 - earlier revisions took index 1 and published the
    5-minute average; see the dated correction in README.md.
    """
    top = subprocess.run(["ps", "-Aceo", "pcpu,comm", "-r"],
                         capture_output=True, text=True).stdout.splitlines()[1:4]
    load = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True,
                          text=True).stdout.strip("{}").split()[0]
    try:
        out = subprocess.run(["iostat", "-d", "-w", "1", "-n", "2"],
                             capture_output=True, text=True, timeout=3).stdout
        row = [l for l in out.splitlines() if l.strip()][-1].split()
        disk = f"{row[1]} tps {row[2]} MB/s"
    except Exception:
        disk = "iostat silent"
    return load, " / ".join(t.strip() for t in top), disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--gap", type=float, default=20.0)
    args = ap.parse_args()
    for i in range(1, args.iterations + 1):
        load, top, disk = snapshot()
        out = subprocess.run([str(BIN), "q35_down_shape", "--ignored", "--nocapture",
                              "--test-threads=1"], capture_output=True, text=True,
                             env=dict(**{k: str(v) for k, v in __import__("os").environ.items()},
                                      ALLPAKA_BENCH_FEEDS="1")).stdout
        meters = "  ".join(f"{m.group(1)} {m.group(2)} GB/s ({m.group(3)}x)"
                           for m in METER_RE.finditer(out))
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())}  load {load:>5}  {meters}"
              f"  | top: {top}  | disk: {disk}", flush=True)
        time.sleep(args.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
