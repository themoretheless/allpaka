#!/usr/bin/env python3
"""Paired A/B of the ROWS verify mapping against the per-row fallback.

Arms differ only by ALLPAKA_NO_EXPERT_ROWS (see preregistration.txt for the
criteria this implements). One process per arm, orders alternated AB/BA.

    python3 drive.py --pairs 14
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
MODEL = ROOT / "models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
BIN = ROOT / "target/release/allpaka"
PP, TG, K = 480, 32, 4

ARMS = {"rows": {}, "norows": {"ALLPAKA_NO_EXPERT_ROWS": "1"}}

DURATION = {"ns": 1e-6, "us": 1e-3, "\u00b5s": 1e-3, "ms": 1.0, "s": 1e3}
VERIFY_RE = re.compile(
    r"^  verify (\d+) tok: ([\d.]+)(ns|us|\u00b5s|ms|s) \((\d+) dispatches, wait ([\d.]+) ms\)$",
    re.M,
)
ACCEPT_RE = re.compile(r"acceptance (\d+)/(\d+)")


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
    """One bench process; returns the per-arm record that goes into a pair row."""
    env = {
        "ALLPAKA_BENCH_PP": str(PP),
        "ALLPAKA_BENCH_TG": str(TG),
        "ALLPAKA_DRAFT_K": str(K),
        "ALLPAKA_MTP_DEBUG": "1",
        "ALLPAKA_ROWS_DEBUG": "1",
        "ALLPAKA_BENCH_REPORT": str(out_dir / f"{arm}-{tag}-report.json"),
    }
    env.update(ARMS[arm])
    log = out_dir / f"{arm}-{tag}-log.txt"
    with log.open("w") as handle:
        subprocess.run(
            [str(out_dir / "allpaka"), "bench", "--engine", str(MODEL)],
            cwd=ROOT,
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    text = log.read_text()
    data = json.loads((out_dir / f"{arm}-{tag}-report.json").read_text())
    medians = {m["name"]: m["summary"]["median"] for m in data["measurements"]}
    spec = next(m for m in data["measurements"] if m["name"] == "mtp-speculative")
    phases = {p["name"]: p for p in spec["phases"]}
    verifies = [
        (float(val) * DURATION[unit], int(disp))
        for _tok, val, unit, disp, _wait in VERIFY_RE.findall(text)
    ]
    accept = ACCEPT_RE.search(text)
    return {
        "arm": arm,
        "at": utc(),
        "load": load1(),
        "plain_tok_s": medians.get("mtp-plain"),
        "spec_tok_s": medians.get("mtp-speculative"),
        "rounds": phases["rounds"]["calls"] if "rounds" in phases else None,
        "spec_dispatches": phases["gpu"]["calls"] if "gpu" in phases else None,
        "spec_gpu_ms": phases["gpu"]["milliseconds"] if "gpu" in phases else None,
        "rows_pipeline_engaged": text.count("verify rows pipelines: true"),
        "verify_rounds_logged": len(verifies),
        "verify_ms": median([ms for ms, _ in verifies]),
        "verify_dispatches": median([d for _ms, d in verifies]),
        "accepted": int(accept.group(1)) if accept else None,
        "drafted": int(accept.group(2)) if accept else None,
        "stream_pass": "mtp stream matches plain greedy: PASS" in text,
        "model_fingerprint": data["metadata"]["model_fingerprint"],
        "git_commit": data["metadata"]["git_commit"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=14)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    out_dir = args.out or ROOT / ".airbug-bench" / (
        "mtp-rows-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pinned = out_dir / "allpaka"
    subprocess.run(["cp", str(BIN), str(pinned)], check=True)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "started": utc(),
        "binary_sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
        "git_head": head,
        "tree": "dirty by design; the ablation switch is ALLPAKA_NO_EXPERT_ROWS",
        "model": str(MODEL.relative_to(ROOT)),
        "pp_tg_k": [PP, TG, K],
        "m_of_round": K + 1,
        "arms": ARMS,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance), flush=True)

    with (out_dir / "pairs.jsonl").open("a") as sink:
        for pair in range(1, args.pairs + 1):
            order = "AB" if pair % 2 else "BA"
            names = ["rows", "norows"] if order == "AB" else ["norows", "rows"]
            row = {"pair": pair, "order": order}
            for arm in names:
                time.sleep(args.cooldown)
                rec = run_arm(arm, f"p{pair}", out_dir)
                row[arm] = rec
                print(
                    f"pair {pair} {order} {arm}: spec {rec['spec_tok_s']} "
                    f"plain {rec['plain_tok_s']} verify {rec['verify_ms']} ms in "
                    f"{rec['verify_dispatches']} dispatches, rows engaged "
                    f"{rec['rows_pipeline_engaged']}, load {rec['load']}",
                    flush=True,
                )
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    print("pairs:", out_dir / "pairs.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
