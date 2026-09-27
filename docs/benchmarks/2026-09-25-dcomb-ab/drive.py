#!/usr/bin/env python3
"""Paired A/B of the decode down+combine fold (ALLPAKA_DCOMB), default ON.

Arms differ only by the DCOMB env var: `plain` leaves it unset (the shipped
default, fold ON), `unfold` sets it to 0 (fold off), and `null` sets it to 1,
which `dcomb()` - `map_or(true, |v| v != "0")` - treats identically to unset, so
the control series measures this machine's noise floor for the same statistic.
Criteria and the prior are in preregistration.txt.

    python3 drive.py --real 14 --control 6
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
MODEL = ROOT / "models/qwen3-30b-a3b-Q4_K_M.gguf"
BIN = ROOT / "target/release/allpaka"
PP, TG = 3072, 320

# Instrument vetoes, added after a contended control series (see the amendment
# in preregistration.txt): LOAD_VETO drops a pair whose 1-min load average
# crossed this at either end of either arm; DRIFT_VETO drops a pair whose
# slower arm falls under this fraction of the running median of all arms seen,
# which catches a bandwidth thief that the load average reports late.
LOAD_VETO = 8.0
DRIFT_VETO = 0.85
# Don't start a 90-minute run into a burst: wait for the machine instead.
LAUNCH_LOAD = 5.0
LAUNCH_WAIT_S = 45 * 60

ARMS = {
    "plain": {},
    "unfold": {"ALLPAKA_DCOMB": "0"},
    "null": {"ALLPAKA_DCOMB": "1"},
}

RESIDENCY_RE = re.compile(r"residency: windows=(\d+) set=(\w+)")
GPU_DECODE_RE = re.compile(
    r"gpu during decode: (\d+) waits, (\d+) dispatches, encode (\d+) ms, wait (\d+) ms "
    r"of (\d+) ms total \((\d+)% outside the GPU\)"
)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load1():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(out.split()[1])


def tenants(n=8):
    out = subprocess.run(["ps", "-Aceo", "pid,pcpu,comm", "-r"],
                         capture_output=True, text=True).stdout
    return out.splitlines()[: n + 1]


def wait_quiet(limit_s=40 * 60, step_s=30.0):
    """Spend wall clock, not data: hold until the machine is quiet before an arm.

    Added after the first two launches of this design were started at load 4.5
    and ran their arms at 8-37.5 (see the amendment at the bottom of
    preregistration.txt). A pair whose arms were both quiet is worth more than
    three pairs sampled through a burst, and the run has no deadline.
    """
    waited = 0.0
    while load1() > LAUNCH_LOAD and waited < limit_s:
        time.sleep(step_s)
        waited += step_s
    if waited:
        print(f"  quiet gate: held {waited:.0f} s, load now {load1()}", flush=True)
    return waited


def median(values):
    v = sorted(values)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def run_arm(arm, tag, out_dir):
    env = {**os.environ, "ALLPAKA_BENCH_PP": str(PP), "ALLPAKA_BENCH_TG": str(TG),
           "ALLPAKA_BENCH_SKIP_MTP": "1", "ALLPAKA_PROFILE": "max-performance",
           "ALLPAKA_BENCH_REPORT": str(out_dir / f"{arm}-{tag}-report.json")}
    env.update(ARMS[arm])
    log = out_dir / f"{arm}-{tag}-log.txt"
    started = time.monotonic()
    load_before = load1()
    census_at, windows, residency_set = None, None, None
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
        "load_before": load_before,
        "load": load1(),
        "decode_tok_s": dec["summary"]["median"],
        "decode_tokens": dec.get("input_tokens"),
        "declines": dec.get("fast_path", {}).get("declines"),
        "waits": int(gpu.group(1)) if gpu else None,
        "dispatches": int(gpu.group(2)) if gpu else None,
        "encode_ms": float(gpu.group(3)) if gpu else None,
        "wait_ms": float(gpu.group(4)) if gpu else None,
        "decode_ms": float(gpu.group(5)) if gpu else None,
        "outside_gpu_pct": int(gpu.group(6)) if gpu else None,
        "windows": windows,
        "residency_set": residency_set,
        "census_secs": round(census_at, 2) if census_at else None,
        "process_secs": round(total_secs, 2),
        "fingerprint": data["metadata"]["model_fingerprint"],
        "git_commit": data["metadata"]["git_commit"],
    }


def series(kind, pairs, out_dir, seen_rates):
    challenge, reference = (("unfold", "plain") if kind == "real" else ("null", "plain"))
    rows_path = out_dir / f"{kind}-pairs.jsonl"
    with rows_path.open("a") as sink:
        for pair in range(1, pairs + 1):
            order = "AB" if pair % 2 else "BA"
            seq = [challenge, reference] if order == "AB" else [reference, challenge]
            row = {"kind": kind, "pair": pair, "order": order}
            for slot, arm in zip(("first", "second"), seq):
                wait_quiet()
                time.sleep(25.0)
                rec = run_arm(arm, f"{kind}{pair}", out_dir)
                if rec["declines"]:
                    raise SystemExit(
                        f"criterion 1: {arm} arm reported {rec['declines']} CPU declines"
                    )
                row[slot] = rec
                if row.get("first") and row.get("second") and \
                        row["first"]["decode_tokens"] != row["second"]["decode_tokens"]:
                    print(f"  {kind} pair {pair}: greedy continuations differ - the "
                          "ablation is not clean on this pair", flush=True)
                    row["tokens_differ"] = True
                print(f"{kind} pair {pair} {order} {arm}: {rec['decode_tok_s']} tok/s, "
                      f"{rec['outside_gpu_pct']}% outside GPU, load {rec['load_before']}"
                      f"->{rec['load']}, census {rec['census_secs']}s", flush=True)
            # Two instrument vetoes, both published as drop counts. Added after
            # three identical-code control pairs gave ratios of 2.03, 1.12 and
            # 0.91 at load averages of 37.5, 18.7 and 8.5 - see
            # contended-control-pairs.jsonl and the amendment in
            # preregistration.txt.
            loads = [x for k in ("first", "second")
                     for x in (row[k]["load_before"], row[k]["load"])]
            if max(loads) > LOAD_VETO:
                row["load_vetoed"] = True
                print(f"  {kind} pair {pair}: load veto (max {max(loads)} > {LOAD_VETO}) "
                      "- pair dropped", flush=True)
            slow = min(row["first"]["decode_tok_s"], row["second"]["decode_tok_s"])
            ref = median(seen_rates)
            if ref and slow < DRIFT_VETO * ref:
                row["drift_vetoed"] = True
                row["drift_reference"] = ref
                print(f"  {kind} pair {pair}: drift veto (slowest arm {slow:.2f} under "
                      f"{DRIFT_VETO:.0%} of running median {ref:.2f}) - pair dropped",
                      flush=True)
            # Appended after the check, so a pair is never compared against a
            # median that already contains its own two arms.
            seen_rates += [row["first"]["decode_tok_s"], row["second"]["decode_tok_s"]]
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return rows_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=14)
    ap.add_argument("--control", type=int, default=6)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    out_dir = args.out or ROOT / ".airbug-bench" / (
        "dcomb-ab-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
        "tree": "clean HEAD worktree; the ablated variable is ALLPAKA_DCOMB",
        "model": str(MODEL.relative_to(ROOT)),
        "pp_tg": [PP, TG],
        "arms": ARMS,
        "top_cpu_at_launch": tenants(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance), flush=True)

    wait_quiet()
    prime = run_arm("plain", "prime", out_dir)
    print(f"priming run: {prime['decode_tok_s']} tok/s, census at {prime['census_secs']}s "
          "(not a result)", flush=True)

    # One rate pool for the whole process, as the 21:15Z amendment words it, so a
    # real pair's drift reference includes the control arms that ran before it.
    seen_rates = [prime["decode_tok_s"]]
    series("control", args.control, out_dir, seen_rates)
    series("real", args.real, out_dir, seen_rates)
    print("pairs:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
