#!/usr/bin/env python3
"""Paired A/B of the NORMFLAG policy and of RFUSE under the policy that reaches it.

Arms differ only by environment variables; see preregistration.txt for the design,
the criteria and what the instrument already resolved on this model today.

    python3 drive.py --probe            # structural, uncounted: are the arms different?
    python3 drive.py --out DIR          # control + policy + family series
    python3 drive.py --series rmt,rtopk # the family members not yet priced
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

LOAD_VETO = 8.0
DRIFT_VETO = 0.85
LAUNCH_LOAD = 5.0

ARMS = {
    "norm": {},
    "null": {"ALLPAKA_NORMFLAG": "1"},
    "off": {"ALLPAKA_NORMFLAG": "0"},
    "off_rf": {"ALLPAKA_NORMFLAG": "0", "ALLPAKA_RFUSE": "1"},
    "off_rmt": {"ALLPAKA_NORMFLAG": "0", "ALLPAKA_RMT": "1"},
    "off_k": {"ALLPAKA_NORMFLAG": "0", "ALLPAKA_RTOPK": "1"},
    # The same three family knobs under the *shipped* policy. The
    # unreachable-arm README says these are no-ops because refs.normflag is true
    # by default; that is a statement about the runtime policy value, while the
    # site that matters (metal.rs `let normflag = runtime && <kernel eligibility>`)
    # can already be false for a given model. Probe, do not cite.
    "rf": {"ALLPAKA_RFUSE": "1"},
    "rmt": {"ALLPAKA_RMT": "1"},
    "k": {"ALLPAKA_RTOPK": "1"},
}

# name: challenge vs reference, pair count, as pre-registered.
# Re-pointed at 23:57Z by the probe amendment: every family arm is compared against
# the shipped default, because the probe showed the family is reachable there.
SERIES = {
    "control": ("null", "norm", 6),
    "policy": ("off", "norm", 6),
    "rfuse": ("rf", "norm", 12),
    "rmt": ("rmt", "norm", 8),
    "rtopk": ("k", "norm", 8),
}

# The five observables of the probe: load-independent, so they are valid while the
# machine is busy, and they are what proves two arms differ before pairs are spent.
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
    waited = 0.0
    while load1() > LAUNCH_LOAD and waited < limit_s:
        time.sleep(step_s)
        waited += step_s
    if waited:
        print(f"  quiet gate: held {waited:.0f} s, load now {load1()}", flush=True)
    return waited


def median(values):
    v = sorted(values)
    if not v:
        return None
    n = len(v)
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
    if not gpu:
        raise SystemExit(f"{arm} arm printed no `gpu during decode:` line; see {log}")
    return {
        "arm": arm,
        "at": utc(),
        "load_before": load_before,
        "load": load1(),
        "decode_tok_s": dec["summary"]["median"],
        "decode_tokens": dec.get("input_tokens"),
        "token_stream": json.dumps(dec.get("input_tokens")),
        "fast_path": json.dumps(dec.get("fast_path")),
        "declines": dec.get("fast_path", {}).get("declines"),
        "waits": int(gpu.group(1)),
        "dispatches": int(gpu.group(2)),
        "encode_ms": float(gpu.group(3)),
        "wait_ms": float(gpu.group(4)),
        "decode_ms": float(gpu.group(5)),
        "outside_gpu_pct": int(gpu.group(6)),
        "windows": windows,
        "residency_set": residency_set,
        "census_secs": round(census_at, 2) if census_at else None,
        "process_secs": round(total_secs, 2),
        "fingerprint": data["metadata"]["model_fingerprint"],
        "git_commit": data["metadata"]["git_commit"],
    }


PROBE_KEYS = ("token_stream", "dispatches", "waits", "fast_path", "windows")


def probe(out_dir, arms):
    """One uncounted run per arm, then the five observables side by side.

    Criterion 1 of the design depends on this: an arm that turns out to encode the
    same buffers as its partner is not a series, it is one arm measured twice, which
    is exactly how the 2026-09-12 RFUSE verdict was built.
    """
    seen = {}
    for arm in arms:
        r = run_arm(arm, "probe", out_dir)
        seen[arm] = r
        print(f"  probe {arm:7s} {r['decode_tok_s']:8.2f} tok/s  disp {r['dispatches']}"
              f"  waits {r['waits']}  declines {r['declines']}  windows {r['windows']}"
              f"  outside {r['outside_gpu_pct']}%  census {r['census_secs']}s", flush=True)
    (out_dir / "probe.json").write_text(json.dumps(seen, indent=1) + "\n")
    ref = "norm"
    print(f"\n  vs {ref} (identical => the arm encodes nothing, and no pair is worth its "
          "time):")
    for arm in arms:
        if arm == ref:
            continue
        diff = [k for k in PROBE_KEYS if seen[arm][k] != seen[ref][k]]
        print(f"    {arm:7s} {'IDENTICAL' if not diff else 'differs in ' + ','.join(diff)}")
    return seen


def series(name, out_dir, seen_rates, arm_rates):
    challenge, reference, pairs = SERIES[name]
    rows_path = out_dir / f"{name}-pairs.jsonl"
    probe_path = out_dir / "probe.json"
    probe_obs = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    with rows_path.open("a") as sink:
        for pair in range(1, pairs + 1):
            order = "AB" if pair % 2 else "BA"
            seq = [challenge, reference] if order == "AB" else [reference, challenge]
            row = {"kind": name, "pair": pair, "order": order}
            for slot, arm in zip(("first", "second"), seq):
                wait_quiet()
                time.sleep(25.0)
                rec = run_arm(arm, f"{name}{pair}", out_dir)
                row[slot] = rec
                # Criterion 1: per-arm determinism against the probe, not cross-arm
                # equality - this is a policy switch, so the arms may legitimately
                # run different engines.
                p = probe_obs.get(arm)
                if p and (rec["dispatches"] != p["dispatches"]
                          or rec["fast_path"] != p["fast_path"]):
                    raise SystemExit(
                        f"criterion 1: {arm} in {name} pair {pair} encoded "
                        f"{rec['dispatches']} dispatches / fast_path "
                        f"{rec['fast_path']}, the probe said {p['dispatches']} / "
                        f"{p['fast_path']}")
                if rec["waits"] != TG:
                    raise SystemExit(
                        f"criterion 1: {arm} in {name} pair {pair} waits "
                        f"{rec['waits']} != TG {TG}")
                if row.get("first") and row.get("second") and \
                        row["first"]["token_stream"] != row["second"]["token_stream"]:
                    print(f"  {name} pair {pair}: greedy continuations differ - the "
                          "ablation is not clean on this pair", flush=True)
                    row["tokens_differ"] = True
                print(f"{name} pair {pair} {order} {arm}: {rec['decode_tok_s']} tok/s, "
                      f"{rec['outside_gpu_pct']}% outside GPU, disp {rec['dispatches']},"
                      f" load {rec['load_before']}->{rec['load']}, census "
                      f"{rec['census_secs']}s", flush=True)
            loads = [x for k in ("first", "second")
                     for x in (row[k]["load_before"], row[k]["load"])]
            if max(loads) > LOAD_VETO:
                row["load_vetoed"] = True
                print(f"  {name} pair {pair}: load veto (max {max(loads)} > {LOAD_VETO}) "
                      "- pair dropped", flush=True)
            slow = min(row["first"]["decode_tok_s"], row["second"]["decode_tok_s"])
            pooled = median(seen_rates)
            # Drift is per arm, not per pair. Against the POOLED median of every
            # arm ever seen, a treatment arm that genuinely runs 18 % slow is
            # indistinguishable from a machine that slowed down: with
            # DRIFT_VETO=0.85 the veto deletes exactly the effect the series was
            # built to measure, which is what happened to 8 of the first 10 rfuse
            # pairs here. An arm is therefore compared against its own history.
            checks = [(k, arm_rates.get(row[k]["arm"], [])) for k in ("first", "second")]
            for k, history in checks:
                ref = median(history) if len(history) >= 3 else 0.0
                if ref and row[k]["decode_tok_s"] < DRIFT_VETO * ref:
                    row["drift_vetoed"] = True
                    row["drift_arm"] = row[k]["arm"]
                    row["drift_reference"] = ref
                    print(f"  {name} pair {pair}: drift veto ({row[k]['arm']} "
                          f"{row[k]['decode_tok_s']:.2f} under {DRIFT_VETO:.0%} of its "
                          f"own running median {ref:.2f}, pooled {pooled:.2f} "
                          f"against slowest arm {slow:.2f}) - pair dropped", flush=True)
            for k in ("first", "second"):
                arm_rates.setdefault(row[k]["arm"], []).append(row[k]["decode_tok_s"])
            seen_rates += [row["first"]["decode_tok_s"], row["second"]["decode_tok_s"]]
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return rows_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-arms", default="norm,null,off,off_rf")
    ap.add_argument("--series", default="control,policy,rfuse,rmt,rtopk",
                    help="comma list of control,policy,rfuse,rmt,rtopk")
    args = ap.parse_args()

    out_dir = args.out or ROOT / ".airbug-bench" / (
        "normflag-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pinned = out_dir / "allpaka"
    if not pinned.exists():
        subprocess.run(["cp", str(BIN), str(pinned)], check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    provenance = {
        "started": utc(),
        "binary_sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
        "git_head": head,
        "tree": "clean HEAD worktree; the ablated variables are NORMFLAG and the MoE "
                "family envs",
        "model": str(MODEL.relative_to(ROOT)),
        "pp_tg": [PP, TG],
        "arms": ARMS,
        "top_cpu_at_launch": tenants(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance), flush=True)

    if args.probe:
        probe(out_dir, [a for a in args.probe_arms.split(",") if a])
        return 0

    want = [s for s in args.series.split(",") if s]
    unknown = [s for s in want if s not in SERIES]
    if unknown:
        raise SystemExit(f"unknown series: {unknown}")
    wait_quiet()
    prime = run_arm("norm", "prime", out_dir)
    print(f"priming run: {prime['decode_tok_s']} tok/s, census at {prime['census_secs']}s "
          "(not a result)", flush=True)
    if not (out_dir / "probe.json").exists():
        print("probing arms (uncounted) before spending pairs", flush=True)
        needed = sorted({a for s in want for a in SERIES[s][:2]})
        probe(out_dir, needed)
    seen_rates = [prime["decode_tok_s"]]
    arm_rates = {"norm": [prime["decode_tok_s"]]}
    for name in want:
        series(name, out_dir, seen_rates, arm_rates)
    print("pairs:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
