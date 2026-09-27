#!/usr/bin/env python3
"""ALLPAKA_MV_ID on GLM-4.5-Air: paired campaign driver (task #9).

Runs the protocol in preregistration.txt beside this file and computes nothing:
analyze-mv-id.py turns the files this leaves behind into the verdict.

  calibration  4 processes of the shipped `on` arm at load < 5, which set gate A3's
               prefill band before any counted pair exists.
  priming      one discarded pair (page cache: whoever first touches a 68 GiB
               mapping pays the fault-in, and that penalty must not land on an arm).
  rounds       8 rounds, each running one real pair (on vs off) and one null pair
               (on vs `MV_ID=1`, an identical encode), with the null pair taking the
               opposite order so each design is 4 AB + 4 BA. Rounds 1/3/5/7 add a
               control pair (on vs `Q5_0_MV=0`, a published ~16% knob) — the
               positive control that makes a null reading here mean something.

Arms differ only by env, on one pinned binary. Every number is read from the
`ALLPAKA_BENCH_REPORT` JSON at full float, plus the per-process GPU counter lines
from that process's own stdout. The report's `metadata.resolved_overrides` is
recorded so the analyzer can assert each arm saw the env it claims, rather than
trusting this driver's bookkeeping.

    python3 drive-mv-id.py --out RUN_DIR                 # calibrate + prime + 8 rounds
    python3 drive-mv-id.py --out RUN_DIR --rounds 2      # short pilot of the same code
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(ROOT, "models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf")
DEFAULT_BINARY = os.path.join(ROOT, "docs/benchmarks/2026-09-26-glm-llama-parity/series-1/allpaka")
ARM_SHA = "7b8ec67aa72c42662ba4608fa1629c139c759c6ce7bacdd6c6aa6fbcc46fe2fe"
PP, TG = 480, 32
COOLDOWN_MS = 1500

BASE_ENV = {
    "ALLPAKA_BENCH_PP": str(PP),
    "ALLPAKA_BENCH_TG": str(TG),
    "ALLPAKA_BENCH_SKIP_MTP": "1",
    "ALLPAKA_PROFILE": "max-performance",
}

# Arm -> the env it adds. `on` is the shipped default and is taken with the
# variable UNSET, so the arm that is compared against the incumbent is the
# incumbent exactly as every other artifact measured it. Only the literals "0"
# and "1" are passed anywhere: both knobs read `!= "0"`, so any other string
# (including "off") means ON.
ARMS = {
    "on": {},
    "off": {"ALLPAKA_MV_ID": "0"},
    "null": {"ALLPAKA_MV_ID": "1"},
    "q5off": {"ALLPAKA_Q5_0_MV": "0"},
}

GPU_RE = re.compile(
    r"gpu during decode: (\d+) waits, (\d+) dispatches, encode (\d+) ms, wait (\d+) ms "
    r"of (\d+) ms total \((\d+)% outside the GPU\)")
CLK_RE = re.compile(
    r"gpu clock during decode: executing (\d+) ms, scheduling (\d+) ms, "
    r"round trips \+ idle (\d+) ms \(of (\d+) ms waited\)")


def load1():
    raw = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True,
                         text=True).stdout
    return float(raw.strip().strip("{}").split()[0])


def top_busy(n=8):
    """Instantaneous per-process CPU + machine memory state.

    `ps %cpu` is a lifetime average and a single-sample `top -l 1` reports
    since-boot averages (it printed 0.0 for processes at 40-88% in the 2026-09-25
    co-tenant artifact), so this takes two samples and reads the last block.
    """
    r = subprocess.run(["top", "-l", "2", "-n", str(n), "-o", "cpu",
                        "-stats", "pid,command,cpu"], capture_output=True, text=True)
    block = r.stdout.split("Processes:")[-1].splitlines()
    summary = [l.strip() for l in block[:12] if l.strip()]
    rows, seen = [], False
    for line in block:
        parts = line.strip().split()
        # -stats pid,command,cpu puts COMMAND before %CPU and commands hold spaces,
        # so the cpu value comes from the right end and the name is the middle.
        if seen and len(parts) >= 3 and parts[0].isdigit():
            try:
                cpu = float(parts[-1].rstrip("%"))
            except ValueError:
                continue
            rows.append({"pid": int(parts[0]), "cpu": cpu, "comm": " ".join(parts[1:-1])})
        seen = seen or line.strip().startswith("PID")
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                          text=True).stdout.strip()
    return {"busy": rows, "summary": summary, "swapusage": swap}


class Sampler(threading.Thread):
    """Append-only covariates from the driver process: load every 5 s, a CPU/memory
    snapshot every 60 s. Pairs are attributed to these afterwards by file
    timestamps, so nothing here runs inside a measured process."""

    def __init__(self, out):
        super().__init__(daemon=True)
        self.stop_at = threading.Event()
        self.loads = open(os.path.join(out, "loads.jsonl"), "a", buffering=1)
        self.snaps = open(os.path.join(out, "snaps.jsonl"), "a", buffering=1)

    def run(self):
        since = 1e9
        while not self.stop_at.is_set():
            now = time.time()
            # A covariate series that dies silently stops being a detector, so the
            # sampler survives its own bugs and leaves a visible gap in the record.
            try:
                self.loads.write(json.dumps({"t": now, "load": load1()}) + "\n")
                if now - since >= 60:
                    self.snaps.write(json.dumps({"t": now, **top_busy()}) + "\n")
                    since = now
            except Exception as e:  # noqa: BLE001
                self.loads.write(json.dumps({"t": now, "error": repr(e)}) + "\n")
            self.stop_at.wait(5)


def sha(path):
    return subprocess.run(["shasum", "-a", "256", path], capture_output=True,
                          text=True).stdout.split()[0]


def build_busy():
    """""""""
    A live compiler or another bench, as a text block ('' when clear). `-x` and
    not `-f cargo`: this tree's shell wrappers carry the word cargo in their
    command lines and are not builds.
    """""""""
    return "\n".join(subprocess.run(["pgrep", "-xl", nm], capture_output=True,
                                     text=True).stdout
                      for nm in ("cargo", "rustc", "llama-bench")).strip()


def wait_quiet(ceiling, tries=60, poll=30):
    """""""""
    Permit one launch: load under `ceiling` *and* no live compiler. The compiler
    check used to run once, at startup; the 2026-09-26 campaign died of that, because
    a build that began two minutes in put rustc at 694% and depressed 8 of the 22
    processes without ever being sampled.
    """""""""
    for _ in range(tries):
        l = load1()
        if l < ceiling and not build_busy():
            return l
        time.sleep(poll)
    return None


def window(path):
    st = os.stat(path)
    return st.st_birthtime, st.st_mtime


def run_arm(binary, out, round_dir, arm, tag):
    """One allpaka bench process of one arm; returns its record or raises."""
    d = os.path.join(out, round_dir)
    os.makedirs(d, exist_ok=True)
    rep = os.path.join(d, f"{arm}-{tag}.json")
    txt = os.path.join(d, f"{arm}-{tag}.txt")
    env = dict(os.environ, **BASE_ENV, **ARMS[arm], ALLPAKA_BENCH_REPORT=rep)
    t0 = time.time()
    with open(txt, "w") as fh:
        rc = subprocess.run([binary, "bench", "--engine", MODEL], cwd=ROOT, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    with open(txt) as fh:
        log = fh.read()
    g, c = GPU_RE.search(log), CLK_RE.search(log)
    ms = {m["name"]: m for m in json.load(open(rep))["measurements"]}
    dec = ms["decode"]
    meta = json.load(open(rep))["metadata"]
    return {
        "arm": arm, "rc": rc, "t0": t0, "t1": time.time(),
        "pp": ms["prefill"]["summary"]["median"],
        "tg": dec["summary"]["median"],
        "samples": dec.get("samples_tok_s"),
        "fast_path": dec["fast_path"],
        "waits": int(g.group(1)) if g else None,
        "dispatches": int(g.group(2)) if g else None,
        "encode_ms": int(g.group(3)) if g else None,
        "wait_ms": int(g.group(4)) if g else None,
        "total_ms": int(g.group(5)) if g else None,
        "outside_pct": int(g.group(6)) if g else None,
        "exec_ms": int(c.group(1)) if c else None,
        "clock_wait_ms": int(c.group(4)) if c else None,
        "git_commit": meta.get("git_commit"),
        "fingerprint": meta.get("model_fingerprint"),
        "overrides": meta.get("resolved_overrides"),
        "win": window(rep),
        "win_log": window(txt),
    }


def run_pair(binary, out, round_dir, a, b, kind, order, prime=False, log=print):
    first, second = (a, b) if order == "AB" else (b, a)
    r1 = run_arm(binary, out, round_dir, first, kind)
    time.sleep(COOLDOWN_MS / 1000.0)
    r2 = run_arm(binary, out, round_dir, second, kind)
    ra, rb = (r1, r2) if order == "AB" else (r2, r1)
    rec = {"round": round_dir, "kind": kind, "order": order, "prime": prime,
           "A": ra, "B": rb}
    with open(os.path.join(out, "pairs.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"  {round_dir} {kind:8s} {order}  {ra['arm']} {ra['tg']:6.2f} / {rb['arm']} {rb['tg']:6.2f}"
        f"  ratio {ra['tg'] / rb['tg']:.4f}  pp {ra['pp']:6.1f}/{rb['pp']:6.1f}"
        f"  outside {ra['outside_pct']}%/{rb['outside_pct']}%"
        f"  disp {ra['dispatches']}/{rb['dispatches']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--calibrate", type=int, default=4)
    ap.add_argument("--no-prime", action="store_true")
    ap.add_argument("--load-gate", type=float, default=5.0)
    args = ap.parse_args()
    # Engine subprocesses run with cwd=ROOT, so every path handed to them must be
    # absolute: a relative --out pinned a binary the spawn could not find.
    args.out = os.path.abspath(args.out)
    args.binary = os.path.abspath(args.binary)

    busy = build_busy()
    if busy:
        print("refusing to start: a build or another bench is live\n" + busy, file=sys.stderr)
        return 2
    got = sha(args.binary)
    if got != ARM_SHA:
        print(f"refusing to start: binary sha {got[:12]} != preregistered {ARM_SHA[:12]}",
              file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    pinned = os.path.join(args.out, "allpaka")
    if not os.path.exists(pinned):
        shutil.copyfile(args.binary, pinned)
        os.chmod(pinned, 0o755)   # copyfile does not carry the exec bit
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = len(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines())
    prov = {"started": time.strftime("%FT%TZ", time.gmtime()),
            "allpaka_sha256": got, "binary_mtime_local": time.strftime(
                "%F %T", time.localtime(os.stat(args.binary).st_mtime)),
            "head_at_pin_time": head, "tree_dirty_files_at_pin_time": dirty,
            "note": "head_at_pin_time is the tree state when the binary was copied, "
                    "not necessarily when it was compiled; the per-process "
                    "metadata.git_commit in each report is the build-time stamp and "
                    "the sha256 is the only exact arm identity.",
            "model": MODEL, "pp": PP, "tg": TG, "base_env": BASE_ENV,
            "arms": ARMS, "load_gate": args.load_gate}
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        fh.write(json.dumps(prov, indent=1) + "\n")
    drv = open(os.path.join(args.out, "driver.txt"), "a", buffering=1)

    def say(s):
        print(s, flush=True)
        drv.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())}  {s}\n")

    say(f"# run {args.out}  binary {got[:16]}  HEAD at pin {head[:8]}  dirty {dirty}")
    sampler = Sampler(args.out)
    sampler.start()
    try:
        cal = []
        for i in range(1, args.calibrate + 1):
            l = wait_quiet(args.load_gate)
            if l is None:
                say(f"  calibration {i}: load gate timed out at {load1():.2f}")
                break
            d = os.path.join(args.out, "calibration")
            os.makedirs(d, exist_ok=True)
            rep = os.path.join(d, f"cal{i}.json")
            with open(os.path.join(d, f"cal{i}.txt"), "w") as fh:
                subprocess.run([pinned, "bench", "--engine", MODEL], cwd=ROOT,
                               env=dict(os.environ, **BASE_ENV, **ARMS["on"],
                                        ALLPAKA_BENCH_REPORT=rep),
                               stdout=fh, stderr=subprocess.STDOUT)
            ms = {m["name"]: m for m in json.load(open(rep))["measurements"]}
            cal.append({"pp": ms["prefill"]["summary"]["median"],
                        "tg": ms["decode"]["summary"]["median"], "load": l})
            say(f"  calibration {i} at load {l:.2f}: prefill {cal[-1]['pp']:.2f}  "
                f"decode {cal[-1]['tg']:.2f}")
        pps = sorted(x["pp"] for x in cal)
        centre = pps[len(pps) // 2] if pps else None
        with open(os.path.join(args.out, "calibration.json"), "w") as fh:
            json.dump({"calibration": cal, "prefill_median": centre,
                       "band_floor": 0.85 * centre if centre else None}, fh, indent=1)
        say(f"  thermometer: median prefill {centre}  A3 floor {0.85 * centre if centre else '-'}")

        if not args.no_prime:
            wait_quiet(args.load_gate)
            run_pair(pinned, args.out, "round-prime", "on", "off", "prime", "AB",
                     prime=True, log=say)
        for r in range(1, args.rounds + 1):
            l = wait_quiet(args.load_gate)
            if l is None:
                say(f"  round {r}: load gate timed out at {load1():.2f} - campaign pauses")
                break
            rd = f"round{r}"
            real_order = "AB" if r % 2 else "BA"
            run_pair(pinned, args.out, rd, "on", "off", "real", real_order, log=say)
            time.sleep(COOLDOWN_MS / 1000.0)
            run_pair(pinned, args.out, rd, "on", "null", "null",
                     "BA" if real_order == "AB" else "AB", log=say)
            if r % 2:
                time.sleep(COOLDOWN_MS / 1000.0)
                run_pair(pinned, args.out, rd, "on", "q5off", "control",
                         "AB" if r % 4 == 1 else "BA", log=say)
    finally:
        sampler.stop_at.set()
        sampler.join(timeout=10)
    say(f"  DONE -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
