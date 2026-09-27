#!/usr/bin/env python3
"""GLM-4.5-Air decode parity vs llama.cpp: paired campaign driver (task #16).

Runs the protocol in preregistration.txt beside this file. It does not compute a
verdict - analyze-glm-parity.py does that from the files this leaves behind.

  calibration  four allpaka-only processes at load < 5, which set the thermometer
               band (gate A3) before any counted pair exists.
  priming      one discarded round: the first process to touch a 68 GiB mapping
               pays the disk fault-in, and that penalty would otherwise land on
               whichever arm happened to run first.
  rounds       `allpaka airbug --repeats 2`, so each round contributes one AB pair
               and one BA pair and campaign drift cannot be handed to one engine.

Every arm's numbers come from the JSON each engine writes, never from a print. The
driver only records the machine state around them (load series, co-tenant snapshots,
per-arm wall windows from file birth/modification times) and the per-process GPU
counters, which is what the gates need and nothing that can move a number.

    python3 drive-glm-parity.py --out RUN_DIR            # calibrate + prime + 6 rounds
    python3 drive-glm-parity.py --out RUN_DIR --rounds 0 --calibrate 4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(ROOT, "models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf")
PP, TG = 480, 32

# airbug sets these for the allpaka arm; the calibration processes must carry the
# identical environment or the thermometer measures a different code path.
BENCH_ENV = {
    "ALLPAKA_BENCH_PP": str(PP),
    "ALLPAKA_BENCH_TG": str(TG),
    "ALLPAKA_BENCH_SKIP_MTP": "1",
    "ALLPAKA_PROFILE": "max-performance",
}


def load1():
    raw = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(raw.strip().strip("{}").split()[0])


def top_busy(n=5):
    """Instantaneous per-process CPU plus the machine's memory state.

    `ps %cpu` is a lifetime average and cannot see a burst, and a single-sample
    `top -l 1` reports since-boot averages, which is why this asks for two samples
    and reads the last one (the 2026-09-25 co-tenant artifact's numbers came from
    exactly that form). Costs ~1.8 s once a minute; the second sample is the one
    whose %CPU means anything, so the summary lines come from the same block.
    """
    r = subprocess.run(["top", "-l", "2", "-n", str(n), "-o", "cpu",
                        "-stats", "pid,command,cpu"], capture_output=True, text=True)
    block = r.stdout.split("Processes:")[-1].splitlines()
    summary = [l.strip() for l in block[:12] if l.strip()]
    rows, seen = [], False
    for line in block:
        parts = line.strip().split()
        # -stats pid,command,cpu prints them in that order, and a command can hold
        # spaces ("Qoder Helper (Renderer)"), so read the cpu from the right end.
        if seen and len(parts) >= 3 and parts[0].isdigit():
            try:
                cpu = float(parts[-1].rstrip("%"))
            except ValueError:
                continue
            rows.append({"pid": int(parts[0]), "cpu": cpu, "comm": " ".join(parts[1:-1])})
        seen = seen or line.strip().startswith("PID")
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return {"busy": rows, "summary": summary, "swapusage": swap}


def sha(path):
    return subprocess.run(["shasum", "-a", "256", path], capture_output=True,
                          text=True).stdout.split()[0]


class Sampler(threading.Thread):
    """Append-only covariates: load every 5 s, a CPU snapshot every 60 s. Writing
    from the measured process would perturb it, so this is a separate thread in the
    driver process and the pairs are attributed afterwards by file timestamps."""

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
            # sampler survives its own bugs and the analyzer can see the gaps.
            try:
                self.loads.write(json.dumps({"t": now, "load": load1()}) + "\n")
                if now - since >= 60:
                    self.snaps.write(json.dumps({"t": now, **top_busy()}) + "\n")
                    since = now
            except Exception as e:  # noqa: BLE001 - the measurement must outlive the sampler
                self.loads.write(json.dumps({"t": now, "error": repr(e)}) + "\n")
            self.stop_at.wait(5)


def wait_quiet(ceiling, tries=60, poll=30):
    for _ in range(tries):
        l = load1()
        if l < ceiling:
            return l
        time.sleep(poll)
    return None


def allpaka_process(binary, out, tag):
    """One allpaka bench process, the same env airbug would give it, for calibration."""
    rep = os.path.join(out, f"{tag}.json")
    with open(os.path.join(out, f"{tag}.txt"), "w") as fh:
        subprocess.run([binary, "bench", "--engine", MODEL], cwd=ROOT,
                       env=dict(os.environ, **BENCH_ENV, ALLPAKA_BENCH_REPORT=rep),
                       stdout=fh, stderr=subprocess.STDOUT)
    ms = {m["name"]: m for m in json.load(open(rep))["measurements"]}
    return ms["prefill"]["summary"]["median"], ms["decode"]["summary"]["median"], rep


def window(path):
    """(birth, modification) of an arm's file: airbug creates the log/json at the
    process start and closes it at the end, so the pair brackets that arm's wall time."""
    st = os.stat(path)
    return st.st_birthtime, st.st_mtime


def run_round(pinned, out, round_dir, prime=False):
    cmd = [pinned, "airbug", MODEL, "--pp", str(PP), "--tg", str(TG),
           "--repeats", "2", "--warmup", "0", "--cooldown-ms", "1500",
           "--threshold", "3.0", "--allpaka", pinned, "--out", round_dir]
    t0 = time.time()
    with open(os.path.join(out, f"round{os.path.basename(round_dir)}.txt"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **BENCH_ENV),
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    raw = os.path.join(round_dir, "raw")
    pairs = []
    for i in (1, 2):
        ap_log = os.path.join(raw, f"allpaka-{i}.log")
        ll_pp = os.path.join(raw, f"llama-pp-{i}.json")
        ll_tg = os.path.join(raw, f"llama-tg-{i}.json")
        if not all(os.path.exists(p) for p in (ap_log, ll_pp, ll_tg)):
            continue
        rec = json.load(open(os.path.join(raw, f"allpaka-{i}.json")))
        meas = {m["name"]: m for m in rec["measurements"]}
        tg_rows = json.load(open(ll_tg))
        pp_rows = json.load(open(ll_pp))
        pairs.append({
            "pair": i, "order": "AB" if i % 2 else "BA", "round": os.path.basename(round_dir),
            "prime": prime, "rc": rc, "round_start": t0, "round_end": time.time(),
            "win": {"ap": window(ap_log), "ll_pp": window(ll_pp), "ll_tg": window(ll_tg),
                    "ap_rep": window(os.path.join(raw, f"allpaka-{i}.json"))},
            "ap_pp": meas["prefill"]["summary"]["median"],
            "ap_tg": meas["decode"]["summary"]["median"],
            "ll_pp": pp_rows[0]["avg_ts"],
            "ll_tg": tg_rows[0]["avg_ts"],
            "ll_build": f"{tg_rows[0].get('build_commit')}/{tg_rows[0].get('build_number')}",
            "fast_path": meas["decode"]["fast_path"],
            "gpu_decode": meas["decode"].get("phases"),
        })
    return pairs, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=6, help="counted rounds (0 to stop after calibration)")
    ap.add_argument("--calibrate", type=int, default=4)
    ap.add_argument("--no-prime", action="store_true")
    ap.add_argument("--load-gate", type=float, default=5.0)
    args = ap.parse_args()

    src = os.path.join(ROOT, "target/release/allpaka")
    # Never measure while a build is live: an XPC GPU process contending with cc
    # moves the number. `-x`, not `-f cargo`: this tree's shell wrappers carry the
    # word cargo in their command lines and are not builds.
    busy = "\n".join(subprocess.run(["pgrep", "-xl", n], capture_output=True, text=True).stdout
                     for n in ("cargo", "rustc", "llama-bench")).strip()
    if busy:
        print("refusing to start: a build or another bench is live\n" + busy, file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    pinned = os.path.join(args.out, "allpaka")
    if not os.path.exists(pinned):
        shutil.copyfile(src, pinned)
        os.chmod(pinned, 0o755)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = len(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines())
    prov = {
        "started": time.strftime("%FT%TZ", time.gmtime()),
        "allpaka_sha256": sha(pinned), "head": head, "tree_dirty_files": dirty,
        "llama_bench": subprocess.run(["llama-bench", "--version"], capture_output=True,
                                      text=True).stdout.strip(),
        "model": MODEL, "pp": PP, "tg": TG, "env": BENCH_ENV, "load_gate": args.load_gate,
    }
    open(os.path.join(args.out, "provenance.json"), "w").write(json.dumps(prov, indent=1) + "\n")
    log = open(os.path.join(args.out, "driver.txt"), "a", buffering=1)

    def say(s):
        print(s, flush=True)
        log.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())}  {s}\n")

    say(f"# run {args.out}  binary {prov['allpaka_sha256'][:16]}  HEAD {head[:8]}  dirty {dirty}")
    say(f"  llama {prov['llama_bench']}  model {os.path.basename(MODEL)}  pp{PP}/tg{TG}")
    sampler = Sampler(args.out)
    sampler.start()
    try:
        cal = []
        for i in range(1, args.calibrate + 1):
            l = wait_quiet(args.load_gate)
            if l is None:
                say(f"  calibration {i}: load gate timed out at {load1():.2f}")
                break
            pp, tg, _ = allpaka_process(pinned, args.out, f"cal{i}")
            cal.append({"pp": pp, "tg": tg, "load": l})
            say(f"  calibration {i} at load {l:.2f}: prefill {pp:.2f}  decode {tg:.2f}")
        med = sorted(x["pp"] for x in cal)
        centre = med[len(med) // 2] if med else None
        json.dump({"calibration": cal, "prefill_median": centre, "band_floor": 0.85 * centre if centre else None},
                  open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
        say(f"  thermometer: median prefill {centre} -> A3 floor {0.85 * centre if centre else '-'}")

        pairs_path = os.path.join(args.out, "pairs.jsonl")
        if not args.no_prime:
            l = wait_quiet(args.load_gate)
            rd = os.path.join(args.out, "round-prime")
            got, rc = run_round(pinned, args.out, rd, prime=True)
            with open(pairs_path, "a") as fh:
                for p in got:
                    fh.write(json.dumps(p) + "\n")
            say(f"  priming round (discarded) at load {l}: " +
                "  ".join(f"{p['order']} ap {p['ap_tg']:.2f}/ll {p['ll_tg']:.2f} = {p['ap_tg'] / p['ll_tg']:.3f}"
                          for p in got) + f"  rc={rc}")
        for r in range(1, args.rounds + 1):
            l = wait_quiet(args.load_gate)
            if l is None:
                say(f"  round {r}: load gate timed out at {load1():.2f} - campaign pauses")
                break
            rd = os.path.join(args.out, f"round{r}")
            got, rc = run_round(pinned, args.out, rd)
            with open(pairs_path, "a") as fh:
                for p in got:
                    fh.write(json.dumps(p) + "\n")
            for p in got:
                say(f"  round {r} pair {p['pair']} {p['order']}  ap_tg {p['ap_tg']:6.2f}  "
                    f"ll_tg {p['ll_tg']:6.2f}  ratio {p['ap_tg'] / p['ll_tg']:.4f}  "
                    f"ap_pp {p['ap_pp']:6.1f}  ll_pp {p['ll_pp']:6.1f}  rc={rc}")
            if rc != 0:
                say(f"  round {r}: airbug exited {rc} - stopping, the binary pair changed under the run")
                break
    finally:
        sampler.stop_at.set()
        sampler.join(timeout=10)
    say(f"  DONE -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
