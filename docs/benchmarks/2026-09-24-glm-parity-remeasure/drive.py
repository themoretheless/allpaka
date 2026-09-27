#!/usr/bin/env python3
"""Task #16 driver: paired allpaka-vs-llama decode on GLM-4.5-Air, tg32.

Replaces the last arithmetic on the status board. The 0.86x row in
../allpaka-vs-llama-metal-status.md is the 2026-09-12 cool matrix (allpaka tg32
39.0 vs llama tg32 45.46, llama build 5266f24da/10809), which predates both the
q5_0_mv port (+16.05% decode, 15/15 pairs, SEM 1.59,
../2026-09-24-q5-0-mv-e2e/paired-32.txt) and the SWFUSE default-off flip. The
current guess from those two numbers is ~parity, and a guess is not a row.

Protocol, fixed in preregistration.txt beside this file:
  - one fresh process per arm, AB/BA alternating by pair so position and thermal
    drift are common-mode rather than a constant bias on one engine;
  - allpaka reads BOTH of its numbers from its own JSON report at full float
    (`measurements[].summary.median`, tok/s) and llama from `avg_ts` in
    `llama-bench -o json`; no number in the analysis path is ever scraped off a
    stdout print, which is invariant 6 of docs/roadmap.md;
  - the llama build commit/number is recorded per run, because the row being
    replaced was measured on build 10809 and the installed one has moved;
  - the 1-minute load average below `--load-max` (default 5.0) is a permission to
    attempt, not an admission criterion: criterion 7 of preregistration.txt, the
    allpaka prefill thermometer, decides whether a pair counts, because the warmup
    and pair 2 of 2026-09-24 passed that gate at 4.72 and 4.93 and still read
    allpaka 16.00/14.20 tok/s against a post-port anchor of ~46.
  - the load average seen at each arm start IS recorded per pair, because
    criterion 5 forbids an "allpaka is faster" claim from a pair whose gate load
    exceeded 5 - llama.cpp puts part of its graph on 12 CPU threads and allpaka's
    decode is a GPU byte-pipe, so background load flatters the ratio.

The binary is copied into the run directory before the first pair, because this is
a shared worktree and another session can rebuild target/release/allpaka mid-run.

    python3 drive.py                      # 20 pairs into a fresh run dir
    PAIRS=12 python3 drive.py --out DIR   # or resume a chosen directory
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(ROOT, "models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf")
PP, TG = 480, 32


def load1():
    raw = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(raw.strip().strip("{}").split()[0])


def wait_quiet(ceiling, tries=40, poll=30):
    for _ in range(tries):
        l = load1()
        if l < ceiling:
            return l
        time.sleep(poll)
    return None


def run_allpaka(out, binary, tag):
    rep = os.path.join(out, f"{tag}.json")
    env = dict(os.environ, ALLPAKA_BENCH_PP=str(PP), ALLPAKA_BENCH_TG=str(TG),
               ALLPAKA_BENCH_SKIP_MTP="1", ALLPAKA_PROFILE="max-performance",
               ALLPAKA_BENCH_REPORT=rep)
    with open(os.path.join(out, f"{tag}.txt"), "w") as fh:
        subprocess.run([binary, "bench", "--engine", MODEL], cwd=ROOT, env=env,
                       stdout=fh, stderr=subprocess.STDOUT)
    ms = {m["name"]: m["summary"]["median"] for m in json.load(open(rep))["measurements"]}
    return ms["prefill"], ms["decode"]


def run_llama(out, tag, phase):
    """One llama-bench process per phase: its JSON reports one test per row."""
    args = ["-p", str(PP), "-n", "0", "-d", "0"] if phase == "prefill" else ["-p", "0", "-n", str(TG), "-d", "1"]
    j = os.path.join(out, f"{tag}.{phase}.json")
    with open(j, "w") as fh:
        subprocess.run(["llama-bench", "-m", MODEL] + args +
                       ["-r", "1", "-ngl", "99", "-ctk", "f16", "-ctv", "f16", "-o", "json"],
                       cwd=ROOT, stdout=fh,
                       stderr=open(os.path.join(out, f"{tag}.{phase}.txt"), "w"))
    rec = json.load(open(j))[0]
    return rec["avg_ts"], f"{rec.get('build_commit')}/{rec.get('build_number')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=int(os.environ.get("PAIRS", "20")))
    ap.add_argument("--warm", type=int, default=int(os.environ.get("WARM", "1")), help="pairs discarded before analysis")
    ap.add_argument("--load-max", type=float, default=5.0)
    ap.add_argument("--cooldown", type=float, default=60.0, help="criterion 8: 60 s between processes in a pair")
    ap.add_argument("--out", default=os.path.join(ROOT, ".airbug-bench", "glm-parity-" + time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pinned = os.path.join(args.out, "allpaka")
    if not os.path.exists(pinned):
        subprocess.run(["cp", os.path.join(ROOT, "target/release/allpaka"), pinned], check=True)
    log = open(os.path.join(args.out, "driver.log"), "w", buffering=1)

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    say(f"# glm-parity run {args.out}  start {time.strftime('%FT%TZ', time.gmtime())}")
    say(f"  pinned binary sha256 {subprocess.run(['shasum', '-a', '256', pinned], capture_output=True, text=True).stdout.split()[0]}")
    say(f"  git HEAD {subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    say(f"  model {MODEL}  pp={PP} tg={TG}  pairs={args.pairs} warm={args.warm} load-max={args.load_max}")

    build = None
    for i in range(1, args.pairs + args.warm + 1):
        order = "AB" if i % 2 else "BA"
        vals, loadmax, loadpre, ok = {}, 0.0, 0.0, True
        for arm in (["ap", "ll"] if order == "AB" else ["ll", "ap"]):
            l = wait_quiet(args.load_max)
            if l is None:
                say(f"  pair {i:2d} {order}: load gate timed out at {load1():.2f} before the {arm} arm - pair dropped")
                ok = False
                break
            loadmax = loadpre = max(loadmax, l)
            tag = f"p{i:02d}_{order}_{arm}"
            if arm == "ap":
                vals["ap_pp"], vals["ap_tg"] = run_allpaka(args.out, pinned, tag)
            else:
                vals["ll_pp"], b1 = run_llama(args.out, tag, "prefill")
                vals["ll_tg"], build = run_llama(args.out, tag, "decode")
                if b1 != build:
                    say(f"  pair {i:2d}: llama build moved mid-run ({b1} vs {build})")
            # A pre-arm reading is bounded by the gate it just passed, so a
            # `loadmax` built only from them cannot reach `load_max` and
            # criterion 5's hot count is empty by construction; the arm's own
            # burst is visible only after it. See the dated amendment in
            # preregistration.txt.
            loadmax = max(loadmax, load1())
            time.sleep(args.cooldown)
        if not ok or len(vals) < 4:
            continue
        say(f"  pair {i:2d} {order} {'warm' if i <= args.warm else '    '} "
            f"ap_tg {vals['ap_tg']:6.2f}  ll_tg {vals['ll_tg']:6.2f}  ratio {vals['ap_tg'] / vals['ll_tg']:.4f}  "
            f"ap_pp {vals['ap_pp']:6.1f}  ll_pp {vals['ll_pp']:6.1f}  loadmax {loadmax:4.2f}")
        with open(os.path.join(args.out, "pairs.jsonl"), "a") as fh:
            fh.write(json.dumps({"pair": i, "order": order, "loadmax": loadmax,
                                 "loadpre_max": loadpre, **vals}) + "\n")
    say(f"  llama build used: {build}")
    say(f"  DONE -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
