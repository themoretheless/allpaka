#!/usr/bin/env python3
"""Price the two attention knobs that carry no number (preregistration.txt).

Two modes, because the two questions need different instruments:

  --e2e     mv vs s32, paired, alternating, primed. Reports the ratio series.
  --stage   one short run per arm with ALLPAKA_DECODE_SPLIT=1, reading only the
            attend stage ms/token across mv/s32/s16/s8/attend4. The probe inflates
            the whole token (13-16% on the 235B), so no throughput claim comes out
            of this mode - only the stage table.

  python3 drive.py --check   # the launch guard's verdict; starts and writes nothing
  python3 drive.py --stage
  python3 drive.py --e2e --pairs 8 --control 4
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/qwen3-30b-a3b-Q4_K_M.gguf"
# The binary this design measures with, chosen before launch and pinned by sha.
# `target/release/allpaka` is what a build of the working tree leaves behind, and this
# tree is edited by another session while these runs are queued - it was mid
# `cargo test --no-run` at the time this line was written. A run whose binary is
# "whatever the tree built last" cannot be compared with the published stage tables or
# the GLM parity anchor, so the default is the sha-verified copy the MV_ID rerun pins,
# and a different binary has to be passed explicitly together with its own sha.
BIN = ROOT / "docs/benchmarks/2026-09-26-glm-mv-id-rerun/series-1/allpaka"
ARM_SHA = "7b8ec67aa72c42662ba4608fa1629c139c759c6ce7bacdd6c6aa6fbcc46fe2fe"
PP, TG = 3072, 320

ARMS = {
    "mv": {},
    "s32": {"ALLPAKA_ATTN_MV": "0"},
    "s16": {"ALLPAKA_ATTN_MV": "0", "ALLPAKA_ATTN_S8": "16"},
    "s8": {"ALLPAKA_ATTN_MV": "0", "ALLPAKA_ATTN_S8": "8"},
    "attend4": {"ALLPAKA_ATTN_MV": "0", "ALLPAKA_ATTN_S8": "4"},
}

GPU_DECODE_RE = re.compile(
    r"gpu during decode: (\d+) waits, (\d+) dispatches, encode (\d+) ms, wait (\d+) ms "
    r"of (\d+) ms total \((\d+)% outside the GPU\)"
)
# The per-label lines come from the ALLPAKA_DECODE_SPLIT print, which emits ONE
# token's stage profile per process (metal.rs:11536-11542), hence two reps/arm.
# Same instrument vetoes as the stagger driver, added after three identical-code
# GLM control pairs returned ratios of 2.03, 1.12 and 0.91 at load 37.5, 18.7 and
# 8.5 (docs/benchmarks/2026-09-25-glm-shared-stagger/contended-control-pairs.jsonl).
LOAD_VETO = 8.0
DRIFT_VETO = 0.85
LAUNCH_LOAD = 5.0
# The launch gate's name classes and its settle, all decided before this design
# next runs: see the "Harness amendment" block at the bottom of preregistration.txt.
BUILD_NAMES = ("cargo", "rustc", "llama-bench")
# Resident daemons, spelled as `pgrep -x` sees them. The 2026-09-26 MV_ID campaign
# gated this class on liveness first, which is unsatisfiable - these are never
# absent - and `-x mdworker` matched nothing at all because the worker is
# `mdworker_shared`.
INDEX_NAMES = ("corespotlightd", "managedcorespotlightd", "mds", "mds_stores",
               "mdworker_shared", "photoanalysisd", "filecoordinationd")
# Percent of one core, from a cputime delta over INDEX_SAMPLE_S, not from `ps` %cpu
# (a lifetime average that cannot show a burst in a three-day-old daemon).
INDEX_SAMPLE_S = 3.0
INDEX_PROC_VETO = 25.0
INDEX_CLASS_VETO = 40.0
SETTLE = 3
LAUNCH_WAIT_S = 45 * 60

ATTEND_RE = re.compile(r"^\s+attend\s+([\d.]+)\s+\(\s*([\d.]+)%", re.M)
# The phases line is telemetry-aggregated and folds attend|wo into `attention`.
PHASE_ATTENTION_RE = re.compile(r"^\s+attention\s+([\d.]+)\s+\(\s*([\d.]+)%", re.M)
PREWARM_RE = re.compile(r"prewarmed ([\d.]+) GiB of weights in ([\d.]+)s")


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load1():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True,
                         text=True).stdout
    return float(out.split()[1])


def build_busy():
    """
    A live compiler or another bench, as a text block ('' when clear). `-x` and not
    `-f cargo`: this tree's shell wrappers carry the word cargo in their command
    lines and are not builds. A builder is vetoed by existence - a live `rustc` is
    by definition working - which is NOT true of the indexer class below.
    """
    return "\n".join(l for nm in BUILD_NAMES for l in subprocess.run(
        ["pgrep", "-xl", nm], capture_output=True, text=True).stdout.splitlines()
                      if l).strip()


def bench_busy():
    """
    Another campaign's bench process, as a text block ('' when clear). This driver
    copies its binary into the run dir and names it `allpaka`, which is what the
    MV_ID campaign does too, so a live `allpaka` that is not this driver's own arm
    means two GPU campaigns sharing the machine - the one contention this gate can
    not interpret afterwards. The driver only ever awaits one arm at a time, so a
    hit here is never its own work.
    """
    return "\n".join(subprocess.run(["pgrep", "-x", "allpaka"], capture_output=True,
                                     text=True).stdout.split())


def _cputimes(pids):
    """Cumulative CPU seconds per pid (`ps -o cputime`, hh:mm:ss.frac)."""
    if not pids:
        return {}
    out = subprocess.run(["ps", "-o", "pid=,cputime=", "-p", ",".join(pids)],
                         capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            secs = 0.0
            for chunk in parts[1].split(":"):
                try:
                    secs = secs * 60 + float(chunk)
                except ValueError:
                    secs = -1.0
                    break
            if secs >= 0:
                vals[parts[0]] = secs
    return vals


def index_busy():
    """
    An indexer working *this instant*, as a text block ('' when idle). These own a
    core and the disk for minutes at a time while load average still reads calm.

    Judged by CPU work, not by liveness: every name here is a resident daemon that
    is never absent, so vetoing on `pgrep` finding them asks for a machine state
    that does not exist and the campaign would wait out its whole cap. And the
    measurement is a `cputime` delta over `INDEX_SAMPLE_S`, not `ps` %cpu, because
    that column divides by the process's entire life - a three-day-old daemon shows
    single digits while indexing at 153%, which is the burst this class is gated for.
    """
    named = {}
    for nm in INDEX_NAMES:
        for p in subprocess.run(["pgrep", "-x", nm], capture_output=True,
                                text=True).stdout.split():
            if p.isdigit():
                named.setdefault(p, nm)
    pids = list(named)
    before = _cputimes(pids)
    time.sleep(INDEX_SAMPLE_S)
    after = _cputimes(pids)
    rows, total = [], 0.0
    for p in pids:
        pct = 100.0 * (after.get(p, 0.0) - before.get(p, 0.0)) / INDEX_SAMPLE_S
        total += pct
        if pct >= INDEX_PROC_VETO:
            rows.append(f"{named[p]}({p}) {pct:.0f}% cpu")
    if total >= INDEX_CLASS_VETO and len(pids) > 1:
        rows.append(f"indexer fleet {total:.0f}% cpu across {len(pids)} pids")
    return "\n".join(rows).strip()


def others_busy():
    return "\n".join(x for x in (build_busy(), bench_busy(), index_busy()) if x).strip()


def wait_quiet(limit_s=40 * 60, step_s=30.0, settle=SETTLE):
    """Spend wall clock, not data: hold until the machine is quiet before an arm.

    Added after the first two launches of this design were started at load 4.5
    and ran their arms at 8-37.5 (see the amendment at the bottom of
    preregistration.txt). A pair whose arms were both quiet is worth more than
    three pairs sampled through a burst, and the run has no deadline.

    `settle` consecutive clean samples, not the first one: the GLM MV_ID campaign
    launched one tick after a burst on the strength of a single reading and had to
    veto the pairs that followed. Load average alone is not the gate either - it
    lags a compiler and is blind to an indexer - hence `others_busy`.
    """
    waited, clean = 0.0, 0
    while waited < limit_s:
        l = load1()
        busy = others_busy()
        if l <= LAUNCH_LOAD and not busy:
            clean += 1
            if clean >= settle:
                if waited:
                    print(f"  quiet gate: held {waited:.0f} s, load now {l}", flush=True)
                return waited
        else:
            clean = 0
            print(f"  quiet gate: load {l:.2f}"
                  + ("" if not busy else "  busy: " + busy.replace("\n", " ")),
                  flush=True)
        time.sleep(step_s)
        waited += step_s
    raise SystemExit(f"quiet gate: no window of {settle} clean samples in "
                     f"{limit_s / 60:.0f} min; refusing to measure into a busy machine")


def run_arm(arm, tag, out_dir, split=False):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ALLPAKA_ATTN_MV", "ALLPAKA_ATTN_S8", "ALLPAKA_SHARED_STAGGER")}
    env.update({"ALLPAKA_BENCH_PP": str(PP), "ALLPAKA_BENCH_TG": str(TG),
                "ALLPAKA_BENCH_SKIP_MTP": "1", "ALLPAKA_PROFILE": "max-performance",
                "ALLPAKA_BENCH_REPORT": str(out_dir / f"{arm}-{tag}-report.json")})
    if split:
        env["ALLPAKA_DECODE_SPLIT"] = "1"
    env.update(ARMS[arm])
    log = out_dir / f"{arm}-{tag}-log.txt"
    # Recorded before the process starts: a burst that begins during an arm
    # shows up in `load` too late to attribute, which is why the veto reads the
    # max of all four readings of a pair.
    load_before = load1()
    with log.open("w") as handle:
        proc = subprocess.Popen([str(out_dir / "allpaka"), "bench", "--engine", str(MODEL)],
                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                text=True)
        rc = proc.wait()
    text = log.read_text()
    if rc != 0:
        raise SystemExit(f"{arm} arm exited {rc}; see {log}")
    data = json.loads((out_dir / f"{arm}-{tag}-report.json").read_text())
    dec = next(m for m in data["measurements"] if m["name"] == "decode")
    gpu, att, pre = GPU_DECODE_RE.search(text), ATTEND_RE.findall(text), PREWARM_RE.search(text)
    ph = PHASE_ATTENTION_RE.findall(text)
    rec = {
        "arm": arm, "at": utc(), "load_before": load_before, "load": load1(),
        "split": split,
        "decode_tok_s": dec["summary"]["median"],
        "decode_tokens": dec.get("input_tokens"),
        "declines": dec.get("fast_path", {}).get("declines"),
        "attend_ms_token": float(att[0][0]) if att else None,
        "attend_pct": float(att[0][1]) if att else None,
        "attention_phase_ms_token": float(ph[-1][0]) if ph else None,
        "dispatches": int(gpu.group(2)) if gpu else None,
        "outside_gpu_pct": int(gpu.group(6)) if gpu else None,
        "prewarm_s": float(pre.group(2)) if pre else None,
        "fingerprint": data["metadata"]["model_fingerprint"],
        "git_commit": data["metadata"]["git_commit"],
    }
    (out_dir / f"{arm}-{tag}-rec.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"  {arm:8} {tag:8} {'split' if split else 'e2e  ':5} "
          f"{rec['decode_tok_s']:.4f} tok/s  attend {rec['attend_ms_token']} ms/token "
          f"({rec['attend_pct']}%)  load {rec['load']}  prewarm {rec['prewarm_s']}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--e2e", action="store_true")
    g.add_argument("--stage", action="store_true")
    g.add_argument("--check", action="store_true",
                   help="print the launch guard's verdict and exit; starts nothing")
    ap.add_argument("--binary", default=str(BIN),
                    help="binary to pin into the run dir; default is the sha-pinned "
                         "campaign copy, not a build of the working tree")
    ap.add_argument("--expect-sha", default=ARM_SHA,
                    help="sha256 the chosen binary must hash to; '' disables the check")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--control", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    if args.check:
        # Added after this session's own probe: a driver whose only way to inspect the
        # gate is to run it will be run. Nothing here touches the GPU or the filesystem.
        got = hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest() \
            if pathlib.Path(args.binary).is_file() else "MISSING"
        busy = others_busy()
        print(f"binary {args.binary}\n  sha256 {got}\n  expects {args.expect_sha or '(unchecked)'}"
              f" -> {'ok' if got == args.expect_sha else 'MISMATCH'}\n"
              f"load {load1():.2f} against ceiling {LAUNCH_LOAD}; "
              f"settle {SETTLE} samples {30}s apart; cap 40 min\n"
              f"  builders:      {build_busy() or 'none'}\n"
              f"  other bench:   {bench_busy() or 'none'}\n"
              f"  indexer work:  {index_busy() or 'none (class is resident and idle)'}")
        return 0

    # Fail closed before a run dir exists: a refused launch leaves nothing behind to
    # mistake for a campaign. The re-checks happen per arm inside wait_quiet.
    busy = others_busy()
    if busy:
        raise SystemExit("refusing to start: a build or another bench is live, or an "
                         "indexer is working\n" + busy)

    out_dir = args.out or ROOT / ".airbug-bench" / (
        ("attn-stage-" if args.stage else "attn-e2e-")
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    src_bin = pathlib.Path(args.binary)
    got = hashlib.sha256(src_bin.read_bytes()).hexdigest()
    if args.expect_sha and got != args.expect_sha:
        raise SystemExit(f"refusing to measure with {src_bin}: sha256 {got} is not the "
                         f"preregistered {args.expect_sha}")
    pinned = out_dir / "allpaka"
    subprocess.run(["cp", str(src_bin), str(pinned)], check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    (out_dir / "provenance.json").write_text(json.dumps({
        "started": utc(), "mode": "stage" if args.stage else "e2e",
        "binary_sha256": got, "binary_source": str(src_bin), "preregistered_sha": args.expect_sha or None,
        "git_head": head, "tree_clean": subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True).stdout.strip() == "",
        "model": str(MODEL.relative_to(ROOT)), "pp_tg": [PP, TG], "arms": ARMS,
        "pairs": args.pairs, "control": args.control,
    }, indent=1) + "\n")

    wait_quiet()
    prime = run_arm("mv", "prime", out_dir, split=args.stage)
    if args.stage:
        # The amendment's corollary: a stage row has no partner to veto against, so
        # the launch gate is the only defense this table has against a burst.
        for arm in ("mv", "s32", "s16", "s8", "attend4"):
            for rep in (1, 2):
                wait_quiet()
                time.sleep(20.0)
                run_arm(arm, f"stage{rep}", out_dir, split=True)
        return 0

    wait_quiet()

    # One pool for the whole process, so a burst that carries from the control
    # series into the real pairs is still visible to the drift veto.
    seen_rates = [prime["decode_tok_s"]]
    for kind, n, challenge in (("control", args.control, "mv"), ("real", args.pairs, "s32")):
        with (out_dir / f"{kind}-pairs.jsonl").open("a") as sink:
            for pair in range(1, n + 1):
                order = "AB" if pair % 2 else "BA"
                seq = [challenge, "mv"] if order == "AB" else ["mv", challenge]
                row = {"kind": kind, "pair": pair, "order": order}
                for slot, arm in zip(("first", "second"), seq):
                    wait_quiet()
                    time.sleep(20.0)
                    rec = run_arm(arm, f"{kind}{pair}{slot}", out_dir)
                    if rec["declines"]:
                        raise SystemExit(f"criterion 1: {arm} reported {rec['declines']} declines")
                    row[slot] = {"arm": arm, **rec}
                loads = [x for k in ("first", "second")
                         for x in (row[k]["load_before"], row[k]["load"])]
                if max(loads) > LOAD_VETO:
                    row["load_vetoed"] = True
                    print(f"  {kind} pair {pair}: load veto (max {max(loads)} > {LOAD_VETO}) "
                          "- pair dropped", flush=True)
                slow = min(row["first"]["decode_tok_s"], row["second"]["decode_tok_s"])
                ref = statistics.median(seen_rates)
                row["drift_reference"] = ref
                if ref and slow < DRIFT_VETO * ref:
                    row["drift_vetoed"] = True
                    print(f"  {kind} pair {pair}: drift veto (slowest arm {slow:.2f} under "
                          f"{DRIFT_VETO:.0%} of running median {ref:.2f}) - pair dropped",
                          flush=True)
                # Appended after the check, so a pair is never compared against
                # a median that already contains its own two arms.
                seen_rates += [row["first"]["decode_tok_s"], row["second"]["decode_tok_s"]]
                if row["first"]["decode_tokens"] != row["second"]["decode_tokens"]:
                    print(f"  {kind} pair {pair}: greedy continuations differ - not a clean "
                          "ablation", flush=True)
                    row["tokens_differ"] = True
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    print("pairs:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
