#!/usr/bin/env python3
"""Task #25 pre-flight: the 2x2 fold/layout harness, run until it is admissible.

`gpu_ffnbench.rs::q35_down_shape_priced_fused_and_plain_across_x_layouts` is the
40 s probe that collapses the pre-registration's +0.23..+0.64 ms/token band to a
point, because it prices the fold at the 35B's own down shape [2048,512] and at
the per-slot activation layout that shape actually dispatches. On 2026-09-25
22:25 UTC it answered nothing: two 13 s processes 90 s apart moved the ratio it
exists to measure from 0.845 to 0.556, while EACH process repeated itself to
1.003x internally (see harness-rejected.txt). The law that artifact buys is that
a within-run ratio is not automatically regime-robust, so a ratio claim needs
the absolute meter gate AND at least two independent processes of it.

This script enforces that and nothing else:

  gate A  the two x-shared plain meters land within 3% of published
          (q4_k [4096,1536] 468.8, q5_k [4096,1536] 344.9),
  gate B  every one of the 16 arm spreads is <= 1.03x,
  gate C  two gate-A/B-passing processes agree on all eight fold/plain ratios
          (4 shapes x 2 activation layouts) within 5%.

Full stdout of every attempt is kept, never grep-filtered, per the second
consequence in harness-rejected.txt. `--then` hands a quiet, proven window to
run-pairs.sh; without it this only says whether the band can be collapsed.
"""

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_BIN = os.path.join(
    ROOT,
    "target/release/build/allpaka-backend/402d2f60b4b17b8a/out/gpu_ffnbench-402d2f60b4b17b8a",
)
PUBLISHED = {"q4_k meter": 468.8, "q5_k meter": 344.9}
SHAPES = ["q4_k meter", "q5_k meter", "q35 down", "q35 gate/up"]
LAYOUTS = ["shared", "shared+sw", "per-slot", "per-slot+sw"]
# The pre-registered cost of the fold at the 35B's routed down: 203.5 MiB of
# q5_k at the meter's 345 GB/s. Saving = t_plain * (1/r - 1) for ratio r.
T_PLAIN_MS = 0.62

SHAPE_RE = re.compile(r"(q4_k meter|q5_k meter|q35 down|q35 gate/up)\s+\[(\d+),(\d+)\]\s+feed=(\d+)")
ARM_RE = re.compile(r"(shared\+sw|per-slot\+sw|shared|per-slot) ([0-9.]+) spread ([0-9.]+)x")
PUB_RE = re.compile(r"vs published ([0-9.]+)x")
FOLD_RE = re.compile(
    r"fold/plain shared ([0-9.]+)  per-slot ([0-9.]+)  \|  layout effect plain ([0-9.]+) fused ([0-9.]+)"
)


def load1():
    raw = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(raw.strip().strip("{}").split()[0])


def parse(text):
    """-> {shape: {"rate": {layout: (best, spread)}, "pub": float|None, "fold": {...}}}"""
    out = {}
    pending = None
    for line in text.splitlines():
        m = SHAPE_RE.search(line)
        if m:
            pending = m.group(1)
            arms = {t: (float(v), float(s)) for t, v, s in ARM_RE.findall(line)}
            pub = PUB_RE.search(line)
            out[pending] = {
                "rate": arms,
                "pub": float(pub.group(1)) if pub else None,
                "shape": [int(m.group(2)), int(m.group(3))],
            }
            continue
        f = FOLD_RE.search(line)
        if f and pending in out:
            out[pending]["fold"] = {
                "shared": float(f.group(1)),
                "per-slot": float(f.group(2)),
                "layout_plain": float(f.group(3)),
                "layout_fused": float(f.group(4)),
            }
    return out


def admit(data):
    """Returns a list of reasons the attempt is not admissible ([] = admissible)."""
    missing = [s for s in SHAPES if s not in data]
    if missing:
        return [f"no line for {' '.join(missing)}"]
    reasons = []
    worst_spread = 0.0
    for label in SHAPES:
        d = data[label]
        if len(d.get("rate", {})) < 4 or "fold" not in d:
            reasons.append(f"{label}: incomplete arms/fold")
            continue
        for tag, (_best, spread) in d["rate"].items():
            worst_spread = max(worst_spread, spread)
        if label in PUBLISHED:
            if d["pub"] is None:
                reasons.append(f"{label}: no published column printed")
            elif d["pub"] < 0.97:
                reasons.append(f"{label} meter at {d['pub']:.2f}x of published")
    if worst_spread > 1.03:
        reasons.append(f"worst arm spread {worst_spread:.3f}x > 1.03x")
    return reasons


def fold_vector(data):
    return {f"{s}/{l}": data[s]["fold"][l] for s in SHAPES for l in ("shared", "per-slot")}


def agree(a, b, tol):
    """-> (ok, worst_key, worst_ratio) comparing two admissible attempts' fold vectors."""
    fa, fb = fold_vector(a), fold_vector(b)
    worst, worst_k = 0.0, None
    for k in fa:
        r = fa[k] / fb[k]
        if abs(r - 1.0) > worst:
            worst, worst_k = abs(r - 1.0), k
    return worst <= tol, worst_k, worst


def table_line(idx, data, reasons):
    cells = []
    for s in SHAPES:
        d = data.get(s, {})
        f = d.get("fold", {})
        cells.append(f"{s.replace(' meter', ''):<11} {f.get('shared', float('nan')):.3f}/{f.get('per-slot', float('nan')):.3f}")
    cells_pub = []
    for s in PUBLISHED:
        p = data.get(s, {}).get("pub")
        cells_pub.append(f"{s.replace(' meter', '')} {p:.2f}x" if p else f"{s.replace(' meter', '')} -")
    meters = "  ".join(cells_pub)
    head = f"  attempt {idx:>2}  meters {meters}  " + " ".join(cells)
    return head + ("  REJECT: " + "; ".join(reasons) if reasons else "  admitted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=DEFAULT_BIN)
    ap.add_argument("--max-wait", type=int, default=10800, help="seconds to keep trying")
    ap.add_argument("--load-max", type=float, default=5.0, help="1-min load average ceiling to attempt at all")
    ap.add_argument("--interval", type=int, default=75, help="seconds between attempts")
    ap.add_argument("--agreement", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, ".airbug-bench", "q35-preflight-" + time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--then", action="store_true", help="run run-pairs.sh on a PASS")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.binary):
        print(f"no harness binary at {args.binary}", file=sys.stderr)
        return 2
    log = open(os.path.join(args.out, "preflight.txt"), "a")

    def emit(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    emit(f"# pre-flight for task #25  start {time.strftime('%FT%TZ', time.gmtime())}")
    emit(f"  harness binary {args.binary}")
    emit(f"  sha256 {subprocess.run(['shasum', '-a', '256', args.binary], capture_output=True, text=True).stdout.split()[0]}")
    emit(f"  mtime {time.strftime('%FT%TZ', time.gmtime(os.path.getmtime(args.binary)))}  git HEAD {subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    emit(f"  gates: meters within 3% of {PUBLISHED}, spreads <= 1.03x, two attempts agreeing to {args.agreement:.0%}")

    deadline = time.time() + args.max_wait
    attempts, admitted = [], []
    n = 0
    while time.time() < deadline:
        L = load1()
        if L > args.load_max:
            time.sleep(args.interval)
            continue
        n += 1
        path = os.path.join(args.out, f"attempt-{n}.txt")
        env = dict(os.environ, ALLPAKA_BENCH_FEEDS="1")
        t0 = time.time()
        with open(path, "w") as fh:
            fh.write(f"# attempt {n}  start {time.strftime('%FT%TZ', time.gmtime())}  1-min load {L:.2f}\n")
            fh.flush()
            subprocess.run([args.binary, "q35_down_shape", "--ignored", "--nocapture", "--test-threads=1"],
                           env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        data = parse(open(path).read())
        reasons = admit(data)
        emit(table_line(n, data, reasons))
        attempts.append((n, data, reasons))
        if not reasons:
            admitted.append((n, data))
            if len(admitted) >= 2:
                (ka, a), (kb, b) = admitted[-2], admitted[-1]
                ok, worst_k, worst = agree(a, b, args.agreement)
                fa, fb = fold_vector(a), fold_vector(b)
                emit(f"  gate C: attempts {ka} vs {kb}, worst disagreement {worst:.1%} at {worst_k}"
                     + ("  -> PASS" if ok else "  -> attempts disagree, keep going"))
                if ok:
                    r = (fa["q35 down/per-slot"] + fb["q35 down/per-slot"]) / 2
                    ctrl = (fa["q5_k meter/per-slot"] + fb["q5_k meter/per-slot"]) / 2
                    saving = T_PLAIN_MS * (1.0 / r - 1.0)
                    emit(f"  VERDICT: the band collapses to a point.")
                    emit(f"    per-slot fold ratio at the 35B's own [2048,512] down = {r:.3f}")
                    emit(f"    control (same format, meter shape [4096,1536])           = {ctrl:.3f}")
                    emit(f"    shape effect vs the borrowed 0.733: {r / 0.733 - 1:+.1%}")
                    emit(f"    predicted cost of the fold on this model = {T_PLAIN_MS:.2f} ms x (1/{r:.3f} - 1) = {saving:.3f} ms/token")
                    emit(f"    (= {saving / 9.82 * 100:.1f}% of the model's 9.82 ms/token warm wall; the band was 2.3-6.5%)")
                    emit(f"  admitted attempts: {[k for k, _ in admitted]}")
                    if args.then:
                        emit(f"  launching run-pairs.sh at {time.strftime('%FT%TZ', time.gmtime())}, load {load1():.2f}")
                        log.close()
                        os.execvpe("bash", ["bash", os.path.join(HERE, "run-pairs.sh")], os.environ)
                    return 0
        time.sleep(args.interval)

    emit(f"  NO PASS within {args.max_wait}s: {n} attempts, {len(admitted)} admitted "
         f"({[k for k, _ in admitted]}); a quiet-window pair was never obtained.")
    emit("  The band therefore stays 0.23-0.64 ms/token and the wall pairs remain the only route.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
