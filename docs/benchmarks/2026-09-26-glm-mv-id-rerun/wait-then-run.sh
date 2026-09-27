#!/bin/bash
# Wait for a genuinely quiet machine, then run the pre-registered MV_ID re-run campaign.
# The driver itself refuses to start while a build or another bench is live, or while
# an indexer is working, and re-checks between rounds; this adds the outer patience so
# the campaign begins in a window rather than being abandoned inside a busy one
# (preregistration.txt, changes 1 and 2).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/series-1}
export CAP_MIN=${CAP_MIN:-240}
export POLL=${POLL:-60}
export NEED=${NEED:-5}
export LOAD_MAX=${LOAD_MAX:-5.0}

python3 - "$HERE" <<'PY'
import importlib.util, os, sys, time
here = sys.argv[1]
need = int(os.environ["NEED"])
poll = int(os.environ["POLL"])
cap = int(os.environ["CAP_MIN"])
load_max = float(os.environ["LOAD_MAX"])
spec = importlib.util.spec_from_file_location("dv", os.path.join(here, "drive-mv-id.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# Printed once, so the log says which guard produced the lines below it: the first
# version of this gate vetoed on daemon liveness and could never clear, and only the
# hash distinguishes that run from the one that can launch.
import hashlib
print("driver sha256 "
      + hashlib.sha256(open(os.path.join(here, "drive-mv-id.py"), "rb").read()).hexdigest()
      + f"  (load<{load_max}, {need} clean samples {poll}s apart, cap {cap}min)",
      flush=True)
end = time.time() + 60 * cap
clean = 0
while True:
    l = m.load1()
    busy = m.others_busy()
    clean = clean + 1 if (l < load_max and not busy) else 0
    stamp = time.strftime("%H:%M:%SZ", time.gmtime())
    print(f"{stamp}  load {l:5.2f}  clean {clean}/{need}"
          + ("" if not busy else "  busy: " + " ".join(busy.split("\n"))), flush=True)
    if clean >= need:
        sys.exit(0)
    if time.time() > end:
        print("cap reached without a quiet window; the campaign did NOT run", flush=True)
        sys.exit(3)
    time.sleep(poll)
PY
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "watcher gave up (rc=$rc)"
  exit $rc
fi

echo "=== quiet window reached; launching the pre-registered campaign ==="
python3 "$HERE/drive-mv-id.py" --out "$OUT"
rc=$?
echo "=== driver exit $rc ==="
mkdir -p "$OUT"
python3 "$HERE/analyze-mv-id.py" "$OUT" > "$OUT/analysis-1.txt" 2>&1
echo "=== analysis -> $OUT/analysis-1.txt (exit $?) ==="
tail -45 "$OUT/analysis-1.txt"
