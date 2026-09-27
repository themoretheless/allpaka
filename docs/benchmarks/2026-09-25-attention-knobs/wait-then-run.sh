#!/bin/bash
# Chain the attention-knob campaign behind the MV_ID rerun, then take it in a window.
#
# Two separate waits, both deliberate (preregistration.txt, harness amendment):
#   1. another GPU campaign must finish first - running the attention design while the
#      MV_ID processes are live is exactly the co-tenant contention that has cost this
#      repo two campaigns, and no post-hoc veto can unbias a pair taken that way;
#   2. then the driver's own gate must see consecutive clean samples - low load average,
#      no live build, no other bench, no indexer doing work.
# Nothing is measured until both say so; if either cap is reached, this exits non-zero
# and the campaign simply does not exist yet.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export STAGE_OUT=${STAGE_OUT:-$HERE/series-1/stage}
export E2E_OUT=${E2E_OUT:-$HERE/series-1/e2e}
export CAP_MIN=${CAP_MIN:-600}
export POLL=${POLL:-60}
export NEED=${NEED:-5}
export LOAD_MAX=${LOAD_MAX:-5.0}
export OTHER=${OTHER:-drive-mv-id.py}

python3 - "$HERE" <<'PY'
import hashlib
import importlib.util
import os
import subprocess
import sys
import time

here = sys.argv[1]
need, poll, cap = int(os.environ["NEED"]), int(os.environ["POLL"]), int(os.environ["CAP_MIN"])
load_max = float(os.environ["LOAD_MAX"])
other = os.environ["OTHER"]
spec = importlib.util.spec_from_file_location("dv", os.path.join(here, "drive.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("driver sha256 "
      + hashlib.sha256(open(os.path.join(here, "drive.py"), "rb").read()).hexdigest()
      + f"  (wait for {other!r} to exit, then load<{load_max} x{need} clean samples "
        f"{poll}s apart, cap {cap}min)", flush=True)


def live(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True,
                          text=True).stdout.split()


end = time.time() + 60 * cap
clean = 0
while True:
    rivals = live(other) + m.bench_busy().split()
    if rivals:
        clean = 0
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())}  another campaign is live "
              f"(pids {' '.join(rivals[:6])}) - not measuring yet", flush=True)
    else:
        l = m.load1()
        busy = m.others_busy()
        clean = clean + 1 if (l < load_max and not busy) else 0
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        print(f"{stamp}  load {l:5.2f}  clean {clean}/{need}"
              + ("" if not busy else "  busy: " + busy.replace("\n", " ")), flush=True)
        if clean >= need:
            sys.exit(0)
    if time.time() > end:
        print("cap reached; the campaign did NOT run", flush=True)
        sys.exit(3)
    time.sleep(poll)
PY
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "watcher gave up (rc=$rc)"
  exit $rc
fi

echo "=== window reached; stage table first, then the e2E pairs ==="
mkdir -p "$STAGE_OUT" "$E2E_OUT"
python3 "$HERE/drive.py" --stage --out "$STAGE_OUT"
src=$?
echo "=== stage exit $src ==="
[[ $src -ne 0 ]] && exit $src
python3 "$HERE/analyze.py" "$STAGE_OUT" > "$STAGE_OUT/analysis-1.txt" 2>&1
tail -30 "$STAGE_OUT/analysis-1.txt"

python3 "$HERE/drive.py" --e2e --pairs 8 --control 4 --out "$E2E_OUT"
rc2=$?
echo "=== e2e exit $rc2 ==="
[[ $rc2 -ne 0 ]] && exit $rc2
python3 "$HERE/analyze.py" "$E2E_OUT" > "$E2E_OUT/analysis-1.txt" 2>&1
echo "=== analysis -> $E2E_OUT/analysis-1.txt (exit $?) ==="
tail -45 "$E2E_OUT/analysis-1.txt"
