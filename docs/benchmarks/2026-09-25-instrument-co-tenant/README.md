# The load average is the wrong gate: a GPU co-tenant moves decode 2.8x at constant load

Observed 2026-09-25T02:37-02:56Z on the M4 Max while running
`../2026-09-25-gpu-window-cost/drive.py`. This is a result about the instrument,
not about allpaka, and it is the reason that series' timing half is on hold.

## The event

Three control-arm processes of the window A/B - arms that differ only by an
environment variable the code treats identically (`ALLPAKA_GPU_WINDOW_GIB=32`
against unset, both computing a 32 GiB window and reporting `windows=1`) - ran
**2.8-3.1x slower** than their pair partners. Exact values, one bench process
each, same model (`Qwen3.6-35B-A3B-UD-Q4_K_M`), same `PP=64 TG=400`:

| pair | order | arm | decode tok/s | dispatches | encode ms | wait ms | total ms | load |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | null | 95.1197 | 304800 | 119 | 4062 | 4205 | 7.36 |
| 1 | AB | wide | 93.4279 | 304800 | 174 | 4069 | 4281 | 6.90 |
| 2 | BA | wide | 93.2975 | 304800 | 153 | 4101 | 4287 | 5.74 |
| 2 | BA | null | 33.9938 | 304800 | 223 | 11490 | 11767 | 5.80 |
| 3 | AB | null | 31.2835 | 304800 | 241 | 12491 | 12786 | 7.61 |
| 3 | AB | wide | 34.0516 | 304800 | 177 | 11531 | 11747 | 10.69 |

What the numbers already rule out, before any external diagnosis:

* **Not the workload.** The dispatch count is identical to the unit in all six
  processes (304800 for 400 tokens, 762/token), so the GPU was asked to do the
  same thing every time.
* **Not the CPU side.** `encode ms` - host time to encode the whole pass - is
  119-241 ms across fast and slow arms alike. It did not grow.
* **Not the clock.** Every process reported the same prefill rate, 512.7-516.2
  tok/s, including the 3x arms. A frequency or thermal limit would have moved
  the compute-bound prefill first; it stayed flat to 0.7%.
* **Not the load.** The fastest arm of the series ran at load 7.36 and a slow one
  at 5.80. `vm.loadavg` does not separate the two populations, and pair 3 shows
  why ordering cannot rescue it: both of its arms were slow, so the pair ratio is
  0.9191 - plausible in sign and wrong by 8% in level, because the contention
  entered between the two processes, not inside one.
* **It is the byte pipe.** All the extra time is in `wait ms` (4062-4101 against
  11490-12491) and inside the GPU's own accounting: `gpu clock during decode`
  reported "executing 4034 ms" for a fast arm against "executing 12389 ms" for a
  slow one, with scheduling (31 against 43 ms) and round trips (36 against 59 ms)
  unchanged. The device spent three times as long executing the same kernels -
  which for a bandwidth-bound matvec path means the bytes arrived three times as
  slowly.

## The co-tenant, seen

A five-probe sweep of the harness's own streamed-byte meters
([`meter-probe.py`](meter-probe.py), kept in this directory; its log archived here as
`meter-probe.txt`) taken at 02:54-02:56Z, after the series was stopped. The `load` column of
this block is the 5-minute average rather than the 1-minute one the table
above uses - see the dated amendment at the end:

```
02:54:39Z  load  6.87  q4_k meter 348.2 GB/s (0.74x)  q5_k meter 265.3 GB/s (0.77x)  | top: 58.9 remoting_me2me_host / 36.0 WindowServer / 35.2 Qoder Helper
02:55:06Z  load  6.65  q4_k meter 337.8 GB/s (0.72x)  q5_k meter 268.6 GB/s (0.78x)  | top: 88.1 remoting_me2me_host / 44.6 WindowServer / 35.2 Qoder Helper
02:55:31Z  load  6.61  q4_k meter 340.7 GB/s (0.73x)  q5_k meter 266.7 GB/s (0.77x)  | top: 64.0 remoting_me2me_host / 36.1 WindowServer / 33.3 Qoder Helper
02:56:04Z  load  6.56  q4_k meter 342.5 GB/s (0.73x)  q5_k meter 262.3 GB/s (0.76x)  | top: 42.9 remoting_me2me_host / 40.3 WindowServer / 33.1 Kimi Helper (Renderer)
02:56:28Z  load  6.80  q4_k meter 126.0 GB/s (0.27x)  q5_k meter 263.1 GB/s (0.76x)  | top: 100.0 fvid-381aa28fb499f77e / 59.0 remoting_me2me_host / 41.2 WindowServer
```

The machine has a live screen-sharing session (`remoting_me2me_host`, 43-88% CPU
sample to sample, with `WindowServer` at 36-45%) that stays at load average
6.5-6.9 all the while. The last line is the interesting one: at the moment a
GPU-side helper (`fvid-...`) took a core, the q4_k meter fell from 342 to
126 GB/s - 0.27x of published, the same magnitude and the same signature as the
slow decode arms above, while the 5-minute load average did not move -
which is what a 5-minute average does to a burst that clears in a minute.

What this establishes: a GPU-attached co-tenant exists on this machine, the
streamed byte rate is roughly halved by its steady state and cut to a quarter by
its bursts, and the load average is blind to it.

What it does not establish: that the co-tenant is the *whole* explanation for the
0.74-0.76x meter readings overnight (it was present for those too, which is a
correlation between 20 rejected pre-flight probes and a remote-desktop session,
not a measured cause), or anything at all about which allpaka kernel is faster.

## Consequence for the harness, and what to do with a gate

The admission rules in `../2026-09-25-q35a3b-postflip/preflight.py` and the
earlier series use `vm.loadavg` as
the state variable. This data says that variable is not measuring the thing that
matters. The bench already prints a better one, for free, inside the process
being gated: for a fixed model and config the counter-derived
`total ms of the decode pass` (and equivalently `wait ms`) is a direct read of
how fast bytes arrived during that pass - 4205-4287 ms against 11747-12786 ms
here, a separation of ~2.9x with no overlap between the populations.

So the rule adopted for the series that need the byte pipe:

1. Gate on the arm's own decode total-ms, not on load: an arm is admissible when
   its ms/token is within 10% of the fastest arm seen so far in the series.
2. Gate the *pair*, not only the ratio - a pair whose two arms sit in different
   populations is discarded even though its ratio looks well-behaved (pair 3
   above).
3. Record the load average and the top CPU tenants anyway, as here, because a
   gate that cannot name its own failure mode is not debuggable.

The cost of rules 1-2 on a machine with a live desktop session is that many pairs
get thrown away, and the honest alternative - relaxing the band to accept them -
would corrupt exactly the absolute-delta claims (`+0.23...0.64 ms/token` for the
SWFUSE fold in `../2026-09-25-q35a3b-postflip/preregistration.txt`) that those
gates exist to protect. Pending series that depend on this: task #25 (35B
post-flip wall A/B), task #16 (GLM parity re-measure), the timing half of the
window A/B, and the GLM stagger A/B in
`../2026-09-25-glm-shared-stagger/preregistration.txt`.

## Reproduce

```sh
# the three-population control arms:
python3 ../2026-09-25-gpu-window-cost/drive.py --control 3 --real 0 \
  --out ./repro-window-cost
# the meter sweep against live tenants:
python3 meter-probe.py --iterations 5 --gap 8
```

The first command needs the 35B and writes per-arm logs plus its control-pairs
JSONL into the out directory; the second needs the pre-built harness meter at
`target/release/build/allpaka-backend/*/out/gpu_ffnbench-*` (build it with
`cargo test -p allpaka-backend --release --test gpu_ffnbench --no-run`, and not
while a GPU measurement is running).

---

## Amendment 2026-09-25T23:18Z - the sweep's `load` column was the 5-minute average

Found while auditing the other drivers for dead instruments, not here.
`meter-probe.py` stripped the braces from `sysctl -n vm.loadavg` and then took
`split()[1]`; once the braces are gone that index is the 5-minute average, since
the line is `{ 1-min 5-min 15-min }`. Verified against a live reading
(`{ 3.51 3.07 3.82 }`): the raw form needs index 1, the brace-stripped form index
0. The probe now reads index 0 and says so in its docstring. The other seven
`vm.loadavg` call sites in `docs/benchmarks/` were checked and all take the
1-minute value (`gpu-window-cost`, `glm-shared-stagger`, `dcomb-ab`,
`attention-knobs`,
`mtp-rows-ablation` split the raw line at index 1; `../2026-09-25-q35a3b-postflip/preflight.py`
and `../2026-09-24-glm-parity-remeasure/drive.py` strip and take index 0), so only
this sweep is affected.

What the correction takes away, and what it leaves:

- The headline stands on the six-process table above, whose `load` column comes
  from the window driver and is 1-minute: the fastest arm of the series ran at
  7.36 while a slow one ran at 5.80. That is the load average failing to separate
  two populations, and it does not depend on the sweep.
- The sweep's flat 6.5-6.9 no longer shows that a *burst* is invisible to the
  load average. A 5-minute average is smooth by construction, so that inference
  is withdrawn; the same burst is caught by a column the probe already records,
  `top: 100.0 fvid-...` in the very sample where the q4_k meter fell to 0.27x.
- The adopted rule is unchanged, and the lag is an extra argument for it: gating
  on a stale aggregate cannot catch a burst that starts and clears inside one
  90-second arm, so the series that need the byte pipe gate on the arm's own
  decode ms and record load only to name a failure after the fact.
