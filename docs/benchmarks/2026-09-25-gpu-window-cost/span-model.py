#!/usr/bin/env python3
"""Check the window-cap cliff against the arithmetic in metal.rs `add_mapping`.

Reproduces the source's loop (WINDOW_CAP = 32 GiB, CHUNK_OVERLAP = 2 GiB,
step = max_buf - CHUNK_OVERLAP when max_buf > overlap else max_buf / 2), prints
the window count and total mapped span for every arm measured in
2026-09-25-gpu-window-cost, and then asks which of the candidate variables
separates correct from broken arms. Run it after editing the source's window
code: a row whose `pred_windows` disagrees with the census in the matching
`raw/*-log.txt` means the model here no longer matches the code.
"""

GIB = 1 << 30
WINDOW_CAP = 32 * GIB
CHUNK_OVERLAP = 2 * GIB

# name -> (shard sizes in bytes, cap in GiB or None for the default, observed
# window census, verdict: True = logits correct)
ARMS = [
    ("235B cap32", [88767695360], None, 3, True),
    ("235B cap24", [88767695360], 24, 4, True),
    ("235B cap16", [88767695360], 16, 6, True),
    ("235B cap12", [88767695360], 12, 9, True),
    ("235B cap10", [88767695360], 10, 11, True),
    ("235B cap9", [88767695360], 9, 12, True),
    ("235B cap8", [88767695360], 8, 14, False),
    ("235B cap6", [88767695360], 6, 21, False),
    ("235B cap4", [88767695360], 4, 41, False),
    ("GLM cap32", [50000746752, 22975001632], None, 3, True),
    ("GLM cap8", [50000746752, 22975001632], 8, 12, True),
    ("GLM cap6", [50000746752, 22975001632], 6, 17, True),
    ("GLM cap5", [50000746752, 22975001632], 5, 22, False),
    ("GLM cap4", [50000746752, 22975001632], 4, 33, False),
    ("GLM cap3", [50000746752, 22975001632], 3, 65, False),
    ("35B cap2", [22134528992], 2, 20, True),
    ("35B cap32", [22134528992], None, 1, True),
]


def mapping(size: int, cap_gib):
    """(windows, total mapped bytes) for one file mapping."""
    max_buf = min(cap_gib * GIB if cap_gib else WINDOW_CAP, WINDOW_CAP)
    step = max_buf - CHUNK_OVERLAP if max_buf > CHUNK_OVERLAP else max_buf // 2
    n, start, span = 0, 0, 0
    while start < size:
        length = min(max_buf, size - start)
        n += 1
        span += length
        if start + length == size:
            break
        start += step
    return n, span


def main():
    rows = []
    print(f"{'arm':<12} {'size GiB':>9} {'span GiB':>9} {'span GB':>8} "
          f"{'pred_w':>6} {'obs_w':>5} {'overlap GB':>10}  verdict")
    for name, sizes, cap, observed, ok in ARMS:
        n = sum(mapping(s, cap)[0] for s in sizes)
        span = sum(mapping(s, cap)[1] for s in sizes)
        total = sum(sizes)
        flag = "ok" if n == observed else f"MISMATCH obs={observed}"
        print(f"{name:<12} {total / GIB:9.2f} {span / GIB:9.2f} {span / 1e9:8.2f} "
              f"{n:6d} {observed:5d} {(span - total) / 1e9:10.2f}  "
              f"{'correct' if ok else 'BROKEN':<8} {flag}")
        rows.append((span, n, total, span - total, ok))

    print("\nDoes a single threshold on each candidate separate the two groups?")
    for label, idx, unit in (("mapped span", 0, GIB), ("window count", 1, 1),
                             ("unique bytes", 2, GIB), ("overlap bytes", 3, GIB)):
        good = sorted(r[idx] for r in rows if r[4])
        bad = sorted(r[idx] for r in rows if not r[4])
        sep = good[-1] < bad[0]
        print(f"  {label:<13} max(correct)={good[-1] / unit:8.2f}  "
              f"min(broken)={bad[0] / unit:8.2f}  "
              f"{'separates' if sep else 'DOES NOT SEPARATE'}"
              + ("  (GiB)" if unit == GIB else "  (windows)"))
        if not sep:
            over = [round(g / unit, 2) for g in good if g >= bad[0]]
            print(f"                  counter-examples: {over} correct at or "
                  f"above the lowest broken value")


if __name__ == "__main__":
    main()
