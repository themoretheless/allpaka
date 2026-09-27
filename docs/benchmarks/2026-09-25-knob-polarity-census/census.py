#!/usr/bin/env python3
"""Classify every ALLPAKA_* env read in the Metal backend by its polarity idiom.

Why: the knob registry in docs/decode-opts.md is written by *verdict*, so a knob
whose env is read as `var_os(..).is_some()` (present == on, so `=0` turns it ON)
sits in the same table as one read as `is_ok_and(|v| v == "1")` and nothing says
so. A sweep that exports `NAME=0` meaning "off" reports the presence-based knobs
backwards, and this repo has already been bitten by a source comment that
inverted a knob's polarity. Run from the repo root:

    python3 docs/benchmarks/2026-09-25-knob-polarity-census/census.py

Classes are named after the source idiom. Anything the patterns miss lands in
`unclassified` for a human to read - the review list is the point, not the
regex. A name whose sites disagree is marked MIXED, and MIXED is not always a
bug: `ALLPAKA_Q2_ILV` is read once as `is_some()` to pick a kernel and once as
`is_none()` to gate the other kernel's NR0 branch, which is the same polarity
expressed twice.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "crates/allpaka-backend/src"

# The Metal decode path only; the CUDA backend has its own env dialect.
SCOPES = [SRC / "gpu/metal.rs", SRC / "gpu/msl", SRC / "runtime.rs", SRC / "profile.rs"]

IDIOMS = [
    ("presence = ON (`=0` turns it ON)", r'var_os\(\s*"{N}"\s*\)\s*\.is_some\(\)'),
    ("presence = OFF (any value turns it OFF)", r'var_os\(\s*"{N}"\s*\)\s*\.is_none\(\)'),
    ("opt-in `=1`", r'var\(\s*"{N}"\s*\)[\s\S]{0,90}?is_ok_and\(\s*\|\s*v\s*\|\s*v\s*==\s*"1"'),
    ("opt-out: default ON, `=0` reverts", r'var\(\s*"{N}"\s*\)[\s\S]{0,90}?is_ok_and\(\s*\|\s*v\s*\|\s*v\s*==\s*"0"'),
    ("default ON, `=0` off, any other value ON", r'"{N}"\s*\)\s*\.map_or\(\s*true\s*,\s*\|\s*v\s*\|\s*v\s*!=\s*"0"'),
    ("default OFF, any other value ON", r'"{N}"\s*\)\s*\.map_or\(\s*false\s*,\s*\|\s*v\s*\|\s*v\s*!=\s*"0"'),
    ("opt-in by word (1/on/true)", r'"{N}"[\s\S]{0,140}?"on"|"{N}"[\s\S]{0,140}?"true"'),
    ("value list (a selector, not a flag)", r'''"{N}"[\s\S]{0,140}?split\(','\)'''),
    # Both parse_bool forms name the knob in the same expression that reaches the
    # parser: the tuple list `("ALLPAKA_X", "field")` that a `for (env, field) in
    # [...]` loop then reads with `env::var(env)` + `parse_bool(&value)`, or a
    # direct `var("ALLPAKA_X")` whose next 240 characters call parse_bool. The
    # entry this replaces matched the bare word `parse_bool` anywhere in a
    # 12-line window, which labelled every knob sitting above such a call rather
    # than every knob read by one - see the dated amendment in results.txt.
    ("policy bool via parse_bool", r'\(\s*"{N}"\s*,\s*"[a-z_]+"\s*\)'),
    # The call form needs its argument's `&`; `fn parse_bool(value: &str)` on the
    # next line would otherwise match every knob defined just above it.
    ("policy bool via parse_bool", r'"{N}"[\s\S]{0,240}?parse_bool\(\s*&'),
    ("numeric sweep (explicit default)", r'"{N}"[\s\S]{0,180}?\.parse\s*\(\s*\)[\s\S]{0,90}?unwrap_or'),
    ("numeric sweep", r'"{N}"[\s\S]{0,180}?\.parse\s*::<\s*(?:usize|u32|f32)\s*>\s*\(\s*\)'),
    # After both numeric-sweep entries, so a `.parse()` that also has an explicit
    # default keeps the class that names the default.
    ("numeric parse (a selector, not a flag)", r'"{N}"[\s\S]{0,240}?\.parse\s*\(\s*\)'),
]

NAME_RE = re.compile(r'"(ALLPAKA_[A-Z0-9_]+)"')


def env_read_sites():
    paths = []
    for scope in SCOPES:
        paths.extend([scope] if scope.is_file() else sorted(scope.rglob("*.rs")))
    for path in paths:
        lines = path.read_text().splitlines()
        for no, line in enumerate(lines):
            for m in NAME_RE.finditer(line):
                ctx = " ".join(lines[no:no + 12])
                if "env::var" not in ctx and "var_os" not in ctx:
                    continue  # a doc string or a log label, not a read
                yield m.group(1), str(path.relative_to(ROOT)), no + 1, ctx


def classify(name, ctx):
    for label, pat in IDIOMS:
        if re.search(pat.replace("{N}", re.escape(name)), ctx):
            return label
    return "unclassified"


def main():
    rows = {}
    for name, path, no, ctx in env_read_sites():
        rows.setdefault(name, []).append((f"{path}:{no}", classify(name, ctx)))

    by_class = {}
    for name in sorted(rows):
        sites = rows[name]
        classes = sorted({c for _, c in sites})
        by_class.setdefault(" / ".join(classes), []).append(name)
        mark = "  <-- MIXED" if len(classes) > 1 else ""
        print(f"| `{name}` | " + "; ".join(f"`{s}` -> {c}" for s, c in sites) + f" |{mark}")

    print("\n## count per class")
    for cls in sorted(by_class):
        print(f"- {cls}: {len(by_class[cls])} ({', '.join(by_class[cls])})")
    print(f"\n{len(rows)} names with an env read in scope")
    presence = sorted(n for n, s in rows.items() if any("presence = ON" in c for _, c in s))
    print(f"presence-based, so `=0` reads as ON: {len(presence)} ({', '.join(presence)})")


if __name__ == "__main__":
    main()
