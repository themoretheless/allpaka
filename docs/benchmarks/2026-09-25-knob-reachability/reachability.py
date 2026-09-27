#!/usr/bin/env python3
"""Which decode knobs can the shipped policy actually reach?

For every env-var knob accessor in `crates/allpaka-backend/src/gpu/metal.rs`,
walk outward from each call site through its enclosing blocks (brace balance,
up to ENC_DEPTH levels) and print every condition on that path that mentions a
`RuntimePolicy` field or another gate word.

Then, and this is the part the first version lacked, RESOLVE each policy field:
the field the encoder tests is not always the bit in the runtime struct.
`LayerRefs.normflag` is `runtime.normflag && <the first three mats are all
matvec_q4_k_mv> && Moe`, so a model can reach a `!normflag` branch with the
policy bit ON - which is exactly what qwen3-30b does in 24 of its 48 layers, and
the reason this file's original "behind `!normflag`, therefore dead under every
default" answer retired three 2026-09-12 numbers that turned out to be real
arms. See README.md's 2026-09-26T01:04Z
amendment and ../2026-09-25-normflag-policy/ for the paired measurement.

Structural, so no GPU and no build: `python3 reachability.py` from
docs/benchmarks/2026-09-25-knob-reachability/, and `--selfcheck` for its
mutation cases.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
TARGETS = {
    "metal": REPO / "crates/allpaka-backend/src/gpu/metal.rs",
    "runtime": REPO / "crates/allpaka-backend/src/runtime.rs",
    "profile": REPO / "crates/allpaka-backend/src/profile.rs",
}
POLICY_FIELDS = (
    "normflag", "decode_serial", "prefill_defer", "prefill_one_buffer",
    "gpu_route", "mm_pipeline", "attention_split",
)
GATE_WORDS = POLICY_FIELDS + ("moe_plain", "mega", "serial", "profile", "get()")
# The whole surface, not one accessor dialect. The original enumeration here was
# `fn (\w+)\(\) -> (?:bool|u32|usize|f32|i32) \{`, a SIGNATURE test, so it silently
# skipped every knob whose reader was not a `fn name() -> bool`: `fn mv_tg() -> u64`
# (u64 is not in the list), `fn attend_kernel() -> &'static str` (a kernel SELECTOR,
# which is how the decode attention ladder is written), `fn serial_dispatch() ->
# metal::MTLDispatchType`. That mattered: the two knobs this file was asked about
# most recently, ATTN_MV and ATTN_S8, are both read only inside attend_kernel, so the
# census that retired the RFUSE row could not see them at all, and the S8 arm's real
# guard - the `ATTN_MV` early return three lines above its own read - is invisible to
# a call-path walk even once the function is enumerated, because it is sequential,
# not nested. Functions with arguments (decode_token, encode_verify_tokens,
# ffn_batch_grouped, lanes_per_row, mm_kernel_for, the prefill blocks) are a third
# shape: their reads are inline in an encode function with no call sites to walk, so
# `inline_reads()` reports the brace nesting at the read line instead of pretending
# they are accessors.
ZERO_ARG_RE = re.compile(r"^\s*(?:pub )?fn (\w+)\(\)(?:\s*->\s*[^\{]+)?\s*\{")
FN_OPEN_RE = re.compile(r"^\s*(?:pub |const |async |unsafe )*fn (\w+)\(")
ENV_RE = re.compile(r'"(ALLPAKA_[A-Z0-9_]+)"')
ENC_DEPTH = 8


def fn_span(lines, i):
    """(start, end) line indices of the body of the function whose header is at i,
    by brace balance. `end` is the index of the line carrying the closing brace."""
    depth = 0
    for j in range(i, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if j > i and depth <= 0:
            return i, j
        if j == i and depth <= 0:
            return i, i
    return i, len(lines) - 1


def sequential_gates(lines, start, end, read_idx):
    """Conditions inside the same body that must have gone FALSE before `read_idx`.

    A selector written as `if knob_a() { return X; } … knob_b()` gates B's arm on A
    without either read sitting inside a block the other is inside, so
    `enclosing_conditions()` - which walks the brace nesting - cannot see it. This
    is the shape the attention ladder uses, and it is the difference between
    "ATTN_S8 is an unmeasured lever" and "ATTN_S8 has four arms, all of which are
    only encodable with ATTN_MV=0".
    """
    out = []
    for k in range(start + 1, min(read_idx, end)):
        line = lines[k]
        if not re.search(r"\bif\b.*\{", line) or line.lstrip().startswith("//"):
            continue
        s, e = fn_span(lines, k)
        if e >= read_idx or e <= k:
            continue
        body = " ".join(lines[k:e + 1])
        if re.search(r"\breturn\b", body):
            cond = re.sub(r"^\s*(?:}\s*else\s+)?if\s+", "", line).rstrip("{ ").strip()
            out.append((k + 1, cond[:120]))
    return out


def accessors(lines):
    """Every zero-argument function that reads an ALLPAKA_* name, whatever it returns."""
    out = {}
    for i, line in enumerate(lines):
        m = ZERO_ARG_RE.match(line)
        if not m:
            continue
        _, end = fn_span(lines, i)
        envs = set()
        for j in range(i, end + 1):
            envs.update(ENV_RE.findall(lines[j]))
        if envs:
            out[m.group(1)] = sorted(envs)
    return out



def enclosing_conditions(lines, call_idx):
    """Walk up from the call line, tracking brace depth; return the opening
    lines of each enclosing block that carry a condition."""
    depth = 0
    path = []
    for j in range(call_idx, -1, -1):
        line = lines[j]
        depth += line.count("}") - line.count("{")
        if depth < 0:
            # `line` opens the block we are standing in
            if any(w in line for w in GATE_WORDS):
                path.append((j + 1, line.strip()[:110]))
            depth = 0
            if len(path) >= ENC_DEPTH:
                break
    return path


def gate_definition(lines, ident):
    """Resolve one level: a bare `if flag {` gate usually hides the policy read
    in the `let flag = ...` that set it, several lines up."""
    pat = re.compile(rf"^\s*let(?:\s+mut)?\s+{re.escape(ident)}\b")
    for i, line in enumerate(lines):
        if not pat.match(line):
            continue
        blob, depth = [], 0
        for j in range(i, min(i + 8, len(lines))):
            blob.append(lines[j].strip())
            depth += sum(c in "([{" for c in lines[j]) - sum(c in ")]}" for c in lines[j])
            if depth <= 0 and blob[-1].endswith((";", "{", ")")):
                break
        return " ".join(blob)[:240]
    return None


# ---------------------------------------------------------------------------
# How a policy FIELD is resolved, as opposed to what the policy VALUE is.
#
# This block exists because the first version of this file answered the wrong
# question. It reported "this accessor sits behind `!normflag`", which reads as
# "dead under every default profile" - and ../2026-09-25-rfuse-unreachable-arm/
# README.md drew exactly that conclusion and closed RFUSE/RMT/RTOPK as knobs that
# select nothing. But the field the encoder tests is `LayerRefs.normflag`, which
# `metal.rs` builds per layer as `runtime.normflag && <q,k,v all matvec_q4_k_mv>
# && Moe`: a conjunction over the MODEL's tensor types. On qwen3-30b that is TRUE
# in 24 of 48 layers and FALSE in the other 24 with the policy bit left at its
# default, and a paired series measured the `rf` arm 18 % slower with exactly
# -7680 dispatches (../2026-09-25-normflag-policy/). So a reachability answer that
# reads only the policy value can be wrong in the direction that deletes a lever.
#
# `gate_resolutions()` therefore separates the two: for each field, the top-level
# clauses of its initializer, split into policy reads and structural reads
# (kernel names, layer-shape matches), plus the sites where the value is carried
# as a struct field and read back off `refs`. A field with a structural clause
# that is carried per layer has NO policy-only verdict available; which model
# reaches it is header data, computable by ../2026-09-25-normflag-policy/
# layer-types.py.
# ---------------------------------------------------------------------------
POLICY_READ = re.compile(r"runtime|policy|\bget\(\)|env::var|self\.")
TYPE_DECL = re.compile(r"[A-Za-z_][\w:<>]*(?:<[^<>]*>)?\s*[;,]?\s*$")
MAX_STATEMENT_LINES = 24


def strip_comment(line):
    return line.split("//", 1)[0].rstrip()


def statement(lines, start):
    """Join a multi-line `let ...;` into one string, comments removed, tracking
    bracket depth so a statement that ends inside a closure is not cut short."""
    parts, depth = [], 0
    for j in range(start, min(start + MAX_STATEMENT_LINES, len(lines))):
        body = strip_comment(lines[j])
        if not body.strip() and parts:
            continue
        depth += sum(c in "([{" for c in body) - sum(c in ")]}" for c in body)
        parts.append(body.strip())
        if depth <= 0 and body.rstrip().endswith(";"):
            break
    return " ".join(parts), start + 1


def split_clauses(text):
    """Split on top-level `&&` and `||` only. The real `normflag` initializer hides
    `&&` inside a `.is_some_and(...)` closure, and a naive split reports six clauses
    for a three-term conjunction - which would overstate how model-dependent the gate
    is by naming the closure body as its own term. `||` splits too because a
    disjunct is equally a second condition on the value (`decode_serial || any_mega`)."""
    out, buf, depth, i = [], [], 0, 0
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0 and text[i:i + 2] in ("&&", "||"):
            out.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        buf.append(c)
        i += 1
    out.append("".join(buf).strip())
    return [c for c in out if c]


HEAD_RE = re.compile(r"^\s*(?:pub\s+)?(?:let(?:\s+mut)?|fn|return|if|match|}\s*else\s*if)\b")
# The two spellings a policy value takes in `metal.rs`: through the process-wide
# `crate::runtime::get()`, and as a field of a `policy` binding.
POLICY_SITE_RE = r"(?:get\(\)|\bpolicy|\bruntime)\s*\.\s*{field}\b"


def statement_head(lines, i, back=8):
    """The line that starts the statement containing line `i`: walk up to the first
    `let`/`fn`/`if`/`return` at shallower-or-equal indentation."""
    indent = len(lines[i]) - len(lines[i].lstrip())
    for j in range(i, max(i - back, -1), -1):
        s = strip_comment(lines[j])
        if not s.strip():
            continue
        if HEAD_RE.match(s) and len(s) - len(s.lstrip()) <= indent:
            return j
    return i


def gate_resolutions(lines, fields=POLICY_FIELDS):
    """field -> the expressions its value is built from, and how those travel."""
    out = {}
    for field in fields:
        site_re = re.compile(POLICY_SITE_RE.format(field=re.escape(field)))
        let_re = re.compile(rf"^\s*(?:pub )?let(?:\s+mut)?\s+{re.escape(field)}\s*=\s*(.*)$")
        sites, seen = [], set()

        def record(i, kind):
            head = statement_head(lines, i)
            if head in seen:
                return
            seen.add(head)
            text, at = statement(lines, head)
            rhs = text.split("=", 1)[1] if "=" in text.split("(")[0] else text
            rhs = rhs.rsplit(";", 1)[0]
            clauses = split_clauses(rhs)
            m = re.match(r"^\s*(?:pub )?let(?:\s+mut)?\s+(\w+)", strip_comment(lines[head]))
            sites.append({
                "kind": kind, "at": at, "local": m.group(1) if m else None,
                "clauses": clauses,
                "structural": [c for c in clauses if not POLICY_READ.search(c)],
                "text": text[:400],
            })

        for i, line in enumerate(lines):
            if let_re.match(strip_comment(line)):
                record(i, "local named after the field")
            elif site_re.search(strip_comment(line)):
                record(i, "inline policy read")
        # `normflag,` (shorthand store) and `normflag: runtime.normflag` are the
        # value travelling into the per-layer struct; `normflag: bool,` is only the
        # struct's type declaration and says nothing about resolution, so it must
        # not count as a carry or every field declared on LayerRefs would look
        # per-layer.
        carried, declared = [], []
        for i, line in enumerate(lines):
            s = strip_comment(line)
            m = re.match(rf"^\s*(?:pub )?{re.escape(field)}\s*:\s*(.+?)\s*$", s)
            if m:
                (declared if TYPE_DECL.match(m.group(1)) else carried).append(i + 1)
            elif re.match(rf"^\s*{re.escape(field)}\s*,\s*$", s):
                carried.append(i + 1)
        reads = [i + 1 for i, line in enumerate(lines)
                 if re.search(rf"\brefs\.{re.escape(field)}\b", strip_comment(line))]
        # A conjunction whose value lands in a local that an `if` then tests is
        # carried too, just not into a per-layer struct - the distinction that
        # decides whether a model can reach the branch at all.
        used = []
        for s in sites:
            if s["local"]:
                used += [i + 1 for i, line in enumerate(lines)
                         if i + 1 != s["at"]
                         and re.search(rf"\b(?:if|else if|while)[^{{}}]*\b{re.escape(s['local'])}\b",
                                       strip_comment(line))]
        conj = [s for s in sites if s["structural"]]
        if not sites:
            verdict = "no read site in this file"
        elif conj and (carried or reads):
            verdict = "PER-LAYER CONJUNCTION"
        elif conj:
            verdict = "CONJUNCTION, not carried per layer"
        elif carried or reads:
            verdict = "POLICY BIT, replicated per layer"
        elif len(sites) > 1:
            verdict = "POLICY VALUE at several sites" if not used else \
                "POLICY VALUE, local tested by an if"
        else:
            verdict = "POLICY VALUE"
        out[field] = {"verdict": verdict, "sites": sites, "carried": carried,
                      "declared": declared, "reads": reads, "used": sorted(set(used))}
    return out


def print_resolutions(res):
    per_layer = [f for f, r in res.items() if r["verdict"] == "PER-LAYER CONJUNCTION"]
    print("policy FIELDS, resolved - what the gate actually depends on, not what the "
          "policy bit says:")
    for field in POLICY_FIELDS:
        r = res[field]
        print(f"  {field:18s} {r['verdict']}")
        for s in r["sites"]:
            local = f" -> `{s['local']}`" if s["local"] else ""
            print(f"      metal.rs:{s['at']:<6d} {s['kind']}{local}: "
                  f"{len(s['clauses'])} clause(s), {len(s['structural'])} structural")
            for c in s["structural"]:
                print(f"        <- {c[:120]}")
        if r["carried"]:
            print(f"      stored per layer at {[f'metal.rs:{i}' for i in r['carried'][:6]]}")
        if r["declared"]:
            print(f"      (declared on a struct at "
                  f"{[f'metal.rs:{i}' for i in r['declared'][:4]]} - a type, not a store)")
        if r["reads"]:
            print(f"      read back as `refs.{field}` at {[f'metal.rs:{i}' for i in r['reads'][:8]]}"
                  f" ({len(r['reads'])} sites)")
        if r["used"] and not (r["carried"] or r["reads"]):
            print(f"      local tested by an `if` at {[f'metal.rs:{i}' for i in r['used'][:6]]}")
    print()
    if per_layer:
        print(f"  {', '.join(per_layer)}: a branch gated on these is decided by the MODEL's "
              f"tensor types, so no 'unreachable under the shipped policy' conclusion is "
              f"available from this file. Count the reachable layers with "
              f"../2026-09-25-normflag-policy/layer-types.py (no GPU).")
    else:
        print("  no per-layer conjunctions: every policy field here is a plain bit, "
              "so the policy value does decide reachability.")
    return per_layer


# A name counts as read only next to a read idiom: the direct calls, and the
# bench harness's own `get("NAME", default)` helper in
# crates/allpaka-backend/tests/gpu_ffnbench.rs, which is how the Q5AB_* set is
# taken in. Both quote styles count: scripts/test-studio.py reads its two
# ALLPAKA_TEST_* vars through os.environ.get('...'), and a double-quote-only
# pattern filed those names as "docs mention it, code does not read it" - the
# same class of miss that hid ALLPAKA_DCOMB for a day.
READ_IDIOM = re.compile(
    r'(?:env::var|env::var_os|std::env::var|os\.environ\.get|getenv|\benv|\bget|\bvar)\s*\(\s*["\']'
    r'(ALLPAKA_[A-Z0-9_]+)["\']'
)
ANY_NAME = re.compile(r"ALLPAKA_[A-Z0-9_]+")
CODE_DIRS = ("crates", "tools", "plugins", "scripts")


def doc_names_without_a_read():
    """The reverse direction of the census check. The polarity census
    (../2026-09-25-knob-polarity-census/) checked code -> docs: every name the
    Metal code reads is listed somewhere. That direction cannot see the two
    failures that matter here, so this checks the rest:
      docs -> code:  a registry row for a name no code reads;
      code -> docs, per file: a knob the Metal decoder reads that no doc names -
      which the census reported as zero only because it scoped `documented` to
      the two registry files and `read` to three source files.
    Read idioms are matched, not bare names, because a name appearing in code
    only inside a comment is exactly the phantom the census found (`ALLPAKA_Q`).
    Registry means docs/*.md at the top level; benchmark READMEs are dated
    experiment records and must not count as registry coverage."""
    read, read_metal, mentioned = set(), set(), set()
    for d in CODE_DIRS:
        root = REPO / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in {".rs", ".sh", ".ts", ".js", ".toml", ".py"}:
                continue
            text = p.read_text(errors="ignore")
            found = set(READ_IDIOM.findall(text))
            read |= found
            if p.name == "metal.rs":
                read_metal |= found
            mentioned.update(ANY_NAME.findall(text))
    registry, records = set(), set()
    for q in sorted((REPO / "docs").glob("*.md")):
        registry.update(ANY_NAME.findall(q.read_text(errors="ignore")))
    for q in sorted((REPO / "docs").rglob("*.md")):
        if q.parent != REPO / "docs":
            records.update(ANY_NAME.findall(q.read_text(errors="ignore")))
    documented = registry | records
    print(f"registry (docs/*.md): {len(registry)} names; benchmark records add "
          f"{len(records - registry)} more; code read sites: {len(read)} "
          f"({len(read_metal)} in gpu/metal.rs)")
    print("\nnamed in docs, never read by any code:")
    for n in sorted(documented - read):
        where = "code mentions it in prose only" if n in mentioned else "absent from code entirely"
        print(f"  {n}: {where}")
    for label, names in (("metal.rs", sorted(read_metal - registry)),
                         ("elsewhere", sorted((read - read_metal) - documented))):
        print(f"\nnames read in {label}, absent from the registry:")
        print("  " + (", ".join(names) if names else "none"))


def selfcheck():
    """Mutation checks on READ_IDIOM and on the field resolver: the scan's own
    answer to the question this file asks about the census. A name-set difference
    inherits the shape of its matcher, so the matcher gets tested the way a timing
    harness does - by feeding it cases it must not get wrong. The single-quote case
    is the one this scan failed at 22:08Z (scripts/test-studio.py's
    ALLPAKA_TEST_NATIVE_VAULT was filed as unread), and the comment case is the
    ALLPAKA_Q phantom.

    The resolver cases are the bug that this file's first version shipped with: it
    could say `behind a RuntimePolicy field` and stop, which is how RFUSE/RMT/RTOPK
    got closed as unreachable. So the fixture is the real `normflag` shape, and the
    cases check that a structural clause is SEEN, that `&&` inside a closure is not
    split into fake terms, that a commented-out clause is ignored, that a plain
    policy bit is still reported as a plain policy bit, and that a conjunction whose
    value never reaches `refs.` is not called per-layer."""
    fixture = r"""
    struct LayerRefs {
        normflag: bool,
        serial: bool,
    }
    fn build(l: &Layer, runtime: &RuntimePolicy) -> LayerRefs {
        let normflag = runtime.normflag
            // && this clause is a comment, and the line above is the only
            && mats_pre.as_ref().is_some_and(|(m, _)| {
                m.iter().take(3).all(|m| m.kernel == "matvec_q4_k_mv"
                    && m.group == 0)
            })
            && matches!(l.ffn, TokenFfn::Moe { .. });
        let serial = runtime.decode_serial;
        let mm_pipeline = env::var("ALLPAKA_MM").is_ok()
            && l.hidden == 4096;
        LayerRefs {
            normflag,
            serial,
        }
    }
    fn decode(refs: &LayerRefs) {
        if refs.normflag { barriers() }
        if !refs.normflag && rfuse() { fold() }
    }
    """
    fl = fixture.splitlines()
    decl_line = next(i + 1 for i, line in enumerate(fl)
                     if line.strip() == "normflag: bool,")
    res = gate_resolutions(fl, ("normflag", "serial", "mm_pipeline", "decode_serial",
                                "prefill_defer"))
    cases = [
        ('std::env::var("ALLPAKA_RUST_DOUBLE")', {"ALLPAKA_RUST_DOUBLE"}),
        ('env::var_os("ALLPAKA_OS")', {"ALLPAKA_OS"}),
        ('os.environ.get("ALLPAKA_PY_DOUBLE")', {"ALLPAKA_PY_DOUBLE"}),
        ("os.environ.get('ALLPAKA_PY_SINGLE')", {"ALLPAKA_PY_SINGLE"}),
        ('env("ALLPAKA_HELPER", 1)', {"ALLPAKA_HELPER"}),
        ("getenv('ALLPAKA_GETENV')", {"ALLPAKA_GETENV"}),
        ('// ALLPAKA_COMMENT_ONLY', set()),
        ('ALLPAKA_BARE_PROSE', set()),
    ]
    bad = 0
    for text, want in cases:
        got = set(READ_IDIOM.findall(text))
        if got != want:
            bad += 1
        print(f"  {'ok ' if got == want else 'FAIL'} {text[:46]:46s} -> {sorted(got)}")
    nf = res["normflag"]["sites"][0]
    verdict_cases = [
        ("the three-term conjunction is not split into the closure body's parts",
         len(nf["clauses"]) == 3),
        ("the kernel-name and layer-shape clauses are structural, the policy read is not",
         len(nf["structural"]) == 2 and "runtime.normflag" in nf["clauses"][0]),
        ("a clause that lives only in a comment is not a clause",
         not any("this clause is a comment" in c for c in nf["clauses"])),
        ("a field carried into the layer loop reads as PER-LAYER",
         res["normflag"]["verdict"] == "PER-LAYER CONJUNCTION"
         and bool(res["normflag"]["reads"])),
        ("a plain policy bit still reads as a bit, not a conjunction",
         res["serial"]["verdict"] == "POLICY BIT, replicated per layer"
         and res["serial"]["sites"][0]["structural"] == []),
        ("a conjunction that never reaches `refs.` is not called per-layer",
         res["mm_pipeline"]["verdict"] == "CONJUNCTION, not carried per layer"),
        ("a `name: bool,` type declaration is not read as the value being stored",
         decl_line in res["normflag"]["declared"]
         and decl_line not in res["normflag"]["carried"]),
        ("a field with no read site here is not invented",
         res["prefill_defer"]["verdict"] == "no read site in this file"),
        ("a value read through a differently named local is still found",
         res["decode_serial"]["sites"][0]["local"] == "serial"),
    ]
    for name, ok in verdict_cases:
        if not ok:
            bad += 1
        print(f"  {'ok ' if ok else 'FAIL'} {name}")

    # The half of the surface the signature test used to miss, checked against a
    # synthetic file rather than against metal.rs: the answer must not depend on
    # whether the reader returns `bool`, whether its signature fits on one line, or
    # whether its guard is a nesting block instead of an earlier `return`.
    synth = [
        'fn old_style() -> bool { std::env::var("ALLPAKA_OLD").unwrap().is_empty() }',
        'fn selector() -> &\'static str {',
        '    static A: OnceLock<bool> = OnceLock::new();',
        '    if *A.get_or_init(|| std::env::var("ALLPAKA_FIRST").map_or(true, |v| v != "0")) {',
        '        return "kernel_a";',
        '    }',
        '    static B: OnceLock<usize> = OnceLock::new();',
        '    let n = std::env::var("ALLPAKA_SECOND").ok().and_then(|v| v.parse().ok()).unwrap_or(32);',
        '    match n { 32 => "k32", _ => "k8" }',
        '}',
        'fn u64_reader() -> u64 {',
        '    static T: OnceLock<u64> = OnceLock::new();',
        '    *T.get_or_init(|| std::env::var("ALLPAKA_TG").ok().and_then(|v| v.parse().ok()).unwrap_or(128))',
        '}',
        'fn encode(',
        '    a: usize,',
        '    b: &str,',
        ') -> Option<Vec<f32>> {',
        '    let split = std::env::var("ALLPAKA_INLINE_SPLIT").is_ok();',
        '    None',
        '}',
        'fn no_env() -> bool { true }',
    ]
    acc_s = accessors(synth)
    sel_end = 9
    second = next(i for i, l in enumerate(synth) if "ALLPAKA_SECOND" in l)
    first = next(i for i, l in enumerate(synth) if "ALLPAKA_FIRST" in l)
    inl = inline_reads(synth, set(acc_s))
    surface_cases = [
        ("a `-> &'static str` kernel selector is enumerated (attend_kernel's shape)",
         set(acc_s.get("selector", [])) == {"ALLPAKA_FIRST", "ALLPAKA_SECOND"}),
        ("a `-> u64` reader is enumerated (mv_tg's shape; the old list had u32/usize)",
         set(acc_s.get("u64_reader", [])) == {"ALLPAKA_TG"}),
        ("a bool accessor is still enumerated", "old_style" in acc_s),
        ("a function with no env read is not", "no_env" not in acc_s),
        ("the second read in a selector reports the earlier knob as its in-body gate",
         any("ALLPAKA_FIRST" in c for _, c in sequential_gates(synth, 1, sel_end, second))),
        ("the FIRST read has no earlier gate to invent",
         sequential_gates(synth, 1, sel_end, first) == []),
        ("a read in a function whose signature spans lines is an inline read",
         set(inl) == {"ALLPAKA_INLINE_SPLIT"} and inl["ALLPAKA_INLINE_SPLIT"]["fn"] == "encode"),
        ("a selector's knobs are not also reported as inline reads",
         not ({"ALLPAKA_FIRST", "ALLPAKA_SECOND", "ALLPAKA_TG", "ALLPAKA_OLD"} & set(inl))),
    ]
    for name, ok in surface_cases:
        if not ok:
            bad += 1
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    total = len(cases) + len(verdict_cases) + len(surface_cases)
    print(f"selfcheck: {total - bad} of {total} cases pass")
    return 1 if bad else 0


def fn_headers(lines):
    """(name, has_args, first_body_line) for every `fn`, including signatures
    written across several lines. FN_HEAD_RE alone misses those, and a scan that
    misses the big encode functions reports a smaller knob surface than the file
    actually has - which is the defect this whole amendment exists to fix."""
    out = []
    for i, line in enumerate(lines):
        m = FN_OPEN_RE.match(line)
        if not m:
            continue
        depth = line.count("(") - line.count(")")
        j = i
        while depth > 0 and j + 1 < len(lines):
            j += 1
            depth += lines[j].count("(") - lines[j].count(")")
        head = " ".join(lines[i:j + 1])
        while j < len(lines) and "{" not in lines[j]:
            j += 1
        args = head[head.index("(") + 1:]
        args = args[:args.index(")")] if ")" in args else args
        out.append((m.group(1), bool(args.strip()), min(j, len(lines) - 1)))
    return out


def inline_reads(lines, enumerated):
    """Reads that live inside a function with arguments - not accessors.

    These are the encode paths themselves (decode_token, ffn_batch_grouped,
    prefill_attn_block, lanes_per_row, mm_kernel_for, attach, add_mapping,
    probe_skip). Their knobs have no call sites to walk, so the gating that applies
    to them is the brace nesting around the read line inside that function, which
    `enclosing_conditions()` gives directly. The debug/observability family
    (DECODE_SPLIT, *_DEBUG, *_TIME, *_ZERO, NO_GPU, SKIP) lives here, which is why a
    census that only enumerates accessors reports a smaller surface than the one
    actually in the file.
    """
    out = {}
    in_selector = set()
    for i, line in enumerate(lines):
        if ZERO_ARG_RE.match(line):
            _, end = fn_span(lines, i)
            in_selector.update(range(i, end + 1))
    for name, has_args, first in fn_headers(lines):
        if not has_args:
            continue
        _, end = fn_span(lines, first)
        for j in range(first, end + 1):
            if j in in_selector:
                continue
            for nm in ENV_RE.findall(lines[j]):
                if nm in enumerated:
                    continue
                rec = out.setdefault(nm, {"fn": name, "lines": [], "gates": set()})
                rec["lines"].append(j + 1)
                for _, text in enclosing_conditions(lines, j):
                    rec["gates"].add(text)
    return out


def main():
    lines = {k: v.read_text().splitlines() for k, v in TARGETS.items()}
    rows = []
    acc = accessors(lines["metal"])
    for fname, envs in acc.items():
        calls = []
        for i, line in enumerate(lines["metal"]):
            if re.search(rf"\b{re.escape(fname)}\(\)", line) and not ZERO_ARG_RE.match(line):
                calls.append(i)
        gates = set()
        in_policy_gate = False
        for c in calls:
            # A call sitting in a `let flag = ...` initializer has no enclosing
            # `if`: the knob *is* part of the gate's definition. Follow it.
            head = None
            for back in range(c, max(c - 4, -1), -1):
                m = re.match(r"^\s*(?:pub )?let(?:\s+mut)?\s+(\w+)\s*=", lines["metal"][back])
                if m:
                    head = m.group(1)
                    break
            if head:
                flag = head
                used = [i + 1 for i, l in enumerate(lines["metal"])
                        if i != c and re.search(rf"\bif\s+{re.escape(flag)}\b", l)]
                text = " ".join(x.strip() for x in lines["metal"][c - 3:c + 3])[:240]
                if used:
                    gates.add(f"{flag} := {text} (tested at {used[:4]})")
                    in_policy_gate |= any(f in text for f in POLICY_FIELDS)
            for ln, text in enclosing_conditions(lines["metal"], c):
                gates.add(text)
                in_policy_gate |= any(f in text for f in POLICY_FIELDS)
                bare = re.fullmatch(r"(?:} else )?if (\w+) \{.*", text)
                if bare:
                    defined = gate_definition(lines["metal"], bare.group(1))
                    if defined:
                        gates.add(f"{bare.group(1)} := {defined}")
                        # The gate word can be several calls deep: `rfuse()` is
                        # read in the initializer of `moe_plain`, and `moe_plain`
                        # is what the `if` tests, so no single line names a
                        # policy field.
                        in_policy_gate |= any(f in defined for f in POLICY_FIELDS)
        # In-body sequential gates: what had to read FALSE for this line to run.
        seq = []
        for i, line in enumerate(lines["metal"]):
            z = ZERO_ARG_RE.match(line)
            if not z or z.group(1) != fname:
                continue
            _, end = fn_span(lines["metal"], i)
            for j in range(i, end + 1):
                for nm in ENV_RE.findall(lines["metal"][j]):
                    for ln, cond in sequential_gates(lines["metal"], i, end, j):
                        seq.append(f"{nm} (metal.rs:{j + 1}) only reached if: {cond}")
        rows.append((fname, envs, calls, gates, in_policy_gate, sorted(set(seq))))

    rows.sort(key=lambda r: (not r[3], r[0]))
    for fname, envs, calls, gates, _, seq in rows:
        print(f"{fname}  env={','.join(envs)}  calls={len(calls)} "
              f"at {[c + 1 for c in calls][:8]}")
        for g in sorted(gates):
            print(f"    gated by: {g}")
        for s in seq:
            print(f"    in-body: {s}")
    print()
    inline = inline_reads(lines["metal"], set(acc))
    if inline:
        print("reads inside argument-taking functions (no call sites to walk; "
              "gates are the brace nesting at the read line):")
        for nm in sorted(inline):
            rec = inline[nm]
            print(f"  {nm}  fn={rec['fn']}  at {rec['lines'][:6]}")
            for g in sorted(rec["gates"])[:4]:
                print(f"      gated by: {g[:150]}")
        print()

    read_all = set(READ_IDIOM.findall("\n".join(lines["metal"])))
    covered = {n for envs in acc.values() for n in envs} | set(inline)
    resid = sorted(read_all - covered)
    print(f"names read in metal.rs: {len(read_all)}; enumerated here: {len(covered)} "
          f"({len(acc)} zero-arg readers + {len(inline)} inline); "
          f"outside both sections: {len(resid)}"
          + (": " + ", ".join(resid) if resid else ""))
    print()
    print(f"{sum(1 for r in rows if r[3])} of {len(rows)} zero-arg env readers sit "
          "behind at least "
          "one gate word on their call path; "
          f"{sum(1 for r in rows if r[4])} "
          "behind a RuntimePolicy field (directly or through a local).")
    print("policy-gated: " + ", ".join(sorted(r[0] for r in rows if r[4])) + "; "
          "ungated: " + ", ".join(sorted(r[0] for r in rows if not r[4])))
    print()
    print_resolutions(gate_resolutions(lines["metal"]))
    print()
    doc_names_without_a_read()


if __name__ == "__main__":
    import sys
    sys.exit(selfcheck() if "--selfcheck" in sys.argv[1:] else main())
