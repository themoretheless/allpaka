//! The Studio panel clamps every swarm input in the browser before sending it, and
//! `swarm::validate` rejects the same ranges on the server. When the two lists
//! disagree the client silently rewrites a value into one the server then refuses,
//! so the user sees a failure with no cause.
//!
//! Both sides are read from source here, and no bound is written as a literal in
//! this file: retuning or renaming a constant in Rust cannot make this test pass by
//! accident. `a_retuned_bound_shows_up_as_drift` mutates the JavaScript in memory to
//! prove the comparison still bites.

use std::path::Path;

fn read(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `pub const NAME: usize = 6;`
fn rust_const(source: &str, name: &str) -> usize {
    let marker = format!("const {name}: ");
    let found = source
        .find(&marker)
        .unwrap_or_else(|| panic!("`{name}` is no longer a const in src/swarm.rs"));
    let after = &source[found + marker.len()..];
    let value = &after[after.find('=').expect("const without `=`") + 1..];
    let digits: String = value
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("const without a number")
}

/// `fn name() -> usize { 6000 }`
fn rust_default(source: &str, name: &str) -> usize {
    let marker = format!("fn {name}() -> ");
    let found = source
        .find(&marker)
        .unwrap_or_else(|| panic!("`{name}` is gone from src/types.rs"));
    let after = &source[found + marker.len()..];
    let body = &after[after.find('{').expect("default fn without a body") + 1..];
    let value = &body[..body.find('}').expect("unterminated default fn")];
    value
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("`{name}` does not return a plain number"))
}

#[derive(Debug)]
struct Clamp {
    subject: String,
    lo: usize,
    hi: usize,
    fallback: usize,
}

/// Every `clampNum(x, lo, hi, fallback)` call, matched across nested parentheses.
fn clamps(js: &str) -> Vec<Clamp> {
    let bytes = js.as_bytes();
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = js[cursor..].find("clampNum(") {
        let open = cursor + rel + "clampNum(".len();
        let mut depth = 1usize;
        let mut at = open;
        while at < bytes.len() && depth > 0 {
            match bytes[at] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            at += 1;
        }
        let args = split_arguments(&js[open..at - 1]);
        if args.len() == 4 {
            if let (Ok(lo), Ok(hi), Ok(fallback)) = (
                args[1].trim().parse(),
                args[2].trim().parse(),
                args[3].trim().parse(),
            ) {
                out.push(Clamp {
                    subject: args[0].trim().to_string(),
                    lo,
                    hi,
                    fallback,
                });
            }
        }
        cursor = at;
    }
    out
}

fn split_arguments(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (index, ch) in args.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(args[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    out.push(args[start..].trim());
    out
}

/// `field: number` inside the `defaultSwarm()` object literal.
fn default_swarm_field(js: &str, field: &str) -> usize {
    let found = js
        .find("function defaultSwarm()")
        .expect("defaultSwarm() is gone from web/app.js");
    let line_end = js[found..].find('\n').expect("unterminated defaultSwarm()");
    let line = &js[found..found + line_end];
    let marker = format!("{field}:");
    let at = line
        .find(&marker)
        .unwrap_or_else(|| panic!("defaultSwarm() no longer sets {field}"));
    let digits: String = line[at + marker.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse()
        .expect("defaultSwarm() field without a number")
}

/// The member-count guards on the cost line: `if(count<2)…участник…`.
fn member_guards(js: &str) -> (usize, usize) {
    let mut min = None;
    let mut max = None;
    for line in js.lines().filter(|l| l.contains("участник")) {
        for (marker, slot) in [("count<", &mut min), ("count>", &mut max)] {
            if let Some(at) = line.find(marker) {
                let digits: String = line[at + marker.len()..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                *slot = digits.parse().ok();
            }
        }
    }
    (
        min.expect("no `count<` member guard in web/app.js"),
        max.expect("no `count>` member guard in web/app.js"),
    )
}

fn violations(server: &str, types: &str, js: &str) -> Vec<String> {
    let mut drift = Vec::new();
    let swarm_clamps: Vec<Clamp> = clamps(js)
        .into_iter()
        .filter(|c| c.subject.contains("swarm"))
        .collect();

    // Each bound is clamped twice: once when the request is built, once when saved
    // preferences are restored into the panel. Both copies must agree with Rust.
    let unwatched: Vec<&Clamp> = swarm_clamps
        .iter()
        .filter(|c| {
            !["rounds", "steps", "report"]
                .iter()
                .any(|n| c.subject.contains(n))
        })
        .collect();
    for clamp in &unwatched {
        drift.push(format!(
            "web/app.js clamps `{}` but this test bounds no constant against it",
            clamp.subject
        ));
    }

    let bounds = [
        (
            "rounds",
            "wave count",
            1usize,
            rust_const(server, "MAX_ROUNDS"),
            rust_default(types, "default_swarm_rounds"),
            "MAX_ROUNDS / default_swarm_rounds",
        ),
        (
            "steps",
            "steps per member",
            1,
            rust_const(server, "MAX_STEPS_PER_MEMBER"),
            rust_default(types, "default_swarm_steps"),
            "MAX_STEPS_PER_MEMBER / default_swarm_steps",
        ),
    ];
    for (needle, label, lo, hi, fallback, source) in bounds {
        let found: Vec<&Clamp> = swarm_clamps
            .iter()
            .filter(|c| c.subject.contains(needle))
            .collect();
        if found.is_empty() {
            drift.push(format!(
                "web/app.js no longer clamps anything matching `{needle}`"
            ));
        }
        for clamp in found {
            if (clamp.lo, clamp.hi, clamp.fallback) != (lo, hi, fallback) {
                let (used_lo, used_hi, used_default) = (clamp.lo, clamp.hi, clamp.fallback);
                drift.push(format!(
                    "{label}: web/app.js uses ({used_lo},{used_hi},{used_default}) but src says {lo}..={hi} default {fallback} ({source})"
                ));
            }
        }
    }

    // The panel counts kilobytes, the server counts bytes.
    let report = (
        rust_const(server, "MIN_REPORT_BYTES"),
        rust_const(server, "MAX_REPORT_BYTES"),
        rust_default(types, "default_swarm_report_bytes"),
    );
    let found: Vec<&Clamp> = swarm_clamps
        .iter()
        .filter(|c| c.subject.contains("report"))
        .collect();
    if found.is_empty() {
        drift.push("web/app.js no longer clamps the report budget".to_string());
    }
    for clamp in found {
        if (clamp.lo * 1000, clamp.hi * 1000, clamp.fallback * 1000) != report {
            drift.push(format!(
                "report budget: web/app.js uses ({}k,{}k,{}k) but src says {}..={} bytes default {} ({})",
                clamp.lo, clamp.hi, clamp.fallback, report.0, report.1, report.2,
                "MIN/MAX_REPORT_BYTES / default_swarm_report_bytes"
            ));
        }
    }

    let (guard_min, guard_max) = member_guards(js);
    let members = (
        rust_const(server, "MIN_MEMBERS"),
        rust_const(server, "MAX_MEMBERS"),
    );
    if (guard_min, guard_max) != members {
        drift.push(format!(
            "member count: web/app.js guards count<{guard_min} / count>{guard_max} but src says MIN_MEMBERS={} MAX_MEMBERS={}",
            members.0, members.1
        ));
    }

    for (field, expected, source) in [
        (
            "rounds",
            rust_default(types, "default_swarm_rounds"),
            "default_swarm_rounds",
        ),
        (
            "max_steps_per_member",
            rust_default(types, "default_swarm_steps"),
            "default_swarm_steps",
        ),
        (
            "report_bytes",
            rust_default(types, "default_swarm_report_bytes"),
            "default_swarm_report_bytes",
        ),
    ] {
        let actual = default_swarm_field(js, field);
        if actual != expected {
            drift.push(format!(
                "defaultSwarm().{field} is {actual} but src says {expected} ({source})"
            ));
        }
    }
    drift
}

#[test]
fn studio_swarm_bounds_match_the_server() {
    let drift = violations(
        &read("src/swarm.rs"),
        &read("src/types.rs"),
        &read("web/app.js"),
    );
    assert!(
        drift.is_empty(),
        "swarm bounds drifted between the Studio panel and the server:\n{}",
        drift.join("\n")
    );
}

/// The guard is only worth having if it fails. Each mutation is one real bound moved
/// by one, done to a copy of the JavaScript so no shared file is touched.
#[test]
fn a_retuned_bound_shows_up_as_drift() {
    let server = read("src/swarm.rs");
    let types = read("src/types.rs");
    let js = read("web/app.js");
    assert!(
        violations(&server, &types, &js).is_empty(),
        "the panel already drifts, so this test proves nothing yet:\n{}",
        violations(&server, &types, &js).join("\n")
    );
    for (from, to) in [
        ("value,1,8,2", "value,1,7,2"),
        ("rounds').value,1,2,1", "rounds').value,1,3,1"),
        ("count>6", "count>9"),
        ("report_bytes:6000", "report_bytes:6500"),
        ("value,1,24,6", "value,1,25,6"),
    ] {
        assert!(
            js.contains(from),
            "`{from}` left web/app.js, so the mutation stopped proving anything"
        );
        let found = violations(&server, &types, &js.replace(from, to));
        assert!(
            !found.is_empty(),
            "mutating `{from}` to `{to}` produced no drift — the comparison is blind"
        );
    }
}
