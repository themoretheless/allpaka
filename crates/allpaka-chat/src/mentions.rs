//! Server-side `@path` / `@path:START-END` attachments for user messages.
//!
//! Every client (web UI, SDK, raw API) gets the same contract: mentions are
//! parsed here, resolved through the tool path policy (`tools::resolve`, no
//! hidden paths, no escapes, no symlinks, conversation storage excluded) and
//! bounded by a per-message context budget. Nothing is cached: each send reads
//! the file again, so an edited file is never served stale.
use crate::{
    context::Project,
    tools,
    types::{Citation, ContextBudget},
};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Most mentions accepted in one message.
pub(crate) const MAX_MENTIONS: usize = 8;
/// Bytes one attached file may contribute.
pub(crate) const PER_FILE_BYTES: usize = 16 * 1024;
/// Lines one attached file may contribute.
pub(crate) const PER_FILE_LINES: usize = 400;
/// Bytes all attachments of one message may contribute.
pub(crate) const TOTAL_BYTES: usize = 48 * 1024;
/// Largest file read before range selection.
const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct Mention {
    pub token: String,
    pub path: String,
    pub range: Option<(usize, usize)>,
}

/// Find `@relative/path[:START[-END]]` tokens. A token counts only when `@`
/// starts a word and the path contains `/` or `.`, so `@alice` or an e-mail
/// address stays ordinary text.
pub(crate) fn parse(text: &str) -> Result<Vec<Mention>> {
    let mut found: Vec<Mention> = Vec::new();
    for word in text.split_whitespace() {
        let Some(body) = word.strip_prefix('@') else {
            continue;
        };
        let body = body.trim_end_matches([',', ';', '!', '?', ')', '"', '\'', '.']);
        let (path, range) = match body.rsplit_once(':') {
            Some((path, spec))
                if !spec.is_empty() && spec.chars().all(|c| c.is_ascii_digit() || c == '-') =>
            {
                let (start, end) = spec.split_once('-').unwrap_or((spec, spec));
                let parsed = start.parse::<usize>().ok().zip(end.parse::<usize>().ok());
                match parsed {
                    Some((start, end)) if start >= 1 && end >= start => (path, Some((start, end))),
                    _ => {
                        bail!("Invalid @ line range in {word}: use START-END with 1 ≤ START ≤ END")
                    }
                }
            }
            _ => (body, None),
        };
        if path.is_empty() || !(path.contains('/') || path.contains('.')) {
            continue;
        }
        if path.len() > 4096 {
            bail!("@ path is too long");
        }
        if found.iter().any(|m| m.path == path && m.range == range) {
            continue;
        }
        found.push(Mention {
            token: word.to_owned(),
            path: path.to_owned(),
            range,
        });
        if found.len() > MAX_MENTIONS {
            bail!("At most {MAX_MENTIONS} @ files can be attached to one message");
        }
    }
    Ok(found)
}

pub(crate) struct Attached {
    /// Model-facing text: the user's message followed by cited file blocks.
    pub text: String,
    pub citations: Vec<Citation>,
    pub budget: Option<ContextBudget>,
}

/// Expand the mentions of `text`. Invalid, escaping or private paths fail
/// with an error naming the token instead of being dropped.
pub(crate) fn attach(project: &Project, excluded: &Path, text: &str) -> Result<Attached> {
    let mentions = parse(text)?;
    if mentions.is_empty() {
        return Ok(Attached {
            text: text.to_owned(),
            citations: vec![],
            budget: None,
        });
    }
    let excluded = excluded
        .canonicalize()
        .unwrap_or_else(|_| excluded.to_path_buf());
    let mut citations = Vec::new();
    let mut blocks = String::new();
    let mut used = 0usize;
    for mention in &mentions {
        let (citation, body) = read(project, &excluded, mention, TOTAL_BYTES - used)
            .with_context(|| format!("Cannot attach {}", mention.token))?;
        used += body.len();
        blocks.push_str(&format!(
            "\n\n<attached_file path=\"{}\" lines=\"{}-{}\"{}>\n{}\n</attached_file>",
            citation.path,
            citation.start_line,
            citation.end_line,
            if citation.truncated {
                " truncated=\"true\""
            } else {
                ""
            },
            body
        ));
        citations.push(citation);
    }
    Ok(Attached {
        text: format!("{text}{blocks}"),
        budget: Some(ContextBudget {
            used_bytes: used,
            limit_bytes: TOTAL_BYTES,
            per_file_bytes: PER_FILE_BYTES,
            per_file_lines: PER_FILE_LINES,
            files: citations.len(),
            truncated: citations.iter().any(|c| c.truncated),
        }),
        citations,
    })
}

fn read(
    project: &Project,
    excluded: &Path,
    mention: &Mention,
    remaining: usize,
) -> Result<(Citation, String)> {
    let first = project.roots.first().context("Project has no folders")?;
    // `@alias/path` picks a non-primary folder; everything else is relative to the primary one.
    let (root, path) = match mention.path.split_once('/') {
        Some((alias, rest)) if project.roots.len() > 1 => project
            .roots
            .iter()
            .find(|r| r.alias == alias)
            .map_or((first, mention.path.as_str()), |r| (r, rest)),
        _ => (first, mention.path.as_str()),
    };
    let canonical_root = root.path.canonicalize()?;
    let file = tools::resolve(&canonical_root, path, false)?;
    if file.starts_with(excluded) {
        bail!("Conversation storage is not available");
    }
    if !file.is_file() {
        bail!("Choose a regular text file");
    }
    let text = tools::read_text(&file, MAX_FILE_BYTES)?;
    if text.contains('\0') {
        bail!("Binary files cannot be attached");
    }
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len().max(1);
    let (start, end) = mention.range.unwrap_or((1, total));
    if start > total {
        bail!("Line {start} is past the end of the file ({total} lines)");
    }
    let end = end.min(total);
    let limit = PER_FILE_BYTES.min(remaining);
    if limit == 0 {
        bail!("Attachments exceed the {TOTAL_BYTES}-byte context budget; narrow them with @path:START-END");
    }
    let mut body = String::new();
    let mut last = start - 1;
    for (index, line) in lines.iter().enumerate().take(end).skip(start - 1) {
        let separator = usize::from(!body.is_empty());
        if index + 2 - start > PER_FILE_LINES || body.len() + separator + line.len() > limit {
            break;
        }
        if separator == 1 {
            body.push('\n');
        }
        body.push_str(line);
        last = index + 1;
    }
    if last < start && !lines.is_empty() {
        bail!("Line {start} alone exceeds the remaining context budget; narrow the attachments");
    }
    let last = last.max(start);
    let relative = file
        .strip_prefix(&canonical_root)?
        .to_string_lossy()
        .replace('\\', "/");
    Ok((
        Citation {
            root: root.alias.clone(),
            path: relative,
            start_line: start,
            end_line: last,
            bytes: body.len(),
            truncated: last < end,
        },
        body,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(dir: &Path) -> Project {
        Project {
            id: "p".into(),
            name: "p".into(),
            instructions: String::new(),
            roots: vec![crate::context::Root {
                alias: "src".into(),
                path: dir.to_path_buf(),
                writable: false,
                repository: false,
                ssh: None,
            }],
        }
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "studio-mentions-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("history")).unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn parses_paths_ranges_and_ignores_plain_words() {
        let found =
            parse("see @src/a.rs:2-4, and @b.md. ping @alice me@mail.com @src/a.rs:2-4").unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].path, "src/a.rs");
        assert_eq!(found[0].range, Some((2, 4)));
        assert_eq!(found[1].path, "b.md");
        assert_eq!(found[1].range, None);
        assert_eq!(parse("@a.rs:7").unwrap()[0].range, Some((7, 7)));
        assert!(parse("@a.rs:5-2").is_err());
        assert!(parse("@a.rs:0-2").is_err());
        let many: Vec<String> = (0..9).map(|i| format!("@f{i}.rs")).collect();
        assert!(parse(&many.join(" ")).is_err());
    }

    #[test]
    fn attaches_with_provenance_and_rejects_bad_paths() {
        let dir = temp("attach");
        std::fs::write(dir.join("hello.rs"), "one\ntwo\nthree\nfour\n").unwrap();
        std::fs::write(dir.join(".secret"), "private").unwrap();
        std::fs::write(dir.join("history/s.json"), "secret").unwrap();
        std::fs::write(dir.join("bin.dat"), b"a\0b").unwrap();
        let p = project(&dir);
        let excluded = dir.join("history");
        let out = attach(&p, &excluded, "explain @hello.rs:2-3").unwrap();
        assert_eq!(out.citations.len(), 1);
        let c = &out.citations[0];
        assert_eq!(
            (c.path.as_str(), c.start_line, c.end_line, c.truncated),
            ("hello.rs", 2, 3, false)
        );
        assert!(out.text.starts_with("explain @hello.rs:2-3"));
        assert!(out.text.contains(
            "<attached_file path=\"hello.rs\" lines=\"2-3\">\ntwo\nthree\n</attached_file>"
        ));
        let budget = out.budget.unwrap();
        assert_eq!(budget.used_bytes, "two\nthree".len());
        assert_eq!(budget.limit_bytes, TOTAL_BYTES);
        let plain = attach(&p, &excluded, "no files here @bob").unwrap();
        assert!(plain.citations.is_empty() && plain.budget.is_none());
        for bad in [
            "@../outside.rs",
            "@.secret",
            "@history/s.json",
            "@bin.dat",
            "@missing.rs",
            "@hello.rs:9-10",
        ] {
            let err = attach(&p, &excluded, bad)
                .err()
                .unwrap_or_else(|| panic!("{bad} was accepted"));
            assert!(format!("{err:#}").contains(bad), "{bad}: {err:#}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn budget_truncates_large_files_and_reports_usage() {
        let dir = temp("budget");
        let big: String = (0..2000).map(|i| format!("line {i}\n")).collect();
        for name in ["a.txt", "b.txt", "c.txt", "d.txt"] {
            std::fs::write(dir.join(name), &big).unwrap();
        }
        let p = project(&dir);
        let out = attach(&p, &dir.join("history"), "@a.txt @b.txt @c.txt @d.txt").unwrap();
        let budget = out.budget.unwrap();
        assert!(budget.truncated);
        assert!(budget.used_bytes <= TOTAL_BYTES);
        for c in &out.citations {
            assert!(c.truncated && c.bytes <= PER_FILE_BYTES);
            assert_eq!(c.end_line - c.start_line + 1, PER_FILE_LINES);
        }
        let long_lines: String = (0..10)
            .map(|_| format!("{}\n", "x".repeat(15_000)))
            .collect();
        for name in ["e.txt", "f.txt", "g.txt", "h.txt"] {
            std::fs::write(dir.join(name), &long_lines).unwrap();
        }
        let err = attach(&p, &dir.join("history"), "@e.txt @f.txt @g.txt @h.txt")
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("@h.txt"), "{err:#}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn edited_file_is_never_served_stale() {
        let dir = temp("fresh");
        std::fs::write(dir.join("note.md"), "old\n").unwrap();
        let p = project(&dir);
        let first = attach(&p, &dir.join("history"), "@note.md").unwrap();
        assert!(first.text.contains("\nold\n"));
        // Same size and possibly the same mtime: there is no cache, the file is re-read.
        std::fs::write(dir.join("note.md"), "new\n").unwrap();
        let second = attach(&p, &dir.join("history"), "@note.md").unwrap();
        assert!(second.text.contains("\nnew\n") && !second.text.contains("\nold\n"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
