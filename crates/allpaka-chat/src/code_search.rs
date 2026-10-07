//! Bounded literal source search. Declaration context is a heuristic, not an AST.
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub(crate) fn search(root: &Path, args: &Value, excluded: Option<&Path>) -> Result<Value> {
    let query = args["query"].as_str().context("query is required")?;
    if query.trim().is_empty() || query.len() > 1000 {
        bail!("query must contain 1–1000 bytes");
    }
    let limit = args
        .get("limit")
        .map(|v| v.as_u64().context("limit must be an integer"))
        .transpose()?
        .unwrap_or(50);
    if !(1..=100).contains(&limit) {
        bail!("limit must be 1–100");
    }
    let start = super::tools::resolve(
        root,
        args["path"].as_str().context("path is required")?,
        false,
    )?;
    let mut pending = vec![start];
    let mut matches = Vec::new();
    let mut scanned = 0usize;
    let mut visited = 0usize;
    let mut bytes = 0u64;
    let mut skipped = 0usize;
    let mut truncated = false;
    while let Some(path) = pending.pop() {
        visited += 1;
        if visited > 20_000 {
            truncated = true;
            break;
        }
        if excluded.is_some_and(|p| path.starts_with(p)) {
            continue;
        }
        let relative = path.strip_prefix(root)?;
        // Apply the same hidden-path and symlink contract to every descendant.
        if super::tools::resolve(root, &relative.to_string_lossy(), false).is_err() {
            skipped += 1;
            continue;
        }
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.is_dir() {
            let mut children = Vec::new();
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                if entry.file_name().to_string_lossy().starts_with('.') {
                    continue;
                }
                if pending.len() + children.len() >= 20_000 {
                    truncated = true;
                    break;
                }
                children.push(entry.path());
            }
            children.sort();
            pending.extend(children.into_iter().rev());
            continue;
        }
        if !metadata.is_file() || metadata.len() > 8 * 1024 * 1024 {
            skipped += 1;
            continue;
        }
        if bytes + metadata.len() > 32 * 1024 * 1024 {
            truncated = true;
            break;
        }
        let content = match super::tools::read_text(&path, 8 * 1024 * 1024) {
            Ok(text) if !text.contains('\0') => text,
            _ => {
                skipped += 1;
                continue;
            }
        };
        bytes += content.len() as u64;
        scanned += 1;
        let mut declaration: Option<(usize, String)> = None;
        for (index, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if declaration_line(trimmed) {
                declaration = Some((index + 1, trimmed.chars().take(240).collect()));
            }
            let Some(position) = line.find(query) else {
                continue;
            };
            if matches.len() == limit as usize {
                truncated = true;
                break;
            }
            let text = excerpt(line, position, query);
            let text_truncated = text.len() < line.len();
            matches.push(json!({"path":relative,"line":index+1,"text":text,"preceding_declaration":declaration.as_ref().map(|(line,text)| json!({"line":line,"text":text})),"text_truncated":text_truncated}));
        }
        if matches.len() == limit as usize {
            truncated |= !pending.is_empty();
            break;
        }
    }
    Ok(
        json!({"matches":matches,"truncated":truncated,"files_scanned":scanned,"bytes_scanned":bytes,"files_skipped":skipped,"context_kind":"preceding declaration heuristic; not guaranteed enclosing scope"}),
    )
}
fn excerpt(line: &str, position: usize, query: &str) -> String {
    let start = line[..position]
        .char_indices()
        .rev()
        .nth(80)
        .map_or(0, |(offset, _)| offset);
    let tail = &line[position..];
    let end = position
        + tail
            .char_indices()
            .nth(query.chars().count() + 240)
            .map_or(tail.len(), |(offset, _)| offset);
    line[start..end].to_owned()
}
fn declaration_line(line: &str) -> bool {
    let line = line.strip_prefix("pub ").unwrap_or(line);
    let line = line.strip_prefix("async ").unwrap_or(line);
    [
        "fn ",
        "struct ",
        "enum ",
        "impl ",
        "trait ",
        "def ",
        "class ",
        "function ",
        "export function ",
        "export class ",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_unicode_line_excerpt_contains_the_match() {
        let line = format!("{}needle{}", "я".repeat(1000), "я".repeat(1000));
        let selected = excerpt(&line, line.find("needle").unwrap(), "needle");
        assert!(selected.contains("needle"));
        assert!(selected.chars().count() < 400);
    }
    #[test]
    fn bounded_search_finds_symbols_and_excludes_private_paths() {
        let root = std::env::temp_dir().join(format!(
            "allpaka-code-search-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("storage")).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(
            root.join("source.rs"),
            "fn example() {\n    // ключ\n    // ключ\n}\n",
        )
        .unwrap();
        std::fs::write(root.join(".secret"), "ключ").unwrap();
        std::fs::write(root.join("storage/history"), "ключ").unwrap();
        std::fs::write(root.join("binary"), b"\0key").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("source.rs"), root.join("link.rs")).unwrap();
        let found = search(
            &root,
            &json!({"path":".","query":"ключ"}),
            Some(&root.join("storage")),
        )
        .unwrap();
        assert_eq!(found["matches"].as_array().unwrap().len(), 2);
        assert_eq!(found["matches"][0]["path"], "source.rs");
        assert_eq!(found["matches"][0]["line"], 2);
        assert_eq!(found["matches"][0]["preceding_declaration"]["line"], 1);
        let found = search(
            &root,
            &json!({"path":"source.rs","query":"ключ","limit":1}),
            None,
        )
        .unwrap();
        assert_eq!(found["matches"].as_array().unwrap().len(), 1);
        assert_eq!(found["truncated"], true);
        for path in ["../outside", ".secret"] {
            assert!(search(&root, &json!({"path":path,"query":"ключ"}), None).is_err());
        }
        assert!(search(&root, &json!({"path":".","query":"","limit":0}), None).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
