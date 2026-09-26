use crate::types::{Mode, PlanItem, Settings};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub fn schemas(settings: &Settings, cmd_enabled: bool) -> Vec<Value> {
    let mut list = vec![
        schema(
            "list_files",
            "List one workspace directory (up to 200 entries).",
            json!({"path":{"type":"string"}}),
            json!(["path"]),
        ),
        schema(
            "read_file",
            "Read a UTF-8 file (up to 8 MiB) with optional inclusive 1-based start_line/end_line. Output at most 128 KiB.",
            json!({"path":{"type":"string"},"start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1}}),
            json!(["path"]),
        ),
        schema(
            "set_plan",
            "Publish or update the task plan. Status is pending, in_progress, or completed.",
            json!({"steps":{"type":"array","maxItems":30,"items":{"type":"object","properties":{"title":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["title","status"],"additionalProperties":false}}}),
            json!(["steps"]),
        ),
    ];
    if settings.mode == Mode::Chat {
        list.truncate(2);
    }
    if matches!(settings.mode, Mode::Auto | Mode::Goal) && settings.allow_writes {
        list.push(schema("edit_file", "Replace one exact, unique non-empty fragment in an existing UTF-8 file. Fails if absent or ambiguous. Returns the actual diff.", json!({"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}}),json!(["path","old_text","new_text"])));
        list.push(schema("write_file", "Create or replace a UTF-8 file inside the workspace. Read existing content first. Cannot access hidden paths or leave the workspace.", json!({"path":{"type":"string"},"content":{"type":"string"}}),json!(["path","content"])));
    }
    if matches!(settings.mode, Mode::Auto | Mode::Goal) && cmd_enabled {
        list.push(schema("run_command", "Run a shell command in the project root directory. Returns stdout, stderr and exit code. Timeout 30s by default (max 120s). Output truncated at 128 KiB.", json!({"command":{"type":"string"},"timeout":{"type":"integer","minimum":1,"maximum":120}}),json!(["command"])));
    }
    list
}
fn schema(name: &str, description: &str, properties: Value, required: Value) -> Value {
    json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}}})
}
fn resolve(root: &Path, path: &str, write: bool) -> Result<PathBuf> {
    let relative = Path::new(path);
    for c in relative.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(s) if !s.to_string_lossy().starts_with('.') => {}
            _ => bail!("Only relative, non-hidden workspace paths are allowed"),
        }
    }
    let candidate = root.join(relative);
    let resolved = if candidate.exists() {
        candidate.canonicalize()?
    } else if write {
        let parent = candidate
            .parent()
            .context("Missing parent")?
            .canonicalize()?;
        parent.join(candidate.file_name().context("Missing file name")?)
    } else {
        bail!("File not found");
    };
    if !resolved.starts_with(root) {
        bail!("Path escapes workspace");
    }
    // Reject symlinks, including dangling links and links into the workspace.
    let mut current = root.to_path_buf();
    for c in relative.components() {
        current.push(c);
        if fs::symlink_metadata(&current).is_ok_and(|m| m.file_type().is_symlink()) {
            bail!("Symlinks are not available to chat tools");
        }
    }
    Ok(resolved)
}
pub fn execute(root: &Path, settings: &Settings, name: &str, args: &Value) -> Result<Value> {
    match name {
        "list_files" => {
            let path = resolve(
                root,
                args["path"].as_str().context("path is required")?,
                false,
            )?;
            let mut entries = vec![];
            for entry in fs::read_dir(path)?.take(201) {
                let e = entry?;
                let name = e.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    entries.push(json!({"name":name,"directory":e.file_type()?.is_dir()}));
                }
            }
            entries.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_owned());
            let truncated = entries.len() > 200;
            entries.truncate(200);
            Ok(json!({"entries":entries,"truncated":truncated}))
        }
        "read_file" => {
            let path = resolve(
                root,
                args["path"].as_str().context("path is required")?,
                false,
            )?;
            if !fs::metadata(&path)?.is_file() {
                bail!("Not a regular file");
            }
            let content = read_text(&path, 8 * 1024 * 1024)?;
            let lines: Vec<&str> = content.split_inclusive('\n').collect();
            let start = line_arg(args, "start_line")?.unwrap_or(1);
            let end = line_arg(args, "end_line")?.unwrap_or(lines.len().max(1));
            if end < start || start > lines.len().max(1) {
                bail!("Invalid line range for {} lines", lines.len());
            }
            let selected = lines
                .get(start - 1..end.min(lines.len()))
                .unwrap_or(&[])
                .concat();
            if selected.len() > 131072 {
                bail!("Output exceeds 128 KiB; request a smaller line range");
            }
            Ok(
                json!({"content":selected,"start_line":start,"end_line":end.min(lines.len()),"total_lines":lines.len()}),
            )
        }
        "edit_file" => {
            if !matches!(settings.mode, Mode::Auto | Mode::Goal) || !settings.allow_writes {
                bail!("Writing requires Auto or Goal mode and enabled workspace writes");
            }
            let name = args["path"].as_str().context("path is required")?;
            let path = resolve(root, name, false)?;
            let before = read_text(&path, 131072)?;
            let old = args["old_text"].as_str().context("old_text is required")?;
            let new = args["new_text"].as_str().context("new_text is required")?;
            if old.is_empty() || before.matches(old).count() != 1 {
                bail!("old_text must match exactly once; read the current file and include more context");
            }
            let after = before.replacen(old, new, 1);
            if after.len() > 131072 {
                bail!("Write exceeds 128 KiB");
            }
            fs::write(&path, &after)?;
            Ok(json!({"written":name,"bytes":after.len(),"diff":unified_diff(name,&before,&after)}))
        }

        "write_file" => {
            if !matches!(settings.mode, Mode::Auto | Mode::Goal) || !settings.allow_writes {
                bail!("Writing requires Auto or Goal mode and enabled workspace writes");
            }
            let path = resolve(
                root,
                args["path"].as_str().context("path is required")?,
                true,
            )?;
            let content = args["content"].as_str().context("content is required")?;
            if content.len() > 131072 {
                bail!("Write exceeds 128 KiB");
            }
            if path.exists() && !path.is_file() {
                bail!("Not a regular file");
            }
            let before = if path.exists() {
                read_text(&path, 131072)?
            } else {
                String::new()
            };
            fs::write(&path, content)?;
            Ok(
                json!({"written":args["path"],"bytes":content.len(),"diff":unified_diff(args["path"].as_str().unwrap(),&before,content)}),
            )
        }
        _ => bail!("Unknown tool: {name}"),
    }
}
pub fn writes(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file")
}
/// Tools that only inspect the connected context. Swarm members get these.
pub fn reads(name: &str) -> bool {
    matches!(name, "list_files" | "read_file")
}
fn line_arg(args: &Value, key: &str) -> Result<Option<usize>> {
    match args.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(
            v.as_u64()
                .filter(|n| *n > 0 && *n <= usize::MAX as u64)
                .context("Line numbers must be positive integers")? as usize,
        )),
    }
}
fn read_text(path: &Path, max: usize) -> Result<String> {
    if !fs::metadata(path)?.is_file() {
        bail!("Not a regular file");
    }
    let f = fs::File::open(path)?;
    let mut bytes = Vec::new();
    f.take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        bail!("File exceeds {} bytes", max);
    }
    String::from_utf8(bytes).context("Not a UTF-8 text file")
}
fn unified_diff(path: &str, before: &str, after: &str) -> String {
    if before == after {
        return String::new();
    }
    let a: Vec<_> = before.split_inclusive('\n').collect();
    let b: Vec<_> = after.split_inclusive('\n').collect();
    let prefix = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suffix = a[prefix..]
        .iter()
        .rev()
        .zip(b[prefix..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let start = prefix.saturating_sub(3);
    let a_end = (a.len() - suffix + 3).min(a.len());
    let b_end = (b.len() - suffix + 3).min(b.len());
    let mut out = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -{},{} +{},{} @@\n",
        if a_end == start { start } else { start + 1 },
        a_end - start,
        if b_end == start { start } else { start + 1 },
        b_end - start
    );
    let mut emit = |sign: char, line: &str| {
        out.push(sign);
        out.push_str(line);
        if !line.ends_with('\n') {
            out.push_str("\n\\ No newline at end of file\n");
        }
    };
    for line in &a[start..prefix] {
        emit(' ', line);
    }
    for line in &a[prefix..a.len() - suffix] {
        emit('-', line);
    }
    for line in &b[prefix..b.len() - suffix] {
        emit('+', line);
    }
    for line in &a[a.len() - suffix..a_end] {
        emit(' ', line);
    }
    out
}
pub fn parse_plan(args: &Value) -> Result<Vec<PlanItem>> {
    let items: Vec<PlanItem> = serde_json::from_value(args["steps"].clone())?;
    if items.is_empty()
        || items.len() > 30
        || items.iter().any(|s| {
            s.title.trim().is_empty()
                || s.title.len() > 500
                || !["pending", "in_progress", "completed"].contains(&s.status.as_str())
        })
    {
        bail!("Invalid plan");
    }
    Ok(items)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_enforces_read_only_even_with_write_flag() {
        let s = Settings {
            verbosity: crate::types::Verbosity::Normal,
            project_id: "default".into(),
            provider: "test".into(),
            model: "test".into(),
            mode: Mode::Plan,
            max_steps: 5,
            max_output_tokens: 8192,
            allow_writes: true,
            swarm: Default::default(),
            auto_compact: true,
            compact_threshold: 24000,
            json_mode: false,
        };
        assert_eq!(schemas(&s, false).len(), 3);
        assert!(execute(
            Path::new("/tmp"),
            &s,
            "write_file",
            &json!({"path":"should-not-exist","content":"bad"})
        )
        .is_err());
    }
    #[test]
    fn ranges_and_precise_edits_preserve_surrounding_text() {
        let root = std::env::temp_dir().join(format!(
            "studio-edit-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        fs::write(root.join("sample.txt"), "first\nunique\nlast\n").unwrap();
        let mut settings: Settings = serde_json::from_value(
            json!({"provider":"local","model":"test","mode":"auto","allow_writes":true}),
        )
        .unwrap();
        let range = execute(
            &root,
            &settings,
            "read_file",
            &json!({"path":"sample.txt","start_line":2,"end_line":2}),
        )
        .unwrap();
        assert_eq!(range["content"], "unique\n");
        assert_eq!(range["total_lines"], 3);
        assert!(execute(
            &root,
            &settings,
            "read_file",
            &json!({"path":"sample.txt","start_line":0})
        )
        .is_err());
        let args = json!({"path":"sample.txt","old_text":"unique","new_text":"changed"});
        let result = execute(&root, &settings, "edit_file", &args).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("sample.txt")).unwrap(),
            "first\nchanged\nlast\n"
        );
        assert!(result["diff"]
            .as_str()
            .unwrap()
            .contains("-unique\n+changed\n"));
        assert!(execute(&root, &settings, "edit_file", &args).is_err());
        fs::write(root.join("sample.txt"), "same same").unwrap();
        assert!(execute(
            &root,
            &settings,
            "edit_file",
            &json!({"path":"sample.txt","old_text":"same","new_text":"x"})
        )
        .is_err());
        settings.mode = Mode::Plan;
        assert!(execute(&root, &settings, "edit_file", &args).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn diff_handles_empty_files_and_missing_newlines() {
        assert_eq!(
            unified_diff("x", "", "new\n"),
            "--- a/x\n+++ b/x\n@@ -0,0 +1,1 @@\n+new\n"
        );
        assert_eq!(
            unified_diff("x", "old\n", ""),
            "--- a/x\n+++ b/x\n@@ -1,1 +0,0 @@\n-old\n"
        );
        assert!(unified_diff("x", "old", "new")
            .contains("-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"));
        assert!(unified_diff("x", "same", "same").is_empty());
    }
    #[test]
    fn traversal_and_hidden_paths_are_rejected() {
        for p in [
            "../private",
            "/etc/passwd",
            ".env",
            "src/../../x",
            "src/.git/config",
        ] {
            assert!(resolve(Path::new("/tmp"), p, false).is_err(), "{p}");
        }
    }
    #[test]
    fn swarm_members_are_read_only_even_with_write_flag() {
        let mut s: Settings = serde_json::from_value(json!({
            "provider": "test",
            "model": "test",
            "mode": "swarm",
            "allow_writes": true
        }))
        .unwrap();
        assert!(matches!(s.mode, Mode::Swarm));
        assert_eq!(schemas(&s, false).len(), 3);
        assert!(execute(
            Path::new("/tmp"),
            &s,
            "write_file",
            &json!({"path": "should-not-exist", "content": "bad"})
        )
        .is_err());
        s.mode = Mode::Chat;
        assert_eq!(schemas(&s, false).len(), 2);
    }
}
