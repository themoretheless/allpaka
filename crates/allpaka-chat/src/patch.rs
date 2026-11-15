//! Reversible file edits (`apply_patch`), glob discovery (`glob_files`),
//! persistent per-session checkpoints with undo, and session working directory control.

use crate::context::{Project, Root};
use crate::ext::{get_session, project_for_session, resolve_project_root};
use crate::storage;
use crate::tools;
use crate::{bad_request, internal, not_found, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const MAX_CHECKPOINTS_PER_SESSION: usize = 100;
const MAX_SNAPSHOT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_GLOB_LIMIT: usize = 1000;
const DEFAULT_GLOB_LIMIT: usize = 200;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FileSnapshot {
    pub path: String,
    pub root_alias: String,
    pub rel_path: String,
    pub existed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_b64: Option<String>,
    #[serde(default)]
    pub byte_len: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CheckpointSummary {
    pub id: String,
    pub created_at_ms: u64,
    pub tool: String,
    pub summary: String,
    pub files: Vec<String>,
    #[serde(default)]
    pub reverted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverted_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CheckpointRecord {
    pub id: String,
    pub created_at_ms: u64,
    pub tool: String,
    pub summary: String,
    pub files: Vec<FileSnapshot>,
    #[serde(default)]
    pub reverted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverted_at_ms: Option<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct CheckpointManager {
    seq: Arc<AtomicU64>,
    io_lock: Arc<Mutex<()>>,
}

fn session_dir(data: &Path, session_id: &str) -> Result<PathBuf> {
    if !storage::valid_id(session_id) {
        bail!("Invalid session ID");
    }
    Ok(data.join("checkpoints").join(session_id))
}

impl CheckpointManager {
    /// Snapshot pre-edit file states before `write_file`, `edit_file`, or `apply_patch`.
    /// Each tuple in `targets` is `(display_path, root_alias, rel_path, abs_path)`.
    pub(crate) fn capture_before_edit(
        &self,
        data: &Path,
        session_id: &str,
        tool: &str,
        summary: &str,
        targets: &[(String, String, String, PathBuf)],
    ) -> Result<String> {
        let dir = session_dir(data, session_id)?;
        let _guard = self.io_lock.lock().unwrap();
        std::fs::create_dir_all(&dir)?;

        let mut seen = BTreeSet::new();
        let mut snapshots = Vec::new();
        let mut files = Vec::new();

        for (display_path, root_alias, rel_path, abs_path) in targets {
            let key = abs_path.to_string_lossy().into_owned();
            if !seen.insert(key) {
                continue;
            }
            let snap = match std::fs::symlink_metadata(abs_path) {
                Ok(meta) => {
                    if meta.file_type().is_symlink() {
                        bail!("Checkpoints refuse symbolic links: {}", abs_path.display());
                    }
                    if meta.len() > MAX_SNAPSHOT_BYTES {
                        bail!(
                            "File {} exceeds {MAX_SNAPSHOT_BYTES}-byte checkpoint limit",
                            display_path
                        );
                    }
                    let bytes = storage::read_bounded_bytes(abs_path, MAX_SNAPSHOT_BYTES as usize)?;
                    let sha = storage::hex_sha256(&bytes);
                    let encoded = storage::base64_encode(&bytes);
                    FileSnapshot {
                        path: display_path.clone(),
                        root_alias: root_alias.clone(),
                        rel_path: rel_path.clone(),
                        existed: true,
                        content_b64: Some(encoded),
                        byte_len: bytes.len(),
                        sha256: Some(sha),
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => FileSnapshot {
                    path: display_path.clone(),
                    root_alias: root_alias.clone(),
                    rel_path: rel_path.clone(),
                    existed: false,
                    content_b64: None,
                    byte_len: 0,
                    sha256: None,
                },
                Err(err) => return Err(err.into()),
            };
            files.push(display_path.clone());
            snapshots.push(snap);
        }

        let now = storage::now_ms();
        let n = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let id = format!("cp-{now}-{n}");
        let record = CheckpointRecord {
            id: id.clone(),
            created_at_ms: now,
            tool: tool.to_owned(),
            summary: summary.to_owned(),
            files: snapshots,
            reverted: false,
            reverted_at_ms: None,
        };
        let record_path = dir.join(format!("{id}.json"));
        storage::atomic_write_bytes(&record_path, &serde_json::to_vec_pretty(&record)?)?;

        let index_path = dir.join("index.json");
        let mut summaries = self.read_index_locked(&index_path)?;
        summaries.push(CheckpointSummary {
            id: id.clone(),
            created_at_ms: now,
            tool: tool.to_owned(),
            summary: summary.to_owned(),
            files,
            reverted: false,
            reverted_at_ms: None,
        });
        while summaries.len() > MAX_CHECKPOINTS_PER_SESSION {
            let evicted = summaries.remove(0);
            let _ = std::fs::remove_file(dir.join(format!("{}.json", evicted.id)));
        }
        storage::atomic_write_bytes(&index_path, &serde_json::to_vec_pretty(&summaries)?)?;
        Ok(id)
    }

    pub(crate) fn list(&self, data: &Path, session_id: &str) -> Result<Vec<CheckpointSummary>> {
        let dir = session_dir(data, session_id)?;
        let _guard = self.io_lock.lock().unwrap();
        self.read_index_locked(&dir.join("index.json"))
    }

    pub(crate) fn undo(
        &self,
        data: &Path,
        project: &Project,
        session_id: &str,
        checkpoint_id: &str,
    ) -> Result<Value> {
        if !storage::valid_id(checkpoint_id) {
            bail!("Invalid checkpoint ID");
        }
        let dir = session_dir(data, session_id)?;
        let _guard = self.io_lock.lock().unwrap();
        let record_path = dir.join(format!("{checkpoint_id}.json"));
        if !record_path.is_file() {
            bail!("Checkpoint `{checkpoint_id}` not found");
        }
        let bytes = storage::read_bounded_bytes(&record_path, 32 * 1024 * 1024)?;
        let mut record: CheckpointRecord = serde_json::from_slice(&bytes)?;
        if record.reverted {
            bail!("Checkpoint `{checkpoint_id}` has already been reverted");
        }

        // Pre-validate every restore target before mutating any file on disk.
        struct RestoreOp<'a> {
            root: &'a Root,
            rel_path: String,
            target: PathBuf,
            existed: bool,
            bytes: Vec<u8>,
            display_path: String,
        }
        let mut ops = Vec::with_capacity(record.files.len());
        let mut lock_roots = BTreeSet::new();
        for snap in &record.files {
            let root = project
                .roots
                .iter()
                .find(|r| r.alias == snap.root_alias)
                .or_else(|| (project.roots.len() == 1).then(|| &project.roots[0]))
                .with_context(|| {
                    format!(
                        "Project root `{}` is unavailable to undo `{}`",
                        snap.root_alias, snap.path
                    )
                })?;
            if !root.writable {
                bail!("Context root `{}` is read-only", root.alias);
            }
            let target = tools::resolve(&root.path, &snap.rel_path, true)?;
            let canonical = target.canonicalize().unwrap_or_else(|_| target.clone());
            if canonical.starts_with(data) {
                bail!("Conversation storage cannot be modified by undo");
            }
            let content = if snap.existed {
                let encoded = snap
                    .content_b64
                    .as_deref()
                    .context("Missing snapshot content")?;
                storage::base64_decode(encoded)?
            } else {
                Vec::new()
            };
            lock_roots.insert(root.path.clone());
            ops.push(RestoreOp {
                root,
                rel_path: snap.rel_path.clone(),
                target,
                existed: snap.existed,
                bytes: content,
                display_path: snap.path.clone(),
            });
        }

        let locks: Vec<_> = lock_roots.iter().map(|r| tools::mutation_lock(r)).collect();
        let _held: Vec<_> = locks.iter().map(|l| l.lock().unwrap()).collect();

        let mut restored = Vec::new();
        for op in ops {
            if op.existed {
                tools::create_parents(&op.root.path, &op.rel_path)?;
                storage::atomic_write_bytes(&op.target, &op.bytes)?;
                restored.push(json!({ "path": op.display_path, "action": "restored" }));
            } else {
                match std::fs::remove_file(&op.target) {
                    Ok(()) => {
                        restored.push(json!({ "path": op.display_path, "action": "removed" }));
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                        restored
                            .push(json!({ "path": op.display_path, "action": "already_absent" }));
                    }
                    Err(err) => return Err(err.into()),
                }
            }
        }

        let now = storage::now_ms();
        record.reverted = true;
        record.reverted_at_ms = Some(now);
        storage::atomic_write_bytes(&record_path, &serde_json::to_vec_pretty(&record)?)?;

        let index_path = dir.join("index.json");
        let mut summaries = self.read_index_locked(&index_path)?;
        if let Some(entry) = summaries.iter_mut().find(|s| s.id == checkpoint_id) {
            entry.reverted = true;
            entry.reverted_at_ms = Some(now);
        }
        storage::atomic_write_bytes(&index_path, &serde_json::to_vec_pretty(&summaries)?)?;

        Ok(json!({
            "reverted": true,
            "checkpoint_id": checkpoint_id,
            "reverted_at_ms": now,
            "restored_files": restored,
        }))
    }

    pub(crate) fn delete_session(&self, data: &Path, session_id: &str) -> Result<()> {
        let dir = session_dir(data, session_id)?;
        let _guard = self.io_lock.lock().unwrap();
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    fn read_index_locked(&self, index_path: &Path) -> Result<Vec<CheckpointSummary>> {
        if !index_path.is_file() {
            return Ok(Vec::new());
        }
        let bytes = storage::read_bounded_bytes(index_path, 4 * 1024 * 1024)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

pub(crate) fn glob_files_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files matching a glob pattern (`*`, `?`, `**`, `{a,b}`) within the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern such as `**/*.rs` or `src/{lib,main}.rs`" },
                    "path": { "type": "string", "description": "Optional base folder or context-alias/subpath (defaults to working directory or `.`)" },
                    "max_results": { "type": "integer", "description": "Maximum results (default 200, max 1000)" }
                },
                "required": ["pattern"]
            }
        }
    })
}

pub(crate) fn apply_patch_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a multi-file patch all-or-nothing (supports `*** Begin Patch` and unified `diff`/`---`/`+++` format) and record a reversible checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": { "type": "string", "description": "Patch content in `*** Begin Patch` or unified diff format" }
                },
                "required": ["patch"]
            }
        }
    })
}

pub(crate) fn working_directory_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "working_directory",
            "description": "Get, set, or reset the session's workspace-relative working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["get", "set", "reset"] },
                    "path": { "type": "string", "description": "Workspace-relative directory path when action is `set`" }
                }
            }
        }
    })
}

fn expand_glob_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_owned()];
    };
    let mut depth = 0usize;
    let mut close = None;
    for (idx, ch) in pattern[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    close = Some(open + idx);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(end) = close else {
        return vec![pattern.to_owned()];
    };
    let prefix = &pattern[..open];
    let inner = &pattern[open + 1..end];
    let suffix = &pattern[end + 1..];
    let mut items = Vec::new();
    let mut cur = String::new();
    let mut inner_depth = 0usize;
    for ch in inner.chars() {
        match ch {
            '{' => {
                inner_depth += 1;
                cur.push(ch);
            }
            '}' => {
                inner_depth = inner_depth.saturating_sub(1);
                cur.push(ch);
            }
            ',' if inner_depth == 0 => {
                items.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    items.push(cur);
    let mut out = Vec::new();
    for item in items {
        for rest in expand_glob_braces(&format!("{prefix}{item}{suffix}")) {
            out.push(rest);
            if out.len() >= 64 {
                return out;
            }
        }
    }
    out
}

fn match_char_class(pat: &[char], ch: char) -> Option<usize> {
    if pat.first() != Some(&'[') {
        return None;
    }
    let mut i = 1;
    let negated = pat.get(i).is_some_and(|c| *c == '!' || *c == '^');
    if negated {
        i += 1;
    }
    let mut matched = false;
    while i < pat.len() {
        if pat[i] == ']' {
            return (matched ^ negated).then_some(i + 1);
        }
        if i + 2 < pat.len() && pat[i + 1] == '-' && pat[i + 2] != ']' {
            if ch >= pat[i] && ch <= pat[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if pat[i] == ch {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

fn match_segment(pat: &[char], text: &[char], depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    if pat.is_empty() {
        return text.is_empty();
    }
    match pat[0] {
        '*' => {
            let mut next = 1;
            while pat.get(next) == Some(&'*') {
                next += 1;
            }
            for cut in 0..=text.len() {
                if match_segment(&pat[next..], &text[cut..], depth + 1) {
                    return true;
                }
            }
            false
        }
        '?' => !text.is_empty() && match_segment(&pat[1..], &text[1..], depth + 1),
        '[' => {
            if text.is_empty() {
                return false;
            }
            if let Some(consumed) = match_char_class(pat, text[0]) {
                return match_segment(&pat[consumed..], &text[1..], depth + 1);
            }
            pat[0] == text[0] && match_segment(&pat[1..], &text[1..], depth + 1)
        }
        c => !text.is_empty() && c == text[0] && match_segment(&pat[1..], &text[1..], depth + 1),
    }
}

fn match_path_segments(pat_segs: &[&str], path_segs: &[&str], depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    if pat_segs.is_empty() {
        return path_segs.is_empty();
    }
    if pat_segs[0] == "**" {
        if match_path_segments(&pat_segs[1..], path_segs, depth + 1) {
            return true;
        }
        return !path_segs.is_empty() && match_path_segments(pat_segs, &path_segs[1..], depth + 1);
    }
    if path_segs.is_empty() {
        return false;
    }
    let p_chars: Vec<char> = pat_segs[0].chars().collect();
    let s_chars: Vec<char> = path_segs[0].chars().collect();
    match_segment(&p_chars, &s_chars, 0)
        && match_path_segments(&pat_segs[1..], &path_segs[1..], depth + 1)
}

fn glob_matches(patterns: &[Vec<String>], rel_posix: &str) -> bool {
    let segs: Vec<&str> = rel_posix.split('/').filter(|s| !s.is_empty()).collect();
    patterns.iter().any(|p| {
        let p_refs: Vec<&str> = p.iter().map(String::as_str).collect();
        match_path_segments(&p_refs, &segs, 0)
    })
}

fn should_descend(name: &str, patterns: &[String]) -> bool {
    if matches!(name, ".git" | "target" | "node_modules") {
        return patterns.iter().any(|p| p.contains(name));
    }
    if name.starts_with('.') {
        return patterns
            .iter()
            .any(|p| p.split('/').any(|seg| seg.starts_with('.')));
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn walk_glob_dir(
    root: &Path,
    cur: &Path,
    data_dir: &Path,
    raw_patterns: &[String],
    compiled: &[Vec<String>],
    out: &mut Vec<String>,
    prefix_alias: Option<&str>,
    truncated: &mut bool,
    limit: usize,
) -> Result<()> {
    let mut entries: Vec<_> = match std::fs::read_dir(cur) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return Ok(()),
    };
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if out.len() >= limit {
            *truncated = true;
            return Ok(());
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let abs = entry.path();
        if abs == data_dir || abs.starts_with(data_dir) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !should_descend(&name, raw_patterns) {
            continue;
        }
        if ft.is_dir() {
            walk_glob_dir(
                root,
                &abs,
                data_dir,
                raw_patterns,
                compiled,
                out,
                prefix_alias,
                truncated,
                limit,
            )?;
        } else if ft.is_file() {
            let Ok(rel) = abs.strip_prefix(root) else {
                continue;
            };
            let rel_posix = rel
                .iter()
                .map(|c| c.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            if glob_matches(compiled, &rel_posix) {
                let display = match prefix_alias {
                    Some(alias) if !alias.is_empty() => format!("{alias}/{rel_posix}"),
                    _ => rel_posix,
                };
                out.push(display);
            }
        }
    }
    Ok(())
}

pub(crate) fn glob_files(
    project: &Project,
    working_dir: Option<&str>,
    data_dir: &Path,
    args: &Value,
) -> Result<Value> {
    let pattern = args["pattern"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .context("pattern is required")?;
    let max_results = args
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_GLOB_LIMIT as u64)
        .clamp(1, MAX_GLOB_LIMIT as u64) as usize;
    let base_arg = args
        .get("path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(working_dir)
        .unwrap_or(".");

    let expanded = expand_glob_braces(pattern);
    let compiled: Vec<Vec<String>> = expanded
        .iter()
        .map(|p| {
            p.trim_start_matches("./")
                .split('/')
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .collect();

    let mut files = Vec::new();
    let mut truncated = false;

    if base_arg == "." && project.roots.len() > 1 {
        for root in &project.roots {
            if root.ssh.is_some() {
                continue;
            }
            if files.len() >= max_results {
                truncated = true;
                break;
            }
            walk_glob_dir(
                &root.path,
                &root.path,
                data_dir,
                &expanded,
                &compiled,
                &mut files,
                Some(&root.alias),
                &mut truncated,
                max_results,
            )?;
        }
    } else {
        let (root, rel) = resolve_project_root(project, base_arg, false)?;
        let start = tools::resolve(&root.path, &rel, false)?;
        let canonical = start.canonicalize().unwrap_or_else(|_| start.clone());
        if canonical.starts_with(data_dir) {
            bail!("Conversation storage is not available to glob_files");
        }
        let prefix = if project.roots.len() > 1 {
            Some(root.alias.as_str())
        } else {
            None
        };
        if start.is_dir() {
            walk_glob_dir(
                &root.path,
                &start,
                data_dir,
                &expanded,
                &compiled,
                &mut files,
                prefix,
                &mut truncated,
                max_results,
            )?;
        }
    }

    Ok(json!({
        "pattern": pattern,
        "count": files.len(),
        "truncated": truncated,
        "files": files,
    }))
}

#[derive(Debug)]
struct Hunk {
    old_start_hint: Option<usize>,
    lines: Vec<(char, String)>, // ' ', '-', '+'
}

#[derive(Debug)]
enum PatchOp {
    Add {
        path: String,
        content: String,
    },
    Delete {
        path: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
}

fn parse_begin_patch(raw: &str) -> Result<Vec<PatchOp>> {
    let lines: Vec<&str> = raw.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim() == "*** Begin Patch")
        .context("Missing `*** Begin Patch` header")?;
    let end = lines[start + 1..]
        .iter()
        .position(|l| l.trim() == "*** End Patch")
        .map(|off| start + 1 + off)
        .context("Missing `*** End Patch` footer")?;

    let mut ops = Vec::new();
    let mut i = start + 1;
    while i < end {
        let line = lines[i];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.trim().to_owned();
            i += 1;
            let mut body = Vec::new();
            while i < end && !lines[i].starts_with("*** ") {
                let l = lines[i];
                let content_line = l
                    .strip_prefix('+')
                    .with_context(|| format!("Add File expects `+` line near line {}", i + 1))?;
                body.push(content_line);
                i += 1;
            }
            let mut joined = body.join("\n");
            if !body.is_empty() {
                joined.push('\n');
            }
            ops.push(PatchOp::Add {
                path,
                content: joined,
            });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            ops.push(PatchOp::Delete {
                path: path.trim().to_owned(),
            });
            i += 1;
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim().to_owned();
            i += 1;
            let mut move_to = None;
            if i < end {
                if let Some(dest) = lines[i].strip_prefix("*** Move to: ") {
                    move_to = Some(dest.trim().to_owned());
                    i += 1;
                }
            }
            let mut hunks = Vec::new();
            while i < end {
                if lines[i].trim() == "*** End of File" {
                    i += 1;
                    continue;
                }
                if lines[i].starts_with("*** Add File:")
                    || lines[i].starts_with("*** Delete File:")
                    || lines[i].starts_with("*** Update File:")
                {
                    break;
                }
                if !lines[i].starts_with("@@") {
                    i += 1;
                    continue;
                }
                let hint = parse_unified_hunk_start(lines[i]);
                i += 1;
                let mut hunk_lines = Vec::new();
                while i < end && !lines[i].starts_with("@@") && !lines[i].starts_with("*** ") {
                    let l = lines[i];
                    if l == r"\ No newline at end of file" {
                        i += 1;
                        continue;
                    }
                    let mut chars = l.chars();
                    let kind = chars.next().unwrap_or(' ');
                    if !matches!(kind, ' ' | '-' | '+') {
                        bail!("Invalid patch hunk line near line {}: `{l}`", i + 1);
                    }
                    hunk_lines.push((kind, chars.as_str().to_owned()));
                    i += 1;
                }
                if !hunk_lines.is_empty() {
                    hunks.push(Hunk {
                        old_start_hint: hint,
                        lines: hunk_lines,
                    });
                }
            }
            ops.push(PatchOp::Update {
                path,
                move_to,
                hunks,
            });
        } else if line.trim().is_empty() {
            i += 1;
        } else {
            bail!("Unexpected line inside patch near line {}: `{line}`", i + 1);
        }
    }
    if ops.is_empty() {
        bail!("Patch contains no file operations");
    }
    Ok(ops)
}

fn parse_unified_hunk_start(header: &str) -> Option<usize> {
    let rest = header.strip_prefix("@@")?.trim_start();
    let minus = rest.strip_prefix('-')?;
    let num_str: String = minus.chars().take_while(|c| c.is_ascii_digit()).collect();
    let line_1based: usize = num_str.parse().ok()?;
    Some(line_1based.saturating_sub(1))
}

fn strip_diff_prefix(raw: &str) -> String {
    let trimmed = raw.trim().split('\t').next().unwrap_or("").trim();
    if let Some(stripped) = trimmed
        .strip_prefix("a/")
        .or_else(|| trimmed.strip_prefix("b/"))
    {
        stripped.to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn parse_unified_diff(raw: &str) -> Result<Vec<PatchOp>> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut ops = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if !lines[i].starts_with("--- ") {
            i += 1;
            continue;
        }
        let old_raw = lines[i].strip_prefix("--- ").unwrap().trim();
        let next = lines
            .get(i + 1)
            .and_then(|l| l.strip_prefix("+++ "))
            .context("Unified diff missing `+++` header after `---`")?
            .trim();
        i += 2;
        let mut hunks = Vec::new();
        while i < lines.len() {
            if lines[i].starts_with("diff ") || lines[i].starts_with("--- ") {
                break;
            }
            if let Some(hint) = parse_unified_hunk_start(lines[i]) {
                i += 1;
                let mut hunk_lines = Vec::new();
                while i < lines.len()
                    && !lines[i].starts_with("@@")
                    && !lines[i].starts_with("diff ")
                    && !lines[i].starts_with("--- ")
                {
                    let l = lines[i];
                    if l == r"\ No newline at end of file" {
                        i += 1;
                        continue;
                    }
                    let mut chars = l.chars();
                    let Some(kind) = chars.next() else {
                        hunk_lines.push((' ', String::new()));
                        i += 1;
                        continue;
                    };
                    if !matches!(kind, ' ' | '-' | '+') {
                        break;
                    }
                    hunk_lines.push((kind, chars.as_str().to_owned()));
                    i += 1;
                }
                hunks.push(Hunk {
                    old_start_hint: Some(hint),
                    lines: hunk_lines,
                });
            } else {
                i += 1;
            }
        }
        if old_raw == "/dev/null" {
            let path = strip_diff_prefix(next);
            let mut added = Vec::new();
            for h in hunks {
                for (k, text) in h.lines {
                    if k == '+' {
                        added.push(text);
                    }
                }
            }
            let mut content = added.join("\n");
            if !added.is_empty() {
                content.push('\n');
            }
            ops.push(PatchOp::Add { path, content });
        } else if next == "/dev/null" {
            ops.push(PatchOp::Delete {
                path: strip_diff_prefix(old_raw),
            });
        } else {
            ops.push(PatchOp::Update {
                path: strip_diff_prefix(next),
                move_to: None,
                hunks,
            });
        }
    }
    if ops.is_empty() {
        bail!(
            "Patch must contain `*** Begin Patch` ... `*** End Patch` or unified `---`/`+++` hunks"
        );
    }
    Ok(ops)
}

fn apply_hunks_to_text(path_label: &str, original: &str, hunks: &[Hunk]) -> Result<String> {
    let had_trailing_nl = original.ends_with('\n');
    let mut lines: Vec<String> = original.lines().map(ToOwned::to_owned).collect();
    let mut offset: isize = 0;

    for (h_idx, hunk) in hunks.iter().enumerate() {
        let old_seq: Vec<&str> = hunk
            .lines
            .iter()
            .filter(|(k, _)| *k == ' ' || *k == '-')
            .map(|(_, s)| s.as_str())
            .collect();
        let new_seq: Vec<String> = hunk
            .lines
            .iter()
            .filter(|(k, _)| *k == ' ' || *k == '+')
            .map(|(_, s)| s.clone())
            .collect();

        if old_seq.is_empty() {
            let pos = hunk
                .old_start_hint
                .map(|h| ((h as isize + offset).max(0) as usize).min(lines.len()))
                .unwrap_or(lines.len());
            lines.splice(pos..pos, new_seq.clone());
            offset += new_seq.len() as isize;
            continue;
        }

        let pos = if let Some(hint) = hunk.old_start_hint {
            let expected = (hint as isize + offset).max(0) as usize;
            if expected + old_seq.len() <= lines.len()
                && lines[expected..expected + old_seq.len()]
                    .iter()
                    .zip(&old_seq)
                    .all(|(a, b)| a == b)
            {
                expected
            } else {
                find_unique_match(&lines, &old_seq, path_label, h_idx)?
            }
        } else {
            find_unique_match(&lines, &old_seq, path_label, h_idx)?
        };

        let old_len = old_seq.len();
        let new_len = new_seq.len();
        lines.splice(pos..pos + old_len, new_seq);
        offset += new_len as isize - old_len as isize;
    }

    let mut out = lines.join("\n");
    if (had_trailing_nl || !lines.is_empty()) && !out.ends_with('\n') && !lines.is_empty() {
        out.push('\n');
    }
    Ok(out)
}

fn find_unique_match(
    lines: &[String],
    needle: &[&str],
    path_label: &str,
    h_idx: usize,
) -> Result<usize> {
    if needle.len() > lines.len() {
        bail!(
            "Patch hunk #{} on `{}` does not match file length",
            h_idx + 1,
            path_label
        );
    }
    let matches: Vec<usize> = (0..=lines.len() - needle.len())
        .filter(|&start| {
            lines[start..start + needle.len()]
                .iter()
                .zip(needle)
                .all(|(a, b)| a == b)
        })
        .collect();
    match matches.len() {
        1 => Ok(matches[0]),
        0 => bail!(
            "Patch hunk #{} failed to match context in `{}`",
            h_idx + 1,
            path_label
        ),
        n => bail!(
            "Patch hunk #{} matched {n} places in `{}`; provide more context",
            h_idx + 1,
            path_label
        ),
    }
}

fn resolve_with_wd(working_dir: Option<&str>, raw_path: &str) -> String {
    let trimmed = raw_path.trim().trim_start_matches("./");
    match working_dir
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != ".")
    {
        Some(wd) if !trimmed.starts_with(wd) && !trimmed.contains(':') => {
            let wd_clean = wd.trim_end_matches('/');
            format!("{wd_clean}/{trimmed}")
        }
        _ => trimmed.to_owned(),
    }
}

pub(crate) fn apply_patch(
    project: &Project,
    working_dir: Option<&str>,
    data_dir: &Path,
    checkpoints: &CheckpointManager,
    session_id: Option<&str>,
    allow_writes: bool,
    args: &Value,
) -> Result<Value> {
    if !allow_writes {
        bail!("Writes are disabled in this mode or conversation");
    }
    let raw = args["patch"].as_str().context("patch is required")?;
    let ops = if raw.contains("*** Begin Patch") {
        parse_begin_patch(raw)?
    } else {
        parse_unified_diff(raw)?
    };

    enum StagedAction<'a> {
        Write {
            display: String,
            root: &'a Root,
            rel: String,
            abs: PathBuf,
            after: String,
            before: String,
            created: bool,
        },
        Delete {
            display: String,
            abs: PathBuf,
            before: String,
        },
    }

    // Acquire root locks sorted by root path before reading/staging to eliminate TOCTOU races.
    let mut all_lock_roots = BTreeSet::new();
    for root in &project.roots {
        if root.writable {
            all_lock_roots.insert(root.path.clone());
        }
    }
    let root_locks: Vec<_> = all_lock_roots
        .iter()
        .map(|r| tools::mutation_lock(r))
        .collect();
    let _held: Vec<_> = root_locks.iter().map(|l| l.lock().unwrap()).collect();

    let mut staged: Vec<StagedAction<'_>> = Vec::new();
    let mut checkpoint_targets: Vec<(String, String, String, PathBuf)> = Vec::new();
    let mut virtual_contents: BTreeMap<PathBuf, Option<String>> = BTreeMap::new();

    for op in ops {
        match op {
            PatchOp::Add { path, content } => {
                let effective = resolve_with_wd(working_dir, &path);
                let (root, rel) = resolve_project_root(project, &effective, true)?;
                let abs = tools::resolve(&root.path, &rel, true)?;
                if abs
                    .canonicalize()
                    .unwrap_or_else(|_| abs.clone())
                    .starts_with(data_dir)
                {
                    bail!("Conversation storage cannot be modified");
                }
                if abs.exists() && virtual_contents.get(&abs).is_none_or(|o| o.is_some()) {
                    bail!("Add File target already exists: `{effective}`");
                }
                virtual_contents.insert(abs.clone(), Some(content.clone()));
                checkpoint_targets.push((
                    effective.clone(),
                    root.alias.clone(),
                    rel.clone(),
                    abs.clone(),
                ));
                staged.push(StagedAction::Write {
                    display: effective,
                    root,
                    rel,
                    abs,
                    after: content,
                    before: String::new(),
                    created: true,
                });
            }
            PatchOp::Delete { path } => {
                let effective = resolve_with_wd(working_dir, &path);
                let (root, rel) = resolve_project_root(project, &effective, true)?;
                let abs = tools::resolve(&root.path, &rel, true)?;
                if abs
                    .canonicalize()
                    .unwrap_or_else(|_| abs.clone())
                    .starts_with(data_dir)
                {
                    bail!("Conversation storage cannot be modified");
                }
                let before = match virtual_contents.get(&abs) {
                    Some(Some(prev)) => prev.clone(),
                    Some(None) => bail!("File `{effective}` was already deleted in patch"),
                    None => {
                        if !abs.is_file() {
                            bail!("Delete File target does not exist: `{effective}`");
                        }
                        tools::read_text(&abs, MAX_SNAPSHOT_BYTES as usize)?
                    }
                };
                virtual_contents.insert(abs.clone(), None);
                checkpoint_targets.push((
                    effective.clone(),
                    root.alias.clone(),
                    rel.clone(),
                    abs.clone(),
                ));
                staged.push(StagedAction::Delete {
                    display: effective,
                    abs,
                    before,
                });
            }
            PatchOp::Update {
                path,
                move_to,
                hunks,
            } => {
                let effective = resolve_with_wd(working_dir, &path);
                let (root, rel) = resolve_project_root(project, &effective, true)?;
                let abs = tools::resolve(&root.path, &rel, true)?;
                if abs
                    .canonicalize()
                    .unwrap_or_else(|_| abs.clone())
                    .starts_with(data_dir)
                {
                    bail!("Conversation storage cannot be modified");
                }
                let before = match virtual_contents.get(&abs) {
                    Some(Some(prev)) => prev.clone(),
                    Some(None) => bail!("File `{effective}` was deleted earlier in patch"),
                    None => {
                        if !abs.is_file() {
                            bail!("Update File target does not exist: `{effective}`");
                        }
                        tools::read_text(&abs, MAX_SNAPSHOT_BYTES as usize)?
                    }
                };
                let after = apply_hunks_to_text(&effective, &before, &hunks)?;
                checkpoint_targets.push((
                    effective.clone(),
                    root.alias.clone(),
                    rel.clone(),
                    abs.clone(),
                ));

                if let Some(dest_raw) = move_to {
                    let dest_effective = resolve_with_wd(working_dir, &dest_raw);
                    let (dest_root, dest_rel) =
                        resolve_project_root(project, &dest_effective, true)?;
                    let dest_abs = tools::resolve(&dest_root.path, &dest_rel, true)?;
                    if dest_abs
                        .canonicalize()
                        .unwrap_or_else(|_| dest_abs.clone())
                        .starts_with(data_dir)
                    {
                        bail!("Conversation storage cannot be modified");
                    }
                    virtual_contents.insert(abs.clone(), None);
                    virtual_contents.insert(dest_abs.clone(), Some(after.clone()));
                    checkpoint_targets.push((
                        dest_effective.clone(),
                        dest_root.alias.clone(),
                        dest_rel.clone(),
                        dest_abs.clone(),
                    ));
                    staged.push(StagedAction::Delete {
                        display: effective,
                        abs,
                        before: before.clone(),
                    });
                    staged.push(StagedAction::Write {
                        display: dest_effective,
                        root: dest_root,
                        rel: dest_rel,
                        abs: dest_abs,
                        after,
                        before: String::new(),
                        created: true,
                    });
                } else {
                    virtual_contents.insert(abs.clone(), Some(after.clone()));
                    staged.push(StagedAction::Write {
                        display: effective,
                        root,
                        rel,
                        abs,
                        after,
                        before,
                        created: false,
                    });
                }
            }
        }
    }

    let checkpoint_id = if let Some(sid) = session_id {
        let summary = format!("apply_patch ({} action(s))", staged.len());
        Some(checkpoints.capture_before_edit(
            data_dir,
            sid,
            "apply_patch",
            &summary,
            &checkpoint_targets,
        )?)
    } else {
        None
    };

    let mut summary_files = Vec::new();
    let mut diffs = Vec::new();
    for action in staged {
        match action {
            StagedAction::Write {
                display,
                root,
                rel,
                abs,
                after,
                before,
                created,
            } => {
                tools::create_parents(&root.path, &rel)?;
                storage::atomic_write_bytes(&abs, after.as_bytes())?;
                diffs.push(tools::unified_diff(&display, &before, &after));
                summary_files.push(json!({
                    "path": display,
                    "action": if created { "added" } else { "updated" },
                }));
            }
            StagedAction::Delete {
                display,
                abs,
                before,
            } => {
                std::fs::remove_file(&abs)?;
                diffs.push(tools::unified_diff(&display, &before, ""));
                summary_files.push(json!({
                    "path": display,
                    "action": "deleted",
                }));
            }
        }
    }

    Ok(json!({
        "applied": true,
        "checkpoint_id": checkpoint_id,
        "files": summary_files,
        "diff": diffs.join("\n"),
    }))
}

pub(crate) fn working_directory_tool(
    app: &App,
    s: &crate::SharedSession,
    project: &Project,
    args: &Value,
) -> Result<Value> {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("get");
    match action {
        "get" => {
            let current = s.lock().unwrap().working_dir.clone();
            Ok(json!({
                "working_dir": current.as_deref().unwrap_or("."),
                "is_default": current.is_none(),
            }))
        }
        "set" => {
            let raw = args
                .get("path")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .context("path is required when action is `set`")?;
            let (root, rel) = resolve_project_root(project, raw, false)?;
            let target = tools::resolve(&root.path, &rel, false)?;
            let canonical = target.canonicalize().unwrap_or_else(|_| target.clone());
            if canonical.starts_with(app.data.as_ref()) {
                bail!("Conversation storage cannot be selected as working directory");
            }
            let meta = std::fs::metadata(&target)
                .with_context(|| format!("Directory not found: {raw}"))?;
            if !meta.is_dir() {
                bail!("working_directory must point to a directory");
            }
            let normalized = if project.roots.len() > 1 {
                if rel == "." {
                    root.alias.clone()
                } else {
                    format!("{}/{}", root.alias, rel.trim_matches('/'))
                }
            } else {
                rel.trim_matches('/').to_owned()
            };
            let stored =
                (!normalized.is_empty() && normalized != ".").then_some(normalized.clone());
            {
                let mut state = s.lock().unwrap();
                state.working_dir = stored.clone();
                crate::save(app, &state)?;
            }
            Ok(json!({
                "working_dir": stored.as_deref().unwrap_or("."),
                "updated": true,
            }))
        }
        "reset" => {
            {
                let mut state = s.lock().unwrap();
                state.working_dir = None;
                crate::save(app, &state)?;
            }
            Ok(json!({
                "working_dir": ".",
                "updated": true,
            }))
        }
        other => bail!("Unsupported working_directory action `{other}`"),
    }
}

pub(crate) async fn list_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    crate::blocking(move || {
        let checkpoints = app.ext.checkpoints.list(&app.data, &id).map_err(internal)?;
        Ok(Json(
            json!({ "session_id": id, "checkpoints": checkpoints }),
        ))
    })
    .await
}

pub(crate) async fn undo_api(
    State(app): State<App>,
    AxumPath((id, cid)): AxumPath<(String, String)>,
) -> ApiResult<Value> {
    if !storage::valid_id(&cid) {
        return Err(bad_request("Invalid checkpoint ID"));
    }
    let shared = get_session(&app, &id)?;
    let (_, settings, project) = project_for_session(&app, &shared)?;
    if !settings.allow_writes {
        return Err(bad_request("Writes are disabled in this conversation"));
    }
    crate::blocking(move || {
        let result = app
            .ext
            .checkpoints
            .undo(&app.data, &project, &id, &cid)
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("not found") {
                    not_found(e)
                } else {
                    bad_request(e)
                }
            })?;
        Ok(Json(result))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Project, Root};

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("allpaka-patch-{label}-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn project_at(root: &Path) -> Project {
        Project {
            id: "p1".into(),
            name: "p1".into(),
            roots: vec![Root {
                alias: "workspace".into(),
                path: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
                writable: true,
                repository: false,
                ssh: None,
            }],
            instructions: String::new(),
        }
    }

    #[test]
    fn glob_matches_nested_and_skips_git() {
        let base = temp_dir("glob");
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(repo.join("src/nested")).unwrap();
        std::fs::create_dir_all(repo.join(".git/objects")).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(repo.join("src/lib.rs"), "fn a(){}").unwrap();
        std::fs::write(repo.join("src/nested/mod.rs"), "fn b(){}").unwrap();
        std::fs::write(repo.join("src/nested/mod.txt"), "hi").unwrap();
        std::fs::write(repo.join(".git/objects/abc.rs"), "skip").unwrap();

        let proj = project_at(&repo);
        let res = glob_files(&proj, None, &data, &json!({ "pattern": "**/*.rs" })).unwrap();
        let files: Vec<String> = res["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(files, vec!["src/lib.rs", "src/nested/mod.rs"]);
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn apply_patch_is_transactional_and_undo_restores_all_files() {
        let base = temp_dir("patch-undo");
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(repo.join("src/a.txt"), "one\ntwo\nthree\n").unwrap();
        std::fs::write(repo.join("src/b.txt"), "alpha\nbeta\n").unwrap();

        let proj = project_at(&repo);
        let mgr = CheckpointManager::default();

        // Failing second file leaves first file untouched!
        let bad_patch = "*** Begin Patch\n*** Update File: src/a.txt\n@@\n one\n-two\n+TWO\n three\n*** Update File: src/b.txt\n@@\n non_matching_line\n-beta\n+BETA\n*** End Patch\n";
        assert!(apply_patch(
            &proj,
            None,
            &data,
            &mgr,
            Some("sess-1"),
            true,
            &json!({ "patch": bad_patch }),
        )
        .is_err());
        assert_eq!(
            std::fs::read_to_string(repo.join("src/a.txt")).unwrap(),
            "one\ntwo\nthree\n"
        );

        // Successful multi-file patch (update + add) creates checkpoint and undo reverts both.
        let good_patch = "*** Begin Patch\n*** Update File: src/a.txt\n@@\n one\n-two\n+TWO\n three\n*** Add File: src/new.txt\n+brand new\n*** End Patch\n";
        let applied = apply_patch(
            &proj,
            None,
            &data,
            &mgr,
            Some("sess-1"),
            true,
            &json!({ "patch": good_patch }),
        )
        .unwrap();
        let cid = applied["checkpoint_id"].as_str().unwrap().to_owned();
        assert_eq!(
            std::fs::read_to_string(repo.join("src/a.txt")).unwrap(),
            "one\nTWO\nthree\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.join("src/new.txt")).unwrap(),
            "brand new\n"
        );

        let undone = mgr.undo(&data, &proj, "sess-1", &cid).unwrap();
        assert_eq!(undone["reverted"], true);
        assert_eq!(
            std::fs::read_to_string(repo.join("src/a.txt")).unwrap(),
            "one\ntwo\nthree\n"
        );
        assert!(!repo.join("src/new.txt").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn apply_patch_supports_unified_diff_format() {
        let base = temp_dir("udiff");
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(repo.join("hello.txt"), "line1\nline2\nline3\n").unwrap();

        let proj = project_at(&repo);
        let mgr = CheckpointManager::default();
        let udiff = "\
--- a/hello.txt
+++ b/hello.txt
@@ -1,3 +1,3 @@
 line1
-line2
+lineTwo
 line3
";
        apply_patch(
            &proj,
            None,
            &data,
            &mgr,
            Some("sess-u"),
            true,
            &json!({ "patch": udiff }),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(repo.join("hello.txt")).unwrap(),
            "line1\nlineTwo\nline3\n"
        );
        let _ = std::fs::remove_dir_all(base);
    }
}
