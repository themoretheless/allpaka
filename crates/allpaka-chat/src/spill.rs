//! Session-scoped spill storage for oversized tool outputs.
//!
//! When a tool response exceeds the inline token/byte budget, the full output is
//! stored atomically under `<data>/spills/<session_id>/<spill_id>.txt` and the
//! model receives a preview plus a `spill_id` it can page through via
//! `read_spill`.

use crate::{bad_request, blocking, internal, not_found, storage, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path as AxumPath, Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) const DEFAULT_INLINE_LIMIT: usize = 128 * 1024;
const MAX_SPILL_BYTES: usize = 8 * 1024 * 1024;
const MAX_SPILLS_PER_SESSION: usize = 200;
const DEFAULT_READ_LIMIT: usize = 32 * 1024;
const MAX_READ_LIMIT: usize = 128 * 1024;

static SPILL_SEQ: AtomicU64 = AtomicU64::new(1);

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "read_spill",
            "description": "Read a slice of an oversized tool output stored in a session spill file. Only spills belonging to the current conversation are accessible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spill_id": { "type": "string", "minLength": 1, "maxLength": 80 },
                    "offset": { "type": "integer", "minimum": 0 },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 131072 }
                },
                "required": ["spill_id"],
                "additionalProperties": false
            }
        }
    })
}

pub(crate) fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    index = index.min(s.len());
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub(crate) fn ceil_char_boundary(s: &str, mut index: usize) -> usize {
    index = index.min(s.len());
    while index < s.len() && !s.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn session_dir(data: &Path, session_id: &str) -> Result<PathBuf> {
    if !storage::valid_id(session_id) {
        bail!("Invalid session ID");
    }
    Ok(data.join("spills").join(session_id))
}

fn spill_path(data: &Path, session_id: &str, spill_id: &str) -> Result<PathBuf> {
    if !storage::valid_id(spill_id) {
        bail!("Invalid spill ID");
    }
    Ok(session_dir(data, session_id)?.join(format!("{spill_id}.txt")))
}

fn evict_oldest_if_needed(dir: &Path) -> Result<()> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "txt") {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            files.push((mtime, path));
        }
    }
    if files.len() < MAX_SPILLS_PER_SESSION {
        return Ok(());
    }
    files.sort_by_key(|(mtime, p)| (*mtime, p.clone()));
    let excess = files.len() + 1 - MAX_SPILLS_PER_SESSION;
    for (_, path) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

pub(crate) fn write_spill(data: &Path, session_id: &str, tool: &str, text: &str) -> Result<String> {
    let dir = session_dir(data, session_id)?;
    std::fs::create_dir_all(&dir)?;
    evict_oldest_if_needed(&dir)?;
    let prefix: String = tool
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(20)
        .collect();
    let seq = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
    let spill_id = format!(
        "sp-{}-{}-{}",
        if prefix.is_empty() { "out" } else { &prefix },
        storage::now_ms(),
        seq
    );
    let path = spill_path(data, session_id, &spill_id)?;
    let bytes = if text.len() > MAX_SPILL_BYTES {
        let cut = floor_char_boundary(text, MAX_SPILL_BYTES);
        format!(
            "{}\n[spill truncated at 8 MiB ({} total bytes)]",
            &text[..cut],
            text.len()
        )
        .into_bytes()
    } else {
        text.as_bytes().to_vec()
    };
    storage::atomic_write_bytes(&path, &bytes)?;
    Ok(spill_id)
}

/// Truncate `text` safely at UTF-8 char boundaries and persist full output as a
/// session spill when `data` and `session_id` are provided.
pub(crate) fn truncate_or_spill(
    data: Option<&Path>,
    session_id: Option<&str>,
    tool: &str,
    text: &str,
    inline_limit: usize,
) -> (String, Option<String>, bool, usize) {
    let total_bytes = text.len();
    if total_bytes <= inline_limit {
        return (text.to_owned(), None, false, total_bytes);
    }
    let cut = floor_char_boundary(text, inline_limit);
    let spill_id = match (data, session_id) {
        (Some(d), Some(sid)) => write_spill(d, sid, tool, text).ok(),
        _ => None,
    };
    let preview = match &spill_id {
        Some(id) => format!(
            "{}… [truncated {} bytes; full output stored in spill_id={} — inspect with read_spill]",
            &text[..cut],
            total_bytes - cut,
            id
        ),
        None => format!("{}… [truncated {} bytes]", &text[..cut], total_bytes - cut),
    };
    (preview, spill_id, true, total_bytes)
}

pub(crate) fn read_spill_range(
    data: &Path,
    session_id: &str,
    spill_id: &str,
    offset: usize,
    limit: usize,
) -> Result<Value> {
    let path = spill_path(data, session_id, spill_id)?;
    if !path.exists() {
        bail!("Spill not found in this session");
    }
    let bytes = storage::read_bounded_bytes(&path, MAX_SPILL_BYTES + 1024)?;
    let text = String::from_utf8_lossy(&bytes);
    let total = text.len();
    let clamped_limit = limit.clamp(1, MAX_READ_LIMIT);
    let start = ceil_char_boundary(&text, offset.min(total));
    let end = floor_char_boundary(&text, start.saturating_add(clamped_limit).min(total));
    let slice = &text[start..end];
    Ok(json!({
        "spill_id": spill_id,
        "offset": start,
        "next_offset": end,
        "total_bytes": total,
        "eof": end >= total,
        "content": slice,
    }))
}

pub(crate) fn execute_read_tool(data: &Path, session_id: &str, args: &Value) -> Result<Value> {
    let spill_id = args["spill_id"].as_str().context("spill_id is required")?;
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(usize::MAX as u64) as usize;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_READ_LIMIT as u64)
        .clamp(1, MAX_READ_LIMIT as u64) as usize;
    read_spill_range(data, session_id, spill_id, offset, limit)
}

pub(crate) fn list_spills(data: &Path, session_id: &str) -> Result<Vec<Value>> {
    let dir = session_dir(data, session_id)?;
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().is_some_and(|e| e == "txt") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        rows.push(json!({
            "spill_id": stem,
            "bytes": meta.len(),
        }));
    }
    rows.sort_by(|a, b| a["spill_id"].as_str().cmp(&b["spill_id"].as_str()));
    Ok(rows)
}

pub(crate) fn delete_session_spills(data: &Path, session_id: &str) -> Result<()> {
    let dir = session_dir(data, session_id)?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

#[derive(Default, Deserialize)]
pub(crate) struct SpillQuery {
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

fn ensure_session_exists(app: &App, id: &str) -> std::result::Result<(), crate::ApiError> {
    if !app.sessions.lock().unwrap().contains_key(id) {
        return Err(not_found("Conversation not found"));
    }
    Ok(())
}

pub(crate) async fn list_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    ensure_session_exists(&app, &id)?;
    blocking(move || {
        let spills = list_spills(&app.data, &id).map_err(bad_request)?;
        Ok(Json(json!({ "session_id": id, "spills": spills })))
    })
    .await
}

pub(crate) async fn get_api(
    State(app): State<App>,
    AxumPath((id, spill_id)): AxumPath<(String, String)>,
    Query(query): Query<SpillQuery>,
) -> ApiResult<Value> {
    ensure_session_exists(&app, &id)?;
    blocking(move || {
        let offset = query.offset.unwrap_or(0);
        let limit = query
            .limit
            .unwrap_or(DEFAULT_READ_LIMIT)
            .clamp(1, MAX_READ_LIMIT);
        let page = read_spill_range(&app.data, &id, &spill_id, offset, limit).map_err(|e| {
            if e.to_string().contains("not found") {
                not_found(e)
            } else {
                bad_request(e)
            }
        })?;
        Ok(Json(page))
    })
    .await
    .map_err(|e| {
        if e.status == axum::http::StatusCode::NOT_FOUND {
            e
        } else {
            internal(e.message)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("allpaka-spill-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_paginate_utf8_and_session_isolation() {
        let data = temp_dir();
        let text = "Привет, мир! ".repeat(200);
        let (preview, spill_id, truncated, total) =
            truncate_or_spill(Some(&data), Some("sess-a"), "run_command", &text, 64);
        assert!(truncated);
        assert_eq!(total, text.len());
        let id = spill_id.expect("spill id created");
        assert!(preview.contains(&id));

        let p1 = read_spill_range(&data, "sess-a", &id, 0, 100).unwrap();
        let p2 = read_spill_range(
            &data,
            "sess-a",
            &id,
            p1["next_offset"].as_u64().unwrap() as usize,
            MAX_READ_LIMIT,
        )
        .unwrap();
        let combined = format!(
            "{}{}",
            p1["content"].as_str().unwrap(),
            p2["content"].as_str().unwrap()
        );
        assert_eq!(combined, text);
        assert_eq!(p2["eof"], true);

        assert!(read_spill_range(&data, "sess-b", &id, 0, 100).is_err());
        assert!(read_spill_range(&data, "sess-a", "../escape", 0, 100).is_err());

        delete_session_spills(&data, "sess-a").unwrap();
        assert!(list_spills(&data, "sess-a").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn fifo_eviction_caps_session_spills() {
        let data = temp_dir();
        for i in 0..(MAX_SPILLS_PER_SESSION + 5) {
            write_spill(&data, "sess-cap", "fetch", &format!("payload-{i}")).unwrap();
        }
        let rows = list_spills(&data, "sess-cap").unwrap();
        assert_eq!(rows.len(), MAX_SPILLS_PER_SESSION);
        let _ = std::fs::remove_dir_all(&data);
    }
}
