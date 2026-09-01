use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

use crate::rag_mcp::{RagMcp, RagMcpConfig};

const DEFAULT_RAG_NOTES_DIR: &str =
    "/Users/themoretheless/.claude/projects/-Users-themoretheless-Documents-Sources-allpaka/memory";
const RAG_SEARCH_MAX_RESULTS: usize = 5;
const RAG_SEARCH_MAX_LINES: usize = 6;
const RAG_TOOL_MAX_ROUNDS: usize = 2;
const RAG_READ_MAX_CHARS: usize = 12_000;

#[derive(Debug, Clone)]
pub(super) struct RagToolConfig {
    pub(super) enabled: bool,
    notes_dir: PathBuf,
    search_max_results: usize,
    search_max_lines: usize,
    pub(super) max_tool_rounds: usize,
    read_max_chars: usize,
    pub(super) inject_tools_when_missing: bool,
}

impl RagToolConfig {
    pub(super) fn load() -> Self {
        let enabled = std::env::var("ALLPAKA_RAG_TOOLS")
            .ok()
            .is_none_or(|v| v != "0");
        let notes_dir = std::env::var("ALLPAKA_RAG_NOTES_DIR")
            .map(PathBuf::from)
            .or_else(|_| std::env::var("RAG_INGEST_ROOTS").map(PathBuf::from))
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_RAG_NOTES_DIR));
        let search_max_results =
            parse_env_usize("ALLPAKA_RAG_SEARCH_MAX_RESULTS").unwrap_or(RAG_SEARCH_MAX_RESULTS);
        let search_max_lines =
            parse_env_usize("ALLPAKA_RAG_SEARCH_MAX_LINES").unwrap_or(RAG_SEARCH_MAX_LINES);
        let max_tool_rounds = parse_env_usize("ALLPAKA_RAG_MAX_TOOL_ROUNDS")
            .unwrap_or(RAG_TOOL_MAX_ROUNDS)
            .max(1);
        let read_max_chars =
            parse_env_usize("ALLPAKA_RAG_READ_MAX_CHARS").unwrap_or(RAG_READ_MAX_CHARS);
        let inject_tools_when_missing = parse_env_bool("ALLPAKA_RAG_AUTO_TOOLS");
        RagToolConfig {
            enabled,
            notes_dir,
            search_max_results,
            search_max_lines,
            max_tool_rounds,
            read_max_chars,
            inject_tools_when_missing,
        }
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Which retrieval backend serves rag_search/rag_read.
///
/// `Auto` (default): rag-mcp when its binary and DuckDB are present, grep
/// otherwise. `Mcp` forces rag-mcp (a missing server is an error in the tool
/// reply, not a silent downgrade). `Grep` keeps the old directory scan, and
/// is also the runtime fallback when the MCP child dies mid-session.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RagBackend {
    Auto,
    Mcp,
    Grep,
}

impl RagBackend {
    fn load() -> Self {
        match std::env::var("ALLPAKA_RAG_BACKEND").as_deref() {
            Ok("mcp") => RagBackend::Mcp,
            Ok("grep") => RagBackend::Grep,
            _ => RagBackend::Auto,
        }
    }
}

/// The tool surface: static config plus the lazily spawned MCP child.
/// RefCell because `handle` sees everything by reference; serve is
/// single-threaded, so there is no real sharing.
pub(super) struct RagTools {
    pub(super) cfg: RagToolConfig,
    backend: RagBackend,
    mcp_cfg: RagMcpConfig,
    mcp: RefCell<Option<RagMcp>>,
}

impl RagTools {
    pub(super) fn load() -> Self {
        let cfg = RagToolConfig::load();
        let backend = RagBackend::load();
        let mcp_cfg = RagMcpConfig::from_env(&cfg.notes_dir);
        let mcp = match backend {
            RagBackend::Grep => None,
            RagBackend::Auto | RagBackend::Mcp if mcp_cfg.available() => {
                match RagMcp::spawn(&mcp_cfg) {
                    Ok(m) => {
                        println!(
                            "rag backend: rag-mcp ({}, BM25 index)",
                            mcp_cfg.db.display()
                        );
                        Some(m)
                    }
                    Err(e) => {
                        if backend == RagBackend::Mcp {
                            println!("rag backend: rag-mcp failed to start ({e:#})");
                        } else {
                            println!("rag backend: rag-mcp unavailable ({e:#}), using grep");
                        }
                        None
                    }
                }
            }
            _ => {
                if backend == RagBackend::Mcp {
                    println!("rag backend: rag-mcp binary/db missing, tool calls will fail");
                } else {
                    println!("rag backend: grep over {}", cfg.notes_dir.display());
                }
                None
            }
        };
        RagTools {
            cfg,
            backend,
            mcp_cfg,
            mcp: RefCell::new(mcp),
        }
    }

    /// Run `f` against the MCP child. On any failure the child is dropped,
    /// the backend degrades to grep for the rest of the process, and the
    /// caller retries with grep. Spawns lazily when Auto found nothing at
    /// startup (the db may have appeared since).
    fn with_mcp(&self, f: impl FnOnce(&mut RagMcp) -> Result<String>) -> Option<String> {
        if self.backend == RagBackend::Grep {
            return None;
        }
        {
            let mut slot = self.mcp.borrow_mut();
            if slot.is_none() && self.backend == RagBackend::Auto && self.mcp_cfg.available() {
                *slot = RagMcp::spawn(&self.mcp_cfg).ok();
            }
            if let Some(m) = slot.as_mut() {
                match f(m) {
                    Ok(s) => return Some(s),
                    Err(e) => {
                        println!("rag-mcp call failed ({e:#}); falling back to grep");
                        *slot = None;
                    }
                }
            }
        }
        if self.backend == RagBackend::Mcp {
            return Some("rag-mcp backend is unavailable".to_string());
        }
        None
    }
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        matches!(
            v.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

fn rag_search(tools: &RagTools, query: &str) -> Result<String> {
    if let Some(s) = tools.with_mcp(|m| rag_search_mcp(&tools.cfg, m, query)) {
        return Ok(s);
    }
    rag_search_grep(&tools.cfg, query)
}

/// Format MCP hits into the same `## title / - line` blocks the grep backend
/// produces, so the model-facing contract does not change with the index.
fn rag_search_mcp(cfg: &RagToolConfig, mcp: &mut RagMcp, query: &str) -> Result<String> {
    let hits = mcp.search(query, cfg.search_max_results)?;
    if hits.is_empty() {
        return Ok(format!("no notes match: {query}"));
    }
    let mut out = Vec::new();
    for hit in hits.iter().take(cfg.search_max_results) {
        let title = hit["document_title"].as_str().unwrap_or("note");
        let score = hit["score"].as_f64().unwrap_or(0.0);
        let body = hit["snippet"]
            .as_str()
            .or_else(|| hit["content"].as_str())
            .unwrap_or("");
        let mut block = format!("## {title} (score: {score:.2})");
        for line in body.lines().take(cfg.search_max_lines) {
            block.push_str(&format!("\n- {line}"));
        }
        out.push(block);
    }
    Ok(out.join("\n\n"))
}

fn rag_read(tools: &RagTools, name: &str) -> Result<String> {
    if let Some(s) = tools.with_mcp(|m| rag_read_mcp(&tools.cfg, m, name)) {
        return Ok(s);
    }
    rag_read_grep(&tools.cfg, name)
}

/// rag_read by file name: find the document whose title matches, then fetch
/// it whole. Falls back to the top hit when no title matches exactly.
fn rag_read_mcp(cfg: &RagToolConfig, mcp: &mut RagMcp, name: &str) -> Result<String> {
    if name.trim().is_empty() {
        return Ok("rag_read: missing name".to_string());
    }
    let hits = mcp.search(name, 5)?;
    let hit = hits
        .iter()
        .find(|h| h["document_title"].as_str() == Some(name))
        .or_else(|| hits.first())
        .with_context(|| format!("no such note: {name}"))?;
    let doc_id = hit["document_id"]
        .as_str()
        .context("hit has no document_id")?;
    let mut text = mcp.get_document(doc_id)?;
    if text.len() > cfg.read_max_chars {
        text = text.chars().take(cfg.read_max_chars).collect();
        text.push_str("\n...\n[truncated]");
    }
    Ok(text)
}

fn rag_search_grep(cfg: &RagToolConfig, query: &str) -> Result<String> {
    let query = query.to_lowercase();
    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();
    if words.is_empty() {
        return Ok("empty query".to_string());
    }
    if !cfg.notes_dir.exists() {
        return Ok(format!(
            "notes root does not exist: {}",
            cfg.notes_dir.display()
        ));
    }
    let mut notes: Vec<PathBuf> = Vec::new();
    collect_notes(&cfg.notes_dir, &mut notes)
        .with_context(|| format!("reading notes directory {}", cfg.notes_dir.display()))?;
    notes.sort();
    let mut matches: Vec<(usize, PathBuf, Vec<String>)> = Vec::new();
    for note in notes {
        let text = match std::fs::read_to_string(&note) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let low = text.to_lowercase();
        let mut score = 0usize;
        for w in &words {
            score += low.matches(w).count();
        }
        if score == 0 {
            continue;
        }
        let mut lines = Vec::new();
        for line in text.lines() {
            let lower_line = line.to_lowercase();
            if words.iter().any(|w| lower_line.contains(w)) {
                lines.push(line.to_string());
                if lines.len() >= cfg.search_max_lines {
                    break;
                }
            }
        }
        if !lines.is_empty() {
            matches.push((score, note, lines));
        }
    }
    if matches.is_empty() {
        return Ok(format!("no notes match: {query}"));
    }
    matches.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out = Vec::new();
    for (score, note, lines) in matches.into_iter().take(cfg.search_max_results) {
        let name = note.file_name().and_then(|n| n.to_str()).unwrap_or("note");
        let mut block = format!("## {name} (hits: {score})");
        for line in lines {
            block.push('\n');
            block.push_str(&format!("- {line}"));
        }
        out.push(block);
    }
    Ok(out.join("\n\n"))
}

fn rag_read_grep(cfg: &RagToolConfig, name: &str) -> Result<String> {
    if name.trim().is_empty() {
        return Ok("rag_read: missing name".to_string());
    }
    let file = cfg
        .notes_dir
        .join(Path::new(name).file_name().unwrap_or_default());
    let mut text = match std::fs::read_to_string(&file) {
        Ok(text) => text,
        Err(_) => return Ok(format!("no such note: {}", file.display())),
    };
    if text.len() > cfg.read_max_chars {
        text = text.chars().take(cfg.read_max_chars).collect();
        text.push_str("\n...\n[truncated]");
    }
    Ok(text)
}

fn parse_rag_tool_args(call: &Value) -> Value {
    let raw = &call["function"]["arguments"];
    if let Some(raw) = raw.as_str() {
        serde_json::from_str::<Value>(raw).unwrap_or(Value::Null)
    } else if raw.is_object() || raw.is_array() || raw.is_null() || raw.is_string() {
        raw.clone()
    } else {
        Value::Null
    }
}

fn collect_notes(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(e) => return Err(anyhow::anyhow!("reading {}: {e}", current.display())),
        };
        let mut entry_paths = Vec::new();
        for entry in entries {
            let entry = entry?;
            entry_paths.push(entry.path());
        }
        entry_paths.sort();
        for path in entry_paths {
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                out.push(path);
            }
        }
    }
    Ok(())
}

pub(super) fn run_rag_tool(call: &Value, tools: &RagTools) -> Result<String> {
    let tool_name = call["function"]["name"].as_str().unwrap_or_default();
    let args = parse_rag_tool_args(call);
    if !tools.cfg.enabled {
        return Ok("rag tools are disabled".to_string());
    }
    match tool_name {
        "rag_search" => {
            let query = args
                .get("query")
                .or_else(|| args.get("q"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            rag_search(tools, query).with_context(|| format!("tool call {tool_name}"))
        }
        "rag_read" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            rag_read(tools, name).with_context(|| format!("tool call {tool_name}"))
        }
        other => Ok(format!(
            "tool `{other}` is not available; supported: rag_search, rag_read"
        )),
    }
}

pub(super) fn rag_default_tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Search local markdown notes by keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "rag_read",
                "description": "Read one note by file name as returned by rag_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "File name of the note, for example engine-status.md."
                        }
                    },
                    "required": ["name"]
                }
            }
        }),
    ]
}
