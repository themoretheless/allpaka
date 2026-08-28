//! An OpenAI-compatible chat endpoint over the engine.
//!
//! `POST /v1/chat/completions` with the usual `{model, messages, max_tokens,
//! temperature}` body; non-streaming, one request at a time. The point is that
//! an agent framework pointed at this URL cannot tell it is not llama-server -
//! same route, same response shape.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::net::{TcpListener, TcpStream};

use allpaka_model::{Model, Session, Tokenizer};
use crate::rag_mcp::{RagMcp, RagMcpConfig};

/// The KV cache carried across requests.
///
/// A chat regenerates the same prompt prefix on every turn: the whole prior
/// conversation, byte for byte, plus one new user message. Keeping the session
/// and the token list lets each request pay only for what is actually new -
/// without this, turn N re-runs the forward pass over every earlier turn and
/// the chat slows down linearly with its own length.
struct ChatState {
    session: Session,
    tokens: Vec<u32>,
    /// SSM snapshots at past prompt boundaries (qwen35moe). The KV cache
    /// rolls back by truncating, but the gated-delta-net recurrence is
    /// irreversible: a prefix hit must restore the state as of that prefix.
    /// One snapshot per request, taken right after its prefill, plus the
    /// prefill's last logits so a fully-covered prompt needs no prefill at
    /// all. Bounded by [SSM_SNAPSHOTS_MAX], oldest evicted first.
    snaps: std::collections::VecDeque<SsmSnap>,
}

/// One prompt-boundary snapshot of the recurrent state.
struct SsmSnap {
    /// Token count of the prompt this state was captured after.
    tokens: usize,
    /// `SsmCache::snapshot()` at the prompt's end.
    state: Vec<f32>,
    /// The prefill's last-position logits, kept so a repeated prompt skips
    /// the prefill entirely (greedy determinism is then bit-exact).
    logits: Vec<f32>,
}

/// SSM snapshots are ~66 MB each on Qwen3.6-35B (30 layers x conv window +
/// 32 heads x 128x128 f32); eight boundaries cover a chat with edits and
/// branches without growing memory without limit.
const SSM_SNAPSHOTS_MAX: usize = 8;

/// Context the persistent session is sized for. f32 KV cache is heavy
/// (~0.4 MB per token on Qwen3-30B), so this is a deliberate budget, not the
/// model's maximum.
const SESSION_TOKENS: usize = 16384;

const DEFAULT_RAG_NOTES_DIR: &str =
    "/Users/themoretheless/.claude/projects/-Users-themoretheless-Documents-Sources-allpaka/memory";
const RAG_SEARCH_MAX_RESULTS: usize = 5;
const RAG_SEARCH_MAX_LINES: usize = 6;
const RAG_TOOL_MAX_ROUNDS: usize = 2;
const RAG_READ_MAX_CHARS: usize = 12_000;

#[derive(Debug, Clone)]
struct RagToolConfig {
    enabled: bool,
    notes_dir: PathBuf,
    search_max_results: usize,
    search_max_lines: usize,
    max_tool_rounds: usize,
    read_max_chars: usize,
    inject_tools_when_missing: bool,
}

impl RagToolConfig {
    fn load() -> Self {
        let enabled = std::env::var("ALLPAKA_RAG_TOOLS")
            .ok()
            .is_none_or(|v| v != "0");
        let notes_dir = std::env::var("ALLPAKA_RAG_NOTES_DIR")
            .map(PathBuf::from)
            .or_else(|_| std::env::var("RAG_INGEST_ROOTS").map(PathBuf::from))
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_RAG_NOTES_DIR));
        let search_max_results = parse_env_usize("ALLPAKA_RAG_SEARCH_MAX_RESULTS")
            .unwrap_or(RAG_SEARCH_MAX_RESULTS);
        let search_max_lines = parse_env_usize("ALLPAKA_RAG_SEARCH_MAX_LINES")
            .unwrap_or(RAG_SEARCH_MAX_LINES);
        let max_tool_rounds = parse_env_usize("ALLPAKA_RAG_MAX_TOOL_ROUNDS")
            .unwrap_or(RAG_TOOL_MAX_ROUNDS)
            .max(1);
        let read_max_chars = parse_env_usize("ALLPAKA_RAG_READ_MAX_CHARS").unwrap_or(RAG_READ_MAX_CHARS);
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
struct RagTools {
    cfg: RagToolConfig,
    backend: RagBackend,
    mcp_cfg: RagMcpConfig,
    mcp: RefCell<Option<RagMcp>>,
}

impl RagTools {
    fn load() -> Self {
        let cfg = RagToolConfig::load();
        let backend = RagBackend::load();
        let mcp_cfg = RagMcpConfig::from_env(&cfg.notes_dir);
        let mcp = match backend {
            RagBackend::Grep => None,
            RagBackend::Auto | RagBackend::Mcp if mcp_cfg.available() => {
                match RagMcp::spawn(&mcp_cfg) {
                    Ok(m) => {
                        println!("rag backend: rag-mcp ({}, BM25 index)", mcp_cfg.db.display());
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
        RagTools { cfg, backend, mcp_cfg, mcp: RefCell::new(mcp) }
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
    std::env::var(name)
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
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
        let body = hit["snippet"].as_str().or_else(|| hit["content"].as_str()).unwrap_or("");
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
    let doc_id = hit["document_id"].as_str().context("hit has no document_id")?;
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
        return Ok(format!("notes root does not exist: {}", cfg.notes_dir.display()));
    }
    let mut notes: Vec<PathBuf> = Vec::new();
    collect_notes(&cfg.notes_dir, &mut notes).with_context(|| {
        format!("reading notes directory {}", cfg.notes_dir.display())
    })?;
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
    let file = cfg.notes_dir.join(Path::new(name).file_name().unwrap_or_default());
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

fn run_rag_tool(call: &Value, tools: &RagTools) -> Result<String> {
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

fn rag_default_tool_schemas() -> Vec<Value> {
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

struct CompletionResult {
    content: String,
    tool_calls: Vec<Value>,
    finish: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill_secs: f64,
    decode_secs: f64,
}

fn run_inference(
    model: &Model,
    tok: &Tokenizer,
    template: &Template,
    chat: &mut ChatState,
    messages: &[(String, String)],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    parse_tools: bool,
    mut stream: Option<&mut TcpStream>,
    emit_done: bool,
) -> Result<CompletionResult> {
    let prompt = template.prompt(tok, messages)?;
    let stops = template.stop_tokens(tok);
    let mut sampler = Sampler::new(temperature, top_p, repeat_penalty);
    // Greedy without a repeat penalty can take the on-GPU argmax: one word
    // read back per token instead of the whole vocabulary.
    let greedy = temperature <= 0.0 && repeat_penalty == 1.0;

    let mut common =
        chat.tokens.iter().zip(&prompt).take_while(|(a, b)| **a == **b).count();
    let mut logits: Vec<f32> = Vec::new();
    let mut skip_prefill = false;
    if prompt.len() + max_tokens + 1 > chat.session.capacity() {
        chat.session = model.new_session(SESSION_TOKENS.max(prompt.len() + max_tokens + 1));
        chat.tokens.clear();
        chat.snaps.clear();
        common = 0;
    } else if chat.session.ssm.is_some() {
        // The gated-delta-net state cannot be rolled back by truncating
        // like the KV cache: a shared prefix is only reusable through the
        // snapshot taken at that prefix's end. Without one, replay from
        // scratch (slow, but correct).
        if common < chat.tokens.len() {
            match chat.snaps.iter().find(|s| s.tokens == common) {
                Some(snap) => {
                    let state = snap.state.clone();
                    chat.session.ssm.as_mut().expect("ssm session").restore(&state);
                    chat.session.truncate(common);
                    chat.tokens.truncate(common);
                    if common == prompt.len() {
                        // The prompt is fully covered: its last logits are
                        // in the snapshot, no prefill at all.
                        logits = snap.logits.clone();
                        skip_prefill = true;
                    }
                }
                None => {
                    chat.session =
                        model.new_session(SESSION_TOKENS.max(prompt.len() + max_tokens + 1));
                    chat.tokens.clear();
                    chat.snaps.clear();
                    common = 0;
                }
            }
        } else if common == prompt.len() {
            // History IS the prompt: the state already sits at its end and
            // only the last logits are missing (KV-only models replay one
            // token here; the SSM cannot step back). The snapshot has them.
            match chat.snaps.iter().find(|s| s.tokens == common) {
                Some(snap) => {
                    logits = snap.logits.clone();
                    skip_prefill = true;
                }
                None => {
                    chat.session =
                        model.new_session(SESSION_TOKENS.max(prompt.len() + max_tokens + 1));
                    chat.tokens.clear();
                    chat.snaps.clear();
                    common = 0;
                }
            }
        }
        // else: common == chat.tokens.len() < prompt.len() - the state is
        // already exactly where the prefix ends; nothing to restore.
    } else {
        if common == prompt.len() {
            common -= 1;
        }
        chat.session.truncate(common);
        chat.tokens.truncate(common);
    }

    let prefill_chunk = allpaka_model::Model::prefill_chunk();
    let t0 = std::time::Instant::now();
    if !skip_prefill {
        for chunk in prompt[common..].chunks(prefill_chunk) {
            logits = model.forward_batch(chunk, &mut chat.session)?;
            chat.tokens.extend_from_slice(chunk);
        }
    }
    let prefill_secs = t0.elapsed().as_secs_f64();

    // Remember the recurrent state at this prompt's boundary (qwen35moe):
    // the next request sharing the prefix restores it instead of replaying.
    // Snapshot before generation, when the state is exactly "prompt consumed".
    if let Some(ssm) = chat.session.ssm.as_ref() {
        if !logits.is_empty() {
            let snap = SsmSnap { tokens: prompt.len(), state: ssm.snapshot(), logits: logits.clone() };
            if let Some(existing) = chat.snaps.iter_mut().find(|s| s.tokens == prompt.len()) {
                *existing = snap;
            } else {
                chat.snaps.push_back(snap);
                while chat.snaps.len() > SSM_SNAPSHOTS_MAX {
                    chat.snaps.pop_front();
                }
            }
        }
    }

    let mut streaming = stream.is_some();
    let think_prefix = template.think_prefix();
    if streaming {
        write_sse_headers(stream.as_mut().unwrap())?;
        // The primed `<think>\n` is part of the reply text but not of the
        // generated token stream; send it ahead of the first real delta.
        if !think_prefix.is_empty() {
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": think_prefix}}],
                }),
            )?;
        }
    }

    let t1 = std::time::Instant::now();
    let mut generated = Vec::new();
    let mut emitted = String::new();
    let mut finish = "length";
    let mut greedy_next: Option<u32> = None;
    for _ in 0..max_tokens {
        let next = match greedy_next.take() {
            Some(t) => t,
            None => sampler.pick(&logits, &chat.tokens),
        };
        if stops.contains(&next) {
            finish = "stop";
            break;
        }
        generated.push(next);
        if streaming {
            let full = tok.decode(&generated);
            let complete = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
            let complete =
                if parse_tools { &complete[..stream_safe_len(complete)] } else { complete };
            if let Some(delta) = complete.strip_prefix(emitted.as_str()) {
                if !delta.is_empty() {
                    let chunk = json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta}}],
                        "tokens_generated": generated.len(),
                    });
                    write_sse_event(stream.as_mut().unwrap(), &chunk)?;
                    emitted = complete.to_string();
                }
            }
        }
        if greedy {
            greedy_next = Some(model.forward_greedy(next, &mut chat.session)?);
        } else {
            logits = model.forward(next, &mut chat.session)?;
        }
        chat.tokens.push(next);
    }
    let decode_secs = t1.elapsed().as_secs_f64();

    let text = format!("{think_prefix}{}", tok.decode(&generated));
    let (content, tool_calls) = if parse_tools {
        parse_tool_calls(&text)
    } else {
        (text.clone(), Vec::new())
    };
    if !tool_calls.is_empty() && finish == "stop" {
        finish = "tool_calls";
    }
    println!(
        "  {} new + {common} cached prompt tok in {prefill_secs:.1}s, \
         {} generated in {decode_secs:.1}s ({:.1} tok/s)",
        prompt.len() - common,
        generated.len(),
        generated.len() as f64 / decode_secs.max(1e-9),
    );

    if streaming {
        // `emitted` tracks the body only; the think prefix went out first.
        if let Some(delta) = content[think_prefix.len()..].strip_prefix(emitted.as_str()) {
            if !delta.is_empty() {
                write_sse_event(
                    stream.as_mut().unwrap(),
                    &json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta}}],
                    }),
                )?;
            }
        }
        if !tool_calls.is_empty() {
            let calls_delta: Vec<Value> = tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut c = c.clone();
                    c["index"] = json!(i);
                    c
                })
                .collect();
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"tool_calls": calls_delta}}],
                }),
            )?;
        }
        if emit_done {
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": prompt.len(),
                        "completion_tokens": generated.len(),
                        "total_tokens": prompt.len() + generated.len(),
                    },
                    "timing": {
                        "prefill_secs": prefill_secs,
                        "decode_secs": decode_secs,
                        "tokens_per_sec": generated.len() as f64 / decode_secs.max(1e-9),
                        "cached_tokens": common,
                        "context_used": chat.tokens.len(),
                        "context_capacity": chat.session.capacity(),
                    },
                }),
            )?;
            stream.as_mut().unwrap().write_all(b"data: [DONE]\n\n")?;
        }
    }

    Ok(CompletionResult {
        content,
        tool_calls,
        finish: finish.to_string(),
        prompt_tokens: prompt.len(),
        completion_tokens: generated.len(),
        cached_tokens: common,
        prefill_secs,
        decode_secs,
    })
}

fn emit_stream_completion(
    stream: &mut TcpStream,
    completion: &CompletionResult,
    emit_content: bool,
    finish: &str,
) -> Result<()> {
    if emit_content && !completion.content.is_empty() {
        write_sse_event(
            stream,
            &json!({
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": completion.content}}],
            }),
        )?;
    }
    write_sse_event(
        stream,
        &json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
            "timing": {
                "prefill_secs": completion.prefill_secs,
                "decode_secs": completion.decode_secs,
                "tokens_per_sec": completion.completion_tokens as f64
                    / completion.decode_secs.max(1e-9),
                "cached_tokens": completion.cached_tokens,
            },
        }),
    )?;
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

/// The chat template families this server can format. Decided by which
/// special tokens the vocabulary actually contains.
enum Template {
    /// `<|im_start|>role\n...<|im_end|>\n` - Qwen and much of the ecosystem.
    /// `force_think` mirrors the GGUF chat template of the thinking Qwen
    /// generations: the assistant turn is primed with a literal `<think>\n`
    /// (llama.cpp renders the same prefix from the jinja template).
    ChatMl { im_start: u32, im_end: u32, think: Option<u32>, force_think: bool },
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>` - Llama 3.
    Llama3 { bos: u32, start_header: u32, end_header: u32, eot: u32 },
    /// `[gMASK]<sop><|user|>\n...<|assistant|>\n` - GLM-4.x (ChatGLM).
    Glm { gmask: Option<u32>, sop: Option<u32>, system: u32, user: u32, assistant: u32 },
}

impl Template {
    fn detect(tok: &Tokenizer) -> Result<Self> {
        if let (Some(im_start), Some(im_end)) =
            (tok.piece_id("<|im_start|>"), tok.piece_id("<|im_end|>"))
        {
            return Ok(Template::ChatMl {
                im_start,
                im_end,
                think: tok.piece_id("<think>"),
                force_think: false,
            });
        }
        if let (Some(start_header), Some(end_header), Some(eot)) = (
            tok.piece_id("<|start_header_id|>"),
            tok.piece_id("<|end_header_id|>"),
            tok.piece_id("<|eot_id|>"),
        ) {
            return Ok(Template::Llama3 {
                bos: tok.bos.context("llama3 template needs a BOS token")?,
                start_header,
                end_header,
                eot,
            });
        }
        if let (Some(system), Some(user), Some(assistant)) = (
            tok.piece_id("<|system|>"),
            tok.piece_id("<|user|>"),
            tok.piece_id("<|assistant|>"),
        ) {
            return Ok(Template::Glm {
                gmask: tok.piece_id("[gMASK]"),
                sop: tok.piece_id("<sop>"),
                system,
                user,
                assistant,
            });
        }
        bail!("the vocabulary has neither ChatML nor Llama 3 nor GLM special tokens");
    }

    /// The literal prefix the assistant turn was primed with (`<think>\n`
    /// for the thinking Qwens, empty otherwise). The reply text gets it back
    /// so clients see the complete `<think>...</think>` block.
    fn think_prefix(&self) -> &'static str {
        match self {
            Template::ChatMl { force_think: true, .. } => "<think>\n",
            _ => "",
        }
    }

    /// Format a conversation into token ids, ending where the assistant
    /// starts writing.
    fn prompt(&self, tok: &Tokenizer, messages: &[(String, String)]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        match self {
            Template::ChatMl { im_start, im_end, think, force_think } => {
                for (role, content) in messages {
                    ids.push(*im_start);
                    ids.extend(tok.encode(&format!("{role}\n{content}"))?);
                    ids.push(*im_end);
                    ids.extend(tok.encode("\n")?);
                }
                ids.push(*im_start);
                ids.extend(tok.encode("assistant\n")?);
                if *force_think {
                    // The special token by id, exactly as the jinja template
                    // renders it - plain BPE would split the literal text.
                    match think {
                        Some(t) => {
                            ids.push(*t);
                            ids.extend(tok.encode("\n")?);
                        }
                        None => ids.extend(tok.encode("<think>\n")?),
                    }
                }
            }
            Template::Llama3 { bos, start_header, end_header, eot } => {
                ids.push(*bos);
                for (role, content) in messages {
                    ids.push(*start_header);
                    ids.extend(tok.encode(role)?);
                    ids.push(*end_header);
                    ids.extend(tok.encode(&format!("\n\n{content}"))?);
                    ids.push(*eot);
                }
                ids.push(*start_header);
                ids.extend(tok.encode("assistant")?);
                ids.push(*end_header);
                ids.extend(tok.encode("\n\n")?);
            }
            Template::Glm { gmask, sop, system, user, assistant } => {
                if let Some(g) = gmask {
                    ids.push(*g);
                }
                if let Some(s) = sop {
                    ids.push(*s);
                }
                for (role, content) in messages {
                    let role_id = match role.as_str() {
                        "system" => *system,
                        "assistant" => *assistant,
                        _ => *user,
                    };
                    ids.push(role_id);
                    ids.extend(tok.encode(&format!("\n{content}"))?);
                }
                ids.push(*assistant);
                ids.extend(tok.encode("\n")?);
            }
        }
        Ok(ids)
    }

    /// Tokens that end an assistant turn.
    fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = Vec::new();
        match self {
            Template::ChatMl { im_end, .. } => stops.push(*im_end),
            Template::Llama3 { eot, .. } => stops.push(*eot),
            Template::Glm { user, .. } => stops.push(*user),
        }
        if let Some(eos) = tok.eos {
            stops.push(eos);
        }
        stops
    }
}

/// A deliberately small sampler: greedy at temperature 0, otherwise
/// repetition penalty, temperature scaling and nucleus (top-p) truncation.
struct Sampler {
    temperature: f32,
    top_p: f32,
    /// llama.cpp-style repeat penalty over the recent window: positive
    /// logits of recent tokens are divided by it, negative multiplied.
    /// Without it a heavily quantised model happily locks into "либо-либо-
    /// либо" forever; 1.1 breaks such loops without visibly changing prose.
    repeat_penalty: f32,
    state: u64,
}

/// How many most recent tokens the repeat penalty looks at.
const PENALTY_WINDOW: usize = 256;

impl Sampler {
    fn new(temperature: f32, top_p: f32, repeat_penalty: f32) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed)
            | 1;
        Self { temperature, top_p, repeat_penalty, state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Candidates that survive the top-k preselect. Nucleus sampling never
    /// needs a total order of the vocabulary, and a full 152k-element sort
    /// costs ~3 ms per token, serial with the forward pass.
    const TOP_K: usize = 256;

    /// Sample the next token. `recent` is the context so far; its tail feeds
    /// the repeat penalty.
    fn pick(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        // Penalise recent tokens once each (not per occurrence), before any
        // selection, so a penalised token can fall out of the candidate set.
        let mut adjusted;
        let logits: &[f32] = if self.repeat_penalty != 1.0 && !recent.is_empty() {
            adjusted = logits.to_vec();
            let start = recent.len().saturating_sub(PENALTY_WINDOW);
            let mut window: Vec<u32> = recent[start..].to_vec();
            window.sort_unstable();
            window.dedup();
            for t in window {
                if let Some(v) = adjusted.get_mut(t as usize) {
                    if *v > 0.0 {
                        *v /= self.repeat_penalty;
                    } else {
                        *v *= self.repeat_penalty;
                    }
                }
            }
            &adjusted
        } else {
            logits
        };

        if self.temperature <= 0.0 {
            return (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap_or(0) as u32;
        }
        // O(n) partition to the top candidates, then softmax and nucleus over
        // just those. The probability mass beyond the top 256 tokens is far
        // below any top_p in practical use.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        let kth = Self::TOP_K.min(order.len()) - 1;
        order.select_nth_unstable_by(kth, |&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(kth + 1);
        order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        let mut scaled: Vec<f32> =
            order.iter().map(|&i| logits[i] / self.temperature).collect();
        allpaka_backend::softmax(&mut scaled);

        // Nucleus: keep the smallest prefix whose mass reaches top_p.
        let mut kept = 0usize;
        let mut mass = 0f32;
        for &p in &scaled {
            kept += 1;
            mass += p;
            if mass >= self.top_p {
                break;
            }
        }
        let mut r = self.next_f32() * mass;
        for (&i, &p) in order[..kept].iter().zip(&scaled) {
            r -= p;
            if r <= 0.0 {
                return i as u32;
            }
        }
        order[kept - 1] as u32
    }
}

/// Convert an OpenAI-shape message list (+ optional `tools`) into plain
/// (role, content) turns in the Hermes convention Qwen3 was trained on:
/// tool schemas inside `<tools>...</tools>` in the system turn, assistant
/// calls as `<tool_call>{json}</tool_call>` blocks, tool results as user
/// turns wrapped in `<tool_response>...</tool_response>` (consecutive
/// results merged into one turn, as the reference chat template does).
fn render_messages(raw: &[Value], tools: Option<&[Value]>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut pending_tool: Vec<String> = Vec::new();
    let flush_tools = |out: &mut Vec<(String, String)>, pending: &mut Vec<String>| {
        if !pending.is_empty() {
            let body = pending
                .iter()
                .map(|c| format!("<tool_response>\n{c}\n</tool_response>"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(("user".into(), body));
            pending.clear();
        }
    };
    for m in raw {
        let role = m["role"].as_str().unwrap_or("user");
        let content = m["content"].as_str().unwrap_or_default().to_string();
        if role == "tool" {
            pending_tool.push(content);
            continue;
        }
        flush_tools(&mut out, &mut pending_tool);
        if role == "assistant" {
            let mut text = content;
            if let Some(calls) = m["tool_calls"].as_array() {
                for c in calls {
                    let name = c["function"]["name"].as_str().unwrap_or_default();
                    // Arguments arrive as a JSON-encoded string per the
                    // OpenAI shape; embed them as the object Hermes expects.
                    let args = c["function"]["arguments"].as_str().unwrap_or("{}");
                    text.push_str(&format!(
                        "\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                    ));
                }
            }
            out.push(("assistant".into(), text));
        } else {
            out.push((role.to_string(), content));
        }
    }
    flush_tools(&mut out, &mut pending_tool);

    if let Some(tools) = tools {
        let schemas = tools
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let section = format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>\n{schemas}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
        );
        match out.iter_mut().find(|(role, _)| role == "system") {
            Some((_, content)) => content.push_str(&section),
            None => out.insert(
                0,
                ("system".into(), format!("You are a helpful assistant.{section}")),
            ),
        }
    }
    out
}

/// Split generated text into visible content and Hermes tool-call blocks,
/// returned in the OpenAI `tool_calls` shape (arguments re-encoded as a JSON
/// string). Malformed blocks stay in the content untouched - the client sees
/// what the model actually wrote instead of a silent drop.
fn parse_tool_calls(text: &str) -> (String, Vec<Value>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        let inner = after[..end].trim();
        let parsed: Option<Value> = serde_json::from_str(inner).ok();
        let valid = parsed
            .as_ref()
            .is_some_and(|v| v["name"].is_string() && !v["arguments"].is_null());
        if valid {
            let v = parsed.unwrap();
            calls.push(json!({
                "id": format!("call_{}", calls.len()),
                "type": "function",
                "function": {
                    "name": v["name"],
                    "arguments": v["arguments"].to_string(),
                },
            }));
            content.push_str(&rest[..start]);
        } else {
            content.push_str(&rest[..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// The longest prefix of `text` that is safe to stream when tool calls may
/// follow: everything up to the first (possibly still incomplete) opening
/// `<tool_call>` tag. A trailing partial match of the tag is held back too.
fn stream_safe_len(text: &str) -> usize {
    const TAG: &str = "<tool_call>";
    if let Some(i) = text.find(TAG) {
        return i;
    }
    let bytes = text.as_bytes();
    let max_tail = (TAG.len() - 1).min(bytes.len());
    for tail in (1..=max_tail).rev() {
        if text.is_char_boundary(bytes.len() - tail)
            && TAG.as_bytes().starts_with(&bytes[bytes.len() - tail..])
        {
            return bytes.len() - tail;
        }
    }
    bytes.len()
}

pub fn run(model_path: &Path, bind: &str) -> Result<()> {
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    for m in file.mappings() {
        prewarm(m);
    }
    let model = Model::load(&file)?;
    let tokenizer = Tokenizer::from_gguf(&file)?;
    tokenizer.self_check()?;
    let mut template = Template::detect(&tokenizer)?;
    // Thinking Qwen generations prime the assistant turn with `<think>\n`
    // in their jinja chat template; mirror that prefix in ours.
    if let Template::ChatMl { force_think, .. } = &mut template {
        *force_think = file
            .meta_str("tokenizer.chat_template")
            .is_some_and(|t| t.contains("<think>"));
    }
    let model_name = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "allpaka".into());

    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
    println!("allpaka serve: {model_name} on http://{bind}/v1/chat/completions");
    println!("  arch {}, vocab {}, template {}", model.config.architecture,
        tokenizer.vocab_size(),
        match template { Template::ChatMl { .. } => "chatml", Template::Llama3 { .. } => "llama3", Template::Glm { .. } => "glm" });

    let mut chat = ChatState {
        session: model.new_session(SESSION_TOKENS),
        tokens: Vec::new(),
        snaps: std::collections::VecDeque::new(),
    };
    let rag_tools = RagTools::load();

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        // One request at a time: the model is the bottleneck, not the socket.
        if let Err(e) = handle(
            stream,
            &model,
            &tokenizer,
            &template,
            &model_name,
            &mut chat,
            &rag_tools,
        ) {
            println!("request failed: {e:#}");
        }
    }
    Ok(())
}

/// Touch every page of the weight mapping once, sequentially.
///
/// Without this the first request faults the whole file in access order
/// during inference - seconds of stutter billed to the first user. One
/// sequential pass at SSD streaming speed moves that cost to startup.
pub fn prewarm(mapping: &[u8]) {
    const PAGE: usize = 16384;
    let t0 = std::time::Instant::now();
    let mut acc = 0u8;
    let mut i = 0;
    while i < mapping.len() {
        acc ^= mapping[i];
        i += PAGE;
    }
    std::hint::black_box(acc);
    println!(
        "prewarmed {:.1} GiB of weights in {:.1}s",
        mapping.len() as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
}

fn handle(
    mut stream: TcpStream,
    model: &Model,
    tok: &Tokenizer,
    template: &Template,
    model_name: &str,
    chat: &mut ChatState,
    rag: &RagTools,
) -> Result<()> {
    let (request_line, body) = read_request(&mut stream)?;

    if request_line.starts_with("GET /health") {
        return respond(&mut stream, 200, &json!({"status": "ok"}));
    }
    if request_line.starts_with("GET /stats") {
        return respond(
            &mut stream,
            200,
            &json!({
                "model": model_name,
                "architecture": model.config.architecture,
                "context_used": chat.tokens.len(),
                "context_capacity": chat.session.capacity(),
                "n_layers": model.config.n_layers,
                "moe": model.config.moe.is_some(),
            }),
        );
    }
    if request_line.starts_with("GET / ") || request_line.starts_with("GET /index") {
        return respond_html(&mut stream, CHAT_PAGE);
    }
    if request_line.starts_with("OPTIONS ") {
        return respond(&mut stream, 200, &json!({}));
    }
    if !request_line.starts_with("POST /v1/chat/completions") {
        return respond(&mut stream, 404, &json!({"error": "unknown route"}));
    }

    let req: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return respond(&mut stream, 400, &json!({"error": format!("bad JSON: {e}")})),
    };
    let Some(raw_messages) = req["messages"].as_array() else {
        return respond(&mut stream, 400, &json!({"error": "messages[] required"}));
    };
    let tool_specs: Option<Vec<Value>> = match req["tools"].as_array() {
        Some(t) if !t.is_empty() => Some(t.to_vec()),
        _ if rag.cfg.enabled && rag.cfg.inject_tools_when_missing => Some(rag_default_tool_schemas()),
        _ => None,
    };
    let tools = tool_specs.as_ref().map(|t| t.as_slice());
    if tools.is_some() && matches!(template, Template::Llama3 { .. }) {
        // The Hermes convention below is what Qwen (ChatML) was trained on;
        // pretending Llama 3 understands it would fail silently.
        return respond(
            &mut stream,
            400,
            &json!({"error": "tools are only supported with the ChatML template"}),
        );
    }
    let messages = render_messages(raw_messages, tools);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.7) as f32;
    let top_p = req["top_p"].as_f64().unwrap_or(0.95) as f32;
    let repeat_penalty = req["repeat_penalty"].as_f64().unwrap_or(1.1) as f32;
    let streaming = req["stream"].as_bool().unwrap_or(false);

    if streaming {
        if tools.is_none() || !rag.cfg.enabled {
            run_inference(
                model,
                tok,
                template,
                chat,
                &messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                false,
                Some(&mut stream),
                true,
            )?;
            return Ok(());
        }
    }

    let completion = if tools.is_none() {
        run_inference(
            model,
            tok,
            template,
            chat,
            &messages,
            max_tokens,
            temperature,
            top_p,
            repeat_penalty,
            false,
            if streaming { Some(&mut stream) } else { None },
            !streaming,
        )?
    } else if !rag.cfg.enabled {
        run_inference(
            model,
            tok,
            template,
            chat,
            &messages,
            max_tokens,
            temperature,
            top_p,
            repeat_penalty,
            true,
            if streaming { Some(&mut stream) } else { None },
            !streaming,
        )?
    } else {
        let mut conversation = raw_messages.to_vec();
        let mut final_outcome: Option<CompletionResult> = None;
        for round in 0..rag.cfg.max_tool_rounds {
            let round_messages = render_messages(&conversation, tools);
            let mut outcome = run_inference(
                model,
                tok,
                template,
                chat,
                &round_messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                true,
                if streaming { Some(&mut stream) } else { None },
                !streaming,
            )?;
            if outcome.tool_calls.is_empty() {
                final_outcome = Some(outcome);
                break;
            }

            conversation.push(json!({
                "role": "assistant",
                "content": outcome.content.clone(),
                "tool_calls": outcome.tool_calls.clone(),
            }));

            for (i, call) in outcome.tool_calls.iter().enumerate() {
                let result = run_rag_tool(call, rag)?;
                let call_id = call["id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("call_{i}"));
                conversation.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }));
            }

            if round + 1 == rag.cfg.max_tool_rounds {
                outcome.tool_calls.clear();
                outcome.content = format!(
                    "tool loop exhausted after {} rounds. Last tool output could not be resolved into a final answer.",
                    rag.cfg.max_tool_rounds
                );
                final_outcome = Some(outcome);
                break;
            }
        }
        final_outcome.context("tool loop completed without output")?
    };

    if streaming && !tools.is_none() && rag.cfg.enabled {
        emit_stream_completion(&mut stream, &completion, false, &completion.finish)?;
        return Ok(());
    }

    respond(
        &mut stream,
        200,
        &json!({
            "id": "chatcmpl-allpaka",
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": Value::String(completion.content.clone()),
                    "tool_calls": if completion.tool_calls.is_empty() {
                        Value::Null
                    } else {
                        Value::Array(completion.tool_calls)
                    },
                },
                "finish_reason": completion.finish,
            }],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
        }),
    )
}

fn read_request(stream: &mut TcpStream) -> Result<(String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-request");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = find_header_end(&buf) {
            header_end = i;
            break;
        }
        if buf.len() > 1 << 20 {
            bail!("request headers over 1 MiB");
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let request_line = head.lines().next().unwrap_or_default().to_string();
    let content_length: usize = head
        .lines()
        .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);

    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-body");
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok((request_line, String::from_utf8_lossy(&body).into_owned()))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn respond(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let text = body.to_string();
    let reason = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
          Connection: close\r\n\r\n",
    )?;
    Ok(())
}

fn write_sse_event(stream: &mut TcpStream, body: &Value) -> Result<()> {
    stream.write_all(format!("data: {body}\n\n").as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn respond_html(stream: &mut TcpStream, page: &str) -> Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_section_lands_in_the_system_turn() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"type":"function","function":{"name":"search","parameters":{}}})];
        let out = render_messages(&msgs, Some(&tools));
        assert_eq!(out[0].0, "system");
        assert!(out[0].1.contains("<tools>"));
        assert!(out[0].1.contains("\"search\""));
        assert_eq!(out[1], ("user".into(), "hi".into()));
        // An existing system message is extended, not duplicated.
        let msgs2 = vec![json!({"role":"system","content":"be brief"}), msgs[0].clone()];
        let out2 = render_messages(&msgs2, Some(&tools));
        assert_eq!(out2.len(), 2);
        assert!(out2[0].1.starts_with("be brief"));
        assert!(out2[0].1.contains("<tools>"));
    }

    #[test]
    fn assistant_calls_and_tool_results_round_trip_into_hermes_text() {
        let msgs = vec![
            json!({"role":"user","content":"weather?"}),
            json!({"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]}),
            json!({"role":"tool","tool_call_id":"call_0","content":"{\"temp\":-3}"}),
            json!({"role":"tool","tool_call_id":"call_1","content":"{\"wind\":5}"}),
        ];
        let out = render_messages(&msgs, None);
        assert!(out[1].1.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Oslo\"}}\n</tool_call>"));
        // Both tool results merge into ONE user turn, each in its own wrapper.
        assert_eq!(out[2].0, "user");
        assert_eq!(out[2].1.matches("<tool_response>").count(), 2);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn tool_call_blocks_parse_and_malformed_ones_stay_in_content() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"rag_search\", \"arguments\": {\"q\": \"kv cache\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "rag_search");
        assert_eq!(calls[0]["function"]["arguments"], "{\"q\":\"kv cache\"}");
        // Broken JSON is not silently dropped.
        let (content2, calls2) = parse_tool_calls("<tool_call>not json</tool_call>done");
        assert!(calls2.is_empty());
        assert_eq!(content2, "<tool_call>not json</tool_call>done");
    }

    #[test]
    fn stream_hold_back_covers_partial_opening_tags() {
        assert_eq!(stream_safe_len("hello"), 5);
        assert_eq!(stream_safe_len("hello <tool_call>{"), 6);
        assert_eq!(stream_safe_len("hello <tool_c"), 6);
        assert_eq!(stream_safe_len("<"), 0);
        // A lone '<' that cannot start the tag is NOT held back.
        assert_eq!(stream_safe_len("a < b"), 5);
    }

    #[test]
    fn greedy_pick_is_argmax() {
        let mut s = Sampler::new(0.0, 0.95, 1.0);
        let mut logits = vec![0.0f32; 1000];
        logits[421] = 3.5;
        assert_eq!(s.pick(&logits, &[]), 421);
    }

    /// A dominant logit must win essentially always at moderate temperature,
    /// and the preselect must never return an index outside the vocabulary.
    #[test]
    fn sampled_pick_follows_the_dominant_token() {
        let mut s = Sampler::new(0.7, 0.95, 1.0);
        let mut logits = vec![0.0f32; 152_000];
        logits[123] = 20.0;
        for _ in 0..50 {
            assert_eq!(s.pick(&logits, &[]), 123);
        }
    }

    /// With a flat-ish distribution the sampler must return varied but valid
    /// tokens from the top candidates.
    #[test]
    fn sampled_pick_stays_within_the_top_candidates() {
        let mut s = Sampler::new(1.0, 0.9, 1.0);
        // 512 tokens clearly above the rest; everything sampled must be one.
        let mut logits = vec![-10.0f32; 152_000];
        for v in logits.iter_mut().take(512) {
            *v = 1.0;
        }
        for _ in 0..100 {
            let t = s.pick(&logits, &[]) as usize;
            assert!(t < 512, "picked {t} from outside the plateau");
        }
    }

    /// The repeat penalty must break a tie in favour of the token that has
    /// NOT just been emitted - the "либо-либо-либо" loop breaker.
    #[test]
    fn repeat_penalty_demotes_a_recently_emitted_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.1);
        let mut logits = vec![0.0f32; 1000];
        logits[7] = 3.0; // the looping token
        logits[8] = 2.9; // the almost-as-good alternative
        // Without history the loop token wins; with itself in the window it
        // is divided down below the alternative.
        assert_eq!(s.pick(&logits, &[]), 7);
        assert_eq!(s.pick(&logits, &[7]), 8);
        // Negative logits move away from zero instead.
        let mut neg = vec![-1.0f32; 10];
        neg[3] = -0.5;
        neg[4] = -0.52;
        assert_eq!(s.pick(&neg, &[3]), 4);
    }

    /// One occurrence or ten in the window, the penalty applies once: the
    /// classic repeat penalty is presence-based, not frequency-based.
    #[test]
    fn repeat_penalty_applies_once_per_distinct_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.3);
        let mut logits = vec![0.0f32; 100];
        logits[5] = 2.0;
        logits[6] = 1.6;
        // 2.0 / 1.3 = 1.538 < 1.6, so one mention already flips the choice;
        // many mentions must not penalise 6 as well by compounding wrongly.
        assert_eq!(s.pick(&logits, &[5, 5, 5, 5, 5]), 6);
    }
}

/// The built-in chat page: zero dependencies, talks to this same server.
const CHAT_PAGE: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>allpaka chat</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #log { display: flex; flex-direction: column; gap: .6rem; margin-bottom: 1rem; }
  .msg { padding: .6rem .9rem; border-radius: .7rem; white-space: pre-wrap; line-height: 1.4; }
  .user { background: #3b82f6; color: white; align-self: flex-end; max-width: 85%; }
  .bot  { background: rgba(127,127,127,.15); align-self: flex-start; max-width: 85%; }
  .meta { opacity: .5; font-size: .75rem; align-self: flex-start; }
  .thinking { opacity: .55; font-style: italic; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .6rem .8rem; border-radius: .6rem; border: 1px solid rgba(127,127,127,.4); font-size: 1rem; }
  button { padding: .6rem 1.1rem; border-radius: .6rem; border: none; background: #3b82f6; color: white; font-size: 1rem; }
  button:disabled { opacity: .5; }
  h1 { font-size: 1.1rem; opacity: .7; }
  details { opacity: .6; font-size: .85rem; }
  #stats { display: flex; align-items: center; gap: .8rem; font-size: .78rem;
           opacity: .75; margin-bottom: 1rem; flex-wrap: wrap; }
  #ctxbar { width: 160px; height: 6px; border-radius: 3px;
            background: rgba(127,127,127,.25); overflow: hidden; }
  #ctxfill { height: 100%; width: 0%; background: #3b82f6; transition: width .3s; }
</style></head><body>
<h1>allpaka · собственный движок · чат</h1>
<div id="stats">
  <span id="s-model">…</span>
  <span>контекст: <span id="s-ctx">0 / 0</span></span>
  <div id="ctxbar"><div id="ctxfill"></div></div>
  <span id="s-speed"></span>
</div>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Спросите что-нибудь..." autofocus autocomplete="off">
<button id="send">→</button></form>
<script>
const log = document.getElementById('log');
const history = [];
function setStats(used, cap, speed) {
  document.getElementById('s-ctx').textContent =
    `${used.toLocaleString('ru')} / ${cap.toLocaleString('ru')} (${Math.round(used/cap*100)}%)`;
  document.getElementById('ctxfill').style.width = Math.min(100, used/cap*100) + '%';
  if (speed) document.getElementById('s-speed').textContent = speed;
}
fetch('/stats').then(r => r.json()).then(s => {
  document.getElementById('s-model').textContent =
    s.model + (s.moe ? ' (MoE)' : '') + ' · ' + s.n_layers + ' слоёв';
  setStats(s.context_used, s.context_capacity, '');
}).catch(() => {});
function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  d.scrollIntoView();
  return d;
}
let aborter = null;
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q');
  const btn = document.getElementById('send');
  if (aborter) { aborter.abort(); return; } // the button doubles as "stop"
  const text = q.value.trim();
  if (!text) return;
  q.value = '';
  aborter = new AbortController();
  btn.textContent = '⏹';
  add('user', text);
  history.push({role: 'user', content: text});
  const busy = add('bot', '…');
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = 'думаю…';
  log.appendChild(meta);
  const t0 = performance.now();
  let raw = '', tokens = 0, timing = null, stopped = false;
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: aborter.signal,
      body: JSON.stringify({messages: history, max_tokens: 1024, temperature: 0.7, stream: true})
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const events = buf.split('\n\n');
      buf = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        const j = JSON.parse(payload);
        const delta = j.choices && j.choices[0].delta && j.choices[0].delta.content;
        if (delta) raw += delta;
        if (j.tokens_generated) tokens = j.tokens_generated;
        if (j.usage) tokens = j.usage.completion_tokens;
        if (j.timing) timing = j.timing;
        // Render: while the model is inside its think block, stream the
        // thoughts dimmed; once the answer starts, show the answer.
        let reply = raw;
        const m = reply.match(/^<think>([\s\S]*?)(<\/think>\s*|$)/);
        let think = null;
        if (m) { think = m[1].trim(); reply = reply.slice(m[0].length); }
        if (reply) {
          busy.classList.remove('thinking');
          busy.textContent = reply;
        } else if (think) {
          busy.classList.add('thinking');
          busy.textContent = '💭 ' + (think.length > 600 ? '…' + think.slice(-600) : think);
        } else {
          busy.textContent = '…';
        }
        const secs = (performance.now() - t0) / 1000;
        const tps = tokens > 0 ? (tokens / secs).toFixed(1) : '?';
        meta.textContent = `${tokens} токенов · ${secs.toFixed(1)} с · ${tps} tok/s`;
        meta.scrollIntoView();
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      stopped = true;
    } else {
      busy.textContent = 'ошибка: ' + err;
      aborter = null;
      btn.textContent = '→';
      q.focus();
      return;
    }
  }
  // Final render: full reply, or whatever arrived before the stop.
  let reply = raw;
  const m = reply.match(/^<think>([\s\S]*?)(<\/think>\s*|$)/);
  let think = null;
  if (m) { think = m[1].trim(); reply = reply.slice(m[0].length); }
  busy.classList.remove('thinking');
  busy.textContent = reply || (stopped ? '(остановлено на размышлениях)' : '(пусто)');
  if (think) {
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = 'размышления';
    det.appendChild(sum);
    const pre = document.createElement('div');
    pre.textContent = think;
    det.appendChild(pre);
    busy.before(det);
  }
  // Partial replies stay in the history: the server cache holds those tokens
  // too, so the next turn still reuses the prefix.
  history.push({role: 'assistant', content: reply});
  if (timing) {
    meta.textContent = `${tokens} токенов · генерация ${timing.decode_secs.toFixed(1)} с · `
      + `${timing.tokens_per_sec.toFixed(1)} tok/s · промпт ${timing.prefill_secs.toFixed(1)} с`
      + (timing.cached_tokens ? ` (кэш: ${timing.cached_tokens} ток)` : '');
    if (timing.context_used) {
      setStats(timing.context_used, timing.context_capacity,
        timing.tokens_per_sec.toFixed(1) + ' tok/s');
    }
  } else if (stopped) {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    meta.textContent = `${tokens} токенов · ${secs} с · остановлено`;
  }
  aborter = null;
  btn.textContent = '→';
  q.focus();
});
</script></body></html>"#;
