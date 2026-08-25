//! Client side of the `serve` API: status probe, one-off chat, and the RAG
//! tool-loop smoke test. Talks to a running `allpaka serve` (or any server
//! with the same routes) over plain HTTP/1.1, same style as verify.rs.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub const DEFAULT_ADDR: &str = "127.0.0.1:8099";

/// The tool schemas `serve` implements server-side. Kept here so `chat --rag`
/// and `rag-test` cannot drift from what the engine actually executes.
fn rag_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "rag_search",
            "description": "Search the local engineering knowledge base notes.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "rag_read",
            "description": "Read one knowledge-base note in full by its file name.",
            "parameters": {"type": "object",
                           "properties": {"name": {"type": "string"}},
                           "required": ["name"]}}},
    ])
}

/// One HTTP request over a fresh connection; no keep-alive.
fn request(method: &str, addr: &str, path: &str, body: Option<&Value>) -> Result<Value> {
    let mut stream =
        TcpStream::connect(addr).with_context(|| format!("connecting to allpaka serve at {addr}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(600)))
        .context("setting read timeout")?;
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    if body.is_some() {
        head += &format!("Content-Type: application/json\r\nContent-Length: {}\r\n", payload.len());
    }
    stream.write_all(format!("{head}\r\n{payload}").as_bytes())?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let text = String::from_utf8_lossy(&response);
    let (head, rest) = text
        .split_once("\r\n\r\n")
        .with_context(|| format!("malformed HTTP response from {path}"))?;
    let status = head.lines().next().unwrap_or("");
    if !status.contains("200") {
        bail!("{path} returned {status}: {}", &rest[..rest.len().min(300)]);
    }
    let body_text = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(rest)?
    } else {
        rest.to_string()
    };
    serde_json::from_str(&body_text).with_context(|| format!("parsing {path} response as JSON"))
}

fn dechunk(raw: &str) -> Result<String> {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let (size_line, tail) = rest.split_once("\r\n").context("truncated chunk header")?;
        let size = usize::from_str_radix(size_line.trim(), 16).context("bad chunk size")?;
        if size == 0 {
            return Ok(out);
        }
        out.push_str(tail.get(..size).context("truncated chunk body")?);
        rest = tail.get(size + 2..).context("truncated chunk trailer")?;
    }
}

fn get(addr: &str, path: &str) -> Result<Value> {
    request("GET", addr, path, None)
}

fn post(addr: &str, path: &str, body: &Value) -> Result<Value> {
    request("POST", addr, path, Some(body))
}

/// GET /health + GET /stats; prints what the server is running.
pub fn status(addr: &str) -> Result<()> {
    get(addr, "/health").context("server is not responding (start it with `allpaka serve`)")?;
    let stats = get(addr, "/stats")?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}

/// One chat round trip. With `rag`, hands the server the rag_search/rag_read
/// schemas so its built-in tool-loop can run.
pub fn chat(addr: &str, prompt: &str, system: Option<&str>, rag: bool, max_tokens: u32, model: &str) -> Result<()> {
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(json!({"role": "system", "content": s}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let mut body = json!({"model": model, "messages": messages, "max_tokens": max_tokens});
    if rag {
        body["tools"] = rag_tools();
    }
    let out = post(addr, "/v1/chat/completions", &body)?;
    let msg = &out["choices"][0]["message"];
    if msg.get("tool_calls").is_some_and(|c| c.is_array()) {
        println!("{}", serde_json::to_string_pretty(&msg["tool_calls"])?);
    }
    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
        println!("{content}");
    }
    let u = &out["usage"];
    eprintln!(
        "--- usage: prompt={} completion={}",
        u["prompt_tokens"], u["completion_tokens"]
    );
    Ok(())
}

/// The RAG smoke test: force rag_search, then check the tool-loop actually
/// ran (the round-2 prompt carries the search results). Note-name citation in
/// the prose is model-dependent, so it is reported but not asserted.
pub fn rag_test(addr: &str, model: &str) -> Result<()> {
    let body = json!({
        "model": model,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": "/no_think You must call the rag_search tool first."},
            {"role": "user", "content": "Search the notes for Metal GPU optimisation."},
        ],
        "tools": rag_tools(),
    });
    let out = post(addr, "/v1/chat/completions", &body)?;
    let answer = out["choices"][0]["message"]["content"].as_str().unwrap_or("");
    println!("{answer}");
    let prompt_tokens = out["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    eprintln!("--- usage: prompt={prompt_tokens} completion={}", out["usage"]["completion_tokens"]);
    if prompt_tokens <= 500 {
        bail!("RAG tool-loop did not run: round-2 prompt is only {prompt_tokens} tokens");
    }
    if answer.trim().is_empty() {
        bail!("empty answer after the tool-loop");
    }
    // Whether the prose happens to mention ".md" file names is up to the
    // model; the loop itself is what this test asserts.
    if !answer.contains(".md") {
        eprintln!("--- note: answer does not cite note file names (model's choice of wording)");
    }
    eprintln!("--- rag-test ok: tool-loop ran");
    Ok(())
}
