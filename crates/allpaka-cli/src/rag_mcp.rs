//! stdio JSON-RPC client for the rag-mcp retrieval server (DuckDB BM25 +
//! wiki index). `serve` spawns it as a child and translates the rag_search /
//! rag_read tool calls into MCP requests, so inference gets the same indexed
//! retrieval the Kimi plugin sees instead of a directory grep.
//!
//! rmcp requires an ordered handshake - requests sent before the
//! `initialized` notification are silently dropped - so spawn() performs
//! initialize/initialized synchronously before serving any call. A reader
//! thread owns the child's stdout and forwards lines; rpc() matches replies
//! by id with a timeout. Any failure poisons the client and the caller falls
//! back to the grep backend.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const RPC_TIMEOUT: Duration = Duration::from_secs(30);

/// Where the pieces live; every field has an env override so the same binary
/// serves the dev repo and the installed plugin.
pub struct RagMcpConfig {
    pub bin: PathBuf,
    pub db: PathBuf,
    pub notes_dir: PathBuf,
    pub search_mode: String,
}

impl RagMcpConfig {
    /// Defaults point at the sibling rag repo; env wins.
    pub fn from_env(notes_dir: &Path) -> Self {
        RagMcpConfig {
            bin: std::env::var("RAG_MCP_BIN").map(PathBuf::from).unwrap_or_else(|_| {
                PathBuf::from("/Users/themoretheless/Documents/Sources/rag/target/release/rag-mcp")
            }),
            db: std::env::var("RAG_DB_PATH").map(PathBuf::from).unwrap_or_else(|_| {
                PathBuf::from(
                    "/Users/themoretheless/Documents/Sources/rag/data/allpaka-notes.duckdb",
                )
            }),
            notes_dir: notes_dir.to_path_buf(),
            search_mode: std::env::var("RAG_MCP_SEARCH_MODE").unwrap_or_else(|_| "lex".into()),
        }
    }

    pub fn available(&self) -> bool {
        self.bin.is_file() && self.db.is_file()
    }
}

pub struct RagMcp {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::Receiver<String>,
    next_id: u64,
}

impl RagMcp {
    /// Spawn the server and complete the MCP handshake.
    pub fn spawn(cfg: &RagMcpConfig) -> Result<Self> {
        let mut child = Command::new(&cfg.bin)
            .env("RAG_DB_PATH", &cfg.db)
            .env("RAG_INGEST_ROOTS", &cfg.notes_dir)
            .env("RAG_DEFAULT_SEARCH_MODE", &cfg.search_mode)
            .env("RAG_LLM_ENABLED", "false")
            .env("RAG_TOOLS", "spine")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning {}", cfg.bin.display()))?;
        let stdin = child.stdin.take().context("rag-mcp stdin")?;
        let stdout = child.stdout.take().context("rag-mcp stdout")?;

        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });

        let mut mcp = RagMcp { child, stdin, lines: rx, next_id: 0 };
        let init = mcp.rpc(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "allpaka-serve", "version": env!("CARGO_PKG_VERSION")},
            }),
        )?;
        let server = init["result"]["serverInfo"]["name"].as_str().unwrap_or("");
        if server != "rag-mcp" {
            bail!("unexpected MCP server: {server:?}");
        }
        mcp.notify("notifications/initialized")?;
        Ok(mcp)
    }

    fn notify(&mut self, method: &str) -> Result<()> {
        let msg = json!({"jsonrpc": "2.0", "method": method});
        self.stdin.write_all(msg.to_string().as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush().context("writing MCP notification")
    }

    /// One request/reply round trip. Replies are matched by id; anything else
    /// on the wire (server logs are on stderr, so this should be rare) is
    /// skipped until the timeout.
    fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.stdin.write_all(msg.to_string().as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush().context("writing MCP request")?;

        let marker = format!("\"id\":{id}");
        let deadline = std::time::Instant::now() + RPC_TIMEOUT;
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(1))
                .with_context(|| format!("MCP {method}: reply timeout / server gone"))?;
            if !line.contains(&marker) {
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "MCP {method}: no reply with id {id}"
                );
                continue;
            }
            let reply: Value =
                serde_json::from_str(&line).with_context(|| format!("MCP {method}: bad JSON"))?;
            if let Some(err) = reply.get("error") {
                bail!("MCP {method} error: {err}");
            }
            return Ok(reply);
        }
    }

    /// tools/call; returns the concatenated text content blocks.
    fn call(&mut self, tool: &str, args: Value) -> Result<String> {
        let reply = self.rpc("tools/call", json!({"name": tool, "arguments": args}))?;
        let mut text = String::new();
        if let Some(blocks) = reply["result"]["content"].as_array() {
            for b in blocks {
                if let Some(t) = b["text"].as_str() {
                    text.push_str(t);
                }
            }
        }
        Ok(text)
    }

    /// Raw hits as parsed JSON values (chunk objects with document_title,
    /// content, score, document_id, ...).
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<Value>> {
        let text = self.call(
            "search",
            json!({"query": query, "mode": "lex", "top_k": top_k}),
        )?;
        let hits: Vec<Value> =
            serde_json::from_str(&text).with_context(|| "parsing rag-mcp search hits")?;
        Ok(hits)
    }

    /// Full text of one document.
    pub fn get_document(&mut self, document_id: &str) -> Result<String> {
        let text = self.call("get_document", json!({"document_id": document_id}))?;
        let doc: Value =
            serde_json::from_str(&text).with_context(|| "parsing rag-mcp document")?;
        doc["content"]
            .as_str()
            .map(str::to_string)
            .with_context(|| "rag-mcp document has no content")
    }
}

impl Drop for RagMcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
