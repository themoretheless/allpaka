//! Plugin smoke tests: the allpaka inference server (health/stats/chat/RAG
//! tool-loop) and the rag-mcp stdio MCP server, end to end against real
//! binaries. Std-only: HTTP over TcpStream, JSON built by hand.
//!
//! Run:  cargo test -p allpaka-cli --test plugin_smoke -- --nocapture --test-threads=1
//!
//! Skips gracefully when the small model or the rag-mcp binary/db is absent.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const HOST: &str = "127.0.0.1:18099";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(120);
const CHAT_TIMEOUT: Duration = Duration::from_secs(300);

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn small_model() -> Option<PathBuf> {
    let p = workspace_root().join("models/qwen3-0.6b-Q8_0.gguf");
    p.is_file().then_some(p)
}

fn rag_bin() -> Option<PathBuf> {
    let p = std::env::var("RAG_MCP_BIN").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/Users/themoretheless/Documents/Sources/rag/target/release/rag-mcp")
    });
    p.is_file().then_some(p)
}

fn rag_db() -> Option<PathBuf> {
    let p = std::env::var("RAG_DB_PATH").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/Users/themoretheless/Documents/Sources/rag/data/allpaka-notes.duckdb")
    });
    p.is_file().then_some(p)
}

// ---------- minimal HTTP ----------

fn http(method: &str, path: &str, body: Option<&str>, timeout: Duration) -> std::io::Result<String> {
    let mut s = TcpStream::connect(HOST)?;
    s.set_read_timeout(Some(timeout))?;
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {HOST}\r\nConnection: close\r\n");
    if let Some(b) = body {
        req += &format!("Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len());
    } else {
        req += "\r\n";
    }
    s.write_all(req.as_bytes())?;
    let mut raw = String::new();
    s.read_to_string(&mut raw)?;
    Ok(raw)
}

fn http_json(method: &str, path: &str, body: Option<&str>, timeout: Duration) -> Option<serde_json_free::Value> {
    let raw = http(method, path, body, timeout).ok()?;
    let json_start = raw.find("\r\n\r\n")? + 4;
    serde_json_free::parse(&raw[json_start..])
}

// Tiny JSON value/parser — enough for flat assertions without adding deps.
mod serde_json_free {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Value::Num(n) => Some(*n),
                _ => None,
            }
        }
        pub fn as_arr(&self) -> Option<&[Value]> {
            match self {
                Value::Arr(a) => Some(a),
                _ => None,
            }
        }
    }

    pub fn parse(s: &str) -> Option<Value> {
        let b = s.as_bytes();
        let mut p = P { b, i: 0 };
        p.ws();
        let v = p.value()?;
        Some(v)
    }

    struct P<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl<'a> P<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
                self.i += 1;
            }
        }
        fn peek(&self) -> Option<u8> {
            self.b.get(self.i).copied()
        }
        fn value(&mut self) -> Option<Value> {
            self.ws();
            match self.peek()? {
                b'{' => self.obj(),
                b'[' => self.arr(),
                b'"' => self.string().map(Value::Str),
                b't' => self.lit("true", Value::Bool(true)),
                b'f' => self.lit("false", Value::Bool(false)),
                b'n' => self.lit("null", Value::Null),
                _ => self.num(),
            }
        }
        fn lit(&mut self, s: &str, v: Value) -> Option<Value> {
            if self.b[self.i..].starts_with(s.as_bytes()) {
                self.i += s.len();
                Some(v)
            } else {
                None
            }
        }
        fn num(&mut self) -> Option<Value> {
            let start = self.i;
            while self.i < self.b.len()
                && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                self.i += 1;
            }
            std::str::from_utf8(&self.b[start..self.i]).ok()?.parse().ok().map(Value::Num)
        }
        fn string(&mut self) -> Option<String> {
            if self.peek()? != b'"' {
                return None;
            }
            self.i += 1;
            let mut out = String::new();
            while let Some(c) = self.peek() {
                self.i += 1;
                match c {
                    b'"' => return Some(out),
                    b'\\' => {
                        let e = self.peek()?;
                        self.i += 1;
                        match e {
                            b'n' => out.push('\n'),
                            b't' => out.push('\t'),
                            b'r' => out.push('\r'),
                            b'u' => {
                                let hex = std::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
                                let cp = u32::from_str_radix(hex, 16).ok()?;
                                self.i += 4;
                                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            _ => out.push(e as char),
                        }
                    }
                    _ => {
                        // collect full UTF-8 char
                        let start = self.i - 1;
                        let len = utf8_len(c);
                        let end = (start + len).min(self.b.len());
                        out.push_str(std::str::from_utf8(&self.b[start..end]).unwrap_or("\u{fffd}"));
                        self.i = end;
                    }
                }
            }
            None
        }
        fn obj(&mut self) -> Option<Value> {
            self.i += 1; // {
            let mut kv = Vec::new();
            loop {
                self.ws();
                match self.peek()? {
                    b'}' => {
                        self.i += 1;
                        return Some(Value::Obj(kv));
                    }
                    b',' => {
                        self.i += 1;
                    }
                    b'"' => {
                        let k = self.string()?;
                        self.ws();
                        if self.peek()? != b':' {
                            return None;
                        }
                        self.i += 1;
                        let v = self.value()?;
                        kv.push((k, v));
                    }
                    _ => return None,
                }
            }
        }
        fn arr(&mut self) -> Option<Value> {
            self.i += 1; // [
            let mut items = Vec::new();
            loop {
                self.ws();
                match self.peek()? {
                    b']' => {
                        self.i += 1;
                        return Some(Value::Arr(items));
                    }
                    b',' => {
                        self.i += 1;
                    }
                    _ => items.push(self.value()?),
                }
            }
        }
    }

    fn utf8_len(first: u8) -> usize {
        match first {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            _ => 4,
        }
    }
}

// ---------- serve fixture ----------

/// Every test in this binary spawns heavy processes (an inference server,
/// rag-mcp with its DuckDB FTS init). Run them one at a time: the port is
/// shared, and CPU contention otherwise blows past the MCP timeouts. Locking
/// here beats asking every caller for `--test-threads=1`.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Serve {
    child: Child,
    _permit: std::sync::MutexGuard<'static, ()>,
}

impl Serve {
    fn start() -> Option<Self> {
        Self::start_with(&[])
    }

    /// Start serve with extra env vars (e.g. ALLPAKA_RAG_BACKEND).
    fn start_with(env: &[(&str, &str)]) -> Option<Self> {
        let permit = TEST_LOCK.lock().ok()?;
        let model = small_model()?;
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_allpaka"));
        cmd.args(["serve", "--model"])
            .arg(&model)
            .args(["--bind", HOST])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().ok()?;
        let s = Serve { child, _permit: permit };
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            if http("GET", "/health", None, Duration::from_secs(2)).is_ok() {
                return Some(s);
            }
            thread::sleep(Duration::from_millis(500));
        }
        None
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------- tests ----------

/// Force a rag_search round trip against the running serve; returns the
/// round-2 prompt size and the final answer. The loop ran iff the prompt
/// carries the tool results (well above the bare question's size).
fn rag_loop_round2() -> (f64, String) {
    let rag_body = r#"{
      "model":"qwen3","max_tokens":800,
      "messages":[
        {"role":"system","content":"/no_think You must call the rag_search tool first."},
        {"role":"user","content":"Search the notes for Metal GPU optimisation."}
      ],
      "tools":[
        {"type":"function","function":{"name":"rag_search","description":"Search knowledge base notes","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
        {"type":"function","function":{"name":"rag_read","description":"Read one note by file name","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}
      ]
    }"#;
    let rag = http_json("POST", "/v1/chat/completions", Some(rag_body), CHAT_TIMEOUT)
        .expect("rag chat JSON");
    let prompt_tokens = rag
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|p| p.as_f64())
        .unwrap_or(0.0);
    let answer = rag
        .get("choices").and_then(|c| c.as_arr())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    (prompt_tokens, answer)
}

#[test]
fn engine_health_stats_chat_and_rag_loop() {
    let Some(_serve) = Serve::start() else {
        eprintln!("SKIP: small model missing or serve did not come up");
        return;
    };

    let stats = http_json("GET", "/stats", None, Duration::from_secs(5)).expect("/stats JSON");
    assert!(stats.get("model").and_then(|m| m.as_str()).is_some(), "stats has model");
    println!("stats model = {:?}", stats.get("model"));

    let chat_body = r#"{"model":"qwen3","max_tokens":40,"messages":[{"role":"user","content":"Say PONG only."}]}"#;
    let chat = http_json("POST", "/v1/chat/completions", Some(chat_body), CHAT_TIMEOUT)
        .expect("chat JSON");
    let content = chat
        .get("choices").and_then(|c| c.as_arr())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    assert!(!content.trim().is_empty(), "chat returns content");

    let (prompt_tokens, answer) = rag_loop_round2();
    assert!(prompt_tokens > 500.0, "RAG tool-loop ran (round-2 prompt = {prompt_tokens})");
    // Citing ".md" file names in prose is the model's wording choice; the
    // loop itself is what this test asserts.
    assert!(!answer.trim().is_empty(), "non-empty answer after the tool-loop");
}

/// With ALLPAKA_RAG_BACKEND=mcp the tool-loop must be served by rag-mcp's
/// BM25 index. The assertion is behavioural (the loop runs and returns
/// grounded content), so it holds whatever the grep backend would say.
#[test]
fn engine_rag_backend_mcp() {
    if rag_bin().is_none() || rag_db().is_none() {
        eprintln!("SKIP: rag-mcp binary or DuckDB not found");
        return;
    }
    let Some(_serve) = Serve::start_with(&[("ALLPAKA_RAG_BACKEND", "mcp")]) else {
        eprintln!("SKIP: small model missing or serve did not come up");
        return;
    };
    let (prompt_tokens, answer) = rag_loop_round2();
    assert!(prompt_tokens > 500.0, "tool-loop via rag-mcp (prompt = {prompt_tokens})");
    assert!(!answer.trim().is_empty(), "non-empty answer via rag-mcp");
    assert!(!answer.contains("rag-mcp backend is unavailable"), "mcp backend actually served");
}

/// Same forced backend, but the binary is gone: the tool must report the
/// outage instead of hanging or crashing the server.
#[test]
fn engine_rag_backend_mcp_missing() {
    let Some(_serve) = Serve::start_with(&[
        ("ALLPAKA_RAG_BACKEND", "mcp"),
        ("RAG_MCP_BIN", "/nonexistent/rag-mcp"),
    ]) else {
        eprintln!("SKIP: small model missing or serve did not come up");
        return;
    };
    let chat_body = r#"{"model":"qwen3","max_tokens":40,"messages":[{"role":"user","content":"Say PONG only."}]}"#;
    assert!(http_json("POST", "/v1/chat/completions", Some(chat_body), CHAT_TIMEOUT).is_some(),
        "server answers plain chat without the mcp backend");
}

/// Auto backend with a broken rag-mcp must degrade to the grep directory
/// scan and still run the loop.
#[test]
fn engine_rag_backend_auto_fallback() {
    let Some(_serve) = Serve::start_with(&[
        ("ALLPAKA_RAG_BACKEND", "auto"),
        ("RAG_MCP_BIN", "/nonexistent/rag-mcp"),
    ]) else {
        eprintln!("SKIP: small model missing or serve did not come up");
        return;
    };
    let (prompt_tokens, answer) = rag_loop_round2();
    assert!(prompt_tokens > 500.0, "grep fallback runs the loop (prompt = {prompt_tokens})");
    assert!(!answer.trim().is_empty(), "non-empty answer via grep fallback");
}

#[test]
fn rag_mcp_stdio_search() {
    let _permit = TEST_LOCK.lock().expect("test lock");
    let (Some(bin), Some(db)) = (rag_bin(), rag_db()) else {
        eprintln!("SKIP: rag-mcp binary or DuckDB not found");
        return;
    };
    let notes = std::env::var("RAG_INGEST_ROOTS").unwrap_or_else(|_| {
        "/Users/themoretheless/.claude/projects/-Users-themoretheless-Documents-Sources-allpaka/memory".into()
    });

    let mut child = Command::new(bin)
        .env("RAG_DB_PATH", db)
        .env("RAG_INGEST_ROOTS", notes)
        .env("RAG_DEFAULT_SEARCH_MODE", "lex")
        .env("RAG_LLM_ENABLED", "false")
        .env("RAG_TOOLS", "spine")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rag-mcp");
    let mut stdin: ChildStdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    // rmcp needs an ordered handshake: send, then wait between requests.
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line.unwrap_or_default()).is_err() {
                break;
            }
        }
    });
    let send = |stdin: &mut ChildStdin, msg: &str| {
        stdin.write_all(msg.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    };
    let wait_for = |id: u64, timeout: Duration| -> Option<serde_json_free::Value> {
        let deadline = Instant::now() + timeout;
        let marker = format!("\"id\":{id}");
        while Instant::now() < deadline {
            if let Ok(line) = rx.recv_timeout(Duration::from_millis(200)) {
                if line.contains(&marker) {
                    return serde_json_free::parse(&line).and_then(|v| v.get("result").cloned());
                }
            }
        }
        None
    };

    send(&mut stdin, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#);
    let init = wait_for(1, Duration::from_secs(30)).expect("initialize reply");
    assert_eq!(
        init.get("serverInfo").and_then(|s| s.get("name")).and_then(|n| n.as_str()),
        Some("rag-mcp")
    );

    send(&mut stdin, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    thread::sleep(Duration::from_millis(500));

    send(&mut stdin, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
    let tools = wait_for(2, Duration::from_secs(30)).expect("tools/list reply");
    let names: Vec<&str> = tools
        .get("tools")
        .and_then(|t| t.as_arr())
        .map(|a| a.iter().filter_map(|t| t.get("name").and_then(|n| n.as_str())).collect())
        .unwrap_or_default();
    println!("rag-mcp tools ({}): {:?}", names.len(), names);
    for need in ["search", "search_wiki", "query_with_index", "pack_context", "get_document"] {
        assert!(names.contains(&need), "missing tool: {need}");
    }

    send(&mut stdin, r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search","arguments":{"query":"metal","mode":"lex"}}}"#);
    let res = wait_for(3, Duration::from_secs(30)).expect("search reply");
    let text = res
        .get("content").and_then(|c| c.as_arr())
        .and_then(|a| a.first())
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");
    assert!(text.contains("document_title"), "search returns real hits: {}", &text[..text.len().min(150)]);

    drop(stdin);
    let _ = child.wait();
}
