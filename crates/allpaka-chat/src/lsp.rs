//! Async stdio Language Server Protocol (LSP) client using `Content-Length` JSON-RPC 2.0
//! framing, cached per `(workspace_root, language)`.

use crate::context::Project;
use crate::ext::resolve_project_root;
use crate::storage;
use crate::tools;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{oneshot, Mutex as AsyncMutex};

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "lsp",
            "description": "Query Language Server Protocol intelligence (`definition`, `references`, `hover`, `document_symbols`, `workspace_symbols`, `diagnostics`, `status`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["definition", "references", "hover", "document_symbols", "workspace_symbols", "diagnostics", "status"]
                    },
                    "path": { "type": "string", "description": "Workspace-relative file path (required for file-scoped operations)" },
                    "line": { "type": "integer", "description": "1-based line number" },
                    "character": { "type": "integer", "description": "1-based column number" },
                    "query": { "type": "string", "description": "Symbol query for workspace_symbols" },
                    "language": { "type": "string", "description": "Optional explicit language key (`rust`, `python`, `typescript`, `go`, `cpp`)" }
                },
                "required": ["operation"]
            }
        }
    })
}

fn detect_language(path: &Path, explicit: Option<&str>) -> Option<&'static str> {
    if let Some(lang) = explicit {
        return match lang.to_ascii_lowercase().as_str() {
            "rust" | "rs" => Some("rust"),
            "typescript" | "ts" | "javascript" | "js" => Some("typescript"),
            "python" | "py" => Some("python"),
            "go" => Some("go"),
            "c" | "cpp" | "c++" => Some("cpp"),
            _ => None,
        };
    }
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "rs" => Some("rust"),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => Some("typescript"),
        "py" | "pyi" => Some("python"),
        "go" => Some("go"),
        "c" | "cc" | "cpp" | "cxx" | "h" | "hpp" => Some("cpp"),
        _ => None,
    }
}

fn command_for_language(lang: &str) -> Vec<String> {
    let env_key = format!("ALLPAKA_LSP_{}", lang.to_ascii_uppercase());
    if let Ok(custom) = std::env::var(&env_key).or_else(|_| std::env::var("ALLPAKA_LSP_CMD")) {
        let parts: Vec<String> = custom.split_whitespace().map(ToOwned::to_owned).collect();
        if !parts.is_empty() {
            return parts;
        }
    }
    match lang {
        "rust" => vec!["rust-analyzer".into()],
        "typescript" => vec!["typescript-language-server".into(), "--stdio".into()],
        "python" => vec!["pyright-langserver".into(), "--stdio".into()],
        "go" => vec!["gopls".into()],
        "cpp" => vec!["clangd".into()],
        _ => vec![],
    }
}

fn path_to_uri(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.to_string_lossy().replace(' ', "%20"))
}

struct LspClient {
    stdin: AsyncMutex<ChildStdin>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>,
    diagnostics: Mutex<HashMap<String, Vec<Value>>>,
    opened_versions: Mutex<HashMap<String, (String, i32)>>,
    _child: AsyncMutex<Child>,
    cmd_label: String,
}

impl LspClient {
    async fn start(root: &Path, argv: &[String]) -> Result<Arc<Self>> {
        if argv.is_empty() {
            bail!("No LSP command configured");
        }
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .current_dir(root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("LSP server `{}` unavailable", argv[0]))?;
        let stdin = child.stdin.take().context("LSP stdin unavailable")?;
        let stdout = child.stdout.take().context("LSP stdout unavailable")?;

        let client = Arc::new(Self {
            stdin: AsyncMutex::new(stdin),
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            diagnostics: Mutex::new(HashMap::new()),
            opened_versions: Mutex::new(HashMap::new()),
            _child: AsyncMutex::new(child),
            cmd_label: argv.join(" "),
        });

        let reader_client = client.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            while let Ok(Some(msg)) = read_lsp_frame(&mut reader).await {
                if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                    if msg.get("method").is_some() {
                        // Server-to-client request: reply null cleanly so server never blocks.
                        let _ = reader_client
                            .send_raw(&json!({ "jsonrpc": "2.0", "id": id, "result": Value::Null }))
                            .await;
                    } else if let Some(tx) = reader_client.pending.lock().unwrap().remove(&id) {
                        if let Some(err) = msg.get("error") {
                            let _ = tx.send(Err(anyhow::anyhow!("LSP error: {err}")));
                        } else {
                            let _ = tx.send(Ok(msg.get("result").cloned().unwrap_or(Value::Null)));
                        }
                    }
                } else if msg.get("method").and_then(Value::as_str)
                    == Some("textDocument/publishDiagnostics")
                {
                    if let Some(params) = msg.get("params") {
                        if let (Some(uri), Some(diags)) = (
                            params.get("uri").and_then(Value::as_str),
                            params.get("diagnostics").and_then(Value::as_array),
                        ) {
                            reader_client
                                .diagnostics
                                .lock()
                                .unwrap()
                                .insert(uri.to_owned(), diags.clone());
                        }
                    }
                }
            }
            // Fail any remaining waiters on EOF.
            let drained: Vec<_> = reader_client
                .pending
                .lock()
                .unwrap()
                .drain()
                .map(|(_, tx)| tx)
                .collect();
            for tx in drained {
                let _ = tx.send(Err(anyhow::anyhow!("LSP connection closed")));
            }
        });

        let root_uri = path_to_uri(root);
        client
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": root_uri,
                    "capabilities": {
                        "textDocument": {
                            "synchronization": { "didSave": true },
                            "hover": { "contentFormat": ["markdown", "plaintext"] },
                            "definition": {},
                            "references": {},
                            "documentSymbol": {},
                            "publishDiagnostics": {}
                        },
                        "workspace": { "symbol": {} }
                    }
                }),
            )
            .await?;
        client.notify("initialized", json!({})).await?;
        Ok(client)
    }

    async fn send_raw(&self, payload: &Value) -> Result<()> {
        let body = serde_json::to_vec(payload)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(header.as_bytes()).await?;
        stdin.write_all(&body).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send_raw(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(err) = self
            .send_raw(&json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            }))
            .await
        {
            self.pending.lock().unwrap().remove(&id);
            return Err(err);
        }
        match tokio::time::timeout(std::time::Duration::from_secs(15), rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => bail!("LSP server closed channel"),
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                bail!("LSP request `{method}` timed out after 15s")
            }
        }
    }

    async fn sync_document(&self, path: &Path, lang: &str) -> Result<String> {
        let uri = path_to_uri(path);
        let text = tools::read_text(path, 2 * 1024 * 1024)?;
        let hash = storage::hex_sha256(text.as_bytes());
        enum SyncAction {
            Open(i32),
            Change(i32),
            None,
        }
        let action = {
            let mut map = self.opened_versions.lock().unwrap();
            match map.get_mut(&uri) {
                None => {
                    map.insert(uri.clone(), (hash, 1));
                    SyncAction::Open(1)
                }
                Some((prev_hash, ver)) if *prev_hash != hash => {
                    *prev_hash = hash;
                    *ver += 1;
                    SyncAction::Change(*ver)
                }
                Some(_) => SyncAction::None,
            }
        };
        match action {
            SyncAction::Open(version) => {
                self.notify(
                    "textDocument/didOpen",
                    json!({
                        "textDocument": {
                            "uri": uri,
                            "languageId": lang,
                            "version": version,
                            "text": text,
                        }
                    }),
                )
                .await?;
            }
            SyncAction::Change(version) => {
                self.notify(
                    "textDocument/didChange",
                    json!({
                        "textDocument": { "uri": uri, "version": version },
                        "contentChanges": [{ "text": text }],
                    }),
                )
                .await?;
            }
            SyncAction::None => {}
        }
        Ok(uri)
    }
}

async fn read_lsp_frame<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("Content-Length") {
                content_length = v.trim().parse().ok();
            }
        }
    }
    let len = content_length.context("Missing Content-Length header in LSP response")?;
    if len > 16 * 1024 * 1024 {
        bail!("LSP frame exceeded 16 MiB bound");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    Ok(Some(serde_json::from_slice(&buf)?))
}

type LspClientMap = HashMap<(PathBuf, String), Arc<LspClient>>;

#[derive(Clone, Default)]
pub(crate) struct LspManager {
    clients: Arc<AsyncMutex<LspClientMap>>,
}

impl LspManager {
    async fn client_for(
        &self,
        root: &Path,
        lang: &str,
    ) -> std::result::Result<Arc<LspClient>, Value> {
        let key = (root.to_path_buf(), lang.to_owned());
        let mut map = self.clients.lock().await;
        if let Some(existing) = map.get(&key) {
            return Ok(existing.clone());
        }
        let argv = command_for_language(lang);
        match LspClient::start(root, &argv).await {
            Ok(client) => {
                map.insert(key, client.clone());
                Ok(client)
            }
            Err(err) => Err(json!({
                "available": false,
                "language": lang,
                "command": argv,
                "message": err.to_string(),
            })),
        }
    }

    pub(crate) async fn execute(
        &self,
        project: &Project,
        data_dir: &Path,
        args: &Value,
    ) -> Result<Value> {
        let op = args["operation"]
            .as_str()
            .context("operation is required")?;
        if op == "status" {
            let map = self.clients.lock().await;
            let servers: Vec<Value> = map
                .iter()
                .map(|((r, l), c)| json!({ "root": r, "language": l, "command": c.cmd_label }))
                .collect();
            return Ok(json!({ "active_servers": servers }));
        }

        let explicit_lang = args.get("language").and_then(Value::as_str);
        if op == "workspace_symbols" {
            let root = project.roots.first().context("No project root available")?;
            let lang = detect_language(Path::new(""), explicit_lang).unwrap_or("rust");
            let client = match self.client_for(&root.path, lang).await {
                Ok(c) => c,
                Err(unavailable) => return Ok(unavailable),
            };
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let res = client
                .request("workspace/symbol", json!({ "query": query }))
                .await?;
            return Ok(json!({ "available": true, "operation": op, "result": res }));
        }

        let rel_arg = args["path"]
            .as_str()
            .context("path is required for file-scoped LSP operations")?;
        let (root, rel) = resolve_project_root(project, rel_arg, false)?;
        let abs = tools::resolve(&root.path, &rel, false)?;
        if abs
            .canonicalize()
            .unwrap_or_else(|_| abs.clone())
            .starts_with(data_dir)
        {
            bail!("Conversation storage is not available to lsp");
        }
        let lang = detect_language(&abs, explicit_lang)
            .with_context(|| format!("Unsupported file language for `{rel_arg}`"))?;
        let client = match self.client_for(&root.path, lang).await {
            Ok(c) => c,
            Err(unavailable) => return Ok(unavailable),
        };
        let uri = client.sync_document(&abs, lang).await?;
        let line0 = args
            .get("line")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .saturating_sub(1);
        let char0 = args
            .get("character")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .saturating_sub(1);
        let position = json!({ "line": line0, "character": char0 });

        match op {
            "definition" => {
                let res = client
                    .request(
                        "textDocument/definition",
                        json!({ "textDocument": { "uri": uri }, "position": position }),
                    )
                    .await?;
                Ok(json!({ "available": true, "operation": op, "result": res }))
            }
            "references" => {
                let res = client
                    .request(
                        "textDocument/references",
                        json!({
                            "textDocument": { "uri": uri },
                            "position": position,
                            "context": { "includeDeclaration": true }
                        }),
                    )
                    .await?;
                Ok(json!({ "available": true, "operation": op, "result": res }))
            }
            "hover" => {
                let res = client
                    .request(
                        "textDocument/hover",
                        json!({ "textDocument": { "uri": uri }, "position": position }),
                    )
                    .await?;
                Ok(json!({ "available": true, "operation": op, "result": res }))
            }
            "document_symbols" => {
                let res = client
                    .request(
                        "textDocument/documentSymbol",
                        json!({ "textDocument": { "uri": uri } }),
                    )
                    .await?;
                Ok(json!({ "available": true, "operation": op, "result": res }))
            }
            "diagnostics" => {
                tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                let diags = client
                    .diagnostics
                    .lock()
                    .unwrap()
                    .get(&uri)
                    .cloned()
                    .unwrap_or_default();
                Ok(json!({ "available": true, "operation": op, "uri": uri, "diagnostics": diags }))
            }
            other => bail!("Unsupported lsp operation `{other}`"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Root;

    #[tokio::test]
    async fn lsp_roundtrips_with_mock_stdio_server_and_degrades_gracefully() {
        let base = std::env::temp_dir().join(format!("allpaka-lsp-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        let repo = repo.canonicalize().unwrap();
        let data = data.canonicalize().unwrap();
        std::fs::write(repo.join("main.py"), "def greet():\n    return 1\n").unwrap();

        let mock_py = base.join("mock_lsp.py");
        std::fs::write(
            &mock_py,
            r#"import sys, json

def read_msg():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b'\r\n', b'\n'):
            break
        k, v = line.decode().split(':', 1)
        headers[k.strip().lower()] = v.strip()
    n = int(headers['content-length'])
    return json.loads(sys.stdin.buffer.read(n).decode())

def send_msg(obj):
    data = json.dumps(obj).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()

while True:
    msg = read_msg()
    if msg is None:
        break
    m = msg.get('method')
    if m == 'initialize':
        send_msg({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'capabilities': {}}})
    elif m == 'textDocument/didOpen':
        uri = msg['params']['textDocument']['uri']
        send_msg({'jsonrpc': '2.0', 'method': 'textDocument/publishDiagnostics', 'params': {'uri': uri, 'diagnostics': [{'message': 'mock-diag'}]}})
    elif m == 'textDocument/hover':
        send_msg({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'contents': 'mock hover info'}})
    elif 'id' in msg:
        send_msg({'jsonrpc': '2.0', 'id': msg['id'], 'result': None})
"#,
        )
        .unwrap();

        let proj = Project {
            id: "p1".into(),
            name: "p1".into(),
            roots: vec![Root {
                alias: "workspace".into(),
                path: repo.clone(),
                writable: false,
                repository: false,
                ssh: None,
            }],
            instructions: String::new(),
        };

        // Call mock server using LspClient::start directly (avoids process-wide env races).
        let argv = vec!["python3".into(), mock_py.to_string_lossy().into_owned()];
        let mgr = LspManager::default();
        let client = LspClient::start(&repo, &argv).await.unwrap();
        mgr.clients
            .lock()
            .await
            .insert((repo.clone(), "python".into()), client);

        let hover = mgr
            .execute(
                &proj,
                &data,
                &json!({ "operation": "hover", "path": "main.py", "line": 1, "character": 5 }),
            )
            .await
            .unwrap();
        assert_eq!(hover["available"], true);
        assert_eq!(hover["result"]["contents"], "mock hover info");

        let diags = mgr
            .execute(
                &proj,
                &data,
                &json!({ "operation": "diagnostics", "path": "main.py" }),
            )
            .await
            .unwrap();
        assert_eq!(diags["diagnostics"][0]["message"], "mock-diag");

        // Graceful unavailable result for missing binary.
        let Err(unavail) = mgr.client_for(&repo, "missing_lang").await else {
            panic!("expected missing_lang LSP server to be unavailable");
        };
        assert_eq!(unavail["available"], false);

        let _ = std::fs::remove_dir_all(base);
    }
}
