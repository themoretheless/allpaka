//! Streamable-HTTP and stdio MCP client supporting tools and resources
//! (`tools/list`, `tools/call`, `resources/list`, `resources/templates/list`, `resources/read`).

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

const STDIO_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

struct StdioProcess {
    _child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
}

enum Transport {
    Http { http: reqwest::Client, url: String },
    Stdio(Arc<Mutex<StdioProcess>>),
}

pub struct Client {
    transport: Transport,
    pub tools: Vec<Value>,
    pub resources: Vec<Value>,
    pub resource_templates: Vec<Value>,
    next_id: AtomicU64,
}

impl Client {
    pub async fn connect(http: &reqwest::Client, url: &str) -> Result<Self> {
        let mut client = Self {
            transport: Transport::Http {
                http: http.clone(),
                url: url.to_string(),
            },
            tools: Vec::new(),
            resources: Vec::new(),
            resource_templates: Vec::new(),
            next_id: AtomicU64::new(1),
        };
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            client.initialize().await?;
            client.tools = client.list_tools().await?;
            client.resources = client.list_resources().await.unwrap_or_default();
            client.resource_templates = client.list_resource_templates().await.unwrap_or_default();
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("MCP connection timed out after 30 seconds")??;
        Ok(client)
    }

    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
        cwd: &Path,
    ) -> Result<Self> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .current_dir(cwd)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP stdio command `{command}`"))?;
        let stdin = child.stdin.take().context("MCP child missing stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("MCP child missing stdout")?);
        let proc = StdioProcess {
            _child: child,
            stdin,
            stdout,
        };
        let mut client = Self {
            transport: Transport::Stdio(Arc::new(Mutex::new(proc))),
            tools: Vec::new(),
            resources: Vec::new(),
            resource_templates: Vec::new(),
            next_id: AtomicU64::new(1),
        };
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            client.initialize().await?;
            client.tools = client.list_tools().await?;
            client.resources = client.list_resources().await.unwrap_or_default();
            client.resource_templates = client.list_resource_templates().await.unwrap_or_default();
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("MCP stdio connection timed out after 30 seconds")??;
        Ok(client)
    }

    pub fn tool_schemas(&self, prefix: &str, read_only: bool) -> Vec<Value> {
        self.tools
            .iter()
            .filter_map(|tool| {
                let name = tool["name"].as_str()?;
                if read_only && !is_read_only(tool, name) {
                    return None;
                }
                let description = tool["description"].as_str().unwrap_or("");
                let parameters = tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type":"object","properties":{}}));
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": format!("{prefix}{name}"),
                        "description": description,
                        "parameters": parameters,
                    }
                }))
            })
            .collect()
    }

    pub async fn call(&self, name: &str, args: &Value) -> Result<Value> {
        let response = self
            .rpc("tools/call", json!({"name": name, "arguments": args}))
            .await?;
        let content = response["result"]["content"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let is_error = response["result"]["isError"].as_bool().unwrap_or(false);
        let text = content
            .iter()
            .filter_map(|item| item["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if is_error {
            Ok(json!({"error": text}))
        } else {
            Ok(serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)))
        }
    }

    async fn initialize(&self) -> Result<()> {
        self.rpc(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "allpaka-studio", "version": "0.1"}
            }),
        )
        .await?;
        let note = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
        match &self.transport {
            Transport::Http { http, url } => {
                let _ = http
                    .post(url)
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("mcp-protocol-version", "2024-11-05")
                    .body(note)
                    .send()
                    .await;
            }
            Transport::Stdio(proc_lock) => {
                let mut guard = proc_lock.lock().await;
                let line = format!("{note}\n");
                let _ = guard.stdin.write_all(line.as_bytes()).await;
                let _ = guard.stdin.flush().await;
            }
        }
        Ok(())
    }

    async fn list_tools(&self) -> Result<Vec<Value>> {
        let response = self.rpc("tools/list", json!({})).await?;
        Ok(response["result"]["tools"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    pub async fn list_resources(&self) -> Result<Vec<Value>> {
        let response = self.rpc("resources/list", json!({})).await?;
        Ok(response["result"]["resources"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    pub async fn list_resource_templates(&self) -> Result<Vec<Value>> {
        let response = self.rpc("resources/templates/list", json!({})).await?;
        Ok(response["result"]["resourceTemplates"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    pub async fn read_resource(&self, uri: &str) -> Result<Value> {
        let response = self.rpc("resources/read", json!({"uri": uri})).await?;
        Ok(response["result"].clone())
    }

    async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        match &self.transport {
            Transport::Http { http, url } => {
                let response = http
                    .post(url)
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("mcp-protocol-version", "2024-11-05")
                    .json(&request)
                    .send()
                    .await
                    .context("MCP HTTP connection failed")?;
                let status = response.status();
                if !status.is_success() {
                    bail!("MCP HTTP {status}");
                }
                let is_sse = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| v.starts_with("text/event-stream"));
                let mut stream = response.bytes_stream();
                let mut buffer = Vec::new();
                while let Some(chunk) = stream.next().await {
                    buffer.extend_from_slice(&chunk.context("MCP response read failed")?);
                    if buffer.len() > 8_000_000 {
                        bail!("MCP response exceeds 8 MB");
                    }
                    if is_sse {
                        while let Some(end) = frame_end(&buffer) {
                            let frame: Vec<_> = buffer.drain(..end).collect();
                            let text = std::str::from_utf8(&frame).context("Invalid MCP UTF-8")?;
                            if let Some(value) = matching_event(text, id)? {
                                return Ok(value);
                            }
                        }
                    }
                }
                parse_rpc(
                    std::str::from_utf8(&buffer).context("Invalid MCP UTF-8")?,
                    id,
                )
            }
            Transport::Stdio(proc_lock) => {
                let line = format!("{}\n", serde_json::to_string(&request)?);
                let mut guard = proc_lock.lock().await;
                guard
                    .stdin
                    .write_all(line.as_bytes())
                    .await
                    .context("Failed writing MCP stdio request")?;
                guard.stdin.flush().await?;
                let mut buf = String::new();
                loop {
                    buf.clear();
                    let read =
                        tokio::time::timeout(STDIO_RPC_TIMEOUT, guard.stdout.read_line(&mut buf))
                            .await
                            .context("MCP stdio response timed out")??;
                    if read == 0 {
                        bail!("MCP stdio server closed stdout");
                    }
                    let trimmed = buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value: Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if value.get("id").and_then(Value::as_u64) == Some(id) {
                        return check_rpc(value, id);
                    }
                }
            }
        }
    }
}

pub(crate) fn resource_schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "list_mcp_resources",
                "description": "List resources exposed by connected MCP servers (optionally filtered by plugin_id).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin_id": { "type": "string" }
                    },
                    "additionalProperties": false
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_mcp_resource_templates",
                "description": "List parameterized resource URI templates exposed by connected MCP servers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin_id": { "type": "string" }
                    },
                    "additionalProperties": false
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_mcp_resource",
                "description": "Read a specific MCP resource URI from a connected MCP plugin.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin_id": { "type": "string" },
                        "uri": { "type": "string" }
                    },
                    "required": ["uri"],
                    "additionalProperties": false
                }
            }
        }),
    ]
}

pub(crate) async fn execute_resource_tool(
    app: &crate::App,
    name: &str,
    args: &Value,
) -> Result<Value> {
    let plugin_filter = args["plugin_id"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let clients: Vec<(String, Arc<Client>)> = {
        let reg = app.plugins.read().unwrap();
        reg.iter()
            .filter_map(|(id, state)| {
                if let Some(filter) = plugin_filter {
                    if id != filter {
                        return None;
                    }
                }
                state.client.as_ref().map(|c| (id.clone(), c.clone()))
            })
            .collect()
    };

    match name {
        "list_mcp_resources" => {
            let mut items = Vec::new();
            for (pid, client) in clients {
                let list = client
                    .list_resources()
                    .await
                    .unwrap_or_else(|_| client.resources.clone());
                for mut item in list {
                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("plugin_id".into(), json!(pid));
                    }
                    items.push(item);
                }
            }
            Ok(json!({ "resources": items }))
        }
        "list_mcp_resource_templates" => {
            let mut items = Vec::new();
            for (pid, client) in clients {
                let list = client
                    .list_resource_templates()
                    .await
                    .unwrap_or_else(|_| client.resource_templates.clone());
                for mut item in list {
                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("plugin_id".into(), json!(pid));
                    }
                    items.push(item);
                }
            }
            Ok(json!({ "resource_templates": items }))
        }
        "read_mcp_resource" => {
            let uri = args["uri"].as_str().context("Missing `uri`")?;
            let (pid, client) = if plugin_filter.is_some() {
                clients
                    .into_iter()
                    .next()
                    .context("Requested MCP plugin is not connected")?
            } else {
                let mut matched = None;
                for (id, c) in &clients {
                    if c.resources.iter().any(|r| r["uri"].as_str() == Some(uri)) {
                        matched = Some((id.clone(), c.clone()));
                        break;
                    }
                }
                matched
                    .or_else(|| clients.into_iter().next())
                    .context("No connected MCP server available")?
            };
            let result = client.read_resource(uri).await?;
            Ok(json!({ "plugin_id": pid, "uri": uri, "result": result }))
        }
        other => bail!("Unsupported MCP resource tool `{other}`"),
    }
}

fn is_read_only(tool: &Value, name: &str) -> bool {
    if let Some(annotations) = tool.get("annotations") {
        if annotations.get("readOnlyHint").and_then(Value::as_bool) == Some(true) {
            return true;
        }
        if annotations.get("destructiveHint").and_then(Value::as_bool) == Some(true) {
            return false;
        }
    }
    const DESTRUCTIVE: &[&str] = &[
        "write",
        "update",
        "delete",
        "ingest",
        "add_",
        "apply",
        "maintain",
        "reembed",
        "compile",
        "consolidate",
        "archive",
        "file_answer",
        "diary_write",
        "checkpoint",
        "append",
        "link_",
        "create",
        "remove",
        "cleanup",
        "vacuum",
        "backup",
        "export",
        "import",
        "sync",
        "prune",
        "merge",
        "refile",
        "set_",
        "upsert",
        "rebuild",
    ];
    !DESTRUCTIVE.iter().any(|d| name.contains(d))
}
fn parse_rpc(body: &str, id: u64) -> Result<Value> {
    let mut data_lines = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.strip_prefix(' ').unwrap_or(data);
            if !data.is_empty() {
                data_lines.push(data);
            }
        }
    }
    if data_lines.is_empty() {
        let value: Value = serde_json::from_str(body).context("Invalid rag-mcp JSON response")?;
        return check_rpc(value, id);
    }
    for data in data_lines.iter().rev() {
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return check_rpc(value, id);
            }
        }
    }
    let value: Value =
        serde_json::from_str(data_lines.last().unwrap()).context("Invalid rag-mcp SSE payload")?;
    check_rpc(value, id)
}

fn check_rpc(value: Value, id: u64) -> Result<Value> {
    if let Some(err) = value.get("error") {
        bail!("rag-mcp error: {err}");
    }
    if value.get("id").and_then(Value::as_u64) != Some(id) {
        bail!("rag-mcp response id mismatch");
    }
    Ok(value)
}

fn frame_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|i| i + 2)
        .into_iter()
        .chain(
            bytes
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|i| i + 4),
        )
        .min()
}
fn matching_event(frame: &str, id: u64) -> Result<Option<Value>> {
    let data = frame
        .lines()
        .filter_map(|l| {
            l.strip_prefix("data:")
                .map(|v| v.strip_prefix(' ').unwrap_or(v))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&data).context("Invalid MCP SSE JSON")?;
    if value.get("id").and_then(Value::as_u64) == Some(id) {
        return check_rpc(value, id).map(Some);
    }
    Ok(None)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn connects_without_waiting_for_sse_stream_to_close() {
        use axum::{response::IntoResponse, routing::post, Json, Router};
        let app = Router::new().route(
            "/mcp",
            post(|Json(request): Json<Value>| async move {
                let Some(id) = request.get("id") else {
                    return axum::http::StatusCode::ACCEPTED.into_response();
                };
                let result = if request["method"] == "tools/list" {
                    json!({"tools":[{"name":"search"}]})
                } else {
                    json!({"protocolVersion":"2024-11-05"})
                };
                let event = format!(
                    "data: {}\n\n",
                    json!({"jsonrpc":"2.0","id":id,"result":result})
                );
                let stream =
                    futures_util::stream::once(
                        async move { Ok::<_, std::convert::Infallible>(event) },
                    )
                    .chain(futures_util::stream::pending());
                (
                    [("content-type", "text/event-stream")],
                    axum::body::Body::from_stream(stream),
                )
                    .into_response()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            Client::connect(&reqwest::Client::new(), &url),
        )
        .await;
        server.abort();
        assert_eq!(result.unwrap().unwrap().tools[0]["name"], "search");
    }
    #[test]
    fn sse_frames_ignore_notifications_and_read_multiline_response() {
        assert_eq!(frame_end(b"data: incomplete"), None);
        assert_eq!(frame_end(b"data: {}\r\n\r\nrest"), Some(12));
        assert!(matching_event("data: {\"method\":\"notification\"}\n\n", 1)
            .unwrap()
            .is_none());
        assert_eq!(
            matching_event(
                "data: {\"id\":1,\n data ignored\ndata: \"result\":{}}\n\n",
                1
            )
            .unwrap()
            .unwrap()["result"],
            json!({})
        );
        assert!(matching_event("data: {\"id\":1,\"error\":{\"code\":-1}}\n\n", 1).is_err());
    }

    #[tokio::test]
    async fn stdio_mcp_client_lists_tools_and_reads_resources() {
        let server_py = r#"
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id")
    if rid is None: continue
    m = req.get("method")
    if m == "initialize":
        res = {"protocolVersion": "2024-11-05", "capabilities": {"resources": {}}}
    elif m == "tools/list":
        res = {"tools": [{"name": "echo", "description": "echo"}]}
    elif m == "resources/list":
        res = {"resources": [{"uri": "memo://welcome", "name": "Welcome"}]}
    elif m == "resources/templates/list":
        res = {"resourceTemplates": [{"uriTemplate": "memo://{id}", "name": "Memo"}]}
    elif m == "resources/read":
        res = {"contents": [{"uri": req["params"]["uri"], "text": "hello resource"}]}
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\n")
    sys.stdout.flush()
"#;
        let args = vec!["-u".into(), "-c".into(), server_py.into()];
        let env = BTreeMap::new();
        let client = Client::connect_stdio("python3", &args, &env, Path::new("."))
            .await
            .unwrap();
        assert_eq!(client.tools.len(), 1);
        assert_eq!(client.resources[0]["uri"], "memo://welcome");
        assert_eq!(client.resource_templates[0]["uriTemplate"], "memo://{id}");
        let read = client.read_resource("memo://welcome").await.unwrap();
        assert_eq!(read["contents"][0]["text"], "hello resource");
    }
}
