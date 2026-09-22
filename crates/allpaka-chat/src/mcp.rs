//! Minimal streamable-HTTP MCP client for the local `rag-mcp` gateway.
//!
//! The gateway is a trusted local process (usually `http://127.0.0.1:7432/mcp`).
//! We discover its tools once at startup and expose them to the model with a
//! `rag_` prefix so they never collide with the built-in workspace tools.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Client {
    http: reqwest::Client,
    url: String,
    pub tools: Vec<Value>,
    next_id: AtomicU64,
}

impl Client {
    pub async fn connect(http: &reqwest::Client, url: &str) -> Result<Self> {
        let mut client = Self {
            http: http.clone(),
            url: url.to_string(),
            tools: Vec::new(),
            next_id: AtomicU64::new(1),
        };
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            client.initialize().await?;
            client.tools = client.list_tools().await?;
            Ok::<_, anyhow::Error>(())
        }).await.context("MCP connection timed out after 30 seconds")??;
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
        let response = self.rpc("tools/call", json!({"name": name, "arguments": args})).await?;
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
        let _ = self
            .http
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2024-11-05")
            .body(json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string())
            .send()
            .await;
        Ok(())
    }

    async fn list_tools(&self) -> Result<Vec<Value>> {
        let response = self.rpc("tools/list", json!({})).await?;
        Ok(response["result"]["tools"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let response = self
            .http
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2024-11-05")
            .json(&request)
            .send()
            .await
            .context("rag-mcp connection failed")?;
        let status = response.status();
        if !status.is_success() {
            bail!("rag-mcp HTTP {status}");
        }
        let is_sse = response.headers().get("content-type").and_then(|v|v.to_str().ok()).is_some_and(|v|v.starts_with("text/event-stream"));
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk.context("MCP response read failed")?);
            if buffer.len() > 8_000_000 { bail!("MCP response exceeds 8 MB"); }
            if is_sse {
                while let Some(end) = frame_end(&buffer) {
                    let frame: Vec<_> = buffer.drain(..end).collect();
                    let text = std::str::from_utf8(&frame).context("Invalid MCP UTF-8")?;
                    if let Some(value) = matching_event(text, id)? { return Ok(value); }
                }
            }
        }
        parse_rpc(std::str::from_utf8(&buffer).context("Invalid MCP UTF-8")?, id)
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
        "write", "update", "delete", "ingest", "add_", "apply", "maintain", "reembed",
        "compile", "consolidate", "archive", "file_answer", "diary_write", "checkpoint",
        "append", "link_", "create", "remove", "cleanup", "vacuum", "backup", "export",
        "import", "sync", "prune", "merge", "refile", "set_", "upsert", "rebuild",
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
    let value: Value = serde_json::from_str(data_lines.last().unwrap())
        .context("Invalid rag-mcp SSE payload")?;
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
    bytes.windows(2).position(|w|w==b"\n\n").map(|i|i+2)
        .into_iter().chain(bytes.windows(4).position(|w|w==b"\r\n\r\n").map(|i|i+4)).min()
}
fn matching_event(frame: &str, id: u64) -> Result<Option<Value>> {
    let data = frame.lines().filter_map(|l|l.strip_prefix("data:").map(|v|v.strip_prefix(' ').unwrap_or(v))).collect::<Vec<_>>().join("\n");
    if data.is_empty() { return Ok(None); }
    let value: Value = serde_json::from_str(&data).context("Invalid MCP SSE JSON")?;
    if value.get("id").and_then(Value::as_u64)==Some(id) { return check_rpc(value,id).map(Some); }
    Ok(None)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn connects_without_waiting_for_sse_stream_to_close() {
        use axum::{routing::post, Json, Router, response::IntoResponse};
        let app = Router::new().route("/mcp", post(|Json(request): Json<Value>| async move {
            let Some(id) = request.get("id") else { return axum::http::StatusCode::ACCEPTED.into_response(); };
            let result = if request["method"] == "tools/list" { json!({"tools":[{"name":"search"}]}) } else {json!({"protocolVersion":"2024-11-05"})};
            let event = format!("data: {}\n\n", json!({"jsonrpc":"2.0","id":id,"result":result}));
            let stream = futures_util::stream::once(async move {Ok::<_, std::convert::Infallible>(event)}).chain(futures_util::stream::pending());
            ([("content-type", "text/event-stream")], axum::body::Body::from_stream(stream)).into_response()
        }));
        let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url=format!("http://{}/mcp",listener.local_addr().unwrap());
        let server=tokio::spawn(async move {axum::serve(listener,app).await.unwrap()});
        let result=tokio::time::timeout(std::time::Duration::from_secs(2),Client::connect(&reqwest::Client::new(),&url)).await;
        server.abort();
        assert_eq!(result.unwrap().unwrap().tools[0]["name"], "search");
    }
    #[test]
    fn sse_frames_ignore_notifications_and_read_multiline_response() {
        assert_eq!(frame_end(b"data: incomplete"),None);
        assert_eq!(frame_end(b"data: {}\r\n\r\nrest"),Some(12));
        assert!(matching_event("data: {\"method\":\"notification\"}\n\n",1).unwrap().is_none());
        assert_eq!(matching_event("data: {\"id\":1,\n data ignored\ndata: \"result\":{}}\n\n",1).unwrap().unwrap()["result"],json!({}));
        assert!(matching_event("data: {\"id\":1,\"error\":{\"code\":-1}}\n\n",1).is_err());
    }
}
