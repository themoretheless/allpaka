//! Agent Client Protocol (ACP) stdio JSON-RPC 2.0 server (`allpaka studio --acp`).

use crate::bundles;
use crate::ext::{default_settings, get_session};
use crate::{
    attach, cancel, evaluation, save, turn_traced, Action, ActionKind, App, Message, Session,
};
use anyhow::Result;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

fn rpc_ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_err(id: Value, code: i64, message: impl ToString) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.to_string() }
    })
}

pub(crate) async fn dispatch_rpc(app: &App, req: &Value) -> Option<Value> {
    let id = req.get("id").cloned()?;
    let method = req["method"].as_str().unwrap_or("");
    let params = &req["params"];

    match method {
        "initialize" => Some(rpc_ok(
            id,
            json!({
                "protocolVersion": "2025-01-01",
                "serverInfo": { "name": "allpaka-studio-acp", "version": "0.1.0" },
                "capabilities": {
                    "sessions": true,
                    "tools": true,
                    "profiles": true,
                }
            }),
        )),
        "ping" => Some(rpc_ok(id, json!({ "ok": true }))),
        "tools/list" => {
            let mut tools = vec![
                "read_file",
                "write_file",
                "edit_file",
                "list_files",
                "agentgrep",
                "run_command",
                "glob_files",
                "apply_patch",
                "lsp",
                "web_fetch",
                "web_search",
                "extract_document",
                "terminal_open",
                "create_worktree",
                "forge_pr",
                "run_code",
                "ralph",
                "workflow",
                "schedule_create",
                "schedule_list",
            ];
            let reg = app.plugins.read().unwrap();
            let plugin_count = reg.values().filter(|s| s.client.is_some()).count();
            tools.sort_unstable();
            Some(rpc_ok(
                id,
                json!({ "tools": tools, "connected_mcp_plugins": plugin_count }),
            ))
        }
        "session/new" => {
            let project_id = params["project_id"].as_str().unwrap_or("default");
            let mut settings = default_settings(app, project_id);
            if let Some(pid) = params["profile"].as_str() {
                match bundles::resolve_profile(app, pid) {
                    Ok(prof) => bundles::apply_profile(&prof, &mut settings),
                    Err(e) => return Some(rpc_err(id, -32602, e)),
                }
            }
            if let Some(m) = params["model"].as_str().filter(|s| !s.is_empty()) {
                settings.model = m.to_owned();
            }
            if let Some(p) = params["provider"].as_str().filter(|s| !s.is_empty()) {
                settings.provider = p.to_owned();
            }
            let sid = evaluation::new_id();
            let session = Session::new(sid.clone(), settings);
            if let Err(e) = save(app, &session) {
                return Some(rpc_err(id, -32603, e));
            }
            attach(app, session);
            Some(rpc_ok(id, json!({ "session_id": sid })))
        }
        "session/get" => {
            let Some(sid) = params["session_id"].as_str() else {
                return Some(rpc_err(id, -32602, "Missing `session_id`"));
            };
            match get_session(app, sid) {
                Ok(shared) => {
                    let s = shared.lock().unwrap().clone();
                    Some(rpc_ok(id, serde_json::to_value(s).unwrap_or(Value::Null)))
                }
                Err(_) => Some(rpc_err(id, -32004, "Session not found")),
            }
        }
        "session/cancel" => {
            let Some(sid) = params["session_id"].as_str() else {
                return Some(rpc_err(id, -32602, "Missing `session_id`"));
            };
            let tx = app
                .sessions
                .lock()
                .unwrap()
                .get(sid)
                .map(|(_, tx)| tx.clone());
            if let Some(tx) = tx {
                let _ = tx.try_send(Action {
                    wake_epoch: None,
                    reply: None,
                    images: Vec::new(),
                    kind: ActionKind::Stop,
                    text: String::new(),
                    settings: None,
                    citations: Vec::new(),
                    context_budget: None,
                });
            }
            if let Ok(shared) = get_session(app, sid) {
                cancel(app, &mut None, &shared).await;
            }
            app.ext.agents.purge_session(sid);
            Some(rpc_ok(id, json!({ "session_id": sid, "cancelled": true })))
        }
        "session/prompt" => {
            let Some(sid) = params["session_id"].as_str() else {
                return Some(rpc_err(id, -32602, "Missing `session_id`"));
            };
            let Some(prompt) = params["prompt"].as_str() else {
                return Some(rpc_err(id, -32602, "Missing `prompt`"));
            };
            let shared = match get_session(app, sid) {
                Ok(s) => s,
                Err(_) => return Some(rpc_err(id, -32004, "Session not found")),
            };
            let project_id = {
                let mut s = shared.lock().unwrap();
                s.messages.push(Message::text("user", prompt));
                s.settings.project_id.clone()
            };
            let trace = match app.observability.begin(sid, &project_id) {
                Ok(t) => t,
                Err(e) => return Some(rpc_err(id, -32603, e)),
            };
            let mut span = trace.span("turn", "acp.prompt", None);
            let outcome = turn_traced(app.clone(), shared.clone(), None, &trace, span.id()).await;
            span.finish(
                if outcome.is_ok() {
                    "completed"
                } else {
                    "failed"
                },
                &Value::Null,
            );
            match outcome {
                Ok(_) => {
                    let s = shared.lock().unwrap();
                    let reply = s
                        .messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "assistant")
                        .map(|m| m.content.clone())
                        .unwrap_or_default();
                    let _ = save(app, &s);
                    Some(rpc_ok(id, json!({ "session_id": sid, "reply": reply })))
                }
                Err(e) => Some(rpc_err(id, -32603, e)),
            }
        }
        other => Some(rpc_err(id, -32601, format!("Unknown ACP method `{other}`"))),
    }
}

pub(crate) async fn run_stdio(app: App) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(text) => {
                    if tx.blocking_send(text).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    while let Some(line) = rx.recv().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(trimmed) {
            Ok(req) => dispatch_rpc(&app, &req).await,
            Err(err) => Some(rpc_err(Value::Null, -32700, format!("Parse error: {err}"))),
        };
        if let Some(resp) = response {
            let mut line_out = serde_json::to_string(&resp)?;
            line_out.push('\n');
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(line_out.as_bytes())?;
            handle.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Project, Root};
    use crate::ext::test_app;
    use crate::storage;

    #[tokio::test]
    async fn acp_initialize_and_session_new_dispatch() {
        let root_dir = std::env::temp_dir().join(format!("allpaka-acp-test-{}", storage::now_ms()));
        let ws = root_dir.join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ws = ws.canonicalize().unwrap();
        let app = test_app(
            &root_dir.join("data"),
            &ws,
            vec![Project {
                id: "default".into(),
                name: "default".into(),
                roots: vec![Root {
                    alias: "workspace".into(),
                    path: ws.clone(),
                    writable: false,
                    repository: false,
                    ssh: None,
                }],
                instructions: String::new(),
            }],
        );

        let init = dispatch_rpc(
            &app,
            &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        )
        .await
        .unwrap();
        assert_eq!(init["result"]["serverInfo"]["name"], "allpaka-studio-acp");

        let created = dispatch_rpc(
            &app,
            &json!({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"profile": "research"}}),
        )
        .await
        .unwrap();
        let sid = created["result"]["session_id"].as_str().unwrap();

        let fetched = dispatch_rpc(
            &app,
            &json!({"jsonrpc": "2.0", "id": 3, "method": "session/get", "params": {"session_id": sid}}),
        )
        .await
        .unwrap();
        assert_eq!(fetched["result"]["settings"]["mode"], "plan");
        let _ = std::fs::remove_dir_all(&root_dir);
    }
}
