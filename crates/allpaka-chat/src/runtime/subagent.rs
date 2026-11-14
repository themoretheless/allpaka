//! Nested headless turns for delegated subagents AND collaborative async subagent tree
//! (`subagent_spawn` w/ `fork_context` and external runtime seams, `list_agents`,
//! `send_agent_message`, `wait_agent`, `interrupt_agent`).

use crate::ext::get_session;
use crate::storage;
use crate::{bad_request, not_found, ApiResult, App, Ephemeral, ExecutionPolicy, Message, Mode, Session, SessionStatus, Settings, SharedSession, TurnOutcome};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{watch, Notify};

const MAX_ACTIVE_PER_SESSION: usize = 6;
const MAX_TOTAL_PER_SESSION: usize = 24;

/// Boxed so that a turn can contain a (subagent) turn.
pub(crate) fn nested_turn<'a>(
    app: App,
    s: SharedSession,
    trace: &'a observability::Trace,
    parent_span: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TurnOutcome>> + Send + 'a>> {
    Box::pin(crate::turn_traced(app, s, None, trace, parent_span))
}

use crate::observability;

/// What a subagent inherits from the turn that delegated to it.
pub(crate) struct SubagentParent<'a> {
    pub(crate) settings: &'a Settings,
    pub(crate) project: &'a crate::context::Project,
    pub(crate) tree: &'a crate::harness::Tree,
    pub(crate) policy: ExecutionPolicy,
}

/// Run one harness subagent synchronously as a nested headless turn (`delegate` tool).
pub(crate) async fn run_subagent(
    app: &App,
    parent: SubagentParent<'_>,
    args: &Value,
    trace: &observability::Trace,
    parent_span: usize,
) -> Result<Value> {
    let SubagentParent {
        settings,
        project,
        tree,
        policy,
    } = parent;
    let name = args["agent"].as_str().context("agent is required")?;
    let task = args["task"].as_str().context("task is required")?;
    if task.trim().is_empty() || task.len() > 16000 {
        bail!("task must be 1–16000 bytes");
    }
    let agent = crate::harness::subagents(tree)
        .into_iter()
        .find(|a| a.name == name)
        .with_context(|| format!("Unknown subagent `{name}`"))?;
    let mut child = settings.clone();
    child.max_steps = (settings.max_steps / 2).max(2);
    child.allow_writes &= agent.writes;
    if child.mode == Mode::Goal {
        child.mode = Mode::Auto;
    }
    let mut session = Session::new(crate::evaluation::new_id(), child);
    session.ephemeral = Some(Arc::new(Ephemeral {
        project: project.clone(),
        harness: Some(tree.clone()),
        system: Some(format!(
            "You are the `{}` subagent, working on a task handed to you by another agent. Finish it and reply with the final result only; your reply is returned to that agent.\n{}",
            agent.name,
            agent.prompt.trim()
        )),
        nested: true,
        policy,
    }));
    session.messages.push(Message::text("user", task));
    let shared: SharedSession = Arc::new(Mutex::new(session));
    let mut span = trace.span("agent", &format!("subagent.{name}"), Some(parent_span));
    let outcome = nested_turn(app.clone(), shared.clone(), trace, span.id()).await;
    span.finish(
        if outcome.is_ok() { "completed" } else { "failed" },
        &Value::Null,
    );
    let state = shared.lock().unwrap();
    let answer = state
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
        .map(|m| m.content.clone())
        .unwrap_or_default();
    Ok(json!({
        "agent": name,
        "outcome": match &outcome { Ok(o) => format!("{o:?}"), Err(e) => format!("error: {e:#}") },
        "answer": answer,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentRunStatus {
    Running,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AgentMailboxMessage {
    pub from: String,
    pub message: String,
    pub sent_at_ms: u64,
}

struct AgentEntry {
    id: String,
    parent_session_id: String,
    name: String,
    role: String,
    external_runtime: Option<String>,
    created_at_ms: u64,
    status: Mutex<AgentRunStatus>,
    finished_at_ms: Mutex<Option<u64>>,
    answer: Mutex<Option<String>>,
    error: Mutex<Option<String>>,
    mailbox: Mutex<VecDeque<AgentMailboxMessage>>,
    session: Option<SharedSession>,
    abort_tx: watch::Sender<bool>,
    handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    notify: Arc<Notify>,
}

impl AgentEntry {
    fn snapshot(&self) -> Value {
        json!({
            "id": self.id,
            "parent_session_id": self.parent_session_id,
            "name": self.name,
            "role": self.role,
            "external_runtime": self.external_runtime,
            "status": *self.status.lock().unwrap(),
            "created_at_ms": self.created_at_ms,
            "finished_at_ms": *self.finished_at_ms.lock().unwrap(),
            "answer": *self.answer.lock().unwrap(),
            "error": *self.error.lock().unwrap(),
            "mailbox_depth": self.mailbox.lock().unwrap().len(),
        })
    }
}

#[derive(Default)]
pub(crate) struct AgentTreeManager {
    seq: AtomicU64,
    agents: Mutex<HashMap<String, Arc<AgentEntry>>>,
}

impl Clone for AgentTreeManager {
    fn clone(&self) -> Self {
        Self {
            seq: AtomicU64::new(self.seq.load(Ordering::Relaxed)),
            agents: Mutex::new(HashMap::new()),
        }
    }
}

pub(crate) fn tree_schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "subagent_spawn",
                "description": "Spawn a live asynchronous subagent (optionally inheriting context via `fork_context` or delegating to an external runtime adapter).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Short worker identifier" },
                        "role": { "type": "string", "description": "Role or specialization instructions" },
                        "task": { "type": "string", "description": "Clear assignment for the subagent" },
                        "fork_context": { "type": "boolean", "description": "Inherit parent conversation history snapshot" },
                        "allow_writes": { "type": "boolean", "description": "Allow file/command mutations bounded by parent permissions" },
                        "max_steps": { "type": "integer", "description": "Maximum reasoning steps (default half of parent)" },
                        "external_runtime": { "type": "string", "description": "Optional external CLI runtime adapter name (resolved via ALLPAKA_EXTERNAL_AGENT_<NAME>)" },
                        "wait": { "type": "boolean", "description": "Optionally block up to 15s for fast completion" }
                    },
                    "required": ["task"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_agents",
                "description": "List all live and completed spawned subagents in the current session.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "send_agent_message",
                "description": "Send a steering/follow-up message to a spawned subagent mailbox (wakes completed in-process subagents unless `wake: false`).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": { "type": "string" },
                        "message": { "type": "string" },
                        "wake": { "type": "boolean" }
                    },
                    "required": ["agent_id", "message"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "wait_agent",
                "description": "Wait for a spawned subagent to finish (up to `timeout_ms`) and inspect its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": { "type": "string" },
                        "timeout_ms": { "type": "integer" }
                    },
                    "required": ["agent_id"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "interrupt_agent",
                "description": "Cancel and stop a running spawned subagent.",
                "parameters": {
                    "type": "object",
                    "properties": { "agent_id": { "type": "string" } },
                    "required": ["agent_id"]
                }
            }
        }),
    ]
}

fn resolve_external_command(runtime: &str) -> Result<Vec<String>> {
    let trimmed = runtime.trim();
    if trimmed.is_empty()
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        bail!("Invalid external_runtime name `{runtime}`");
    }
    let key = format!("ALLPAKA_EXTERNAL_AGENT_{}", trimmed.to_ascii_uppercase().replace('-', "_"));
    let val = std::env::var(&key)
        .with_context(|| format!("External runtime `{trimmed}` is not configured (`{key}` env var not set)"))?;
    let parts: Vec<String> = val.split_whitespace().map(ToOwned::to_owned).collect();
    if parts.is_empty() {
        bail!("Configured external runtime command `{key}` is empty");
    }
    Ok(parts)
}

async fn run_external_runtime_argv(
    argv: &[String],
    cwd: &std::path::Path,
    payload: &Value,
    mut abort_rx: watch::Receiver<bool>,
) -> Result<String> {
    if argv.is_empty() {
        bail!("External runtime argv is empty");
    }
    let mut cmd = tokio::process::Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to launch external agent `{}`", argv[0]))?;
    if let Some(mut stdin) = child.stdin.take() {
        let bytes = serde_json::to_vec(payload)?;
        stdin.write_all(&bytes).await?;
        stdin.shutdown().await?;
    }
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let stdout_fut = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(mut s) = stdout_pipe.take() {
            let _ = s.read_to_end(&mut buf).await;
        }
        buf
    });
    let stderr_fut = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(mut s) = stderr_pipe.take() {
            let _ = s.read_to_end(&mut buf).await;
        }
        buf
    });
    tokio::select! {
        _ = abort_rx.changed() => {
            let _ = child.kill().await;
            let _ = stdout_fut.await;
            let _ = stderr_fut.await;
            bail!("Interrupted");
        }
        status_res = tokio::time::timeout(std::time::Duration::from_secs(120), child.wait()) => {
            let status = status_res.map_err(|_| anyhow::anyhow!("External agent timed out after 120s"))??;
            let out_bytes = stdout_fut.await.unwrap_or_default();
            let err_bytes = stderr_fut.await.unwrap_or_default();
            if !status.success() {
                bail!(
                    "External agent exited {}: {}",
                    status,
                    String::from_utf8_lossy(&err_bytes).trim()
                );
            }
            Ok(String::from_utf8_lossy(&out_bytes).trim().to_owned())
        }
    }
}

impl AgentTreeManager {
    pub(crate) async fn spawn(
        &self,
        app: App,
        parent_session_id: &str,
        parent_session: &SharedSession,
        parent: SubagentParent<'_>,
        args: &Value,
    ) -> Result<Value> {
        let task = args["task"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .context("task is required")?;
        if task.len() > 24_000 {
            bail!("task must be <= 24000 bytes");
        }
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("worker")
            .to_owned();
        let role = args
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("collaborator")
            .to_owned();
        let fork_context = args
            .get("fork_context")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let req_writes = args
            .get("allow_writes")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let external_runtime = args
            .get("external_runtime")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);

        {
            let map = self.agents.lock().unwrap();
            let active = map
                .values()
                .filter(|e| {
                    e.parent_session_id == parent_session_id
                        && *e.status.lock().unwrap() == AgentRunStatus::Running
                })
                .count();
            if active >= MAX_ACTIVE_PER_SESSION {
                bail!("Maximum {MAX_ACTIVE_PER_SESSION} concurrent subagents active");
            }
            let total = map
                .values()
                .filter(|e| e.parent_session_id == parent_session_id)
                .count();
            if total >= MAX_TOTAL_PER_SESSION {
                bail!("Maximum {MAX_TOTAL_PER_SESSION} subagents created in this session");
            }
        }

        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let agent_id = format!("agent-{}-{}", storage::now_ms(), seq);
        let (abort_tx, abort_rx) = watch::channel(false);

        if let Some(ext_name) = external_runtime.clone() {
            let argv = resolve_external_command(&ext_name)?;
            let cwd = parent
                .project
                .roots
                .first()
                .map(|r| r.path.clone())
                .unwrap_or_else(|| app.data.to_path_buf());
            let entry = Arc::new(AgentEntry {
                id: agent_id.clone(),
                parent_session_id: parent_session_id.to_owned(),
                name: name.clone(),
                role: role.clone(),
                external_runtime: Some(ext_name.clone()),
                created_at_ms: storage::now_ms(),
                status: Mutex::new(AgentRunStatus::Running),
                finished_at_ms: Mutex::new(None),
                answer: Mutex::new(None),
                error: Mutex::new(None),
                mailbox: Mutex::new(VecDeque::new()),
                session: None,
                abort_tx,
                handle: Mutex::new(None),
                notify: Arc::new(Notify::new()),
            });
            self.agents.lock().unwrap().insert(agent_id.clone(), entry.clone());
            let payload = json!({
                "agent_id": agent_id,
                "name": name,
                "role": role,
                "task": task,
                "parent_session_id": parent_session_id,
            });
            let e_task = entry.clone();
            let handle = tokio::spawn(async move {
                let res = run_external_runtime_argv(&argv, &cwd, &payload, abort_rx).await;
                let mut st = e_task.status.lock().unwrap();
                if *st != AgentRunStatus::Interrupted {
                    match res {
                        Ok(out) => {
                            *e_task.answer.lock().unwrap() = Some(out);
                            *st = AgentRunStatus::Completed;
                        }
                        Err(err) => {
                            *e_task.error.lock().unwrap() = Some(err.to_string());
                            *st = AgentRunStatus::Failed;
                        }
                    }
                    *e_task.finished_at_ms.lock().unwrap() = Some(storage::now_ms());
                }
                e_task.notify.notify_waiters();
            });
            *entry.handle.lock().unwrap() = Some(handle);
            let wait_now = args.get("wait").and_then(Value::as_bool).unwrap_or(false);
            if wait_now {
                return self.wait(parent_session_id, &agent_id, 15_000).await;
            }
            return Ok(entry.snapshot());
        }

        // In-process collaborative child session.
        let mut child = parent.settings.clone();
        child.allow_writes &= req_writes;
        if let Some(steps) = args.get("max_steps").and_then(Value::as_u64) {
            child.max_steps = (steps as usize).clamp(1, parent.settings.max_steps.max(1));
        } else {
            child.max_steps = (parent.settings.max_steps / 2).max(4);
        }
        if child.mode == Mode::Goal {
            child.mode = Mode::Auto;
        }
        let mut child_session = Session::new(crate::evaluation::new_id(), child);
        if fork_context {
            let parent_msgs = parent_session.lock().unwrap().messages.clone();
            let start = parent_msgs.len().saturating_sub(40);
            for mut m in parent_msgs[start..].to_vec() {
                m.images.clear();
                child_session.messages.push(m);
            }
        }
        child_session.ephemeral = Some(Arc::new(Ephemeral {
            project: parent.project.clone(),
            harness: Some(parent.tree.clone()),
            system: Some(format!(
                "You are spawned subagent `{name}` (role: {role}). Complete the assigned task cleanly and report concrete verified results."
            )),
            nested: true,
            policy: parent.policy,
        }));
        child_session.messages.push(Message::text("user", task));
        let child_shared: SharedSession = Arc::new(Mutex::new(child_session));

        let entry = Arc::new(AgentEntry {
            id: agent_id.clone(),
            parent_session_id: parent_session_id.to_owned(),
            name: name.clone(),
            role,
            external_runtime: None,
            created_at_ms: storage::now_ms(),
            status: Mutex::new(AgentRunStatus::Running),
            finished_at_ms: Mutex::new(None),
            answer: Mutex::new(None),
            error: Mutex::new(None),
            mailbox: Mutex::new(VecDeque::new()),
            session: Some(child_shared.clone()),
            abort_tx,
            handle: Mutex::new(None),
            notify: Arc::new(Notify::new()),
        });
        self.agents.lock().unwrap().insert(agent_id.clone(), entry.clone());
        let handle = spawn_inprocess_turn(app, entry.clone(), child_shared);
        *entry.handle.lock().unwrap() = Some(handle);

        if args.get("wait").and_then(Value::as_bool).unwrap_or(false) {
            return self.wait(parent_session_id, &agent_id, 15_000).await;
        }
        Ok(entry.snapshot())
    }

    pub(crate) fn list(&self, parent_session_id: &str) -> Vec<Value> {
        let mut rows: Vec<_> = self
            .agents
            .lock()
            .unwrap()
            .values()
            .filter(|e| e.parent_session_id == parent_session_id)
            .cloned()
            .collect();
        rows.sort_by_key(|e| e.created_at_ms);
        rows.into_iter().map(|e| e.snapshot()).collect()
    }

    pub(crate) fn send_message(
        &self,
        app: &App,
        parent_session_id: &str,
        agent_id: &str,
        message: &str,
        wake: bool,
    ) -> Result<Value> {
        let entry = self
            .agents
            .lock()
            .unwrap()
            .get(agent_id)
            .cloned()
            .with_context(|| format!("Agent `{agent_id}` not found"))?;
        if entry.parent_session_id != parent_session_id {
            bail!("Agent `{agent_id}` does not belong to this session");
        }
        entry.mailbox.lock().unwrap().push_back(AgentMailboxMessage {
            from: "parent".into(),
            message: message.to_owned(),
            sent_at_ms: storage::now_ms(),
        });
        if wake {
            let mut st = entry.status.lock().unwrap();
            if *st == AgentRunStatus::Completed {
                if let Some(child_shared) = &entry.session {
                    *st = AgentRunStatus::Running;
                    *entry.finished_at_ms.lock().unwrap() = None;
                    let drained: Vec<AgentMailboxMessage> =
                        entry.mailbox.lock().unwrap().drain(..).collect();
                    let body = drained
                        .iter()
                        .map(|m| m.message.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    child_shared.lock().unwrap().messages.push(Message::text("user", &body));
                    let handle = spawn_inprocess_turn(app.clone(), entry.clone(), child_shared.clone());
                    *entry.handle.lock().unwrap() = Some(handle);
                }
            }
        }
        Ok(entry.snapshot())
    }

    pub(crate) async fn wait(
        &self,
        parent_session_id: &str,
        agent_id: &str,
        timeout_ms: u64,
    ) -> Result<Value> {
        let entry = self
            .agents
            .lock()
            .unwrap()
            .get(agent_id)
            .cloned()
            .with_context(|| format!("Agent `{agent_id}` not found"))?;
        if entry.parent_session_id != parent_session_id {
            bail!("Agent `{agent_id}` does not belong to this session");
        }
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms.clamp(10, 60_000));
        loop {
            if *entry.status.lock().unwrap() != AgentRunStatus::Running {
                break;
            }
            if tokio::time::timeout_at(deadline, entry.notify.notified())
                .await
                .is_err()
            {
                break;
            }
        }
        Ok(entry.snapshot())
    }

    pub(crate) fn interrupt(&self, parent_session_id: &str, agent_id: &str) -> Result<Value> {
        let entry = self
            .agents
            .lock()
            .unwrap()
            .get(agent_id)
            .cloned()
            .with_context(|| format!("Agent `{agent_id}` not found"))?;
        if entry.parent_session_id != parent_session_id {
            bail!("Agent `{agent_id}` does not belong to this session");
        }
        let _ = entry.abort_tx.send(true);
        if let Some(child_shared) = &entry.session {
            let mut s = child_shared.lock().unwrap();
            s.status = SessionStatus::Paused;
            s.close_pending_tools();
        }
        if let Some(handle) = entry.handle.lock().unwrap().take() {
            handle.abort();
        }
        *entry.status.lock().unwrap() = AgentRunStatus::Interrupted;
        *entry.finished_at_ms.lock().unwrap() = Some(storage::now_ms());
        entry.notify.notify_waiters();
        Ok(entry.snapshot())
    }

    pub(crate) fn purge_session(&self, parent_session_id: &str) {
        let entries: Vec<_> = {
            let mut map = self.agents.lock().unwrap();
            let ids: Vec<String> = map
                .values()
                .filter(|e| e.parent_session_id == parent_session_id)
                .map(|e| e.id.clone())
                .collect();
            ids.into_iter().filter_map(|id| map.remove(&id)).collect()
        };
        for entry in entries {
            let _ = entry.abort_tx.send(true);
            if let Some(child) = &entry.session {
                child.lock().unwrap().status = SessionStatus::Paused;
            }
            if let Some(h) = entry.handle.lock().unwrap().take() {
                h.abort();
            }
        }
    }
}

fn spawn_inprocess_turn(
    app: App,
    entry: Arc<AgentEntry>,
    child_shared: SharedSession,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let trace_res = {
            let s = child_shared.lock().unwrap();
            app.observability.begin(&s.id, &s.settings.project_id)
        };
        let outcome = match trace_res {
            Ok(trace) => {
                let mut span = trace.span("agent", &format!("spawn.{}", entry.name), None);
                let r = nested_turn(app, child_shared.clone(), &trace, span.id()).await;
                span.finish(if r.is_ok() { "completed" } else { "failed" }, &Value::Null);
                r
            }
            Err(e) => Err(e),
        };
        let mut st = entry.status.lock().unwrap();
        if *st != AgentRunStatus::Interrupted {
            match outcome {
                Ok(_) => {
                    let answer = child_shared
                        .lock()
                        .unwrap()
                        .messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                        .map(|m| m.content.clone())
                        .unwrap_or_default();
                    *entry.answer.lock().unwrap() = Some(answer);
                    *st = AgentRunStatus::Completed;
                }
                Err(err) => {
                    *entry.error.lock().unwrap() = Some(format!("{err:#}"));
                    *st = AgentRunStatus::Failed;
                }
            }
            *entry.finished_at_ms.lock().unwrap() = Some(storage::now_ms());
        }
        entry.notify.notify_waiters();
    })
}

pub(crate) async fn list_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    Ok(Json(json!({
        "session_id": id,
        "agents": app.ext.agents.list(&id),
    })))
}

#[derive(Deserialize)]
pub(crate) struct SendAgentMessageBody {
    message: String,
    #[serde(default)]
    wake: Option<bool>,
}

pub(crate) async fn message_api(
    State(app): State<App>,
    AxumPath((id, aid)): AxumPath<(String, String)>,
    Json(body): Json<SendAgentMessageBody>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    app.ext
        .agents
        .send_message(&app, &id, &aid, &body.message, body.wake.unwrap_or(true))
        .map(Json)
        .map_err(|e| {
            if e.to_string().contains("not found") {
                not_found(e)
            } else {
                bad_request(e)
            }
        })
}

pub(crate) async fn interrupt_api(
    State(app): State<App>,
    AxumPath((id, aid)): AxumPath<(String, String)>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    app.ext
        .agents
        .interrupt(&id, &aid)
        .map(Json)
        .map_err(|e| {
            if e.to_string().contains("not found") {
                not_found(e)
            } else {
                bad_request(e)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn external_runtime_runs_and_interrupts_cleanly() {
        let base = std::env::temp_dir().join(format!("allpaka-subagent-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let mock_agent = base.join("mock_agent.py");
        std::fs::write(
            &mock_agent,
            "#!/usr/bin/env python3\nimport sys, json\ndata = json.loads(sys.stdin.read())\nprint(json.dumps({'ack': data['task']}))\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&mock_agent).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&mock_agent, perms).unwrap();
        }
        let (_tx, rx) = watch::channel(false);
        let argv = vec!["python3".into(), mock_agent.to_string_lossy().into_owned()];
        let out = run_external_runtime_argv(&argv, &base, &json!({ "task": "ping" }), rx)
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["ack"], "ping");
        let _ = std::fs::remove_dir_all(base);
    }
}
