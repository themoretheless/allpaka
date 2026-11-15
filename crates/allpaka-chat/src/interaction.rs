//! Interactive user confirmations (`PendingApproval`) and structured clarifying
//! questions (`ask_user_question` / `PendingQuestion`).
//!
//! When a turn needs human input mid-flight, it registers a one-shot waiter in
//! [`Broker`], records the pending prompt on the [`Session`] (persisted to
//! disk), and awaits the reply up to [`INTERACTION_TIMEOUT_SECS`]. Cancellation
//! or drop clears both the session prompt and the waiter.

use crate::{
    bad_request, error, not_found,
    sandbox::ApprovalPolicy,
    save, storage,
    types::{Mode, Session},
    ApiResult, App, SharedSession,
};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::oneshot;

const INTERACTION_TIMEOUT_SECS: u64 = 300;
static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct PendingApproval {
    pub id: String,
    pub tool: String,
    pub summary: String,
    pub args_preview: String,
    pub created_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct QuestionItem {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default)]
    pub multi_select: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct PendingQuestion {
    pub id: String,
    pub questions: Vec<QuestionItem>,
    pub created_ms: u64,
}

#[derive(Debug)]
pub(crate) enum InteractionReply {
    ApproveOnce,
    ApproveSession,
    Deny { reason: String },
    Answer { answers: Value },
}

type WaiterMap = HashMap<String, (String, oneshot::Sender<InteractionReply>)>;

#[derive(Clone, Default)]
pub(crate) struct Broker {
    waiters: Arc<Mutex<WaiterMap>>,
    session_allowlist: Arc<Mutex<HashMap<String, HashSet<String>>>>,
}

fn waiter_key(session_id: &str, request_id: &str) -> String {
    format!("{session_id}:{request_id}")
}

impl Broker {
    pub(crate) fn is_session_approved(&self, session_id: &str, tool: &str) -> bool {
        self.session_allowlist
            .lock()
            .unwrap()
            .get(session_id)
            .is_some_and(|set| set.contains(tool))
    }

    pub(crate) fn allow_for_session(&self, session_id: &str, tool: &str) {
        self.session_allowlist
            .lock()
            .unwrap()
            .entry(session_id.to_owned())
            .or_default()
            .insert(tool.to_owned());
    }

    pub(crate) fn register(
        &self,
        session_id: &str,
        request_id: &str,
    ) -> oneshot::Receiver<InteractionReply> {
        let (tx, rx) = oneshot::channel();
        self.waiters.lock().unwrap().insert(
            waiter_key(session_id, request_id),
            (session_id.to_owned(), tx),
        );
        rx
    }

    pub(crate) fn remove(&self, session_id: &str, request_id: &str) {
        self.waiters
            .lock()
            .unwrap()
            .remove(&waiter_key(session_id, request_id));
    }

    pub(crate) fn cancel_waiters(&self, session_id: &str) {
        let mut waiters = self.waiters.lock().unwrap();
        let keys: Vec<String> = waiters
            .iter()
            .filter(|(_, (sid, _))| sid == session_id)
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys {
            if let Some((_, tx)) = waiters.remove(&key) {
                let _ = tx.send(InteractionReply::Deny {
                    reason: "Cancelled".into(),
                });
            }
        }
    }

    pub(crate) fn clear_session(&self, session_id: &str) {
        self.cancel_waiters(session_id);
        self.session_allowlist.lock().unwrap().remove(session_id);
    }

    pub(crate) fn deliver(
        &self,
        session_id: &str,
        request_id: &str,
        reply: InteractionReply,
    ) -> Result<()> {
        let key = waiter_key(session_id, request_id);
        let (_, tx) = self
            .waiters
            .lock()
            .unwrap()
            .remove(&key)
            .context("No pending interaction matching request_id")?;
        if let InteractionReply::ApproveSession = &reply {
            // Caller (request_approval) also records tool name in session_allowlist;
            // delivering ApproveSession wakes the waiter immediately.
        }
        tx.send(reply)
            .map_err(|_| anyhow::anyhow!("Waiting turn is no longer active"))
    }
}

pub(crate) fn ask_question_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": "Pause and ask the user 1–4 clarifying multiple-choice or free-form questions when requirements or preferences are genuinely ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "minLength": 1, "maxLength": 64 },
                                "question": { "type": "string", "minLength": 1, "maxLength": 1000 },
                                "options": {
                                    "type": "array",
                                    "maxItems": 8,
                                    "items": { "type": "string", "maxLength": 300 }
                                },
                                "multi_select": { "type": "boolean" }
                            },
                            "required": ["question"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": false
            }
        }
    })
}

pub(crate) fn clear_pending_fields(session: &mut Session) {
    session.pending_approval = None;
    session.pending_question = None;
}

fn next_request_id(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}",
        storage::now_ms(),
        REQ_SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

fn is_read_only_builtin(tool: &str, args: &Value) -> bool {
    if tool == "working_directory" {
        return args.get("action").and_then(Value::as_str) != Some("set");
    }
    if tool == "forge_pr" {
        return matches!(
            args.get("action").and_then(Value::as_str),
            Some("status" | "detect" | "checks")
        );
    }
    if tool == "plugin_manager" {
        return args.get("action").and_then(Value::as_str) == Some("list");
    }
    matches!(
        tool,
        "list_files"
            | "read_file"
            | "agentgrep"
            | "git_diff"
            | "glob_files"
            | "read_spill"
            | "conversation_search"
            | "memory_recall"
            | "extract_document"
            | "runtime_inspect"
            | "ask_user_question"
            | "list_agents"
            | "wait_agent"
            | "team_task_list"
            | "team_task_get"
            | "user_terminal_list"
            | "user_terminal_read"
            | "terminal_list"
            | "terminal_read"
            | "schedule_list"
            | "schedule_view"
            | "list_mcp_resources"
            | "list_mcp_resource_templates"
            | "read_mcp_resource"
            | "lsp"
            | "web_fetch"
            | "web_search"
    )
}

fn is_command_or_exec_tool(tool: &str) -> bool {
    matches!(
        tool,
        "run_command"
            | "background"
            | "run_script"
            | "terminal_open"
            | "terminal_send"
            | "terminal_signal"
            | "terminal_close"
            | "run_code"
            | "ralph"
            | "workflow"
            | "create_worktree"
            | "forge_pr"
            | "subagent_spawn"
            | "send_agent_message"
            | "interrupt_agent"
            | "browser_use"
            | "computer_use"
            | "plugin_manager"
    )
}

fn is_write_tool(tool: &str, args: &Value) -> bool {
    if crate::tools::writes(tool)
        || matches!(tool, "apply_patch" | "schedule_create" | "schedule_edit")
    {
        return true;
    }
    if tool == "working_directory" && args.get("action").and_then(Value::as_str) == Some("set") {
        return true;
    }
    if let Some(mcp_name) = tool.strip_prefix("mcp_") {
        return !is_read_only_builtin(mcp_name, args)
            && [
                "write", "update", "delete", "create", "remove", "apply", "set_",
            ]
            .iter()
            .any(|k| mcp_name.contains(k));
    }
    false
}

pub(crate) fn requires_approval(policy: ApprovalPolicy, tool: &str, args: &Value) -> bool {
    if tool == "ask_user_question" {
        return false;
    }
    match policy {
        ApprovalPolicy::Never => false,
        ApprovalPolicy::Commands => {
            is_command_or_exec_tool(tool) && !is_read_only_builtin(tool, args)
        }
        ApprovalPolicy::WritesAndCommands => {
            (is_command_or_exec_tool(tool) || is_write_tool(tool, args))
                && !is_read_only_builtin(tool, args)
        }
        ApprovalPolicy::Always => true,
    }
}

struct PendingGuard {
    app: App,
    s: SharedSession,
    session_id: String,
    request_id: String,
    is_question: bool,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.app
            .interactions
            .remove(&self.session_id, &self.request_id);
        if let Ok(mut session) = self.s.lock() {
            if self.is_question {
                if session
                    .pending_question
                    .as_ref()
                    .is_some_and(|p| p.id == self.request_id)
                {
                    session.pending_question = None;
                }
            } else if session
                .pending_approval
                .as_ref()
                .is_some_and(|p| p.id == self.request_id)
            {
                session.pending_approval = None;
            }
            let _ = save(&self.app, &session);
        }
    }
}

pub(crate) async fn request_approval_if_needed(
    app: &App,
    s: &SharedSession,
    policy: ApprovalPolicy,
    tool: &str,
    args: &Value,
) -> Result<()> {
    if !requires_approval(policy, tool, args) {
        return Ok(());
    }
    let session_id = s.lock().unwrap().id.clone();
    if app.interactions.is_session_approved(&session_id, tool) {
        return Ok(());
    }
    let request_id = next_request_id("apr");
    let raw_preview = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
    let cut = crate::spill::floor_char_boundary(&raw_preview, 1500);
    let args_preview = if raw_preview.len() > cut {
        format!("{}…", &raw_preview[..cut])
    } else {
        raw_preview
    };
    let summary = format!("Approve tool `{tool}`");
    let rx = app.interactions.register(&session_id, &request_id);
    let _guard = PendingGuard {
        app: app.clone(),
        s: s.clone(),
        session_id: session_id.clone(),
        request_id: request_id.clone(),
        is_question: false,
    };
    {
        let mut state = s.lock().unwrap();
        state.pending_approval = Some(PendingApproval {
            id: request_id.clone(),
            tool: tool.to_owned(),
            summary,
            args_preview,
            created_ms: storage::now_ms(),
        });
        save(app, &state)?;
    }

    let reply = tokio::time::timeout(Duration::from_secs(INTERACTION_TIMEOUT_SECS), rx)
        .await
        .map_err(|_| anyhow::anyhow!("Tool approval timed out after {INTERACTION_TIMEOUT_SECS}s"))?
        .map_err(|_| anyhow::anyhow!("Approval request cancelled"))?;
    match reply {
        InteractionReply::ApproveOnce => Ok(()),
        InteractionReply::ApproveSession => {
            app.interactions.allow_for_session(&session_id, tool);
            Ok(())
        }
        InteractionReply::Deny { reason } => {
            let clean = if reason.trim().is_empty() {
                "Denied by user".to_string()
            } else {
                reason.trim().to_string()
            };
            bail!("Tool `{tool}` denied by user: {clean}");
        }
        InteractionReply::Answer { .. } => bail!("Invalid interaction reply for approval"),
    }
}

pub(crate) async fn ask_user_question(app: &App, s: &SharedSession, args: &Value) -> Result<Value> {
    let raw_questions = args["questions"]
        .as_array()
        .context("questions array is required")?;
    if raw_questions.is_empty() || raw_questions.len() > 4 {
        bail!("questions must contain 1–4 items");
    }
    let mut items = Vec::with_capacity(raw_questions.len());
    for (idx, q) in raw_questions.iter().enumerate() {
        let prompt = q["question"]
            .as_str()
            .map(str::trim)
            .filter(|t| !t.is_empty() && t.len() <= 1000)
            .context("Each question must be 1–1000 characters")?
            .to_owned();
        let id = q
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| storage::valid_id(id))
            .map(str::to_owned)
            .unwrap_or_else(|| format!("q{}", idx + 1));
        let options = q
            .get("options")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::trim))
                    .filter(|s| !s.is_empty())
                    .take(8)
                    .map(|s| s.chars().take(300).collect::<String>())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let multi_select = q
            .get("multi_select")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        items.push(QuestionItem {
            id,
            question: prompt,
            options,
            multi_select,
        });
    }

    let (session_id, mode) = {
        let state = s.lock().unwrap();
        (state.id.clone(), state.settings.mode)
    };
    if !matches!(mode, Mode::Plan | Mode::Auto | Mode::Goal) {
        bail!("ask_user_question requires Plan, Auto, or Goal mode");
    }
    let request_id = next_request_id("ask");
    let rx = app.interactions.register(&session_id, &request_id);
    let _guard = PendingGuard {
        app: app.clone(),
        s: s.clone(),
        session_id: session_id.clone(),
        request_id: request_id.clone(),
        is_question: true,
    };
    {
        let mut state = s.lock().unwrap();
        state.pending_question = Some(PendingQuestion {
            id: request_id.clone(),
            questions: items.clone(),
            created_ms: storage::now_ms(),
        });
        save(app, &state)?;
    }

    let reply = tokio::time::timeout(Duration::from_secs(INTERACTION_TIMEOUT_SECS), rx)
        .await
        .map_err(|_| anyhow::anyhow!("User did not respond within {INTERACTION_TIMEOUT_SECS}s"))?
        .map_err(|_| anyhow::anyhow!("Question cancelled"))?;
    match reply {
        InteractionReply::Answer { answers } => Ok(json!({
            "answered": true,
            "request_id": request_id,
            "answers": answers,
        })),
        InteractionReply::Deny { reason } => Ok(json!({
            "answered": false,
            "dismissed_reason": reason,
        })),
        _ => bail!("Invalid interaction reply for question"),
    }
}

#[derive(Deserialize)]
pub(crate) struct InteractionInput {
    request_id: String,
    kind: String,
    #[serde(default)]
    decision: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    answers: Option<Value>,
}

pub(crate) async fn respond_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(input): Json<InteractionInput>,
) -> ApiResult<Value> {
    if !app.sessions.lock().unwrap().contains_key(&id) {
        return Err(not_found("Conversation not found"));
    }
    if input.request_id.trim().is_empty() || input.request_id.len() > 100 {
        return Err(bad_request("Invalid request_id"));
    }
    let reply = match input.kind.as_str() {
        "approval" => match input.decision.as_deref().unwrap_or("") {
            "approve_once" => InteractionReply::ApproveOnce,
            "approve_session" => InteractionReply::ApproveSession,
            "deny" => InteractionReply::Deny {
                reason: input.reason.unwrap_or_default().chars().take(500).collect(),
            },
            _ => {
                return Err(bad_request(
                    "decision must be approve_once, approve_session, or deny",
                ))
            }
        },
        "question" => {
            if input.decision.as_deref() == Some("dismiss") {
                InteractionReply::Deny {
                    reason: input.reason.unwrap_or_else(|| "Dismissed by user".into()),
                }
            } else {
                let answers = input
                    .answers
                    .ok_or_else(|| bad_request("answers payload is required"))?;
                InteractionReply::Answer { answers }
            }
        }
        _ => return Err(bad_request("kind must be approval or question")),
    };
    app.interactions
        .deliver(&id, &input.request_id, reply)
        .map_err(|e| error(StatusCode::CONFLICT, e))?;
    Ok(Json(
        json!({ "accepted": true, "request_id": input.request_id }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_policy_matrix_matches_expectation() {
        assert!(!requires_approval(
            ApprovalPolicy::Never,
            "run_command",
            &json!({"command":"ls"})
        ));
        assert!(requires_approval(
            ApprovalPolicy::Commands,
            "run_command",
            &json!({"command":"ls"})
        ));
        assert!(!requires_approval(
            ApprovalPolicy::Commands,
            "write_file",
            &json!({"path":"a","content":"b"})
        ));
        assert!(requires_approval(
            ApprovalPolicy::WritesAndCommands,
            "write_file",
            &json!({"path":"a","content":"b"})
        ));
        assert!(!requires_approval(
            ApprovalPolicy::WritesAndCommands,
            "read_file",
            &json!({"path":"a"})
        ));
        assert!(requires_approval(
            ApprovalPolicy::Always,
            "read_file",
            &json!({"path":"a"})
        ));
        assert!(!requires_approval(
            ApprovalPolicy::Always,
            "ask_user_question",
            &json!({"questions":[]})
        ));
    }

    #[test]
    fn broker_delivers_and_isolates_waiters() {
        let broker = Broker::default();
        let rx = broker.register("s1", "r1");
        assert!(broker
            .deliver("s2", "r1", InteractionReply::ApproveOnce)
            .is_err());
        broker
            .deliver("s1", "r1", InteractionReply::ApproveSession)
            .unwrap();
        let reply = rx.blocking_recv().unwrap();
        assert!(matches!(reply, InteractionReply::ApproveSession));
    }
}
