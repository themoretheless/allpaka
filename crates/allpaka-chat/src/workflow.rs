//! Multi-stage DAG workflows (`workflow`) and iterative verify-until-pass loops (`ralph`).

use crate::context::Project;
use crate::ext::{get_session, project_for_session};
use crate::runtime::subagent::nested_turn;
use crate::sandbox::SandboxMode;
use crate::{
    bad_request, conflict, not_found, state_file, storage, ApiResult, App, Ephemeral,
    ExecutionPolicy, Message, Mode, Session, Settings, SharedSession,
};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const MAX_RALPH_ITERS: usize = 8;
const MAX_WORKFLOW_STAGES: usize = 20;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct WorkflowStage {
    pub id: String,
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_command: Option<String>,
    #[serde(default)]
    pub allow_writes: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct WorkflowSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub stages: Vec<WorkflowStage>,
    #[serde(default)]
    pub created_at_ms: u64,
}

pub(crate) fn ralph_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "ralph",
            "description": "Run a bounded iterative verify-until-pass autonomous loop (max 8 iterations) with optional verification command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": { "type": "string", "description": "Concrete task objective" },
                    "verify_command": { "type": "string", "description": "Optional shell command that must exit 0 to pass verification" },
                    "max_iterations": { "type": "integer", "description": "Iteration budget (1-8, default 3)" },
                    "allow_writes": { "type": "boolean" }
                },
                "required": ["goal"]
            }
        }
    })
}

pub(crate) fn workflow_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "workflow",
            "description": "Manage and run multi-stage DAG workflows (`list`, `save`, `run`, `delete`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "save", "run", "delete"] },
                    "workflow_id": { "type": "string" },
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "stages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "name": { "type": "string" },
                                "prompt": { "type": "string" },
                                "depends_on": { "type": "array", "items": { "type": "string" } },
                                "verify_command": { "type": "string" },
                                "allow_writes": { "type": "boolean" }
                            },
                            "required": ["id", "name", "prompt"]
                        }
                    }
                },
                "required": ["action"]
            }
        }
    })
}

pub(crate) fn validate_workflow(spec: &WorkflowSpec) -> Result<()> {
    if !storage::valid_id(&spec.id) {
        bail!("Invalid workflow id `{}`", spec.id);
    }
    if spec.name.trim().is_empty() || spec.name.len() > 120 {
        bail!("Workflow name must be 1-120 characters");
    }
    if spec.stages.is_empty() || spec.stages.len() > MAX_WORKFLOW_STAGES {
        bail!("Workflow must have 1-{MAX_WORKFLOW_STAGES} stages");
    }
    let _ = topo_order(&spec.stages)?;
    Ok(())
}

pub(crate) fn topo_order(stages: &[WorkflowStage]) -> Result<Vec<usize>> {
    let mut by_id: HashMap<&str, usize> = HashMap::new();
    for (idx, st) in stages.iter().enumerate() {
        if !storage::valid_id(&st.id) {
            bail!("Invalid stage id `{}`", st.id);
        }
        if by_id.insert(st.id.as_str(), idx).is_some() {
            bail!("Duplicate workflow stage id `{}`", st.id);
        }
    }
    let mut visited = vec![0u8; stages.len()];
    let mut order = Vec::with_capacity(stages.len());
    fn visit(
        idx: usize,
        stages: &[WorkflowStage],
        by_id: &HashMap<&str, usize>,
        visited: &mut [u8],
        order: &mut Vec<usize>,
    ) -> Result<()> {
        match visited[idx] {
            1 => bail!(
                "Cycle detected in workflow involving stage `{}`",
                stages[idx].id
            ),
            2 => return Ok(()),
            _ => {}
        }
        visited[idx] = 1;
        for dep in &stages[idx].depends_on {
            if dep == &stages[idx].id {
                bail!("Stage `{dep}` cannot depend on itself");
            }
            let &dep_idx = by_id
                .get(dep.as_str())
                .with_context(|| format!("Unknown dependency stage `{dep}`"))?;
            visit(dep_idx, stages, by_id, visited, order)?;
        }
        visited[idx] = 2;
        order.push(idx);
        Ok(())
    }
    for i in 0..stages.len() {
        if visited[i] == 0 {
            visit(i, stages, &by_id, &mut visited, &mut order)?;
        }
    }
    Ok(order)
}

pub(crate) fn load_workflows(app: &App) -> Result<Vec<WorkflowSpec>> {
    state_file::load_list(
        app.data.as_path(),
        "workflows",
        "workflow",
        validate_workflow,
    )
}

#[allow(clippy::too_many_arguments)]
async fn run_stage_turn(
    app: &App,
    settings: &Settings,
    project: &Project,
    tree: &crate::harness::Tree,
    policy: ExecutionPolicy,
    system: &str,
    prompt: &str,
    allow_writes: bool,
) -> Result<String> {
    let mut child = settings.clone();
    child.allow_writes &= allow_writes;
    child.max_steps = (settings.max_steps / 2).max(4);
    if child.mode == Mode::Goal {
        child.mode = Mode::Auto;
    }
    let mut s = Session::new(crate::evaluation::new_id(), child);
    s.ephemeral = Some(Arc::new(Ephemeral {
        project: project.clone(),
        harness: Some(tree.clone()),
        system: Some(system.to_owned()),
        nested: true,
        policy,
    }));
    s.messages.push(Message::text("user", prompt));
    let shared: SharedSession = Arc::new(Mutex::new(s));
    let trace = app
        .observability
        .begin(&shared.lock().unwrap().id, &settings.project_id)?;
    let mut span = trace.span("workflow", "stage", None);
    let outcome = nested_turn(app.clone(), shared.clone(), &trace, span.id()).await;
    span.finish(
        if outcome.is_ok() {
            "completed"
        } else {
            "failed"
        },
        &Value::Null,
    );
    outcome?;
    let answer = shared
        .lock()
        .unwrap()
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
        .map(|m| m.content.clone())
        .unwrap_or_default();
    Ok(answer)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_ralph(
    app: &App,
    session_id: &str,
    settings: &Settings,
    project: &Project,
    tree: &crate::harness::Tree,
    policy: ExecutionPolicy,
    sandbox_mode: SandboxMode,
    writable_roots: &[PathBuf],
    cwd: &std::path::Path,
    args: &Value,
) -> Result<Value> {
    let goal = args["goal"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .context("goal is required")?;
    let max_iters = args
        .get("max_iterations")
        .and_then(Value::as_u64)
        .unwrap_or(3)
        .clamp(1, MAX_RALPH_ITERS as u64) as usize;
    let verify_cmd = args
        .get("verify_command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let allow_writes = settings.allow_writes
        && args
            .get("allow_writes")
            .and_then(Value::as_bool)
            .unwrap_or(true);

    let mut history_log = Vec::new();
    let mut feedback = String::new();
    for iter in 1..=max_iters {
        let prompt = if feedback.is_empty() {
            format!("Iteration {iter}/{max_iters} goal:\n{goal}")
        } else {
            format!(
                "Iteration {iter}/{max_iters} goal:\n{goal}\n\nPrevious verification attempt failed:\n{feedback}"
            )
        };
        let answer = run_stage_turn(
            app,
            settings,
            project,
            tree,
            policy,
            "You are executing a Ralph verify-until-pass loop iteration. Make targeted progress toward passing the verification check.",
            &prompt,
            allow_writes,
        )
        .await?;

        if let Some(cmd) = verify_cmd {
            let ctx = crate::ExecContext {
                data: Some(app.data.as_ref()),
                session_id: Some(session_id),
                sandbox_mode,
                writable_roots,
                allow_loopback_ipc: false,
            };
            let v =
                crate::execute_command_ctx(cwd, &json!({ "command": cmd, "timeout": 30 }), &ctx)
                    .await?;
            let code = v["exit_code"].as_i64().unwrap_or(-1);
            history_log.push(json!({
                "iteration": iter,
                "exit_code": code,
                "answer": answer,
            }));
            if code == 0 {
                return Ok(json!({
                    "passed": true,
                    "iterations": iter,
                    "summary": answer,
                    "verification": v,
                    "history": history_log,
                }));
            }
            feedback = format!(
                "exit_code={code}\nstdout:\n{}\nstderr:\n{}",
                v["stdout"].as_str().unwrap_or(""),
                v["stderr"].as_str().unwrap_or("")
            );
        } else {
            history_log.push(json!({ "iteration": iter, "answer": answer }));
            return Ok(json!({
                "passed": true,
                "iterations": iter,
                "summary": answer,
                "history": history_log,
            }));
        }
    }

    Ok(json!({
        "passed": false,
        "iterations": max_iters,
        "last_failure": feedback,
        "history": history_log,
    }))
}

pub(crate) fn save_spec_from_args(app: &App, args: &Value) -> Result<Value> {
    let id = args
        .get("workflow_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("wf-{}", storage::now_ms()));
    let stages: Vec<WorkflowStage> =
        serde_json::from_value(args.get("stages").cloned().context("stages is required")?)?;
    let spec = WorkflowSpec {
        id: id.clone(),
        name: args
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(&id)
            .to_owned(),
        description: args
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        stages,
        created_at_ms: storage::now_ms(),
    };
    validate_workflow(&spec)?;
    let mut list = load_workflows(app)?;
    if let Some(existing) = list.iter_mut().find(|w| w.id == spec.id) {
        *existing = spec.clone();
    } else {
        list.push(spec.clone());
    }
    state_file::save_list(app.data.as_path(), "workflows", &list)?;
    Ok(json!({ "saved": spec }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_workflow(
    app: &App,
    session_id: &str,
    settings: &Settings,
    project: &Project,
    tree: &crate::harness::Tree,
    policy: ExecutionPolicy,
    sandbox_mode: SandboxMode,
    writable_roots: &[PathBuf],
    cwd: &std::path::Path,
    args: &Value,
) -> Result<Value> {
    let action = args["action"].as_str().context("action is required")?;
    match action {
        "list" => {
            let list = load_workflows(app)?;
            Ok(json!({ "workflows": list }))
        }
        "save" => save_spec_from_args(app, args),
        "delete" => {
            let wid = args["workflow_id"]
                .as_str()
                .context("workflow_id is required")?;
            let mut list = load_workflows(app)?;
            let before = list.len();
            list.retain(|w| w.id != wid);
            if list.len() == before {
                bail!("Workflow `{wid}` not found");
            }
            state_file::save_list(app.data.as_path(), "workflows", &list)?;
            Ok(json!({ "deleted": wid }))
        }
        "run" => {
            let stages: Vec<WorkflowStage> = if let Some(s_val) = args.get("stages") {
                serde_json::from_value(s_val.clone())?
            } else {
                let wid = args["workflow_id"]
                    .as_str()
                    .context("workflow_id or stages is required for workflow run")?;
                let list = load_workflows(app)?;
                list.into_iter()
                    .find(|w| w.id == wid)
                    .with_context(|| format!("Workflow `{wid}` not found"))?
                    .stages
            };
            if stages.is_empty() || stages.len() > MAX_WORKFLOW_STAGES {
                bail!("Workflow must have 1-{MAX_WORKFLOW_STAGES} stages");
            }
            let order = topo_order(&stages)?;
            let mut outputs: HashMap<String, String> = HashMap::new();
            let mut results = Vec::new();
            let mut failed_stage: Option<String> = None;

            for idx in order {
                let stage = &stages[idx];
                let mut context_blocks = Vec::new();
                for dep in &stage.depends_on {
                    if let Some(out) = outputs.get(dep) {
                        context_blocks.push(format!("[Upstream stage `{dep}` output]\n{out}"));
                    }
                }
                let prompt = if context_blocks.is_empty() {
                    stage.prompt.clone()
                } else {
                    format!("{}\n\n{}", context_blocks.join("\n\n"), stage.prompt)
                };
                let answer = run_stage_turn(
                    app,
                    settings,
                    project,
                    tree,
                    policy,
                    &format!(
                        "You are executing workflow stage `{}` ({}).",
                        stage.name, stage.id
                    ),
                    &prompt,
                    settings.allow_writes && stage.allow_writes,
                )
                .await?;
                let mut verified = true;
                let mut verify_detail = Value::Null;
                if let Some(cmd) = &stage.verify_command {
                    let ctx = crate::ExecContext {
                        data: Some(app.data.as_ref()),
                        session_id: Some(session_id),
                        sandbox_mode,
                        writable_roots,
                        allow_loopback_ipc: false,
                    };
                    let v = crate::execute_command_ctx(
                        cwd,
                        &json!({ "command": cmd, "timeout": 30 }),
                        &ctx,
                    )
                    .await?;
                    verified = v["exit_code"].as_i64() == Some(0);
                    verify_detail = v;
                }
                outputs.insert(stage.id.clone(), answer.clone());
                results.push(json!({
                    "stage_id": stage.id,
                    "name": stage.name,
                    "verified": verified,
                    "verification": verify_detail,
                    "output": answer,
                }));
                if !verified {
                    failed_stage = Some(stage.id.clone());
                    break;
                }
            }
            let completed: HashSet<&str> = outputs.keys().map(String::as_str).collect();
            Ok(json!({
                "ok": failed_stage.is_none(),
                "failed_stage": failed_stage,
                "completed_count": completed.len(),
                "stages": results,
            }))
        }
        other => bail!("Unsupported workflow action `{other}`"),
    }
}

pub(crate) async fn list_api(State(app): State<App>) -> ApiResult<Value> {
    let workflows = load_workflows(&app).map_err(crate::internal)?;
    Ok(Json(json!({ "workflows": workflows })))
}

pub(crate) async fn save_api(State(app): State<App>, Json(body): Json<Value>) -> ApiResult<Value> {
    save_spec_from_args(&app, &body)
        .map(Json)
        .map_err(bad_request)
}

pub(crate) async fn delete_api(
    State(app): State<App>,
    AxumPath(wid): AxumPath<String>,
) -> ApiResult<Value> {
    let mut list = load_workflows(&app).map_err(crate::internal)?;
    let before = list.len();
    list.retain(|w| w.id != wid);
    if list.len() == before {
        return Err(not_found("Workflow not found"));
    }
    state_file::save_list(app.data.as_path(), "workflows", &list).map_err(crate::internal)?;
    Ok(Json(json!({ "deleted": wid })))
}

pub(crate) async fn run_session_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(mut body): Json<Value>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, settings, project) = project_for_session(&app, &shared)?;
    if settings.mode.is_read_only() {
        return Err(conflict("Running workflows requires non-read-only mode"));
    }
    body["action"] = json!("run");
    let tree =
        crate::harness::active_tree(&app.data, &settings.project_id).map_err(crate::internal)?;
    let writable_roots: Vec<PathBuf> = if settings.allow_writes {
        project
            .roots
            .iter()
            .filter(|r| r.writable)
            .map(|r| r.path.clone())
            .collect()
    } else {
        Vec::new()
    };
    let cwd = project
        .roots
        .first()
        .map(|r| r.path.clone())
        .ok_or_else(|| bad_request("No project root available"))?;
    let res = execute_workflow(
        &app,
        &id,
        &settings,
        &project,
        &tree,
        ExecutionPolicy::trusted_unsandboxed(),
        settings.sandbox_mode,
        &writable_roots,
        &cwd,
        &body,
    )
    .await
    .map_err(bad_request)?;
    Ok(Json(res))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_topo_order_respects_dag_and_rejects_cycles() {
        let stages = vec![
            WorkflowStage {
                id: "test".into(),
                name: "Test".into(),
                prompt: "run tests".into(),
                depends_on: vec!["build".into()],
                ..Default::default()
            },
            WorkflowStage {
                id: "build".into(),
                name: "Build".into(),
                prompt: "compile".into(),
                depends_on: vec![],
                ..Default::default()
            },
        ];
        let order = topo_order(&stages).unwrap();
        assert_eq!(order, vec![1, 0]);

        let cyclic = vec![
            WorkflowStage {
                id: "a".into(),
                name: "a".into(),
                prompt: "a".into(),
                depends_on: vec!["b".into()],
                ..Default::default()
            },
            WorkflowStage {
                id: "b".into(),
                name: "b".into(),
                prompt: "b".into(),
                depends_on: vec!["a".into()],
                ..Default::default()
            },
        ];
        assert!(topo_order(&cyclic).is_err());
    }
}
