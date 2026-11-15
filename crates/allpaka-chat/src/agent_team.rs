//! Coordinated Agent Team shared task list with DAG dependency and cycle validation.

use crate::ext::get_session;
use crate::storage;
use crate::{bad_request, conflict, not_found, ApiResult, App, SharedSession};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeamTaskStatus {
    #[default]
    Pending,
    InProgress,
    Blocked,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TeamTask {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub status: TeamTaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignee: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

pub(crate) fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "team_task_create",
                "description": "Create a task in the Agent Team DAG task list (validating dependencies against cycles).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Optional unique task identifier" },
                        "title": { "type": "string" },
                        "description": { "type": "string" },
                        "assignee": { "type": "string" },
                        "dependencies": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["title"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "team_task_list",
                "description": "List tasks in the Agent Team task graph with optional status or assignee filters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string", "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"] },
                        "assignee": { "type": "string" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "team_task_get",
                "description": "Inspect full details of a single team task by ID.",
                "parameters": {
                    "type": "object",
                    "properties": { "task_id": { "type": "string" } },
                    "required": ["task_id"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "team_task_update",
                "description": "Update status, assignee, dependencies, or result on an Agent Team task (enforces DAG acyclicity and dependency completion).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": { "type": "string" },
                        "title": { "type": "string" },
                        "description": { "type": "string" },
                        "status": { "type": "string", "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"] },
                        "assignee": { "type": "string" },
                        "dependencies": { "type": "array", "items": { "type": "string" } },
                        "result": { "type": "string" }
                    },
                    "required": ["task_id"]
                }
            }
        }),
    ]
}

pub(crate) fn validate_dag(tasks: &[TeamTask]) -> Result<()> {
    let mut by_id: HashMap<&str, usize> = HashMap::with_capacity(tasks.len());
    for (idx, task) in tasks.iter().enumerate() {
        if !storage::valid_id(&task.id) {
            bail!("Invalid task ID `{}`", task.id);
        }
        if by_id.insert(task.id.as_str(), idx).is_some() {
            bail!("Duplicate team task ID `{}`", task.id);
        }
    }
    for task in tasks {
        for dep in &task.dependencies {
            if dep == &task.id {
                bail!("Task `{}` cannot depend on itself", task.id);
            }
            let Some(&dep_idx) = by_id.get(dep.as_str()) else {
                bail!("Task `{}` depends on unknown task `{dep}`", task.id);
            };
            if task.status == TeamTaskStatus::Completed
                && tasks[dep_idx].status != TeamTaskStatus::Completed
            {
                bail!(
                    "Task `{}` cannot be marked completed while dependency `{dep}` is {:?}",
                    task.id,
                    tasks[dep_idx].status
                );
            }
        }
    }

    // 0 = unvisited, 1 = visiting, 2 = visited
    let mut state = vec![0u8; tasks.len()];
    fn dfs(
        idx: usize,
        tasks: &[TeamTask],
        by_id: &HashMap<&str, usize>,
        state: &mut [u8],
    ) -> Result<()> {
        match state[idx] {
            1 => bail!(
                "Dependency cycle detected involving task `{}`",
                tasks[idx].id
            ),
            2 => return Ok(()),
            _ => {}
        }
        state[idx] = 1;
        for dep in &tasks[idx].dependencies {
            if let Some(&dep_idx) = by_id.get(dep.as_str()) {
                dfs(dep_idx, tasks, by_id, state)?;
            }
        }
        state[idx] = 2;
        Ok(())
    }
    for i in 0..tasks.len() {
        if state[i] == 0 {
            dfs(i, tasks, &by_id, &mut state)?;
        }
    }
    Ok(())
}

fn summary_json(tasks: &[TeamTask]) -> Value {
    let mut pending = 0usize;
    let mut in_progress = 0usize;
    let mut blocked = 0usize;
    let mut completed = 0usize;
    let mut cancelled = 0usize;
    for t in tasks {
        match t.status {
            TeamTaskStatus::Pending => pending += 1,
            TeamTaskStatus::InProgress => in_progress += 1,
            TeamTaskStatus::Blocked => blocked += 1,
            TeamTaskStatus::Completed => completed += 1,
            TeamTaskStatus::Cancelled => cancelled += 1,
        }
    }
    json!({
        "total": tasks.len(),
        "pending": pending,
        "in_progress": in_progress,
        "blocked": blocked,
        "completed": completed,
        "cancelled": cancelled,
    })
}

pub(crate) fn execute_tool(
    app: &App,
    s: &SharedSession,
    name: &str,
    args: &Value,
) -> Result<Value> {
    match name {
        "team_task_create" => {
            let title = args["title"]
                .as_str()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .context("title is required")?;
            let now = storage::now_ms();
            let id = args
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("task-{}-{}", now, crate::evaluation::new_id()));
            let deps: Vec<String> = args
                .get("dependencies")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(|d| d.trim().to_owned())
                        .filter(|d| !d.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let task = TeamTask {
                id,
                title: title.to_owned(),
                description: args
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                status: TeamTaskStatus::Pending,
                assignee: args
                    .get("assignee")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned),
                dependencies: deps,
                result: None,
                created_at_ms: now,
                updated_at_ms: now,
            };
            let mut state = s.lock().unwrap();
            let mut candidate = state.team_tasks.clone();
            candidate.push(task.clone());
            validate_dag(&candidate)?;
            state.team_tasks = candidate;
            crate::save(app, &state)?;
            Ok(json!({ "created": task, "summary": summary_json(&state.team_tasks) }))
        }
        "team_task_list" => {
            let status_filter = args
                .get("status")
                .and_then(Value::as_str)
                .map(|v| serde_json::from_value::<TeamTaskStatus>(json!(v)))
                .transpose()?;
            let assignee_filter = args.get("assignee").and_then(Value::as_str);
            let state = s.lock().unwrap();
            let filtered: Vec<&TeamTask> = state
                .team_tasks
                .iter()
                .filter(|t| status_filter.is_none_or(|st| t.status == st))
                .filter(|t| assignee_filter.is_none_or(|a| t.assignee.as_deref() == Some(a)))
                .collect();
            Ok(json!({
                "tasks": filtered,
                "summary": summary_json(&state.team_tasks),
            }))
        }
        "team_task_get" => {
            let task_id = args["task_id"].as_str().context("task_id is required")?;
            let state = s.lock().unwrap();
            let task = state
                .team_tasks
                .iter()
                .find(|t| t.id == task_id)
                .with_context(|| format!("Team task `{task_id}` not found"))?;
            Ok(json!({ "task": task }))
        }
        "team_task_update" => {
            let task_id = args["task_id"].as_str().context("task_id is required")?;
            let mut state = s.lock().unwrap();
            let mut candidate = state.team_tasks.clone();
            let pos = candidate
                .iter()
                .position(|t| t.id == task_id)
                .with_context(|| format!("Team task `{task_id}` not found"))?;
            if let Some(title) = args
                .get("title")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                candidate[pos].title = title.to_owned();
            }
            if let Some(desc) = args.get("description").and_then(Value::as_str) {
                candidate[pos].description = desc.to_owned();
            }
            if let Some(st) = args.get("status") {
                candidate[pos].status = serde_json::from_value(st.clone())?;
            }
            if let Some(assignee) = args.get("assignee") {
                candidate[pos].assignee = assignee
                    .as_str()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned);
            }
            if let Some(deps) = args.get("dependencies").and_then(Value::as_array) {
                candidate[pos].dependencies = deps
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|d| d.trim().to_owned())
                    .filter(|d| !d.is_empty())
                    .collect();
            }
            if let Some(res) = args.get("result") {
                candidate[pos].result = res.as_str().map(ToOwned::to_owned);
            }
            candidate[pos].updated_at_ms = storage::now_ms();
            validate_dag(&candidate)?;
            let updated = candidate[pos].clone();
            state.team_tasks = candidate;
            crate::save(app, &state)?;
            Ok(json!({ "task": updated, "summary": summary_json(&state.team_tasks) }))
        }
        _ => bail!("Unknown team tool `{name}`"),
    }
}

pub(crate) async fn get_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let state = shared.lock().unwrap();
    Ok(Json(json!({
        "session_id": id,
        "team_mode": state.settings.team_mode,
        "tasks": state.team_tasks,
        "summary": summary_json(&state.team_tasks),
    })))
}

pub(crate) async fn post_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let action = body
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("create");
    match action {
        "create" => execute_tool(&app, &shared, "team_task_create", &body)
            .map(Json)
            .map_err(bad_request),
        "update" => execute_tool(&app, &shared, "team_task_update", &body)
            .map(Json)
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    not_found(e)
                } else {
                    conflict(e)
                }
            }),
        "delete" => {
            let task_id = body["task_id"]
                .as_str()
                .ok_or_else(|| bad_request("task_id is required"))?;
            let mut state = shared.lock().unwrap();
            let mut candidate = state.team_tasks.clone();
            let before_len = candidate.len();
            candidate.retain(|t| t.id != task_id);
            if candidate.len() == before_len {
                return Err(not_found("Team task not found"));
            }
            validate_dag(&candidate).map_err(conflict)?;
            state.team_tasks = candidate;
            crate::save(&app, &state).map_err(crate::internal)?;
            Ok(Json(json!({
                "deleted": task_id,
                "summary": summary_json(&state.team_tasks),
            })))
        }
        other => Err(bad_request(format!("Unsupported team action `{other}`"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: &str, deps: &[&str], status: TeamTaskStatus) -> TeamTask {
        TeamTask {
            id: id.into(),
            title: id.into(),
            description: String::new(),
            status,
            assignee: None,
            dependencies: deps.iter().map(|s| (*s).to_owned()).collect(),
            result: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    #[test]
    fn dag_validation_rejects_cycles_and_premature_completion() {
        // Valid DAG: t1 (completed) -> t2 (completed)
        let ok_graph = vec![
            mk("t1", &[], TeamTaskStatus::Completed),
            mk("t2", &["t1"], TeamTaskStatus::Completed),
        ];
        assert!(validate_dag(&ok_graph).is_ok());

        // Cycle: t1 -> t2 -> t1
        let cyc = vec![
            mk("t1", &["t2"], TeamTaskStatus::Pending),
            mk("t2", &["t1"], TeamTaskStatus::Pending),
        ];
        assert!(validate_dag(&cyc).is_err());

        // Premature completion: t2 completed while t1 still pending
        let premature = vec![
            mk("t1", &[], TeamTaskStatus::Pending),
            mk("t2", &["t1"], TeamTaskStatus::Completed),
        ];
        assert!(validate_dag(&premature).is_err());
    }
}
