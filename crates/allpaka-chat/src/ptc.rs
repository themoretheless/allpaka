//! Programmatic Tool Calling (`run_code`): runs sandboxed Python with a loopback
//! token-authenticated tool-call HTTP bridge (`studio.call_tool(...)`).

use crate::context::Project;
use crate::ext::resolve_project_root;
use crate::sandbox::{self, SandboxMode};
use crate::{
    evaluation, interaction, patch, spill, storage, tools, App, ExecutionPolicy, Mode, Settings,
    SharedSession,
};
use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;

const MAX_BRIDGE_CALLS: usize = 32;
const MAX_CODE_BYTES: usize = 64 * 1024;

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute sandboxed Python 3 code with programmatic workspace tool calling via `studio.call_tool(name, **kwargs)` over an authenticated loopback bridge (max 32 tool calls, max 60s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "Python 3 source code using optional `studio.call_tool(tool_name, ...)`" },
                    "timeout": { "type": "integer", "description": "Execution timeout in seconds (1-60, default 30)" }
                },
                "required": ["code"]
            }
        }
    })
}

#[derive(Clone)]
struct BridgeState {
    app: App,
    session: SharedSession,
    project: Project,
    settings: Settings,
    policy: ExecutionPolicy,
    sandbox_mode: SandboxMode,
    writable_roots: Vec<PathBuf>,
    token: String,
    calls: Arc<AtomicUsize>,
    cwd: PathBuf,
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut diff = ab.len() ^ bb.len();
    let n = ab.len().min(bb.len());
    for i in 0..n {
        diff |= (ab[i] ^ bb[i]) as usize;
    }
    diff == 0
}

fn is_bridge_allowed(tool: &str) -> bool {
    !matches!(
        tool,
        "run_code"
            | "ralph"
            | "workflow"
            | "subagent_spawn"
            | "ask_user_question"
            | "terminal_open"
            | "terminal_send"
            | "terminal_signal"
            | "terminal_close"
            | "schedule_create"
            | "schedule_edit"
    )
}

async fn bridge_dispatch(
    state: &BridgeState,
    name: &str,
    args: Value,
) -> std::result::Result<Value, String> {
    if !is_bridge_allowed(name) {
        return Err(format!(
            "Tool `{name}` cannot be called recursively from `run_code`"
        ));
    }
    if state.policy == ExecutionPolicy::files_only() {
        return Err("Tool calls are disabled under files-only policy".into());
    }
    let session_id = state.session.lock().unwrap().id.clone();

    interaction::request_approval_if_needed(
        &state.app,
        &state.session,
        state.settings.approval_policy,
        name,
        &args,
    )
    .await
    .map_err(|e| e.to_string())?;

    let allow_writes = !state.settings.mode.is_read_only()
        && state.settings.allow_writes
        && state.project.roots.iter().any(|r| r.writable);

    if name == "glob_files" {
        let wd = state.session.lock().unwrap().working_dir.clone();
        return patch::glob_files(&state.project, wd.as_deref(), &state.app.data, &args)
            .map_err(|e| e.to_string());
    }
    if name == "apply_patch" {
        let wd = state.session.lock().unwrap().working_dir.clone();
        return patch::apply_patch(
            &state.project,
            wd.as_deref(),
            &state.app.data,
            &state.app.ext.checkpoints,
            Some(&session_id),
            allow_writes,
            &args,
        )
        .map_err(|e| e.to_string());
    }
    if name == "run_command" {
        if !matches!(state.settings.mode, Mode::Auto | Mode::Goal)
            || state.policy.commands == crate::CommandExecution::Disabled
        {
            return Err("run_command is not permitted in this session".into());
        }
        let ctx = crate::ExecContext {
            data: Some(state.app.data.as_ref()),
            session_id: Some(session_id.as_str()),
            sandbox_mode: state.sandbox_mode,
            writable_roots: &state.writable_roots,
            allow_loopback_ipc: false,
        };
        return crate::execute_command_ctx(&state.cwd, &args, &ctx)
            .await
            .map_err(|e| e.to_string());
    }
    if matches!(
        name,
        "team_task_create" | "team_task_list" | "team_task_get" | "team_task_update"
    ) {
        if !state.settings.team_mode {
            return Err("team_mode is disabled".into());
        }
        return crate::agent_team::execute_tool(&state.app, &state.session, name, &args)
            .map_err(|e| e.to_string());
    }
    if !matches!(
        name,
        "read_file" | "write_file" | "edit_file" | "list_files" | "agentgrep"
    ) {
        return Err(format!("Unsupported tool `{name}` in run_code bridge"));
    }

    let path_str = args.get("path").and_then(Value::as_str).unwrap_or("");
    let is_write = tools::writes(name);
    let (root_ref, rel) =
        resolve_project_root(&state.project, path_str, is_write).map_err(|e| e.to_string())?;
    let abs = root_ref.path.join(&rel);
    if abs
        .canonicalize()
        .unwrap_or_else(|_| abs.clone())
        .starts_with(state.app.data.as_ref())
    {
        return Err("Conversation storage is not available to tools".into());
    }
    if is_write && !allow_writes {
        return Err("Writes are disabled in this session".into());
    }
    if is_write {
        if let Ok(resolved) = tools::resolve(&root_ref.path, &rel, true) {
            let _ = state.app.ext.checkpoints.capture_before_edit(
                state.app.data.as_ref(),
                &session_id,
                name,
                &format!("ptc {name} {rel}"),
                &[(rel.clone(), root_ref.alias.clone(), rel.clone(), resolved)],
            );
        }
    }
    let mut scoped_args = args;
    scoped_args["path"] = json!(rel);
    tools::execute_scoped(
        &root_ref.path,
        &state.settings,
        name,
        &scoped_args,
        Some(state.app.data.as_ref()),
    )
    .map_err(|e| e.to_string())
}

async fn handle_bridge_call(
    State(state): State<Arc<BridgeState>>,
    headers: HeaderMap,
    Json(req): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let expected = format!("Bearer {}", state.token);
    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !constant_time_eq(provided, &expected) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "ok": false, "error": "Invalid PTC bridge token" })),
        );
    }
    let count = state.calls.fetch_add(1, Ordering::Relaxed) + 1;
    if count > MAX_BRIDGE_CALLS {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({ "ok": false, "error": "PTC bridge tool call budget exceeded" })),
        );
    }
    let Some(tool) = req.get("tool").and_then(Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "Missing `tool`" })),
        );
    };
    let args = req.get("args").cloned().unwrap_or_else(|| json!({}));
    match bridge_dispatch(&state, tool, args).await {
        Ok(result) => (
            StatusCode::OK,
            Json(json!({ "ok": true, "result": result })),
        ),
        Err(err) => (StatusCode::OK, Json(json!({ "ok": false, "error": err }))),
    }
}

const PYTHON_BOOTSTRAP: &str = r#"
import base64, json, os, sys, types, urllib.request

_URL = os.environ["ALLPAKA_PTC_URL"]
_TOKEN = os.environ["ALLPAKA_PTC_TOKEN"]
_CODE = base64.b64decode(os.environ["ALLPAKA_PTC_CODE_B64"]).decode("utf-8")

def _call_tool(name, args=None, **kwargs):
    payload_args = dict(args or {})
    payload_args.update(kwargs)
    body = json.dumps({"tool": name, "args": payload_args}).encode("utf-8")
    req = urllib.request.Request(
        _URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + _TOKEN,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    if not res.get("ok"):
        raise RuntimeError(res.get("error") or "tool call failed")
    return res.get("result")

studio = types.SimpleNamespace(call_tool=_call_tool)
globals_ns = {"__name__": "__main__", "studio": studio, "call_tool": _call_tool}
exec(compile(_CODE, "<run_code>", "exec"), globals_ns)
"#;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute(
    app: &App,
    session: &SharedSession,
    project: &Project,
    settings: &Settings,
    sandbox_mode: SandboxMode,
    writable_roots: &[PathBuf],
    cwd: &std::path::Path,
    args: &Value,
) -> Result<Value> {
    let code = args["code"].as_str().context("code is required")?;
    if code.len() > MAX_CODE_BYTES {
        bail!("Python code exceeds {MAX_CODE_BYTES}-byte limit");
    }
    let timeout_secs = args
        .get("timeout")
        .and_then(Value::as_u64)
        .unwrap_or(30)
        .clamp(1, 60);

    let effective_policy = if sandbox_mode == SandboxMode::Unsandboxed {
        ExecutionPolicy::trusted_unsandboxed()
    } else {
        ExecutionPolicy::sandboxed(sandbox_mode)
    };

    let token = format!(
        "{}-{}",
        evaluation::new_id(),
        storage::hex_sha256(code.as_bytes())
    );
    let session_id = session.lock().unwrap().id.clone();

    let state = Arc::new(BridgeState {
        app: app.clone(),
        session: session.clone(),
        project: project.clone(),
        settings: settings.clone(),
        policy: effective_policy,
        sandbox_mode,
        writable_roots: writable_roots.to_vec(),
        token: token.clone(),
        calls: Arc::new(AtomicUsize::new(0)),
        cwd: cwd.to_path_buf(),
    });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("Failed to bind PTC loopback bridge")?;
    let addr = listener.local_addr()?;
    let router = Router::new()
        .route("/call", post(handle_bridge_call))
        .with_state(state.clone());
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    let py_args = vec!["-I".into(), "-c".into(), PYTHON_BOOTSTRAP.into()];
    let mut cmd = sandbox::build_command(
        "python3",
        &py_args,
        cwd,
        sandbox_mode,
        writable_roots,
        app.data.as_ref(),
        true,
    )?;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("ALLPAKA_PTC_URL", format!("http://{addr}/call"))
        .env("ALLPAKA_PTC_TOKEN", &token)
        .env(
            "ALLPAKA_PTC_CODE_B64",
            storage::base64_encode(code.as_bytes()),
        )
        .kill_on_drop(true);

    let child = cmd
        .spawn()
        .context("Failed to spawn python3 for run_code")?;
    let out_res = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        child.wait_with_output(),
    )
    .await;

    let _ = shutdown_tx.send(());
    let _ = server_handle.await;

    let out = match out_res {
        Ok(Ok(o)) => o,
        Ok(Err(err)) => return Err(err.into()),
        Err(_) => bail!("run_code timed out after {timeout_secs}s"),
    };

    let stdout_raw = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr_raw = String::from_utf8_lossy(&out.stderr).into_owned();
    let (stdout, stdout_spill, _, _) = spill::truncate_or_spill(
        Some(app.data.as_ref()),
        Some(&session_id),
        "run_code_stdout",
        &stdout_raw,
        spill::DEFAULT_INLINE_LIMIT,
    );
    let (stderr, stderr_spill, _, _) = spill::truncate_or_spill(
        Some(app.data.as_ref()),
        Some(&session_id),
        "run_code_stderr",
        &stderr_raw,
        spill::DEFAULT_INLINE_LIMIT,
    );

    Ok(json!({
        "exit_code": out.status.code().unwrap_or(-1),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_spill_id": stdout_spill,
        "stderr_spill_id": stderr_spill,
        "tool_calls": state.calls.load(Ordering::Relaxed).min(MAX_BRIDGE_CALLS),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Root;
    use crate::Session;
    use std::sync::Mutex;

    #[tokio::test]
    async fn python_loopback_bridge_calls_workspace_tool_and_blocks_recursion() {
        let base = std::env::temp_dir().join(format!("allpaka-ptc-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        let repo = repo.canonicalize().unwrap();
        let data = data.canonicalize().unwrap();
        std::fs::write(repo.join("data.txt"), "alpha\nbeta\ngamma\n").unwrap();

        let settings: Settings = serde_json::from_value(json!({
            "provider": "fixture",
            "model": "tiny",
            "project_id": "p1",
            "mode": "auto",
            "allow_writes": false,
            "sandbox_mode": "read_only"
        }))
        .unwrap();
        let sess = Session::new("ptc-s1".into(), settings.clone());
        let shared = Arc::new(Mutex::new(sess));

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

        let app = crate::ext::test_app(&data, &repo, vec![proj.clone()]);

        let script = r#"
res = studio.call_tool("read_file", path="data.txt")
print("LINES=" + str(len(res["content"].strip().splitlines())))
try:
    studio.call_tool("run_code", code="print(1)")
    print("RECURSION_UNEXPECTED")
except RuntimeError as e:
    print("BLOCKED=" + str(e))
"#;
        let out = execute(
            &app,
            &shared,
            &proj,
            &settings,
            SandboxMode::ReadOnly,
            &[],
            &repo,
            &json!({ "code": script, "timeout": 15 }),
        )
        .await
        .unwrap();

        assert_eq!(out["exit_code"], 0, "stderr: {}", out["stderr"]);
        let stdout = out["stdout"].as_str().unwrap();
        assert!(stdout.contains("LINES=3"), "stdout={stdout:?}");
        assert!(stdout.contains("BLOCKED="), "stdout={stdout:?}");
        assert_eq!(out["tool_calls"], 2);

        let _ = std::fs::remove_dir_all(base);
    }
}
