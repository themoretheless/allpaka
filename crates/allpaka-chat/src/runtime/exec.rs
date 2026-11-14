//! Shell command execution for the command tool, OS sandbox wrapping,
//! output spill handling, and MCP tool lookup.

use crate::{sandbox, spill, *};

pub(crate) fn mcp_lookup(app: &App, name: &str) -> Option<(Arc<mcp::Client>, String)> {
    let registry = app.plugins.read().unwrap();
    for (id, state) in registry.iter() {
        let prefix = format!("mcp_{id}_");
        if let Some(tool) = name.strip_prefix(&prefix) {
            return state
                .client
                .as_ref()
                .map(|client| (client.clone(), tool.to_string()));
        }
    }
    None
}

pub(crate) struct ExecContext<'a> {
    pub(crate) data: Option<&'a std::path::Path>,
    pub(crate) session_id: Option<&'a str>,
    pub(crate) sandbox_mode: sandbox::SandboxMode,
    pub(crate) writable_roots: &'a [std::path::PathBuf],
    pub(crate) allow_loopback_ipc: bool,
}

impl<'a> Default for ExecContext<'a> {
    fn default() -> Self {
        Self {
            data: None,
            session_id: None,
            sandbox_mode: sandbox::SandboxMode::Unsandboxed,
            writable_roots: &[],
            allow_loopback_ipc: false,
        }
    }
}

pub(crate) async fn execute_command(cwd: &std::path::Path, args: &Value) -> Result<Value> {
    execute_command_ctx(cwd, args, &ExecContext::default()).await
}

pub(crate) async fn execute_command_ctx(
    cwd: &std::path::Path,
    args: &Value,
    ctx: &ExecContext<'_>,
) -> Result<Value> {
    let command = args["command"].as_str().context("command is required")?;
    if command.trim().is_empty() {
        bail!("command must not be empty");
    }
    let timeout_secs = args["timeout"].as_u64().unwrap_or(30).clamp(1, 120);
    #[cfg(unix)]
    let (program, prog_args) = ("/bin/sh", vec!["-c".to_string(), command.to_string()]);
    #[cfg(windows)]
    let (program, prog_args) = ("cmd.exe", vec!["/C".to_string(), command.to_string()]);

    let data_fallback = std::path::Path::new("/nonexistent");
    let data_dir = ctx.data.unwrap_or(data_fallback);
    let mut cmd = sandbox::build_command(
        program,
        &prog_args,
        cwd,
        ctx.sandbox_mode,
        ctx.writable_roots,
        data_dir,
        ctx.allow_loopback_ipc,
    )?;
    cmd.kill_on_drop(true);

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    )
    .await
    .context("Command timed out")?
    .context("Failed to execute command")?;

    let stdout_raw = String::from_utf8_lossy(&output.stdout);
    let stderr_raw = String::from_utf8_lossy(&output.stderr);
    let (stdout, stdout_spill_id, stdout_truncated, _) = spill::truncate_or_spill(
        ctx.data,
        ctx.session_id,
        "cmd_stdout",
        &stdout_raw,
        spill::DEFAULT_INLINE_LIMIT,
    );
    let (stderr, stderr_spill_id, stderr_truncated, _) = spill::truncate_or_spill(
        ctx.data,
        ctx.session_id,
        "cmd_stderr",
        &stderr_raw,
        spill::DEFAULT_INLINE_LIMIT,
    );
    let mut result = json!({
        "exit_code": output.status.code(),
        "stdout": stdout,
        "stderr": stderr,
    });
    if stdout_truncated || stderr_truncated {
        result["truncated"] = json!(true);
    }
    if let Some(id) = stdout_spill_id {
        result["stdout_spill_id"] = json!(id);
    }
    if let Some(id) = stderr_spill_id {
        result["stderr_spill_id"] = json!(id);
    }
    Ok(result)
}

pub(crate) fn truncate_output(bytes: &[u8]) -> String {
    const MAX: usize = 131072;
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= MAX {
        text.into_owned()
    } else {
        let cut = spill::floor_char_boundary(&text, MAX);
        format!("{}… [truncated {} bytes]", &text[..cut], text.len() - cut)
    }
}
