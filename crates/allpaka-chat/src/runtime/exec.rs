//! Shell command execution for the command tool, output truncation, and
//! MCP tool lookup.

use crate::*;

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
pub(crate) async fn execute_command(cwd: &std::path::Path, args: &Value) -> Result<Value> {
    let command = args["command"].as_str().context("command is required")?;
    if command.trim().is_empty() {
        bail!("command must not be empty");
    }
    let timeout_secs = args["timeout"].as_u64().unwrap_or(30).clamp(1, 120);
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokio::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .output(),
    )
    .await
    .context("Command timed out")?
    .context("Failed to execute command")?;
    let stdout = truncate_output(&output.stdout);
    let stderr = truncate_output(&output.stderr);
    Ok(json!({
        "exit_code": output.status.code(),
        "stdout": stdout,
        "stderr": stderr,
    }))
}
pub(crate) fn truncate_output(bytes: &[u8]) -> String {
    const MAX: usize = 131072;
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= MAX {
        text.into_owned()
    } else {
        format!("{}… [truncated {} bytes]", &text[..MAX], text.len() - MAX)
    }
}
