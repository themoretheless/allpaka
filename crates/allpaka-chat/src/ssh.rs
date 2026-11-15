//! Remote SSH workspace transport seam: strict host/option validation, `BatchMode=yes`,
//! remote filesystem/command execution when `Root.ssh` is set, and `POST /api/ssh/test`.

use crate::context::{Project, Root};
use crate::ext::resolve_project_root;
use crate::{bad_request, storage, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct SshTarget {
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_hosts_policy: Option<String>,
}

fn valid_ssh_token(s: &str, allow_path_chars: bool) -> bool {
    !s.is_empty()
        && s.len() <= 255
        && !s.starts_with('-')
        && !s.chars().any(|c| {
            c.is_control()
                || c.is_whitespace()
                || matches!(
                    c,
                    '\'' | '"' | ';' | '&' | '|' | '$' | '`' | '>' | '<' | '(' | ')' | '{' | '}'
                )
                || (!allow_path_chars && c == '/')
        })
}

pub fn validate_target(target: &SshTarget) -> Result<()> {
    let host = target.host.trim();
    if !valid_ssh_token(host, false) {
        bail!("Invalid SSH host `{}`", target.host);
    }
    if let Some(user) = target
        .user
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if !valid_ssh_token(user, false) || user.contains('@') {
            bail!("Invalid SSH user `{user}`");
        }
    }
    if target.port == Some(0) {
        bail!("SSH port must be 1-65535");
    }
    if let Some(id) = target
        .identity_file
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if !valid_ssh_token(id, true) || id.contains("..") {
            bail!("Invalid SSH identity_file `{id}`");
        }
    }
    if let Some(policy) = target.known_hosts_policy.as_deref() {
        if !matches!(policy, "yes" | "accept-new" | "ask") {
            bail!("SSH known_hosts_policy must be `yes`, `accept-new`, or `ask`");
        }
    }
    Ok(())
}

pub fn validate_remote_root(path: &Path) -> Result<()> {
    let s = path
        .to_str()
        .context("Remote SSH root path must be valid UTF-8")?;
    if !s.starts_with('/') || s.len() > 1024 {
        bail!("Remote SSH root path must be an absolute POSIX path");
    }
    if s.chars().any(|c| c.is_control()) || s.split('/').any(|seg| seg == "..") {
        bail!("Remote SSH root path contains disallowed traversal or control characters");
    }
    Ok(())
}

fn join_remote_posix(root: &Path, rel: &str) -> Result<String> {
    validate_remote_root(root)?;
    let trimmed = rel.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(root.to_string_lossy().into_owned());
    }
    if trimmed.starts_with('/') || trimmed.chars().any(|c| c.is_control()) {
        bail!("Remote relative path cannot be absolute or contain control chars");
    }
    let mut parts: Vec<&str> = Vec::new();
    for seg in trimmed.split('/') {
        match seg {
            "" | "." => {}
            ".." => bail!("Path traversal `..` is forbidden on SSH workspaces"),
            other => {
                if other.starts_with(".git") {
                    bail!("Access to `.git` is forbidden");
                }
                parts.push(other);
            }
        }
    }
    let base = root.to_string_lossy();
    let base_clean = base.trim_end_matches('/');
    if parts.is_empty() {
        Ok(if base_clean.is_empty() {
            "/".into()
        } else {
            base_clean.to_owned()
        })
    } else {
        Ok(format!("{base_clean}/{}", parts.join("/")))
    }
}

fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

pub(crate) fn build_ssh_args(target: &SshTarget, remote_script: &str) -> Result<Vec<String>> {
    validate_target(target)?;
    let policy = target.known_hosts_policy.as_deref().unwrap_or("accept-new");
    let mut args = vec![
        "-T".into(),
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=10".into(),
        "-o".into(),
        format!("StrictHostKeyChecking={policy}"),
    ];
    if let Some(port) = target.port {
        args.extend(["-p".into(), port.to_string()]);
    }
    if let Some(id) = target
        .identity_file
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        args.extend(["-i".into(), id.to_owned()]);
    }
    let dest = match target
        .user
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(u) => format!("{u}@{}", target.host.trim()),
        None => target.host.trim().to_owned(),
    };
    args.push("--".into());
    args.push(dest);
    args.push(remote_script.to_owned());
    Ok(args)
}

async fn run_ssh_with_bin(
    bin: &str,
    target: &SshTarget,
    remote_script: &str,
    stdin_bytes: Option<&[u8]>,
    timeout_secs: u64,
) -> Result<(i32, String, String)> {
    let argv = build_ssh_args(target, remote_script)?;
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(&argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stdin_bytes.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to launch SSH binary `{bin}`"))?;
    if let Some(bytes) = stdin_bytes {
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(bytes).await?;
            stdin.shutdown().await?;
        }
    }
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs.clamp(1, 120)),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("SSH command timed out"))??;
    let code = out.status.code().unwrap_or(-1);
    Ok((
        code,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn default_ssh_bin() -> String {
    std::env::var("ALLPAKA_SSH_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "ssh".into())
}

pub(crate) async fn test_target_with_bin(
    bin: &str,
    target: &SshTarget,
    root_path: &Path,
) -> Result<Value> {
    validate_target(target)?;
    validate_remote_root(root_path)?;
    let quoted = shell_quote(&root_path.to_string_lossy());
    let script = format!("test -d {quoted} && echo ALLPAKA_SSH_OK");
    let start = storage::now_ms();
    let (code, stdout, stderr) = run_ssh_with_bin(bin, target, &script, None, 15).await?;
    let latency = storage::now_ms().saturating_sub(start);
    if code != 0 || !stdout.contains("ALLPAKA_SSH_OK") {
        bail!("SSH check failed (exit {code}): {}", stderr.trim());
    }
    Ok(json!({
        "ok": true,
        "host": target.host,
        "path": root_path,
        "latency_ms": latency,
        "output": stdout.trim(),
    }))
}

pub(crate) async fn execute_remote_tool(
    bin: &str,
    root: &Root,
    target: &SshTarget,
    rel_path: &str,
    name: &str,
    args: &Value,
    allow_writes: bool,
) -> Result<Value> {
    let full = join_remote_posix(&root.path, rel_path)?;
    let quoted = shell_quote(&full);
    match name {
        "read_file" => {
            let start_line = args
                .get("start_line")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            let end_line = args.get("end_line").and_then(Value::as_u64);
            let script = match end_line {
                Some(end) => format!("sed -n '{start_line},{end}p' -- {quoted}"),
                None => format!("head -c 262144 -- {quoted}"),
            };
            let (code, stdout, stderr) = run_ssh_with_bin(bin, target, &script, None, 20).await?;
            if code != 0 {
                bail!("Remote read_file failed: {}", stderr.trim());
            }
            Ok(json!({ "path": rel_path, "content": stdout, "remote": true }))
        }
        "write_file" => {
            if !allow_writes || !root.writable {
                bail!("This SSH context root is read-only");
            }
            let content = args["content"].as_str().context("content is required")?;
            let dir = Path::new(&full)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| "/".into());
            let script = format!("mkdir -p -- {} && cat > {quoted}", shell_quote(&dir));
            let (code, _, stderr) =
                run_ssh_with_bin(bin, target, &script, Some(content.as_bytes()), 20).await?;
            if code != 0 {
                bail!("Remote write_file failed: {}", stderr.trim());
            }
            Ok(json!({ "path": rel_path, "written": content.len(), "remote": true }))
        }
        "edit_file" => {
            if !allow_writes || !root.writable {
                bail!("This SSH context root is read-only");
            }
            let old_text = args["old_text"].as_str().context("old_text is required")?;
            let new_text = args["new_text"].as_str().context("new_text is required")?;
            let read_script = format!("cat -- {quoted}");
            let (code, before, stderr) =
                run_ssh_with_bin(bin, target, &read_script, None, 20).await?;
            if code != 0 {
                bail!("Remote edit_file read failed: {}", stderr.trim());
            }
            let count = before.matches(old_text).count();
            if count == 0 {
                bail!("old_text not found in remote file `{rel_path}`");
            }
            if count > 1
                && !args
                    .get("replace_all")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            {
                bail!("old_text matched {count} times in remote file `{rel_path}`");
            }
            let after = if args
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                before.replace(old_text, new_text)
            } else {
                before.replacen(old_text, new_text, 1)
            };
            let write_script = format!("cat > {quoted}");
            let (wcode, _, wstderr) =
                run_ssh_with_bin(bin, target, &write_script, Some(after.as_bytes()), 20).await?;
            if wcode != 0 {
                bail!("Remote edit_file write failed: {}", wstderr.trim());
            }
            Ok(json!({
                "path": rel_path,
                "edited": true,
                "remote": true,
                "diff": crate::tools::unified_diff(rel_path, &before, &after),
            }))
        }
        "list_files" => {
            let script = format!("find {quoted} -maxdepth 3 -not -path '*/.*' | head -n 500");
            let (code, stdout, stderr) = run_ssh_with_bin(bin, target, &script, None, 20).await?;
            if code != 0 {
                bail!("Remote list_files failed: {}", stderr.trim());
            }
            let entries: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
            Ok(json!({ "path": rel_path, "entries": entries, "remote": true }))
        }
        "agentgrep" => {
            let query = args["query"].as_str().context("query is required")?;
            let script = format!(
                "grep -rnF --exclude-dir=.git -- {} {quoted} | head -n 200",
                shell_quote(query)
            );
            let (code, stdout, stderr) = run_ssh_with_bin(bin, target, &script, None, 20).await?;
            if code > 1 {
                bail!("Remote agentgrep failed: {}", stderr.trim());
            }
            let lines: Vec<&str> = stdout.lines().collect();
            Ok(json!({ "matches": lines, "remote": true }))
        }
        other => bail!("Tool `{other}` is not supported over remote SSH root"),
    }
}

pub(crate) async fn execute_remote_command(
    project: &Project,
    args: &Value,
) -> Result<Option<Value>> {
    let Some(root) = project.roots.first() else {
        return Ok(None);
    };
    let Some(target) = &root.ssh else {
        return Ok(None);
    };
    let command = args["command"].as_str().context("command is required")?;
    let timeout = args.get("timeout").and_then(Value::as_u64).unwrap_or(30);
    let quoted_root = shell_quote(&root.path.to_string_lossy());
    let script = format!("cd {quoted_root} && {command}");
    let bin = default_ssh_bin();
    let (code, stdout, stderr) = run_ssh_with_bin(&bin, target, &script, None, timeout).await?;
    Ok(Some(json!({
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "remote": true,
        "host": target.host,
    })))
}

pub(crate) async fn try_remote_fs(
    project: &Project,
    name: &str,
    args: &Value,
    allow_writes: bool,
) -> Result<Option<Value>> {
    if !matches!(
        name,
        "read_file" | "write_file" | "edit_file" | "list_files" | "agentgrep"
    ) {
        return Ok(None);
    }
    let path_arg = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let (root, rel) = resolve_project_root(project, path_arg, crate::tools::writes(name))?;
    let Some(target) = &root.ssh else {
        return Ok(None);
    };
    let bin = default_ssh_bin();
    let res = execute_remote_tool(&bin, root, target, &rel, name, args, allow_writes).await?;
    Ok(Some(res))
}

#[derive(Deserialize)]
pub(crate) struct SshTestRequest {
    ssh: SshTarget,
    #[serde(default)]
    path: Option<String>,
}

pub(crate) async fn test_api(
    axum::extract::State(_app): axum::extract::State<App>,
    Json(req): Json<SshTestRequest>,
) -> ApiResult<Value> {
    let path = req
        .path
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("/");
    let bin = default_ssh_bin();
    test_target_with_bin(&bin, &req.ssh, Path::new(path))
        .await
        .map(Json)
        .map_err(bad_request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ssh_option_injection_and_traversal() {
        assert!(validate_target(&SshTarget {
            host: "-oProxyCommand=id".into(),
            ..Default::default()
        })
        .is_err());
        assert!(validate_target(&SshTarget {
            host: "example.com".into(),
            user: Some("-bad".into()),
            ..Default::default()
        })
        .is_err());
        assert!(validate_target(&SshTarget {
            host: "example.com".into(),
            known_hosts_policy: Some("no".into()),
            ..Default::default()
        })
        .is_err());
        assert!(validate_remote_root(Path::new("relative/path")).is_err());
        assert!(validate_remote_root(Path::new("/srv/../etc")).is_err());
        assert!(join_remote_posix(Path::new("/srv/repo"), "../etc/passwd").is_err());
    }

    #[tokio::test]
    async fn ssh_executes_remote_tools_against_mock_ssh_binary() {
        let base = std::env::temp_dir().join(format!("allpaka-ssh-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let mock_ssh = base.join("mock_ssh.py");
        std::fs::write(
            &mock_ssh,
            r#"#!/usr/bin/env python3
import sys, subprocess
args = sys.argv[1:]
if "--" not in args:
    sys.exit(2)
idx = args.index("--")
script = args[idx + 2]
res = subprocess.run(["/bin/sh", "-c", script], stdin=sys.stdin, capture_output=False)
sys.exit(res.returncode)
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&mock_ssh).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&mock_ssh, perms).unwrap();
        }

        let remote_root = base.join("remote_ws");
        std::fs::create_dir_all(&remote_root).unwrap();
        let target = SshTarget {
            host: "devbox.internal".into(),
            user: Some("alice".into()),
            port: Some(2222),
            identity_file: None,
            known_hosts_policy: Some("yes".into()),
        };
        let root = Root {
            alias: "remote".into(),
            path: remote_root.clone(),
            writable: true,
            repository: false,
            ssh: Some(target.clone()),
        };
        let bin = mock_ssh.to_string_lossy().into_owned();
        let checked = test_target_with_bin(&bin, &target, &remote_root)
            .await
            .unwrap();
        assert_eq!(checked["ok"], true);

        execute_remote_tool(
            &bin,
            &root,
            &target,
            "hello.txt",
            "write_file",
            &json!({ "content": "one\ntwo\n" }),
            true,
        )
        .await
        .unwrap();
        let read = execute_remote_tool(
            &bin,
            &root,
            &target,
            "hello.txt",
            "read_file",
            &json!({}),
            false,
        )
        .await
        .unwrap();
        assert_eq!(read["content"], "one\ntwo\n");

        execute_remote_tool(
            &bin,
            &root,
            &target,
            "hello.txt",
            "edit_file",
            &json!({ "old_text": "two", "new_text": "TWO" }),
            true,
        )
        .await
        .unwrap();
        let read2 = execute_remote_tool(
            &bin,
            &root,
            &target,
            "hello.txt",
            "read_file",
            &json!({}),
            false,
        )
        .await
        .unwrap();
        assert_eq!(read2["content"], "one\nTWO\n");

        let _ = std::fs::remove_dir_all(base);
    }
}
