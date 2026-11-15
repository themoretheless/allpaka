//! OS-level process sandboxing and permission presets for command execution.
//!
//! Uses `sandbox-exec` (Seatbelt SBPL profiles) on macOS and `bwrap` (bubblewrap)
//! on Linux. Fail-closed semantics: requesting a sandboxed mode (`ReadOnly` or
//! `WorkspaceWrite`) on a host without an available backend returns an error
//! rather than silently running unsandboxed.

use anyhow::{bail, Result};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxMode {
    #[default]
    Unsandboxed,
    ReadOnly,
    WorkspaceWrite,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    #[default]
    Never,
    Commands,
    WritesAndCommands,
    Always,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PermissionPreset {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub allow_writes: bool,
    pub sandbox_mode: SandboxMode,
    pub approval_policy: ApprovalPolicy,
}

const PRESETS: &[PermissionPreset] = &[
    PermissionPreset {
        id: "readonly_audit",
        label: "Read-only audit",
        description: "Workspace read-only; commands run inside a read-only offline sandbox with command approval.",
        allow_writes: false,
        sandbox_mode: SandboxMode::ReadOnly,
        approval_policy: ApprovalPolicy::Commands,
    },
    PermissionPreset {
        id: "workspace_sandboxed",
        label: "Sandboxed workspace",
        description: "Writes and commands allowed inside writable project roots; OS sandbox denies external writes and Studio data access.",
        allow_writes: true,
        sandbox_mode: SandboxMode::WorkspaceWrite,
        approval_policy: ApprovalPolicy::Never,
    },
    PermissionPreset {
        id: "interactive_guarded",
        label: "Interactive guarded",
        description: "OS-sandboxed workspace with explicit user confirmation before edits and commands.",
        allow_writes: true,
        sandbox_mode: SandboxMode::WorkspaceWrite,
        approval_policy: ApprovalPolicy::WritesAndCommands,
    },
    PermissionPreset {
        id: "full_trusted",
        label: "Trusted unsandboxed",
        description: "Direct host execution without OS sandbox or approval prompts.",
        allow_writes: true,
        sandbox_mode: SandboxMode::Unsandboxed,
        approval_policy: ApprovalPolicy::Never,
    },
];

pub fn presets() -> &'static [PermissionPreset] {
    PRESETS
}

pub fn resolve_preset(id: &str) -> Option<&'static PermissionPreset> {
    PRESETS.iter().find(|preset| preset.id == id)
}

pub(crate) async fn presets_api() -> Json<Value> {
    Json(json!({
        "presets": PRESETS,
        "capabilities": capabilities(),
    }))
}

#[cfg(target_os = "linux")]
fn find_in_path(binary: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn capabilities() -> Value {
    #[cfg(target_os = "macos")]
    {
        let bin = Path::new("/usr/bin/sandbox-exec");
        let available = bin.is_file();
        json!({
            "os": "macos",
            "backend": if available { "sandbox-exec" } else { "none" },
            "available": available,
        })
    }
    #[cfg(target_os = "linux")]
    {
        let available = find_in_path("bwrap").is_some();
        json!({
            "os": "linux",
            "backend": if available { "bwrap" } else { "none" },
            "available": available,
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        json!({
            "os": std::env::consts::OS,
            "backend": "none",
            "available": false,
        })
    }
}

fn sbpl_escape_path(path: &Path) -> Result<String> {
    let raw = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Non-UTF-8 path cannot be used inside sandbox profile"))?;
    if raw.chars().any(|c| c == '"' || c == '\\' || c.is_control()) {
        bail!("Unsafe characters in path for sandbox profile");
    }
    Ok(raw.to_owned())
}

fn expand_path_variants(path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let raw = path.to_path_buf();
    out.push(raw.clone());
    if let Ok(canon) = path.canonicalize() {
        if canon != raw {
            out.push(canon);
        }
    }
    out
}

pub(crate) fn build_sbpl_profile(
    mode: SandboxMode,
    writable_roots: &[PathBuf],
    data_dir: &Path,
    allow_loopback_ipc: bool,
) -> Result<String> {
    if mode == SandboxMode::Unsandboxed {
        return Ok("(version 1)\n(allow default)\n".into());
    }
    let mut lines = vec![
        "(version 1)".to_string(),
        "(allow default)".to_string(),
        "(deny file-write*)".to_string(),
        "(allow file-write* (literal \"/dev/null\") (literal \"/dev/tty\") (literal \"/dev/ptmx\") (regex #\"^/dev/tty.*\") (regex #\"^/dev/ptys.*\"))".to_string(),
    ];
    if mode == SandboxMode::ReadOnly {
        lines.push("(deny network-outbound)".to_string());
    } else {
        for root in writable_roots {
            for variant in expand_path_variants(root) {
                let escaped = sbpl_escape_path(&variant)?;
                lines.push(format!("(allow file-write* (subpath \"{escaped}\"))"));
            }
        }
    }
    if allow_loopback_ipc {
        lines.push("(allow network-outbound (remote ip \"localhost:*\"))".to_string());
    }
    // Always protect Studio data directory last so more general rules cannot expose it.
    for variant in expand_path_variants(data_dir) {
        let escaped = sbpl_escape_path(&variant)?;
        lines.push(format!(
            "(deny file-read* file-write* (subpath \"{escaped}\"))"
        ));
    }
    Ok(lines.join("\n"))
}

pub(crate) fn build_command(
    program: &str,
    args: &[String],
    cwd: &Path,
    mode: SandboxMode,
    writable_roots: &[PathBuf],
    data_dir: &Path,
    allow_loopback_ipc: bool,
) -> Result<tokio::process::Command> {
    if mode == SandboxMode::Unsandboxed {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args).current_dir(cwd);
        return Ok(cmd);
    }

    #[cfg(target_os = "macos")]
    {
        let bin = Path::new("/usr/bin/sandbox-exec");
        if !bin.is_file() {
            bail!("Sandbox mode {mode:?} requested, but /usr/bin/sandbox-exec is unavailable");
        }
        let profile = build_sbpl_profile(mode, writable_roots, data_dir, allow_loopback_ipc)?;
        let mut cmd = tokio::process::Command::new(bin);
        cmd.arg("-p")
            .arg(profile)
            .arg(program)
            .args(args)
            .current_dir(cwd);
        Ok(cmd)
    }

    #[cfg(target_os = "linux")]
    {
        let bwrap = find_in_path("bwrap").ok_or_else(|| {
            anyhow::anyhow!("Sandbox mode {mode:?} requested, but `bwrap` is not installed on PATH")
        })?;
        let mut cmd = tokio::process::Command::new(bwrap);
        cmd.args([
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--die-with-parent",
            "--unshare-pid",
        ]);
        if mode == SandboxMode::ReadOnly {
            if !allow_loopback_ipc {
                cmd.arg("--unshare-net");
            }
        } else {
            for root in writable_roots {
                if let Ok(canon) = root.canonicalize() {
                    if let Some(s) = canon.to_str() {
                        cmd.args(["--bind", s, s]);
                    }
                }
            }
        }
        if let Ok(canon_data) = data_dir.canonicalize() {
            if let Some(s) = canon_data.to_str() {
                cmd.args(["--tmpfs", s]);
            }
        }
        cmd.arg("--").arg(program).args(args).current_dir(cwd);
        return Ok(cmd);
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (
            program,
            args,
            cwd,
            writable_roots,
            data_dir,
            allow_loopback_ipc,
        );
        bail!(
            "Sandbox mode {mode:?} is not supported on OS `{}` (fail-closed)",
            std::env::consts::OS
        );
    }
}

/// Convert `build_command` parameters into a wrapped `(program, args)` pair so callers such
/// as interactive PTY spawning can invoke the sandboxed command directly via `Command` or `execvp`.
pub(crate) fn sandbox_argv(
    program: &str,
    args: &[String],
    mode: SandboxMode,
    writable_roots: &[PathBuf],
    data_dir: &Path,
    allow_loopback_ipc: bool,
) -> Result<(String, Vec<String>)> {
    if mode == SandboxMode::Unsandboxed {
        return Ok((program.to_owned(), args.to_vec()));
    }
    #[cfg(target_os = "macos")]
    {
        if !Path::new("/usr/bin/sandbox-exec").is_file() {
            bail!("Sandbox mode {mode:?} requested, but /usr/bin/sandbox-exec is unavailable");
        }
        let profile = build_sbpl_profile(mode, writable_roots, data_dir, allow_loopback_ipc)?;
        let mut wrapped = vec!["-p".into(), profile, program.to_owned()];
        wrapped.extend(args.iter().cloned());
        Ok(("/usr/bin/sandbox-exec".into(), wrapped))
    }
    #[cfg(target_os = "linux")]
    {
        let bwrap = find_in_path("bwrap").ok_or_else(|| {
            anyhow::anyhow!("Sandbox mode {mode:?} requested, but `bwrap` is not installed on PATH")
        })?;
        let mut wrapped = vec![
            "--ro-bind".into(),
            "/".into(),
            "/".into(),
            "--dev".into(),
            "/dev".into(),
            "--proc".into(),
            "/proc".into(),
            "--die-with-parent".into(),
            "--unshare-pid".into(),
        ];
        if mode == SandboxMode::ReadOnly && !allow_loopback_ipc {
            wrapped.push("--unshare-net".into());
        } else if mode == SandboxMode::WorkspaceWrite {
            for root in writable_roots {
                if let Ok(canon) = root.canonicalize() {
                    if let Some(s) = canon.to_str() {
                        wrapped.extend(["--bind".into(), s.to_owned(), s.to_owned()]);
                    }
                }
            }
        }
        if let Ok(canon_data) = data_dir.canonicalize() {
            if let Some(s) = canon_data.to_str() {
                wrapped.extend(["--tmpfs".into(), s.to_owned()]);
            }
        }
        wrapped.push("--".into());
        wrapped.push(program.to_owned());
        wrapped.extend(args.iter().cloned());
        return Ok((bwrap.to_string_lossy().into_owned(), wrapped));
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (program, args, writable_roots, data_dir, allow_loopback_ipc);
        bail!("Sandbox mode {mode:?} is not supported on this OS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbpl_profile_protects_data_dir_and_scopes_writes() {
        let profile = build_sbpl_profile(
            SandboxMode::WorkspaceWrite,
            &[PathBuf::from("/tmp/work")],
            Path::new("/tmp/studio-data"),
            false,
        )
        .unwrap();
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("(allow file-write* (subpath \"/tmp/work\"))"));
        assert!(profile.contains("(deny file-read* file-write* (subpath \"/tmp/studio-data\"))"));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn macos_sandbox_exec_enforces_write_bounds_and_hides_data_dir() {
        let base =
            std::env::temp_dir().join(format!("allpaka-sbx-{}", crate::evaluation::new_id()));
        let work = base.join("writable");
        let ro = base.join("readonly");
        let data = base.join("data");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&ro).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("secret.txt"), "top-secret").unwrap();
        let work_c = work.canonicalize().unwrap();
        let ro_c = ro.canonicalize().unwrap();
        let data_c = data.canonicalize().unwrap();

        // WorkspaceWrite allows writing to `work`, forbids `ro`, and forbids reading `data`.
        let cmd_ok = build_command(
            "/bin/sh",
            &[
                "-c".into(),
                format!(
                    "echo ok > '{}/a.txt' && cat '{}/a.txt'",
                    work_c.display(),
                    work_c.display()
                ),
            ],
            &work_c,
            SandboxMode::WorkspaceWrite,
            std::slice::from_ref(&work_c),
            &data_c,
            false,
        )
        .unwrap()
        .output()
        .await
        .unwrap();
        assert!(
            cmd_ok.status.success(),
            "writable root write failed: {}",
            String::from_utf8_lossy(&cmd_ok.stderr)
        );
        assert_eq!(
            std::fs::read_to_string(work_c.join("a.txt"))
                .unwrap()
                .trim(),
            "ok"
        );

        let cmd_ro_write = build_command(
            "/bin/sh",
            &["-c".into(), format!("touch '{}/pwned'", ro_c.display())],
            &work_c,
            SandboxMode::WorkspaceWrite,
            std::slice::from_ref(&work_c),
            &data_c,
            false,
        )
        .unwrap()
        .output()
        .await
        .unwrap();
        assert!(!cmd_ro_write.status.success());
        assert!(!ro_c.join("pwned").exists());

        let cmd_secret_read = build_command(
            "/bin/sh",
            &[
                "-c".into(),
                format!("cat '{}/secret.txt'", data_c.display()),
            ],
            &work_c,
            SandboxMode::WorkspaceWrite,
            std::slice::from_ref(&work_c),
            &data_c,
            false,
        )
        .unwrap()
        .output()
        .await
        .unwrap();
        assert!(!cmd_secret_read.status.success());
        assert!(!String::from_utf8_lossy(&cmd_secret_read.stdout).contains("top-secret"));

        // ReadOnly forbids writing even inside `work`.
        let cmd_readonly = build_command(
            "/bin/sh",
            &["-c".into(), format!("touch '{}/blocked'", work_c.display())],
            &work_c,
            SandboxMode::ReadOnly,
            std::slice::from_ref(&work_c),
            &data_c,
            false,
        )
        .unwrap()
        .output()
        .await
        .unwrap();
        assert!(!cmd_readonly.status.success());
        assert!(!work_c.join("blocked").exists());

        let _ = std::fs::remove_dir_all(&base);
    }
}
