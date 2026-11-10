//! One place that turns a POSIX shell snippet into a process.
//!
//! Tool commands, hooks, transforms and external executors are written for a
//! POSIX shell. Unix runs them with `/bin/sh -c`. Windows has no `/bin/sh`,
//! so it uses `sh` from `PATH` (Git for Windows, MSYS2) and falls back to
//! `cmd.exe /C` only when no POSIX shell is installed.

use std::sync::OnceLock;

fn program() -> &'static (&'static str, &'static str) {
    static PROGRAM: OnceLock<(&'static str, &'static str)> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        if cfg!(unix) {
            ("/bin/sh", "-c")
        } else if posix_sh_on_path() {
            ("sh", "-c")
        } else {
            ("cmd.exe", "/C")
        }
    })
}

fn posix_sh_on_path() -> bool {
    std::process::Command::new("sh")
        .args(["-c", "exit 0"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// `sh -c <script>` as a tokio command, ready for cwd, env and stdio.
pub(crate) fn command(script: &str) -> tokio::process::Command {
    let (shell, flag) = program();
    let mut c = tokio::process::Command::new(shell);
    c.arg(flag).arg(script);
    c
}
