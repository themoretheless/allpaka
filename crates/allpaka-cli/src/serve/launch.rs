//! Disk catalog and a detached relaunch of `allpaka serve` on a chosen GGUF.
//!
//! The CUDA weight windows stay for the life of the process, so a different
//! model is a new process, not a second attach.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static BIND: OnceLock<String> = OnceLock::new();

pub fn remember_bind(bind: &str) {
    let _ = BIND.set(bind.to_string());
}

pub fn list_models() -> Vec<Value> {
    let mut found = Vec::new();
    for root in catalog_roots() {
        walk(&root, &mut found, 0);
    }
    found.sort_by(|a, b| a["path"].as_str().unwrap_or("").cmp(b["path"].as_str().unwrap_or("")));
    found.dedup_by(|a, b| a["path"] == b["path"]);
    found.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
    });
    found
}

pub fn schedule(path: &str, bind: &str) -> Result<()> {
    let model = check_model_path(path)?;
    let bind = if bind.is_empty() {
        BIND.get().map(String::as_str).unwrap_or("127.0.0.1:8099")
    } else {
        bind
    };
    if bind.chars().any(|c| c.is_whitespace() || "&|<>".contains(c)) {
        bail!("refusing bind address {bind}");
    }
    let exe = std::env::current_exe().context("locating allpaka.exe")?;
    let log = std::env::temp_dir().join("allpaka-serve.log");
    let bat = std::env::temp_dir().join("allpaka-relaunch.cmd");
    let cuda = std::env::var("CUDA_PATH").unwrap_or_default();
    let cuda_line = if cuda.is_empty() {
        String::new()
    } else {
        format!("set CUDA_PATH={cuda}\r\nset PATH=%CUDA_PATH%\\bin\\x64;%CUDA_PATH%\\bin;%PATH%\r\n")
    };
    let script = format!(
        "@echo off\r\n{cuda_line}ping -n 4 127.0.0.1 >nul\r\necho ===== relaunch %DATE% %TIME% =====>> \"{}\"\r\n\"{}\" serve --bind {bind} --model \"{}\" >> \"{}\" 2>&1\r\n",
        log.display(),
        exe.display(),
        model.display(),
        log.display(),
    );
    std::fs::write(&bat, script).with_context(|| format!("writing {}", bat.display()))?;
    spawn_detached(&bat)?;
    Ok(())
}

fn check_model_path(path: &str) -> Result<PathBuf> {
    if path.is_empty() || path.chars().any(|c| c == '"' || c == '\n' || c == '\r' || "&|<>".contains(c)) {
        bail!("model path is empty or not safe to relaunch");
    }
    let model = PathBuf::from(path);
    if model.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        bail!("model must be a .gguf file");
    }
    if !model.is_file() {
        bail!("model not found: {}", model.display());
    }
    Ok(model)
}

fn catalog_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        roots.push(
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }
    roots.push(PathBuf::from("models"));
    roots
}

fn walk(dir: &Path, out: &mut Vec<Value>, depth: usize) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "blobs" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk(&path, out, depth + 1);
            continue;
        }
        if !name.ends_with(".gguf") {
            continue;
        }
        let bytes = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        let stem = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.into_owned());
        out.push(json!({
            "name": stem,
            "path": path.display().to_string(),
            "bytes": bytes,
        }));
    }
}

fn spawn_detached(bat: &Path) -> Result<()> {
    let mut cmd = std::process::Command::new("cmd");
    cmd.arg("/c").arg(bat);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x00000008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    cmd.spawn()
        .with_context(|| format!("starting {}", bat.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_model_path;

    #[test]
    fn launch_rejects_a_path_that_is_not_a_gguf_file() {
        assert!(check_model_path("").is_err());
        assert!(check_model_path("model.bin").is_err());
        assert!(check_model_path("a&b.gguf").is_err());
    }
}
