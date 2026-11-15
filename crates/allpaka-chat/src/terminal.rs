//! Persistent PTY terminal manager with strict Model vs User ownership isolation,
//! sandboxed child invocation via `sandbox::sandbox_argv`, and a 256 KiB ring buffer.

use crate::ext::{get_session, project_for_session};
use crate::sandbox::{self, SandboxMode};
use crate::storage;
use crate::{bad_request, conflict, not_found, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

const RING_CAPACITY: usize = 256 * 1024;
const MAX_TERMINALS_PER_SESSION: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TerminalOwner {
    Model,
    User,
}

struct OutputRing {
    bytes: VecDeque<u8>,
    total_written: u64,
    alive: bool,
    exit_code: Option<i32>,
}

impl OutputRing {
    fn new() -> Self {
        Self {
            bytes: VecDeque::with_capacity(4096),
            total_written: 0,
            alive: true,
            exit_code: None,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        for &b in chunk {
            if self.bytes.len() == RING_CAPACITY {
                self.bytes.pop_front();
            }
            self.bytes.push_back(b);
            self.total_written += 1;
        }
    }

    fn read_from(&self, offset: u64, max_bytes: usize) -> Value {
        let len = self.bytes.len() as u64;
        let earliest = self.total_written.saturating_sub(len);
        let truncated = offset < earliest;
        let start = offset.max(earliest);
        let avail = self.total_written.saturating_sub(start) as usize;
        let take = avail.min(max_bytes);
        let drop_front = (start - earliest) as usize;
        let slice: Vec<u8> = self
            .bytes
            .iter()
            .skip(drop_front)
            .take(take)
            .copied()
            .collect();
        let next_offset = start + slice.len() as u64;
        let text = String::from_utf8_lossy(&slice).into_owned();
        json!({
            "output": text,
            "offset": start,
            "next_offset": next_offset,
            "total_bytes": self.total_written,
            "truncated_before": truncated,
            "alive": self.alive,
            "exit_code": self.exit_code,
        })
    }
}

struct TerminalInstance {
    id: String,
    session_id: String,
    owner: TerminalOwner,
    title: String,
    cwd: PathBuf,
    cols: Mutex<u16>,
    rows: Mutex<u16>,
    created_at_ms: u64,
    ring: Arc<Mutex<OutputRing>>,
    notify: Arc<Notify>,
    #[cfg(unix)]
    write_file: Mutex<std::fs::File>,
    #[cfg(unix)]
    child_pid: i32,
}

impl TerminalInstance {
    fn summary(&self) -> Value {
        let r = self.ring.lock().unwrap();
        let tail_start = r.total_written.saturating_sub(2048);
        let tail_preview = r.read_from(tail_start, 2048)["output"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        json!({
            "id": self.id,
            "session_id": self.session_id,
            "owner": self.owner,
            "title": self.title,
            "cwd": self.cwd,
            "cols": *self.cols.lock().unwrap(),
            "rows": *self.rows.lock().unwrap(),
            "created_at_ms": self.created_at_ms,
            "alive": r.alive,
            "exit_code": r.exit_code,
            "total_bytes": r.total_written,
            "tail_preview": tail_preview,
        })
    }

    fn write_bytes(&self, data: &[u8]) -> Result<()> {
        if !self.ring.lock().unwrap().alive {
            bail!("Terminal `{}` has already exited", self.id);
        }
        #[cfg(unix)]
        {
            let mut file = self.write_file.lock().unwrap();
            file.write_all(data)?;
            file.flush()?;
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = data;
            bail!("PTY terminals require a Unix host");
        }
    }

    fn resize(&self, cols: u16, rows: u16) -> Result<()> {
        *self.cols.lock().unwrap() = cols;
        *self.rows.lock().unwrap() = rows;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let file = self.write_file.lock().unwrap();
            let ws = libc::winsize {
                ws_row: rows,
                ws_col: cols,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            let rc = unsafe { libc::ioctl(file.as_raw_fd(), libc::TIOCSWINSZ, &ws) };
            if rc != 0 {
                bail!("TIOCSWINSZ failed: {}", std::io::Error::last_os_error());
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            bail!("PTY resize requires a Unix host");
        }
    }

    fn send_signal(&self, sig: i32) -> Result<()> {
        #[cfg(unix)]
        {
            if !self.ring.lock().unwrap().alive {
                return Ok(());
            }
            let rc = unsafe { libc::kill(-self.child_pid, sig) };
            if rc != 0 {
                let _ = unsafe { libc::kill(self.child_pid, sig) };
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = sig;
            bail!("Signals require a Unix host");
        }
    }

    fn kill(&self) {
        let _ = self.send_signal(libc_sigkill());
    }
}

impl Drop for TerminalInstance {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(unix)]
fn libc_sigkill() -> i32 {
    libc::SIGKILL
}
#[cfg(not(unix))]
fn libc_sigkill() -> i32 {
    9
}

#[derive(Clone, Default)]
pub(crate) struct TerminalManager {
    seq: Arc<AtomicU64>,
    terminals: Arc<Mutex<HashMap<String, Arc<TerminalInstance>>>>,
}

pub(crate) fn schema_list() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "terminal_open",
                "description": "Open a persistent interactive PTY terminal owned by the model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "command": { "type": "string", "description": "Optional initial command passed to /bin/sh -lc; omit for interactive shell" },
                        "cwd": { "type": "string" },
                        "cols": { "type": "integer" },
                        "rows": { "type": "integer" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "terminal_send",
                "description": "Send stdin text to a model-owned PTY terminal (never allowed on user-owned terminals).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terminal_id": { "type": "string" },
                        "input": { "type": "string" }
                    },
                    "required": ["terminal_id", "input"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "terminal_read",
                "description": "Read incremental PTY output from a model-owned terminal starting at `offset`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terminal_id": { "type": "string" },
                        "offset": { "type": "integer" },
                        "max_bytes": { "type": "integer" },
                        "wait_ms": { "type": "integer" }
                    },
                    "required": ["terminal_id"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "terminal_signal",
                "description": "Send SIGINT, SIGTERM, or SIGKILL to a model-owned PTY terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terminal_id": { "type": "string" },
                        "signal": { "type": "string", "enum": ["INT", "TERM", "KILL"] }
                    },
                    "required": ["terminal_id", "signal"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "terminal_close",
                "description": "Close and terminate a model-owned PTY terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terminal_id": { "type": "string" }
                    },
                    "required": ["terminal_id"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "terminal_list",
                "description": "List model-owned PTY terminals in the current session.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "user_terminal_list",
                "description": "List user-owned interactive PTY terminals (read-only metadata).",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "user_terminal_read",
                "description": "Read recent output from a user-owned terminal (strictly read-only).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terminal_id": { "type": "string" },
                        "offset": { "type": "integer" },
                        "max_bytes": { "type": "integer" }
                    },
                    "required": ["terminal_id"]
                }
            }
        }),
    ]
}

impl TerminalManager {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        &self,
        session_id: &str,
        owner: TerminalOwner,
        title: Option<&str>,
        command: Option<&str>,
        cwd: &std::path::Path,
        cols: u16,
        rows: u16,
        sandbox_mode: SandboxMode,
        writable_roots: &[PathBuf],
        data_dir: &std::path::Path,
    ) -> Result<Value> {
        {
            let map = self.terminals.lock().unwrap();
            let count = map.values().filter(|t| t.session_id == session_id).count();
            if count >= MAX_TERMINALS_PER_SESSION {
                bail!("Maximum {MAX_TERMINALS_PER_SESSION} terminals active in this session");
            }
        }
        let cols = cols.clamp(20, 400);
        let rows = rows.clamp(5, 200);

        #[cfg(unix)]
        {
            use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
            use std::os::unix::process::CommandExt;

            let mut master_fd: libc::c_int = 0;
            let mut slave_fd: libc::c_int = 0;
            let mut ws = libc::winsize {
                ws_row: rows,
                ws_col: cols,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            let rc = unsafe {
                libc::openpty(
                    &mut master_fd,
                    &mut slave_fd,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut ws,
                )
            };
            if rc != 0 {
                bail!("openpty failed: {}", std::io::Error::last_os_error());
            }
            let master_owned = unsafe { OwnedFd::from_raw_fd(master_fd) };
            let _ =
                unsafe { libc::fcntl(master_owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };
            let _ = unsafe { libc::fcntl(slave_fd, libc::F_SETFD, libc::FD_CLOEXEC) };

            let raw_args: Vec<String> = match command {
                Some(cmd) if !cmd.trim().is_empty() => vec!["-lc".into(), cmd.to_owned()],
                _ => vec!["-i".into()],
            };
            let (prog, prog_args) = sandbox::sandbox_argv(
                "/bin/sh",
                &raw_args,
                sandbox_mode,
                writable_roots,
                data_dir,
                false,
            )?;

            let mut cmd = std::process::Command::new(&prog);
            cmd.args(&prog_args)
                .current_dir(cwd)
                .env("TERM", "xterm-256color")
                .env("PS1", "$ ");
            unsafe {
                cmd.pre_exec(move || {
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::ioctl(slave_fd, libc::TIOCSCTTY as _, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    for target in [0, 1, 2] {
                        if libc::dup2(slave_fd, target) == -1 {
                            return Err(std::io::Error::last_os_error());
                        }
                    }
                    if slave_fd > 2 {
                        libc::close(slave_fd);
                    }
                    Ok(())
                });
            }

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(err) => {
                    unsafe { libc::close(slave_fd) };
                    return Err(err.into());
                }
            };
            unsafe { libc::close(slave_fd) };
            let child_pid = child.id() as i32;

            let write_dup = unsafe { libc::dup(master_owned.as_raw_fd()) };
            if write_dup < 0 {
                let _ = child.kill();
                bail!("dup failed: {}", std::io::Error::last_os_error());
            }
            let _ = unsafe { libc::fcntl(write_dup, libc::F_SETFD, libc::FD_CLOEXEC) };
            let write_file = unsafe { std::fs::File::from_raw_fd(write_dup) };
            let mut read_file = std::fs::File::from(master_owned);

            let ring = Arc::new(Mutex::new(OutputRing::new()));
            let notify = Arc::new(Notify::new());

            let ring_reader = ring.clone();
            let notify_reader = notify.clone();
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    match read_file.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            ring_reader.lock().unwrap().push(&buf[..n]);
                            notify_reader.notify_waiters();
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            });

            let ring_reaper = ring.clone();
            let notify_reaper = notify.clone();
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code());
                let mut r = ring_reaper.lock().unwrap();
                r.alive = false;
                r.exit_code = code;
                notify_reaper.notify_waiters();
            });

            let id = format!(
                "term-{}-{}",
                storage::now_ms(),
                self.seq.fetch_add(1, Ordering::Relaxed) + 1
            );
            let default_title = match owner {
                TerminalOwner::Model => "Model Shell",
                TerminalOwner::User => "User Shell",
            };
            let inst = Arc::new(TerminalInstance {
                id: id.clone(),
                session_id: session_id.to_owned(),
                owner,
                title: title.unwrap_or(default_title).to_owned(),
                cwd: cwd.to_path_buf(),
                cols: Mutex::new(cols),
                rows: Mutex::new(rows),
                created_at_ms: storage::now_ms(),
                ring,
                notify,
                write_file: Mutex::new(write_file),
                child_pid,
            });
            let summary = inst.summary();
            self.terminals.lock().unwrap().insert(id, inst);
            Ok(summary)
        }
        #[cfg(not(unix))]
        {
            let _ = (
                session_id,
                owner,
                title,
                command,
                cwd,
                sandbox_mode,
                writable_roots,
                data_dir,
            );
            bail!("PTY terminals require a Unix host");
        }
    }

    fn get_owned(
        &self,
        session_id: &str,
        tid: &str,
        required_owner: Option<TerminalOwner>,
    ) -> Result<Arc<TerminalInstance>> {
        let inst = self
            .terminals
            .lock()
            .unwrap()
            .get(tid)
            .cloned()
            .with_context(|| format!("Terminal `{tid}` not found"))?;
        if inst.session_id != session_id {
            bail!("Terminal `{tid}` does not belong to this session");
        }
        if let Some(req) = required_owner {
            if inst.owner != req {
                bail!(
                    "Access denied: terminal `{tid}` is owned by {:?}, expected {:?}",
                    inst.owner,
                    req
                );
            }
        }
        Ok(inst)
    }

    pub(crate) async fn read(
        &self,
        session_id: &str,
        tid: &str,
        required_owner: Option<TerminalOwner>,
        offset: u64,
        max_bytes: usize,
        wait_ms: u64,
    ) -> Result<Value> {
        let inst = self.get_owned(session_id, tid, required_owner)?;
        let cap = max_bytes.clamp(1, RING_CAPACITY);
        if wait_ms > 0 {
            let deadline =
                tokio::time::Instant::now() + std::time::Duration::from_millis(wait_ms.min(10_000));
            loop {
                {
                    let r = inst.ring.lock().unwrap();
                    if r.total_written > offset || !r.alive {
                        return Ok(r.read_from(offset, cap));
                    }
                }
                let notified = inst.notify.notified();
                if tokio::time::timeout_at(deadline, notified).await.is_err() {
                    break;
                }
            }
        }
        let out = inst.ring.lock().unwrap().read_from(offset, cap);
        Ok(out)
    }

    pub(crate) fn write(
        &self,
        session_id: &str,
        tid: &str,
        required_owner: Option<TerminalOwner>,
        input: &str,
    ) -> Result<Value> {
        let inst = self.get_owned(session_id, tid, required_owner)?;
        inst.write_bytes(input.as_bytes())?;
        Ok(json!({ "terminal_id": tid, "bytes_written": input.len() }))
    }

    pub(crate) fn signal(
        &self,
        session_id: &str,
        tid: &str,
        required_owner: Option<TerminalOwner>,
        sig_name: &str,
    ) -> Result<Value> {
        let inst = self.get_owned(session_id, tid, required_owner)?;
        #[cfg(unix)]
        let sig = match sig_name {
            "INT" | "SIGINT" => libc::SIGINT,
            "TERM" | "SIGTERM" => libc::SIGTERM,
            "KILL" | "SIGKILL" => libc::SIGKILL,
            other => bail!("Unsupported signal `{other}`"),
        };
        #[cfg(not(unix))]
        let sig = 9;
        inst.send_signal(sig)?;
        Ok(json!({ "terminal_id": tid, "signal": sig_name }))
    }

    pub(crate) fn resize(
        &self,
        session_id: &str,
        tid: &str,
        cols: u16,
        rows: u16,
    ) -> Result<Value> {
        let inst = self.get_owned(session_id, tid, None)?;
        inst.resize(cols.clamp(20, 400), rows.clamp(5, 200))?;
        Ok(json!({ "terminal_id": tid, "cols": cols, "rows": rows }))
    }

    pub(crate) fn close(
        &self,
        session_id: &str,
        tid: &str,
        required_owner: Option<TerminalOwner>,
    ) -> Result<Value> {
        let inst = self.get_owned(session_id, tid, required_owner)?;
        inst.kill();
        self.terminals.lock().unwrap().remove(tid);
        Ok(json!({ "closed": tid }))
    }

    pub(crate) fn list(&self, session_id: &str, owner_filter: Option<TerminalOwner>) -> Vec<Value> {
        let mut items: Vec<_> = self
            .terminals
            .lock()
            .unwrap()
            .values()
            .filter(|t| t.session_id == session_id && owner_filter.is_none_or(|o| t.owner == o))
            .cloned()
            .collect();
        items.sort_by_key(|t| t.created_at_ms);
        items.into_iter().map(|t| t.summary()).collect()
    }

    pub(crate) fn purge_session(&self, session_id: &str) {
        let removed: Vec<_> = {
            let mut map = self.terminals.lock().unwrap();
            let ids: Vec<String> = map
                .values()
                .filter(|t| t.session_id == session_id)
                .map(|t| t.id.clone())
                .collect();
            ids.into_iter().filter_map(|id| map.remove(&id)).collect()
        };
        for inst in removed {
            inst.kill();
        }
    }
}

pub(crate) async fn execute_tool(
    app: &App,
    session_id: &str,
    default_cwd: &std::path::Path,
    sandbox_mode: SandboxMode,
    writable_roots: &[PathBuf],
    name: &str,
    args: &Value,
) -> Result<Value> {
    let mgr = &app.ext.terminals;
    match name {
        "terminal_open" => {
            let cwd_arg = args.get("cwd").and_then(Value::as_str);
            let cwd = if let Some(rel) = cwd_arg {
                let shared =
                    get_session(app, session_id).map_err(|e| anyhow::anyhow!(e.message))?;
                let (_, _, project) =
                    project_for_session(app, &shared).map_err(|e| anyhow::anyhow!(e.message))?;
                let (root, rel_p) = crate::ext::resolve_project_root(&project, rel, false)?;
                crate::tools::resolve(&root.path, &rel_p, false)?
            } else {
                default_cwd.to_path_buf()
            };
            mgr.open(
                session_id,
                TerminalOwner::Model,
                args.get("title").and_then(Value::as_str),
                args.get("command").and_then(Value::as_str),
                &cwd,
                args.get("cols").and_then(Value::as_u64).unwrap_or(100) as u16,
                args.get("rows").and_then(Value::as_u64).unwrap_or(32) as u16,
                sandbox_mode,
                writable_roots,
                app.data.as_ref(),
            )
        }
        "terminal_send" => {
            let tid = args["terminal_id"]
                .as_str()
                .context("terminal_id is required")?;
            let input = args["input"].as_str().context("input is required")?;
            mgr.write(session_id, tid, Some(TerminalOwner::Model), input)
        }
        "terminal_read" => {
            let tid = args["terminal_id"]
                .as_str()
                .context("terminal_id is required")?;
            let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0);
            let max_bytes = args
                .get("max_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(32768) as usize;
            let wait_ms = args.get("wait_ms").and_then(Value::as_u64).unwrap_or(200);
            mgr.read(
                session_id,
                tid,
                Some(TerminalOwner::Model),
                offset,
                max_bytes,
                wait_ms,
            )
            .await
        }
        "terminal_signal" => {
            let tid = args["terminal_id"]
                .as_str()
                .context("terminal_id is required")?;
            let sig = args["signal"].as_str().context("signal is required")?;
            mgr.signal(session_id, tid, Some(TerminalOwner::Model), sig)
        }
        "terminal_close" => {
            let tid = args["terminal_id"]
                .as_str()
                .context("terminal_id is required")?;
            mgr.close(session_id, tid, Some(TerminalOwner::Model))
        }
        "terminal_list" => Ok(json!({
            "terminals": mgr.list(session_id, Some(TerminalOwner::Model)),
        })),
        "user_terminal_list" => Ok(json!({
            "terminals": mgr.list(session_id, Some(TerminalOwner::User)),
        })),
        "user_terminal_read" => {
            let tid = args["terminal_id"]
                .as_str()
                .context("terminal_id is required")?;
            let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0);
            let max_bytes = args
                .get("max_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(32768) as usize;
            mgr.read(
                session_id,
                tid,
                Some(TerminalOwner::User),
                offset,
                max_bytes,
                0,
            )
            .await
        }
        _ => bail!("Unknown terminal tool `{name}`"),
    }
}

#[derive(Deserialize)]
pub(crate) struct CreateTerminalInput {
    #[serde(default)]
    owner: Option<TerminalOwner>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    cols: Option<u16>,
    #[serde(default)]
    rows: Option<u16>,
}

#[derive(Default, Deserialize)]
pub(crate) struct ReadTerminalQuery {
    #[serde(default)]
    offset: Option<u64>,
    #[serde(default)]
    max_bytes: Option<usize>,
    #[serde(default)]
    wait_ms: Option<u64>,
}

#[derive(Deserialize)]
pub(crate) struct InputRequest {
    input: String,
}

#[derive(Deserialize)]
pub(crate) struct ResizeRequest {
    cols: u16,
    rows: u16,
}

pub(crate) async fn list_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    Ok(Json(json!({
        "session_id": id,
        "terminals": app.ext.terminals.list(&id, None),
    })))
}

pub(crate) async fn create_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<CreateTerminalInput>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, settings, project) = project_for_session(&app, &shared)?;
    if settings.mode.is_read_only() {
        return Err(conflict("Terminals cannot be spawned in read-only mode"));
    }
    let cwd = project
        .roots
        .first()
        .map(|r| r.path.clone())
        .ok_or_else(|| bad_request("No project root available"))?;
    let writable: Vec<PathBuf> = if settings.allow_writes {
        project
            .roots
            .iter()
            .filter(|r| r.writable)
            .map(|r| r.path.clone())
            .collect()
    } else {
        Vec::new()
    };
    let created = app
        .ext
        .terminals
        .open(
            &id,
            req.owner.unwrap_or(TerminalOwner::User),
            req.title.as_deref(),
            req.command.as_deref(),
            &cwd,
            req.cols.unwrap_or(100),
            req.rows.unwrap_or(32),
            settings.sandbox_mode,
            &writable,
            app.data.as_ref(),
        )
        .map_err(bad_request)?;
    Ok(Json(created))
}

pub(crate) async fn read_api(
    State(app): State<App>,
    AxumPath((id, tid)): AxumPath<(String, String)>,
    Query(q): Query<ReadTerminalQuery>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let page = app
        .ext
        .terminals
        .read(
            &id,
            &tid,
            None,
            q.offset.unwrap_or(0),
            q.max_bytes.unwrap_or(32768),
            q.wait_ms.unwrap_or(0),
        )
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                not_found(e)
            } else {
                bad_request(e)
            }
        })?;
    Ok(Json(page))
}

pub(crate) async fn input_api(
    State(app): State<App>,
    AxumPath((id, tid)): AxumPath<(String, String)>,
    Json(req): Json<InputRequest>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let out = app
        .ext
        .terminals
        .write(&id, &tid, None, &req.input)
        .map_err(bad_request)?;
    Ok(Json(out))
}

pub(crate) async fn resize_api(
    State(app): State<App>,
    AxumPath((id, tid)): AxumPath<(String, String)>,
    Json(req): Json<ResizeRequest>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let out = app
        .ext
        .terminals
        .resize(&id, &tid, req.cols, req.rows)
        .map_err(bad_request)?;
    Ok(Json(out))
}

pub(crate) async fn close_api(
    State(app): State<App>,
    AxumPath((id, tid)): AxumPath<(String, String)>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let out = app.ext.terminals.close(&id, &tid, None).map_err(|e| {
        if e.to_string().contains("not found") {
            not_found(e)
        } else {
            bad_request(e)
        }
    })?;
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pty_executes_and_isolates_model_and_user_terminals() {
        let base = std::env::temp_dir().join(format!("allpaka-pty-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        let work = base.join("work");
        let data = base.join("data");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&data).unwrap();

        let mgr = TerminalManager::default();
        let model_term = mgr
            .open(
                "s1",
                TerminalOwner::Model,
                Some("m"),
                None,
                &work,
                80,
                24,
                SandboxMode::Unsandboxed,
                std::slice::from_ref(&work),
                &data,
            )
            .unwrap();
        let model_id = model_term["id"].as_str().unwrap().to_owned();

        let user_term = mgr
            .open(
                "s1",
                TerminalOwner::User,
                Some("u"),
                None,
                &work,
                80,
                24,
                SandboxMode::Unsandboxed,
                std::slice::from_ref(&work),
                &data,
            )
            .unwrap();
        let user_id = user_term["id"].as_str().unwrap().to_owned();

        // Model cannot write to or close User terminal.
        assert!(mgr
            .write("s1", &user_id, Some(TerminalOwner::Model), "echo BAD\n")
            .is_err());
        assert!(mgr
            .close("s1", &user_id, Some(TerminalOwner::Model))
            .is_err());

        // Model CAN write to Model terminal and read output back incrementally.
        mgr.write(
            "s1",
            &model_id,
            Some(TerminalOwner::Model),
            "echo PTY_OK_42\n",
        )
        .unwrap();
        let mut collected = String::new();
        let mut offset = 0u64;
        for _ in 0..10 {
            let chunk = mgr
                .read(
                    "s1",
                    &model_id,
                    Some(TerminalOwner::Model),
                    offset,
                    16384,
                    300,
                )
                .await
                .unwrap();
            collected.push_str(chunk["output"].as_str().unwrap_or(""));
            offset = chunk["next_offset"].as_u64().unwrap();
            if collected.contains("PTY_OK_42") {
                break;
            }
        }
        assert!(collected.contains("PTY_OK_42"), "got: {collected:?}");

        // Cloned manager handles (as cloned per HTTP request by Axum State) share live terminals.
        let cloned = mgr.clone();
        assert_eq!(cloned.list("s1", None).len(), 2);

        mgr.purge_session("s1");
        assert!(mgr.list("s1", None).is_empty());
        let _ = std::fs::remove_dir_all(base);
    }
}
