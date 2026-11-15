//! Grouped runtime extension managers and shared session/project helpers for harness waves.

use crate::context::{Project, Root};
use crate::lsp::LspManager;
use crate::patch::CheckpointManager;
use crate::runtime::AgentTreeManager;
use crate::terminal::TerminalManager;
use crate::types::{Mode, SwarmConfig, Verbosity};
use crate::{
    attach, bad_request, not_found, save, storage, ApiError, App, Session, Settings, SharedSession,
};
use anyhow::{bail, Context, Result};
use std::io::Read as _;
use std::sync::atomic::{AtomicU64, Ordering};

static TOKEN_SEQ: AtomicU64 = AtomicU64::new(1);

/// Generate a cryptographically random hex string of `2 * byte_len` hex characters.
pub(crate) fn random_hex_token(byte_len: usize) -> String {
    let mut buf = vec![0u8; byte_len];
    let filled = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok();
    if !filled {
        let seed = format!(
            "{}:{}:{}",
            storage::now_ms(),
            std::process::id(),
            TOKEN_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let h = storage::hex_sha256(seed.as_bytes());
        return h.chars().cycle().take(byte_len * 2).collect();
    }
    let mut hex = String::with_capacity(byte_len * 2);
    for b in buf {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// Consolidated extension state mounted at `App.ext`.
#[derive(Clone, Default)]
pub(crate) struct RuntimeExt {
    pub(crate) checkpoints: CheckpointManager,
    pub(crate) terminals: TerminalManager,
    pub(crate) lsp: LspManager,
    pub(crate) agents: AgentTreeManager,
}

/// Resolve an active or persisted session by ID (rejecting malformed IDs).
pub(crate) fn get_session(app: &App, id: &str) -> std::result::Result<SharedSession, ApiError> {
    if !storage::valid_id(id) {
        return Err(bad_request("Invalid session ID"));
    }
    if let Some((shared, _)) = app.sessions.lock().unwrap().get(id) {
        return Ok(shared.clone());
    }
    let path = app.data.join(format!("{id}.json"));
    if path.is_file() {
        let bytes = storage::read_bounded_bytes(&path, 16 * 1024 * 1024)
            .map_err(|_| not_found("Conversation not found"))?;
        let session: Session =
            serde_json::from_slice(&bytes).map_err(|_| not_found("Conversation not found"))?;
        attach(app, session);
        if let Some((shared, _)) = app.sessions.lock().unwrap().get(id) {
            return Ok(shared.clone());
        }
    }
    Err(not_found("Conversation not found"))
}

/// Snapshot `(session_id, settings, project)` for a validated `SharedSession`.
pub(crate) fn project_for_session(
    app: &App,
    shared: &SharedSession,
) -> std::result::Result<(String, Settings, Project), ApiError> {
    let (id, settings) = {
        let s = shared.lock().unwrap();
        (s.id.clone(), s.settings.clone())
    };
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == settings.project_id)
        .cloned()
        .ok_or_else(|| bad_request("Project not found"))?;
    Ok((id, settings, project))
}

/// Resolve `path` against `project`: tries explicit `alias/...` prefix first, and falls back to
/// the sole root for single-root projects while still enforcing `write => root.writable`.
pub(crate) fn resolve_project_root<'a>(
    project: &'a Project,
    path: &str,
    write: bool,
) -> Result<(&'a Root, String)> {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "." {
        let root = project.roots.first().context("No project root available")?;
        if write && !root.writable {
            bail!("This context folder is read-only");
        }
        return Ok((root, ".".into()));
    }
    let (head, rest) = trimmed.split_once('/').unwrap_or((trimmed, "."));
    if let Some(root) = project.roots.iter().find(|r| r.alias == head) {
        if write && !root.writable {
            bail!("This context folder is read-only");
        }
        return Ok((root, rest.into()));
    }
    if project.roots.len() == 1 {
        let root = &project.roots[0];
        if write && !root.writable {
            bail!("This context folder is read-only");
        }
        return Ok((root, trimmed.into()));
    }
    bail!("Use context-alias/relative-path; list context roots with path '.'")
}

pub(crate) fn default_settings(app: &App, project_id: &str) -> Settings {
    let provider = {
        let ps = app.providers.lock().unwrap();
        ps.iter()
            .find(|p| !p.key.is_empty())
            .or_else(|| ps.first())
            .map(|p| p.id.clone())
            .unwrap_or_else(|| "openai".into())
    };
    let model = match provider.as_str() {
        "deepseek" => "deepseek-chat",
        "anthropic" => "claude-sonnet-4-5",
        _ => "gpt-4o-mini",
    }
    .to_owned();
    Settings {
        prepared_guardrails: None,
        guardrails: None,
        verbosity: Verbosity::Normal,
        project_id: project_id.to_owned(),
        provider,
        model,
        mode: Mode::Auto,
        max_steps: 12,
        max_output_tokens: 8192,
        allow_writes: false,
        swarm: SwarmConfig::default(),
        auto_compact: true,
        compact_threshold: 24_000,
        json_mode: false,
        reasoning_effort: None,
        sandbox_mode: crate::sandbox::SandboxMode::default(),
        approval_policy: crate::sandbox::ApprovalPolicy::default(),
        permission_preset: None,
        team_mode: false,
        budget: crate::types::Budget::default(),
    }
}

pub(crate) fn spawn_scheduled_session(
    app: &App,
    title: String,
    settings: Settings,
    prompt: String,
) -> Result<String> {
    let id = crate::evaluation::new_id();
    let mut session = Session::new(id.clone(), settings.clone());
    session.title = title;
    session.scheduled_origin = true;
    save(app, &session)?;
    attach(app, session);
    let tx = app
        .sessions
        .lock()
        .unwrap()
        .get(&id)
        .map(|(_, tx)| tx.clone())
        .context("Failed to locate attached scheduled session actor")?;
    tx.try_send(crate::Action {
        wake_epoch: None,
        reply: None,
        images: Vec::new(),
        kind: crate::ActionKind::Send,
        text: prompt,
        settings: Some(settings),
        citations: Vec::new(),
        context_budget: None,
    })
    .map_err(|_| anyhow::anyhow!("Failed to enqueue prompt on scheduled session"))?;
    Ok(id)
}

#[cfg(test)]
pub(crate) fn test_app(
    data: &std::path::Path,
    workspace: &std::path::Path,
    projects: Vec<Project>,
) -> App {
    std::fs::create_dir_all(data).unwrap();
    std::fs::create_dir_all(workspace).unwrap();
    let data = data.canonicalize().unwrap();
    let workspace = workspace.canonicalize().unwrap();
    App {
        projects: std::sync::Arc::new(std::sync::Mutex::new(projects)),
        sessions: Default::default(),
        background: Default::default(),
        observability: crate::observability::Store::new(&data).unwrap(),
        feedback: crate::feedback::Store::new(&data).unwrap(),
        review_queues: crate::review_queues::Store::new(&data).unwrap(),
        evaluation: crate::evaluation::Store::new(&data).unwrap(),
        memory: crate::memory::Store::new(&data).unwrap(),
        judge_runs: crate::judge_runs::Manager::new(&data).unwrap(),
        judges: std::sync::Arc::new(tokio::sync::Semaphore::new(2)),
        memory_extractions: std::sync::Arc::new(tokio::sync::Semaphore::new(2)),
        memory_extraction_cancels: Default::default(),
        prompts: crate::prompts::Store::new(&data).unwrap(),
        experiments: crate::experiments::Manager::new(&data).unwrap(),
        matrix_jobs: crate::experiments::matrix_jobs::Manager::new(&data).unwrap(),
        providers: Default::default(),
        key_mutations: Default::default(),
        client: crate::provider::client().unwrap(),
        workspace: std::sync::Arc::new(workspace),
        data: std::sync::Arc::new(data),
        origin: "http://127.0.0.1:0".into(),
        plugins: Default::default(),
        harness_jobs: Default::default(),
        interactions: Default::default(),
        ext: Default::default(),
    }
}
