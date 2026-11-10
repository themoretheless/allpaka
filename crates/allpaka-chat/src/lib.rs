mod api;
mod background;
mod bookmarks;
mod cancel;
mod code_search;
mod compact;
mod context;
mod conversation_search;
mod credentials;
mod data_lock;
mod evaluation;
mod experiments;
mod feedback;
mod files;
mod guardrail_policies;
mod harness;
mod harness_library;
mod harness_prompts;
mod harness_refine;
mod history;
mod judge;
mod judge_presets;
mod judge_runs;
mod mcp;
mod memory;
mod memory_extract;
mod model_evaluators;
mod observability;
mod online_evaluation;
mod online_sources;
mod plan;
mod plugins;
mod progress;
mod prompts;
mod provider;
mod quality_jobs;
mod rag;
mod remote_worker;
mod review_queues;
mod runtime;
mod state_file;
pub mod storage;
mod strict_json;
mod swarm;
mod tools;
mod types;

use anyhow::{bail, Context, Result};
use api::*;
use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use runtime::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{sync::mpsc, task::JoinHandle};
use types::*;

type SharedSession = Arc<Mutex<Session>>;
type SessionRegistry = Arc<Mutex<HashMap<String, (SharedSession, mpsc::Sender<Action>)>>>;
#[derive(Clone)]
pub(crate) struct App {
    pub(crate) sessions: SessionRegistry,
    pub(crate) background: background::Manager,
    pub(crate) observability: observability::Store,
    pub(crate) evaluation: evaluation::Store,
    pub(crate) memory: memory::Store,
    pub(crate) judge_runs: judge_runs::Manager,
    pub(crate) judges: Arc<tokio::sync::Semaphore>,
    pub(crate) memory_extractions: Arc<tokio::sync::Semaphore>,
    pub(crate) prompts: prompts::Store,
    pub(crate) experiments: experiments::Manager,
    pub(crate) matrix_jobs: experiments::matrix_jobs::Manager,
    pub(crate) feedback: feedback::Store,
    pub(crate) review_queues: review_queues::Store,
    pub(crate) providers: Arc<Mutex<Vec<provider::Provider>>>,
    pub(crate) projects: Arc<Mutex<Vec<context::Project>>>,
    pub(crate) client: reqwest::Client,
    pub(crate) workspace: Arc<PathBuf>,
    pub(crate) data: Arc<PathBuf>,
    pub(crate) origin: String,
    pub(crate) key_mutations: Arc<tokio::sync::Mutex<()>>,
    pub(crate) plugins: plugins::Registry,
    pub(crate) harness_jobs: harness_refine::Manager,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommandExecution {
    Disabled,
    TrustedUnsandboxed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExecutionPolicy {
    pub(crate) commands: CommandExecution,
    pub(crate) external_tools: bool,
}
impl ExecutionPolicy {
    pub(crate) const fn files_only() -> Self {
        Self {
            commands: CommandExecution::Disabled,
            external_tools: false,
        }
    }
    pub(crate) const fn trusted_unsandboxed() -> Self {
        Self {
            commands: CommandExecution::TrustedUnsandboxed,
            external_tools: true,
        }
    }
}
/// What a headless session runs with instead of the registry: its own
/// project (a probe works on copied roots), an explicit harness and an extra
/// system prompt for a subagent. `nested` turns cannot delegate again.
#[derive(Debug)]
pub(crate) struct Ephemeral {
    pub(crate) project: context::Project,
    pub(crate) harness: Option<harness::Tree>,
    pub(crate) system: Option<String>,
    pub(crate) nested: bool,
    /// Applies to tool schemas and direct calls, and is inherited by subagents.
    pub(crate) policy: ExecutionPolicy,
}
#[derive(Deserialize)]
struct Action {
    #[serde(skip)]
    wake_epoch: Option<u64>,
    #[serde(skip)]
    reply: Option<tokio::sync::oneshot::Sender<std::result::Result<(), String>>>,
    #[serde(default)]
    images: Vec<context::Image>,
    kind: ActionKind,
    #[serde(default)]
    text: String,
    settings: Option<Settings>,
}
/// The only kinds `POST /api/sessions/:id/actions` accepts. Serde rejects
/// anything else, and the actor must name every variant it does not handle,
/// so a new kind cannot be added to one list and forgotten in another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ActionKind {
    #[serde(skip)]
    BackgroundDone,
    Send,
    SendNow,
    Steer,
    Stop,
    Resume,
    ClearQueue,
    Compact,
    /// Re-run one member of the last Swarm turn and rebuild the synthesis. The
    /// member is named by `text`; nothing about the roster is required to match.
    RetryMember,
    Rename,
    Move,
}
/// A handler failure: an HTTP status and the message sent as `{"error": message}`.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

pub(crate) type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

pub(crate) fn error(status: StatusCode, message: impl ToString) -> ApiError {
    ApiError {
        status,
        message: message.to_string(),
    }
}
pub(crate) fn bad_request(message: impl ToString) -> ApiError {
    error(StatusCode::BAD_REQUEST, message)
}
pub(crate) fn not_found(message: impl ToString) -> ApiError {
    error(StatusCode::NOT_FOUND, message)
}
pub(crate) fn internal(message: impl ToString) -> ApiError {
    error(StatusCode::INTERNAL_SERVER_ERROR, message)
}
/// Execute a synchronous filesystem/store handler body on Tokio's blocking pool
/// so Axum worker threads are never stalled by directory scans or `sync_all`.
pub(crate) async fn blocking<T, F>(f: F) -> ApiResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> ApiResult<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f).await.map_err(internal)?
}
pub fn run(bind: SocketAddr, workspace: PathBuf, data_dir: Option<PathBuf>) -> Result<()> {
    if !bind.ip().is_loopback() {
        bail!("Studio is a local single-user service: bind to a loopback address");
    }
    let workspace = workspace
        .canonicalize()
        .context("Workspace does not exist")?;
    if !workspace.is_dir() {
        bail!("Workspace must be a directory");
    }
    let _workspace_lock = data_lock::DataLock::acquire_workspace(&workspace)?;
    let data = data_dir.unwrap_or_else(|| {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        home.join(".allpaka").join("studio")
    });
    std::fs::create_dir_all(&data)?;
    let data = data.canonicalize()?;
    let _data_lock = data_lock::DataLock::acquire(&data)?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(bind).await?;
            let addr = listener.local_addr()?;
            let projects_file = data.join("projects.state");
            let projects: Vec<context::Project> = if projects_file.exists() {
                state_file::load_list(&data, "projects", "project", |_| Ok(()))?
            } else {
                vec![context::Project {
                    id: "default".into(),
                    name: "allpaka".into(),
                    roots: vec![context::Root {
                        alias: "workspace".into(),
                        path: workspace.clone(),
                        writable: false,
                        repository: workspace.join(".git").exists(),
                    }],
                    instructions: String::new(),
                }]
            };
            let mut providers=provider::defaults();
            let claude_workspace = provider::load_workspace_id(&data)?;
            for p in &mut providers {
                if p.anthropic && p.workspace_id.is_empty() { p.workspace_id = claude_workspace.clone(); }
            }
            for c in provider::load_custom_providers(&data)? {
                providers.push(provider::custom_to_provider(&c));
            }
            let store=credentials::Store::new(&data);
            for p in &mut providers {
                match store.saved() {
                    Ok(ids)=>p.saved=ids.contains(&p.id),
                    Err(_)=>{p.key_error=Some("Не удалось прочитать индекс сохранённых ключей".into());continue;}
                }
                if p.key.is_empty() && p.saved {
                    match store.load(&credentials::SystemVault,p.id.as_str()) {
                        Ok(Some(key))=>{p.key=key;p.key_source="system".into();},
                        Ok(None)=>p.key_error=Some("Сохранённый ключ не найден в системном хранилище".into()),
                        Err(_)=>p.key_error=Some("Системное хранилище ключей заблокировано или недоступно".into()),
                    }
                }
            }
            let http = provider::client()?;
            let mut plugin_configs = plugins::load(&data)?;
            if plugin_configs.is_empty() {
                plugin_configs = plugins::defaults();
            }
            if let Ok(url) = std::env::var("RAG_MCP_URL") {
                let url = url.trim().to_string();
                if !url.is_empty() && !plugin_configs.iter().any(|p| p.id == "rag") {
                    plugin_configs.push(plugins::PluginConfig {
                        id: "rag".into(),
                        name: "rag-mcp".into(),
                        kind: plugins::PluginKind::Mcp,
                        url,
                        enabled: true,
                        instructions: None,
                        unlocks_tool: None,
                    });
                }
            }
            let plugin_registry = plugins::Registry::default();
            for config in &plugin_configs {
                plugin_registry.write().unwrap().insert(
                    config.id.clone(),
                    plugins::PluginState {
                        config: config.clone(),
                        client: None,
                        error: None,
                    },
                );
            }
            if !plugin_configs.is_empty() {
                let _ = plugins::save(&data, &plugin_configs);
            }
            let app = App {
                projects: Arc::new(Mutex::new(projects)),
                sessions: Default::default(),
                background: background::Manager::open(&data)?,
                observability: observability::Store::new(&data)?,
                evaluation: evaluation::Store::new(&data)?,
                memory: memory::Store::new(&data)?,
                judge_runs: judge_runs::Manager::new(&data)?,
                judges: Arc::new(tokio::sync::Semaphore::new(2)),
                memory_extractions: Arc::new(tokio::sync::Semaphore::new(2)),
                prompts: prompts::Store::new(&data)?,
                experiments: experiments::Manager::new(&data)?,
                matrix_jobs: experiments::matrix_jobs::Manager::new(&data)?,
                feedback: feedback::Store::new(&data)?,
                review_queues: review_queues::Store::new(&data)?,
                providers: Arc::new(Mutex::new(providers)),
                key_mutations:Default::default(),
                client: http,
                workspace: Arc::new(workspace),
                data: Arc::new(data),
                origin: format!("http://{addr}"),
                plugins: plugin_registry,
                harness_jobs: Default::default(),
            };
            app.judge_runs.recover()?;
            let recovered_traces = app.observability.recover()?;
            let recovered_runs = app.experiments.recover()?;
            app.matrix_jobs.recover()?;
            quality_jobs::recover(&app.data)?;
            harness_refine::recover(&app.data)?;
            let _quality_worker=quality_jobs::spawn(app.clone());
            let _online_worker = (std::env::var("ALLPAKA_ONLINE_WORKER").as_deref()!=Ok("0")).then(||online_evaluation::spawn_worker(app.data.clone()));
            if recovered_traces > 0 || recovered_runs > 0 {
                eprintln!("Studio recovered {recovered_traces} interrupted traces and {recovered_runs} interrupted evaluation runs");
            }
            for config in &plugin_configs {
                if config.enabled {
                    plugins::spawn_plugin_connect(app.clone(), config.id.clone());
                }
            }
            rag::spawn_rag_maintenance(app.clone());
            for entry in std::fs::read_dir(app.data.as_ref())? {
                let path = entry?.path();
                if path.extension().is_some_and(|e| e == "json") {
                    let data = std::fs::read(&path)?;
                    let mut session: Session =
                        serde_json::from_slice(&data).with_context(|| {
                            format!("Invalid conversation file: {}", path.display())
                        })?;
                    if !valid_id(&session.id) {
                        bail!("Invalid saved session ID");
                    }
                    let old_plan_revision=session.plan_revision;
                    plan::restore(&mut session)?;
                    if session.plan_revision!=old_plan_revision {save(&app,&session)?;}
                    if session.status == SessionStatus::Running {
                        session.status = SessionStatus::Paused;
                        session.error = Some(
                            "Server restarted. Partial output retained; resume explicitly.".into(),
                        );
                        session.close_pending_tools();
                    }
                    if session.error.as_deref().is_some_and(|e|e == "Provider stopped: length" || e == "Provider stopped: max_tokens") {
                    session.error=None;session.status=SessionStatus::Paused;
                    session.notice=Some("Предыдущий ответ достиг лимита токенов. Увеличьте лимит ответа и нажмите «Продолжить».".into());
                    if let Some(m)=session.messages.last_mut().filter(|m|m.role=="assistant") {m.truncated=true;}
                }
                attach(&app, session);
                }
            }
            let router = routes(app.clone());
            eprintln!(
                "allpaka studio: {}\nWorkspace: {}\nHistory: {}",
                app.origin,
                app.workspace.display(),
                app.data.display()
            );
            axum::serve(listener, router).await?;
            Ok(())
        })
}
