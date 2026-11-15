//! Runtime plugin registry.
//!
//! Two kinds of plugins:
//! - **mcp**: MCP server endpoints connected in the background over HTTP.
//! - **skill**: instruction-only plugins that inject text into the system prompt
//!   and (optionally) gate a built-in tool such as `run_command`.
//!
//! Both kinds are stored in `plugins.state`. Adding or removing a plugin does
//! not require a server restart: the next turn rebuilds schemas and prompts.

use crate::mcp;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PluginKind {
    #[default]
    Mcp,
    McpStdio,
    Skill,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub kind: PluginKind,
    #[serde(default)]
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub env: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub instructions: Option<String>,
    /// Built-in tool this skill plugin unlocks (e.g. "run_command" for cmd).
    #[serde(default)]
    pub unlocks_tool: Option<String>,
}

pub struct PluginState {
    pub config: PluginConfig,
    pub client: Option<Arc<mcp::Client>>,
    pub error: Option<String>,
}

pub type Registry = Arc<RwLock<HashMap<String, PluginState>>>;

pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() < 80
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

pub fn validate(config: &PluginConfig) -> Result<()> {
    if !valid_id(&config.id) {
        bail!("Plugin ID must be 1–79 ASCII letters, digits, '-' or '_'");
    }
    if config.name.trim().is_empty()
        || config.name.chars().count() > 100
        || config.name.chars().any(char::is_control)
    {
        bail!("Plugin name must be 1–100 characters");
    }
    match config.kind {
        PluginKind::Mcp => {
            if !(config.url.starts_with("http://") || config.url.starts_with("https://"))
                || config.url.len() > 500
                || config.url.chars().any(char::is_control)
            {
                bail!("MCP plugin URL must start with http:// or https:// and be at most 500 characters");
            }
        }
        PluginKind::McpStdio => {
            let cmd = config.command.as_deref().map(str::trim).unwrap_or("");
            if cmd.is_empty() || cmd.len() > 500 || cmd.chars().any(char::is_control) {
                bail!("MCP stdio plugin requires a non-empty command (<= 500 chars)");
            }
            if config.args.len() > 32
                || config
                    .args
                    .iter()
                    .any(|a| a.len() > 1000 || a.chars().any(char::is_control))
            {
                bail!("MCP stdio args must contain at most 32 safe strings");
            }
            if config.env.len() > 32 {
                bail!("MCP stdio env must contain at most 32 variables");
            }
            for (k, v) in &config.env {
                let upper = k.to_ascii_uppercase();
                if matches!(
                    upper.as_str(),
                    "LD_PRELOAD"
                        | "LD_LIBRARY_PATH"
                        | "DYLD_INSERT_LIBRARIES"
                        | "DYLD_LIBRARY_PATH"
                ) {
                    bail!("Restricted loader environment variable `{k}` is not permitted");
                }
                if k.trim().is_empty()
                    || k.chars().any(|c| c.is_control() || c == '=')
                    || v.chars().any(char::is_control)
                {
                    bail!("Invalid environment variable `{k}` in MCP stdio config");
                }
            }
        }
        PluginKind::Skill => {
            if config
                .instructions
                .as_ref()
                .is_none_or(|s| s.trim().is_empty())
            {
                bail!("Skill plugin must have instructions");
            }
            if config
                .instructions
                .as_ref()
                .is_some_and(|s| s.len() > 16000)
            {
                bail!("Skill instructions must be at most 16 000 characters");
            }
        }
    }
    Ok(())
}

pub fn load(data: &Path) -> Result<Vec<PluginConfig>> {
    crate::state_file::load_list(data, "plugins", "plugin", validate)
}

pub fn save(data: &Path, list: &[PluginConfig]) -> Result<()> {
    crate::state_file::save_list(data, "plugins", list)
}

/// Collect instructions from all enabled skill plugins.
pub fn skill_instructions(registry: &Registry) -> Vec<(String, String)> {
    let reg = registry.read().unwrap();
    reg.values()
        .filter(|s| s.config.enabled && s.config.kind == PluginKind::Skill)
        .filter_map(|s| {
            s.config
                .instructions
                .as_ref()
                .map(|text| (s.config.name.clone(), text.clone()))
        })
        .collect()
}

/// Check whether a built-in tool is unlocked by at least one enabled plugin.
pub fn tool_unlocked(registry: &Registry, tool_name: &str) -> bool {
    let reg = registry.read().unwrap();
    reg.values().any(|s| {
        s.config.enabled
            && s.config.kind == PluginKind::Skill
            && s.config.unlocks_tool.as_deref() == Some(tool_name)
    })
}

/// Default skill plugins seeded on first startup.
pub fn defaults() -> Vec<PluginConfig> {
    vec![
        PluginConfig {
            id: "cmd".into(),
            name: "Shell".into(),
            kind: PluginKind::Skill,
            url: String::new(),
            command: None,
            args: Vec::new(),
            env: std::collections::BTreeMap::new(),
            enabled: true,
            instructions: Some("Shell commands are available via run_command. Use for build, test, lint, and other dev tasks. Timeout 30s by default (max 120s). Output truncated at 128 KiB. Prefer dedicated tools (read_file, write_file, edit_file) for file operations. Commands run in the project root directory.".into()),
            unlocks_tool: Some("run_command".into()),
        },
        PluginConfig {
            id: "git".into(),
            name: "Git".into(),
            kind: PluginKind::Skill,
            url: String::new(),
            command: None,
            args: Vec::new(),
            env: std::collections::BTreeMap::new(),
            enabled: true,
            instructions: Some("Git operations via run_command. Read-only by default: status, diff, log, branch. Write operations (add, commit, push, reset, stash) only on explicit request. Never run reset --hard, clean -fdx, push --force, or delete branches without confirmation. Show what will be affected before destructive commands. Uncommitted changes are someone's work — don't stash or switch branches without asking.".into()),
            unlocks_tool: None,
        },
        PluginConfig {
            id: "gh".into(),
            name: "GitHub".into(),
            kind: PluginKind::Skill,
            url: String::new(),
            command: None,
            args: Vec::new(),
            env: std::collections::BTreeMap::new(),
            enabled: true,
            instructions: Some("GitHub CLI (gh) via run_command. Read-only by default: pr list/view, issue list/view, run list/view, repo view. Write operations (pr create/merge/close, issue create/close, release create, workflow run) only on explicit request. Show what will be published before creating PRs, issues, or releases. Check auth with 'gh auth status' first. Respect rate limits — don't retry on 403/429.".into()),
            unlocks_tool: None,
        },
    ]
}

use crate::error;
use crate::ApiResult;
use crate::App;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
pub(crate) fn spawn_plugin_connect(app: App, id: String) {
    tokio::spawn(async move {
        let cfg = {
            let registry = app.plugins.read().unwrap();
            let Some(state) = registry.get(&id) else {
                return;
            };
            if !state.config.enabled {
                return;
            }
            state.config.clone()
        };
        if cfg.kind == PluginKind::Skill {
            return;
        }
        let connected = match cfg.kind {
            PluginKind::Mcp => mcp::Client::connect(&app.client, &cfg.url).await,
            PluginKind::McpStdio => {
                let cmd = cfg.command.as_deref().unwrap_or("");
                mcp::Client::connect_stdio(cmd, &cfg.args, &cfg.env, &app.workspace).await
            }
            PluginKind::Skill => return,
        };
        match connected {
            Ok(client) => {
                let count = client.tools.len();
                if let Some(state) = app.plugins.write().unwrap().get_mut(&id) {
                    state.client = Some(Arc::new(client));
                    state.error = None;
                }
                eprintln!("plugin {id} connected: {count} tools");
            }
            Err(err) => {
                if let Some(state) = app.plugins.write().unwrap().get_mut(&id) {
                    state.client = None;
                    state.error = Some(format!("{err:#}"));
                }
                eprintln!("plugin {id} unavailable: {err:#}");
            }
        }
    });
}

pub(crate) fn reload_all(app: &App) {
    let ids: Vec<String> = {
        let mut reg = app.plugins.write().unwrap();
        let mut enabled_ids = Vec::new();
        for (id, state) in reg.iter_mut() {
            state.client = None;
            state.error = None;
            if state.config.enabled && state.config.kind != PluginKind::Skill {
                enabled_ids.push(id.clone());
            }
        }
        enabled_ids
    };
    for id in ids {
        spawn_plugin_connect(app.clone(), id);
    }
}

#[derive(serde::Deserialize)]
pub(crate) struct PluginInput {
    #[serde(default)]
    id: String,
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    url: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: std::collections::BTreeMap<String, String>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    unlocks_tool: Option<String>,
}
pub(crate) fn default_true() -> bool {
    true
}
pub(crate) async fn list_plugins(State(app): State<App>) -> Json<Value> {
    let registry = app.plugins.read().unwrap();
    let plugins: Vec<Value> = registry
        .values()
        .map(|state| {
            json!({
                "id": state.config.id,
                "name": state.config.name,
                "kind": state.config.kind,
                "url": state.config.url,
                "command": state.config.command,
                "args": state.config.args,
                "enabled": state.config.enabled,
                "instructions": state.config.instructions,
                "unlocks_tool": state.config.unlocks_tool,
                "connected": state.client.is_some(),
                "tools": state.client.as_ref().map(|c| c.tools.len()).unwrap_or(0),
                "resources": state.client.as_ref().map(|c| c.resources.len()).unwrap_or(0),
                "resource_templates": state.client.as_ref().map(|c| c.resource_templates.len()).unwrap_or(0),
                "error": state.error,
            })
        })
        .collect();
    Json(json!({"plugins": plugins}))
}
pub(crate) async fn save_plugin(
    State(app): State<App>,
    Json(mut input): Json<PluginInput>,
) -> ApiResult<Value> {
    if input.id.trim().is_empty() {
        input.id = format!(
            "plugin-{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
    }
    let kind = match input.kind.as_deref() {
        Some("skill") => PluginKind::Skill,
        Some("mcp_stdio") => PluginKind::McpStdio,
        _ => PluginKind::Mcp,
    };
    let config = PluginConfig {
        id: input.id.trim().to_string(),
        name: input.name.trim().to_string(),
        kind: kind.clone(),
        url: input.url.trim().to_string(),
        command: input.command.map(|c| c.trim().to_string()),
        args: input.args,
        env: input.env,
        enabled: input.enabled,
        instructions: input.instructions,
        unlocks_tool: input.unlocks_tool,
    };
    validate(&config).map_err(crate::bad_request)?;
    {
        let mut registry = app.plugins.write().unwrap();
        let mut configs: Vec<PluginConfig> = registry
            .values()
            .filter(|state| state.config.id != config.id)
            .map(|state| state.config.clone())
            .collect();
        configs.push(config.clone());
        save(app.data.as_path(), &configs).map_err(crate::internal)?;
        match registry.get_mut(&config.id) {
            Some(state) => {
                let changed = state.config.url != config.url
                    || state.config.kind != config.kind
                    || state.config.command != config.command
                    || state.config.args != config.args
                    || state.config.env != config.env
                    || state.config.instructions != config.instructions;
                state.config = config.clone();
                if changed || !config.enabled {
                    state.client = None;
                    state.error = None;
                }
            }
            None => {
                registry.insert(
                    config.id.clone(),
                    PluginState {
                        config: config.clone(),
                        client: None,
                        error: None,
                    },
                );
            }
        }
    }
    if config.enabled {
        spawn_plugin_connect(app.clone(), config.id.clone());
    }
    Ok(Json(json!({"id": config.id})))
}
pub(crate) async fn delete_plugin(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    {
        let mut registry = app.plugins.write().unwrap();
        if !registry.contains_key(&id) {
            return Err(error(StatusCode::NOT_FOUND, "Plugin not found"));
        }
        let configs: Vec<PluginConfig> = registry
            .values()
            .filter(|state| state.config.id != id)
            .map(|state| state.config.clone())
            .collect();
        save(app.data.as_path(), &configs).map_err(crate::internal)?;
        registry.remove(&id);
    }
    Ok(Json(json!({"deleted": id})))
}
pub(crate) async fn reload_plugin(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let enabled = {
        let mut registry = app.plugins.write().unwrap();
        let Some(state) = registry.get_mut(&id) else {
            return Err(error(StatusCode::NOT_FOUND, "Plugin not found"));
        };
        state.client = None;
        state.error = None;
        state.config.enabled
    };
    if enabled {
        spawn_plugin_connect(app.clone(), id.clone());
    }
    Ok(Json(json!({"id": id, "connecting": enabled})))
}

pub(crate) fn manager_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "plugin_manager",
            "description": "Inspect, toggle enablement of, or reload already-configured plugins and MCP servers (`list`, `toggle`, `reload`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "toggle", "reload"] },
                    "plugin_id": { "type": "string" },
                    "enabled": { "type": "boolean" }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        }
    })
}

pub(crate) fn execute_manager(app: &App, args: &Value) -> Result<Value> {
    let action = args["action"].as_str().context("Missing `action`")?;
    match action {
        "list" => {
            let reg = app.plugins.read().unwrap();
            let list: Vec<Value> = reg
                .values()
                .map(|s| {
                    json!({
                        "id": s.config.id,
                        "name": s.config.name,
                        "kind": s.config.kind,
                        "enabled": s.config.enabled,
                        "connected": s.client.is_some(),
                        "tools": s.client.as_ref().map(|c| c.tools.len()).unwrap_or(0),
                        "resources": s.client.as_ref().map(|c| c.resources.len()).unwrap_or(0),
                        "error": s.error,
                    })
                })
                .collect();
            Ok(json!({ "plugins": list }))
        }
        "toggle" => {
            let id = args["plugin_id"]
                .as_str()
                .context("`toggle` requires `plugin_id`")?
                .trim();
            let enable = args["enabled"]
                .as_bool()
                .context("`toggle` requires boolean `enabled`")?;
            let configs = {
                let mut reg = app.plugins.write().unwrap();
                let state = reg
                    .get_mut(id)
                    .with_context(|| format!("Unknown plugin `{id}`"))?;
                state.config.enabled = enable;
                if !enable {
                    state.client = None;
                    state.error = None;
                }
                reg.values().map(|s| s.config.clone()).collect::<Vec<_>>()
            };
            save(app.data.as_path(), &configs)?;
            if enable {
                spawn_plugin_connect(app.clone(), id.to_owned());
            }
            Ok(json!({ "plugin_id": id, "enabled": enable }))
        }
        "reload" => {
            let pid = args["plugin_id"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty());
            if let Some(id) = pid {
                let enabled = {
                    let mut reg = app.plugins.write().unwrap();
                    let state = reg
                        .get_mut(id)
                        .with_context(|| format!("Unknown plugin `{id}`"))?;
                    state.client = None;
                    state.error = None;
                    state.config.enabled
                };
                if enabled {
                    spawn_plugin_connect(app.clone(), id.to_owned());
                }
                Ok(json!({ "plugin_id": id, "reloading": enabled }))
            } else {
                reload_all(app);
                Ok(json!({ "reloading": "all" }))
            }
        }
        other => bail!("Unsupported plugin_manager action `{other}`"),
    }
}
