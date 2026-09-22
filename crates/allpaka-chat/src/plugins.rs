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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PluginKind {
    Mcp,
    Skill,
}

impl Default for PluginKind {
    fn default() -> Self {
        Self::Mcp
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub kind: PluginKind,
    #[serde(default)]
    pub url: String,
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
        PluginKind::Skill => {
            if config.instructions.as_ref().is_none_or(|s| s.trim().is_empty()) {
                bail!("Skill plugin must have instructions");
            }
            if config.instructions.as_ref().is_some_and(|s| s.len() > 16000) {
                bail!("Skill instructions must be at most 16 000 characters");
            }
        }
    }
    Ok(())
}

pub fn load(data: &Path) -> Result<Vec<PluginConfig>> {
    let path = data.join("plugins.state");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() > 65536 {
        bail!("Invalid plugin file");
    }
    let list: Vec<PluginConfig> = serde_json::from_slice(&bytes).context("Invalid plugin file")?;
    for config in &list {
        validate(config)?;
    }
    Ok(list)
}

pub fn save(data: &Path, list: &[PluginConfig]) -> Result<()> {
    let path = data.join("plugins.state");
    let temp = data.join("plugins.tmp");
    std::fs::write(&temp, serde_json::to_vec(list)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
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
            enabled: true,
            instructions: Some("Shell commands are available via run_command. Use for build, test, lint, and other dev tasks. Timeout 30s by default (max 120s). Output truncated at 128 KiB. Prefer dedicated tools (read_file, write_file, edit_file) for file operations. Commands run in the project root directory.".into()),
            unlocks_tool: Some("run_command".into()),
        },
        PluginConfig {
            id: "git".into(),
            name: "Git".into(),
            kind: PluginKind::Skill,
            url: String::new(),
            enabled: true,
            instructions: Some("Git operations via run_command. Read-only by default: status, diff, log, branch. Write operations (add, commit, push, reset, stash) only on explicit request. Never run reset --hard, clean -fdx, push --force, or delete branches without confirmation. Show what will be affected before destructive commands. Uncommitted changes are someone's work — don't stash or switch branches without asking.".into()),
            unlocks_tool: None,
        },
        PluginConfig {
            id: "gh".into(),
            name: "GitHub".into(),
            kind: PluginKind::Skill,
            url: String::new(),
            enabled: true,
            instructions: Some("GitHub CLI (gh) via run_command. Read-only by default: pr list/view, issue list/view, run list/view, repo view. Write operations (pr create/merge/close, issue create/close, release create, workflow run) only on explicit request. Show what will be published before creating PRs, issues, or releases. Check auth with 'gh auth status' first. Respect rate limits — don't retry on 403/429.".into()),
            unlocks_tool: None,
        },
    ]
}
