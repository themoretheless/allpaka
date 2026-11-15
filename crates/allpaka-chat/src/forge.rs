//! GitHub (`gh`) / GitLab (`glab`) Pull/Merge Request automation and persisted `Session.forge_pr` state.

use crate::context::Project;
use crate::ext::{get_session, project_for_session};
use crate::storage;
use crate::{bad_request, conflict, ApiResult, App, SharedSession};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ForgePrState {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub branch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_branch: Option<String>,
    pub title: String,
    pub state: String,
    pub updated_at_ms: u64,
}

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "forge_pr",
            "description": "Inspect, create, comment on, or check CI status for GitHub (`gh`) or GitLab (`glab`) pull/merge requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["status", "create", "comment", "checks"] },
                    "title": { "type": "string" },
                    "body": { "type": "string" },
                    "base": { "type": "string" },
                    "draft": { "type": "boolean" },
                    "pr_number": { "type": "integer" }
                },
                "required": ["action"]
            }
        }
    })
}

fn valid_branch_token(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('-')
        && !s.contains("..")
        && !s.chars().any(|c| c.is_control() || c.is_whitespace())
}

fn detect_forge(repo: &Path) -> (String, Option<String>) {
    let url = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let provider = match url.as_deref() {
        Some(u) if u.contains("gitlab") => "gitlab",
        _ => "github",
    };
    (provider.to_owned(), url)
}

fn current_branch(repo: &Path) -> Result<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("git rev-parse failed")?;
    if !out.status.success() {
        bail!(
            "Failed to read git branch: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn forge_bin(provider: &str) -> String {
    if let Ok(b) = std::env::var("ALLPAKA_FORGE_BIN") {
        if !b.trim().is_empty() {
            return b;
        }
    }
    match provider {
        "gitlab" => std::env::var("ALLPAKA_GLAB_BIN").unwrap_or_else(|_| "glab".into()),
        _ => std::env::var("ALLPAKA_GH_BIN").unwrap_or_else(|_| "gh".into()),
    }
}

async fn run_forge(bin: &str, cwd: &Path, args: &[String]) -> Result<Value> {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(args).current_dir(cwd);
    let out = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(err)) => {
            return Ok(json!({
                "available": false,
                "binary": bin,
                "message": err.to_string(),
            }));
        }
        Err(_) => bail!("Forge command timed out after 30s"),
    };
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        bail!(
            "`{bin}` exited with status {}: {}",
            out.status,
            stderr.trim()
        );
    }
    let parsed: Value =
        serde_json::from_str(stdout.trim()).unwrap_or_else(|_| json!({ "raw": stdout.trim() }));
    Ok(json!({
        "available": true,
        "binary": bin,
        "output": parsed,
        "raw": stdout.trim(),
    }))
}

pub(crate) async fn execute(
    app: &App,
    shared: &SharedSession,
    project: &Project,
    allow_writes: bool,
    args: &Value,
) -> Result<Value> {
    let action = args["action"].as_str().context("action is required")?;
    let root = project.roots.first().context("No project root available")?;
    let (provider, remote_url) = detect_forge(&root.path);
    let branch = current_branch(&root.path).unwrap_or_else(|_| "HEAD".into());
    let bin = forge_bin(&provider);

    match action {
        "status" => {
            let stored = shared.lock().unwrap().forge_pr.clone();
            let cli_args = if provider == "gitlab" {
                vec!["mr".into(), "view".into(), "--output".into(), "json".into()]
            } else {
                vec![
                    "pr".into(),
                    "view".into(),
                    "--json".into(),
                    "number,title,url,state,baseRefName,isDraft".into(),
                ]
            };
            let live = run_forge(&bin, &root.path, &cli_args).await.ok();
            Ok(json!({
                "provider": provider,
                "remote_url": remote_url,
                "branch": branch,
                "stored": stored,
                "live": live,
            }))
        }
        "checks" => {
            let cli_args = if provider == "gitlab" {
                vec!["ci".into(), "status".into()]
            } else {
                vec!["pr".into(), "checks".into()]
            };
            run_forge(&bin, &root.path, &cli_args).await
        }
        "create" => {
            if !allow_writes {
                bail!("Creating PR/MR requires a write-enabled session");
            }
            let title = args["title"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .context("title is required")?;
            let body = args.get("body").and_then(Value::as_str).unwrap_or("");
            let base = args
                .get("base")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty());
            if let Some(b) = base {
                if !valid_branch_token(b) {
                    bail!("Invalid base branch `{b}`");
                }
            }
            let draft = args.get("draft").and_then(Value::as_bool).unwrap_or(false);
            let mut cli_args = if provider == "gitlab" {
                let mut v = vec![
                    "mr".into(),
                    "create".into(),
                    "--title".into(),
                    title.to_owned(),
                    "--description".into(),
                    body.to_owned(),
                ];
                if let Some(b) = base {
                    v.extend(["--target-branch".into(), b.to_owned()]);
                }
                if draft {
                    v.push("--draft".into());
                }
                v
            } else {
                let mut v = vec![
                    "pr".into(),
                    "create".into(),
                    "--title".into(),
                    title.to_owned(),
                    "--body".into(),
                    body.to_owned(),
                ];
                if let Some(b) = base {
                    v.extend(["--base".into(), b.to_owned()]);
                }
                if draft {
                    v.push("--draft".into());
                }
                v
            };
            let _ = &mut cli_args;
            let res = run_forge(&bin, &root.path, &cli_args).await?;
            let pr_url = res["output"]
                .get("url")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    res["raw"]
                        .as_str()
                        .filter(|s| s.starts_with("http"))
                        .map(|s| s.lines().next().unwrap_or(s).to_owned())
                });
            let pr_num = res["output"].get("number").and_then(Value::as_u64);
            let pr_state = ForgePrState {
                provider,
                number: pr_num,
                url: pr_url,
                branch,
                base_branch: base.map(ToOwned::to_owned),
                title: title.to_owned(),
                state: if draft { "draft" } else { "open" }.into(),
                updated_at_ms: storage::now_ms(),
            };
            if res.get("available") == Some(&json!(true)) {
                let mut state = shared.lock().unwrap();
                state.forge_pr = Some(pr_state.clone());
                crate::save(app, &state)?;
            }
            Ok(
                json!({ "created": res.get("available") == Some(&json!(true)), "pr": pr_state, "details": res }),
            )
        }
        "comment" => {
            if !allow_writes {
                bail!("Commenting on a PR/MR requires a write-enabled session");
            }
            let body = args["body"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .context("body is required")?;
            let number = args
                .get("pr_number")
                .and_then(Value::as_u64)
                .or_else(|| {
                    shared
                        .lock()
                        .unwrap()
                        .forge_pr
                        .as_ref()
                        .and_then(|p| p.number)
                })
                .map(|n| n.to_string());
            let mut cli_args = if provider == "gitlab" {
                vec!["mr".into(), "note".into()]
            } else {
                vec!["pr".into(), "comment".into()]
            };
            if let Some(n) = number {
                cli_args.push(n);
            }
            cli_args.extend([
                if provider == "gitlab" {
                    "--message".into()
                } else {
                    "--body".into()
                },
                body.to_owned(),
            ]);
            run_forge(&bin, &root.path, &cli_args).await
        }
        other => bail!("Unsupported forge_pr action `{other}`"),
    }
}

pub(crate) async fn get_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, _, project) = project_for_session(&app, &shared)?;
    execute(
        &app,
        &shared,
        &project,
        false,
        &json!({ "action": "status" }),
    )
    .await
    .map(Json)
    .map_err(bad_request)
}

pub(crate) async fn post_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(args): Json<Value>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, settings, project) = project_for_session(&app, &shared)?;
    if settings.mode.is_read_only() {
        return Err(conflict("Forge actions require non-read-only mode"));
    }
    execute(&app, &shared, &project, settings.allow_writes, &args)
        .await
        .map(Json)
        .map_err(bad_request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_branch_tokens_against_flag_injection() {
        assert!(valid_branch_token("main"));
        assert!(valid_branch_token("feature/pr-1"));
        assert!(!valid_branch_token("--repo=evil"));
        assert!(!valid_branch_token("a..b"));
    }
}
