//! Isolated git worktree management outside `app.data`, with optional Studio project registration.

use crate::context::{Project, Root};
use crate::ext::{get_session, project_for_session};
use crate::{bad_request, conflict, state_file, storage, ApiResult, App, SharedSession};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "create_worktree",
            "description": "Create, list, or remove an isolated git worktree outside Studio data storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["create", "list", "remove"] },
                    "slug": { "type": "string", "description": "Alphanumeric/dash/underscore worktree identifier" },
                    "branch": { "type": "string", "description": "Optional new branch name to create (`git worktree add -b`)" },
                    "base_ref": { "type": "string", "description": "Optional base commit or ref (defaults to HEAD)" },
                    "attach_project": { "type": "boolean", "description": "When true on create, registers a writable Studio project for the worktree" }
                }
            }
        }
    })
}

fn valid_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug.len() <= 64
        && !slug.starts_with('-')
        && slug
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn valid_git_ref(r: &str) -> bool {
    !r.is_empty()
        && r.len() <= 128
        && !r.starts_with('-')
        && !r.contains("..")
        && !r.contains("//")
        && !r.contains('\\')
        && !r.chars().any(|c| {
            c.is_control() || c.is_whitespace() || matches!(c, '~' | '^' | ':' | '?' | '*' | '[')
        })
}

fn git_toplevel(repo: &Path) -> Result<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("Failed to execute git")?;
    if !out.status.success() {
        bail!(
            "Not a git repository: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let path_str = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    Ok(PathBuf::from(path_str))
}

fn worktree_target(toplevel: &Path, slug: &str, data_dir: &Path) -> Result<PathBuf> {
    if !valid_slug(slug) {
        bail!("Invalid worktree slug `{slug}` (use 1-64 ASCII letters, digits, '-' or '_')");
    }
    let parent = toplevel.parent().unwrap_or(Path::new("/tmp"));
    let name = toplevel
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo");
    let wt_dir = parent.join(".allpaka-worktrees").join(name).join(slug);
    let canon_data = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf());
    if wt_dir.starts_with(data_dir) || wt_dir.starts_with(&canon_data) {
        bail!("Worktree path cannot reside inside Studio data storage");
    }
    Ok(wt_dir)
}

pub(crate) fn execute(
    app: &App,
    session_id: &str,
    shared: &SharedSession,
    project: &Project,
    allow_writes: bool,
    args: &Value,
) -> Result<Value> {
    let _ = session_id;
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("create");
    let root = project.roots.first().context("No project root available")?;
    let toplevel = git_toplevel(&root.path)?;

    match action {
        "list" => {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&toplevel)
                .args(["worktree", "list", "--porcelain"])
                .output()?;
            if !out.status.success() {
                bail!(
                    "git worktree list failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            let text = String::from_utf8_lossy(&out.stdout);
            let mut entries = Vec::new();
            let mut current = serde_json::Map::new();
            for line in text.lines() {
                if line.trim().is_empty() {
                    if !current.is_empty() {
                        entries.push(Value::Object(std::mem::take(&mut current)));
                    }
                    continue;
                }
                if let Some((k, v)) = line.split_once(' ') {
                    current.insert(k.to_owned(), json!(v));
                } else {
                    current.insert(line.to_owned(), json!(true));
                }
            }
            if !current.is_empty() {
                entries.push(Value::Object(current));
            }
            Ok(json!({ "repository": toplevel, "worktrees": entries }))
        }
        "create" => {
            if !allow_writes {
                bail!("Worktree creation requires write-enabled Auto/Goal session");
            }
            let default_slug = format!("wt-{}", storage::now_ms());
            let slug = args
                .get("slug")
                .and_then(Value::as_str)
                .unwrap_or(&default_slug);
            let wt_path = worktree_target(&toplevel, slug, &app.data)?;
            if wt_path.exists() {
                bail!("Worktree `{}` already exists", wt_path.display());
            }
            if let Some(parent) = wt_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut cmd = std::process::Command::new("git");
            cmd.arg("-C").arg(&toplevel).args(["worktree", "add"]);
            let branch = args
                .get("branch")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty());
            if let Some(b) = branch {
                if !valid_git_ref(b) {
                    bail!("Invalid git branch name `{b}`");
                }
                cmd.arg("-b").arg(b);
            }
            cmd.arg(&wt_path);
            if let Some(base) = args
                .get("base_ref")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                if !valid_git_ref(base) {
                    bail!("Invalid base git ref `{base}`");
                }
                cmd.arg(base);
            } else if branch.is_none() {
                cmd.arg("HEAD");
            }
            let out = cmd.output()?;
            if !out.status.success() {
                bail!(
                    "git worktree add failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }

            let attach = args
                .get("attach_project")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let mut attached_project_id = None;
            if attach {
                let mut wt_proj = Project {
                    id: format!("wt-{slug}"),
                    name: format!("{} ({slug})", project.name),
                    roots: vec![Root {
                        alias: "workspace".into(),
                        path: wt_path.clone(),
                        writable: true,
                        repository: true,
                        ssh: None,
                    }],
                    instructions: project.instructions.clone(),
                };
                wt_proj.validate(&app.data)?;
                let mut projects = app.projects.lock().unwrap();
                let mut updated = projects.clone();
                if let Some(existing) = updated.iter_mut().find(|p| p.id == wt_proj.id) {
                    *existing = wt_proj.clone();
                } else {
                    updated.push(wt_proj.clone());
                }
                state_file::save_list(app.data.as_path(), "projects", &updated)?;
                *projects = updated;
                drop(projects);
                {
                    let mut state = shared.lock().unwrap();
                    state.settings.project_id = wt_proj.id.clone();
                    state.working_dir = None;
                    crate::save(app, &state)?;
                }
                attached_project_id = Some(wt_proj.id);
            }

            Ok(json!({
                "created": true,
                "slug": slug,
                "worktree_path": wt_path,
                "branch": branch,
                "project_id": attached_project_id,
            }))
        }
        "remove" => {
            if !allow_writes {
                bail!("Worktree removal requires write-enabled session");
            }
            let slug = args["slug"]
                .as_str()
                .context("slug is required for remove")?;
            let wt_path = worktree_target(&toplevel, slug, &app.data)?;
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&toplevel)
                .args(["worktree", "remove", "--force"])
                .arg(&wt_path)
                .output()?;
            if !out.status.success() {
                bail!(
                    "git worktree remove failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            Ok(json!({ "removed": slug, "worktree_path": wt_path }))
        }
        other => bail!("Unsupported create_worktree action `{other}`"),
    }
}

pub(crate) async fn list_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, _, project) = project_for_session(&app, &shared)?;
    crate::blocking(move || {
        let out = execute(
            &app,
            &id,
            &shared,
            &project,
            false,
            &json!({ "action": "list" }),
        )
        .map_err(bad_request)?;
        Ok(Json(out))
    })
    .await
}

pub(crate) async fn action_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    Json(args): Json<Value>,
) -> ApiResult<Value> {
    let shared = get_session(&app, &id)?;
    let (_, settings, project) = project_for_session(&app, &shared)?;
    if settings.mode.is_read_only() {
        return Err(conflict("Worktree mutation requires non-read-only mode"));
    }
    crate::blocking(move || {
        let out = execute(&app, &id, &shared, &project, settings.allow_writes, &args)
            .map_err(bad_request)?;
        Ok(Json(out))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_slug_refs_and_keeps_worktrees_outside_data_dir() {
        assert!(valid_slug("feat-123_ok"));
        assert!(!valid_slug("-bad"));
        assert!(!valid_slug("a/b"));
        assert!(valid_git_ref("feature/parity-1"));
        assert!(!valid_git_ref("--upload-pack=evil"));
        assert!(!valid_git_ref("main..other"));

        let base = std::env::temp_dir().join(format!("allpaka-wt-{}", storage::now_ms()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let data = base.join("data");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        assert!(std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["init", "-q"])
            .status()
            .unwrap()
            .success());
        let _ = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args([
                "-c",
                "user.email=t@t.local",
                "-c",
                "user.name=T",
                "commit",
                "--allow-empty",
                "-m",
                "init",
                "-q",
            ])
            .status()
            .unwrap();

        let top = git_toplevel(&repo).unwrap();
        let target = worktree_target(&top, "iso1", &data).unwrap();
        assert!(!target.starts_with(&data));
        let ok = std::process::Command::new("git")
            .arg("-C")
            .arg(&top)
            .args(["worktree", "add", "-b", "iso-branch"])
            .arg(&target)
            .arg("HEAD")
            .output()
            .unwrap();
        assert!(
            ok.status.success(),
            "{}",
            String::from_utf8_lossy(&ok.stderr)
        );
        assert!(target.exists());
        let _ = std::fs::remove_dir_all(base);
    }
}
