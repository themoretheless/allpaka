//! Read-only, bounded project browsing for Studio. Same path policy as tools.
use crate::{context::Project, error, tools, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    project_id: String,
    root: String,
    #[serde(default = "dot")]
    path: String,
    #[serde(default)]
    query: String,
}
fn dot() -> String {
    ".".into()
}
fn checked(root: &Path, path: &str, excluded: &Path) -> Result<std::path::PathBuf> {
    let file = tools::resolve(root, path, false)?;
    if file.starts_with(excluded) {
        bail!("Conversation storage is not available");
    }
    Ok(file)
}
fn browse(project: &Project, input: &Request, excluded: &Path, action: &str) -> Result<Value> {
    if input.path.len() > 4096 || input.query.len() > 256 {
        bail!("Path or search query is too long");
    }
    let root = &project
        .roots
        .iter()
        .find(|r| r.alias == input.root)
        .context("Unknown project folder")?
        .path;
    let start = checked(root, &input.path, excluded)?;
    if action == "read" {
        if !start.is_file() {
            bail!("Choose a regular text file");
        }
        let text = tools::read_text(&start, 100_000)?;
        if text.contains('\0') {
            bail!("Binary files cannot be added as text");
        }
        return Ok(
            json!({"project_id":project.id,"root":input.root,"path":input.path,"text":text,"bytes":text.len()}),
        );
    }
    if !start.is_dir() {
        bail!("Choose a folder");
    }
    let recursive = action == "search" && !input.query.trim().is_empty();
    let query = input.query.to_lowercase();
    let mut pending = vec![start];
    let mut entries = Vec::new();
    let mut visited = 0usize;
    let mut truncated = false;
    let deadline = Instant::now() + Duration::from_millis(250);
    'scan: while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir)? {
            visited += 1;
            if visited > 20_000 || Instant::now() > deadline {
                truncated = true;
                break 'scan;
            }
            let Ok(entry) = entry else { continue };
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() || (!kind.is_file() && !kind.is_dir()) {
                continue;
            }
            let path = entry.path();
            let Some(relative) = path.strip_prefix(root)?.to_str().map(str::to_owned) else {
                continue;
            };
            if checked(root, &relative, excluded).is_err() {
                continue;
            }
            if recursive
                && kind.is_dir()
                && !matches!(
                    name.as_str(),
                    "target" | "node_modules" | "vendor" | "dist" | "build"
                )
            {
                pending.push(path);
            }
            if recursive && !kind.is_file() {
                continue;
            }
            let Some(score) = (if recursive {
                fuzzy(&relative.to_lowercase(), &query)
            } else {
                Some(0)
            }) else {
                continue;
            };
            entries.push((
                score,
                json!({"name":name,"path":relative,"directory":kind.is_dir()}),
            ));
            if !recursive && entries.len() >= 400 {
                truncated = true;
                break 'scan;
            }
        }
    }
    entries.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1["directory"].as_bool().cmp(&a.1["directory"].as_bool()))
            .then_with(|| a.1["path"].as_str().cmp(&b.1["path"].as_str()))
    });
    if entries.len() > 150 && recursive {
        entries.truncate(150);
        truncated = true;
    }
    Ok(
        json!({"project_id":project.id,"root":input.root,"path":input.path,"entries":entries.into_iter().map(|(_,v)|v).collect::<Vec<_>>(),"truncated":truncated,"visited":visited,"skips_build_folders":recursive}),
    )
}
fn fuzzy(path: &str, query: &str) -> Option<usize> {
    let mut letters = query.chars().filter(|c| !c.is_whitespace()).peekable();
    let mut score = 0;
    for (i, c) in path.chars().enumerate() {
        if letters.peek() == Some(&c) {
            letters.next();
            score += i;
        }
        if letters.peek().is_none() {
            return Some(score);
        }
    }
    None
}
async fn run(app: App, input: Request, action: &'static str) -> ApiResult<Value> {
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == input.project_id)
        .cloned()
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Unknown project"))?;
    static SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
    let permit = SLOTS.try_acquire().map_err(|_| {
        error(
            StatusCode::TOO_MANY_REQUESTS,
            "File browser is busy; try again",
        )
    })?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        browse(&project, &input, &app.data, action)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    result
        .map(Json)
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn list(State(app): State<App>, Json(input): Json<Request>) -> ApiResult<Value> {
    run(app, input, "list").await
}
pub(crate) async fn search(State(app): State<App>, Json(input): Json<Request>) -> ApiResult<Value> {
    run(app, input, "search").await
}
pub(crate) async fn read(State(app): State<App>, Json(input): Json<Request>) -> ApiResult<Value> {
    run(app, input, "read").await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn browsing_is_scoped_and_bounded() {
        let dir = std::env::temp_dir().join(format!("studio-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("history")).unwrap();
        std::fs::write(dir.join("hello.rs"), "one\ntwo\nthree\n").unwrap();
        std::fs::write(dir.join(".secret"), "private").unwrap();
        std::fs::write(dir.join("binary"), b"a\0b").unwrap();
        std::fs::write(dir.join("history/private"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.join("hello.rs"), dir.join("link")).unwrap();
        let p = Project {
            id: "p".into(),
            name: "p".into(),
            instructions: String::new(),
            roots: vec![crate::context::Root {
                alias: "src".into(),
                path: dir.canonicalize().unwrap(),
                writable: false,
                repository: false,
            }],
        };
        let mut q = Request {
            project_id: "p".into(),
            root: "src".into(),
            path: ".".into(),
            query: "hlrs".into(),
        };
        let excluded = dir.canonicalize().unwrap().join("history");
        let result = browse(&p, &q, &excluded, "search").unwrap();
        assert_eq!(result["entries"].as_array().unwrap().len(), 1);
        assert_eq!(result["entries"][0]["name"], "hello.rs");
        for path in [
            "../hello.rs",
            ".secret",
            "history/private",
            "link",
            "binary",
        ] {
            q.path = path.into();
            assert!(browse(&p, &q, &excluded, "read").is_err(), "{path}");
        }
        q.path = "hello.rs".into();
        assert_eq!(
            browse(&p, &q, &excluded, "read").unwrap()["text"],
            "one\ntwo\nthree\n"
        );
        std::fs::write(dir.join("large"), vec![b'a'; 100001]).unwrap();
        q.path = "large".into();
        assert!(browse(&p, &q, &excluded, "read").is_err());
        q.path = ".".into();
        q.query = "".into();
        for i in 0..405 {
            std::fs::write(dir.join(format!("f{i}")), "").unwrap();
        }
        assert_eq!(
            browse(&p, &q, &excluded, "list").unwrap()["truncated"],
            true
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
