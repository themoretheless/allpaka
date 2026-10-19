//! Harnessing-level recursion: a per-project library of chosen harnesses
//! and the lessons the reflector drew from each refinement job.
//!
//! ```text
//! <data>/harness/library/<project>/INDEX.md
//! <data>/harness/library/<project>/entries/<job>/harness.json
//! <data>/harness/library/<project>/entries/<job>/EXPERIENCE.md
//! <data>/harness/library/<project>/entries/<job>/entry.json
//! ```
//!
//! An entry is kept only when the reflection names a cue, a lesson and the
//! evidence behind it. Later jobs of the same project read the index and the
//! entries whose cue shares the most words with their task; a job never sees
//! its own entry.

use crate::{error, harness, harness::Tree, ApiResult, App};
use anyhow::{bail, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Entry {
    pub job_id: String,
    pub created_ms: u64,
    pub chosen: String,
    pub cue: String,
    pub lesson: String,
    pub evidence: String,
    pub confidence: String,
    pub skills: usize,
    pub scripts: usize,
    pub always_on_chars: usize,
}

fn root(data: &Path, project: &str) -> Result<PathBuf> {
    if project.is_empty()
        || project.len() > 80
        || !project
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("Invalid project ID");
    }
    Ok(data.join("harness/library").join(project))
}

fn field(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or("").trim().to_string()
}
pub fn has_experience(reflection: &Value) -> bool {
    let e = &reflection["experience"];
    e.is_object()
        && ["cue", "lesson", "evidence"]
            .iter()
            .all(|k| !field(e, k).is_empty())
}

fn experience_md(entry: &Entry, reflection: &Value, why: &str) -> String {
    let mut out = format!(
        "# Experience from harness job {}\n\nChosen: {}\n\n## Why this was chosen\n{}\n\n## What each edit did\n",
        entry.job_id,
        entry.chosen,
        if why.trim().is_empty() { "(no reason recorded)" } else { why.trim() }
    );
    for row in reflection["edits"].as_array().into_iter().flatten() {
        out.push_str(&format!(
            "- {} [{}]: {}; observed: {}; carried by {}\n",
            field(row, "component"),
            field(row, "verdict"),
            field(row, "change"),
            field(row, "observed"),
            field(row, "candidates"),
        ));
    }
    out.push_str(&format!(
        "\n## Transferable lesson\n- **cue**: {}\n- **lesson**: {}\n- **evidence**: {}\n- **confidence**: {}\n",
        entry.cue, entry.lesson, entry.evidence, entry.confidence
    ));
    out
}

/// Store the chosen harness and its lesson. Returns the entry ID, or None
/// when the reflection declined to write a lesson.
pub fn add(
    data: &Path,
    project: &str,
    job_id: &str,
    tree: &Tree,
    reflection: &Value,
    chosen: &str,
    why: &str,
) -> Result<Option<String>> {
    if !has_experience(reflection) {
        return Ok(None);
    }
    let e = &reflection["experience"];
    let clip = |s: String| s.chars().take(2000).collect::<String>();
    let entry = Entry {
        job_id: job_id.into(),
        created_ms: harness::now_ms(),
        chosen: chosen.into(),
        cue: clip(field(e, "cue")),
        lesson: clip(field(e, "lesson")),
        evidence: clip(field(e, "evidence")),
        confidence: clip(field(e, "confidence")),
        skills: harness::skills(tree).len(),
        scripts: harness::scripts(tree).len(),
        always_on_chars: harness::always_on_chars(tree),
    };
    let dir = root(data, project)?.join("entries").join(job_id);
    std::fs::create_dir_all(&dir)?;
    crate::harness_refine::write_json(&dir.join("harness.json"), tree)?;
    std::fs::write(
        dir.join("EXPERIENCE.md"),
        experience_md(&entry, reflection, why),
    )?;
    crate::harness_refine::write_json(&dir.join("entry.json"), &entry)?;
    rebuild_index(data, project)?;
    Ok(Some(job_id.into()))
}

pub fn entries(data: &Path, project: &str) -> Result<Vec<Entry>> {
    let mut list = vec![];
    let Ok(dirs) = std::fs::read_dir(root(data, project)?.join("entries")) else {
        return Ok(list);
    };
    for dir in dirs.flatten() {
        if let Ok(bytes) = std::fs::read(dir.path().join("entry.json")) {
            if let Ok(entry) = serde_json::from_slice::<Entry>(&bytes) {
                list.push(entry);
            }
        }
    }
    list.sort_by_key(|e| e.created_ms);
    Ok(list)
}

pub fn rebuild_index(data: &Path, project: &str) -> Result<()> {
    let mut out = "# Harness library\n\nOne line per refinement job: the stored harness, its size, and the situation its lesson applies to.\n\n".to_string();
    for e in entries(data, project)? {
        out.push_str(&format!(
            "- entries/{}/ — {} skills, {} scripts, {} always-on chars — cue: {}\n",
            e.job_id, e.skills, e.scripts, e.always_on_chars, e.cue
        ));
    }
    let dir = root(data, project)?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("INDEX.md"), out)?;
    Ok(())
}

fn words(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 3)
        .map(str::to_lowercase)
        .collect()
}

/// Library evidence for a proposer: the index, then whole entries in order
/// of how many words their cue and lesson share with the task, within
/// `budget` characters.
pub fn context_for(
    data: &Path,
    project: &str,
    task: &str,
    exclude_job: Option<&str>,
    budget: usize,
) -> Result<String> {
    let task_words = words(task);
    let mut ranked: Vec<(usize, Entry)> = entries(data, project)?
        .into_iter()
        .filter(|e| Some(e.job_id.as_str()) != exclude_job)
        .map(|e| {
            let shared = words(&format!("{} {}", e.cue, e.lesson))
                .intersection(&task_words)
                .count();
            (shared, e)
        })
        .collect();
    if ranked.is_empty() {
        return Ok(String::new());
    }
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.created_ms.cmp(&a.1.created_ms)));
    let base = root(data, project)?;
    let mut out = std::fs::read_to_string(base.join("INDEX.md")).unwrap_or_default();
    for (_, entry) in ranked {
        let dir = base.join("entries").join(&entry.job_id);
        let experience = std::fs::read_to_string(dir.join("EXPERIENCE.md")).unwrap_or_default();
        let tree: Tree = std::fs::read(dir.join("harness.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let files: String = tree
            .iter()
            .map(|(p, t)| format!("=== entries/{}/{p} ===\n{t}\n", entry.job_id))
            .collect();
        let block = format!("\n{experience}\n{files}");
        if out.len() + block.len() > budget {
            // A lesson without its harness still teaches something.
            if out.len() + experience.len() < budget {
                out.push('\n');
                out.push_str(&experience);
            }
            continue;
        }
        out.push_str(&block);
    }
    Ok(out)
}

#[derive(Deserialize)]
pub(crate) struct ProjectQuery {
    project_id: String,
}
pub(crate) async fn list_api(
    State(app): State<App>,
    Query(q): Query<ProjectQuery>,
) -> ApiResult<Value> {
    let list = entries(&app.data, &q.project_id).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let base = root(&app.data, &q.project_id).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let entries: Vec<Value> = list
        .iter()
        .rev()
        .map(|e| {
            json!({
                "entry": e,
                "experience": std::fs::read_to_string(base.join("entries").join(&e.job_id).join("EXPERIENCE.md")).unwrap_or_default(),
            })
        })
        .collect();
    Ok(Json(json!({"entries": entries})))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reflection(cue: &str) -> Value {
        json!({"edits":[{"component":"skills/cite","candidates":"inv1","change":"added a citation skill","observed":"inv1 quoted paths","verdict":"helped"}],
               "experience":{"cue":cue,"lesson":"add a skill that quotes file paths","evidence":"inv1 vs base1","confidence":"medium"}})
    }

    #[test]
    fn entries_need_a_complete_lesson_and_rank_by_task_overlap() {
        let data =
            std::env::temp_dir().join(format!("allpaka-library-{}", crate::evaluation::new_id()));
        let tree = Tree::from([("system.md".to_string(), "Be exact.".to_string())]);
        assert!(!has_experience(&json!({"edits":[],"experience":null})));
        assert!(!has_experience(
            &json!({"experience":{"cue":"x","lesson":"","evidence":"e"}})
        ));
        assert_eq!(
            add(
                &data,
                "p",
                "a1",
                &tree,
                &json!({"experience":null}),
                "agg1",
                ""
            )
            .unwrap(),
            None
        );
        assert_eq!(
            add(
                &data,
                "p",
                "a1",
                &tree,
                &reflection("summarising a repository layout"),
                "agg1",
                "complete"
            )
            .unwrap(),
            Some("a1".into())
        );
        add(
            &data,
            "p",
            "b2",
            &tree,
            &reflection("fixing failing database migrations"),
            "c2",
            "",
        )
        .unwrap();
        assert!(root(&data, "../x").is_err());

        let index = std::fs::read_to_string(data.join("harness/library/p/INDEX.md")).unwrap();
        assert!(
            index.contains("entries/a1/")
                && index.contains("cue: fixing failing database migrations")
        );
        let md = std::fs::read_to_string(data.join("harness/library/p/entries/a1/EXPERIENCE.md"))
            .unwrap();
        assert!(
            md.contains("- skills/cite [helped]")
                && md.contains("- **cue**: summarising a repository layout")
        );

        let context = context_for(
            &data,
            "p",
            "Write migrations for the database",
            None,
            100_000,
        )
        .unwrap();
        let (b, a) = (
            context.find("job b2").unwrap(),
            context.find("job a1").unwrap(),
        );
        assert!(b < a, "the overlapping lesson comes first");
        assert!(context.contains("=== entries/b2/system.md ==="));
        assert!(!context_for(&data, "p", "x", Some("b2"), 100_000)
            .unwrap()
            .contains("job b2"));
        assert_eq!(context_for(&data, "other", "x", None, 1000).unwrap(), "");
        std::fs::remove_dir_all(data).unwrap();
    }
}
