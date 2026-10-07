//! Human-authored annotations live separately from metadata-only traces.
//! Every change appends an immutable revision, including recoverable removal.
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
};
const MAX_BYTES: usize = 1024 * 1024;
const MAX_VERSIONS: u64 = 2000;
#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    writes: Arc<Mutex<()>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Annotation {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    span_id: Option<usize>,
    author: String,
    #[serde(default)]
    metric: Option<String>,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    comment: Option<String>,
    #[serde(default)]
    correction: Option<String>,
    #[serde(default)]
    deleted: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Revision {
    trace_id: String,
    version: u64,
    #[serde(default)]
    saved_ms: u64,
    annotations: Vec<Annotation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Save {
    base_version: u64,
    annotation: Annotation,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Search {
    version: Option<u64>,
}
#[derive(Debug)]
struct Conflict;
impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Feedback revision changed; reload before saving")
    }
}
impl std::error::Error for Conflict {}
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 100 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}
fn validate(a: &Annotation) -> Result<()> {
    if a.id.as_ref().is_some_and(|id| !valid_id(id))
        || a.author.trim().is_empty()
        || a.author.len() > 200
    {
        bail!("Invalid annotation ID or reviewer name");
    }
    if a.metric
        .as_ref()
        .is_some_and(|s| s.trim().is_empty() || s.len() > 100)
        || a.category
            .as_ref()
            .is_some_and(|s| s.trim().is_empty() || s.len() > 200)
        || a.comment.as_ref().is_some_and(|s| s.len() > 16000)
        || a.correction.as_ref().is_some_and(|s| s.len() > 65536)
        || a.value
            .is_some_and(|v| !v.is_finite() || v.abs() > 1_000_000.0)
    {
        bail!("Annotation field exceeds its limit");
    }
    if a.value.is_some() && a.category.is_some()
        || a.metric.is_some() != (a.value.is_some() || a.category.is_some())
    {
        bail!("A named score requires either a numeric value or a category");
    }
    if a.metric.is_none()
        && a.comment.as_ref().is_none_or(|s| s.trim().is_empty())
        && a.correction.as_ref().is_none_or(|s| s.is_empty())
    {
        bail!("Provide a score, comment or corrected answer");
    }
    Ok(())
}
impl Store {
    pub(crate) fn validate_review_reference(&self,trace:&str,version:u64,annotation_id:&str,span:Option<usize>,reviewer:&str)->Result<()> {
        if version==0 {bail!("Review completion requires a saved feedback version");}
        let revision=self.load(trace,Some(version))?;
        if !revision.annotations.iter().any(|a|a.id.as_deref()==Some(annotation_id)&&!a.deleted&&a.span_id==span&&a.author==reviewer){bail!("Review evidence does not match target and assigned reviewer");}
        Ok(())
    }
    pub(crate) fn export_snapshot(&self,id:&str)->Result<Value>{
        self.load(id,None).map(response)
    }
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("observability/feedback");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if !root.starts_with(data) {
            bail!("Feedback storage escapes Studio data directory");
        }
        Ok(Self {
            root: Arc::new(root),
            writes: Default::default(),
        })
    }
    fn directory(&self, id: &str) -> Result<PathBuf> {
        if !valid_id(id) || !id.starts_with("trace-") {
            bail!("Invalid trace ID");
        }
        let path = self.root.join(id);
        match std::fs::symlink_metadata(&path) {
            Ok(m) if !m.is_dir() => bail!("Invalid feedback directory"),
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
        Ok(path)
    }
    fn load(&self, id: &str, version: Option<u64>) -> Result<Revision> {
        let directory = self.directory(id)?;
        let mut latest = 0;
        if directory.exists() {
            let mut count = 0;
            for entry in std::fs::read_dir(&directory)? {
                let entry = entry?;
                if entry.path().extension().is_none_or(|e| e != "json") {
                    continue;
                }
                count += 1;
                if count > MAX_VERSIONS {
                    bail!("Feedback revision limit reached");
                }
                let version = entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .context("Invalid revision filename")?
                    .parse::<u64>()?;
                latest = latest.max(version);
            }
        }
        let version = version.unwrap_or(latest);
        if version == 0 {
            return Ok(Revision {
                trace_id: id.into(),
                version: 0,
                saved_ms: 0,
                annotations: vec![],
            });
        }
        if version > MAX_VERSIONS {
            bail!("Invalid feedback version");
        }
        let path = directory.join(format!("{version}.json"));
        let m = std::fs::symlink_metadata(&path)?;
        if !m.is_file() || m.len() > MAX_BYTES as u64 {
            bail!("Invalid feedback revision");
        }
        let mut bytes = vec![];
        std::fs::File::open(path)?
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_BYTES {
            bail!("Oversize feedback revision");
        }
        let revision: Revision = serde_json::from_slice(&bytes)?;
        if revision.trace_id != id
            || revision.version != version
            || revision.annotations.len() > 1000
        {
            bail!("Feedback identity mismatch");
        }
        let mut ids = std::collections::HashSet::new();
        for a in &revision.annotations {
            validate(a)?;
            if a.id.is_none() || !ids.insert(a.id.as_ref()) {
                bail!("Invalid stored annotation identity");
            }
        }
        Ok(revision)
    }
    fn save(&self, id: &str, request: Save) -> Result<Revision> {
        validate(&request.annotation)?;
        let _write = self.writes.lock().unwrap();
        let mut revision = self.load(id, None)?;
        if revision.version != request.base_version {
            return Err(Conflict.into());
        }
        if revision.version >= MAX_VERSIONS {
            bail!("Feedback revision limit reached");
        }
        let mut a = request.annotation;
        if let Some(annotation_id) = &a.id {
            let existing = revision
                .annotations
                .iter_mut()
                .find(|n| n.id.as_ref() == Some(annotation_id))
                .context("Annotation not found")?;
            if existing.span_id != a.span_id || existing.author != a.author {
                bail!("Annotation target and reviewer cannot change");
            }
            *existing = a;
        } else {
            if a.deleted || revision.annotations.len() >= 1000 {
                bail!("Invalid new annotation or annotation limit reached");
            }
            a.id = Some(crate::evaluation::new_id());
            revision.annotations.push(a);
        }
        revision.version += 1;
        revision.saved_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64;
        if serde_json::to_vec(&revision)?.len() > MAX_BYTES {
            bail!("Feedback revision exceeds 1 MiB");
        }
        let directory = self.directory(id)?;
        std::fs::create_dir_all(&directory)?;
        if !directory.canonicalize()?.starts_with(self.root.as_ref()) {
            bail!("Feedback storage escapes its root");
        }
        let path = directory.join(format!("{}.json", revision.version));
        crate::evaluation::commit_new(&path, &revision).map_err(|e| {
            if path.exists() {
                anyhow::Error::new(Conflict)
            } else {
                e
            }
        })?;
        Ok(revision)
    }
}
fn response(revision: Revision) -> Value {
    let mut scores: BTreeMap<(Option<usize>, String), (f64, usize, BTreeMap<String, usize>)> =
        BTreeMap::new();
    for a in revision.annotations.iter().filter(|a| !a.deleted) {
        if let Some(metric) = &a.metric {
            let entry = scores.entry((a.span_id, metric.clone())).or_default();
            if let Some(value) = a.value {
                entry.0 += value;
                entry.1 += 1;
            }
            if let Some(category) = &a.category {
                *entry.2.entry(category.clone()).or_default() += 1;
            }
        }
    }
    let summaries:Vec<_>=scores.into_iter().map(|((span_id,metric),(sum,count,categories))|json!({"span_id":span_id,"metric":metric,"count":count,"mean":if count>0 {Some(sum/count as f64)}else{None},"categories":categories})).collect();
    json!({"trace_id":revision.trace_id,"version":revision.version,"saved_ms":revision.saved_ms,"annotations":revision.annotations,"summaries":summaries,"reviewer_identity":"self_reported","content_capture":"explicit_annotations_only"})
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Query(q): Query<Search>,
) -> crate::ApiResult<Value> {
    app.observability
        .validate_target(&id, None)
        .and_then(|_| app.feedback.load(&id, q.version))
        .map(response)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryQuery {
    #[serde(default)] offset: usize,
    #[serde(default = "history_limit")] limit: usize,
}
fn history_limit() -> usize { 20 }
pub(crate) async fn history(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        app.observability.validate_target(&id, None)?;
        app.feedback.history(&id, query.offset, query.limit)
    })();
    result.map(Json).map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
impl Store {
    fn history(&self, id: &str, offset: usize, limit: usize) -> Result<Value> {
        if offset > MAX_VERSIONS as usize || !(1..=100).contains(&limit) {
            bail!("Invalid feedback history page");
        }
        let latest = self.load(id, None)?.version;
        let total = latest as usize;
        let mut rows = Vec::new();
        for index in offset..total.min(offset + limit) {
            let revision = self.load(id, Some(latest - index as u64))?;
            rows.push(json!({"version":revision.version,"saved_ms":revision.saved_ms,
                "annotation_count":revision.annotations.len(),
                "active_count":revision.annotations.iter().filter(|a|!a.deleted).count()}));
        }
        Ok(json!({"kind":"feedback_history","trace_id":id,"latest_version":latest,
            "total":total,"offset":offset,"limit":limit,"has_more":offset+rows.len()<total,
            "order":"version_desc","versions":rows,"provider_calls":0,
            "reviewer_identity":"self_reported"}))
    }
}
pub(crate) async fn save(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Json(request): Json<Save>,
) -> crate::ApiResult<Value> {
    app.observability
        .validate_target(&id, request.annotation.span_id)
        .and_then(|_| app.feedback.save(&id, request))
        .map(response)
        .map(Json)
        .map_err(|e| {
            crate::error(
                if e.is::<Conflict>() {
                    StatusCode::CONFLICT
                } else {
                    StatusCode::BAD_REQUEST
                },
                e,
            )
        })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn annotation(author: &str, value: f64) -> Annotation {
        Annotation {
            id: None,
            span_id: Some(1),
            author: author.into(),
            metric: Some("quality".into()),
            value: Some(value),
            category: None,
            comment: Some("review".into()),
            correction: None,
            deleted: false,
        }
    }
    #[test]
    fn immutable_reviews_conflict_aggregate_remove_restore_and_restart() {
        let data = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let store = Store::new(&data).unwrap();
        let id = "trace-test";
        let first = store
            .save(
                id,
                Save {
                    base_version: 0,
                    annotation: annotation("a", 1.0),
                },
            )
            .unwrap();
        assert!(store
            .save(
                id,
                Save {
                    base_version: 0,
                    annotation: annotation("b", 0.0)
                }
            )
            .unwrap_err()
            .is::<Conflict>());
        let second = store
            .save(
                id,
                Save {
                    base_version: 1,
                    annotation: annotation("b", 0.0),
                },
            )
            .unwrap();
        assert_eq!(response(second)["summaries"][0]["mean"], 0.5);
        let mut removal = first.annotations[0].clone();
        removal.deleted = true;
        let third = store
            .save(
                id,
                Save {
                    base_version: 2,
                    annotation: removal.clone(),
                },
            )
            .unwrap();
        assert_eq!(response(third)["summaries"][0]["mean"], 0.0);
        removal.deleted = false;
        let fourth = store
            .save(
                id,
                Save {
                    base_version: 3,
                    annotation: removal,
                },
            )
            .unwrap();
        assert_eq!(response(fourth)["summaries"][0]["mean"], 0.5);
        let restarted = Store::new(&data).unwrap();
        assert_eq!(restarted.load(id, None).unwrap().version, 4);
        assert_eq!(restarted.load(id, Some(1)).unwrap().annotations.len(), 1);
        let page = restarted.history(id, 0, 2).unwrap();
        assert_eq!(page["total"], 4); assert_eq!(page["has_more"], true);
        assert_eq!(page["versions"][0]["version"], 4);
        assert_eq!(page["versions"][1]["active_count"], 1);
        assert_eq!(restarted.history(id, 2, 2).unwrap()["versions"][1]["version"], 1);
        assert!(restarted.history(id, 0, 101).is_err());
        assert!(restarted.history(id, 2001, 20).is_err());
        assert_eq!(restarted.history("trace-empty", 0, 20).unwrap()["total"], 0);
        assert!(store.load("../escape", None).is_err());
        assert!(validate(&annotation("a", f64::NAN)).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn concurrent_review_writes_preserve_a_single_revision() {
        let data = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let store = Store::new(&data).unwrap();
        let gate = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let store = Store::new(&data).unwrap();
                let gate = gate.clone();
                std::thread::spawn(move || {
                    gate.wait();
                    store
                        .save(
                            "trace-test",
                            Save {
                                base_version: 0,
                                annotation: annotation(&i.to_string(), 1.0),
                            },
                        )
                        .is_ok()
                })
            })
            .collect();
        assert_eq!(
            handles
                .into_iter()
                .filter_map(|h| h.join().ok())
                .filter(|v| *v)
                .count(),
            1
        );
        assert_eq!(store.load("trace-test", None).unwrap().annotations.len(), 1);
        std::fs::remove_dir_all(data).unwrap();
    }
}
