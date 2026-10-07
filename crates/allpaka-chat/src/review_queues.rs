//! Explicit, project-scoped sources for durable human review queues.
use anyhow::{Context, Result, bail};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    io::{Read, Write},
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
};
const MAX_BYTES: usize = 1024 * 1024;
#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    writes: Arc<Mutex<()>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Target {
    trace_id: String,
    #[serde(default)]
    span_id: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Create {
    project_id: String,
    name: String,
    #[serde(default)]
    instructions: String,
    targets: Vec<Target>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Queue {
    id: String,
    project_id: String,
    name: String,
    instructions: String,
    version: u64,
    #[serde(default)]
    archived: bool,
    targets: Vec<Target>,
    #[serde(default)]
    assignments: std::collections::BTreeMap<usize, String>,
    #[serde(default)]
    completions: std::collections::BTreeMap<usize, Evidence>,
    #[serde(default)]
    history: Vec<Change>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    version: u64,
    action: String,
    target_index: Option<usize>,
    reviewer: Option<String>,
    feedback_version: Option<u64>,
    annotation_id: Option<String>,
    archived: bool,
}
impl Queue {
    fn record(&mut self, action: &str, index: Option<usize>) -> Result<()> {
        if self.history.len() >= 2000 {
            bail!("Review queue history limit reached");
        }
        let reviewer = index.and_then(|i| self.assignments.get(&i)).cloned();
        let evidence = index.and_then(|i| self.completions.get(&i));
        self.history.push(Change {
            version: self.version,
            action: action.into(),
            target_index: index,
            reviewer,
            feedback_version: evidence.map(|e| e.feedback_version),
            annotation_id: evidence.map(|e| e.annotation_id.clone()),
            archived: self.archived,
        });
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    reviewer: String,
    feedback_version: u64,
    annotation_id: String,
}
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 100 && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
}
impl Store {
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("observability/review-queues");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if !root.starts_with(data) {
            bail!("Review queue storage escapes data directory");
        }
        Ok(Self {
            root: Arc::new(root),
            writes: Default::default(),
        })
    }
    fn read(&self, id: &str) -> Result<Queue> {
        if !valid_id(id) {
            bail!("Invalid review queue ID");
        }
        let path = self.root.join(format!("{id}.json"));
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() > MAX_BYTES as u64 {
            bail!("Invalid review queue artifact");
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_BYTES {
            bail!("Review queue artifact exceeds limit");
        }
        let queue: Queue = serde_json::from_slice(&bytes)?;
        if queue.id != id || queue.version == 0 {
            bail!("Review queue identity differs");
        }
        Ok(queue)
    }
    fn write_locked(&self, queue: &Queue, bytes: &[u8]) -> Result<()> {
        let final_path = self.root.join(format!("{}.json", queue.id));
        let temp = self.root.join(format!("{}.tmp", queue.id));
        let result = (|| -> Result<()> {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temp, final_path)?;
            std::fs::File::open(self.root.as_ref())?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temp);
        }
        result?;
        Ok(())
    }
    fn create(&self, input: Create) -> Result<Queue> {
        if input.project_id.trim().is_empty()
            || input.project_id.len() > 100
            || input.name.trim().is_empty()
            || input.name.len() > 200
            || input.instructions.len() > 16000
            || input.targets.is_empty()
            || input.targets.len() > 200
        {
            bail!("Invalid review queue fields");
        }
        let mut unique = std::collections::BTreeSet::new();
        for target in &input.targets {
            if !valid_id(&target.trace_id)
                || !unique.insert((target.trace_id.clone(), target.span_id))
            {
                bail!("Invalid or duplicate review queue target");
            }
        }
        let mut queue = Queue {
            id: crate::evaluation::new_id(),
            project_id: input.project_id,
            name: input.name,
            instructions: input.instructions,
            version: 1,
            archived: false,
            targets: input.targets,
            assignments: Default::default(),
            completions: Default::default(),
            history: Default::default(),
        };
        queue.record("create", None)?;
        let bytes = serde_json::to_vec(&queue)?;
        if bytes.len() > MAX_BYTES {
            bail!("Review queue artifact exceeds limit");
        }
        let _writer = self.writes.lock().unwrap();
        let mut count = 0usize;
        let mut total = bytes.len() as u64;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            if entry.path().extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            count += 1;
            let metadata = entry.metadata()?;
            if !entry.file_type()?.is_file() {
                bail!("Invalid review queue catalogue entry");
            }
            total = total
                .checked_add(metadata.len())
                .context("Review queue capacity overflow")?;
            if count >= 1000 || total > 128 * 1024 * 1024 {
                bail!("Review queue capacity reached");
            }
        }
        self.write_locked(&queue, &bytes)?;
        Ok(queue)
    }
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Search {
    project_id: String,
    status: Option<String>,
    reviewer: Option<String>,
    name: Option<String>,
    archived: Option<bool>,
    #[serde(default)]
    offset: usize,
    #[serde(default = "page_limit")]
    limit: usize,
}
fn page_limit() -> usize {
    20
}
impl Store {
    fn list(&self, query: &Search) -> Result<Value> {
        if query.project_id.is_empty()
            || query.project_id.len() > 100
            || query.offset > 1000
            || !(1..=100).contains(&query.limit)
        {
            bail!("Invalid review queue page");
        }
        if query
            .status
            .as_deref()
            .is_some_and(|s| !matches!(s, "pending" | "completed" | "unassigned"))
            || query
                .reviewer
                .as_ref()
                .is_some_and(|s| s.trim().is_empty() || s.len() > 200)
        {
            bail!("Invalid review queue filter");
        }
        if query
            .name
            .as_ref()
            .is_some_and(|s| s.trim().is_empty() || s.len() > 200)
        {
            bail!("Invalid review queue name search");
        }
        let name_search = query.name.as_ref().map(|s| s.to_lowercase());
        let mut rows = Vec::new();
        let mut count = 0;
        let mut bytes = 0u64;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            count += 1;
            if !entry.file_type()?.is_file() {
                bail!("Invalid review queue catalog entry");
            }
            bytes = bytes
                .checked_add(entry.metadata()?.len())
                .context("Review queue capacity overflow")?;
            if count > 1000 || bytes > 128 * 1024 * 1024 {
                bail!("Review queue catalog exceeds limit");
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("Invalid review queue filename")?;
            let queue = self.read(id)?;
            let matches_status = match query.status.as_deref() {
                Some("pending") => queue.completions.len() < queue.targets.len(),
                Some("completed") => queue.completions.len() == queue.targets.len(),
                Some("unassigned") => queue.assignments.len() < queue.targets.len(),
                _ => true,
            };
            let matches_reviewer = query
                .reviewer
                .as_ref()
                .is_none_or(|reviewer| queue.assignments.values().any(|name| name == reviewer));
            if query
                .archived
                .is_none_or(|archived| queue.archived == archived)
                && queue.project_id == query.project_id
                && matches_status
                && matches_reviewer
                && name_search
                    .as_ref()
                    .is_none_or(|name| queue.name.to_lowercase().contains(name))
            {
                rows.push(serde_json::json!({"id":queue.id,"project_id":queue.project_id,"name":queue.name,"version":queue.version,"target_count":queue.targets.len(),"assigned_count":queue.assignments.len(),"completed_count":queue.completions.len(),"archived":queue.archived}));
            }
        }
        rows.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
        let total = rows.len();
        let rows = rows
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect::<Vec<_>>();
        Ok(
            serde_json::json!({"kind":"review_queue_catalog","project_id":query.project_id,"total":total,"offset":query.offset,"limit":query.limit,"has_more":query.offset+rows.len()<total,"order":"id_desc","queues":rows,"provider_calls":0,"filters":{"status":query.status,"reviewer":query.reviewer,"name":query.name,"archived":query.archived}}),
        )
    }
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<Search>,
) -> crate::ApiResult<Value> {
    app.review_queues
        .list(&query)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryPage {
    #[serde(default)]
    offset: usize,
    #[serde(default = "page_limit")]
    limit: usize,
}
pub(crate) async fn history(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Query(page): Query<HistoryPage>,
) -> crate::ApiResult<Value> {
    (||->Result<Value> {
        if page.offset>2000 || !(1..=100).contains(&page.limit) {bail!("Invalid review queue history page");}
        let queue=app.review_queues.read(&id)?;
        let total=queue.history.len();
        let entries=queue.history.iter().rev().skip(page.offset).take(page.limit).collect::<Vec<_>>();
        Ok(serde_json::json!({"kind":"review_queue_history","queue_id":id,"project_id":queue.project_id,"queue_version":queue.version,"total":total,"offset":page.offset,"limit":page.limit,"has_more":page.offset+entries.len()<total,"order":"version_desc","history_complete":queue.history.first().is_some_and(|e|e.version==1),"entries":entries,"provider_calls":0,"reviewer_identity":"self_reported"}))
    })().map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Assignment {
    base_version: u64,
    target_index: usize,
    reviewer: Option<String>,
}
#[derive(Debug)]
struct Conflict;
impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Review queue version changed; reload before assigning")
    }
}
impl std::error::Error for Conflict {}
impl Store {
    fn persist_update(&self, queue: &Queue) -> Result<()> {
        let bytes = serde_json::to_vec(&queue)?;
        if bytes.len() > MAX_BYTES {
            bail!("Review queue artifact exceeds limit");
        }
        let mut total = bytes.len() as u64;
        let mut count = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            count += 1;
            if !entry.file_type()?.is_file() {
                bail!("Invalid review queue catalogue entry");
            }
            if path != self.root.join(format!("{}.json", queue.id)) {
                total = total
                    .checked_add(entry.metadata()?.len())
                    .context("Review queue capacity overflow")?;
            }
            if count > 1000 || total > 128 * 1024 * 1024 {
                bail!("Review queue capacity reached");
            }
        }
        self.write_locked(&queue, &bytes)?;
        Ok(())
    }
    fn assign(&self, id: &str, input: Assignment) -> Result<Queue> {
        if input
            .reviewer
            .as_ref()
            .is_some_and(|name| name.trim().is_empty() || name.len() > 200)
        {
            bail!("Invalid reviewer name");
        }
        let _writer = self.writes.lock().unwrap();
        let mut queue = self.read(id)?;
        if queue.archived {
            bail!("Restore archived review queue before modifying targets");
        }
        if queue.version != input.base_version {
            return Err(Conflict.into());
        }
        if input.target_index >= queue.targets.len() {
            bail!("Review queue target does not exist");
        }
        if queue.completions.contains_key(&input.target_index) {
            bail!("Reopen completed review before changing its assignment");
        }
        match input.reviewer {
            Some(name) => {
                queue.assignments.insert(input.target_index, name);
            }
            None => {
                queue.assignments.remove(&input.target_index);
            }
        }
        queue.version = queue
            .version
            .checked_add(1)
            .context("Review queue version overflow")?;
        queue.record("assignment", Some(input.target_index))?;
        self.persist_update(&queue)?;
        Ok(queue)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Lifecycle {
    base_version: u64,
    archived: bool,
}
impl Store {
    fn lifecycle(&self, id: &str, input: Lifecycle) -> Result<Queue> {
        let _writer = self.writes.lock().unwrap();
        let mut queue = self.read(id)?;
        if queue.version != input.base_version {
            return Err(Conflict.into());
        }
        if queue.archived == input.archived {
            bail!("Review queue already has requested archive state");
        }
        queue.archived = input.archived;
        queue.version = queue
            .version
            .checked_add(1)
            .context("Review queue version overflow")?;
        queue.record(if input.archived { "archive" } else { "restore" }, None)?;
        self.persist_update(&queue)?;
        Ok(queue)
    }
}
pub(crate) async fn lifecycle(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Json(input): Json<Lifecycle>,
) -> crate::ApiResult<Value> {
    app.review_queues
        .lifecycle(&id, input)
        .and_then(|queue| Ok(serde_json::to_value(queue)?))
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Completion {
    base_version: u64,
    target_index: usize,
    action: String,
    feedback_version: Option<u64>,
    annotation_id: Option<String>,
}
impl Store {
    fn complete(
        &self,
        id: &str,
        input: Completion,
        feedback: &crate::feedback::Store,
    ) -> Result<Queue> {
        let _writer = self.writes.lock().unwrap();
        let mut queue = self.read(id)?;
        if queue.archived {
            bail!("Restore archived review queue before modifying targets");
        }
        if queue.version != input.base_version {
            return Err(Conflict.into());
        }
        let target = queue
            .targets
            .get(input.target_index)
            .context("Review target does not exist")?;
        match input.action.as_str() {
            "complete" => {
                if queue.completions.contains_key(&input.target_index) {
                    bail!("Review target is already completed");
                }
                let reviewer = queue
                    .assignments
                    .get(&input.target_index)
                    .context("Assign a reviewer before completion")?
                    .clone();
                let version = input.feedback_version.context("Select feedback version")?;
                let annotation = input.annotation_id.context("Select feedback annotation")?;
                feedback.validate_review_reference(
                    &target.trace_id,
                    version,
                    &annotation,
                    target.span_id,
                    &reviewer,
                )?;
                queue.completions.insert(
                    input.target_index,
                    Evidence {
                        reviewer,
                        feedback_version: version,
                        annotation_id: annotation,
                    },
                );
            }
            "reopen" => {
                if input.feedback_version.is_some() || input.annotation_id.is_some() {
                    bail!("Reopen does not accept replacement evidence");
                }
                if queue.completions.remove(&input.target_index).is_none() {
                    bail!("Review target is not completed");
                }
            }
            _ => bail!("Use complete or reopen review action"),
        }
        queue.version = queue
            .version
            .checked_add(1)
            .context("Review queue version overflow")?;
        queue.record(&input.action, Some(input.target_index))?;
        self.persist_update(&queue)?;
        Ok(queue)
    }
}
pub(crate) async fn complete(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Json(input): Json<Completion>,
) -> crate::ApiResult<Value> {
    app.review_queues
        .complete(&id, input, &app.feedback)
        .and_then(|queue| Ok(serde_json::to_value(queue)?))
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
pub(crate) async fn assign(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Json(input): Json<Assignment>,
) -> crate::ApiResult<Value> {
    app.review_queues
        .assign(&id, input)
        .and_then(|queue| Ok(serde_json::to_value(queue)?))
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
pub(crate) async fn create(
    State(app): State<crate::App>,
    Json(input): Json<Create>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        if input.targets.is_empty() || input.targets.len() > 200 {
            bail!("Use 1-200 review targets");
        }
        if !app
            .projects
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.id == input.project_id)
        {
            bail!("Review queue project does not exist");
        }
        for target in &input.targets {
            app.observability.validate_project_target(
                &target.trace_id,
                target.span_id,
                &input.project_id,
            )?;
        }
        Ok(serde_json::to_value(app.review_queues.create(input)?)?)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn detail(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    app.review_queues
        .read(&id)
        .and_then(|queue| Ok(serde_json::to_value(queue)?))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn durable_queue_preserves_sources_and_rejects_duplicates() {
        let root = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let store = Store::new(&root).unwrap();
        let target = Target {
            trace_id: "trace-test".into(),
            span_id: Some(1),
        };
        let input = || Create {
            project_id: "default".into(),
            name: "Review".into(),
            instructions: "Check accuracy".into(),
            targets: vec![target.clone()],
        };
        let queue = store.create(input()).unwrap();
        let loaded = Store::new(&root).unwrap().read(&queue.id).unwrap();
        assert_eq!(loaded.instructions, "Check accuracy");
        assert_eq!(loaded.targets[0].span_id, Some(1));
        let assigned = store
            .assign(
                &queue.id,
                Assignment {
                    base_version: 1,
                    target_index: 0,
                    reviewer: Some("Reviewer".into()),
                },
            )
            .unwrap();
        assert_eq!(assigned.version, 2);
        assert_eq!(
            Store::new(&root)
                .unwrap()
                .read(&queue.id)
                .unwrap()
                .assignments[&0],
            "Reviewer"
        );
        assert!(
            store
                .assign(
                    &queue.id,
                    Assignment {
                        base_version: 1,
                        target_index: 0,
                        reviewer: None
                    }
                )
                .unwrap_err()
                .is::<Conflict>()
        );
        assert!(
            store
                .assign(
                    &queue.id,
                    Assignment {
                        base_version: 2,
                        target_index: 1,
                        reviewer: None
                    }
                )
                .is_err()
        );
        let cleared = store
            .assign(
                &queue.id,
                Assignment {
                    base_version: 2,
                    target_index: 0,
                    reviewer: None,
                },
            )
            .unwrap();
        assert_eq!(cleared.version, 3);
        assert!(cleared.assignments.is_empty());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut workers = Vec::new();
        for name in ["a", "b"] {
            let manager = store.clone();
            let id = queue.id.clone();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                manager
                    .assign(
                        &id,
                        Assignment {
                            base_version: 3,
                            target_index: 0,
                            reviewer: Some(name.into()),
                        },
                    )
                    .is_ok()
            }));
        }
        assert_eq!(
            workers
                .into_iter()
                .map(|worker| usize::from(worker.join().unwrap()))
                .sum::<usize>(),
            1
        );
        assert_eq!(store.read(&queue.id).unwrap().version, 4);

        let page = store
            .list(&Search {
                project_id: "default".into(),
                offset: 0,
                limit: 1,
                ..Search::default()
            })
            .unwrap();
        assert_eq!(page["total"], 1);
        assert_eq!(page["queues"][0]["id"], queue.id);
        assert!(page["queues"][0].get("instructions").is_none());
        assert_eq!(
            store
                .list(&Search {
                    project_id: "foreign".into(),
                    offset: 0,
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()["total"],
            0
        );
        assert!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    offset: 0,
                    limit: 101,
                    ..Search::default()
                })
                .is_err()
        );

        let assigned_name = store.read(&queue.id).unwrap().assignments[&0].clone();
        let filtered = |status: &str, reviewer: Option<String>| {
            store
                .list(&Search {
                    project_id: "default".into(),
                    status: Some(status.into()),
                    reviewer,
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()
        };
        assert_eq!(filtered("pending", Some(assigned_name))["total"], 1);
        assert_eq!(filtered("completed", None)["total"], 0);
        assert_eq!(filtered("unassigned", None)["total"], 0);
        assert_eq!(filtered("pending", Some("absent".into()))["total"], 0);
        assert!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    status: Some("unknown".into()),
                    limit: 20,
                    ..Search::default()
                })
                .is_err()
        );
        let named = store.read(&queue.id).unwrap().name.to_uppercase();
        assert_eq!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    name: Some(named),
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()["total"],
            1
        );
        assert_eq!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    name: Some("absent name".into()),
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()["total"],
            0
        );
        assert!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    name: Some(" ".into()),
                    limit: 20,
                    ..Search::default()
                })
                .is_err()
        );
        let current = store.read(&queue.id).unwrap();
        let archived = store
            .lifecycle(
                &queue.id,
                Lifecycle {
                    base_version: current.version,
                    archived: true,
                },
            )
            .unwrap();
        assert!(Store::new(&root).unwrap().read(&queue.id).unwrap().archived);
        assert!(
            store
                .assign(
                    &queue.id,
                    Assignment {
                        base_version: archived.version,
                        target_index: 0,
                        reviewer: None
                    }
                )
                .is_err()
        );
        assert!(
            store
                .lifecycle(
                    &queue.id,
                    Lifecycle {
                        base_version: current.version,
                        archived: false
                    }
                )
                .unwrap_err()
                .is::<Conflict>()
        );
        assert_eq!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    archived: Some(false),
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()["total"],
            0
        );
        assert_eq!(
            store
                .list(&Search {
                    project_id: "default".into(),
                    archived: Some(true),
                    limit: 20,
                    ..Search::default()
                })
                .unwrap()["total"],
            1
        );
        let restored = store
            .lifecycle(
                &queue.id,
                Lifecycle {
                    base_version: archived.version,
                    archived: false,
                },
            )
            .unwrap();
        assert!(!restored.archived);
        assert_eq!(
            serde_json::to_value(&restored.targets).unwrap(),
            serde_json::to_value(&current.targets).unwrap()
        );
        assert_eq!(restored.assignments, current.assignments);
        assert_eq!(restored.history.first().unwrap().action, "create");
        assert_eq!(restored.history.last().unwrap().action, "restore");
        assert_eq!(restored.history.last().unwrap().version, restored.version);
        assert!(
            restored
                .history
                .windows(2)
                .all(|w| w[0].version < w[1].version)
        );
        let mut legacy_value = serde_json::to_value(&restored).unwrap();
        legacy_value.as_object_mut().unwrap().remove("history");
        let mut legacy: Queue = serde_json::from_value(legacy_value).unwrap();
        assert!(legacy.history.is_empty());
        legacy.version += 1;
        legacy.record("archive", None).unwrap();
        assert_ne!(legacy.history[0].version, 1);
        let mut duplicate = input();
        duplicate.targets.push(target.clone());
        assert!(store.create(duplicate).is_err());
        assert!(store.read("../escape").is_err());
        for _ in 0..21 {
            store.create(input()).unwrap();
        }
        assert_eq!(filtered("unassigned", None)["total"], 21);
        let mut completed = store.read(&queue.id).unwrap();
        completed.completions.insert(
            0,
            Evidence {
                reviewer: completed.assignments[&0].clone(),
                feedback_version: 1,
                annotation_id: "saved".into(),
            },
        );
        store.persist_update(&completed).unwrap();
        assert_eq!(filtered("completed", None)["total"], 1);
        assert_eq!(filtered("pending", None)["total"], 21);
        let paged = store
            .list(&Search {
                project_id: "default".into(),
                status: Some("pending".into()),
                offset: 20,
                limit: 20,
                ..Search::default()
            })
            .unwrap();
        assert_eq!(paged["total"], 21);
        assert_eq!(paged["queues"].as_array().unwrap().len(), 1);
        assert_eq!(paged["has_more"], false);
        let first = store
            .list(&Search {
                project_id: "default".into(),
                offset: 0,
                limit: 20,
                ..Search::default()
            })
            .unwrap();
        let second = store
            .list(&Search {
                project_id: "default".into(),
                offset: 20,
                limit: 20,
                ..Search::default()
            })
            .unwrap();
        assert_eq!(first["total"], 22);
        assert_eq!(first["queues"].as_array().unwrap().len(), 20);
        assert_eq!(first["has_more"], true);
        assert_eq!(second["queues"].as_array().unwrap().len(), 2);
        assert_eq!(second["has_more"], false);
        assert!(
            first["queues"][19]["id"].as_str().unwrap()
                > second["queues"][0]["id"].as_str().unwrap()
        );

        std::fs::remove_dir_all(root).unwrap();
    }
}
