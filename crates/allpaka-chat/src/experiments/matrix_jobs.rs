//! Server-owned matrix sequencing. Reservations are journalled before inference.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
const MAX_BYTES: usize = 2 * 1024 * 1024;
const MAX_JOBS:usize=1000;
const MAX_CATALOG_BYTES:usize=16*1024*1024;
// Includes 16 bounded run IDs, status and a 1000-character JSON-escaped error.
const MUTABLE_RESERVE:usize=16*1024;
fn reserved_size(job:&Job)->Result<usize>{
    let mut fixed=job.clone();fixed.run_ids=vec![None;job.variants.len()];fixed.run_attempted=vec![false;job.variants.len()];fixed.status.clear();fixed.error=None;
    serde_json::to_vec(&fixed)?.len().checked_add(MUTABLE_RESERVE).context("Matrix job capacity overflow")
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Variant {
    label: String,
    sha256: String,
    request: Start,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_sha256: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Input {
    project_id: String,
    variants: Vec<Variant>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    schema_version: u32,
    id: String,
    project_id: String,
    variants: Vec<Variant>,
    run_ids: Vec<Option<String>>,
    run_attempted: Vec<bool>,
    provider_routes: Vec<String>,
    status: String,
    matrix_id: String,
    error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retry_of: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_run_ids: Option<Vec<Option<String>>>,
}
impl Job {
    fn view(&self) -> Value {
        json!({"kind":"experiment_matrix_job","schema_version":1,"id":self.id,
            "project_id":self.project_id,"status":self.status,"error":self.error,
            "matrix_id":if self.status=="completed" {Some(&self.matrix_id)} else {None},
            "variants":self.variants.iter().zip(&self.run_ids).map(|(v,id)|json!({"label":v.label,"run_id":id})).collect::<Vec<_>>(),
            "retry_of":self.retry_of,"previous_run_ids":self.previous_run_ids,
            "automatic_replay":false,"automatic_promotion":false})
    }
}
#[derive(Clone)]
pub(crate) struct Manager {
    root: Arc<PathBuf>,
    active: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    writes: Arc<Mutex<()>>,
}
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')
}
fn provider_route(provider: &provider::Provider) -> String {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(
        &json!({"id":provider.id,"base":provider.base,"anthropic":provider.anthropic}),
    )
    .unwrap();
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
impl Manager {
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("evaluation/matrix-jobs");
        std::fs::create_dir_all(&root)?;
        if !std::fs::symlink_metadata(&root)?.is_dir() || !root.canonicalize()?.starts_with(data) {
            bail!("Invalid matrix job directory");
        }
        Ok(Self {
            root: Arc::new(root.canonicalize()?),
            active: Default::default(),
            writes: Default::default(),
        })
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        if !valid_id(id) {
            bail!("Invalid matrix job ID");
        }
        Ok(self.root.join(format!("{id}.json")))
    }
    fn save(&self, job: &Job) -> Result<()> {
        use std::io::Write;
        let bytes = serde_json::to_vec(job)?;
        if bytes.len() > MAX_BYTES {
            bail!("Matrix job exceeds 2 MiB");
        }
        let _lock = self.writes.lock().unwrap();
        let path = self.path(&job.id)?;
        if !path.try_exists()? {
            let ids=self.ids()?;
            if ids.len()>=MAX_JOBS {bail!("Matrix job retention limit reached; new job was not started");}
            let mut reserved=reserved_size(job)?;
            if reserved>MAX_BYTES {bail!("Matrix job leaves insufficient space for durable progress");}
            for id in ids {reserved=reserved.checked_add(reserved_size(&self.load(&id)?)?).context("Matrix catalog capacity overflow")?;}
            if reserved>MAX_CATALOG_BYTES {bail!("Matrix job storage capacity reached; new job was not started");}
        }
        let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
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
            std::fs::rename(&temp, &path)?;
            std::fs::File::open(self.root.as_ref())?.sync_all()?;
            Ok(())
        })();
        let _ = std::fs::remove_file(temp);
        result
    }
    fn load(&self, id: &str) -> Result<Job> {
        let path = self.path(id)?;
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() || meta.len() > MAX_BYTES as u64 {
            bail!("Invalid matrix job file");
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_BYTES {
            bail!("Oversize matrix job");
        }
        let job: Job =
            serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)?;
        if job.id != id
            || job.schema_version != 1
            || !valid_id(&job.matrix_id)
            || !(2..=16).contains(&job.variants.len())
            || job.variants.len() != job.run_ids.len()
            || job.variants.len() != job.run_attempted.len()
            || job
                .run_ids
                .iter()
                .zip(&job.run_attempted)
                .any(|(id, attempted)| id.is_none() && *attempted)
            || job.provider_routes.len() != job.variants.len()
            || job
                .provider_routes
                .iter()
                .any(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
            || !matches!(
                job.status.as_str(),
                "running" | "interrupted" | "completed" | "failed" | "cancelled"
            )
            || job.run_ids.iter().flatten().any(|id| !valid_id(id))
        {
            bail!("Invalid matrix job receipt");
        }
        let mut seen = std::collections::HashSet::new();
        if self::invalid_retry_lineage(&job) {
            bail!("Invalid matrix retry lineage");
        }
        if job.run_ids.iter().flatten().any(|id| !seen.insert(id)) {
            bail!("Duplicate matrix job run");
        }
        Ok(job)
    }
    fn ids(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        let mut total = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|v| v != "json") {
                continue;
            }
            total += entry.metadata()?.len();
            if ids.len() >= MAX_JOBS || total > MAX_CATALOG_BYTES as u64 || !entry.file_type()?.is_file() {
                bail!("Matrix job scan bounds exceeded");
            }
            ids.push(
                path.file_stem()
                    .and_then(|v| v.to_str())
                    .context("Invalid matrix job filename")?
                    .to_owned(),
            );
        }
        ids.sort_by(|a, b| b.cmp(a));
        Ok(ids)
    }
    pub(crate) fn recover(&self) -> Result<()> {
        for id in self.ids()? {
            let mut job = self.load(&id)?;
            if job.status == "running" {
                job.status = "interrupted".into();
                job.error = Some("process_restart; resume requires an explicit request".into());
                self.save(&job)?;
            }
        }
        Ok(())
    }
    fn launch(
        &self,
        app: crate::App,
        mut job: Job,
        prepared: Vec<(Start, Snapshot, provider::Provider)>,
    ) -> Result<Value> {
        let mut active = self.active.lock().unwrap();
        if active.contains_key(&job.id) || active.len() >= 2 {
            bail!("Matrix job already active or two jobs are running");
        }
        job.status = "running".into();
        job.error = None;
        self.save(&job)?;
        let stop = Arc::new(AtomicBool::new(false));
        active.insert(job.id.clone(), stop.clone());
        let view = job.view();
        let manager = self.clone();
        tokio::spawn(async move {
            let mut guard = JobGuard { manager, job };
            if let Err(error) = execute_job(&app, &mut guard, &stop, prepared).await {
                guard.job.status = if stop.load(Ordering::SeqCst) {
                    "cancelled"
                } else {
                    "failed"
                }
                .into();
                guard.job.error = Some(error.to_string().chars().take(1000).collect());
            }
        });
        Ok(view)
    }
}
struct JobGuard {
    manager: Manager,
    job: Job,
}
impl Drop for JobGuard {
    fn drop(&mut self) {
        if self.job.status == "running" {
            self.job.status = "interrupted".into();
        }
        if let Err(error) = self.manager.save(&self.job) {
            eprintln!("Matrix job persistence failed: {error}");
        }
        self.manager.active.lock().unwrap().remove(&self.job.id);
    }
}
fn prepare(
    app: &crate::App,
    project: &str,
    variants: &[Variant],
) -> Result<Vec<(Start, Snapshot, provider::Provider)>> {
    crate::evaluation::project_exists(app, project)?;
    if !(2..=16).contains(&variants.len()) {
        bail!("Select 2–16 variants");
    }
    let mut labels = std::collections::HashSet::new();
    let mut prepared = Vec::new();
    for variant in variants {
        if variant.label.trim().is_empty()
            || variant.label.len() > 200
            || !labels.insert(&variant.label)
        {
            bail!("Use unique bounded variant labels");
        }
        let (mut request, snapshot, provider) = super::prepare_start(app, variant.request.clone())?;
        if let Some(hash) = variant.prompt_sha256.as_ref() {
            if request
                .prompt_snapshot
                .as_ref()
                .map(|prompt| &prompt.sha256)
                != Some(hash)
            {
                bail!("Pinned prompt hash mismatch");
            }
        }
        request.metrics.sort();
        if snapshot.project_id != project || variant.sha256 != snapshot.sha256 {
            bail!("Matrix project or dataset hash mismatch");
        }
        if let Some((base, base_snapshot, _)) = prepared.first() {
            let base: &Start = base;
            let base_snapshot: &Snapshot = base_snapshot;
            if request.metrics != base.metrics
                || snapshot.id != base_snapshot.id
                || snapshot.version != base_snapshot.version
                || snapshot.sha256 != base_snapshot.sha256
            {
                bail!("Matrix requires identical dataset and metrics");
            }
        }
        prepared.push((request, snapshot, provider));
    }
    Ok(prepared)
}
fn verify_variant(run: &Run, request: &Start, snapshot: &Snapshot) -> Result<()> {
    super::verify_run(run, snapshot)?;
    verify_variant_configuration(run, request, snapshot)
}
fn verify_variant_configuration(run: &Run, request: &Start, snapshot: &Snapshot) -> Result<()> {
    if run.metrics != request.metrics
        || run.project_id != snapshot.project_id
        || run.dataset_sha256 != snapshot.sha256
        || run.prompt_template != request.prompt_template
        || run.dataset_id != snapshot.id
        || run.dataset_version != snapshot.version
        || run.concurrency != request.concurrency
        || run.item_timeout_secs != request.item_timeout_secs
        || serde_json::to_value(&run.settings)? != serde_json::to_value(&request.settings)?
        || serde_json::to_value(&run.prompt_snapshot)?
            != serde_json::to_value(&request.prompt_snapshot)?
    {
        bail!("Reserved run configuration mismatch");
    }
    Ok(())
}
async fn execute_job(
    app: &crate::App,
    guard: &mut JobGuard,
    stop: &AtomicBool,
    prepared: Vec<(Start, Snapshot, provider::Provider)>,
) -> Result<()> {
    for (index, (request, snapshot, provider)) in prepared.into_iter().enumerate() {
        if stop.load(Ordering::SeqCst) {
            bail!("Matrix stopped");
        }
        let id = guard.job.run_ids[index]
            .get_or_insert_with(crate::evaluation::new_id)
            .clone();
        // Persist ownership BEFORE the native run may issue any provider call.
        guard.manager.save(&guard.job)?;
        if !app.experiments.path(&id)?.exists() {
            if guard.job.run_attempted[index] {
                bail!("Attempted experiment receipt is missing; replay refused");
            }
            guard.job.run_attempted[index] = true;
            guard.manager.save(&guard.job)?;
            let started = app.experiments.start_reserved(
                app.clone(),
                request.clone(),
                snapshot.clone(),
                provider,
                id.clone(),
            );
            if let Err(error) = started {
                // A returned error with no run file proves this admission never spawned inference.
                if !app.experiments.path(&id)?.exists() {
                    guard.job.run_attempted[index] = false;
                    guard.manager.save(&guard.job)?;
                }
                return Err(error);
            }
        } else if !guard.job.run_attempted[index] {
            bail!("Unexpected experiment at an unsent reservation");
        }
        loop {
            let run = app.experiments.load(&id)?;
            if run.status != "running" {
                verify_variant(&run, &request, &snapshot)?;
                break;
            }
            if !app.experiments.active.lock().unwrap().contains_key(&id) {
                let final_run = app.experiments.load(&id)?;
                if final_run.status == "running" {
                    bail!("Experiment stopped without a terminal receipt");
                }
                verify_variant(&final_run, &request, &snapshot)?;
                break;
            }
            if stop.load(Ordering::SeqCst) {
                app.experiments.cancel(&id)?;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if stop.load(Ordering::SeqCst) {
            bail!("Matrix stopped");
        }
    }
    let input = MatrixSave {
        project_id: guard.job.project_id.clone(),
        baseline_id: guard.job.run_ids[0].clone().context("Missing baseline")?,
        variants: guard
            .job
            .variants
            .iter()
            .zip(&guard.job.run_ids)
            .map(|(variant, id)| {
                Ok(MatrixVariant {
                    label: variant.label.clone(),
                    run_id: id.clone().context("Missing variant")?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let receipt = super::matrix_receipt(app, &guard.job.matrix_id, &input)?;
    let root = app.data.join("evaluation/matrices");
    std::fs::create_dir_all(&root)?;
    if !std::fs::symlink_metadata(&root)?.is_dir()
        || !root.canonicalize()?.starts_with(app.data.as_ref())
        || serde_json::to_vec(&receipt)?.len() > 4 * 1024 * 1024
    {
        bail!("Invalid matrix storage");
    }
    let path = root.join(format!("{}.json", guard.job.matrix_id));
    if path.exists() {
        let Json(existing) =
            super::read_matrix(State(app.clone()), Path(guard.job.matrix_id.clone()))
                .await
                .map_err(|_| anyhow::anyhow!("Cannot verify saved matrix"))?;
        if existing != receipt {
            bail!("Saved matrix differs from job results");
        }
    } else {
        crate::evaluation::commit_new(&path, &receipt)?;
    }
    guard.job.status = "completed".into();
    guard.manager.save(&guard.job)?;
    Ok(())
}
pub(crate) async fn start(
    State(app): State<crate::App>,
    Json(input): Json<Input>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        let prepared = prepare(&app, &input.project_id, &input.variants)?;
        let job = Job {
            schema_version: 1,
            id: crate::evaluation::new_id(),
            project_id: input.project_id,
            run_ids: vec![None; input.variants.len()],
            run_attempted: vec![false; input.variants.len()],
            provider_routes: prepared
                .iter()
                .map(|(_, _, provider)| provider_route(provider))
                .collect(),
            variants: input.variants,
            status: "running".into(),
            matrix_id: crate::evaluation::new_id(),
            error: None,
            retry_of: None,
            previous_run_ids: None,
        };
        app.matrix_jobs.launch(app.clone(), job, prepared)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn detail(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    app.matrix_jobs
        .load(&id)
        .map(|job| Json(job.view()))
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<MatrixCatalog>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        crate::evaluation::project_exists(&app, &query.project_id)?;
        if query.limit == 0 || query.limit > 100 || query.offset > 1000 {
            bail!("Invalid matrix job page");
        }
        let mut rows = Vec::new();
        for id in app.matrix_jobs.ids()? {
            let job = app.matrix_jobs.load(&id)?;
            if job.project_id == query.project_id {
                rows.push(job.view());
            }
        }
        let has_more = rows.len() > query.offset + query.limit;
        Ok(
            json!({"jobs":rows.into_iter().skip(query.offset).take(query.limit).collect::<Vec<_>>(),"offset":query.offset,"has_more":has_more}),
        )
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn cancel(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        app.matrix_jobs.load(&id)?;
        let active = app.matrix_jobs.active.lock().unwrap();
        let requested = active.get(&id).is_some_and(|flag| {
            flag.store(true, Ordering::SeqCst);
            true
        });
        Ok(json!({"cancel_requested":requested}))
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn resume(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        let job = app.matrix_jobs.load(&id)?;
        if !matches!(job.status.as_str(), "interrupted" | "failed") {
            bail!("Only interrupted or failed jobs can be resumed");
        }
        let prepared = prepare(&app, &job.project_id, &job.variants)?;
        if prepared
            .iter()
            .map(|(_, _, provider)| provider_route(provider))
            .collect::<Vec<_>>()
            != job.provider_routes
        {
            bail!("Provider routing changed since matrix admission");
        }
        for (((request, snapshot, _), id), attempted) in
            prepared.iter().zip(&job.run_ids).zip(&job.run_attempted)
        {
            if let Some(id) = id {
                if app.experiments.path(id)?.exists() {
                    if !attempted {
                        bail!("Unexpected experiment at an unsent reservation");
                    }
                    let run = app.experiments.load(id)?;
                    verify_variant(&run, request, snapshot)?;
                } else if *attempted {
                    bail!("Attempted experiment receipt is missing; replay refused");
                }
            }
        }
        app.matrix_jobs.launch(app.clone(), job, prepared)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}

fn invalid_retry_lineage(job: &Job) -> bool {
    match (&job.retry_of, &job.previous_run_ids) {
        (None, None) => false,
        (Some(parent), Some(previous)) => {
            !valid_id(parent)
                || parent == &job.id
                || previous.len() != job.variants.len()
                || previous.iter().flatten().any(|id| !valid_id(id))
        }
        _ => true,
    }
}
pub(crate) async fn retry(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        let original = app.matrix_jobs.load(&id)?;
        if !matches!(
            original.status.as_str(),
            "interrupted" | "failed" | "cancelled"
        ) {
            bail!("Only interrupted, failed or cancelled jobs can be retried");
        }
        if app.matrix_jobs.active.lock().unwrap().contains_key(&id) {
            bail!("Original matrix job is still active");
        }
        let prepared = prepare(&app, &original.project_id, &original.variants)?;
        if prepared
            .iter()
            .map(|(_, _, provider)| provider_route(provider))
            .collect::<Vec<_>>()
            != original.provider_routes
        {
            bail!("Provider routing changed since matrix admission");
        }
        let mut next = original.clone();
        next.id = crate::evaluation::new_id();
        next.matrix_id = crate::evaluation::new_id();
        next.retry_of = Some(original.id.clone());
        next.previous_run_ids = Some(original.run_ids.clone());
        for (index, (request, snapshot, _)) in prepared.iter().enumerate() {
            let mut reusable = false;
            if let Some(run_id) = original.run_ids[index].as_ref() {
                if app.experiments.active.lock().unwrap().contains_key(run_id) {
                    bail!("Original experiment is still active");
                }
                if app.experiments.path(run_id)?.exists() {
                    if !original.run_attempted[index] {
                        bail!("Unexpected experiment at an unsent reservation");
                    }
                    let run = app.experiments.load(run_id)?;
                    verify_variant_configuration(&run, request, snapshot)?;
                    if run.status == "running" {
                        bail!("Original experiment has no confirmed terminal state");
                    }
                    if run.status == "completed" && run.strict_quality {
                        verify_variant(&run, request, snapshot)?;
                        reusable = true;
                    }
                }
            }
            if !reusable {
                next.run_ids[index] = None;
                next.run_attempted[index] = false;
            }
        }
        app.matrix_jobs.launch(app.clone(), next, prepared)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PathBuf, Manager, Job) {
        let data = std::env::temp_dir().join(format!(
            "allpaka-matrix-jobs-{}",
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let manager = Manager::new(&data).unwrap();
        let request:Start=serde_json::from_value(json!({"dataset_id":"dataset","dataset_version":1,"settings":{"project_id":"project","provider":"local","model":"mock","mode":"chat","allow_writes":false},"prompt_template":"{{input}}","metrics":["exact_match"]})).unwrap();
        let job = Job {
            schema_version: 1,
            id: crate::evaluation::new_id(),
            project_id: "project".into(),
            variants: ["baseline", "candidate"]
                .into_iter()
                .map(|label| Variant {
                    label: label.into(),
                    sha256: "a".repeat(64),
                    request: request.clone(),
                    prompt_sha256: None,
                })
                .collect(),
            run_ids: vec![Some(crate::evaluation::new_id()), None],
            run_attempted: vec![false; 2],
            provider_routes: vec!["a".repeat(64); 2],
            status: "running".into(),
            matrix_id: crate::evaluation::new_id(),
            error: None,
            retry_of: None,
            previous_run_ids: None,
        };
        (data, manager, job)
    }
    #[test]
    fn full_catalog_rejects_admission_but_retains_updates(){
        let(data,manager,mut job)=fixture();
        for i in 0..MAX_JOBS {let mut existing=job.clone();existing.id=format!("{i:x}");std::fs::write(manager.path(&existing.id).unwrap(),serde_json::to_vec(&existing).unwrap()).unwrap();}
        assert_eq!(manager.ids().unwrap().len(),MAX_JOBS);assert!(manager.save(&job).unwrap_err().to_string().contains("retention limit"));assert!(!manager.path(&job.id).unwrap().exists());
        job.id="0".into();job.status="completed".into();manager.save(&job).unwrap();assert_eq!(manager.load("0").unwrap().status,"completed");assert_eq!(manager.ids().unwrap().len(),MAX_JOBS);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn competing_admissions_share_remaining_reserved_bytes(){
        let(data,manager,mut job)=fixture();job.variants[0].request.prompt_template="x".repeat(1024*1024);let capacity=MAX_CATALOG_BYTES/reserved_size(&job).unwrap();
        for i in 0..capacity-1 {let mut existing=job.clone();existing.id=format!("{i:x}");std::fs::write(manager.path(&existing.id).unwrap(),serde_json::to_vec(&existing).unwrap()).unwrap();}
        let barrier=Arc::new(std::sync::Barrier::new(2));let threads:Vec<_>=(0..2).map(|_|{let manager=manager.clone();let mut job=job.clone();job.id=crate::evaluation::new_id();let barrier=barrier.clone();std::thread::spawn(move||{barrier.wait();manager.save(&job).is_ok()})}).collect();
        assert_eq!(threads.into_iter().filter_map(|t|t.join().ok()).filter(|ok|*ok).count(),1);assert_eq!(manager.ids().unwrap().len(),capacity);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn reserved_catalog_capacity_keeps_space_for_progress(){
        let(data,manager,mut job)=fixture();
        job.variants[0].request.prompt_template="x".repeat(1024*1024);
        let weight=reserved_size(&job).unwrap();assert!(weight<MAX_BYTES);
        let count=MAX_CATALOG_BYTES/weight;
        for i in 0..count {let mut existing=job.clone();existing.id=format!("{i:x}");std::fs::write(manager.path(&existing.id).unwrap(),serde_json::to_vec(&existing).unwrap()).unwrap();}
        assert!(manager.save(&job).unwrap_err().to_string().contains("storage capacity"));
        job.id="0".into();let before=reserved_size(&job).unwrap();job.run_ids=vec![Some("a".repeat(80));2];job.error=Some("\u{0000}".repeat(1000));job.status="interrupted".into();assert_eq!(reserved_size(&job).unwrap(),before);assert!(serde_json::to_vec(&job).unwrap().len()<=before);manager.save(&job).unwrap();assert_eq!(manager.ids().unwrap().len(),count);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn provider_route_pins_identity_endpoint_and_protocol_without_credentials() {
        let mut provider = provider::defaults().remove(0);
        let original = provider_route(&provider);
        provider.key = "rotated credential".into();
        provider.name = "renamed".into();
        assert_eq!(provider_route(&provider), original);
        let mut changed = provider.clone();
        changed.base.push_str("/different");
        assert_ne!(provider_route(&changed), original);
        changed = provider.clone();
        changed.anthropic = !changed.anthropic;
        assert_ne!(provider_route(&changed), original);
        changed = provider;
        changed.id.push_str("-different");
        assert_ne!(provider_route(&changed), original);
    }
    #[test]
    fn retry_lineage_retains_original_ids_and_rejects_inconsistent_links() {
        let (data, manager, original) = fixture();
        manager.save(&original).unwrap();
        let mut next = original.clone();
        next.id = crate::evaluation::new_id();
        next.retry_of = Some(original.id.clone());
        next.previous_run_ids = Some(original.run_ids.clone());
        manager.save(&next).unwrap();
        let view = manager.load(&next.id).unwrap().view();
        assert_eq!(view["retry_of"], original.id);
        assert_eq!(
            view["previous_run_ids"],
            serde_json::to_value(&original.run_ids).unwrap()
        );
        assert_eq!(
            manager.load(&original.id).unwrap().run_ids,
            original.run_ids
        );
        next.retry_of = Some(next.id.clone());
        manager.save(&next).unwrap();
        assert!(manager.load(&next.id).is_err());
        next.retry_of = Some(original.id.clone());
        next.previous_run_ids = Some(vec![]);
        manager.save(&next).unwrap();
        assert!(manager.load(&next.id).is_err());
        next.previous_run_ids = None;
        manager.save(&next).unwrap();
        assert!(manager.load(&next.id).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn journal_recovery_retains_reservations_without_replay() {
        let (data, manager, job) = fixture();
        manager.save(&job).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(manager.path(&job.id).unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        let reopened = Manager::new(&data).unwrap();
        reopened.recover().unwrap();
        let recovered = reopened.load(&job.id).unwrap();
        assert_eq!(recovered.status, "interrupted");
        assert_eq!(recovered.run_ids, job.run_ids);
        assert_eq!(recovered.matrix_id, job.matrix_id);
        assert!(reopened.active.lock().unwrap().is_empty());
        let view = recovered.view();
        assert_eq!(view["automatic_replay"], false);
        assert!(view.get("request").is_none());
        assert!(!view.to_string().contains("prompt_template"));
        reopened.recover().unwrap();
        assert_eq!(reopened.load(&job.id).unwrap().view(), view);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn journal_rejects_duplicate_keys_and_duplicate_run_ownership() {
        let (data, manager, mut job) = fixture();
        job.run_ids[1] = job.run_ids[0].clone();
        manager.save(&job).unwrap();
        assert!(manager.load(&job.id).is_err());
        let path = manager.path(&job.id).unwrap();
        let bytes = serde_json::to_string(&job).unwrap();
        std::fs::write(&path, format!("{{\"id\":\"fake\",{}", &bytes[1..])).unwrap();
        assert!(manager.load(&job.id).is_err());
        assert!(manager.path("../escape").is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
}
