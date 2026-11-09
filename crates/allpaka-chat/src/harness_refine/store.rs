use super::*;

// ---- jobs ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProbeExecution {
    FilesOnly,
    TrustedUnsandboxed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidationRequest {
    #[serde(default = "default_validation_runs")]
    pub(super) runs: usize,
}

impl Default for ValidationRequest {
    fn default() -> Self {
        Self {
            runs: default_validation_runs(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartRequest {
    pub(super) session_id: String,
    #[serde(default = "default_k")]
    pub(super) k: usize,
    #[serde(default)]
    pub(super) mode: Recursion,
    /// Proposer model; defaults to the session's model.
    #[serde(default)]
    pub(super) provider: Option<String>,
    #[serde(default)]
    pub(super) model: Option<String>,
    /// Proposers edit files as agents (default) or answer with JSON.
    #[serde(default = "default_agentic")]
    pub(super) agentic: bool,
    /// Optional rubric: the base and chosen rollouts are scored for the
    /// report only.
    #[serde(default)]
    pub(super) rubric: Option<String>,
    /// Independent fresh Base/Champion comparisons after selection.
    #[serde(default)]
    pub(super) validation: ValidationRequest,
    /// A stream of tasks refined in order; each sees the library entries of
    /// the ones before it. Empty means the conversation's own task.
    #[serde(default)]
    pub(super) tasks: Vec<String>,
    /// Take the task stream from a Studio dataset instead: each sample's
    /// input (with its contexts) is a task and its expected output goes to
    /// the rubric judge as the reference.
    #[serde(default)]
    pub(super) dataset: Option<DatasetRef>,
    /// Internal probes are files-only unless command execution is explicitly
    /// trusted. Workspace copies are not a security sandbox.
    #[serde(default)]
    pub(super) execution: Option<ProbeExecution>,
    /// Run rollouts with an external agent CLI instead of Studio's loop.
    /// Needs the command plugin and explicit trusted_unsandboxed execution.
    #[serde(default)]
    pub(super) external: Option<External>,
}

impl StartRequest {
    fn policy(&self) -> crate::ExecutionPolicy {
        match self.execution {
            Some(ProbeExecution::TrustedUnsandboxed) => {
                crate::ExecutionPolicy::trusted_unsandboxed()
            }
            Some(ProbeExecution::FilesOnly) | None => crate::ExecutionPolicy::files_only(),
        }
    }
}

pub(super) const ROLLOUT_CACHE_SCHEMA: &str = "harness-rollout-cache-v2";
#[allow(clippy::too_many_arguments)]
pub(super) fn cache_namespace(
    task: &str,
    executor: &Settings,
    proposer_provider: &str,
    proposer_model: &str,
    external: Option<&External>,
    execution: crate::ExecutionPolicy,
    workspace_fingerprint: &str,
    sample_id: Option<&str>,
) -> Result<String> {
    Ok(harness::fingerprint(&Tree::from([
        ("cache_schema".to_string(), ROLLOUT_CACHE_SCHEMA.to_string()),
        ("task".to_string(), task.to_string()),
        (
            "executor_settings".to_string(),
            serde_json::to_string(executor)?,
        ),
        ("executor_provider".to_string(), executor.provider.clone()),
        ("executor_model".to_string(), executor.model.clone()),
        (
            "proposer_provider".to_string(),
            proposer_provider.to_string(),
        ),
        ("proposer_model".to_string(), proposer_model.to_string()),
        ("external".to_string(), serde_json::to_string(&external)?),
        ("execution_policy".to_string(), format!("{execution:?}")),
        (
            "workspace_fingerprint".to_string(),
            workspace_fingerprint.to_string(),
        ),
        (
            "dataset_sample_id".to_string(),
            sample_id.unwrap_or("").to_string(),
        ),
    ])))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetRef {
    pub(super) id: String,
    #[serde(default)]
    pub(super) version: Option<u64>,
    #[serde(default)]
    pub(super) limit: Option<usize>,
}
pub(super) fn default_k() -> usize {
    1
}
pub(super) fn default_agentic() -> bool {
    true
}
pub(super) fn default_validation_runs() -> usize {
    1
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TaskResult {
    pub(super) index: usize,
    pub(super) task: String,
    pub(super) outcome: Outcome,
    pub(super) revision: Option<String>,
    pub(super) library_entry: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct JobRecord {
    pub(super) id: String,
    pub(super) project_id: String,
    pub(super) request: StartRequest,
    pub(super) tasks: Vec<String>,
    /// Dataset references, parallel to `tasks`; empty without a dataset.
    #[serde(default)]
    pub(super) references: Vec<Option<String>>,
    /// Stable dataset sample IDs, parallel to `tasks`; empty without a dataset.
    #[serde(default)]
    pub(super) sample_ids: Vec<Option<String>>,
    pub(super) executor: Settings,
    pub(super) base_revision: Option<String>,
    pub(super) created_ms: u64,
    pub(super) status: String,
    #[serde(default)]
    pub(super) events: Vec<String>,
    #[serde(default)]
    pub(super) results: Vec<TaskResult>,
    #[serde(default)]
    pub(super) warnings: Vec<String>,
    #[serde(default)]
    pub(super) error: Option<String>,
}

#[derive(Clone, Default)]
pub(crate) struct Manager {
    pub(super) running: Arc<Mutex<HashMap<String, tokio::task::AbortHandle>>>,
}
pub(super) fn jobs_dir(data: &std::path::Path) -> PathBuf {
    data.join("harness/jobs")
}
pub(super) fn valid_job(id: &str) -> bool {
    !id.is_empty() && id.len() < 80 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}
pub(super) fn load_job(data: &std::path::Path, id: &str) -> Result<JobRecord> {
    if !valid_job(id) {
        bail!("Invalid job ID");
    }
    read_json(&jobs_dir(data).join(id).join("job.json")).context("Harness job not found")
}
pub(super) fn update_job(
    data: &std::path::Path,
    id: &str,
    f: impl FnOnce(&mut JobRecord),
) -> Result<JobRecord> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    let mut job = load_job(data, id)?;
    f(&mut job);
    if job.events.len() > 200 {
        job.events.drain(..job.events.len() - 200);
    }
    write_json(&jobs_dir(data).join(id).join("job.json"), &job)?;
    Ok(job)
}
/// Jobs left running by a stopped server can be resumed explicitly.
pub(crate) fn recover(data: &std::path::Path) -> Result<()> {
    let Ok(entries) = std::fs::read_dir(jobs_dir(data)) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let id = entry.file_name().to_string_lossy().to_string();
        if let Ok(job) = load_job(data, &id) {
            if job.status == "running" {
                update_job(data, &id, |j| {
                    j.status = "interrupted".into();
                    j.events.push(
                        "server restarted; resume reuses finished rollouts and stages".into(),
                    );
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn spawn(app: App, id: String) {
    let manager = app.harness_jobs.clone();
    let task_app = app.clone();
    let task_id = id.clone();
    let handle = tokio::spawn(async move {
        let data = task_app.data.clone();
        let result = execute(&task_app, &task_id).await;
        let _ = update_job(&data, &task_id, |j| match result {
            Ok(()) => j.status = "completed".into(),
            Err(e) => {
                j.status = "failed".into();
                j.error = Some(format!("{e:#}"));
            }
        });
        task_app
            .harness_jobs
            .running
            .lock()
            .unwrap()
            .remove(&task_id);
    });
    manager
        .running
        .lock()
        .unwrap()
        .insert(id, handle.abort_handle());
}

pub(super) const LEGACY_EXTERNAL_WARNING: &str = "Legacy external executor job resumed as trusted_unsandboxed; workspace copies do not sandbox shell commands.";

pub(super) async fn execute(app: &App, id: &str) -> Result<()> {
    let mut legacy_external = false;
    let job = update_job(&app.data, id, |j| {
        legacy_external = j.request.external.is_some() && j.request.execution.is_none();
        if legacy_external {
            j.request.execution = Some(ProbeExecution::TrustedUnsandboxed);
            if !j.warnings.iter().any(|w| w == LEGACY_EXTERNAL_WARNING) {
                j.warnings.push(LEGACY_EXTERNAL_WARNING.into());
            }
            if !j.events.iter().any(|e| e == LEGACY_EXTERNAL_WARNING) {
                j.events.push(LEGACY_EXTERNAL_WARNING.into());
            }
        }
        j.status = "running".into();
        j.error = None;
        j.events.push("started".into());
    })?;
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == job.project_id)
        .cloned()
        .context("Project not found")?;
    let base = match &job.base_revision {
        Some(id) => harness::load_revision(&app.data, id)?.tree,
        None => Tree::new(),
    };
    let mut proposer = job.executor.clone();
    proposer.mode = Mode::Chat;
    proposer.allow_writes = false;
    proposer.guardrails = None;
    proposer.json_mode = false;
    proposer.max_output_tokens = proposer.max_output_tokens.max(16_384);
    if let Some(provider) = &job.request.provider {
        proposer.provider = provider.clone();
    }
    if let Some(model) = &job.request.model {
        proposer.model = model.clone();
    }
    let proposer_provider = app
        .providers
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == proposer.provider)
        .cloned()
        .context("Proposer provider not found")?;
    let mut proposer_agent = proposer.clone();
    proposer_agent.mode = Mode::Auto;
    proposer_agent.allow_writes = true;
    proposer_agent.max_steps = 40;
    for (index, task) in job.tasks.iter().enumerate() {
        if job.results.iter().any(|r| r.index == index) {
            continue;
        }
        let label = if job.tasks.len() > 1 {
            format!("task {}/{}: ", index + 1, job.tasks.len())
        } else {
            String::new()
        };
        let scratch = std::env::temp_dir()
            .join("allpaka-harness")
            .join(id)
            .join(index.to_string());
        let workspace = WorkspaceSnapshot::capture(&project, &scratch.join("snapshot"))?;
        let cache_namespace = cache_namespace(
            task,
            &job.executor,
            &proposer.provider,
            &proposer.model,
            job.request.external.as_ref(),
            job.request.policy(),
            &workspace.fingerprint,
            job.sample_ids.get(index).and_then(Option::as_deref),
        )?;
        let engine = StudioEngine {
            app: app.clone(),
            executor: job.executor.clone(),
            proposer: proposer.clone(),
            proposer_agent: proposer_agent.clone(),
            proposer_provider: proposer_provider.clone(),
            project: project.clone(),
            workspace,
            task: task.clone(),
            scratch: scratch.clone(),
            external: job.request.external.clone(),
            execution: job.request.policy(),
            rollout_timeout: std::time::Duration::from_secs(15 * 60),
        };
        let entry_id = format!("{id}-{index}");
        let library = harness_library::context_for(
            &app.data,
            &job.project_id,
            task,
            Some(&entry_id),
            24_000,
        )?;
        let data = app.data.clone();
        let job_id = id.to_string();
        let prefix = label.clone();
        let progress = move |event: String| {
            let _ = update_job(&data, &job_id, |j| {
                j.events.push(format!("{prefix}{event}"))
            });
        };
        let outcome = run(Plan {
            engine: &engine,
            dir: jobs_dir(&app.data)
                .join(id)
                .join("tasks")
                .join(index.to_string()),
            k: job.request.k,
            mode: job.request.mode,
            task: task.clone(),
            base: base.clone(),
            cache_namespace,
            workspace_fingerprint: engine.workspace.fingerprint.clone(),
            library,
            width: 2,
            agentic: job.request.agentic,
            rubric: job.request.rubric.clone(),
            validation_runs: job.request.validation.runs,
            reference: job.references.get(index).cloned().flatten(),
            progress: &progress,
        })
        .await;
        let _ = std::fs::remove_dir_all(&scratch);
        let outcome = outcome?;
        let revision = if outcome.tree != base {
            Some(
                harness::save_revision(
                    &app.data,
                    &job.project_id,
                    &outcome.tree,
                    job.base_revision.clone(),
                    &format!(
                        "harness job {id}{}: {} ({:?}, k={})",
                        if job.tasks.len() > 1 {
                            format!(" task {}", index + 1)
                        } else {
                            String::new()
                        },
                        outcome.chosen,
                        outcome.mode,
                        outcome.k
                    ),
                )?
                .id,
            )
        } else {
            None
        };
        let library_entry = match &outcome.reflection {
            Some(reflection) => harness_library::add(
                &app.data,
                &job.project_id,
                &entry_id,
                &outcome.tree,
                reflection,
                &outcome.chosen,
                &outcome.selection_why,
            )?,
            None => None,
        };
        update_job(&app.data, id, |j| {
            j.events.push(format!(
                "{label}chosen {} ({})",
                outcome.chosen, outcome.chosen_kind
            ));
            j.results.push(TaskResult {
                index,
                task: task.clone(),
                outcome,
                revision,
                library_entry,
            });
            j.results.sort_by_key(|r| r.index);
            j.warnings = stream_warnings(&j.results, legacy_external);
        })?;
    }
    Ok(())
}

/// Across a stream of tasks (after Harness² report.py): refinement that
/// keeps choosing the base harness is not adapting anything.
pub(super) fn stream_warnings(results: &[TaskResult], legacy_external: bool) -> Vec<String> {
    let mut out = if legacy_external {
        vec![LEGACY_EXTERNAL_WARNING.into()]
    } else {
        vec![]
    };
    if results.len() >= 2 {
        let base = results
            .iter()
            .filter(|r| r.outcome.chosen_kind == "base")
            .count();
        if base * 5 >= results.len() * 4 {
            out.push(format!(
                "The base harness was kept in {base} of {} tasks.",
                results.len()
            ));
        }
        let fell = results
            .iter()
            .filter(|r| !r.outcome.fallbacks.is_empty())
            .count();
        if fell > 0 {
            out.push(format!(
                "Stages fell back in {fell} of {} tasks.",
                results.len()
            ));
        }
    }
    out
}

