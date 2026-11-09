use super::*;

pub(super) fn bad(e: impl ToString) -> crate::ApiError {
    error(StatusCode::BAD_REQUEST, e)
}
pub(crate) async fn start_api(
    State(app): State<App>,
    Json(mut request): Json<StartRequest>,
) -> ApiResult<Value> {
    if !(1..=4).contains(&request.k) {
        return Err(bad("k must be between 1 and 4"));
    }
    if request.validation.runs > 4 {
        return Err(bad("validation.runs must be between 0 and 4"));
    }
    if request.tasks.len() > 20
        || request
            .tasks
            .iter()
            .any(|t| t.trim().is_empty() || t.len() > 16_000)
    {
        return Err(bad("Give at most 20 non-empty tasks of up to 16000 bytes"));
    }
    if request
        .rubric
        .as_ref()
        .is_some_and(|r| r.trim().is_empty() || r.len() > 16_000)
    {
        return Err(bad("Rubric must be 1–16000 bytes"));
    }
    if app.harness_jobs.running.lock().unwrap().len() >= 2 {
        return Err(error(
            StatusCode::TOO_MANY_REQUESTS,
            "Two harness jobs are already running",
        ));
    }
    let session = app
        .sessions
        .lock()
        .unwrap()
        .get(&request.session_id)
        .map(|(s, _)| s.clone())
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
    let (mut executor, task) = {
        let s = session.lock().unwrap();
        let task = s
            .goal
            .as_ref()
            .and_then(|g| s.messages.get(g.message_index))
            .or_else(|| {
                s.messages.iter().rev().find(|m| {
                    m.role == "user"
                        && !m.content.starts_with("Continue the task")
                        && !m.content.starts_with("Goal continuation")
                })
            })
            .map(|m| m.content.clone())
            .unwrap_or_default();
        (s.settings.clone(), task)
    };
    let (tasks, references, sample_ids) = match &request.dataset {
        Some(source) => {
            if !request.tasks.is_empty() {
                return Err(bad("Give either tasks or a dataset, not both"));
            }
            let dataset = app
                .evaluation
                .load(&source.id, source.version)
                .map_err(bad)?;
            if dataset.project_id != executor.project_id {
                return Err(bad("The dataset belongs to another project"));
            }
            let limit = source.limit.unwrap_or(20).clamp(1, 20);
            dataset
                .samples
                .iter()
                .take(limit)
                .map(|sample| {
                    let mut task = sample.input.clone();
                    if !sample.contexts.is_empty() {
                        task.push_str(&format!("\n\nContext:\n{}", sample.contexts.join("\n\n")));
                    }
                    (
                        task,
                        sample.expected_output.clone(),
                        Some(sample.id.clone()),
                    )
                })
                .fold(
                    (vec![], vec![], vec![]),
                    |(mut tasks, mut references, mut sample_ids), (task, reference, sample_id)| {
                        tasks.push(task);
                        references.push(reference);
                        sample_ids.push(sample_id);
                        (tasks, references, sample_ids)
                    },
                )
        }
        None if request.tasks.is_empty() => (vec![task], vec![], vec![]),
        None => (request.tasks.clone(), vec![], vec![]),
    };
    if tasks.is_empty() || tasks[0].trim().is_empty() {
        return Err(bad("The conversation has no task to refine on"));
    }
    if tasks.iter().any(|t| t.len() > 16_000) {
        return Err(bad("A task exceeds 16000 bytes"));
    }
    if let Some(external) = &request.external {
        if request.execution != Some(ProbeExecution::TrustedUnsandboxed) {
            return Err(bad(
                "An external executor requires execution=trusted_unsandboxed. Workspace copies do not sandbox shell commands.",
            ));
        }
        if external.command.trim().is_empty() || external.command.len() > 2000 {
            return Err(bad("The external command must be 1–2000 bytes"));
        }
        if !crate::plugins::tool_unlocked(&app.plugins, "run_command") {
            return Err(bad(
                "An external executor runs a command: enable the command plugin first",
            ));
        }
    } else if request.execution.is_none() {
        request.execution = Some(ProbeExecution::FilesOnly);
    }
    match executor.mode {
        Mode::Swarm => {
            return Err(bad(
                "Harness refinement runs single-agent modes; switch away from Swarm",
            ))
        }
        Mode::Goal => executor.mode = Mode::Auto,
        _ => {}
    }
    crate::validate_settings(&executor, &app).map_err(bad)?;
    let base_revision = harness::active_revision(&app.data, &executor.project_id)
        .map_err(bad)?
        .map(|r| r.id);
    let id = crate::evaluation::new_id();
    let job = JobRecord {
        id: id.clone(),
        project_id: executor.project_id.clone(),
        request,
        tasks,
        references,
        sample_ids,
        executor,
        base_revision,
        created_ms: harness::now_ms(),
        status: "queued".into(),
        events: vec![],
        results: vec![],
        warnings: vec![],
        error: None,
    };
    write_json(&jobs_dir(&app.data).join(&id).join("job.json"), &job).map_err(bad)?;
    spawn(app.clone(), id);
    Ok(Json(json!(job)))
}
#[derive(Deserialize)]
pub(crate) struct ProjectQuery {
    pub(super) project_id: String,
}
pub(crate) async fn list_api(
    State(app): State<App>,
    Query(q): Query<ProjectQuery>,
) -> ApiResult<Value> {
    let mut jobs = vec![];
    if let Ok(entries) = std::fs::read_dir(jobs_dir(&app.data)) {
        for entry in entries.flatten() {
            if let Ok(job) = load_job(&app.data, &entry.file_name().to_string_lossy()) {
                if job.project_id == q.project_id {
                    jobs.push(json!({"id":job.id,"status":job.status,"created_ms":job.created_ms,"k":job.request.k,"mode":job.request.mode,"agentic":job.request.agentic,"tasks":job.tasks.len(),"done":job.results.len(),"task":job.tasks[0].chars().take(120).collect::<String>()}));
                }
            }
        }
    }
    jobs.sort_by_key(|j| std::cmp::Reverse(j["created_ms"].as_u64()));
    jobs.truncate(50);
    Ok(Json(json!({"jobs": jobs})))
}
pub(crate) async fn get_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(crate::not_found)?;
    let base = job
        .base_revision
        .as_deref()
        .and_then(|r| harness::load_revision(&app.data, r).ok())
        .map(|r| r.tree)
        .unwrap_or_default();
    let diffs: Vec<harness::Diff> = job
        .results
        .iter()
        .map(|r| harness::diff(&base, &r.outcome.tree))
        .collect();
    let active = harness::active_revision(&app.data, &job.project_id)
        .ok()
        .flatten()
        .map(|r| r.id);
    Ok(Json(json!({"job": job, "diffs": diffs, "active": active})))
}
pub(crate) async fn cancel_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let handle = app.harness_jobs.running.lock().unwrap().remove(&id);
    if let Some(handle) = handle {
        handle.abort();
    }
    let job = update_job(&app.data, &id, |j| {
        if matches!(j.status.as_str(), "running" | "queued") {
            j.status = "cancelled".into();
            j.events.push("cancelled".into());
        }
    })
    .map_err(crate::not_found)?;
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("allpaka-harness").join(&id));
    Ok(Json(json!(job)))
}
pub(crate) async fn resume_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(crate::not_found)?;
    if app.harness_jobs.running.lock().unwrap().contains_key(&id) || job.status == "completed" {
        return Err(bad("Job is running or already completed"));
    }
    spawn(app.clone(), id);
    Ok(Json(json!({"resumed": true})))
}
/// Make one task's result (`{"task": index}`, default the last task with a
/// new revision) the project's active harness.
pub(crate) async fn adopt_api(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(crate::not_found)?;
    let result = match body.get("task").and_then(Value::as_u64) {
        Some(index) => job.results.iter().find(|r| r.index as u64 == index),
        None => job.results.iter().rev().find(|r| r.revision.is_some()),
    };
    if result.is_some_and(|r| {
        r.outcome
            .validation
            .as_ref()
            .is_some_and(|validation| validation.status == ValidationStatus::RegressionObserved)
    }) && body.get("override_regression").and_then(Value::as_bool) != Some(true)
    {
        return Err(error(
            StatusCode::CONFLICT,
            "Fresh validation observed a regression. Adopt only after reviewing the comparisons; to override, set override_regression=true.",
        ));
    }
    let revision = result
        .and_then(|r| r.revision.clone())
        .ok_or_else(|| bad("The job kept the current harness; nothing to adopt"))?;
    harness::activate(&app.data, &job.project_id, Some(&revision)).map_err(bad)?;
    Ok(Json(json!({"active": revision})))
}
