//! Session, plan, background-task and project handlers, with the settings
//! and identifier validation they share.

use crate::*;

pub(crate) async fn config(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"credential_storage":credentials::AVAILABLE,"providers":app.providers.lock().unwrap().iter().map(|p|p.public()).collect::<Vec<_>>(),"workspace":app.workspace.display().to_string(),"projects":*app.projects.lock().unwrap()}),
    )
}
pub(crate) fn validate_settings(s: &Settings, app: &App) -> Result<()> {
    if let Some(selection) = &s.guardrails {
        if matches!(s.mode, Mode::Swarm)
            && s.swarm
                .members
                .iter()
                .any(|member| !member.worker.is_empty())
        {
            bail!("Guardrails for remote Swarm workers are not yet supported");
        }
        selection.prepare(&guardrail_policies::Store::open(&app.data)?)?;
    }
    if !(4096..=1000000).contains(&s.compact_threshold) {
        bail!("Compaction threshold must be between 4096 and 1000000 estimated tokens");
    }
    let output_limit = if s.provider == "deepseek"
        && ["deepseek-v4-pro", "deepseek-flash", "deepseek-v4-flash"].contains(&s.model.as_str())
    {
        393216
    } else {
        131072
    };
    if !(256..=output_limit).contains(&s.max_output_tokens) {
        bail!("Output token limit must be between 256 and {output_limit}; the selected model may have a lower maximum");
    }
    if !app
        .projects
        .lock()
        .unwrap()
        .iter()
        .any(|p| p.id == s.project_id)
    {
        bail!("Unknown project");
    }
    if s.model.trim().is_empty() || s.model.len() > 200 {
        bail!("Choose a model ID");
    }
    if !(1..=50).contains(&s.max_steps) {
        bail!("Step limit must be between 1 and 50");
    }
    if s.mode == Mode::Swarm {
        swarm::validate(&s.swarm)?;
        let providers = app.providers.lock().unwrap();
        for member in &s.swarm.members {
            if !member.worker.is_empty() {
                continue;
            }
            let member_provider = providers
                .iter()
                .find(|p| p.id == member.provider)
                .with_context(|| format!("Участник {}: неизвестный провайдер", member.label))?;
            if !member_provider.public().configured {
                bail!(
                    "Участник {}: задайте {} или введите ключ в «Подключениях»",
                    member.label,
                    member_provider.env
                );
            }
        }
        let synthesis = if s.swarm.synthesis_provider.trim().is_empty() {
            s.provider.clone()
        } else {
            s.swarm.synthesis_provider.clone()
        };
        let synthesis_provider = providers
            .iter()
            .find(|p| p.id == synthesis)
            .with_context(|| format!("Синтез: неизвестный провайдер {synthesis}"))?;
        if !synthesis_provider.public().configured {
            bail!(
                "Синтез: задайте {} или введите ключ в «Подключениях»",
                synthesis_provider.env
            );
        }
    }
    let providers = app.providers.lock().unwrap();
    let p = providers
        .iter()
        .find(|p| p.id == s.provider)
        .context("Unknown provider")?;
    if !p.public().configured {
        bail!("Set {} or enter the API key in Connections", p.env);
    }
    Ok(())
}
pub(crate) fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() < 80 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}
pub(crate) async fn create_session(
    State(app): State<App>,
    Json(settings): Json<Settings>,
) -> ApiResult<Value> {
    let app_for_attach = app.clone();
    let (id, session) = tokio::task::spawn_blocking(move || -> std::result::Result<_, ApiError> {
        validate_settings(&settings, &app).map_err(crate::bad_request)?;
        let id = format!(
            "{:x}-{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        );
        let session = Session::new(id.clone(), settings);
        save(&app, &session).map_err(crate::internal)?;
        Ok((id, session))
    })
    .await
    .map_err(crate::internal)??;
    attach(&app_for_attach, session);
    Ok(Json(json!({"id":id})))
}
pub(crate) fn attach(app: &App, session: Session) {
    let id = session.id.clone();
    let shared = Arc::new(Mutex::new(session));
    let (tx, rx) = mpsc::channel(64);
    app.sessions
        .lock()
        .unwrap()
        .insert(id, (shared.clone(), tx));
    tokio::spawn(actor(app.clone(), shared, rx));
}
#[derive(Deserialize)]
pub(crate) struct BranchRequest {
    pub(crate) message_count: usize,
}
pub(crate) async fn branch_session(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(request): Json<BranchRequest>,
) -> ApiResult<Value> {
    let source = {
        let registry = app.sessions.lock().unwrap();
        let (source, _) = registry
            .get(&id)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
        let snapshot = source.lock().unwrap().clone();
        snapshot
    };
    if source.status == SessionStatus::Running {
        return Err(error(
            StatusCode::CONFLICT,
            "Stop generation before branching",
        ));
    }
    let n = request.message_count;
    if n > source.messages.len()
        || (n > 0 && {
            let last = &source.messages[n - 1];
            last.role != "assistant" || !last.tool_calls.is_empty() || last.truncated
        })
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Branch at a completed assistant response or before the first message",
        ));
    }
    let new_id = format!(
        "{:x}-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    );
    let mut branch = Session::new(new_id.clone(), source.settings);
    branch.title = format!("Ветка · {}", source.title);
    branch.messages = source.messages[..n].to_vec();
    branch.bookmarks = source
        .bookmarks
        .into_iter()
        .filter(|m| m.message_index < n)
        .collect();
    branch.parent = Some(BranchOrigin {
        session_id: id,
        message_count: n,
    });
    let save_app = app.clone();
    let branch = tokio::task::spawn_blocking(move || -> std::result::Result<_, ApiError> {
        save(&save_app, &branch).map_err(crate::internal)?;
        Ok(branch)
    })
    .await
    .map_err(crate::internal)??;
    attach(&app, branch);
    Ok(Json(json!({"id":new_id})))
}
pub(crate) async fn session(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let shared = {
        let sessions = app.sessions.lock().unwrap();
        let (s, _) = sessions
            .get(&id)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
        s.clone()
    };
    let (mut value, stats) = {
        let session = shared.lock().unwrap();
        let val = serde_json::to_value(&*session).map_err(crate::internal)?;
        let st = compact::statistics(&session);
        (val, st)
    };
    value["context_stats"] = stats;
    value["background"] = app.background.list(&id);
    Ok(Json(value))
}
pub(crate) async fn background_action(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(args): Json<Value>,
) -> ApiResult<Value> {
    let settings = {
        let sessions = app.sessions.lock().unwrap();
        let (shared, _) = sessions
            .get(&id)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
        let session = shared.lock().unwrap();
        if args["action"] == "start" && session.folder != HistoryFolder::Active {
            return Err(error(
                StatusCode::CONFLICT,
                "Restore the conversation before starting background work",
            ));
        }
        session.settings.clone()
    };
    if args["action"] == "start"
        && (!matches!(settings.mode, Mode::Auto | Mode::Goal)
            || !plugins::tool_unlocked(&app.plugins, "run_command"))
    {
        return Err(error(
            StatusCode::FORBIDDEN,
            "Background commands require Auto/Goal mode and the command plugin",
        ));
    }
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == settings.project_id)
        .cloned()
        .ok_or_else(|| error(StatusCode::BAD_REQUEST, "Project not found"))?;
    let root = project
        .roots
        .first()
        .ok_or_else(|| error(StatusCode::BAD_REQUEST, "No project root available"))?;
    execute_background(&app, &id, &root.path, &args)
        .await
        .map(Json)
        .map_err(crate::bad_request)
}

pub(crate) async fn execute_background(
    app: &App,
    owner: &str,
    cwd: &std::path::Path,
    args: &Value,
) -> Result<Value> {
    let epoch = app
        .sessions
        .lock()
        .unwrap()
        .get(owner)
        .map(|(session, _)| session.lock().unwrap().background_continuation_epoch)
        .context("Conversation not found")?;
    let result = app.background.execute(owner, cwd, args).await?;
    if args["action"] == "start" {
        if let (Some(text), Some(id)) = (args["follow_up"].as_str(), result["id"].as_str()) {
            let app = app.clone();
            let owner = owner.to_string();
            let id = id.to_string();
            let text = text.to_string();
            let cwd = cwd.to_path_buf();
            tokio::spawn(async move {
                loop {
                    let waited = app
                        .background
                        .execute(
                            &owner,
                            &cwd,
                            &json!({"action":"wait","task_ids":[id],"wait_seconds":60}),
                        )
                        .await;
                    let Ok(waited) = waited else {
                        break;
                    };
                    if waited["tasks"][0]["status"] != "running" {
                        let sender = app
                            .sessions
                            .lock()
                            .unwrap()
                            .get(&owner)
                            .map(|(_, sender)| sender.clone());
                        if let Some(sender) = sender {
                            let _ = sender
                                .send(Action {
                                    wake_epoch: Some(epoch),
                                    reply: None,
                                    images: vec![],
                                    kind: ActionKind::BackgroundDone,
                                    text: format!(
                                        "Background task {id} terminated with status {}. {text}",
                                        waited["tasks"][0]["status"].as_str().unwrap_or("unknown")
                                    ),
                                    settings: None,
                                })
                                .await;
                        }
                        break;
                    }
                }
            });
        }
    }
    Ok(result)
}

pub(crate) async fn delete_session(
    State(app): State<App>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    let folder = {
        let sessions = app.sessions.lock().unwrap();
        let Some((shared, _)) = sessions.get(&id) else {
            return Err(error(StatusCode::NOT_FOUND, "Conversation not found"));
        };
        let folder = shared.lock().unwrap().folder;
        folder
    };
    if folder != HistoryFolder::Trash {
        return Err(error(
            StatusCode::CONFLICT,
            "Only conversations in Trash can be permanently deleted",
        ));
    }
    app.background.cancel_owned(&id);
    app.sessions.lock().unwrap().remove(&id);
    let path = app.data.join(format!("{id}.json"));
    let _ = tokio::task::spawn_blocking(move || std::fs::remove_file(path)).await;
    Ok(Json(json!({"deleted": id})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlanInput {
    pub(crate) steps: Vec<PlanItem>,
    #[serde(default)]
    pub(crate) base_revision: Option<u64>,
    #[serde(default)]
    pub(crate) allow_reopen: bool,
}
pub(crate) async fn read_plan(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let sessions = app.sessions.lock().unwrap();
    let (shared, _) = sessions
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
    let state = shared.lock().unwrap();
    Ok(Json(
        json!({"session_id":state.id,"goal":state.goal,"revision":state.plan_revision,"steps":state.plan,"checkpoints":state.plan_checkpoints,"history_truncated":state.plan_checkpoints.first().is_some_and(|entry|entry.revision>1),"evidence_basis":"reported","reported_completion_ready":plan::reported_ready(&state),"automatic_replay":false}),
    ))
}
pub(crate) async fn save_plan(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(input): Json<PlanInput>,
) -> ApiResult<Value> {
    crate::blocking(move || {
        if input.steps.len() > 30 {
            return Err(error(StatusCode::BAD_REQUEST, "Too many plan steps"));
        }
        for step in &input.steps {
            if step.title.trim().is_empty() || step.title.chars().count() > 300 {
                return Err(error(StatusCode::BAD_REQUEST, "Invalid plan step title"));
            }
        }
        let sessions = app.sessions.lock().unwrap();
        let (shared, _) = sessions
            .get(&id)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
        let mut state = shared.lock().unwrap();
        if state.folder != HistoryFolder::Active {
            return Err(error(
                StatusCode::CONFLICT,
                "Conversation is archived or in trash; restore it first",
            ));
        }
        let mut next = state.clone();
        plan::update(
            &mut next,
            input.steps,
            input.base_revision,
            "user",
            input.allow_reopen,
        )
        .map_err(|e| error(StatusCode::CONFLICT, e))?;
        save(&app, &next).map_err(crate::internal)?;
        *state = next;
        Ok(Json(
            json!({"updated": true,"revision":state.plan_revision,"steps":state.plan,"evidence_basis":"reported"}),
        ))
    })
    .await
}
pub(crate) async fn action(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(mut action): Json<Action>,
) -> ApiResult<Value> {
    if matches!(action.kind, ActionKind::Rename | ActionKind::Move) {
        return crate::blocking(move || {
            let sessions = app.sessions.lock().unwrap();
            let (shared, _) = sessions
                .get(&id)
                .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
            let mut state = shared.lock().unwrap();
            if action.kind == ActionKind::Move && state.status == SessionStatus::Running {
                return Err(error(
                    StatusCode::CONFLICT,
                    "Stop generation before moving the conversation",
                ));
            }
            if action.kind == ActionKind::Rename {
                let title = action.text.trim();
                if title.is_empty() || title.chars().count() > 100 {
                    return Err(error(
                        StatusCode::BAD_REQUEST,
                        "Title must be 1–100 characters",
                    ));
                }
                state.title = title.to_string();
            } else {
                state.folder = match action.text.as_str() {
                    "active" => HistoryFolder::Active,
                    "archived" => HistoryFolder::Archived,
                    "trash" => HistoryFolder::Trash,
                    _ => return Err(error(StatusCode::BAD_REQUEST, "Unknown history folder")),
                };
            }
            save(&app, &state).map_err(crate::internal)?;
            Ok(Json(json!({"accepted":true})))
        })
        .await;
    }
    if matches!(
        action.kind,
        ActionKind::Send | ActionKind::SendNow | ActionKind::Steer
    ) && (action.text.trim().is_empty() || action.text.len() > 131072)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Message must contain 1–131072 bytes",
        ));
    }
    if action.kind == ActionKind::RetryMember
        && (action.text.trim().is_empty() || action.text.chars().count() > 60)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Name the swarm member to retry (1–60 characters)",
        ));
    }
    if let Some(settings) = &action.settings {
        validate_settings(settings, &app).map_err(crate::bad_request)?;
    }
    let (current, folder) = {
        let sessions = app.sessions.lock().unwrap();
        let (s, _) = sessions
            .get(&id)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
        let state = s.lock().unwrap();
        (state.settings.clone(), state.folder)
    };
    if folder != HistoryFolder::Active
        && !matches!(action.kind, ActionKind::Stop | ActionKind::ClearQueue)
    {
        return Err(error(
            StatusCode::CONFLICT,
            "Conversation is archived or in trash; restore it first",
        ));
    }
    let chosen = action.settings.as_ref().unwrap_or(&current);
    if chosen.project_id != current.project_id {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Start a new chat to change project",
        ));
    }
    context::validate_images(&action.images, &chosen.provider).map_err(crate::bad_request)?;
    if action.kind == ActionKind::Steer && !action.images.is_empty() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Use Send now or Queue for images",
        ));
    }
    let tx = app
        .sessions
        .lock()
        .unwrap()
        .get(&id)
        .map(|(_, tx)| tx.clone())
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
    let (reply, receipt) = tokio::sync::oneshot::channel();
    action.reply = Some(reply);
    tx.try_send(action)
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Control queue is full"))?;
    receipt
        .await
        .map_err(|_| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Conversation worker stopped",
            )
        })?
        .map_err(|message| error(StatusCode::TOO_MANY_REQUESTS, message))?;
    Ok(Json(json!({"accepted":true})))
}

pub(crate) async fn save_project(
    State(app): State<App>,
    Json(mut project): Json<context::Project>,
) -> ApiResult<Value> {
    crate::blocking(move || {
        project.validate(&app.data).map_err(crate::bad_request)?;
        if project.id.is_empty() {
            project.id = format!(
                "{:x}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
        }
        if project.id != "default" && !valid_id(&project.id) {
            return Err(error(StatusCode::BAD_REQUEST, "Invalid project ID"));
        }
        let mut projects = app.projects.lock().unwrap();
        let mut updated = projects.clone();
        if let Some(p) = updated.iter_mut().find(|p| p.id == project.id) {
            *p = project.clone();
        } else {
            updated.push(project.clone());
        }
        state_file::save_list(app.data.as_path(), "projects", &updated).map_err(crate::internal)?;
        *projects = updated;
        Ok(Json(json!(project)))
    })
    .await
}
