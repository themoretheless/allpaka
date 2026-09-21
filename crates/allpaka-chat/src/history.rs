use crate::{types::*, *};
use axum::extract::Query;
use std::collections::HashSet;

#[derive(Default, Deserialize)]
pub struct Search {
    #[serde(default)]
    q: String,
    project: Option<String>,
    #[serde(default)]
    folder: HistoryFolder,
}
pub async fn list(State(app): State<App>, Query(search): Query<Search>) -> ApiResult<Value> {
    if search.q.len() > 1000 {
        return Err(error(StatusCode::BAD_REQUEST, "Search query too long"));
    }
    let query = search.q.to_lowercase();
    let mut rows = Vec::new();
    for (session, _) in app.sessions.lock().unwrap().values() {
        let s = session.lock().unwrap();
        if s.folder != search.folder
            || search
                .project
                .as_ref()
                .is_some_and(|p| p != &s.settings.project_id)
        {
            continue;
        }
        let matched = if query.is_empty() {
            None
        } else {
            s.messages
                .iter()
                .find(|m| m.content.to_lowercase().contains(&query))
        };
        if !query.is_empty() && !s.title.to_lowercase().contains(&query) && matched.is_none() {
            continue;
        }
        rows.push(json!({"id":s.id,"title":s.title,"status":s.status,"provider":s.settings.provider,"project_id":s.settings.project_id,"folder":s.folder,"match_preview":matched.map(|m|m.content.chars().take(160).collect::<String>())}));
    }
    rows.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    Ok(Json(json!(rows)))
}
#[derive(Deserialize)]
pub struct Import {
    session: Session,
    project_id: String,
}
fn validate_messages(messages: &[Message]) -> Result<()> {
    if messages.len() > 10000 {
        bail!("Too many messages");
    }
    let mut pending = HashSet::new();
    for m in messages {
        if !["user", "assistant", "tool"].contains(&m.role.as_str()) {
            bail!("Unsupported message role");
        }
        context::validate_images(&m.images, "import")?;
        if m.role != "tool" && !pending.is_empty() {
            bail!("Tool calls must be followed by their results");
        }
        if m.role == "tool" {
            let id = m.tool_call_id.as_ref().context("Tool result has no ID")?;
            if !pending.remove(id) {
                bail!("Orphan or duplicate tool result");
            }
        } else if m.tool_call_id.is_some() {
            bail!("Unexpected tool result ID");
        }
        if !m.tool_calls.is_empty() && m.role != "assistant" {
            bail!("Only assistant messages may call tools");
        }
        for call in &m.tool_calls {
            let id = call["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .context("Tool call has no ID")?;
            if !pending.insert(id.to_owned()) {
                bail!("Duplicate tool call ID");
            }
            serde_json::from_str::<Value>(
                call["function"]["arguments"]
                    .as_str()
                    .context("Missing tool arguments")?,
            )?;
            if call["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .is_none()
            {
                bail!("Missing tool name");
            }
        }
    }
    if !pending.is_empty() {
        bail!("Unfinished tool exchange in import");
    }
    Ok(())
}
pub async fn import(State(app): State<App>, Json(input): Json<Import>) -> ApiResult<Value> {
    if !app
        .projects
        .lock()
        .unwrap()
        .iter()
        .any(|p| p.id == input.project_id)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Unknown destination project",
        ));
    }
    validate_messages(&input.session.messages).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    if input.session.title.chars().count() > 100 || input.session.title.trim().is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "Invalid title"));
    }
    let id = format!(
        "{:x}-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    );
    let mut settings = input.session.settings;
    settings.project_id = input.project_id;
    if settings.model.trim().is_empty() || settings.model.len() > 200 {
        return Err(error(StatusCode::BAD_REQUEST, "Invalid imported model ID"));
    }
    // Imported settings never enable writes or execute queued actions.
    settings.allow_writes = false;
    settings.mode = Mode::Chat;
    settings.max_steps = settings.max_steps.clamp(1, 50);
    settings.max_output_tokens = settings.max_output_tokens.clamp(256, 131072);
    settings.compact_threshold = settings.compact_threshold.clamp(4096, 1000000);
    if !app
        .providers
        .lock()
        .unwrap()
        .iter()
        .any(|p| p.id == settings.provider)
    {
        return Err(error(StatusCode::BAD_REQUEST, "Unknown imported provider"));
    }
    let mut session = Session::new(id.clone(), settings);
    session.title = input.session.title;
    session.messages = input.session.messages;
    // Restore original history; imported summaries are not authoritative replacements.
    session.notice = Some(
        "Импортирована копия. Выбран режим Chat; очередь и разрешения записи не перенесены.".into(),
    );
    save(&app, &session).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    attach(&app, session);
    Ok(Json(json!({"id":id})))
}
