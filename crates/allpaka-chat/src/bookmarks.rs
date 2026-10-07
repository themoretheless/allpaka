//! Conversation navigation metadata; never part of model context.
use crate::{types::*, App};
use anyhow::{bail, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;

pub(crate) fn validate(session: &Session) -> Result<()> {
    if session.bookmarks.len() > 200 {
        bail!("Conversation bookmark limit is 200");
    }
    let mut indices = HashSet::new();
    for mark in &session.bookmarks {
        if mark.label.trim().is_empty()
            || mark.label.len() > 200
            || !indices.insert(mark.message_index)
            || !session
                .messages
                .get(mark.message_index)
                .is_some_and(|m| matches!(m.role.as_str(), "user" | "assistant"))
        {
            bail!("Invalid conversation bookmark");
        }
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Save {
    message_index: usize,
    label: String,
}
pub(crate) async fn save(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(input): Json<Save>,
) -> crate::ApiResult<Value> {
    mutate(&app, &id, |s| {
        if let Some(mark) = s
            .bookmarks
            .iter_mut()
            .find(|m| m.message_index == input.message_index)
        {
            mark.label = input.label.trim().to_owned();
        } else {
            s.bookmarks.push(Bookmark {
                message_index: input.message_index,
                label: input.label.trim().to_owned(),
                created_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
                    .min(u64::MAX as u128) as u64,
            });
        }
        s.bookmarks.sort_by_key(|m| m.message_index);
    })
}
pub(crate) async fn remove(
    State(app): State<App>,
    Path((id, index)): Path<(String, usize)>,
) -> crate::ApiResult<Value> {
    mutate(&app, &id, |s| {
        s.bookmarks.retain(|m| m.message_index != index)
    })
}
fn mutate(app: &App, id: &str, update: impl FnOnce(&mut Session)) -> crate::ApiResult<Value> {
    let shared = {
        let registry = app.sessions.lock().unwrap();
        registry
            .get(id)
            .map(|(s, _)| s.clone())
            .ok_or_else(|| crate::error(StatusCode::NOT_FOUND, "Conversation not found"))?
    };
    let mut session = shared.lock().unwrap();
    let mut candidate = session.clone();
    update(&mut candidate);
    validate(&candidate).map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))?;
    crate::save(app, &candidate).map_err(|e| crate::error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    *session = candidate;
    Ok(Json(json!({"bookmarks":session.bookmarks})))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bookmark_bounds_and_tool_targets_are_rejected() {
        let settings = serde_json::from_value(json!({"provider":"local","model":"mock"})).unwrap();
        let mut session = Session::new("test".into(), settings);
        session.messages = (0..201).map(|_| Message::text("user", "content")).collect();
        session.bookmarks = (0..200)
            .map(|index| Bookmark {
                message_index: index,
                label: format!("mark {index}"),
                created_ms: 1,
            })
            .collect();
        validate(&session).unwrap();
        session.bookmarks.push(Bookmark {
            message_index: 200,
            label: "extra".into(),
            created_ms: 1,
        });
        assert!(validate(&session).is_err());
        session.bookmarks.pop();
        session.bookmarks[0].message_index = 1;
        assert!(validate(&session).is_err());
        session.bookmarks[0].message_index = 0;
        session.messages[0].role = "tool".into();
        assert!(validate(&session).is_err());
        session.messages[0].role = "user".into();
        session.bookmarks[0].label = " ".into();
        assert!(validate(&session).is_err());
        session.bookmarks[0].label = "x".repeat(201);
        assert!(validate(&session).is_err());
        session.bookmarks[0].label = " valid ".into();
        session.bookmarks[0].message_index = 201;
        assert!(validate(&session).is_err());
    }
}
