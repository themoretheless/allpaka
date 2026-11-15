//! Revocable, expirable, SHA-256 capability-token read-only LiveShare links.

use crate::ext::{get_session, random_hex_token};
use crate::{bad_request, forbidden, not_found, state_file, storage, ApiResult, App, Session};
use anyhow::{bail, Result};
use axum::extract::{Path as AxumPath, State};
use axum::response::Html;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MAX_SHARES_TOTAL: usize = 512;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ShareEntry {
    pub id: String,
    pub session_id: String,
    pub token_sha256: String,
    pub label: String,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
    #[serde(default)]
    pub revoked: bool,
}

fn validate_share(s: &ShareEntry) -> Result<()> {
    if !storage::valid_id(&s.id) || !storage::valid_id(&s.session_id) {
        bail!("Invalid share or session ID");
    }
    if s.token_sha256.len() != 64 || !s.token_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("Invalid share token digest");
    }
    Ok(())
}

fn load_shares(app: &App) -> Result<Vec<ShareEntry>> {
    state_file::load_list(app.data.as_path(), "shares", "share", validate_share)
}

fn save_shares(app: &App, list: &[ShareEntry]) -> Result<()> {
    state_file::save_list(app.data.as_path(), "shares", list)
}

fn constant_time_hex_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut diff = ab.len() ^ bb.len();
    let n = ab.len().min(bb.len());
    for i in 0..n {
        diff |= (ab[i] ^ bb[i]) as usize;
    }
    diff == 0
}

fn resolve_active_share(app: &App, token: &str) -> Result<ShareEntry, crate::ApiError> {
    if token.len() < 16 || token.len() > 160 {
        return Err(forbidden("Share link is invalid, expired, or revoked"));
    }
    let digest = storage::hex_sha256(token.as_bytes());
    let list = load_shares(app).map_err(crate::internal)?;
    let now = storage::now_ms();
    for item in list {
        if constant_time_hex_eq(&item.token_sha256, &digest) {
            if item.revoked || item.expires_at_ms.is_some_and(|exp| now > exp) {
                return Err(forbidden("Share link is invalid, expired, or revoked"));
            }
            return Ok(item);
        }
    }
    Err(forbidden("Share link is invalid, expired, or revoked"))
}

fn sanitize_session_snapshot(session: &Session, share: &ShareEntry) -> Value {
    let messages: Vec<Value> = session
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
            })
        })
        .collect();
    json!({
        "share": {
            "id": share.id,
            "label": share.label,
            "created_at_ms": share.created_at_ms,
            "expires_at_ms": share.expires_at_ms,
            "read_only": true,
        },
        "session": {
            "id": session.id,
            "title": session.title,
            "mode": session.settings.mode,
            "model": session.settings.model,
            "messages": messages,
            "plan": session.plan,
        }
    })
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub(crate) async fn list_session_shares_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let shares: Vec<ShareEntry> = load_shares(&app)
        .map_err(crate::internal)?
        .into_iter()
        .filter(|s| s.session_id == id)
        .collect();
    Ok(Json(json!({ "shares": shares })))
}

pub(crate) async fn create_session_share_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    body: Option<Json<Value>>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let payload = body.map(|Json(v)| v).unwrap_or_else(|| json!({}));
    let mut list = load_shares(&app).map_err(crate::internal)?;
    if list.len() >= MAX_SHARES_TOTAL {
        return Err(bad_request("Maximum share links reached"));
    }
    let label = payload["label"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("Read-only snapshot")
        .to_owned();
    let now = storage::now_ms();
    let expires_at_ms = payload["ttl_secs"]
        .as_u64()
        .filter(|&ttl| ttl > 0)
        .map(|ttl| now.saturating_add(ttl.min(365 * 24 * 3600) * 1000));
    let token = format!("shr_{}", random_hex_token(32));
    let entry = ShareEntry {
        id: format!("shr-{}", crate::evaluation::new_id()),
        session_id: id,
        token_sha256: storage::hex_sha256(token.as_bytes()),
        label,
        created_at_ms: now,
        expires_at_ms,
        revoked: false,
    };
    list.push(entry.clone());
    save_shares(&app, &list).map_err(crate::internal)?;
    Ok(Json(json!({
        "share": entry,
        "token": token,
        "url": format!("/share/{token}"),
        "api_url": format!("/api/shares/{token}"),
    })))
}

pub(crate) async fn revoke_session_share_api(
    State(app): State<App>,
    AxumPath((id, sid)): AxumPath<(String, String)>,
) -> ApiResult<Value> {
    let _ = get_session(&app, &id)?;
    let mut list = load_shares(&app).map_err(crate::internal)?;
    let Some(item) = list.iter_mut().find(|s| s.id == sid && s.session_id == id) else {
        return Err(not_found("Share not found"));
    };
    item.revoked = true;
    save_shares(&app, &list).map_err(crate::internal)?;
    Ok(Json(json!({ "revoked": sid })))
}

pub(crate) async fn get_public_share_api(
    State(app): State<App>,
    AxumPath(token): AxumPath<String>,
) -> ApiResult<Value> {
    let share = resolve_active_share(&app, &token)?;
    let shared = get_session(&app, &share.session_id)?;
    let guard = shared.lock().unwrap();
    Ok(Json(sanitize_session_snapshot(&guard, &share)))
}

pub(crate) async fn get_public_share_html(
    State(app): State<App>,
    AxumPath(token): AxumPath<String>,
) -> Result<Html<String>, crate::ApiError> {
    let share = resolve_active_share(&app, &token)?;
    let shared = get_session(&app, &share.session_id)?;
    let snap = {
        let guard = shared.lock().unwrap();
        sanitize_session_snapshot(&guard, &share)
    };
    let title = html_escape(
        snap["session"]["title"]
            .as_str()
            .unwrap_or("Shared session"),
    );
    let model = html_escape(snap["session"]["model"].as_str().unwrap_or(""));
    let label = html_escape(&share.label);

    let mut body_items = String::new();
    if let Some(msgs) = snap["session"]["messages"].as_array() {
        for m in msgs {
            let role = html_escape(m["role"].as_str().unwrap_or("message"));
            let content = html_escape(m["content"].as_str().unwrap_or(""));
            body_items.push_str(&format!(
                "<article class=\"msg {}\"><header><strong>{}</strong></header><pre>{}</pre></article>",
                role, role, content
            ));
        }
    }

    let page = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{title} · Read-Only Share</title><link rel=\"stylesheet\" href=\"/style.css\"></head><body><main class=\"ops-panel\"><h1>{title}</h1><p class=\"muted\">Read-only LiveShare ({label}) · Model: {model}</p><section id=\"messages\">{body_items}</section></main></body></html>"
    );
    Ok(Html(page))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Project, Root};
    use crate::ext::test_app;

    #[test]
    fn capability_token_is_hashed_and_revocation_forbids_access() {
        let root_dir =
            std::env::temp_dir().join(format!("allpaka-share-test-{}", storage::now_ms()));
        let ws = root_dir.join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ws = ws.canonicalize().unwrap();
        let app = test_app(
            &root_dir.join("data"),
            &ws,
            vec![Project {
                id: "default".into(),
                name: "default".into(),
                roots: vec![Root {
                    alias: "workspace".into(),
                    path: ws.clone(),
                    writable: false,
                    repository: false,
                    ssh: None,
                }],
                instructions: String::new(),
            }],
        );
        let token = format!("shr_{}", random_hex_token(32));
        let digest = storage::hex_sha256(token.as_bytes());
        assert_ne!(token, digest);
        save_shares(
            &app,
            &[ShareEntry {
                id: "shr-1".into(),
                session_id: "sess1".into(),
                token_sha256: digest,
                label: "Audit".into(),
                created_at_ms: storage::now_ms(),
                expires_at_ms: None,
                revoked: false,
            }],
        )
        .unwrap();

        assert!(resolve_active_share(&app, &token).is_ok());

        let mut list = load_shares(&app).unwrap();
        list[0].revoked = true;
        save_shares(&app, &list).unwrap();
        assert!(resolve_active_share(&app, &token).is_err());
        let _ = std::fs::remove_dir_all(&root_dir);
    }
}
