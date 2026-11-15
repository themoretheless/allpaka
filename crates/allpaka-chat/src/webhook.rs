//! HMAC-SHA256 authenticated inbound webhooks (`POST /api/webhooks/trigger/:id`)
//! spawning isolated `scheduled_origin` sessions.

use crate::ext::{default_settings, random_hex_token, spawn_scheduled_session};
use crate::{bad_request, not_found, state_file, storage, unauthorized, ApiResult, App};
use anyhow::{bail, Result};
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MAX_WEBHOOKS: usize = 64;
const MAX_TRIGGER_BODY: usize = 256 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WebhookEntry {
    pub id: String,
    pub name: String,
    pub project_id: String,
    pub secret: String,
    pub prompt_template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_triggered_ms: Option<u64>,
    #[serde(default)]
    pub trigger_count: u64,
}

fn default_true() -> bool {
    true
}

/// RFC 2104 HMAC-SHA256.
pub(crate) fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k_pad = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        k_pad[..32].copy_from_slice(&hashed);
    } else {
        k_pad[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_pad[i];
        opad[i] ^= k_pad[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_hash);
    let out = outer.finalize();
    let mut res = [0u8; 32];
    res.copy_from_slice(&out);
    res
}

fn decode_hex_32(s: &str) -> Option<[u8; 32]> {
    let hex = s.trim();
    let hex = hex
        .strip_prefix("sha256=")
        .or_else(|| hex.strip_prefix("SHA256="))
        .unwrap_or(hex);
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let pair = std::str::from_utf8(&bytes[i * 2..i * 2 + 2]).ok()?;
        *slot = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

pub(crate) fn verify_signature(secret: &str, body: &[u8], header_val: &str) -> bool {
    let Some(provided) = decode_hex_32(header_val) else {
        return false;
    };
    let expected = hmac_sha256(secret.as_bytes(), body);
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= expected[i] ^ provided[i];
    }
    diff == 0
}

fn validate_webhook(w: &WebhookEntry) -> Result<()> {
    if !storage::valid_id(&w.id) {
        bail!("Invalid webhook id");
    }
    if w.name.trim().is_empty() || w.name.chars().count() > 120 {
        bail!("Webhook name must be 1-120 characters");
    }
    if w.secret.len() < 16 {
        bail!("Webhook secret must be at least 16 characters");
    }
    if w.prompt_template.trim().is_empty() || w.prompt_template.len() > 16_000 {
        bail!("Webhook prompt_template must be 1-16000 bytes");
    }
    Ok(())
}

fn load_webhooks(app: &App) -> Result<Vec<WebhookEntry>> {
    state_file::load_list(app.data.as_path(), "webhooks", "webhook", validate_webhook)
}

fn save_webhooks(app: &App, list: &[WebhookEntry]) -> Result<()> {
    state_file::save_list(app.data.as_path(), "webhooks", list)
}

fn redacted(w: &WebhookEntry) -> Value {
    json!({
        "id": w.id,
        "name": w.name,
        "project_id": w.project_id,
        "prompt_template": w.prompt_template,
        "profile": w.profile,
        "enabled": w.enabled,
        "created_at_ms": w.created_at_ms,
        "last_triggered_ms": w.last_triggered_ms,
        "trigger_count": w.trigger_count,
        "has_secret": true,
    })
}

pub(crate) async fn list_api(State(app): State<App>) -> ApiResult<Value> {
    let items: Vec<Value> = load_webhooks(&app)
        .map_err(crate::internal)?
        .iter()
        .map(redacted)
        .collect();
    Ok(Json(json!({ "webhooks": items })))
}

pub(crate) async fn create_api(
    State(app): State<App>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    let mut list = load_webhooks(&app).map_err(crate::internal)?;
    if list.len() >= MAX_WEBHOOKS {
        return Err(bad_request(format!(
            "Maximum {MAX_WEBHOOKS} webhooks allowed"
        )));
    }
    let name = body["name"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad_request("Missing `name`"))?
        .to_owned();
    let project_id = body["project_id"].as_str().unwrap_or("default").to_owned();
    let prompt_template = body["prompt_template"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| bad_request("Missing `prompt_template`"))?
        .to_owned();
    let secret = format!("whsec_{}", random_hex_token(32));
    let entry = WebhookEntry {
        id: format!("whk-{}", crate::evaluation::new_id()),
        name,
        project_id,
        secret: secret.clone(),
        prompt_template,
        profile: body["profile"].as_str().map(str::to_owned),
        enabled: body["enabled"].as_bool().unwrap_or(true),
        created_at_ms: storage::now_ms(),
        last_triggered_ms: None,
        trigger_count: 0,
    };
    validate_webhook(&entry).map_err(bad_request)?;
    list.push(entry.clone());
    save_webhooks(&app, &list).map_err(crate::internal)?;
    Ok(Json(json!({
        "webhook": redacted(&entry),
        "secret": secret,
    })))
}

pub(crate) async fn delete_api(
    State(app): State<App>,
    AxumPath(wid): AxumPath<String>,
) -> ApiResult<Value> {
    let mut list = load_webhooks(&app).map_err(crate::internal)?;
    let before = list.len();
    list.retain(|w| w.id != wid);
    if list.len() == before {
        return Err(not_found("Webhook not found"));
    }
    save_webhooks(&app, &list).map_err(crate::internal)?;
    Ok(Json(json!({ "deleted": wid })))
}

pub(crate) async fn trigger_api(
    State(app): State<App>,
    AxumPath(wid): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Value> {
    if body.len() > MAX_TRIGGER_BODY {
        return Err(bad_request("Webhook payload exceeds 256 KiB"));
    }
    let sig = headers
        .get("x-allpaka-signature-256")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| unauthorized("Missing X-Allpaka-Signature-256 header"))?;

    let mut list = load_webhooks(&app).map_err(crate::internal)?;
    let pos = list
        .iter()
        .position(|w| w.id == wid)
        .ok_or_else(|| not_found("Webhook not found"))?;
    if !list[pos].enabled {
        return Err(unauthorized("Webhook is disabled"));
    }
    if !verify_signature(&list[pos].secret, &body, sig) {
        return Err(unauthorized("Invalid X-Allpaka-Signature-256 signature"));
    }

    let payload_str = String::from_utf8_lossy(&body).into_owned();
    let prompt = if list[pos].prompt_template.contains("{{payload}}") {
        list[pos]
            .prompt_template
            .replace("{{payload}}", &payload_str)
    } else {
        format!("{}\n\nPayload:\n{}", list[pos].prompt_template, payload_str)
    };

    let mut settings = default_settings(&app, &list[pos].project_id);
    if let Some(pid) = &list[pos].profile {
        if let Ok(prof) = crate::bundles::resolve_profile(&app, pid) {
            crate::bundles::apply_profile(&prof, &mut settings);
        }
    }
    let session_id = spawn_scheduled_session(
        &app,
        format!("Webhook: {}", list[pos].name),
        settings,
        prompt,
    )
    .map_err(crate::internal)?;

    list[pos].last_triggered_ms = Some(storage::now_ms());
    list[pos].trigger_count += 1;
    save_webhooks(&app, &list).map_err(crate::internal)?;

    Ok(Json(json!({
        "accepted": true,
        "webhook_id": wid,
        "session_id": session_id,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_sha256_matches_rfc4231_test_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert!(verify_signature(
            "Jefe",
            b"what do ya want for nothing?",
            "sha256=5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        ));
        assert!(!verify_signature(
            "Jefe",
            b"tampered",
            "sha256=5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        ));
    }
}
