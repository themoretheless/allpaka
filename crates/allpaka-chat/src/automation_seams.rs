//! External runtime seams for browser automation (`browser_use`),
//! desktop GUI automation (`computer_use`), and Speech-to-Text (`POST /api/voice/transcribe`).

use crate::{bad_request, spill, storage, web_tools, App};
use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

const MAX_AUDIO_BYTES: usize = 6 * 1024 * 1024;
const DRIVER_TIMEOUT: Duration = Duration::from_secs(45);

pub(crate) fn browser_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "browser_use",
            "description": "Control an external headless browser automation driver (configured via ALLPAKA_BROWSER_CMD). Public URLs are SSRF-validated before navigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["navigate", "snapshot", "click", "type", "screenshot", "eval", "close"] },
                    "url": { "type": "string" },
                    "selector": { "type": "string" },
                    "text": { "type": "string" },
                    "script": { "type": "string" },
                    "timeout_ms": { "type": "integer", "minimum": 500, "maximum": 45000 }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        }
    })
}

pub(crate) fn computer_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": "Control an external desktop GUI automation driver (configured via ALLPAKA_COMPUTER_CMD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["screenshot", "click", "move", "type", "key", "scroll"] },
                    "x": { "type": "integer" },
                    "y": { "type": "integer" },
                    "button": { "type": "string", "enum": ["left", "right", "middle"] },
                    "text": { "type": "string" },
                    "key": { "type": "string" },
                    "delta_x": { "type": "integer" },
                    "delta_y": { "type": "integer" }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        }
    })
}

async fn invoke_json_driver(
    cmd: &str,
    payload: &Value,
    timeout: Duration,
    cwd: Option<&Path>,
) -> Result<Value> {
    let mut command = tokio::process::Command::new("sh");
    command
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("Failed to spawn driver `{cmd}`"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let bytes = serde_json::to_vec(payload)?;
        stdin.write_all(&bytes).await?;
        let _ = stdin.shutdown().await;
    }
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .context("Automation driver timed out")??;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        bail!(
            "Driver exited with {}: {}",
            output.status,
            if stderr.is_empty() { &stdout } else { &stderr }
        );
    }
    if let Ok(val) = serde_json::from_str::<Value>(&stdout) {
        Ok(val)
    } else {
        Ok(json!({ "ok": true, "output": stdout, "stderr": stderr }))
    }
}

pub(crate) async fn execute_browser_with_cmd(
    data: Option<&Path>,
    session_id: &str,
    cwd: Option<&Path>,
    args: &Value,
    cmd_override: Option<&str>,
) -> Result<Value> {
    let action = args["action"].as_str().context("Missing `action`")?;
    if action == "navigate" {
        let url = args["url"].as_str().context("`navigate` requires `url`")?;
        let allow_local = std::env::var("ALLPAKA_BROWSER_ALLOW_LOCAL")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        if !allow_local {
            web_tools::validate_public_url_syntax(url)?;
        }
    }

    let env_cmd = std::env::var("ALLPAKA_BROWSER_CMD").ok();
    let cmd = cmd_override
        .map(str::to_owned)
        .or_else(|| env_cmd.filter(|s| !s.trim().is_empty()));

    let Some(driver_cmd) = cmd else {
        return Ok(json!({
            "available": false,
            "backend": "none",
            "action": action,
            "error": "Browser automation driver not configured (set ALLPAKA_BROWSER_CMD)",
            "hint": "Configure ALLPAKA_BROWSER_CMD or use web_fetch for HTTP content",
        }));
    };

    let timeout = args["timeout_ms"]
        .as_u64()
        .map(|ms| Duration::from_millis(ms.clamp(500, 45_000)))
        .unwrap_or(DRIVER_TIMEOUT);
    let payload = json!({
        "session_id": session_id,
        "workspace": cwd.map(|p| p.display().to_string()),
        "args": args,
    });
    let res = invoke_json_driver(&driver_cmd, &payload, timeout, cwd).await?;
    let serialized = serde_json::to_string_pretty(&res)?;
    let (preview, spill_id, truncated, _) = spill::truncate_or_spill(
        data,
        Some(session_id),
        "browser_use",
        &serialized,
        spill::DEFAULT_INLINE_LIMIT,
    );
    if truncated {
        return Ok(json!({
            "available": true,
            "truncated": true,
            "preview": preview,
            "spill_id": spill_id,
        }));
    }
    Ok(res)
}

pub(crate) async fn execute_computer_with_cmd(
    data: Option<&Path>,
    session_id: &str,
    cwd: Option<&Path>,
    args: &Value,
    cmd_override: Option<&str>,
) -> Result<Value> {
    let action = args["action"].as_str().context("Missing `action`")?;
    let env_cmd = std::env::var("ALLPAKA_COMPUTER_CMD").ok();
    let cmd = cmd_override
        .map(str::to_owned)
        .or_else(|| env_cmd.filter(|s| !s.trim().is_empty()));

    let Some(driver_cmd) = cmd else {
        return Ok(json!({
            "available": false,
            "backend": "none",
            "action": action,
            "error": "Computer-use desktop driver not configured (set ALLPAKA_COMPUTER_CMD)",
        }));
    };

    let payload = json!({
        "session_id": session_id,
        "workspace": cwd.map(|p| p.display().to_string()),
        "args": args,
    });
    let res = invoke_json_driver(&driver_cmd, &payload, DRIVER_TIMEOUT, cwd).await?;
    let serialized = serde_json::to_string_pretty(&res)?;
    let (preview, spill_id, truncated, _) = spill::truncate_or_spill(
        data,
        Some(session_id),
        "computer_use",
        &serialized,
        spill::DEFAULT_INLINE_LIMIT,
    );
    if truncated {
        return Ok(json!({
            "available": true,
            "truncated": true,
            "preview": preview,
            "spill_id": spill_id,
        }));
    }
    Ok(res)
}

pub(crate) async fn execute_stt_with_overrides(
    audio_b64: &str,
    mime_type: Option<&str>,
    language: Option<&str>,
    cmd_override: Option<&str>,
    url_override: Option<&str>,
) -> Result<Value> {
    let bytes = storage::base64_decode(audio_b64)?;
    if bytes.is_empty() {
        bail!("Audio payload is empty");
    }
    if bytes.len() > MAX_AUDIO_BYTES {
        bail!("Audio payload exceeds {MAX_AUDIO_BYTES}-byte limit");
    }

    let env_cmd = std::env::var("ALLPAKA_STT_CMD").ok();
    let stt_cmd = cmd_override
        .map(str::to_owned)
        .or_else(|| env_cmd.filter(|s| !s.trim().is_empty()));

    if let Some(cmd) = stt_cmd {
        let temp_dir = std::env::temp_dir().join(format!("allpaka-stt-{}", storage::now_ms()));
        std::fs::create_dir_all(&temp_dir)?;
        let ext = match mime_type.unwrap_or("audio/webm") {
            m if m.contains("wav") => "wav",
            m if m.contains("mp3") || m.contains("mpeg") => "mp3",
            m if m.contains("ogg") => "ogg",
            _ => "webm",
        };
        let audio_path = temp_dir.join(format!("input.{ext}"));
        std::fs::write(&audio_path, &bytes)?;

        let mut command = tokio::process::Command::new("sh");
        command
            .arg("-c")
            .arg(&cmd)
            .env("ALLPAKA_AUDIO_FILE", &audio_path)
            .env("ALLPAKA_AUDIO_MIME", mime_type.unwrap_or("audio/webm"))
            .env("ALLPAKA_AUDIO_LANG", language.unwrap_or(""))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let output = tokio::time::timeout(DRIVER_TIMEOUT, command.output())
            .await
            .context("STT driver timed out")?;
        let _ = std::fs::remove_dir_all(&temp_dir);
        let output = output?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("STT driver failed: {}", err.trim());
        }
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if let Ok(mut parsed) = serde_json::from_str::<Value>(&stdout) {
            if let Some(obj) = parsed.as_object_mut() {
                obj.entry("available").or_insert(json!(true));
                obj.entry("backend").or_insert(json!("cmd"));
            }
            return Ok(parsed);
        }
        return Ok(json!({
            "available": true,
            "backend": "cmd",
            "text": stdout,
        }));
    }

    let env_url = std::env::var("ALLPAKA_STT_URL").ok();
    let stt_url = url_override
        .map(str::to_owned)
        .or_else(|| env_url.filter(|s| !s.trim().is_empty()));

    if let Some(url) = stt_url {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        let mut req = client.post(&url).json(&json!({
            "audio_b64": audio_b64,
            "mime_type": mime_type.unwrap_or("audio/webm"),
            "language": language,
        }));
        if let Ok(key) = std::env::var("ALLPAKA_STT_KEY") {
            if !key.trim().is_empty() {
                req = req.bearer_auth(key.trim());
            }
        }
        let resp = req.send().await.context("STT HTTP request failed")?;
        let status = resp.status();
        let val: Value = resp.json().await.context("Invalid JSON from STT service")?;
        if !status.is_success() {
            bail!("STT service returned HTTP {status}: {val}");
        }
        return Ok(json!({
            "available": true,
            "backend": "http",
            "text": val.get("text").and_then(Value::as_str).unwrap_or(""),
            "raw": val,
        }));
    }

    Ok(json!({
        "available": false,
        "backend": "none",
        "error": "Speech-to-text backend not configured (set ALLPAKA_STT_CMD or ALLPAKA_STT_URL)",
    }))
}

pub(crate) async fn transcribe_api(
    State(_app): State<App>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let Some(audio_b64) = body["audio_b64"].as_str() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Missing `audio_b64`" })),
        );
    };
    let mime = body["mime_type"].as_str();
    let lang = body["language"].as_str();
    match execute_stt_with_overrides(audio_b64, mime, lang, None, None).await {
        Ok(val) if val.get("available").and_then(Value::as_bool) == Some(false) => {
            (StatusCode::NOT_IMPLEMENTED, Json(val))
        }
        Ok(val) => (StatusCode::OK, Json(val)),
        Err(err) => {
            let e = bad_request(err);
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("{e:?}") })),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn browser_and_computer_unconfigured_and_mocked() {
        let unconf = execute_browser_with_cmd(
            None,
            "s1",
            None,
            &json!({"action": "navigate", "url": "https://93.184.216.34/"}),
            None,
        )
        .await
        .unwrap();
        assert_eq!(unconf["available"], false);

        // SSRF check must still reject private targets before spawning any driver
        assert!(execute_browser_with_cmd(
            None,
            "s1",
            None,
            &json!({"action": "navigate", "url": "http://127.0.0.1:8080/"}),
            Some("cat")
        )
        .await
        .is_err());

        let mocked = execute_browser_with_cmd(
            None,
            "s1",
            None,
            &json!({"action": "navigate", "url": "https://93.184.216.34/"}),
            Some("python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({\"available\":True,\"seen\":d[\"args\"][\"url\"]}))'"),
        )
        .await
        .unwrap();
        assert_eq!(mocked["available"], true);
        assert_eq!(mocked["seen"], "https://93.184.216.34/");

        let cu_unconf =
            execute_computer_with_cmd(None, "s1", None, &json!({"action": "screenshot"}), None)
                .await
                .unwrap();
        assert_eq!(cu_unconf["available"], false);

        let cu_mock = execute_computer_with_cmd(
            None,
            "s1",
            None,
            &json!({"action": "click", "x": 10, "y": 20}),
            Some("python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({\"clicked\":[d[\"args\"][\"x\"],d[\"args\"][\"y\"]]}))'"),
        )
        .await
        .unwrap();
        assert_eq!(cu_mock["clicked"], json!([10, 20]));
    }

    #[tokio::test]
    async fn stt_unconfigured_and_mock_cmd() {
        let b64 = storage::base64_encode(b"RIFFWAVE_MOCK");
        let unconf = execute_stt_with_overrides(&b64, Some("audio/wav"), Some("en"), None, None)
            .await
            .unwrap();
        assert_eq!(unconf["available"], false);

        let ok = execute_stt_with_overrides(
            &b64,
            Some("audio/wav"),
            Some("en"),
            Some("python3 -c 'import os; print(\"transcribed \" + os.environ[\"ALLPAKA_AUDIO_MIME\"])'"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(ok["available"], true);
        assert_eq!(ok["text"], "transcribed audio/wav");
    }
}
