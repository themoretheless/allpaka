//! OpenTelemetry OTLP/HTTP JSON trace serialization (`GET /api/observability/traces/:id/otlp`)
//! and SSRF-checked exporter push (`POST /api/observability/traces/:id/export`).

use crate::{bad_request, not_found, storage, web_tools, ApiResult, App};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde_json::{json, Value};
use std::time::Duration;

fn deterministic_trace_id(id: &str) -> String {
    let hex = storage::hex_sha256(id.as_bytes());
    hex.chars().take(32).collect()
}

fn deterministic_span_id(trace_id: &str, idx: usize) -> String {
    let hex = storage::hex_sha256(format!("{trace_id}:{idx}").as_bytes());
    hex.chars().take(16).collect()
}

fn kv_str(k: &str, v: &str) -> Value {
    json!({ "key": k, "value": { "stringValue": v } })
}

pub(crate) fn trace_to_otlp(record: &Value) -> Value {
    let rec_id = record["id"].as_str().unwrap_or("trace");
    let trace_id = deterministic_trace_id(rec_id);
    let base_started_ms = record["started_ms"].as_u64().unwrap_or(0);
    let project_id = record["project_id"].as_str().unwrap_or("default");

    let spans = record["spans"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(idx, sp)| {
                    let span_id = deterministic_span_id(rec_id, idx);
                    let parent_span_id = sp["parent_id"]
                        .as_u64()
                        .map(|p| deterministic_span_id(rec_id, p as usize))
                        .unwrap_or_default();
                    let start_ms = sp["started_ms"].as_u64().unwrap_or(base_started_ms);
                    let dur_ms = sp["duration_ms"].as_u64().unwrap_or(0);
                    let start_ns = (start_ms as u128) * 1_000_000;
                    let end_ns = ((start_ms.saturating_add(dur_ms)) as u128) * 1_000_000;
                    let status_str = sp["status"].as_str().unwrap_or("unset");
                    let code = match status_str {
                        "completed" | "ok" => 1,
                        "failed" | "error" => 2,
                        _ => 0,
                    };
                    let mut attrs = vec![
                        kv_str("allpaka.trace_id", rec_id),
                        kv_str("allpaka.project_id", project_id),
                        kv_str(
                            "allpaka.span_kind",
                            sp["kind"].as_str().unwrap_or("internal"),
                        ),
                        kv_str("allpaka.status", status_str),
                    ];
                    if let Some(pid) = sp["provider_id"].as_str() {
                        attrs.push(kv_str("llm.provider", pid));
                    }
                    json!({
                        "traceId": trace_id,
                        "spanId": span_id,
                        "parentSpanId": parent_span_id,
                        "name": sp["name"].as_str().unwrap_or("span"),
                        "kind": 1,
                        "startTimeUnixNano": start_ns.to_string(),
                        "endTimeUnixNano": end_ns.to_string(),
                        "attributes": attrs,
                        "status": { "code": code },
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    kv_str("service.name", "allpaka-studio"),
                    kv_str("allpaka.project_id", project_id),
                ]
            },
            "scopeSpans": [{
                "scope": { "name": "allpaka-chat", "version": "0.1.0" },
                "spans": spans,
            }]
        }]
    })
}

pub(crate) async fn get_otlp_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Value> {
    let record = app
        .observability
        .trace_record_json(&id)
        .map_err(|_| not_found("Trace not found"))?;
    Ok(Json(trace_to_otlp(&record)))
}

pub(crate) async fn export_otlp_api(
    State(app): State<App>,
    AxumPath(id): AxumPath<String>,
    body: Option<Json<Value>>,
) -> ApiResult<Value> {
    let record = app
        .observability
        .trace_record_json(&id)
        .map_err(|_| not_found("Trace not found"))?;
    let payload = trace_to_otlp(&record);

    let req_endpoint = body
        .as_ref()
        .and_then(|Json(v)| v["endpoint"].as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        })
        .ok_or_else(|| bad_request("Missing OTLP `endpoint` (or OTEL_EXPORTER_OTLP_ENDPOINT)"))?;

    let allow_local = std::env::var("ALLPAKA_OTEL_ALLOW_LOCAL")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if !allow_local {
        web_tools::validate_public_url_syntax(&req_endpoint).map_err(bad_request)?;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(crate::internal)?;
    let resp = client
        .post(&req_endpoint)
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(bad_request)?;
    let status = resp.status();
    if !status.is_success() {
        return Err(bad_request(format!(
            "OTLP collector returned HTTP {status}"
        )));
    }
    Ok(Json(json!({
        "exported": true,
        "trace_id": id,
        "endpoint": req_endpoint,
        "status": status.as_u16(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_conversion_builds_deterministic_ids_and_parent_links() {
        let rec = json!({
            "id": "trace-abc-123",
            "project_id": "proj1",
            "started_ms": 1700000000000u64,
            "spans": [
                { "parent_id": null, "kind": "turn", "name": "root", "started_ms": 1700000000000u64, "duration_ms": 50, "status": "completed" },
                { "parent_id": 0, "kind": "tool", "name": "read_file", "started_ms": 1700000000010u64, "duration_ms": 15, "status": "completed", "provider_id": "openai" }
            ]
        });
        let otlp = trace_to_otlp(&rec);
        let spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array()
            .unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0]["traceId"].as_str().unwrap().len(), 32);
        assert_eq!(spans[0]["spanId"].as_str().unwrap().len(), 16);
        assert_eq!(spans[0]["spanId"], spans[1]["parentSpanId"]);
        assert_eq!(spans[1]["startTimeUnixNano"], "1700000000010000000");
        assert_eq!(spans[1]["endTimeUnixNano"], "1700000000025000000");
        assert_eq!(spans[1]["status"]["code"], 1);
    }
}
