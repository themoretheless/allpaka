//! Query parameter types and the axum handlers for listing, summaries,
//! evaluation summaries, export, removal and restore.

use super::*;

#[derive(Default, Deserialize)]
pub(crate) struct Search {
    pub(super) status: Option<String>,
    pub(super) since_ms: Option<u64>,
    pub(super) until_ms: Option<u64>,
    pub(super) guardrail: Option<GuardrailFilter>,
    pub(super) guardrail_policy_sha256: Option<String>,
    pub(super) removed: Option<bool>,
    pub(super) session_id: Option<String>,
    pub(super) project_id: Option<String>,
    pub(super) offset: Option<usize>,
    pub(super) limit: Option<usize>,
}
#[derive(Default)]
pub(super) struct ConversationTotals {
    pub(super) traces: u64,
    pub(super) statuses: std::collections::BTreeMap<String, u64>,
    pub(super) first: Option<u64>,
    pub(super) last: u64,
    pub(super) external_calls: u64,
    pub(super) models: ModelTotals,
}
#[derive(Default)]
pub(super) struct TimeBucket {
    pub(super) traces: u64,
    pub(super) statuses: std::collections::BTreeMap<String, u64>,
    pub(super) models: ModelTotals,
    pub(super) guardrails: GuardrailTotals,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeSearch {
    pub(super) status: Option<String>,
    pub(super) project_id: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) since_ms: u64,
    pub(super) until_ms: u64,
    pub(super) bucket_ms: u64,
}
pub(crate) async fn time_series(
    State(app): State<crate::App>,
    Query(query): Query<TimeSearch>,
) -> crate::ApiResult<Value> {
    app.observability
        .time_series(&query)
        .map(Json)
        .map_err(|error| crate::error(StatusCode::BAD_REQUEST, error))
}
#[derive(Default, Deserialize)]
pub(crate) struct SummarySearch {
    pub(super) status: Option<String>,
    pub(super) conversation_offset: Option<usize>,
    pub(super) conversation_limit: Option<usize>,
    pub(super) project_id: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) since_ms: Option<u64>,
    pub(super) until_ms: Option<u64>,
}
pub(crate) async fn summary(
    State(app): State<crate::App>,
    Query(query): Query<SummarySearch>,
) -> crate::ApiResult<Value> {
    app.observability
        .summary(&query)
        .map(Json)
        .map_err(|error| crate::error(StatusCode::BAD_REQUEST, error))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationSearch {
    pub(super) project_id: String,
    pub(super) since_ms: Option<u64>,
    pub(super) until_ms: Option<u64>,
}
impl Store {
    pub(super) fn evaluation_summary(&self, query: &EvaluationSearch) -> Result<Value> {
        if query.project_id.is_empty()
            || query.project_id.len() > 100
            || !query
                .project_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
            || query
                .since_ms
                .zip(query.until_ms)
                .is_some_and(|(a, b)| a > b)
        {
            bail!("Invalid evaluation summary scope");
        }
        let records = self.scan(&Search {
            project_id: Some(query.project_id.clone()),
            since_ms: query.since_ms,
            until_ms: query.until_ms,
            ..Default::default()
        })?;
        let mut selected = 0u64;
        let mut skipped = 0u64;
        let mut failed = 0u64;
        let mut completed = 0u64;
        let mut groups = std::collections::BTreeMap::<
            (Option<String>, Option<u64>, String),
            (u64, f64, f64, f64),
        >::new();
        let mut attempts =
            std::collections::BTreeMap::<(Option<String>, Option<u64>), (u64, u64)>::new();
        for record in &records {
            for span in &record.spans {
                if let Some(receipt) = valid_evaluation_sampling(&span.usage["evaluation_sampling"])
                {
                    if receipt["selected"] == true {
                        selected += 1;
                    } else {
                        skipped += 1;
                    }
                }
                if span.kind != "external_tool" {
                    continue;
                }
                let reference = valid_evaluator_ref(&span.usage["evaluator_ref"]);
                let scores = valid_evaluation_scores(&span.usage["evaluation_scores"]);
                if (span.name == "evaluation" || reference.is_some() || scores.is_some())
                    && matches!(span.status.as_str(), "failed" | "completed")
                {
                    let key = (
                        reference
                            .as_ref()
                            .and_then(|r| r["id"].as_str())
                            .map(str::to_owned),
                        reference.as_ref().and_then(|r| r["version"].as_u64()),
                    );
                    let row = attempts.entry(key).or_default();
                    if span.status == "failed" {
                        failed += 1;
                        row.1 += 1;
                    } else {
                        completed += 1;
                        row.0 += 1;
                    }
                    if attempts.len() > 4000 {
                        bail!("Evaluation summary evaluator limit reached");
                    }
                }
                if span.status != "completed" {
                    continue;
                }
                if let Some(scores) = scores {
                    for (metric, value) in scores.as_object().unwrap() {
                        let key = (
                            reference
                                .as_ref()
                                .and_then(|r| r["id"].as_str())
                                .map(str::to_owned),
                            reference.as_ref().and_then(|r| r["version"].as_u64()),
                            metric.clone(),
                        );
                        let score = value.as_f64().unwrap();
                        let row = groups.entry(key).or_insert((0, 0.0, score, score));
                        row.0 += 1;
                        row.1 += score;
                        row.2 = row.2.min(score);
                        row.3 = row.3.max(score);
                        if groups.len() > 4000 {
                            bail!("Evaluation summary group limit reached");
                        }
                    }
                }
            }
        }
        let metrics=groups.into_iter().map(|((id,version,metric),(count,sum,min,max))|json!({"evaluator_id":id,"evaluator_version":version,"metric":metric,"count":count,"mean":(sum/count as f64).clamp(min,max),"min":min,"max":max})).collect::<Vec<_>>();
        let evaluators=attempts.into_iter().map(|((id,version),(completed,failed))|json!({"evaluator_id":id,"evaluator_version":version,"completed_assessments":completed,"failed_assessments":failed})).collect::<Vec<_>>();
        Ok(
            json!({"kind":"callback_evaluation_summary","project_id":query.project_id,"since_ms":query.since_ms,"until_ms":query.until_ms,"trace_count":records.len(),"selected_tasks":selected,"skipped_tasks":skipped,"completed_assessments":completed,"failed_assessments":failed,"evaluators":evaluators,"metrics":metrics,"assessment_source":"caller_reported","provider_calls":0,"automatic_promotion":false}),
        )
    }
}
pub(crate) async fn evaluation_summary(
    State(app): State<crate::App>,
    Query(query): Query<EvaluationSearch>,
) -> crate::ApiResult<Value> {
    app.observability
        .evaluation_summary(&query)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<Search>,
) -> crate::ApiResult<Value> {
    app.observability
        .list(&query)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn get(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> crate::ApiResult<Value> {
    app.observability
        .read_validated_record(&id, None, None)
        .and_then(|record| Ok(serde_json::to_value(record)?))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportOptions {
    pub(super) include_feedback: Option<bool>,
}
pub(super) fn export_packet(app: &crate::App, id: &str, include_feedback: bool) -> Result<Value> {
    let record = app.observability.read_validated_record(id, None, None)?;
    if record.status == "running" || record.spans.iter().any(|span| span.status == "running") {
        bail!("Export requires a completed trace snapshot");
    }
    let feedback = if include_feedback {
        Some(app.feedback.export_snapshot(id)?)
    } else {
        None
    };
    let packet = json!({"kind":"trace_export","schema_version":1,"exported_ms":now(),"trace":record,"feedback":feedback,
        "privacy":{"runtime_content_capture":false,"feedback_included":include_feedback},"provider_calls":0});
    if serde_json::to_vec(&packet)?.len() > 3 * 1024 * 1024 {
        bail!("Trace export exceeds 3 MiB");
    }
    Ok(packet)
}
pub(crate) async fn export(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(options): Query<ExportOptions>,
) -> crate::ApiResult<Value> {
    export_packet(&app, &id, options.include_feedback.unwrap_or(false))
        .map(Json)
        .map_err(|error| crate::error(StatusCode::BAD_REQUEST, error))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BulkExportRequest {
    pub(super) project_id: String,
    pub(super) trace_ids: Vec<String>,
    #[serde(default)]
    pub(super) include_feedback: bool,
}
pub(crate) async fn bulk_export(
    State(app): State<crate::App>,
    Json(request): Json<BulkExportRequest>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        if request.project_id.is_empty()
            || request.project_id.len() > 80
            || request.trace_ids.is_empty()
            || request.trace_ids.len() > 100
        {
            bail!("Choose a project and 1-100 unique traces");
        }
        let mut seen = std::collections::HashSet::new();
        for id in &request.trace_ids {
            if !seen.insert(id) {
                bail!("Duplicate export trace ID");
            }
        }
        let mut packets = Vec::new();
        let mut bytes = 0usize;
        for id in &request.trace_ids {
            let packet = export_packet(&app, id, request.include_feedback)?;
            if packet["trace"]["project_id"] != request.project_id {
                bail!("Trace export project mismatch");
            }
            bytes = bytes
                .checked_add(serde_json::to_vec(&packet)?.len())
                .context("Export size overflow")?;
            if bytes > 16 * 1024 * 1024 {
                bail!("Bulk trace export exceeds 16 MiB");
            }
            packets.push(packet);
        }
        let packet = json!({"kind":"trace_export_batch","schema_version":1,"project_id":request.project_id,"exported_ms":now(),
            "trace_count":packets.len(),"traces":packets,"privacy":{"runtime_content_capture":false,"feedback_included":request.include_feedback},"provider_calls":0});
        if serde_json::to_vec(&packet)?.len() > 16 * 1024 * 1024 {
            bail!("Bulk trace export exceeds 16 MiB");
        }
        Ok(packet)
    })();
    result
        .map(Json)
        .map_err(|error| crate::error(StatusCode::BAD_REQUEST, error))
}
pub(crate) async fn remove(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> crate::ApiResult<Value> {
    app.observability
        .change_visibility(&id, true)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn restore(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> crate::ApiResult<Value> {
    app.observability
        .change_visibility(&id, false)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
