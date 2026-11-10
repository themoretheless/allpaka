//! Ingestion of externally produced traces: wire types, validation, keyed
//! idempotent writes, and removal and restore of a trace.

use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalTrace {
    pub(super) project_id: String,
    #[serde(default)]
    pub(super) idempotency_key: Option<String>,
    pub(super) correlation_id: String,
    pub(super) started_ms: u64,
    pub(super) spans: Vec<ExternalSpan>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalSpan {
    #[serde(default)]
    pub(super) provider_id: Option<String>,
    pub(super) parent_id: Option<usize>,
    pub(super) kind: String,
    pub(super) name: String,
    pub(super) status: String,
    pub(super) started_ms: u64,
    pub(super) duration_ms: Option<u64>,
    #[serde(default)]
    pub(super) usage: ExternalUsage,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalUsage {
    pub(super) guardrail_receipt: Option<Value>,
    pub(super) evaluation_scores: Option<Value>,
    pub(super) evaluator_ref: Option<Value>,
    pub(super) evaluation_sampling: Option<Value>,
    pub(super) first_text_ms: Option<u64>,
    pub(super) input_tokens: Option<u64>,
    pub(super) output_tokens: Option<u64>,
    pub(super) cache_read_input_tokens: Option<u64>,
    pub(super) cache_creation_input_tokens: Option<u64>,
    pub(super) cost: Option<f64>,
    pub(super) cost_currency: Option<String>,
}
impl Store {
    pub(super) fn is_removed(&self, id: &str) -> Result<bool> {
        let directory = self.root.join("removed");
        match std::fs::symlink_metadata(&directory) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        if !std::fs::symlink_metadata(&directory)?.is_dir()
            || !directory.canonicalize()?.starts_with(self.root.as_ref())
        {
            bail!("Invalid removed trace directory");
        }
        let path = directory.join(format!("{id}.json"));
        match std::fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() > 1024 {
            bail!("Invalid trace removal marker");
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1025)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 {
            bail!("Oversize trace removal marker");
        }
        let marker: Value = serde_json::from_slice(&bytes)?;
        if marker["trace_id"] != id || marker["removed_ms"].as_u64().is_none() {
            bail!("Invalid trace removal identity");
        }
        Ok(true)
    }
    pub(super) fn change_visibility(&self, id: &str, removed: bool) -> Result<Value> {
        let _lock = self.lifecycle.lock().unwrap();
        let record = self.read_record(id)?;
        if record.status == "running" || record.spans.iter().any(|span| span.status == "running") {
            bail!("Active traces cannot be removed or restored");
        }
        let previous = self.is_removed(id)?;
        if removed && !previous {
            let directory = self.root.join("removed");
            std::fs::create_dir_all(&directory)?;
            if !std::fs::symlink_metadata(&directory)?.is_dir()
                || !directory.canonicalize()?.starts_with(self.root.as_ref())
            {
                bail!("Invalid removed trace directory");
            }
            crate::evaluation::commit_new(
                &directory.join(format!("{id}.json")),
                &json!({"trace_id":id,"removed_ms":now()}),
            )?;
        } else if !removed && previous {
            std::fs::remove_file(self.root.join("removed").join(format!("{id}.json")))?;
        }
        Ok(
            json!({"trace_id":id,"removed":removed,"changed":previous!=removed,"content_preserved":true}),
        )
    }
    pub(super) fn ingest(&self, request: ExternalTrace) -> Result<Value> {
        let identifier = |s: &str| {
            !s.is_empty()
                && s.len() <= 100
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-:/".contains(&b))
        };
        if !identifier(&request.correlation_id)
            || request.spans.is_empty()
            || request.spans.len() > 200
        {
            bail!("External traces require a bounded correlation ID and 1–200 spans");
        }
        if request
            .idempotency_key
            .as_ref()
            .is_some_and(|key| !identifier(key))
        {
            bail!("Invalid external trace idempotency key");
        }
        let mut spans = Vec::new();
        for (index, span) in request.spans.into_iter().enumerate() {
            if (index == 0 && span.parent_id.is_some())
                || (index > 0 && !span.parent_id.is_some_and(|parent| parent < index))
            {
                bail!("Provide one root and parent-before-child spans");
            }
            if !identifier(&span.name)
                || !matches!(span.kind.as_str(), "agent" | "model" | "tool")
                || span
                    .provider_id
                    .as_ref()
                    .is_some_and(|provider| !identifier(provider))
                || !matches!(span.status.as_str(), "completed" | "failed" | "interrupted")
            {
                bail!("Invalid external span metadata");
            }
            if span.started_ms < request.started_ms
                || span
                    .duration_ms
                    .is_some_and(|d| span.started_ms.checked_add(d).is_none())
            {
                bail!("Invalid external span timing");
            }
            if let Some(parent) = span.parent_id {
                let parent: &Span = &spans[parent];
                if span.started_ms < parent.started_ms {
                    bail!("Child starts before its parent");
                }
                if parent
                    .duration_ms
                    .is_some_and(|d| span.started_ms > parent.started_ms + d)
                {
                    bail!("Child starts after its parent ends");
                }
                if let (Some(pd), Some(cd)) = (parent.duration_ms, span.duration_ms) {
                    if span.started_ms + cd > parent.started_ms + pd {
                        bail!("Child ends after its parent");
                    }
                }
            }
            let usage = span.usage;
            if let Some(receipt) = &usage.guardrail_receipt {
                let clean = public_guardrail_receipt(receipt)
                    .context("Invalid external guardrail receipt")?;
                let name = format!(
                    "guardrail.{}.{}.{}.{}",
                    clean["stage"].as_str().unwrap(),
                    clean["action"].as_str().unwrap(),
                    if clean["passed"] == true {
                        "pass"
                    } else {
                        "fail"
                    },
                    clean["policy_sha256"].as_str().unwrap()
                );
                if clean != *receipt
                    || span.kind != "tool"
                    || span.name != name
                    || span.status
                        != if clean["blocked"] == true {
                            "failed"
                        } else {
                            "completed"
                        }
                {
                    bail!("External guardrail receipt metadata mismatch");
                }
            }

            if let Some(sampling) = &usage.evaluation_sampling {
                let clean = valid_evaluation_sampling(sampling)
                    .context("Invalid evaluation sampling receipt")?;
                let rate = clean["sample_rate"].as_f64().unwrap();
                let key = serde_json::to_vec(&json!([
                    "allpaka-evaluation-sampling-v1",
                    request.project_id,
                    request.correlation_id,
                    span.name,
                    index
                ]))?;
                let hash = Sha256::digest(key);
                let choice = u64::from_be_bytes(hash[..8].try_into().unwrap()) as u128;
                let expected = rate == 1.0
                    || rate > 0.0 && choice < (rate * 18446744073709551616.0).ceil() as u128;
                if clean["selected"] != json!(expected) || span.status != "completed" {
                    bail!("Evaluation sampling decision mismatch");
                }
            }
            if let Some(reference) = &usage.evaluator_ref {
                if valid_evaluator_ref(reference).is_none() || span.kind != "tool" {
                    bail!("Invalid evaluator reference");
                }
            }
            if let Some(scores) = &usage.evaluation_scores {
                if valid_evaluation_scores(scores).is_none()
                    || span.kind != "tool"
                    || span.status != "completed"
                {
                    bail!("Invalid external evaluation scores");
                }
            }
            if usage.cost.is_some_and(|n| !n.is_finite() || n < 0.0)
                || usage
                    .cost_currency
                    .as_ref()
                    .is_some_and(|s| s.len() != 3 || !s.bytes().all(|b| b.is_ascii_uppercase()))
            {
                bail!("Invalid reported external cost");
            }
            if usage
                .first_text_ms
                .zip(span.duration_ms)
                .is_some_and(|(first, total)| first > total)
            {
                bail!("First text time exceeds span duration");
            }
            spans.push(Span{provider_id:span.provider_id,id:index,parent_id:span.parent_id,kind:format!("external_{}",span.kind),name:span.name,
                status:span.status,started_ms:span.started_ms,duration_ms:span.duration_ms,linked_trace_id:None,
                usage:public_usage(&json!({"input_tokens":usage.input_tokens,"output_tokens":usage.output_tokens,
                    "evaluation_sampling":usage.evaluation_sampling,"evaluator_ref":usage.evaluator_ref,"evaluation_scores":usage.evaluation_scores,"guardrail_receipt":usage.guardrail_receipt,"first_text_ms":usage.first_text_ms,"cache_read_input_tokens":usage.cache_read_input_tokens,"cache_creation_input_tokens":usage.cache_creation_input_tokens,"cost":usage.cost,"cost_currency":usage.cost_currency}))});
        }
        let keyed = request.idempotency_key.is_some();
        let id = if let Some(key) = request.idempotency_key {
            // Scope keys to the project; no raw key is retained in metadata.
            format!(
                "trace-external-{}",
                crate::storage::hex_sha256(&serde_json::to_vec(&(
                    request.project_id.as_str(),
                    key
                ))?)
            )
        } else {
            format!("trace-external-{}", crate::evaluation::new_id())
        };
        let record = Record {
            id: id.clone(),
            project_id: request.project_id,
            session_id: format!("external-{}", request.correlation_id),
            started_ms: request.started_ms,
            status: spans[0].status.clone(),
            recovered_ms: None,
            spans,
        };
        let path = self.root.join(format!("{id}.json"));
        let replay = || -> Result<Value> {
            if self.is_removed(&id)? {
                bail!("External trace was removed; restore it explicitly");
            }
            let existing = self.read_record(&id)?;
            if serde_json::to_value(&existing)? != serde_json::to_value(&record)? {
                bail!("External trace idempotency conflict");
            }
            Ok(
                json!({"id":id,"source":"external","span_count":record.spans.len(),"provider_calls":0,"deduplicated":true}),
            )
        };
        if keyed && path.try_exists()? {
            return replay();
        }
        if let Err(error) = crate::evaluation::commit_new(&path, &record) {
            if keyed && path.try_exists()? {
                return replay();
            }
            return Err(error);
        }
        update_cache(&self.cache, &path, &record);
        Ok(
            json!({"id":id,"source":"external","span_count":record.spans.len(),"provider_calls":0,"deduplicated":false}),
        )
    }
}
pub(crate) async fn ingest(
    State(app): State<crate::App>,
    Json(request): Json<ExternalTrace>,
) -> crate::ApiResult<Value> {
    crate::blocking(move || {
        if !app
            .projects
            .lock()
            .unwrap()
            .iter()
            .any(|project| project.id == request.project_id)
        {
            return Err(crate::error(
                StatusCode::BAD_REQUEST,
                "Unknown external trace project",
            ));
        }
        let project = request.project_id.clone();
        let completed = request
            .spans
            .first()
            .is_some_and(|span| span.status == "completed")
            && request.spans.iter().all(|span| span.status != "running");
        app.observability
            .ingest(request)
            .map(|receipt| {
                if completed {
                    if let Some(id) = receipt["id"].as_str() {
                        if let Err(error) =
                            crate::online_evaluation::enqueue_completed(&app.data, &project, id)
                        {
                            eprintln!("External online evaluation admission failed: {error}");
                        }
                    }
                }
                Json(receipt)
            })
            .map_err(|e| {
                let status = if e.to_string() == "External trace idempotency conflict" {
                    StatusCode::CONFLICT
                } else {
                    StatusCode::BAD_REQUEST
                };
                crate::error(status, e)
            })
    })
    .await
}
