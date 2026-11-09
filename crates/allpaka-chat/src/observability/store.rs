//! Reading the trace store: validated record access, startup recovery,
//! scanning with filters, listings, time series and summaries.

use super::*;

impl Store {
    pub(super) fn read_record(&self, id: &str) -> Result<Record> {
        if !id.starts_with("trace-")
            || id.len() > 100
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            bail!("Invalid trace ID");
        }
        let path = self.root.join(format!("{id}.json"));
        let metadata = std::fs::symlink_metadata(&path).context("Trace not found")?;
        if !metadata.is_file() || metadata.len() > 1024 * 1024 {
            bail!("Invalid trace record");
        }
        let mtime = metadata.modified().ok();
        if mtime.is_some() {
            if let Some((cached_len, cached_mtime, cached_record)) =
                self.cache.lock().unwrap().get(id)
            {
                if *cached_len == metadata.len() && *cached_mtime == mtime {
                    return Ok(cached_record.clone());
                }
            }
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 {
            bail!("Oversize trace record");
        }
        let record: Record = serde_json::from_slice(&bytes)?;
        if record.id != id {
            bail!("Trace or span identity does not exist");
        }
        if mtime.is_some() {
            self.cache
                .lock()
                .unwrap()
                .insert(id.to_owned(), (metadata.len(), mtime, record.clone()));
        }
        Ok(record)
    }
    pub(super) fn read_validated_record(
        &self,
        id: &str,
        span: Option<usize>,
        project: Option<&str>,
    ) -> Result<Record> {
        let record = self.read_record(id)?;
        if self.is_removed(id)? {
            bail!("Trace was removed");
        }
        if span.is_some_and(|n| !record.spans.iter().any(|s| s.id == n)) {
            bail!("Trace span does not exist");
        }
        if project.is_some_and(|p| record.project_id != p) {
            bail!("Review target belongs to another project");
        }
        Ok(record)
    }
    pub(crate) fn completed_trace_health(&self, id: &str, project: &str) -> Result<(String, f64)> {
        let record = self.read_validated_record(id, None, Some(project))?;
        if record.status != "completed"
            || record.spans.is_empty()
            || record.spans.iter().any(|span| span.status == "running")
        {
            bail!("Assessment requires a completed trace with spans");
        }
        let hash = crate::storage::hex_sha256(&serde_json::to_vec(&record)?);
        let score = record
            .spans
            .iter()
            .filter(|span| span.status == "completed")
            .count() as f64
            / record.spans.len() as f64;
        Ok((hash, score))
    }
    pub(crate) fn completed_trace_fingerprint(&self, id: &str, project: &str) -> Result<String> {
        let record = self.read_validated_record(id, None, Some(project))?;
        if record.status != "completed" || record.spans.iter().any(|span| span.status == "running")
        {
            bail!("Online selection requires a completed trace");
        }
        Ok(crate::storage::hex_sha256(&serde_json::to_vec(&record)?))
    }
    pub(crate) fn validate_project_target(
        &self,
        id: &str,
        span: Option<usize>,
        project: &str,
    ) -> Result<()> {
        self.read_validated_record(id, span, Some(project))
            .map(|_| ())
    }
    pub(crate) fn validate_target(&self, id: &str, span: Option<usize>) -> Result<()> {
        self.read_validated_record(id, span, None).map(|_| ())
    }
    pub(crate) fn new(data: &std::path::Path) -> Result<Self> {
        let root = data.join("observability/traces");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if !root.starts_with(data) {
            bail!("Trace storage escapes Studio data directory");
        }
        Ok(Self {
            root: Arc::new(root),
            lifecycle: Arc::new(Mutex::new(())),
            cache: Default::default(),
        })
    }
    /// Called once at startup while the data-directory OS lock is held.
    pub(crate) fn recover(&self) -> Result<usize> {
        let mut recovered = 0;
        let mut count = 0;
        let mut total_bytes = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|s| s != "json") || !entry.file_type()?.is_file() {
                continue;
            }
            let size = entry.metadata()?.len();
            count += 1;
            total_bytes += size;
            if count > 10_000 || total_bytes > 64 * 1024 * 1024 || size > 1024 * 1024 {
                bail!("Trace recovery scan limit reached");
            }
            let mut bytes = Vec::new();
            std::fs::File::open(&path)?
                .take(1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > 1024 * 1024 {
                bail!("Oversize trace during recovery");
            }
            let mut record: Record = serde_json::from_slice(&bytes)?;
            if path.file_stem().and_then(|s| s.to_str()) != Some(record.id.as_str()) {
                bail!("Trace identity mismatch during recovery");
            }
            if record.status == "running" || record.spans.iter().any(|s| s.status == "running") {
                record.status = "interrupted".into();
                record.recovered_ms = Some(now());
                for span in &mut record.spans {
                    if span.status == "running" {
                        span.status = "interrupted".into();
                    }
                }
                // The restart time is not the operation end time. Unknown durations stay unknown.
                atomic_write(&path, &record)?;
                recovered += 1;
            }
            update_cache(&self.cache, &path, &record);
        }
        Ok(recovered)
    }
    pub(crate) fn begin(&self, session: &str, project: &str) -> Result<Trace> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = format!(
            "trace-{}-{}-{}",
            now(),
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let trace = Trace {
            path: Arc::new(self.root.join(format!("{id}.json"))),
            record: Arc::new(Mutex::new(Record {
                id,
                session_id: session.into(),
                project_id: project.into(),
                started_ms: now(),
                status: "running".into(),
                recovered_ms: None,
                spans: vec![],
            })),
            writer: Arc::new(Mutex::new(())),
            cache: self.cache.clone(),
        };
        trace.persist()?;
        Ok(trace)
    }
    pub(super) fn scan(&self, query: &Search) -> Result<Vec<Record>> {
        let mut rows = Vec::new();
        let mut scanned_bytes = 0u64;
        let mut scanned_records = 0usize;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|s| s != "json") || !entry.file_type()?.is_file() {
                continue;
            }
            scanned_records += 1;
            scanned_bytes += entry.metadata()?.len();
            if scanned_records > 10_000 || scanned_bytes > 64 * 1024 * 1024 {
                bail!("Trace scan limit reached; retention is required");
            }
            if entry.metadata()?.len() > 1024 * 1024 {
                bail!("Oversize trace record");
            }
            let id = path
                .file_stem()
                .and_then(|id| id.to_str())
                .context("Invalid trace filename")?;
            let record = self.read_record(id)?;
            if self.is_removed(id)? != query.removed.unwrap_or(false) {
                continue;
            }
            if query
                .session_id
                .as_ref()
                .is_some_and(|id| id != &record.session_id)
                || query
                    .project_id
                    .as_ref()
                    .is_some_and(|id| id != &record.project_id)
            {
                continue;
            }
            if query
                .status
                .as_ref()
                .is_some_and(|status| status != &record.status)
                || query
                    .since_ms
                    .is_some_and(|since| record.started_ms < since)
                || query
                    .until_ms
                    .is_some_and(|until| record.started_ms > until)
            {
                continue;
            }
            if (query.guardrail.is_some() || query.guardrail_policy_sha256.is_some())
                && !record.spans.iter().any(|span| {
                    query
                        .guardrail
                        .as_ref()
                        .unwrap_or(&GuardrailFilter::Any)
                        .matches(span)
                        && query
                            .guardrail_policy_sha256
                            .as_ref()
                            .is_none_or(|hash| span.name.rsplit('.').next() == Some(hash.as_str()))
                })
            {
                continue;
            }
            rows.push(record);
            if rows.len() > 100_000 {
                bail!("Trace listing capacity exceeded; retention is required");
            }
        }
        rows.sort_by(|a, b| {
            b.started_ms
                .cmp(&a.started_ms)
                .then_with(|| b.id.cmp(&a.id))
        });
        Ok(rows)
    }
    pub(super) fn list(&self, query: &Search) -> Result<Value> {
        if query
            .since_ms
            .zip(query.until_ms)
            .is_some_and(|(since, until)| since > until)
        {
            bail!("Invalid trace time range");
        }
        if query.status.as_ref().is_some_and(|status| {
            status.is_empty()
                || status.len() > 40
                || !status.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        }) {
            bail!("Invalid trace status filter");
        }
        if query
            .guardrail_policy_sha256
            .as_ref()
            .is_some_and(|hash| !valid_policy_hash(hash))
        {
            bail!("Invalid guardrail policy SHA-256");
        }
        let limit = query.limit.unwrap_or(50);
        if limit == 0 || limit > 200 || query.offset.unwrap_or(0) > 100_000 {
            bail!("Invalid trace pagination");
        }
        let rows = self.scan(query)?;
        let total = rows.len();
        let rows: Vec<_> = rows
            .into_iter()
            .skip(query.offset.unwrap_or(0))
            .take(limit)
            .collect();
        Ok(
            json!({"traces":rows,"total":total,"privacy":{"content_capture":false,"credentials":false,"system_prompts":false}}),
        )
    }
    pub(super) fn time_series(&self, query: &TimeSearch) -> Result<Value> {
        if query.status.as_ref().is_some_and(|status| {
            status.is_empty()
                || status.len() > 40
                || !status.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        }) {
            bail!("Invalid trace series status");
        }
        if query.since_ms > query.until_ms
            || query.until_ms == u64::MAX
            || query.bucket_ms == 0
            || query.bucket_ms > 86_400_000
        {
            bail!("Invalid trace series time range or interval");
        }
        let count = (query.until_ms - query.since_ms) / query.bucket_ms + 1;
        if count > 500 {
            bail!("Trace series exceeds 500 intervals");
        }
        let mut buckets: Vec<TimeBucket> = (0..count).map(|_| TimeBucket::default()).collect();
        let records = self.scan(&Search {
            status: query.status.clone(),
            project_id: query.project_id.clone(),
            session_id: query.session_id.clone(),
            since_ms: Some(query.since_ms),
            until_ms: Some(query.until_ms),
            ..Default::default()
        })?;
        for record in records {
            let index = ((record.started_ms - query.since_ms) / query.bucket_ms) as usize;
            let bucket = &mut buckets[index];
            bucket.traces += 1;
            *bucket.statuses.entry(record.status).or_default() += 1;
            for span in &record.spans {
                bucket.guardrails.observe(span);
                if matches!(span.kind.as_str(), "model" | "external_model") {
                    bucket.models.observe(span)?;
                }
            }
        }
        let rows:Vec<_>=buckets.into_iter().enumerate().map(|(index,bucket)| {
            let start=query.since_ms+(index as u64)*query.bucket_ms;
            let end=start.saturating_add(query.bucket_ms).min(query.until_ms+1);
            let calls=bucket.models.calls;
            let mut usage=bucket.models.receipt("mixed".into(),None,"all_models".into());
            if let Some(object)=usage.as_object_mut(){object.remove("source");object.remove("provider_id");object.remove("model");}
            json!({"start_ms":start,"end_exclusive_ms":end,"trace_count":bucket.traces,"trace_statuses":bucket.statuses,"model_calls":calls,"model_usage":usage,"guardrails":bucket.guardrails.receipt()})
        }).collect();
        Ok(
            json!({"kind":"trace_time_series","schema_version":1,"status":query.status,"since_ms":query.since_ms,"until_ms":query.until_ms,"bucket_ms":query.bucket_ms,"bucket_count":count,"bucket_limit":500,"buckets":rows,"assignment":"trace_started_ms","project_id":query.project_id,"session_id":query.session_id,"provider_calls":0,"privacy":{"content_capture":false}}),
        )
    }
    pub(super) fn summary(&self, query: &SummarySearch) -> Result<Value> {
        if query.status.as_ref().is_some_and(|status| {
            status.is_empty()
                || status.len() > 40
                || !status.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        }) {
            bail!("Invalid summary trace status");
        }
        let conversation_offset = query.conversation_offset.unwrap_or(0);
        let conversation_limit = query.conversation_limit.unwrap_or(100);
        if conversation_offset > 10_000 || !(1..=100).contains(&conversation_limit) {
            bail!("Invalid conversation pagination");
        }
        if query
            .since_ms
            .zip(query.until_ms)
            .is_some_and(|(a, b)| a > b)
        {
            bail!("Invalid summary time range");
        }
        let records = self.scan(&Search {
            status: query.status.clone(),
            project_id: query.project_id.clone(),
            session_id: query.session_id.clone(),
            ..Default::default()
        })?;
        let mut statuses = std::collections::BTreeMap::<String, u64>::new();
        let mut external_model_calls = 0u64;
        let mut model_calls = 0u64;
        let mut traces = 0u64;
        let mut input = 0u64;
        let mut output = 0u64;
        let mut input_unknown = 0u64;
        let mut output_unknown = 0u64;
        let mut cost_unknown = 0u64;
        let mut currency_unknown = 0u64;
        let mut currencies = std::collections::BTreeMap::<String, f64>::new();
        let mut cost_examples = Vec::new();
        let mut models =
            std::collections::BTreeMap::<(String, Option<String>, String), ModelTotals>::new();
        let mut guardrails = GuardrailSummary::default();
        let mut conversations =
            std::collections::BTreeMap::<(String, String), ConversationTotals>::new();
        for record in records {
            if query
                .since_ms
                .is_some_and(|since| record.started_ms < since)
                || query
                    .until_ms
                    .is_some_and(|until| record.started_ms > until)
            {
                continue;
            }
            traces += 1;
            *statuses.entry(record.status.clone()).or_default() += 1;
            let conversation = conversations
                .entry((record.project_id.clone(), record.session_id.clone()))
                .or_default();
            conversation.traces += 1;
            *conversation
                .statuses
                .entry(record.status.clone())
                .or_default() += 1;
            conversation.first = Some(
                conversation
                    .first
                    .map_or(record.started_ms, |first| first.min(record.started_ms)),
            );
            conversation.last = conversation.last.max(record.started_ms);
            for span in &record.spans {
                guardrails.observe(span);
            }
            for span in record
                .spans
                .iter()
                .filter(|span| span.kind == "model" || span.kind == "external_model")
            {
                conversation.models.observe(span)?;
                if span.kind == "external_model" {
                    conversation.external_calls += 1;
                }
                let source = if span.kind == "external_model" {
                    "external"
                } else {
                    "native"
                };
                models
                    .entry((source.into(), span.provider_id.clone(), span.name.clone()))
                    .or_default()
                    .observe(span)?;
                model_calls += 1;
                if span.kind == "external_model" {
                    external_model_calls += 1;
                }
                match span.usage["prompt_tokens"]
                    .as_u64()
                    .or_else(|| span.usage["input_tokens"].as_u64())
                {
                    Some(n) => {
                        input = input.checked_add(n).context("Input token total overflow")?
                    }
                    None => input_unknown += 1,
                }
                match span.usage["completion_tokens"]
                    .as_u64()
                    .or_else(|| span.usage["output_tokens"].as_u64())
                {
                    Some(n) => {
                        output = output
                            .checked_add(n)
                            .context("Output token total overflow")?
                    }
                    None => output_unknown += 1,
                }
                if let Some(cost) = span.usage["cost"]
                    .as_f64()
                    .filter(|cost| cost.is_finite() && *cost >= 0.0)
                {
                    let currency = span.usage["cost_currency"].as_str().filter(|code| {
                        code.len() == 3 && code.bytes().all(|b| b.is_ascii_uppercase())
                    });
                    if let Some(currency) = currency {
                        let total = currencies.entry(currency.into()).or_default();
                        *total += cost;
                        if !total.is_finite() {
                            bail!("Cost total overflow");
                        }
                    } else {
                        currency_unknown += 1;
                    }
                    if cost_examples.len() < 100 {
                        cost_examples.push(json!({"trace_id":record.id,"span_id":span.id,"model":span.name,"value":cost,"currency":currency}));
                    }
                } else {
                    cost_unknown += 1;
                }
            }
        }
        let (cache_write, cache_read, cache_write_unknown, cache_read_unknown) = models
            .values()
            .try_fold((0u64, 0u64, 0u64, 0u64), |a, t| -> Result<_> {
                Ok((
                    a.0.checked_add(t.cache_write)
                        .context("Cache creation total overflow")?,
                    a.1.checked_add(t.cache_read)
                        .context("Cache read total overflow")?,
                    a.2 + t.cache_write_unknown,
                    a.3 + t.cache_read_unknown,
                ))
            })?;
        let first_text = first_text_receipt(
            models
                .values()
                .flat_map(|model| model.first_text.iter().copied())
                .collect(),
            models.values().map(|model| model.first_text_unknown).sum(),
        );
        let models: Vec<_> = models
            .into_iter()
            .map(|((source, provider_id, model), totals)| {
                totals.receipt(source, provider_id, model)
            })
            .collect();
        let conversation_count = conversations.len();
        let mut conversations: Vec<_> = conversations.into_iter().collect();
        conversations.sort_by(|a, b| b.1.last.cmp(&a.1.last).then_with(|| a.0.cmp(&b.0)));
        let conversations:Vec<_>=conversations.into_iter().skip(conversation_offset).take(conversation_limit).map(|((project_id,session_id),totals)| {
            let mut usage=totals.models.receipt("mixed".into(),None,"all_models".into());
            if let Some(object)=usage.as_object_mut(){object.remove("source");object.remove("provider_id");object.remove("model");}
            json!({"project_id":project_id,"session_id":session_id,"trace_count":totals.traces,"trace_statuses":totals.statuses,"first_started_ms":totals.first,"last_started_ms":totals.last,"external_model_calls":totals.external_calls,"model_usage":usage})
        }).collect();
        Ok(
            json!({"status":query.status,"conversations":conversations,"conversation_count":conversation_count,"conversation_offset":conversation_offset,"conversation_limit":conversation_limit,"conversation_has_more":conversation_offset+conversations.len()<conversation_count,"conversations_truncated":conversation_offset>0||conversation_count>conversations.len(),"conversation_order":"last_started_ms_desc_project_session_asc","first_text":first_text,"guardrails":guardrails.receipt(),"models":models,"cache_creation_input_tokens":cache_write,"cache_read_input_tokens":cache_read,"cache_creation_unknown_calls":cache_write_unknown,"cache_read_unknown_calls":cache_read_unknown,"trace_count":traces,"trace_statuses":statuses,"model_calls":model_calls,"external_model_calls":external_model_calls,"input_tokens":input,"output_tokens":output,"input_tokens_unknown_calls":input_unknown,"output_tokens_unknown_calls":output_unknown,"reported_cost_by_currency":currencies,"cost_unknown_calls":cost_unknown,"cost_currency_unknown_calls":currency_unknown,"reported_cost_examples":cost_examples,"cost_examples_limit":100,"scope":"model_spans","price_estimates":false,"unknown_currency_costs_summed":false,"since_ms":query.since_ms,"until_ms":query.until_ms}),
        )
    }
}
