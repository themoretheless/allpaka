//! The persisted trace record and its spans, plus per-model usage totals
//! and latency quantiles derived from them.

use super::*;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Record {
    pub(super) id: String,
    pub(super) session_id: String,
    pub(super) project_id: String,
    pub(super) started_ms: u64,
    pub(super) status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) recovered_ms: Option<u64>,
    pub(super) spans: Vec<Span>,
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Span {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) provider_id: Option<String>,
    pub(super) id: usize,
    pub(super) parent_id: Option<usize>,
    pub(super) kind: String,
    pub(super) name: String,
    pub(super) status: String,
    pub(super) started_ms: u64,
    pub(super) duration_ms: Option<u64>,
    pub(super) usage: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) linked_trace_id: Option<String>,
}
#[derive(Default)]
pub(super) struct ModelTotals {
    pub(super) calls: u64,
    pub(super) input: u64,
    pub(super) output: u64,
    pub(super) input_unknown: u64,
    pub(super) output_unknown: u64,
    pub(super) cache_write: u64,
    pub(super) cache_read: u64,
    pub(super) cache_write_unknown: u64,
    pub(super) cache_read_unknown: u64,
    pub(super) cost_unknown: u64,
    pub(super) currency_unknown: u64,
    pub(super) duration_unknown: u64,
    pub(super) first_text_unknown: u64,
    pub(super) first_text: Vec<u64>,
    pub(super) statuses: std::collections::BTreeMap<String, u64>,
    pub(super) costs: std::collections::BTreeMap<String, f64>,
    pub(super) durations: Vec<u64>,
}
impl ModelTotals {
    pub(super) fn observe(&mut self, span: &Span) -> Result<()> {
        self.calls += 1;
        *self.statuses.entry(span.status.clone()).or_default() += 1;
        match span.usage["prompt_tokens"]
            .as_u64()
            .or_else(|| span.usage["input_tokens"].as_u64())
        {
            Some(n) => {
                self.input = self
                    .input
                    .checked_add(n)
                    .context("Model input total overflow")?
            }
            None => self.input_unknown += 1,
        }
        match span.usage["completion_tokens"]
            .as_u64()
            .or_else(|| span.usage["output_tokens"].as_u64())
        {
            Some(n) => {
                self.output = self
                    .output
                    .checked_add(n)
                    .context("Model output total overflow")?
            }
            None => self.output_unknown += 1,
        }
        match span.usage["cache_creation_input_tokens"].as_u64() {
            Some(n) => {
                self.cache_write = self
                    .cache_write
                    .checked_add(n)
                    .context("Cache creation total overflow")?
            }
            None => self.cache_write_unknown += 1,
        }
        match span.usage["cache_read_input_tokens"]
            .as_u64()
            .or_else(|| span.usage["cached_tokens"].as_u64())
        {
            Some(n) => {
                self.cache_read = self
                    .cache_read
                    .checked_add(n)
                    .context("Cache read total overflow")?
            }
            None => self.cache_read_unknown += 1,
        }
        match span.duration_ms {
            Some(n) => self.durations.push(n),
            None => self.duration_unknown += 1,
        }
        match span.usage["first_text_ms"]
            .as_u64()
            .filter(|first| span.duration_ms.is_none_or(|total| *first <= total))
        {
            Some(first) => self.first_text.push(first),
            None => self.first_text_unknown += 1,
        }
        if let Some(cost) = span.usage["cost"]
            .as_f64()
            .filter(|n| n.is_finite() && *n >= 0.0)
        {
            if let Some(currency) = span.usage["cost_currency"]
                .as_str()
                .filter(|s| s.len() == 3 && s.bytes().all(|b| b.is_ascii_uppercase()))
            {
                let total = self.costs.entry(currency.into()).or_default();
                *total += cost;
                if !total.is_finite() {
                    bail!("Model cost overflow");
                }
            } else {
                self.currency_unknown += 1;
            }
        } else {
            self.cost_unknown += 1;
        }
        Ok(())
    }
    pub(super) fn receipt(mut self, source: String, provider_id: Option<String>, model: String) -> Value {
        self.durations.sort_unstable();
        let percentile = |percent: usize| {
            if self.durations.is_empty() {
                None
            } else {
                Some(self.durations[(self.durations.len() * percent).div_ceil(100) - 1])
            }
        };
        json!({"first_text":first_text_receipt(self.first_text,self.first_text_unknown),"source":source,"provider_id":provider_id,"model":model,"calls":self.calls,"statuses":self.statuses,
            "input_tokens":self.input,"output_tokens":self.output,"input_tokens_unknown_calls":self.input_unknown,"output_tokens_unknown_calls":self.output_unknown,
            "cache_creation_input_tokens":self.cache_write,"cache_read_input_tokens":self.cache_read,"cache_creation_unknown_calls":self.cache_write_unknown,"cache_read_unknown_calls":self.cache_read_unknown,
            "reported_cost_by_currency":self.costs,"cost_unknown_calls":self.cost_unknown,"cost_currency_unknown_calls":self.currency_unknown,
            "duration_known_calls":self.durations.len(),"duration_unknown_calls":self.duration_unknown,
            "duration_min_ms":self.durations.first(),"duration_max_ms":self.durations.last(),"duration_p50_ms":percentile(50),"duration_p95_ms":percentile(95),
            "duration_quantile_method":"nearest_rank","duration_scope":"whole_model_span"})
    }
}
pub(super) fn first_text_receipt(mut values: Vec<u64>, unknown: u64) -> Value {
    values.sort_unstable();
    let percentile = |percent: usize| {
        if values.is_empty() {
            None
        } else {
            Some(values[(values.len() * percent).div_ceil(100) - 1])
        }
    };
    json!({"known_calls":values.len(),"unknown_calls":unknown,"min_ms":values.first(),"max_ms":values.last(),"p50_ms":percentile(50),"p95_ms":percentile(95),"quantile_method":"nearest_rank","scope":"client_observed_first_text","server_token_time":false})
}
