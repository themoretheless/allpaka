//! Bounded, cancellable dataset runs using the native provider implementation.
pub(crate) mod matrix_jobs;
use crate::{
    evaluation::{Sample, Snapshot},
    provider,
    types::{Message, Mode, Settings},
};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
const MAX_RUN_BYTES: usize = 32 * 1024 * 1024;
const MAX_RUNS:usize=1000;
const MAX_CATALOG_BYTES:u64=128*1024*1024;
const MAX_OUTPUT_JSON_BYTES:usize=128*1024+16;
const PROGRESS_RESERVE:usize=16*1024;
fn run_reservation(run:&Run)->Result<usize>{
    let mut fixed=run.clone();fixed.status.clear();fixed.trace_id=None;fixed.recovered_ms=None;fixed.strict_quality=false;fixed.mean_scores.clear();
    for item in &mut fixed.items {item.status.clear();item.output=None;item.output_truncated=false;item.scores.clear();item.usage=Value::Null;item.duration_ms=None;item.error=None;}
    let size=serde_json::to_vec(&fixed)?.len()+PROGRESS_RESERVE+fixed.items.len()*(MAX_OUTPUT_JSON_BYTES+PROGRESS_RESERVE);
    if size>MAX_RUN_BYTES {bail!("Experiment cannot reserve durable output space");}Ok(size)
}
fn bound_output(mut output:String)->Result<(String,bool)>{
    let original=output.len();let mut end=original.min(65536);while !output.is_char_boundary(end){end-=1;}output.truncate(end);
    if serde_json::to_vec(&output)?.len()>MAX_OUTPUT_JSON_BYTES {
        let(mut low,mut high)=(0,output.len());while low<high {let mid=(low+high+1)/2;let mut boundary=mid;while !output.is_char_boundary(boundary){boundary-=1;}if serde_json::to_vec(&output[..boundary])?.len()<=MAX_OUTPUT_JSON_BYTES{low=mid;}else{high=mid-1;}}
        while !output.is_char_boundary(low){low-=1;}output.truncate(low);
    }
    let truncated=output.len()!=original;Ok((output,truncated))
}
#[derive(Clone)]
pub(crate) struct Manager {
    root: Arc<PathBuf>,
    writes: Arc<Mutex<()>>,
    active: Arc<Mutex<HashMap<String, Option<oneshot::Sender<()>>>>>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Metric {
    ExactMatch,
    ContainsReference,
    JsonValid,
    WhitespaceTokenF1,
    CharacterBigramF1,
    JsonEquals,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OfflineScore {
    dataset_id: String,
    dataset_version: u64,
    metrics: Vec<Metric>,
    outputs: BTreeMap<String, String>,
}
pub(crate) async fn offline_score(
    State(app): State<crate::App>,
    Json(input): Json<OfflineScore>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        if input.dataset_version == 0 {
            bail!("Pin an explicit dataset version");
        }
        let snapshot = app
            .evaluation
            .load(&input.dataset_id, Some(input.dataset_version))?;
        validate_metrics(&input.metrics, &snapshot)?;
        if snapshot.samples.is_empty()
            || snapshot.samples.len() > 200
            || input.outputs.len() != snapshot.samples.len()
        {
            bail!("Provide exactly one output per sample, up to 200 samples");
        }
        let mut items = Vec::new();
        let mut means = BTreeMap::<String, f64>::new();
        for sample in &snapshot.samples {
            let output = input
                .outputs
                .get(&sample.id)
                .context("Missing sample output")?;
            if output.len() > 64000 {
                bail!("Offline output exceeds 64000 bytes");
            }
            let scores = score(sample, output, &input.metrics);
            for (metric, value) in &scores {
                *means.entry(metric.clone()).or_default() += value / snapshot.samples.len() as f64;
            }
            items.push(json!({"sample_id":sample.id,"output":output,"scores":scores}));
        }
        let id = crate::evaluation::new_id();
        let receipt = json!({"id":id,"kind":"offline_score","project_id":snapshot.project_id,"dataset_id":snapshot.id,"dataset_version":snapshot.version,"dataset_sha256":snapshot.sha256,"metrics":input.metrics,"items":items,"mean_scores":means,"provider_calls":0});
        let directory = app.data.join("evaluation/offline_scores");
        std::fs::create_dir_all(&directory)?;
        let directory = directory.canonicalize()?;
        if !directory.starts_with(app.data.as_ref()) {
            bail!("Offline scoring storage escapes Studio data");
        }
        crate::evaluation::commit_new(&directory.join(format!("{id}.json")), &receipt)?;
        Ok(receipt)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn read_offline_score(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        if id.is_empty()
            || id.len() > 80
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            bail!("Invalid score ID");
        }
        let path = app
            .data
            .join("evaluation/offline_scores")
            .join(format!("{id}.json"));
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
            bail!("Invalid offline receipt file");
        }
        let path = path.canonicalize()?;
        if !path.starts_with(app.data.as_ref()) {
            bail!("Offline receipt escapes Studio data");
        }
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 16 * 1024 * 1024 {
            bail!("Offline receipt too large");
        }
        let receipt: Value = crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        if receipt["id"] != id
            || receipt["kind"] != "offline_score"
            || receipt["provider_calls"] != 0
        {
            bail!("Offline identity mismatch");
        }
        let version = receipt["dataset_version"]
            .as_u64()
            .filter(|v| *v > 0)
            .context("Missing pinned version")?;
        let snapshot = app.evaluation.load(
            receipt["dataset_id"].as_str().context("Missing dataset")?,
            Some(version),
        )?;
        if receipt["dataset_sha256"] != snapshot.sha256
            || receipt["project_id"] != snapshot.project_id
        {
            bail!("Offline dataset mismatch");
        }
        let metrics: Vec<Metric> = serde_json::from_value(receipt["metrics"].clone())?;
        validate_metrics(&metrics, &snapshot)?;
        let items = receipt["items"].as_array().context("Missing score items")?;
        if snapshot.samples.is_empty()
            || snapshot.samples.len() > 200
            || items.len() != snapshot.samples.len()
        {
            bail!("Offline sample coverage mismatch");
        }
        let mut means = BTreeMap::<String, f64>::new();
        for (sample, item) in snapshot.samples.iter().zip(items) {
            let output = item["output"].as_str().context("Missing output")?;
            if item["sample_id"] != sample.id || output.len() > 64000 {
                bail!("Offline sample mismatch");
            }
            let scores = score(sample, output, &metrics);
            if item["scores"] != serde_json::to_value(&scores)? {
                bail!("Offline score integrity failure");
            }
            for (metric, value) in scores {
                *means.entry(metric).or_default() += value / snapshot.samples.len() as f64;
            }
        }
        if receipt["mean_scores"] != serde_json::to_value(means)? {
            bail!("Offline mean integrity failure");
        }
        Ok(receipt)
    })();
    result.map(Json).map_err(|_| {
        crate::error(
            StatusCode::BAD_REQUEST,
            "Offline scoring receipt unavailable or invalid",
        )
    })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OfflineCatalog {
    project_id: String,
    dataset_id: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}
pub(crate) async fn list_offline_scores(
    State(app): State<crate::App>,
    Query(query): Query<OfflineCatalog>,
) -> crate::ApiResult<Value> {
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(20);
    let prepare = (|| -> Result<Vec<String>> {
        if query.project_id.is_empty()
            || query.project_id.len() > 80
            || offset > 1000
            || !(1..=100).contains(&limit)
        {
            bail!("Invalid offline catalog bounds");
        }
        let directory = app.data.join("evaluation/offline_scores");
        if !directory.exists() {
            return Ok(Vec::new());
        }
        if std::fs::symlink_metadata(&directory)?
            .file_type()
            .is_symlink()
            || !directory.canonicalize()?.starts_with(app.data.as_ref())
        {
            bail!("Invalid offline catalog directory");
        }
        let mut ids = Vec::new();
        for (index, entry) in std::fs::read_dir(directory)?.enumerate() {
            if index >= 10000 {
                bail!("Offline catalog directory exceeds scan bound");
            }
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(id) = path.file_stem().and_then(|name| name.to_str()) {
                    ids.push(id.to_owned());
                }
            }
        }
        ids.sort();
        ids.reverse();
        Ok(ids)
    })()
    .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))?;
    let mut rows = Vec::new();
    let mut invalid = 0usize;
    let mut scanned = 0usize;
    let mut bytes = 0u64;
    let mut truncated = false;
    for id in prepare {
        let path = app
            .data
            .join("evaluation/offline_scores")
            .join(format!("{id}.json"));
        let size = std::fs::symlink_metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        if scanned >= 1000 || bytes.saturating_add(size) > 64 * 1024 * 1024 {
            truncated = true;
            break;
        }
        scanned += 1;
        bytes = bytes.saturating_add(size);
        match read_offline_score(State(app.clone()), Path(id)).await {
            Ok(Json(receipt)) => {
                if receipt["project_id"] != query.project_id
                    || query
                        .dataset_id
                        .as_ref()
                        .is_some_and(|id| receipt["dataset_id"] != *id)
                {
                    continue;
                }
                rows.push(json!({"id":receipt["id"],"project_id":receipt["project_id"],"dataset_id":receipt["dataset_id"],
                    "dataset_version":receipt["dataset_version"],"dataset_sha256":receipt["dataset_sha256"],
                    "metrics":receipt["metrics"],"mean_scores":receipt["mean_scores"],"sample_count":receipt["items"].as_array().map_or(0,Vec::len),"provider_calls":0}));
            }
            Err(_) => invalid += 1,
        }
    }
    let total = rows.len();
    let items = rows
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();
    Ok(Json(
        json!({"scores":items,"offset":offset,"limit":limit,"total":total,"has_more":offset.saturating_add(limit)<total,
        "scanned":scanned,"invalid_receipts":invalid,"truncated":truncated,"order":"id_desc","provider_calls":0}),
    ))
}
pub(crate) async fn compare_offline_scores(
    State(app): State<crate::App>,
    Json(input): Json<Compare>,
) -> crate::ApiResult<Value> {
    let baseline = read_offline_score(State(app.clone()), Path(input.baseline_id))
        .await?
        .0;
    let candidate = read_offline_score(State(app.clone()), Path(input.candidate_id))
        .await?
        .0;
    let result = (|| -> Result<Value> {
        for field in [
            "project_id",
            "dataset_id",
            "dataset_version",
            "dataset_sha256",
            "metrics",
        ] {
            if baseline[field] != candidate[field] {
                bail!("Offline comparison requires identical dataset and metrics");
            }
        }
        let mut regressions = 0usize;
        let mut improvements = 0usize;
        let mut pairs = Vec::new();
        for (base, next) in baseline["items"]
            .as_array()
            .context("Missing baseline rows")?
            .iter()
            .zip(
                candidate["items"]
                    .as_array()
                    .context("Missing candidate rows")?,
            )
        {
            for (metric, score) in base["scores"].as_object().context("Missing scores")? {
                let score = score.as_f64().context("Invalid score")?;
                let value = next["scores"][metric]
                    .as_f64()
                    .context("Missing candidate score")?;
                let delta = value - score;
                regressions += usize::from(delta < 0.0);
                improvements += usize::from(delta > 0.0);
                pairs.push(json!({"sample_id":base["sample_id"],"metric":metric,"baseline":score,"candidate":value,"delta":delta}));
            }
        }
        let id = crate::evaluation::new_id();
        let receipt = json!({"id":id,"kind":"offline_comparison","baseline_id":baseline["id"],"candidate_id":candidate["id"],"project_id":baseline["project_id"],"dataset_id":baseline["dataset_id"],"dataset_version":baseline["dataset_version"],"dataset_sha256":baseline["dataset_sha256"],"metrics":baseline["metrics"],"eligible":regressions==0&&improvements>0,"regressions":regressions,"improvements":improvements,"pairs":pairs,"observational_only":true,"provider_calls":0});
        let directory = app.data.join("evaluation/offline_comparisons");
        std::fs::create_dir_all(&directory)?;
        let directory = directory.canonicalize()?;
        if !directory.starts_with(app.data.as_ref()) {
            bail!("Offline comparison storage escapes Studio data");
        }
        crate::evaluation::commit_new(&directory.join(format!("{id}.json")), &receipt)?;
        Ok(receipt)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn metric_catalog() -> Json<Value> {
    let metrics = [
        (Metric::ExactMatch, "Exact text match", "required", "Exact UTF-8 string equality; no trimming or case normalization"),
        (Metric::ContainsReference, "Reference containment", "nonempty", "Case-sensitive literal reference substring in output"),
        (Metric::JsonValid, "Valid JSON", "none", "Output parses as a serde_json value"),
        (Metric::WhitespaceTokenF1, "Whitespace token F1", "nonempty_tokens", "Case-sensitive Unicode-whitespace multiset overlap: 2*overlap/(reference_tokens+output_tokens)"),
        (Metric::CharacterBigramF1, "Character bigram F1", "required", "Case-sensitive Unicode scalar adjacent-pair multiset F1; preserves whitespace and punctuation; strings shorter than two scalars use exact equality"),
        (Metric::JsonEquals, "Structural JSON equality", "valid_json", "JSON values with unique object keys are equal; ignores object key order and formatting, preserves array order and types"),
    ].into_iter().map(|(id,name,reference,description)|json!({"id":id,"name":name,"reference_requirement":reference,"description":description,"min_score":0,"max_score":1,"higher_is_better":true,"provider_calls":0,"kind":"deterministic"})).collect::<Vec<_>>();
    Json(json!({"metrics":metrics,"max_metrics_per_run":6}))
}
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Item {
    sample_id: String,
    status: String,
    output: Option<String>,
    #[serde(default)]
    output_truncated: bool,
    scores: BTreeMap<String, f64>,
    usage: Value,
    duration_ms: Option<u64>,
    error: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Run {
    id: String,
    project_id: String,
    dataset_id: String,
    dataset_version: u64,
    dataset_sha256: String,
    #[serde(default)]
    trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recovered_ms: Option<u64>,
    settings: Settings,
    prompt_template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_snapshot: Option<crate::prompts::Snapshot>,
    metrics: Vec<Metric>,
    #[serde(default,skip_serializing_if="is_false")]
    playground:bool,
    concurrency: usize,
    item_timeout_secs: u64,
    status: String,
    items: Vec<Item>,
    strict_quality: bool,
    mean_scores: BTreeMap<String, f64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Start {
    dataset_id: String,
    dataset_version: u64,
    settings: Settings,
    #[serde(default)]
    prompt_template: String,
    #[serde(default)]
    prompt_ref: Option<PromptRef>,
    #[serde(skip)]
    prompt_snapshot: Option<crate::prompts::Snapshot>,
    #[serde(skip)]
    playground:bool,
    metrics: Vec<Metric>,
    #[serde(default = "default_concurrency")]
    concurrency: usize,
    #[serde(default = "default_timeout")]
    item_timeout_secs: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRef {
    id: String,
    version: u64,
}
fn is_false(value:&bool)->bool{!*value}
fn default_concurrency() -> usize {
    4
}
fn default_timeout() -> u64 {
    120
}
fn render_prompt(template: &str, sample: &Sample) -> String {
    let contexts = sample.contexts.join("\n\n");
    template
        .split("{{input}}")
        .map(|part| part.replace("{{contexts}}", &contexts))
        .collect::<Vec<_>>()
        .join(&sample.input)
}
// Case-sensitive Unicode-whitespace tokens; repeated tokens count as a multiset.
fn whitespace_token_f1(reference: &str, output: &str) -> f64 {
    let mut counts = BTreeMap::new();
    let mut reference_count = 0usize;
    for token in reference.split_whitespace() {
        *counts.entry(token).or_insert(0usize) += 1;
        reference_count += 1;
    }
    let mut output_count = 0usize;
    let mut overlap = 0usize;
    for token in output.split_whitespace() {
        output_count += 1;
        if let Some(count) = counts.get_mut(token) {
            if *count > 0 {
                overlap += 1;
                *count -= 1;
            }
        }
    }
    if reference_count + output_count == 0 {
        return 1.0;
    }
    2.0 * overlap as f64 / (reference_count + output_count) as f64
}
fn character_bigram_f1(reference: &str, output: &str) -> f64 {
    let reference: Vec<char> = reference.chars().collect();
    let output: Vec<char> = output.chars().collect();
    if reference.len() < 2 || output.len() < 2 {
        return f64::from(reference == output);
    }
    let mut counts = BTreeMap::new();
    for pair in reference.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0usize) += 1;
    }
    let mut overlap = 0usize;
    for pair in output.windows(2) {
        if let Some(count) = counts.get_mut(&(pair[0], pair[1])) {
            if *count > 0 {
                overlap += 1;
                *count -= 1;
            }
        }
    }
    2.0 * overlap as f64 / (reference.len() + output.len() - 2) as f64
}
fn json_equals(reference: &str, output: &str) -> bool {
    match (
        crate::strict_json::parse(reference),
        crate::strict_json::parse(output),
    ) {
        (Ok(reference), Ok(output)) => reference == output,
        _ => false,
    }
}
fn score(sample: &Sample, output: &str, metrics: &[Metric]) -> BTreeMap<String, f64> {
    metrics
        .iter()
        .map(|metric| match metric {
            Metric::ExactMatch => (
                "exact_match".into(),
                f64::from(sample.expected_output.as_deref() == Some(output)),
            ),
            Metric::ContainsReference => (
                "contains_reference".into(),
                f64::from(
                    sample
                        .expected_output
                        .as_ref()
                        .is_some_and(|s| output.contains(s)),
                ),
            ),
            Metric::CharacterBigramF1 => (
                "character_bigram_f1".into(),
                character_bigram_f1(sample.expected_output.as_deref().unwrap_or(""), output),
            ),
            Metric::JsonEquals => (
                "json_equals".into(),
                f64::from(json_equals(
                    sample.expected_output.as_deref().unwrap_or(""),
                    output,
                )),
            ),
            Metric::WhitespaceTokenF1 => (
                "whitespace_token_f1".into(),
                whitespace_token_f1(sample.expected_output.as_deref().unwrap_or(""), output),
            ),
            Metric::JsonValid => (
                "json_valid".into(),
                f64::from(serde_json::from_str::<Value>(output).is_ok()),
            ),
        })
        .collect()
}
fn validate_metrics(metrics: &[Metric], snapshot: &Snapshot) -> Result<()> {
    if metrics.is_empty()
        || metrics.len() > 6
        || metrics
            .iter()
            .enumerate()
            .any(|(i, m)| metrics[..i].contains(m))
    {
        bail!("Select 1–6 unique metrics");
    }
    if metrics.iter().any(|m| {
        matches!(
            m,
            Metric::ExactMatch
                | Metric::CharacterBigramF1
                | Metric::ContainsReference
                | Metric::WhitespaceTokenF1
                | Metric::JsonEquals
        )
    }) && snapshot.samples.iter().any(|s| s.expected_output.is_none())
    {
        bail!("Reference metrics require expected_output on every sample");
    }
    if metrics.contains(&Metric::ContainsReference)
        && snapshot
            .samples
            .iter()
            .any(|s| s.expected_output.as_deref() == Some(""))
    {
        bail!("Contains-reference metric requires nonempty references");
    }
    if metrics.contains(&Metric::WhitespaceTokenF1)
        && snapshot.samples.iter().any(|s| {
            s.expected_output
                .as_deref()
                .is_none_or(|text| text.split_whitespace().next().is_none())
        })
    {
        bail!("Token F1 requires nonempty token references");
    }
    if metrics.contains(&Metric::JsonEquals)
        && snapshot.samples.iter().any(|s| {
            s.expected_output
                .as_deref()
                .is_none_or(|text| crate::strict_json::parse(text).is_err())
        })
    {
        bail!("JSON equality requires valid JSON references");
    }
    Ok(())
}
fn validate(request: &Start, snapshot: &Snapshot) -> Result<()> {
    if !(1..=8).contains(&request.concurrency)
        || !(1..=600).contains(&request.item_timeout_secs)
        || snapshot.samples.len() > 200
    {
        bail!("Runs support 1–200 samples, 1–8 concurrent requests and 1–600 seconds per item");
    }
    let chat=request.prompt_snapshot.as_ref().is_some_and(|p|p.messages.is_some());
    if !chat&&(request.prompt_template.len() > 16000 || !request.prompt_template.contains("{{input}}")) {
        bail!("Prompt template must contain {{input}} and fit 16000 bytes");
    }
    if request.playground&&snapshot.samples.len()!=1 {bail!("Playground accepts one sample");}
    if !request.playground||!request.metrics.is_empty(){validate_metrics(&request.metrics, snapshot)?;}
    if request.settings.project_id != snapshot.project_id {
        bail!("Dataset and run projects must match");
    }
    if request.settings.mode != Mode::Chat || request.settings.allow_writes {
        bail!("Evaluation runs require Chat settings without workspace writes");
    }
    Ok(())
}
impl Manager {
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("evaluation/runs");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if !root.starts_with(data) {
            bail!("Experiment storage escapes Studio data directory");
        }
        Ok(Self {
            root: Arc::new(root),
            writes: Default::default(),
            active: Default::default(),
        })
    }
    /// Called only at startup under the exclusive Studio directory lock.
    pub(crate) fn recover(&self) -> Result<usize> {
        let mut count = 0;
        let mut bytes = 0;
        let mut recovered = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|s| s != "json") || !entry.file_type()?.is_file() {
                continue;
            }
            count += 1;
            bytes += entry.metadata()?.len();
            if count > MAX_RUNS || bytes > MAX_CATALOG_BYTES {
                bail!("Experiment recovery scan limit reached");
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("Invalid experiment filename")?;
            let mut run = self.load(id)?;
            if run.status == "running" {
                run.status = "interrupted".into();
                run.strict_quality = false;
                run.mean_scores.clear();
                run.recovered_ms = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                        .min(u64::MAX as u128) as u64,
                );
                for item in &mut run.items {
                    if item.status == "pending" || item.status == "running" {
                        item.status = "interrupted".into();
                        item.error = Some("process_restart".into());
                    }
                }
                self.save(&run)?;
                recovered += 1;
            }
        }
        Ok(recovered)
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty() || id.len() > 80 || !id.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')
        {
            bail!("Invalid experiment ID");
        }
        Ok(self.root.join(format!("{id}.json")))
    }
    fn save(&self,run:&Run)->Result<()>{self.save_with(run,||Ok(()))}
    fn save_with(&self, run: &Run,before_write:impl FnOnce()->Result<()>) -> Result<()> {
        use std::io::Write;
        let bytes = serde_json::to_vec(run)?;
        if bytes.len() > MAX_RUN_BYTES {
            bail!("Experiment artifact exceeds 32 MiB");
        }
        let _writer = self.writes.lock().unwrap();
        let path = self.path(&run.id)?;
        if !path.try_exists()? {
            let mut count=0usize;let mut reserved=run_reservation(run)? as u64;
            for entry in std::fs::read_dir(self.root.as_ref())? {
                let entry=entry?;let path=entry.path();if path.extension().is_none_or(|ext|ext!="json"){continue;}
                count+=1;if count>=MAX_RUNS {bail!("Experiment retention limit reached; new run was not started");}
                let metadata=entry.file_type()?;if !metadata.is_file(){bail!("Invalid experiment catalog entry");}
                let id=path.file_stem().and_then(|id|id.to_str()).context("Invalid experiment filename")?;let existing=self.load(id)?;
                let weight=if existing.status=="running" {run_reservation(&existing)? as u64}else{entry.metadata()?.len().checked_add(PROGRESS_RESERVE as u64).context("Experiment capacity overflow")?};
                reserved=reserved.checked_add(weight).context("Experiment capacity overflow")?;
                if reserved>MAX_CATALOG_BYTES {bail!("Experiment storage capacity reached; new run was not started");}
            }
        }
        before_write()?;
        let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
        let result = (|| -> Result<()> {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
            let mut file = options.open(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temp, path)?;
            std::fs::File::open(self.root.as_ref())?.sync_all()?;
            Ok(())
        })();
        let _ = std::fs::remove_file(temp);
        result
    }
    fn load(&self, id: &str) -> Result<Run> {
        let path = self.path(id)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() > MAX_RUN_BYTES as u64 {
            bail!("Invalid experiment artifact");
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX_RUN_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_RUN_BYTES {
            bail!("Oversize experiment artifact");
        }
        let run: Run = serde_json::from_slice(&bytes)?;
        if run.id != id {
            bail!("Experiment identity mismatch");
        }
        Ok(run)
    }
    fn list(&self, query:&ListQuery) -> Result<Value> {
        let offset=query.offset.unwrap_or(0);let paged=query.offset.is_some()||query.limit.is_some();let limit=query.limit.unwrap_or(if paged {100}else{1000});
        if offset>1000||limit==0||(paged&&limit>100){bail!("Invalid experiment pagination");}
        if query.status.as_deref().is_some_and(|s|!matches!(s,"running"|"completed"|"failed"|"cancelled"|"interrupted")){bail!("Invalid experiment status filter");}
        for id in [query.provider.as_deref(),query.dataset_id.as_deref()].into_iter().flatten(){if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_'){bail!("Invalid experiment identity filter");}}
        if query.model.as_ref().is_some_and(|m|m.chars().count()>200||m.len()>800){bail!("Experiment model search exceeds bounds");}
        let model=query.model.as_deref().unwrap_or("").trim().to_lowercase();
        let mut rows = Vec::new();
        let mut bytes = 0u64;
        let mut count = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|s| s != "json") || !entry.file_type()?.is_file() {
                continue;
            }
            count += 1;
            bytes += entry.metadata()?.len();
            if count > 1000 || bytes > 128 * 1024 * 1024 {
                bail!("Experiment catalog scan limit reached");
            }
            let run = self.load(
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .context("Invalid experiment file")?,
            )?;
            if run.project_id == query.project_id&&query.status.as_ref().is_none_or(|s|s==&run.status)&&query.provider.as_ref().is_none_or(|p|p==&run.settings.provider)&&query.dataset_id.as_ref().is_none_or(|id|id==&run.dataset_id)&&query.playground.is_none_or(|v|v==run.playground)&&(model.is_empty()||run.settings.model.to_lowercase().contains(&model)) {
                rows.push(json!({"project_id":run.project_id,"id":run.id,"dataset_id":run.dataset_id,"dataset_version":run.dataset_version,"dataset_sha256":run.dataset_sha256,"status":run.status,"strict_quality":run.strict_quality,"mean_scores":run.mean_scores,"items":run.items.len(),"provider":run.settings.provider,"model":run.settings.model,"playground":run.playground}));
            }
        }
        rows.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
        let total=rows.len();let rows:Vec<_>=rows.into_iter().skip(offset).take(limit).collect();
        Ok(json!({"runs":rows,"total":total,"offset":offset,"limit":limit,"has_more":offset+rows.len()<total,"order":"id_desc","provider_calls":0,"summaries_verified":false}))
    }
    fn start(
        &self,
        app: crate::App,
        request: Start,
        snapshot: Snapshot,
        p: provider::Provider,
    ) -> Result<Run> {
        self.start_reserved(app, request, snapshot, p, crate::evaluation::new_id())
    }
    fn start_reserved(
        &self,
        app: crate::App,
        mut request: Start,
        snapshot: Snapshot,
        p: provider::Provider,
        id: String,
    ) -> Result<Run> {
        validate(&request, &snapshot)?;
        request.metrics.sort();
        let mut active = self.active.lock().unwrap();
        if active.len() >= 4 {
            bail!("Four experiments are already running");
        }
        let run = Run {
            id,
            project_id: snapshot.project_id.clone(),
            dataset_id: snapshot.id.clone(),
            dataset_version: snapshot.version,
            dataset_sha256: snapshot.sha256.clone(),
            trace_id: None,
            recovered_ms: None,
            settings: request.settings,
            prompt_template: request.prompt_template,
            prompt_snapshot: request.prompt_snapshot,
            metrics: request.metrics,
            playground:request.playground,
            concurrency: request.concurrency,
            item_timeout_secs: request.item_timeout_secs,
            status: "running".into(),
            strict_quality: false,
            mean_scores: BTreeMap::new(),
            items: snapshot
                .samples
                .iter()
                .map(|s| Item {
                    sample_id: s.id.clone(),
                    status: "pending".into(),
                    output: None,
                    output_truncated: false,
                    scores: BTreeMap::new(),
                    usage: Value::Null,
                    duration_ms: None,
                    error: None,
                })
                .collect(),
        };
        if self.path(&run.id)?.exists() { bail!("Reserved experiment already exists"); }
        self.save_with(&run,||{if run.playground{app.evaluation.commit_single(snapshot.clone())?;}Ok(())})?;
        let (tx, rx) = oneshot::channel();
        active.insert(run.id.clone(), Some(tx));
        let response = run.clone();
        let manager = self.clone();
        let guard = RunGuard {
            manager: manager.clone(),
            run,
        };
        tokio::spawn(async move {
            execute(app, p, snapshot, guard, rx).await;
        });
        Ok(response)
    }
    fn cancel(&self, id: &str) -> Result<Value> {
        let _ = self.load(id)?;
        let requested = self
            .active
            .lock()
            .unwrap()
            .get_mut(id)
            .and_then(Option::take)
            .is_some_and(|tx| tx.send(()).is_ok());
        Ok(json!({"cancel_requested":requested}))
    }
}
struct RunGuard {
    manager: Manager,
    run: Run,
}
impl Drop for RunGuard {
    fn drop(&mut self) {
        if self.run.status == "running" {
            self.run.status = "interrupted".into();
            for item in &mut self.run.items {
                if item.status == "pending" {
                    item.status = "interrupted".into();
                }
            }
        }
        if let Err(e) = self.manager.save(&self.run) {
            eprintln!("Experiment persistence failed: {e}");
        }
        self.manager.active.lock().unwrap().remove(&self.run.id);
    }
}
async fn execute(
    app: crate::App,
    p: provider::Provider,
    snapshot: Snapshot,
    mut guard: RunGuard,
    mut cancel: oneshot::Receiver<()>,
) {
    let trace = match app.observability.begin(
        &format!("experiment-{}", guard.run.id),
        &guard.run.project_id,
    ) {
        Ok(trace) => trace,
        Err(error) => {
            guard.run.status = "failed".into();
            for item in &mut guard.run.items {
                item.status = "error".into();
                item.error = Some("trace_storage_error".into());
            }
            eprintln!("Experiment trace creation failed: {error}");
            return;
        }
    };
    let mut root = trace.span("experiment", "dataset evaluation", None);
    guard.run.trace_id = Some(trace.id());
    if let Err(error) = guard.manager.save(&guard.run) {
        guard.run.status = "failed".into();
        root.finish("failed", &Value::Null);
        eprintln!("Experiment trace receipt persistence failed: {error}");
        return;
    }
    let parent = root.id();
    let settings = guard.run.settings.clone();
    let template = guard.run.prompt_template.clone();
    let system = guard
        .run
        .prompt_snapshot
        .as_ref()
        .map(|s| s.system.clone())
        .unwrap_or_else(crate::prompts::default_system);
    let chat_prompt=guard.run.prompt_snapshot.clone();
    let metrics = guard.run.metrics.clone();
    let timeout = guard.run.item_timeout_secs;
    let mut stream = futures_util::stream::iter(snapshot.samples.into_iter().enumerate())
        .map(|(index, sample)| {
            let p = p.clone();
            let trace = trace.clone();
            let client = app.client.clone();
            let settings = settings.clone();
            let template = template.clone();
            let system = system.clone();
            let metrics = metrics.clone();
            let chat_prompt=chat_prompt.clone();
            async move {
                let mut span = trace.span("evaluation_item", &sample.id, Some(parent));
                let started = Instant::now();
                let result = tokio::time::timeout(Duration::from_secs(timeout), async {
                    let history=if let Some(prompt)=chat_prompt.as_ref().filter(|p|p.messages.is_some()){crate::prompts::render_messages(prompt,&sample.input,&sample.contexts)?.into_iter().map(|m|Message::text(&m.role,m.content)).collect::<Vec<_>>()}else{vec![Message::text("user",render_prompt(&template,&sample))]};
                    if let Some(prepared) = &settings.prepared_guardrails {
                        crate::enforce_guardrail(
                            prepared.input(&history.iter().map(|m|m.content.as_str()).collect::<Vec<_>>().join("\n\n"))?,
                            &trace,
                            span.id(),
                        )?;
                    }
                    let (message, usage) = trace
                        .generation(
                            &settings.model,
                            &settings.provider,
                            span.id(),
                            provider::generate(
                                &client,
                                &p,
                                &settings,
                                &system,
                                &history,
                                &[],
                                |_, _| {},
                            ),
                        )
                        .await?;
                    let output_error = settings.prepared_guardrails.as_ref().and_then(|prepared| {
                        prepared
                            .output(&message.content)
                            .and_then(|receipt| {
                                crate::enforce_guardrail(receipt, &trace, span.id())
                            })
                            .err()
                    });
                    Ok::<_, anyhow::Error>((message, usage, output_error))
                })
                .await;
                let mut item = Item {
                    sample_id: sample.id.clone(),
                    status: "error".into(),
                    output: None,
                    output_truncated: false,
                    scores: BTreeMap::new(),
                    usage: Value::Null,
                    duration_ms: Some(started.elapsed().as_millis().min(u64::MAX as u128) as u64),
                    error: None,
                };
                match result {
                    Ok(Ok((message, usage, output_error))) => {
                        item.usage = crate::observability::public_usage(&usage);
                        if output_error.is_some() {
                            item.error = Some("guardrail_blocked".into());
                            span.finish("failed", &Value::Null);
                            return (index, item);
                        }
                        let (output,storage_truncated)=match bound_output(message.content){Ok(value)=>value,Err(_)=>{item.error=Some("output_storage_error".into());return(index,item);}};
                        let incomplete=message.truncated||!message.tool_calls.is_empty()||storage_truncated;
                        item.output_truncated=message.truncated||storage_truncated;
                        if incomplete {
                            item.error = Some("incomplete_or_oversize_output".into());
                        } else {
                            item.status = "completed".into();
                            item.scores = score(&sample, &output, &metrics);
                        }
                        item.output = Some(output);
                    }
                    Ok(Err(error)) => {
                        item.error = Some(
                            if error.to_string() == "guardrail_blocked" {
                                "guardrail_blocked"
                            } else {
                                "provider_error"
                            }
                            .into(),
                        )
                    }
                    Err(_) => item.error = Some("timeout".into()),
                }
                span.finish(
                    if item.error.as_deref() == Some("timeout") {
                        "timeout"
                    } else if item.status == "completed" {
                        "completed"
                    } else {
                        "failed"
                    },
                    &Value::Null,
                );
                (index, item)
            }
        })
        .buffer_unordered(guard.run.concurrency);
    loop {
        tokio::select! {
            biased;
            _=&mut cancel=>{guard.run.status="cancelled".into();for item in &mut guard.run.items{if item.status=="pending"{item.status="cancelled".into();}}break;},
            next=stream.next()=>{
                let Some((index,item))=next else {guard.run.status=if guard.run.items.iter().all(|i|i.status=="completed"){"completed"}else{"failed"}.into();break;};
                guard.run.items[index]=item;
                if let Err(error)=guard.manager.save(&guard.run){guard.run.status="failed".into();eprintln!("Experiment persistence failed: {error}");break;}
            }
        }
    }
    drop(stream);
    root.finish(&guard.run.status, &Value::Null);
    guard.run.strict_quality = !guard.run.metrics.is_empty() && guard.run.status == "completed"
        && guard.run.items.iter().all(|i| {
            i.status == "completed"
                && i.error.is_none()
                && i.scores.len() == guard.run.metrics.len()
        });
    if guard.run.strict_quality {
        for item in &guard.run.items {
            for (name, score) in &item.scores {
                *guard.run.mean_scores.entry(name.clone()).or_default() +=
                    score / guard.run.items.len() as f64;
            }
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ListQuery {
    project_id:String,offset:Option<usize>,limit:Option<usize>,status:Option<String>,provider:Option<String>,model:Option<String>,dataset_id:Option<String>,playground:Option<bool>,
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<ListQuery>,
) -> crate::ApiResult<Value> {
    crate::evaluation::project_exists(&app, &query.project_id)
        .and_then(|_| app.experiments.list(&query))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
fn prepare_start(app: &crate::App,input:Start)->Result<(Start,Snapshot,provider::Provider)>{
    let snapshot=app.evaluation.load(&input.dataset_id,Some(input.dataset_version))?;
    prepare_with_snapshot(app,input,snapshot)
}
fn prepare_with_snapshot(app: &crate::App, mut input: Start,snapshot:Snapshot) -> Result<(Start, Snapshot, provider::Provider)> {
        crate::validate_settings(&input.settings, app)?;
        input.settings.prepared_guardrails = input
            .settings
            .guardrails
            .as_ref()
            .map(|selection| {
                selection
                    .prepare(&crate::guardrail_policies::Store::open(&app.data)?)
                    .map(Arc::new)
            })
            .transpose()?;
        if let Some(reference) = input.prompt_ref.as_ref() {
            if !input.prompt_template.is_empty() {
                bail!("Choose inline template or prompt_ref, not both");
            }
            let prompt = app.prompts.load(&reference.id, Some(reference.version))?;
            if prompt.project_id != input.settings.project_id {
                bail!("Prompt belongs to another project");
            }
            input.prompt_template = prompt.template.clone();
            input.prompt_snapshot = Some(prompt);
        }

        let p = app
            .providers
            .lock()
            .unwrap()
            .iter()
            .find(|p| p.id == input.settings.provider)
            .cloned()
            .context("Provider not found")?;
        validate(&input, &snapshot)?;
        Ok((input, snapshot, p))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlaygroundInput {
    settings:Settings,prompt_ref:PromptRef,prompt_sha256:String,input:String,
    #[serde(default)]contexts:Vec<String>,
    #[serde(default)]expected_output:Option<String>,
    #[serde(default)]metrics:Vec<Metric>,
    #[serde(default="default_timeout")]item_timeout_secs:u64,
}
pub(crate) async fn playground(State(app):State<crate::App>,Json(input):Json<PlaygroundInput>)->crate::ApiResult<Value>{
    let result=(||->Result<Run>{
        let sample=Sample{id:"input".into(),input:input.input,expected_output:input.expected_output,contexts:input.contexts,metadata:BTreeMap::from([("kind".into(),json!("playground"))])};
        let snapshot=crate::evaluation::Store::draft_single(&input.settings.project_id,"Playground",sample)?;
        let request=Start{dataset_id:snapshot.id.clone(),dataset_version:1,settings:input.settings,prompt_template:String::new(),prompt_ref:Some(input.prompt_ref),prompt_snapshot:None,playground:true,metrics:input.metrics,concurrency:1,item_timeout_secs:input.item_timeout_secs};
        let(request,snapshot,provider)=prepare_with_snapshot(&app,request,snapshot)?;
        let prompt=request.prompt_snapshot.as_ref().context("Playground requires a saved prompt")?;
        if prompt.sha256!=input.prompt_sha256 {bail!("Playground prompt hash differs from reviewed version");}
        crate::prompts::preview_receipt(prompt,&crate::prompts::PreviewInput{project_id:snapshot.project_id.clone(),input:snapshot.samples[0].input.clone(),contexts:snapshot.samples[0].contexts.clone()})?;
        app.experiments.start(app.clone(),request,snapshot,provider)
    })();
    result.and_then(|run|Ok(serde_json::to_value(run)?)).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn start(
    State(app): State<crate::App>,
    Json(input): Json<Start>,
) -> crate::ApiResult<Value> {
    let result = prepare_start(&app, input).and_then(|(input, snapshot, p)| {
        app.experiments.start(app.clone(), input, snapshot, p)
    });
    result.and_then(|r| Ok(serde_json::to_value(r)?)).map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn detail(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    app.experiments
        .load(&id)
        .and_then(|r| Ok(serde_json::to_value(r)?))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportQuery {
    #[serde(default)]
    include_outputs: bool,
}
fn export_receipt(run: &Run, include_outputs: bool) -> Value {
    let items=run.items.iter().map(|item| {
        let mut row=json!({"sample_id":item.sample_id,"status":item.status,
            "scores":item.scores,"duration_ms":item.duration_ms,
            "has_error":item.error.is_some(),"output_truncated":item.output_truncated});
        if include_outputs {row["output"]=json!(item.output);}
        row
    }).collect::<Vec<_>>();
    json!({"kind":"experiment_export","schema_version":1,"run_id":run.id,
        "project_id":run.project_id,"dataset_id":run.dataset_id,
        "dataset_version":run.dataset_version,"dataset_sha256":run.dataset_sha256,
        "status":run.status,"strict_quality":run.strict_quality,
        "provider":run.settings.provider,"model":run.settings.model,"trace_id":run.trace_id,
        "prompt_ref":run.prompt_snapshot.as_ref().map(|prompt|json!({"id":prompt.id,"version":prompt.version,"sha256":prompt.sha256})),
        "concurrency":run.concurrency,"item_timeout_secs":run.item_timeout_secs,
        "metrics":run.metrics,"mean_scores":run.mean_scores,"items":items,
        "outputs_included":include_outputs,"automatic_promotion":false,"playground":run.playground})
}
pub(crate) async fn export(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Query(query): Query<ExportQuery>,
) -> crate::ApiResult<Value> {
    app.experiments.load(&id).map(|run| Json(export_receipt(&run,query.include_outputs)))
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn cancel(
    State(app): State<crate::App>,
    Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    app.experiments
        .cancel(&id)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Compare {
    baseline_id: String,
    candidate_id: String,
}
fn verify_run(run: &Run, snapshot: &Snapshot) -> Result<()> {
    validate_metrics(&run.metrics, snapshot)?;
    if let Some(prompt) = run.prompt_snapshot.as_ref() {
        crate::prompts::verify(prompt)?;
        if prompt.project_id != run.project_id || prompt.template != run.prompt_template {
            bail!("Experiment prompt receipt mismatch");
        }
    }
    if run.status != "completed"
        || !run.strict_quality
        || run.items.len() != snapshot.samples.len()
        || run.dataset_sha256 != snapshot.sha256
        || run.project_id != snapshot.project_id
        || run.settings.project_id != snapshot.project_id
    {
        bail!("Comparison requires complete, strict-quality runs on an intact dataset snapshot");
    }
    let mut ids = std::collections::HashSet::new();
    let mut means = BTreeMap::<String, f64>::new();
    for item in &run.items {
        let sample = snapshot
            .samples
            .iter()
            .find(|s| s.id == item.sample_id)
            .context("Run contains an unknown sample")?;
        if !ids.insert(&item.sample_id)
            || item.status != "completed"
            || item.error.is_some()
            || item.output_truncated
        {
            bail!("Run contains duplicate, incomplete or failed rows");
        }
        let output = item.output.as_deref().context("Run output missing")?;
        if item.scores != score(sample, output, &run.metrics)
            || item.scores.len() != run.metrics.len()
        {
            bail!("Run metric receipt does not match its output");
        }
        for (name, value) in &item.scores {
            *means.entry(name.clone()).or_default() += value / run.items.len() as f64;
        }
    }
    if run.mean_scores != means {
        bail!("Run mean metric receipt does not match its rows");
    }
    Ok(())
}
fn paired_comparison(baseline: &Run, candidate: &Run, snapshot: &Snapshot) -> Result<Value> {
    if baseline.project_id != candidate.project_id
        || baseline.dataset_id != candidate.dataset_id
        || baseline.dataset_version != candidate.dataset_version
        || baseline.dataset_sha256 != candidate.dataset_sha256
        || baseline.metrics != candidate.metrics
    {
        bail!(
            "Comparison requires identical projects, dataset snapshots and metric configurations"
        );
    }
    verify_run(baseline, snapshot)?;
    verify_run(candidate, snapshot)?;
    let mut pairs = Vec::new();
    let mut regressions = 0;
    let mut improvements = 0;
    for base in &baseline.items {
        let next = candidate
            .items
            .iter()
            .find(|i| i.sample_id == base.sample_id)
            .context("Missing paired sample")?;
        for (metric, score) in &base.scores {
            let value = next.scores.get(metric).context("Missing paired metric")?;
            let delta = value - score;
            regressions += usize::from(delta < 0.0);
            improvements += usize::from(delta > 0.0);
            pairs.push(json!({"sample_id":base.sample_id,"metric":metric,"baseline":score,"candidate":value,"delta":delta}));
        }
    }
    Ok(
        json!({"id":crate::evaluation::new_id(),"baseline_id":baseline.id,"candidate_id":candidate.id,"project_id":baseline.project_id,"dataset_id":baseline.dataset_id,"dataset_version":baseline.dataset_version,"dataset_sha256":baseline.dataset_sha256,"eligible":regressions==0&&improvements>0,"regressions":regressions,"improvements":improvements,"pairs":pairs,"reason":if regressions>0{"paired_regression"}else if improvements==0{"no_improvement"}else{"strict_paired_improvement"},"observational_only":true}),
    )
}
pub(crate) async fn compare(
    State(app): State<crate::App>,
    Json(input): Json<Compare>,
) -> crate::ApiResult<Value> {
    let result = (|| -> Result<Value> {
        let baseline = app.experiments.load(&input.baseline_id)?;
        let candidate = app.experiments.load(&input.candidate_id)?;
        let snapshot = app
            .evaluation
            .load(&baseline.dataset_id, Some(baseline.dataset_version))?;
        let receipt = paired_comparison(&baseline, &candidate, &snapshot)?;
        let root = app.data.join("evaluation/comparisons");
        std::fs::create_dir_all(&root)?;
        if !root.canonicalize()?.starts_with(app.data.as_ref()) {
            bail!("Comparison storage escapes Studio data directory");
        }
        let path = root.join(format!("{}.json", receipt["id"].as_str().unwrap()));
        crate::evaluation::commit_new(&path, &receipt)?;
        Ok(receipt)
    })();
    result
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComparisonCatalog {
    project_id:String,
    #[serde(default)] offset:usize,
    #[serde(default="comparison_page_size")] limit:usize,
}
fn comparison_page_size()->usize{30}
pub(crate) async fn list_comparisons(
    State(app):State<crate::App>, Query(query):Query<ComparisonCatalog>,
)->crate::ApiResult<Value>{
    let result=(||->Result<Value>{
        crate::evaluation::project_exists(&app,&query.project_id)?;
        if query.limit==0||query.limit>100||query.offset>2000{bail!("Invalid comparison page");}
        let root=app.data.join("evaluation/comparisons");
        if !root.exists(){return Ok(json!({"comparisons":[],"offset":query.offset,"has_more":false,"summaries_verified":false}));}
        if !std::fs::symlink_metadata(&root)?.is_dir()||!root.canonicalize()?.starts_with(app.data.as_ref()){bail!("Invalid comparison directory");}
        let mut rows=Vec::new();let mut files=0;let mut total=0;
        for entry in std::fs::read_dir(root)?{
            let entry=entry?;let path=entry.path();if path.extension().is_none_or(|ext|ext!="json"){continue;}
            files+=1;let size=entry.metadata()?.len();total+=size;
            if files>2000||total>16*1024*1024||size>4*1024*1024||!entry.file_type()?.is_file(){bail!("Comparison catalog bounds exceeded");}
            let mut bytes=Vec::new();std::fs::File::open(&path)?.take(4*1024*1024+1).read_to_end(&mut bytes)?;
            if bytes.len()>4*1024*1024{bail!("Oversize comparison receipt");}
            let receipt=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
            let id=receipt["id"].as_str().context("Missing comparison ID")?;
            if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-')||path.file_stem().and_then(|v|v.to_str())!=Some(id){bail!("Comparison catalog identity mismatch");}
            if receipt["project_id"]!=query.project_id{continue;}
            rows.push(json!({"id":id,"baseline_id":receipt["baseline_id"],"candidate_id":receipt["candidate_id"],"dataset_id":receipt["dataset_id"],"dataset_version":receipt["dataset_version"],"regressions":receipt["regressions"],"improvements":receipt["improvements"],"reason":receipt["reason"]}));
        }
        rows.sort_by(|a,b|b["id"].as_str().cmp(&a["id"].as_str()));
        let has_more=rows.len()>query.offset+query.limit;
        Ok(json!({"comparisons":rows.into_iter().skip(query.offset).take(query.limit).collect::<Vec<_>>(),"offset":query.offset,"has_more":has_more,"summaries_verified":false}))
    })();result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn read_comparison(
    State(app): State<crate::App>, Path(id): Path<String>,
) -> crate::ApiResult<Value> {
    let result=(|| -> Result<Value> {
        if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'){bail!("Invalid comparison ID");}
        let path=app.data.join("evaluation/comparisons").join(format!("{id}.json"));
        let metadata=std::fs::symlink_metadata(&path)?;
        if !metadata.is_file()||metadata.len()>4*1024*1024 {bail!("Invalid comparison receipt");}
        let path=path.canonicalize()?;if !path.starts_with(app.data.as_ref()){bail!("Comparison escapes Studio data");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(4*1024*1024+1).read_to_end(&mut bytes)?;
        if bytes.len()>4*1024*1024 {bail!("Oversize comparison receipt");}
        let receipt=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        if receipt["id"]!=id {bail!("Comparison identity mismatch");}
        let baseline=app.experiments.load(receipt["baseline_id"].as_str().context("Missing baseline")?)?;
        let candidate=app.experiments.load(receipt["candidate_id"].as_str().context("Missing candidate")?)?;
        let snapshot=app.evaluation.load(&baseline.dataset_id,Some(baseline.dataset_version))?;
        let mut expected=paired_comparison(&baseline,&candidate,&snapshot)?;expected["id"]=json!(id);
        if expected!=receipt {bail!("Comparison receipt does not match pinned results");}
        Ok(receipt)
    })();
    result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}

#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixVariant {label:String,run_id:String}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixSave {project_id:String,baseline_id:String,variants:Vec<MatrixVariant>}
fn matrix_receipt(app:&crate::App,id:&str,input:&MatrixSave)->Result<Value>{
    crate::evaluation::project_exists(app,&input.project_id)?;
    if input.variants.len()<2||input.variants.len()>16||input.variants[0].run_id!=input.baseline_id{bail!("Select 2-16 variants, with the baseline first");}
    let baseline=app.experiments.load(&input.baseline_id)?;
    if baseline.project_id!=input.project_id{bail!("Matrix project mismatch");}
    let snapshot=app.evaluation.load(&baseline.dataset_id,Some(baseline.dataset_version))?;
    let mut ids=std::collections::HashSet::new();let mut labels=std::collections::HashSet::new();let mut variants=Vec::new();let mut passed=true;
    for variant in &input.variants {
        if variant.label.trim().is_empty()||variant.label.len()>200||!labels.insert(&variant.label)||!ids.insert(&variant.run_id){bail!("Select unique runs and bounded unique labels");}
        let run=app.experiments.load(&variant.run_id)?;
        let mut comparison=paired_comparison(&baseline,&run,&snapshot)?;comparison.as_object_mut().unwrap().remove("id");
        passed&=comparison["regressions"]==0;
        variants.push(json!({"label":variant.label,"run_id":run.id,"provider":run.settings.provider,"model":run.settings.model,"mean_scores":run.mean_scores,"comparison":comparison}));
    }
    Ok(json!({"kind":"experiment_matrix","schema_version":1,"id":id,"project_id":input.project_id,"baseline_id":input.baseline_id,"dataset_id":baseline.dataset_id,"dataset_version":baseline.dataset_version,"dataset_sha256":baseline.dataset_sha256,"metrics":baseline.metrics,"request":input,"variants":variants,"passed":passed,"gate_policy":"no_paired_regression","automatic_promotion":false}))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixCatalog {
    project_id:String,
    #[serde(default)] offset:usize,
    #[serde(default="comparison_page_size")] limit:usize,
}
pub(crate) async fn list_matrices(State(app):State<crate::App>,Query(query):Query<MatrixCatalog>)->crate::ApiResult<Value>{
    let result=(||->Result<Value>{
        crate::evaluation::project_exists(&app,&query.project_id)?;
        if query.limit==0||query.limit>100||query.offset>2000{bail!("Invalid matrix page");}
        let root=app.data.join("evaluation/matrices");
        if !root.exists(){return Ok(json!({"matrices":[],"offset":query.offset,"has_more":false,"summaries_verified":false}));}
        if !std::fs::symlink_metadata(&root)?.is_dir()||!root.canonicalize()?.starts_with(app.data.as_ref()){bail!("Invalid matrix directory");}
        let mut rows=Vec::new();let mut files=0;let mut total=0;
        for entry in std::fs::read_dir(root)?{
            let entry=entry?;let path=entry.path();if path.extension().is_none_or(|ext|ext!="json"){continue;}
            files+=1;let size=entry.metadata()?.len();total+=size;
            if files>2000||total>16*1024*1024||size>4*1024*1024||!entry.file_type()?.is_file(){bail!("Matrix catalog bounds exceeded");}
            let mut bytes=Vec::new();std::fs::File::open(&path)?.take(4*1024*1024+1).read_to_end(&mut bytes)?;
            if bytes.len()>4*1024*1024{bail!("Oversize matrix receipt");}
            let receipt=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
            let id=receipt["id"].as_str().context("Missing matrix ID")?;
            if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-')||path.file_stem().and_then(|v|v.to_str())!=Some(id)||receipt["kind"]!="experiment_matrix"||receipt["schema_version"]!=1{bail!("Matrix catalog identity mismatch");}
            if receipt["project_id"]!=query.project_id{continue;}
            let count=receipt["variants"].as_array().context("Missing matrix variants")?.len();
            if !(2..=16).contains(&count){bail!("Invalid matrix variant count");}
            rows.push(json!({"id":id,"dataset_id":receipt["dataset_id"],"dataset_version":receipt["dataset_version"],"baseline_id":receipt["baseline_id"],"variant_count":count}));
        }
        rows.sort_by(|a,b|b["id"].as_str().cmp(&a["id"].as_str()));let has_more=rows.len()>query.offset+query.limit;
        Ok(json!({"matrices":rows.into_iter().skip(query.offset).take(query.limit).collect::<Vec<_>>(),"offset":query.offset,"has_more":has_more,"summaries_verified":false}))
    })();result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn save_matrix(State(app):State<crate::App>,Json(input):Json<MatrixSave>)->crate::ApiResult<Value>{
    let result=(||->Result<Value>{
        let id=crate::evaluation::new_id();let receipt=matrix_receipt(&app,&id,&input)?;
        let root=app.data.join("evaluation/matrices");std::fs::create_dir_all(&root)?;
        if !std::fs::symlink_metadata(&root)?.is_dir()||!root.canonicalize()?.starts_with(app.data.as_ref()){bail!("Invalid matrix directory");}
        if serde_json::to_vec(&receipt)?.len()>4*1024*1024{bail!("Oversize matrix artifact");}
        crate::evaluation::commit_new(&root.join(format!("{id}.json")),&receipt)?;Ok(receipt)
    })();result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn read_matrix(State(app):State<crate::App>,Path(id):Path<String>)->crate::ApiResult<Value>{
    let result=(||->Result<Value>{
        if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'){bail!("Invalid matrix ID");}
        let path=app.data.join("evaluation/matrices").join(format!("{id}.json"));let metadata=std::fs::symlink_metadata(&path)?;
        if !metadata.is_file()||metadata.len()>4*1024*1024{bail!("Invalid matrix artifact");}
        let path=path.canonicalize()?;if !path.starts_with(app.data.as_ref()){bail!("Matrix escapes Studio data");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(4*1024*1024+1).read_to_end(&mut bytes)?;
        if bytes.len()>4*1024*1024{bail!("Oversize matrix artifact");}
        let receipt=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        let input:MatrixSave=serde_json::from_value(receipt["request"].clone())?;
        if matrix_receipt(&app,&id,&input)?!=receipt{bail!("Matrix does not match pinned results");}
        Ok(receipt)
    })();result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capacity_fixture()->(PathBuf,Manager,Run){
        let root=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let manager=Manager::new(&root).unwrap();
        let run:Run=serde_json::from_value(json!({"id":crate::evaluation::new_id(),"project_id":"default","dataset_id":"cases","dataset_version":1,"dataset_sha256":"a".repeat(64),"settings":{"project_id":"default","provider":"local","model":"mock","mode":"chat","allow_writes":false},"prompt_template":"{{input}}","metrics":["json_valid"],"concurrency":1,"item_timeout_secs":120,"status":"running","items":[{"sample_id":"input","status":"pending","output":null,"scores":{},"usage":null,"duration_ms":null,"error":null}],"strict_quality":false,"mean_scores":{}})).unwrap();(root,manager,run)
    }
    #[test]
    fn escaped_output_bounds_preserve_utf8(){
        let plain="x".repeat(65536);assert_eq!(bound_output(plain.clone()).unwrap(),(plain,false));
        let(output,truncated)=bound_output("\u{0000}".repeat(65536)).unwrap();assert!(truncated);assert!(serde_json::to_vec(&output).unwrap().len()<=MAX_OUTPUT_JSON_BYTES);
        let(output,truncated)=bound_output("я".repeat(40000)).unwrap();assert!(truncated);assert_eq!(output.len(),65536);assert!(output.chars().all(|c|c=='я'));
        let(root,_manager,mut run)=capacity_fixture();run.items=vec![run.items[0].clone();200];assert!(run_reservation(&run).unwrap()<MAX_RUN_BYTES);std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn full_run_catalog_rejects_new_write_callbacks_and_keeps_updates(){
        let(root,manager,mut run)=capacity_fixture();run.status="completed".into();for i in 0..MAX_RUNS{let mut existing=run.clone();existing.id=format!("{i:x}");std::fs::write(manager.path(&existing.id).unwrap(),serde_json::to_vec(&existing).unwrap()).unwrap();}
        let called=std::sync::atomic::AtomicBool::new(false);assert!(manager.save_with(&run,||{called.store(true,std::sync::atomic::Ordering::Relaxed);Ok(())}).unwrap_err().to_string().contains("retention limit"));assert!(!called.load(std::sync::atomic::Ordering::Relaxed));assert!(!manager.path(&run.id).unwrap().exists());run.id="0".into();run.status="interrupted".into();manager.save(&run).unwrap();assert_eq!(manager.load("0").unwrap().status,"interrupted");std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn output_reservations_serialize_competing_admissions(){
        let(root,manager,mut run)=capacity_fixture();run.items=vec![run.items[0].clone();200];let capacity=MAX_CATALOG_BYTES as usize/run_reservation(&run).unwrap();for i in 0..capacity-1 {let mut existing=run.clone();existing.id=format!("{i:x}");std::fs::write(manager.path(&existing.id).unwrap(),serde_json::to_vec(&existing).unwrap()).unwrap();}
        let barrier=Arc::new(std::sync::Barrier::new(2));let threads:Vec<_>=(0..2).map(|_|{let mut run=run.clone();run.id=crate::evaluation::new_id();let manager=manager.clone();let barrier=barrier.clone();std::thread::spawn(move||{barrier.wait();manager.save(&run).is_ok()})}).collect();assert_eq!(threads.into_iter().filter_map(|t|t.join().ok()).filter(|v|*v).count(),1);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn catalog_filters_before_pagination_and_keeps_legacy_full_list(){
        let root=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let manager=Manager::new(&root).unwrap();
        for i in 0..26 {let run:Run=serde_json::from_value(json!({"id":format!("{i:04x}"),"project_id":if i==25 {"foreign"}else{"default"},"dataset_id":if i<20 {"cases"}else{"other"},"dataset_version":1,"dataset_sha256":"a".repeat(64),"settings":{"project_id":"default","provider":"local","model":if i%2==0 {"MODEL-A"}else{"Model-B"},"mode":"chat","allow_writes":false},"prompt_template":"{{input}}","metrics":["json_valid"],"playground":i%2==0,"concurrency":1,"item_timeout_secs":120,"status":if i%3==0 {"failed"}else{"completed"},"items":[],"strict_quality":false,"mean_scores":{}})).unwrap();manager.save(&run).unwrap();}
        let mut query:ListQuery=serde_json::from_value(json!({"project_id":"default","offset":0,"limit":20})).unwrap();let first=manager.list(&query).unwrap();assert_eq!(first["total"],25);assert_eq!(first["runs"].as_array().unwrap().len(),20);assert_eq!(first["has_more"],true);query.offset=Some(20);assert_eq!(manager.list(&query).unwrap()["runs"].as_array().unwrap().len(),5);
        query.offset=Some(0);query.model=Some("model-a".into());assert_eq!(manager.list(&query).unwrap()["total"],13);query.playground=Some(false);assert_eq!(manager.list(&query).unwrap()["total"],0);query.playground=Some(true);query.dataset_id=Some("cases".into());query.status=Some("failed".into());assert_eq!(manager.list(&query).unwrap()["total"],4);query.limit=Some(101);assert!(manager.list(&query).is_err());
        let legacy:ListQuery=serde_json::from_value(json!({"project_id":"default"})).unwrap();assert_eq!(manager.list(&legacy).unwrap()["runs"].as_array().unwrap().len(),25);std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn only_single_input_playground_can_skip_metrics(){
        let mut request:Start=serde_json::from_value(json!({"dataset_id":"dataset","dataset_version":1,"settings":{"project_id":"default","provider":"local","model":"mock","mode":"chat","allow_writes":false},"prompt_template":"{{input}}","metrics":[]})).unwrap();
        let mut snapshot=crate::evaluation::Store::draft_single("default","One",Sample{id:"input".into(),input:"Question".into(),expected_output:None,contexts:vec![],metadata:BTreeMap::new()}).unwrap();
        assert!(validate(&request,&snapshot).is_err());request.playground=true;assert!(validate(&request,&snapshot).is_ok());assert!(serde_json::to_value(&request).unwrap().get("playground").is_none());snapshot.samples.push(snapshot.samples[0].clone());assert!(validate(&request,&snapshot).is_err());
    }
    #[test]
    fn character_bigrams_preserve_adjacency_unicode_and_multiplicity() {
        assert_eq!(character_bigram_f1("abc", "abc"), 1.0);
        assert_eq!(character_bigram_f1("abc", "cba"), 0.0);
        assert_eq!(character_bigram_f1("aaaa", "aa"), 0.5);
        assert_eq!(character_bigram_f1("Привет", "Привёт"), 0.6);
        assert_eq!(character_bigram_f1("", ""), 1.0);
        assert_eq!(character_bigram_f1("а", "а"), 1.0);
        assert_eq!(character_bigram_f1("a", "aa"), 0.0);
        assert_eq!(character_bigram_f1("a b", "ab"), 0.0);
    }
    #[test]
    fn json_equality_ignores_object_order_but_preserves_array_order_and_types() {
        assert!(json_equals(
            r#"{"a":1,"b":[true,null]}"#,
            r#"{ "b": [true, null], "a": 1 }"#
        ));
        assert!(!json_equals(r#"{"a":2}"#, r#"{"a":1,"a":2}"#));
        assert!(!json_equals("[1,2]", "[2,1]"));
        assert!(!json_equals("1", "\"1\""));
        assert!(!json_equals("null", "{}"));
        assert!(!json_equals("invalid", "invalid"));
        assert!(!json_equals("{}", "{} trailing"));
    }
    #[test]
    fn token_f1_counts_duplicate_tokens_and_preserves_case_and_punctuation() {
        assert_eq!(whitespace_token_f1("a a b", "a b b"), 2.0 / 3.0);
        assert_eq!(whitespace_token_f1("Привет мир", "мир\nПривет"), 1.0);
        assert_eq!(whitespace_token_f1("A", "a"), 0.0);
        assert_eq!(whitespace_token_f1("word.", "word"), 0.0);
        assert_eq!(whitespace_token_f1("a", ""), 0.0);
        assert_eq!(whitespace_token_f1("a b", "a"), 2.0 / 3.0);
    }
    #[test]
    fn prompt_placeholders_inside_sample_data_are_not_expanded() {
        let sample = Sample {
            id: "a".into(),
            input: "user {{contexts}}".into(),
            expected_output: None,
            contexts: vec!["data {{input}}".into()],
            metadata: BTreeMap::new(),
        };
        assert_eq!(
            render_prompt("{{contexts}} / {{input}}", &sample),
            "data {{input}} / user {{contexts}}"
        );
    }
    #[test]
    fn metrics_distinguish_wrong_answers_and_invalid_json() {
        let sample = Sample {
            id: "a".into(),
            input: "q".into(),
            expected_output: Some("answer".into()),
            contexts: vec![],
            metadata: BTreeMap::new(),
        };
        let metrics = [
            Metric::ExactMatch,
            Metric::ContainsReference,
            Metric::JsonValid,
        ];
        let scores = score(&sample, "answer extra", &metrics);
        assert_eq!(scores["exact_match"], 0.0);
        assert_eq!(scores["contains_reference"], 1.0);
        assert_eq!(scores["json_valid"], 0.0);
        assert_eq!(
            score(&sample, "{\"value\":true}", &metrics)["json_valid"],
            1.0
        );
    }
}
