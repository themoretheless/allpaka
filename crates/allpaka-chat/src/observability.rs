//! Native, metadata-only trace storage. No prompts, tool inputs or outputs are stored.
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest,Sha256};
use std::{
    io::Read,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    lifecycle: Arc<Mutex<()>>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    session_id: String,
    project_id: String,
    started_ms: u64,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recovered_ms: Option<u64>,
    spans: Vec<Span>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Span {
    #[serde(default, skip_serializing_if="Option::is_none")]
    provider_id: Option<String>,
    id: usize,
    parent_id: Option<usize>,
    kind: String,
    name: String,
    status: String,
    started_ms: u64,
    duration_ms: Option<u64>,
    usage: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    linked_trace_id: Option<String>,
}
#[derive(Default)]
struct ModelTotals {
    calls:u64,input:u64,output:u64,input_unknown:u64,output_unknown:u64,cache_write:u64,cache_read:u64,cache_write_unknown:u64,cache_read_unknown:u64,
    cost_unknown:u64,currency_unknown:u64,duration_unknown:u64,first_text_unknown:u64,first_text:Vec<u64>,
    statuses:std::collections::BTreeMap<String,u64>,
    costs:std::collections::BTreeMap<String,f64>,durations:Vec<u64>,
}
impl ModelTotals {
    fn observe(&mut self,span:&Span)->Result<()> {
        self.calls+=1;*self.statuses.entry(span.status.clone()).or_default()+=1;
        match span.usage["prompt_tokens"].as_u64().or_else(||span.usage["input_tokens"].as_u64()) {
            Some(n)=>self.input=self.input.checked_add(n).context("Model input total overflow")?,None=>self.input_unknown+=1,
        }
        match span.usage["completion_tokens"].as_u64().or_else(||span.usage["output_tokens"].as_u64()) {
            Some(n)=>self.output=self.output.checked_add(n).context("Model output total overflow")?,None=>self.output_unknown+=1,
        }
        match span.usage["cache_creation_input_tokens"].as_u64(){Some(n)=>self.cache_write=self.cache_write.checked_add(n).context("Cache creation total overflow")?,None=>self.cache_write_unknown+=1}
        match span.usage["cache_read_input_tokens"].as_u64().or_else(||span.usage["cached_tokens"].as_u64()){Some(n)=>self.cache_read=self.cache_read.checked_add(n).context("Cache read total overflow")?,None=>self.cache_read_unknown+=1}
        match span.duration_ms {Some(n)=>self.durations.push(n),None=>self.duration_unknown+=1}
        match span.usage["first_text_ms"].as_u64().filter(|first|span.duration_ms.is_none_or(|total|*first<=total)){Some(first)=>self.first_text.push(first),None=>self.first_text_unknown+=1}
        if let Some(cost)=span.usage["cost"].as_f64().filter(|n|n.is_finite()&&*n>=0.0) {
            if let Some(currency)=span.usage["cost_currency"].as_str().filter(|s|s.len()==3&&s.bytes().all(|b|b.is_ascii_uppercase())) {
                let total=self.costs.entry(currency.into()).or_default();*total+=cost;if !total.is_finite(){bail!("Model cost overflow");}
            }else{self.currency_unknown+=1;}
        }else{self.cost_unknown+=1;}
        Ok(())
    }
    fn receipt(mut self,source:String,provider_id:Option<String>,model:String)->Value {
        self.durations.sort_unstable();
        let percentile=|percent:usize|if self.durations.is_empty(){None}else{Some(self.durations[(self.durations.len()*percent).div_ceil(100)-1])};
        json!({"first_text":first_text_receipt(self.first_text,self.first_text_unknown),"source":source,"provider_id":provider_id,"model":model,"calls":self.calls,"statuses":self.statuses,
            "input_tokens":self.input,"output_tokens":self.output,"input_tokens_unknown_calls":self.input_unknown,"output_tokens_unknown_calls":self.output_unknown,
            "cache_creation_input_tokens":self.cache_write,"cache_read_input_tokens":self.cache_read,"cache_creation_unknown_calls":self.cache_write_unknown,"cache_read_unknown_calls":self.cache_read_unknown,
            "reported_cost_by_currency":self.costs,"cost_unknown_calls":self.cost_unknown,"cost_currency_unknown_calls":self.currency_unknown,
            "duration_known_calls":self.durations.len(),"duration_unknown_calls":self.duration_unknown,
            "duration_min_ms":self.durations.first(),"duration_max_ms":self.durations.last(),"duration_p50_ms":percentile(50),"duration_p95_ms":percentile(95),
            "duration_quantile_method":"nearest_rank","duration_scope":"whole_model_span"})
    }
}
fn first_text_receipt(mut values:Vec<u64>,unknown:u64)->Value {
    values.sort_unstable();
    let percentile=|percent:usize|if values.is_empty(){None}else{Some(values[(values.len()*percent).div_ceil(100)-1])};
    json!({"known_calls":values.len(),"unknown_calls":unknown,"min_ms":values.first(),"max_ms":values.last(),"p50_ms":percentile(50),"p95_ms":percentile(95),"quantile_method":"nearest_rank","scope":"client_observed_first_text","server_token_time":false})
}
#[derive(Clone)]
pub(crate) struct Trace {
    record: Arc<Mutex<Record>>,
    path: Arc<PathBuf>,
    writer: Arc<Mutex<()>>,
}
pub(crate) struct Guard {
    trace: Trace,
    index: usize,
    started: Instant,
    finished: bool,
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
fn atomic_write(path: &std::path::Path, record: &Record) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec(record)?;
    let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(temp);
    result
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalTrace {
    project_id: String,
    #[serde(default)]
    idempotency_key: Option<String>,
    correlation_id: String,
    started_ms: u64,
    spans: Vec<ExternalSpan>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalSpan {
    #[serde(default)]
    provider_id: Option<String>,
    parent_id: Option<usize>,
    kind: String,
    name: String,
    status: String,
    started_ms: u64,
    duration_ms: Option<u64>,
    #[serde(default)]
    usage: ExternalUsage,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalUsage {
    guardrail_receipt: Option<Value>,
    evaluation_scores: Option<Value>,
    evaluator_ref: Option<Value>,
    evaluation_sampling: Option<Value>,
    first_text_ms: Option<u64>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cost: Option<f64>,
    cost_currency: Option<String>,
}
impl Store {
    fn is_removed(&self,id:&str)->Result<bool>{
        let directory=self.root.join("removed");
        match std::fs::symlink_metadata(&directory){Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(false),Err(error)=>return Err(error.into()),Ok(_)=>{}}
        if !std::fs::symlink_metadata(&directory)?.is_dir()||!directory.canonicalize()?.starts_with(self.root.as_ref()) {bail!("Invalid removed trace directory");}
        let path=directory.join(format!("{id}.json"));
        match std::fs::symlink_metadata(&path){Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(false),Err(error)=>return Err(error.into()),Ok(_)=>{}}
        let metadata=std::fs::symlink_metadata(&path)?;
        if !metadata.is_file()||metadata.len()>1024 {bail!("Invalid trace removal marker");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(1025).read_to_end(&mut bytes)?;
        if bytes.len()>1024 {bail!("Oversize trace removal marker");}
        let marker:Value=serde_json::from_slice(&bytes)?;
        if marker["trace_id"]!=id||marker["removed_ms"].as_u64().is_none() {bail!("Invalid trace removal identity");}
        Ok(true)
    }
    fn change_visibility(&self,id:&str,removed:bool)->Result<Value>{
        let _lock=self.lifecycle.lock().unwrap();
        let record=self.read_record(id)?;
        if record.status=="running"||record.spans.iter().any(|span|span.status=="running") {bail!("Active traces cannot be removed or restored");}
        let previous=self.is_removed(id)?;
        if removed&&!previous {
            let directory=self.root.join("removed");std::fs::create_dir_all(&directory)?;
            if !std::fs::symlink_metadata(&directory)?.is_dir()||!directory.canonicalize()?.starts_with(self.root.as_ref()) {bail!("Invalid removed trace directory");}
            crate::evaluation::commit_new(&directory.join(format!("{id}.json")),&json!({"trace_id":id,"removed_ms":now()}))?;
        }else if !removed&&previous {std::fs::remove_file(self.root.join("removed").join(format!("{id}.json")))?;}
        Ok(json!({"trace_id":id,"removed":removed,"changed":previous!=removed,"content_preserved":true}))
    }
    fn ingest(&self, request: ExternalTrace) -> Result<Value> {
        let identifier=|s:&str| !s.is_empty()&&s.len()<=100&&s.bytes().all(|b|b.is_ascii_alphanumeric()||b"._-:/".contains(&b));
        if !identifier(&request.correlation_id)||request.spans.is_empty()||request.spans.len()>200 {
            bail!("External traces require a bounded correlation ID and 1–200 spans");
        }
        if request.idempotency_key.as_ref().is_some_and(|key|!identifier(key)) {
            bail!("Invalid external trace idempotency key");
        }
        let mut spans=Vec::new();
        for (index, span) in request.spans.into_iter().enumerate() {
            if (index==0&&span.parent_id.is_some())||(index>0&&!span.parent_id.is_some_and(|parent|parent<index)) {
                bail!("Provide one root and parent-before-child spans");
            }
            if !identifier(&span.name)||!matches!(span.kind.as_str(),"agent"|"model"|"tool")
                ||span.provider_id.as_ref().is_some_and(|provider|!identifier(provider))
                ||!matches!(span.status.as_str(),"completed"|"failed"|"interrupted") {
                bail!("Invalid external span metadata");
            }
            if span.started_ms<request.started_ms||span.duration_ms.is_some_and(|d|span.started_ms.checked_add(d).is_none()) {
                bail!("Invalid external span timing");
            }
            if let Some(parent)=span.parent_id {
                let parent:&Span=&spans[parent];
                if span.started_ms<parent.started_ms {bail!("Child starts before its parent");}
                if parent.duration_ms.is_some_and(|d|span.started_ms>parent.started_ms+d) {bail!("Child starts after its parent ends");}
                if let (Some(pd),Some(cd))=(parent.duration_ms,span.duration_ms) {
                    if span.started_ms+cd>parent.started_ms+pd {bail!("Child ends after its parent");}
                }
            }
            let usage=span.usage;
            if let Some(receipt)=&usage.guardrail_receipt {
                let clean=public_guardrail_receipt(receipt).context("Invalid external guardrail receipt")?;
                let name=format!("guardrail.{}.{}.{}.{}",clean["stage"].as_str().unwrap(),clean["action"].as_str().unwrap(),if clean["passed"]==true {"pass"}else{"fail"},clean["policy_sha256"].as_str().unwrap());
                if clean!=*receipt||span.kind!="tool"||span.name!=name||span.status!=if clean["blocked"]==true {"failed"}else{"completed"} {bail!("External guardrail receipt metadata mismatch");}
            }

            if let Some(sampling)=&usage.evaluation_sampling {
                let clean=valid_evaluation_sampling(sampling).context("Invalid evaluation sampling receipt")?;
                let rate=clean["sample_rate"].as_f64().unwrap();
                let key=serde_json::to_vec(&json!(["allpaka-evaluation-sampling-v1",request.project_id,request.correlation_id,span.name,index]))?;
                let hash=Sha256::digest(key);let choice=u64::from_be_bytes(hash[..8].try_into().unwrap()) as u128;
                let expected=rate==1.0 || rate>0.0 && choice < (rate*18446744073709551616.0).ceil() as u128;
                if clean["selected"]!=json!(expected)||span.status!="completed" {bail!("Evaluation sampling decision mismatch");}
            }
            if let Some(reference)=&usage.evaluator_ref {if valid_evaluator_ref(reference).is_none()||span.kind!="tool" {bail!("Invalid evaluator reference");}}
            if let Some(scores)=&usage.evaluation_scores { if valid_evaluation_scores(scores).is_none() || span.kind!="tool" || span.status!="completed" {bail!("Invalid external evaluation scores");} }
            if usage.cost.is_some_and(|n|!n.is_finite()||n<0.0)
                ||usage.cost_currency.as_ref().is_some_and(|s|s.len()!=3||!s.bytes().all(|b|b.is_ascii_uppercase())) {
                bail!("Invalid reported external cost");
            }
            if usage.first_text_ms.zip(span.duration_ms).is_some_and(|(first,total)|first>total) { bail!("First text time exceeds span duration"); }
            spans.push(Span{provider_id:span.provider_id,id:index,parent_id:span.parent_id,kind:format!("external_{}",span.kind),name:span.name,
                status:span.status,started_ms:span.started_ms,duration_ms:span.duration_ms,linked_trace_id:None,
                usage:public_usage(&json!({"input_tokens":usage.input_tokens,"output_tokens":usage.output_tokens,
                    "evaluation_sampling":usage.evaluation_sampling,"evaluator_ref":usage.evaluator_ref,"evaluation_scores":usage.evaluation_scores,"guardrail_receipt":usage.guardrail_receipt,"first_text_ms":usage.first_text_ms,"cache_read_input_tokens":usage.cache_read_input_tokens,"cache_creation_input_tokens":usage.cache_creation_input_tokens,"cost":usage.cost,"cost_currency":usage.cost_currency}))});
        }
        let keyed=request.idempotency_key.is_some();
        let id=if let Some(key)=request.idempotency_key {
            use sha2::{Digest,Sha256};
            // Scope keys to the project; no raw key is retained in metadata.
            let digest=Sha256::digest(serde_json::to_vec(&(request.project_id.as_str(),key))?);
            format!("trace-external-{}",digest.iter().map(|byte|format!("{byte:02x}")).collect::<String>())
        }else{format!("trace-external-{}",crate::evaluation::new_id())};
        let record=Record{id:id.clone(),project_id:request.project_id,session_id:format!("external-{}",request.correlation_id),
            started_ms:request.started_ms,status:spans[0].status.clone(),recovered_ms:None,spans};
        let path=self.root.join(format!("{id}.json"));
        let replay=|| -> Result<Value> {
            if self.is_removed(&id)? {bail!("External trace was removed; restore it explicitly");}
            let existing=self.read_record(&id)?;
            if serde_json::to_value(&existing)?!=serde_json::to_value(&record)? {bail!("External trace idempotency conflict");}
            Ok(json!({"id":id,"source":"external","span_count":record.spans.len(),"provider_calls":0,"deduplicated":true}))
        };
        if keyed&&path.try_exists()? {return replay();}
        if let Err(error)=crate::evaluation::commit_new(&path,&record) {
            if keyed&&path.try_exists()? {return replay();}
            return Err(error);
        }
        Ok(json!({"id":id,"source":"external","span_count":record.spans.len(),"provider_calls":0,"deduplicated":false}))
    }
}
pub(crate) async fn ingest(State(app):State<crate::App>,Json(request):Json<ExternalTrace>)->crate::ApiResult<Value>{
    if !app.projects.lock().unwrap().iter().any(|project|project.id==request.project_id) {
        return Err(crate::error(StatusCode::BAD_REQUEST,"Unknown external trace project"));
    }
    let project=request.project_id.clone();
    let completed=request.spans.first().is_some_and(|span|span.status=="completed") && request.spans.iter().all(|span|span.status!="running");
    app.observability.ingest(request).map(|receipt|{
        if completed {if let Some(id)=receipt["id"].as_str(){if let Err(error)=crate::online_evaluation::enqueue_completed(&app.data,&project,id){eprintln!("External online evaluation admission failed: {error}");}}}
        Json(receipt)
    }).map_err(|e|{
        let status=if e.to_string()=="External trace idempotency conflict"{StatusCode::CONFLICT}else{StatusCode::BAD_REQUEST};
        crate::error(status,e)
    })
}
impl Store {
    fn read_record(&self, id: &str) -> Result<Record> {
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
        Ok(record)
    }
    pub(crate) fn completed_trace_health(&self,id:&str,project:&str)->Result<(String,f64)> {
        self.validate_project_target(id,None,project)?;
        let record=self.read_record(id)?;
        if record.status!="completed"||record.spans.is_empty()||record.spans.iter().any(|span|span.status=="running"){bail!("Assessment requires a completed trace with spans");}
        let hash=Sha256::digest(serde_json::to_vec(&record)?).iter().map(|b|format!("{b:02x}")).collect();
        let score=record.spans.iter().filter(|span|span.status=="completed").count() as f64/record.spans.len() as f64;
        Ok((hash,score))
    }
    pub(crate) fn completed_trace_fingerprint(&self,id:&str,project:&str)->Result<String> {
        self.validate_project_target(id,None,project)?;
        let record=self.read_record(id)?;
        if record.status!="completed"||record.spans.iter().any(|span|span.status=="running"){bail!("Online selection requires a completed trace");}
        Ok(Sha256::digest(serde_json::to_vec(&record)?).iter().map(|b|format!("{b:02x}")).collect())
    }
    pub(crate) fn validate_project_target(&self,id:&str,span:Option<usize>,project:&str)->Result<()> {
        self.validate_target(id,span)?;
        if self.read_record(id)?.project_id!=project {bail!("Review target belongs to another project");}
        Ok(())
    }
    pub(crate) fn validate_target(&self, id: &str, span: Option<usize>) -> Result<()> {
        let record=self.read_record(id)?;
        if self.is_removed(id)? {bail!("Trace was removed");}
        if span.is_some_and(|n| !record.spans.iter().any(|s|s.id==n)){bail!("Trace span does not exist");}
        Ok(())
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
        };
        trace.persist()?;
        Ok(trace)
    }
    fn scan(&self, query: &Search) -> Result<Vec<Record>> {
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
            let id=path.file_stem().and_then(|id|id.to_str()).context("Invalid trace filename")?;
            let record=self.read_record(id)?;
            if self.is_removed(id)?!=query.removed.unwrap_or(false) {continue;}
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
            if query.status.as_ref().is_some_and(|status|status!=&record.status)
                || query.since_ms.is_some_and(|since|record.started_ms<since)
                || query.until_ms.is_some_and(|until|record.started_ms>until) {continue;}
            if (query.guardrail.is_some() || query.guardrail_policy_sha256.is_some()) && !record.spans.iter().any(|span| {
                query.guardrail.as_ref().unwrap_or(&GuardrailFilter::Any).matches(span)
                    && query.guardrail_policy_sha256.as_ref().is_none_or(|hash|span.name.rsplit('.').next()==Some(hash.as_str()))
            }) { continue; }
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
    fn list(&self, query:&Search)->Result<Value>{
        if query.since_ms.zip(query.until_ms).is_some_and(|(since,until)|since>until) {bail!("Invalid trace time range");}
        if query.status.as_ref().is_some_and(|status|status.is_empty() || status.len()>40 || !status.bytes().all(|b|b.is_ascii_lowercase() || b==b'_')) {bail!("Invalid trace status filter");}
        if query.guardrail_policy_sha256.as_ref().is_some_and(|hash| !valid_policy_hash(hash)) { bail!("Invalid guardrail policy SHA-256"); }
        let limit=query.limit.unwrap_or(50);
        if limit==0||limit>200||query.offset.unwrap_or(0)>100_000{bail!("Invalid trace pagination");}
        let rows=self.scan(query)?;
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
    fn time_series(&self,query:&TimeSearch)->Result<Value>{
        if query.status.as_ref().is_some_and(|status|status.is_empty() || status.len()>40 || !status.bytes().all(|b|b.is_ascii_lowercase() || b==b'_')) {bail!("Invalid trace series status");}
        if query.since_ms>query.until_ms || query.until_ms==u64::MAX || query.bucket_ms==0 || query.bucket_ms>86_400_000 {bail!("Invalid trace series time range or interval");}
        let count=(query.until_ms-query.since_ms)/query.bucket_ms+1;
        if count>500 {bail!("Trace series exceeds 500 intervals");}
        let mut buckets:Vec<TimeBucket>=(0..count).map(|_|TimeBucket::default()).collect();
        let records=self.scan(&Search{status:query.status.clone(),project_id:query.project_id.clone(),session_id:query.session_id.clone(),since_ms:Some(query.since_ms),until_ms:Some(query.until_ms),..Default::default()})?;
        for record in records {
            let index=((record.started_ms-query.since_ms)/query.bucket_ms) as usize;
            let bucket=&mut buckets[index];bucket.traces+=1;*bucket.statuses.entry(record.status).or_default()+=1;
            for span in &record.spans {
                bucket.guardrails.observe(span);
                if matches!(span.kind.as_str(),"model"|"external_model") {bucket.models.observe(span)?;}
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
        Ok(json!({"kind":"trace_time_series","schema_version":1,"status":query.status,"since_ms":query.since_ms,"until_ms":query.until_ms,"bucket_ms":query.bucket_ms,"bucket_count":count,"bucket_limit":500,"buckets":rows,"assignment":"trace_started_ms","project_id":query.project_id,"session_id":query.session_id,"provider_calls":0,"privacy":{"content_capture":false}}))
    }
    fn summary(&self,query:&SummarySearch)->Result<Value>{
        if query.status.as_ref().is_some_and(|status|status.is_empty() || status.len()>40 || !status.bytes().all(|b|b.is_ascii_lowercase() || b==b'_')) {bail!("Invalid summary trace status");}
        let conversation_offset=query.conversation_offset.unwrap_or(0);
        let conversation_limit=query.conversation_limit.unwrap_or(100);
        if conversation_offset>10_000 || !(1..=100).contains(&conversation_limit) {bail!("Invalid conversation pagination");}
        if query.since_ms.zip(query.until_ms).is_some_and(|(a,b)|a>b){bail!("Invalid summary time range");}
        let records=self.scan(&Search{status:query.status.clone(),project_id:query.project_id.clone(),session_id:query.session_id.clone(),..Default::default()})?;
        let mut statuses=std::collections::BTreeMap::<String,u64>::new();
        let mut external_model_calls=0u64;let mut model_calls=0u64;let mut traces=0u64;let mut input=0u64;let mut output=0u64;let mut input_unknown=0u64;let mut output_unknown=0u64;
        let mut cost_unknown=0u64;let mut currency_unknown=0u64;let mut currencies=std::collections::BTreeMap::<String,f64>::new();let mut cost_examples=Vec::new();
        let mut models=std::collections::BTreeMap::<(String,Option<String>,String),ModelTotals>::new();
        let mut guardrails=GuardrailSummary::default();
        let mut conversations=std::collections::BTreeMap::<(String,String),ConversationTotals>::new();
        for record in records{
            if query.since_ms.is_some_and(|since|record.started_ms<since)||query.until_ms.is_some_and(|until|record.started_ms>until){continue;}
            traces+=1;*statuses.entry(record.status.clone()).or_default()+=1;
            let conversation=conversations.entry((record.project_id.clone(),record.session_id.clone())).or_default();
            conversation.traces+=1;*conversation.statuses.entry(record.status.clone()).or_default()+=1;
            conversation.first=Some(conversation.first.map_or(record.started_ms,|first|first.min(record.started_ms)));
            conversation.last=conversation.last.max(record.started_ms);
            for span in &record.spans { guardrails.observe(span); }
            for span in record.spans.iter().filter(|span|span.kind=="model"||span.kind=="external_model"){
                conversation.models.observe(span)?;
                if span.kind=="external_model" {conversation.external_calls+=1;}
                let source=if span.kind=="external_model"{"external"}else{"native"};
                models.entry((source.into(),span.provider_id.clone(),span.name.clone())).or_default().observe(span)?;
                model_calls+=1;if span.kind=="external_model"{external_model_calls+=1;}
                match span.usage["prompt_tokens"].as_u64().or_else(||span.usage["input_tokens"].as_u64()){Some(n)=>input=input.checked_add(n).context("Input token total overflow")?,None=>input_unknown+=1}
                match span.usage["completion_tokens"].as_u64().or_else(||span.usage["output_tokens"].as_u64()){Some(n)=>output=output.checked_add(n).context("Output token total overflow")?,None=>output_unknown+=1}
                if let Some(cost)=span.usage["cost"].as_f64().filter(|cost|cost.is_finite()&&*cost>=0.0){
                    let currency=span.usage["cost_currency"].as_str().filter(|code|code.len()==3&&code.bytes().all(|b|b.is_ascii_uppercase()));
                    if let Some(currency)=currency{let total=currencies.entry(currency.into()).or_default();*total+=cost;if !total.is_finite(){bail!("Cost total overflow");}}else{currency_unknown+=1;}
                    if cost_examples.len()<100{cost_examples.push(json!({"trace_id":record.id,"span_id":span.id,"model":span.name,"value":cost,"currency":currency}));}
                }else{cost_unknown+=1;}
            }
        }
        let (cache_write,cache_read,cache_write_unknown,cache_read_unknown)=models.values().try_fold((0u64,0u64,0u64,0u64),|a,t|->Result<_>{Ok((a.0.checked_add(t.cache_write).context("Cache creation total overflow")?,a.1.checked_add(t.cache_read).context("Cache read total overflow")?,a.2+t.cache_write_unknown,a.3+t.cache_read_unknown))})?;
        let first_text=first_text_receipt(models.values().flat_map(|model|model.first_text.iter().copied()).collect(),models.values().map(|model|model.first_text_unknown).sum());
        let models:Vec<_>=models.into_iter().map(|((source,provider_id,model),totals)|totals.receipt(source,provider_id,model)).collect();
        let conversation_count=conversations.len();
        let mut conversations:Vec<_>=conversations.into_iter().collect();
        conversations.sort_by(|a,b|b.1.last.cmp(&a.1.last).then_with(||a.0.cmp(&b.0)));
        let conversations:Vec<_>=conversations.into_iter().skip(conversation_offset).take(conversation_limit).map(|((project_id,session_id),totals)| {
            let mut usage=totals.models.receipt("mixed".into(),None,"all_models".into());
            if let Some(object)=usage.as_object_mut(){object.remove("source");object.remove("provider_id");object.remove("model");}
            json!({"project_id":project_id,"session_id":session_id,"trace_count":totals.traces,"trace_statuses":totals.statuses,"first_started_ms":totals.first,"last_started_ms":totals.last,"external_model_calls":totals.external_calls,"model_usage":usage})
        }).collect();
        Ok(json!({"status":query.status,"conversations":conversations,"conversation_count":conversation_count,"conversation_offset":conversation_offset,"conversation_limit":conversation_limit,"conversation_has_more":conversation_offset+conversations.len()<conversation_count,"conversations_truncated":conversation_offset>0||conversation_count>conversations.len(),"conversation_order":"last_started_ms_desc_project_session_asc","first_text":first_text,"guardrails":guardrails.receipt(),"models":models,"cache_creation_input_tokens":cache_write,"cache_read_input_tokens":cache_read,"cache_creation_unknown_calls":cache_write_unknown,"cache_read_unknown_calls":cache_read_unknown,"trace_count":traces,"trace_statuses":statuses,"model_calls":model_calls,"external_model_calls":external_model_calls,"input_tokens":input,"output_tokens":output,"input_tokens_unknown_calls":input_unknown,"output_tokens_unknown_calls":output_unknown,"reported_cost_by_currency":currencies,"cost_unknown_calls":cost_unknown,"cost_currency_unknown_calls":currency_unknown,"reported_cost_examples":cost_examples,"cost_examples_limit":100,"scope":"model_spans","price_estimates":false,"unknown_currency_costs_summed":false,"since_ms":query.since_ms,"until_ms":query.until_ms}))
    }
}
impl Trace {
    pub(crate) fn id(&self) -> String {
        self.record.lock().unwrap().id.clone()
    }
    pub(crate) async fn generation(
        &self,
        name: &str,
        provider_id: &str,
        parent: usize,
        future: impl std::future::Future<Output = Result<(crate::types::Message, Value)>>,
    ) -> Result<(crate::types::Message, Value)> {
        let mut span = self.span("model", name, Some(parent));
        span.set_provider(provider_id);
        let result = future.await;
        match &result {
            Ok((message, usage)) => span.finish(
                if message.truncated {
                    "token_limit"
                } else {
                    "completed"
                },
                usage,
            ),
            Err(_) => span.finish("failed", &Value::Null),
        }
        result
    }
    pub(crate) fn span(&self, kind: &str, name: &str, parent: Option<usize>) -> Guard {
        let index = {
            let mut record = self.record.lock().unwrap();
            let index = record.spans.len();
            record.spans.push(Span {
                id: index,
                parent_id: parent,
                kind: kind.into(),
                name: name.chars().take(200).collect(),
                status: "running".into(),
                started_ms: now(),
                duration_ms: None,
                usage: Value::Null,
                linked_trace_id: None,
                provider_id: None,
            });
            index
        };
        self.flush();
        Guard {
            trace: self.clone(),
            index,
            started: Instant::now(),
            finished: false,
        }
    }
    fn persist(&self) -> Result<()> {
        let _writer = self.writer.lock().unwrap();
        atomic_write(self.path.as_ref(), &self.record.lock().unwrap())
    }
    fn flush(&self) {
        if let Err(error) = self.persist() {
            eprintln!("Studio trace persistence failed: {error}");
        }
    }
}
impl Guard {
    pub(crate) fn set_provider(&mut self,provider:&str){
        if provider.is_empty()||provider.len()>100||!provider.bytes().all(|b|b.is_ascii_alphanumeric()||b"._-:/".contains(&b)){return;}
        self.trace.record.lock().unwrap().spans[self.index].provider_id=Some(provider.into());
        self.trace.flush();
    }
    pub(crate) fn id(&self) -> usize {
        self.index
    }
    pub(crate) fn link_trace(&mut self, id: &str) {
        if id.is_empty() || id.len()>80 || !id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_') { return; }
        self.trace.record.lock().unwrap().spans[self.index].linked_trace_id=Some(id.into());
        self.trace.flush();
    }
    pub(crate) fn finish(&mut self, status: &str, usage: &Value) {
        if self.finished {
            return;
        }
        self.finished = true;
        {
            let mut record = self.trace.record.lock().unwrap();
            let span = &mut record.spans[self.index];
            span.status = status.into();
            span.duration_ms =
                Some(self.started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            span.usage = public_usage(usage);
            if self.index == 0 {
                record.status = status.into();
            }
        }
        self.trace.flush();
        if self.index==0&&status=="completed" {
            let (project,id)={let record=self.trace.record.lock().unwrap();(record.project_id.clone(),record.id.clone())};
            if let Some(data)=self.trace.path.parent().and_then(|path|path.parent()).and_then(|path|path.parent()) {
                if let Err(error)=crate::online_evaluation::enqueue_completed(data,&project,&id){eprintln!("Online evaluation admission failed: {error}");}
            }
        }
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if !self.finished {
            self.finish("interrupted", &Value::Null);
        }
    }
}
pub(crate) fn public_usage(usage: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for name in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cost",
    ] {
        if let Some(value) = usage
            .get(name)
            .filter(|v| v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0))
        {
            result.insert(name.into(), value.clone());
        }
    }
    if let Some(first)=usage["first_text_ms"].as_u64(){result.insert("first_text_ms".into(),json!(first));}
    if let Some(cached) = usage["prompt_tokens_details"]["cached_tokens"].as_u64() {
        result.insert("cached_tokens".into(), json!(cached));
    }
    if let Some(currency)=usage["cost_currency"].as_str().filter(|code|code.len()==3&&code.bytes().all(|b|b.is_ascii_uppercase())){result.insert("cost_currency".into(),json!(currency));}
    if let Some(reference)=valid_evaluator_ref(&usage["evaluator_ref"]) {result.insert("evaluator_ref".into(),reference);}
    if let Some(sampling)=valid_evaluation_sampling(&usage["evaluation_sampling"]) {result.insert("evaluation_sampling".into(),sampling);}
    if let Some(scores)=valid_evaluation_scores(&usage["evaluation_scores"]) {result.insert("evaluation_scores".into(),scores);}
    if let Some(receipt)=public_guardrail_receipt(&usage["guardrail_receipt"]) {result.insert("guardrail_receipt".into(),receipt);}
    json!(result)
}
fn valid_evaluator_ref(value:&Value)->Option<Value>{
    if value.as_object()?.len()!=2||!value["version"].as_u64().is_some_and(|version|version>0)||!value["id"].as_str().is_some_and(|id|!id.is_empty()&&id.len()<=100&&id.bytes().all(|c|c.is_ascii_alphanumeric()||b"._-".contains(&c))) {return None;}
    Some(value.clone())
}
fn valid_evaluation_sampling(value:&Value)->Option<Value>{
    let fields=value.as_object()?;
    if fields.len()!=3 || value["method"]!="sha256_v1" || !value["sample_rate"].as_f64().is_some_and(|rate|rate.is_finite()&&(0.0..=1.0).contains(&rate)) || !value["selected"].is_boolean() {return None;}
    Some(value.clone())
}
fn valid_evaluation_scores(value:&Value)->Option<Value>{
    let scores=value.as_object()?;
    if scores.is_empty()||scores.len()>20||scores.iter().any(|(metric,value)|metric.is_empty()||metric.len()>100||!metric.bytes().all(|c|c.is_ascii_alphanumeric()||b"._-".contains(&c))||!value.as_f64().is_some_and(|score|score.is_finite()&&(0.0..=1.0).contains(&score))) {return None;}
    Some(value.clone())
}
fn public_guardrail_receipt(value:&Value)->Option<Value> {
    let stage=value["stage"].as_str()?;let action=value["action"].as_str()?;let hash=value["policy_sha256"].as_str()?;
    if value["kind"]!="local_guardrail"||value["schema_version"]!=1||!["input","output"].contains(&stage)||!["observe","block"].contains(&action)||!valid_policy_hash(hash)||value["provider_calls"]!=0||value["content_captured"]!=false {return None;}
    let passed=value["passed"].as_bool()?;let blocked=value["blocked"].as_bool()?;
    if blocked!=(!passed&&action=="block") {return None;}
    let raw=value["rules"].as_array()?;if !(1..=32).contains(&raw.len()) {return None;}
    let mut ids=std::collections::HashSet::new();let mut rules=Vec::new();
    for rule in raw {let id=rule["rule_id"].as_str()?;let kind=rule["kind"].as_str()?;let outcome=rule["passed"].as_bool()?;
        if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b"._-".contains(&b))||!ids.insert(id)||!["min_bytes","max_bytes","json_valid","forbidden_substrings","required_substrings"].contains(&kind) {return None;}
        rules.push(json!({"rule_id":id,"kind":kind,"passed":outcome}));
    }
    if rules.iter().all(|rule|rule["passed"]==true)!=passed {return None;}
    Some(json!({"kind":"local_guardrail","schema_version":1,"stage":stage,"action":action,"passed":passed,"blocked":blocked,"rules":rules,"policy_sha256":hash,"provider_calls":0,"content_captured":false}))
}
fn valid_policy_hash(hash:&str)->bool {
    hash.len()==64 && hash.bytes().all(|b|b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Deserialize)]
#[serde(rename_all="snake_case")]
enum GuardrailFilter { Any, Failed, Blocked }
impl GuardrailFilter {
    fn matches(&self, span:&Span)->bool {
        if !matches!(span.kind.as_str(),"tool"|"external_tool") { return false; }
        let parts:Vec<_>=span.name.split('.').collect();
        if parts.len()!=5 || parts[0]!="guardrail" || !matches!(parts[1],"input"|"output")
            || !matches!(parts[2],"observe"|"block") || !matches!(parts[3],"pass"|"fail")
            || !valid_policy_hash(parts[4]) { return false; }
        match self { Self::Any=>true, Self::Failed=>parts[3]=="fail", Self::Blocked=>parts[2]=="block" && parts[3]=="fail" && span.status=="failed" }
    }
}
#[derive(Default)]
struct GuardrailTotals { checks:u64, passed:u64, violations:u64, blocked:u64, incomplete:u64, input:u64, output:u64 }
impl GuardrailTotals {
    fn observe(&mut self,span:&Span) {
        if !GuardrailFilter::Any.matches(span) { return; }
        let parts:Vec<_>=span.name.split('.').collect();
        self.checks+=1;
        if parts[1]=="input" { self.input+=1; } else { self.output+=1; }
        if parts[3]=="pass" && span.status=="completed" { self.passed+=1; }
        else if parts[3]=="fail" && ((parts[2]=="observe" && span.status=="completed") || (parts[2]=="block" && span.status=="failed")) {
            self.violations+=1;
            if parts[2]=="block" { self.blocked+=1; }
        } else { self.incomplete+=1; }
    }
    fn receipt(&self)->Value {
        json!({"checks":self.checks,"passed":self.passed,"violations":self.violations,"blocked":self.blocked,"incomplete_or_inconsistent":self.incomplete,"input_checks":self.input,"output_checks":self.output,"scope":"reported_guardrail_spans","content_capture":false})
    }
}
#[derive(Default)]
struct GuardrailSummary {
    totals:GuardrailTotals,
    policies:std::collections::BTreeMap<String,GuardrailTotals>,
    truncated:bool,
}
impl GuardrailSummary {
    fn observe(&mut self,span:&Span) {
        if !GuardrailFilter::Any.matches(span) { return; }
        self.totals.observe(span);
        let hash=span.name.rsplit('.').next().unwrap();
        if !self.policies.contains_key(hash) && self.policies.len()==100 {
            self.truncated=true;
            let largest=self.policies.last_key_value().unwrap().0.clone();
            if hash>=largest.as_str() { return; }
            self.policies.remove(&largest);
        }
        self.policies.entry(hash.into()).or_default().observe(span);
    }
    fn receipt(&self)->Value {
        let mut receipt=self.totals.receipt();
        let policies:Vec<_>=self.policies.iter().map(|(hash,totals)| {
            let mut row=totals.receipt();row["policy_sha256"]=json!(hash);row
        }).collect();
        receipt["policies"]=json!(policies);
        receipt["policy_groups_limit"]=json!(100);
        receipt["policy_groups_truncated"]=json!(self.truncated);
        receipt["policy_group_order"]=json!("policy_sha256_ascending");
        receipt
    }
}
#[derive(Default, Deserialize)]
pub(crate) struct Search {
    status: Option<String>,
    since_ms: Option<u64>,
    until_ms: Option<u64>,
    guardrail: Option<GuardrailFilter>,
    guardrail_policy_sha256: Option<String>,
    removed: Option<bool>,
    session_id: Option<String>,
    project_id: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}
#[derive(Default)]
struct ConversationTotals {
    traces:u64,
    statuses:std::collections::BTreeMap<String,u64>,
    first:Option<u64>,
    last:u64,
    external_calls:u64,
    models:ModelTotals,
}
#[derive(Default)]
struct TimeBucket { traces:u64,statuses:std::collections::BTreeMap<String,u64>,models:ModelTotals,guardrails:GuardrailTotals }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeSearch { status:Option<String>,project_id:Option<String>,session_id:Option<String>,since_ms:u64,until_ms:u64,bucket_ms:u64 }
pub(crate) async fn time_series(State(app):State<crate::App>,Query(query):Query<TimeSearch>)->crate::ApiResult<Value>{
    app.observability.time_series(&query).map(Json).map_err(|error|crate::error(StatusCode::BAD_REQUEST,error))
}
#[derive(Default, Deserialize)]
pub(crate) struct SummarySearch{status:Option<String>,conversation_offset:Option<usize>,conversation_limit:Option<usize>,project_id:Option<String>,session_id:Option<String>,since_ms:Option<u64>,until_ms:Option<u64>}
pub(crate) async fn summary(State(app):State<crate::App>,Query(query):Query<SummarySearch>)->crate::ApiResult<Value>{
    app.observability.summary(&query).map(Json).map_err(|error|crate::error(StatusCode::BAD_REQUEST,error))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationSearch {project_id:String,since_ms:Option<u64>,until_ms:Option<u64>}
impl Store {
    fn evaluation_summary(&self,query:&EvaluationSearch)->Result<Value>{
        if query.project_id.is_empty()||query.project_id.len()>100||!query.project_id.bytes().all(|c|c.is_ascii_alphanumeric()||b"-_".contains(&c))||query.since_ms.zip(query.until_ms).is_some_and(|(a,b)|a>b) {bail!("Invalid evaluation summary scope");}
        let records=self.scan(&Search {project_id:Some(query.project_id.clone()),since_ms:query.since_ms,until_ms:query.until_ms,..Default::default()})?;
        let mut selected=0u64;let mut skipped=0u64;let mut failed=0u64;let mut completed=0u64;
        let mut groups=std::collections::BTreeMap::<(Option<String>,Option<u64>,String),(u64,f64,f64,f64)>::new();
        let mut attempts=std::collections::BTreeMap::<(Option<String>,Option<u64>),(u64,u64)>::new();
        for record in &records {for span in &record.spans {
            if let Some(receipt)=valid_evaluation_sampling(&span.usage["evaluation_sampling"]) {if receipt["selected"]==true {selected+=1;} else {skipped+=1;}}
            if span.kind!="external_tool" {continue;}
            let reference=valid_evaluator_ref(&span.usage["evaluator_ref"]);
            let scores=valid_evaluation_scores(&span.usage["evaluation_scores"]);
            if (span.name=="evaluation"||reference.is_some()||scores.is_some()) && matches!(span.status.as_str(),"failed"|"completed") {
                let key=(reference.as_ref().and_then(|r|r["id"].as_str()).map(str::to_owned),reference.as_ref().and_then(|r|r["version"].as_u64()));
                let row=attempts.entry(key).or_default();
                if span.status=="failed" {failed+=1;row.1+=1;}else {completed+=1;row.0+=1;}
                if attempts.len()>4000 {bail!("Evaluation summary evaluator limit reached");}
            }
            if span.status!="completed" {continue;}
            if let Some(scores)=scores {for (metric,value) in scores.as_object().unwrap() {
                let key=(reference.as_ref().and_then(|r|r["id"].as_str()).map(str::to_owned),reference.as_ref().and_then(|r|r["version"].as_u64()),metric.clone());
                let score=value.as_f64().unwrap();let row=groups.entry(key).or_insert((0,0.0,score,score));row.0+=1;row.1+=score;row.2=row.2.min(score);row.3=row.3.max(score);
                if groups.len()>4000 {bail!("Evaluation summary group limit reached");}
            }}
        }}
        let metrics=groups.into_iter().map(|((id,version,metric),(count,sum,min,max))|json!({"evaluator_id":id,"evaluator_version":version,"metric":metric,"count":count,"mean":(sum/count as f64).clamp(min,max),"min":min,"max":max})).collect::<Vec<_>>();
        let evaluators=attempts.into_iter().map(|((id,version),(completed,failed))|json!({"evaluator_id":id,"evaluator_version":version,"completed_assessments":completed,"failed_assessments":failed})).collect::<Vec<_>>();
        Ok(json!({"kind":"callback_evaluation_summary","project_id":query.project_id,"since_ms":query.since_ms,"until_ms":query.until_ms,"trace_count":records.len(),"selected_tasks":selected,"skipped_tasks":skipped,"completed_assessments":completed,"failed_assessments":failed,"evaluators":evaluators,"metrics":metrics,"assessment_source":"caller_reported","provider_calls":0,"automatic_promotion":false}))
    }
}
pub(crate) async fn evaluation_summary(State(app):State<crate::App>,Query(query):Query<EvaluationSearch>)->crate::ApiResult<Value>{
    app.observability.evaluation_summary(&query).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
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
pub(crate) async fn get(State(app):State<crate::App>,axum::extract::Path(id):axum::extract::Path<String>)->crate::ApiResult<Value>{
    app.observability.validate_target(&id,None).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))?;
    app.observability.read_record(&id).and_then(|record|Ok(serde_json::to_value(record)?)).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
#[derive(Default,Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportOptions{include_feedback:Option<bool>}
fn export_packet(app:&crate::App,id:&str,include_feedback:bool)->Result<Value>{
    app.observability.validate_target(id,None)?;
    let record=app.observability.read_record(id)?;
    if record.status=="running"||record.spans.iter().any(|span|span.status=="running"){bail!("Export requires a completed trace snapshot");}
    let feedback=if include_feedback{Some(app.feedback.export_snapshot(id)?)}else{None};
    let packet=json!({"kind":"trace_export","schema_version":1,"exported_ms":now(),"trace":record,"feedback":feedback,
        "privacy":{"runtime_content_capture":false,"feedback_included":include_feedback},"provider_calls":0});
    if serde_json::to_vec(&packet)?.len()>3*1024*1024 {bail!("Trace export exceeds 3 MiB");}
    Ok(packet)
}
pub(crate) async fn export(State(app):State<crate::App>,axum::extract::Path(id):axum::extract::Path<String>,Query(options):Query<ExportOptions>)->crate::ApiResult<Value>{
    export_packet(&app,&id,options.include_feedback.unwrap_or(false)).map(Json).map_err(|error|crate::error(StatusCode::BAD_REQUEST,error))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BulkExportRequest{project_id:String,trace_ids:Vec<String>,#[serde(default)]include_feedback:bool}
pub(crate) async fn bulk_export(State(app):State<crate::App>,Json(request):Json<BulkExportRequest>)->crate::ApiResult<Value>{
    let result=(||->Result<Value>{
        if request.project_id.is_empty()||request.project_id.len()>80||request.trace_ids.is_empty()||request.trace_ids.len()>100{bail!("Choose a project and 1-100 unique traces");}
        let mut seen=std::collections::HashSet::new();
        for id in &request.trace_ids{if !seen.insert(id){bail!("Duplicate export trace ID");}}
        let mut packets=Vec::new();let mut bytes=0usize;
        for id in &request.trace_ids{
            let packet=export_packet(&app,id,request.include_feedback)?;
            if packet["trace"]["project_id"]!=request.project_id{bail!("Trace export project mismatch");}
            bytes=bytes.checked_add(serde_json::to_vec(&packet)?.len()).context("Export size overflow")?;
            if bytes>16*1024*1024{bail!("Bulk trace export exceeds 16 MiB");}
            packets.push(packet);
        }
        let packet=json!({"kind":"trace_export_batch","schema_version":1,"project_id":request.project_id,"exported_ms":now(),
            "trace_count":packets.len(),"traces":packets,"privacy":{"runtime_content_capture":false,"feedback_included":request.include_feedback},"provider_calls":0});
        if serde_json::to_vec(&packet)?.len()>16*1024*1024{bail!("Bulk trace export exceeds 16 MiB");}
        Ok(packet)
    })();
    result.map(Json).map_err(|error|crate::error(StatusCode::BAD_REQUEST,error))
}
pub(crate) async fn remove(State(app):State<crate::App>,axum::extract::Path(id):axum::extract::Path<String>)->crate::ApiResult<Value>{
    app.observability.change_visibility(&id,true).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn restore(State(app):State<crate::App>,axum::extract::Path(id):axum::extract::Path<String>)->crate::ApiResult<Value>{
    app.observability.change_visibility(&id,false).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_trace_health_counts_failed_spans_and_rejects_running_or_removed(){
        let data=std::env::temp_dir().join(format!("allpaka-health-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let trace=store.begin("health","project").unwrap();let mut root=trace.span("turn","agent",None);let mut failed=trace.span("tool","failed-tool",Some(root.id()));let mut successful=trace.span("model","model",Some(root.id()));
        assert!(store.completed_trace_health(&trace.id(),"project").is_err());
        root.finish("completed",&Value::Null);failed.finish("failed",&Value::Null);
        assert!(store.completed_trace_health(&trace.id(),"project").is_err());
        successful.finish("completed",&Value::Null);
        let (hash,score)=store.completed_trace_health(&trace.id(),"project").unwrap();assert_eq!(score,2.0/3.0);assert_eq!(hash,store.completed_trace_fingerprint(&trace.id(),"project").unwrap());
        assert_eq!(Store::new(&data).unwrap().completed_trace_health(&trace.id(),"project").unwrap(),(hash,score));
        assert!(store.completed_trace_health(&trace.id(),"foreign").is_err());
        store.change_visibility(&trace.id(),true).unwrap();assert!(store.completed_trace_health(&trace.id(),"project").is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn completed_trace_fingerprint_is_stable_and_detects_changed_metadata(){
        let data=std::env::temp_dir().join(format!("allpaka-trace-pin-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let trace=store.begin("source","project").unwrap();let mut root=trace.span("turn","agent",None);
        assert!(store.completed_trace_fingerprint(&trace.id(),"project").is_err());
        root.finish("completed",&Value::Null);
        let hash=store.completed_trace_fingerprint(&trace.id(),"project").unwrap();
        assert_eq!(Store::new(&data).unwrap().completed_trace_fingerprint(&trace.id(),"project").unwrap(),hash);
        assert!(store.completed_trace_fingerprint(&trace.id(),"other").is_err());
        trace.span("tool","extra",Some(root.id())).finish("completed",&Value::Null);
        assert_ne!(store.completed_trace_fingerprint(&trace.id(),"project").unwrap(),hash);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn callback_summary_separates_evaluator_versions_and_project_scope() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        for (project,version,score,status) in [("p",1,0.2,"completed"),("p",1,0.8,"completed"),("p",2,1.0,"completed"),("other",1,0.0,"completed"),("p",3,0.0,"failed")] {
            let usage=if status=="failed" {json!({"evaluator_ref":{"id":"check","version":version}})} else {json!({"evaluation_scores":{"quality":score},"evaluator_ref":{"id":"check","version":version}})};
            store.ingest(serde_json::from_value(json!({"project_id":project,"correlation_id":"summary","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":status,"started_ms":1000,"duration_ms":1,"usage":usage}]})).unwrap()).unwrap();
        }
        let query=EvaluationSearch {project_id:"p".into(),since_ms:None,until_ms:None};let report=store.evaluation_summary(&query).unwrap();assert_eq!(report["trace_count"],4);assert_eq!(report["failed_assessments"],1);assert_eq!(report["completed_assessments"],3);assert_eq!(report["metrics"].as_array().unwrap().len(),2);assert_eq!(report["metrics"][0]["count"],2);assert_eq!(report["metrics"][0]["mean"],0.5);assert_eq!(report["metrics"][1]["evaluator_version"],2);
        assert_eq!(report["evaluators"],json!([
            {"evaluator_id":"check","evaluator_version":1,"completed_assessments":2,"failed_assessments":0},
            {"evaluator_id":"check","evaluator_version":2,"completed_assessments":1,"failed_assessments":0},
            {"evaluator_id":"check","evaluator_version":3,"completed_assessments":0,"failed_assessments":1}
        ]));
        store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"legacy-failure","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":"failed","started_ms":1000,"duration_ms":1}]})).unwrap()).unwrap();
        let legacy=store.evaluation_summary(&query).unwrap();assert_eq!(legacy["evaluators"][0],json!({"evaluator_id":null,"evaluator_version":null,"completed_assessments":0,"failed_assessments":1}));assert_eq!(legacy["failed_assessments"],2);
        for _ in 0..20 {store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"rounding","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":"completed","started_ms":1000,"duration_ms":1,"usage":{"evaluation_scores":{"constant":0.1},"evaluator_ref":{"id":"constant-check","version":1}}}]})).unwrap()).unwrap();}
        let constant=store.evaluation_summary(&query).unwrap();let group=constant["metrics"].as_array().unwrap().iter().find(|row|row["metric"]=="constant").unwrap();assert_eq!(group["mean"],group["max"]);assert_eq!(group["count"],20);
        assert_eq!(store.evaluation_summary(&EvaluationSearch {since_ms:Some(1001),..query}).unwrap()["trace_count"],0);
        assert!(store.evaluation_summary(&EvaluationSearch {project_id:"p".into(),since_ms:Some(2),until_ms:Some(1)}).is_err());std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn sampling_receipts_reverify_sdk_hash_decisions() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let request=|rate:f64,selected:bool|json!({"project_id":"p","correlation_id":"sampling","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"task","status":"completed","started_ms":1000,"duration_ms":1,"usage":{"evaluation_sampling":{"method":"sha256_v1","sample_rate":rate,"selected":selected}}}]});
        for (rate,selected) in [(0.0,false),(1.0,true),(0.5,true)] {assert!(store.ingest(serde_json::from_value(request(rate,selected)).unwrap()).is_ok());assert!(store.ingest(serde_json::from_value(request(rate,!selected)).unwrap()).is_err());}
        assert!(valid_evaluation_sampling(&json!({"method":"sha256_v1","sample_rate":true,"selected":false})).is_none());
        assert!(valid_evaluation_sampling(&json!({"method":"sha256_v1","sample_rate":0.5,"selected":true,"private":"text"})).is_none());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn evaluation_scores_are_bounded_numeric_metadata() {
        assert!(valid_evaluator_ref(&json!({"id":"check","version":2})).is_some());
        for invalid in [json!({"id":"check","version":true}),json!({"id":"private text","version":1}),json!({"id":"check","version":1,"private":"text"})] {assert!(valid_evaluator_ref(&invalid).is_none());}
        assert_eq!(public_usage(&json!({"evaluation_scores":{"quality":0.5}}))["evaluation_scores"]["quality"],0.5);
        for invalid in [json!({}),json!({"metric":true}),json!({"metric":1.1}),json!({"private text":0.5}),json!({"metric":"PRIVATE"})] {assert!(valid_evaluation_scores(&invalid).is_none());assert!(public_usage(&json!({"evaluation_scores":invalid}))["evaluation_scores"].is_null());}
    }
    #[test]
    fn guardrail_receipts_strip_private_values_and_reject_inconsistent_results() {
        let mut receipt=json!({"kind":"local_guardrail","schema_version":1,"stage":"input","action":"block","passed":false,"blocked":true,"rules":[{"rule_id":"limit","kind":"max_bytes","passed":false,"value":"PRIVATE"}],"policy_sha256":"a".repeat(64),"provider_calls":0,"content_captured":false,"text":"PRIVATE"});
        let clean=public_usage(&json!({"guardrail_receipt":receipt}));assert!(clean["guardrail_receipt"].is_object());assert!(!clean.to_string().contains("PRIVATE"));
        receipt["passed"]=json!(true);assert!(public_usage(&json!({"guardrail_receipt":receipt}))["guardrail_receipt"].is_null());
    }
    #[test]
    fn same_model_keeps_provider_unknown_and_external_groups_distinct() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let trace=store.begin("s","p").unwrap();let mut root=trace.span("turn","agent",None);
        for (provider,tokens) in [(Some("a"),1),(Some("b"),2),(None,3)] {
            let mut model=trace.span("model","same-model",Some(root.id()));if let Some(provider)=provider {model.set_provider(provider);}
            model.finish("completed",&json!({"input_tokens":tokens}));
        }
        root.finish("completed",&Value::Null);
        store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"ext","started_ms":1000,"spans":[
            {"parent_id":null,"kind":"model","name":"same-model","provider_id":"a","status":"completed","started_ms":1000,"duration_ms":0,"usage":{"input_tokens":4}}]})).unwrap()).unwrap();
        let receipt=store.summary(&SummarySearch::default()).unwrap();let groups=receipt["models"].as_array().unwrap();
        assert_eq!(groups.len(),4);assert_eq!(receipt["input_tokens"],10);
        assert!(groups.iter().any(|group|group["source"]=="native"&&group["provider_id"].is_null()&&group["input_tokens"]==3));
        assert!(groups.iter().any(|group|group["source"]=="external"&&group["provider_id"]=="a"&&group["input_tokens"]==4));
        assert_eq!(groups.iter().map(|group|group["input_tokens"].as_u64().unwrap()).sum::<u64>(),10);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn model_quantiles_keep_unknown_durations_and_cost_currencies_separate() {
        let mut totals=ModelTotals::default();
        for (index,duration) in [Some(10),Some(20),Some(100),None].into_iter().enumerate() {
            let span:Span=serde_json::from_value(json!({"id":index,"parent_id":null,"kind":"model","name":"m",
                "status":if index==2{"failed"}else{"completed"},"started_ms":0,"duration_ms":duration,
                "usage":if index==0{json!({"prompt_tokens":3,"input_tokens":99,"cost":0.2,"cost_currency":"USD"})}
                    else if index==1{json!({"input_tokens":2,"cost":0.3,"cost_currency":"EUR"})}else{json!({})}})).unwrap();
            totals.observe(&span).unwrap();
        }
        let result=totals.receipt("native".into(),None,"m".into());
        assert_eq!(result["calls"],4);assert_eq!(result["input_tokens"],5);assert_eq!(result["input_tokens_unknown_calls"],2);
        assert_eq!(result["duration_known_calls"],3);assert_eq!(result["duration_unknown_calls"],1);
        assert_eq!(result["duration_p50_ms"],20);assert_eq!(result["duration_p95_ms"],100);assert_eq!(result["statuses"]["failed"],1);
        assert_eq!(result["reported_cost_by_currency"]["USD"],0.2);assert_eq!(result["reported_cost_by_currency"]["EUR"],0.3);
        assert!(ModelTotals::default().receipt("external".into(),None,"m".into())["duration_p50_ms"].is_null());
    }
    #[test]
    fn time_series_assigns_trace_start_and_preserves_empty_intervals_and_usage() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        for (started,status) in [(10,"completed"),(19,"failed"),(30,"completed")] {
            let trace=store.begin("session","project").unwrap();trace.record.lock().unwrap().started_ms=started;
            let mut root=trace.span("agent","root",None);let mut model=trace.span("model","model",Some(0));model.finish("completed",&json!({"input_tokens":7,"first_text_ms":0}));root.finish(status,&json!({}));
        }
        let query=TimeSearch{status:None,project_id:Some("project".into()),session_id:None,since_ms:10,until_ms:30,bucket_ms:10};
        let result=store.time_series(&query).unwrap();assert_eq!(result["bucket_count"],3);let rows=result["buckets"].as_array().unwrap();
        assert_eq!(rows[0]["trace_count"],2);assert_eq!(rows[0]["trace_statuses"]["failed"],1);assert_eq!(rows[0]["model_usage"]["input_tokens"],14);assert_eq!(rows[0]["model_usage"]["first_text"]["known_calls"],2);
        assert_eq!(rows[1]["trace_count"],0);assert!(rows[1]["model_usage"]["first_text"]["p50_ms"].is_null());assert_eq!(rows[2]["trace_count"],1);assert_eq!(rows[2]["end_exclusive_ms"],31);
        let failed=store.time_series(&TimeSearch{status:Some("failed".into()),project_id:query.project_id.clone(),session_id:None,since_ms:10,until_ms:30,bucket_ms:10}).unwrap();
        assert_eq!(failed["status"],"failed");assert_eq!(failed["buckets"][0]["trace_count"],1);assert_eq!(failed["buckets"][0]["model_usage"]["input_tokens"],7);assert_eq!(failed["buckets"][2]["trace_count"],0);
        assert!(store.time_series(&TimeSearch{status:Some("INVALID".into()),project_id:None,session_id:None,since_ms:10,until_ms:30,bucket_ms:10}).is_err());
        assert!(store.time_series(&TimeSearch{bucket_ms:0,..query}).is_err());
        assert!(store.time_series(&TimeSearch{status:None,project_id:None,session_id:None,since_ms:0,until_ms:500,bucket_ms:1}).is_err());
        assert_eq!(store.time_series(&TimeSearch{status:None,project_id:Some("other".into()),session_id:None,since_ms:10,until_ms:30,bucket_ms:10}).unwrap()["buckets"][0]["trace_count"],0);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn trace_status_and_time_filters_are_inclusive_and_precede_pagination() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        for (started,status) in [(10,"failed"),(20,"completed"),(30,"failed")] {
            let trace=store.begin("session","project").unwrap();trace.record.lock().unwrap().started_ms=started;
            let mut root=trace.span("agent","root",None);root.finish(status,&json!({}));
        }
        let result=store.list(&Search{status:Some("failed".into()),since_ms:Some(10),until_ms:Some(30),limit:Some(1),offset:Some(1),..Default::default()}).unwrap();
        assert_eq!(result["total"],2);assert_eq!(result["traces"][0]["started_ms"],10);
        assert_eq!(store.list(&Search{since_ms:Some(20),until_ms:Some(20),..Default::default()}).unwrap()["total"],1);
        assert_eq!(store.list(&Search{status:Some("failed".into()),project_id:Some("other".into()),..Default::default()}).unwrap()["total"],0);
        assert!(store.list(&Search{since_ms:Some(30),until_ms:Some(10),..Default::default()}).is_err());
        assert!(store.list(&Search{status:Some("PRIVATE STATUS".into()),..Default::default()}).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn first_text_quantiles_preserve_zero_unknowns_and_reject_inconsistent_values() {
        let mut totals=ModelTotals::default();
        for value in [json!(0),json!(10),json!(100),json!(201),json!(null),json!(-1)] {
            let span:Span=serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_model","name":"model","status":"completed","started_ms":0,"duration_ms":200,"usage":{"first_text_ms":value}})).unwrap();totals.observe(&span).unwrap();
        }
        let receipt=totals.receipt("external".into(),None,"model".into());let first=&receipt["first_text"];
        assert_eq!(first["known_calls"],3);assert_eq!(first["unknown_calls"],3);assert_eq!(first["min_ms"],0);assert_eq!(first["max_ms"],100);assert_eq!(first["p50_ms"],10);assert_eq!(first["p95_ms"],100);
        let empty=first_text_receipt(vec![],2);assert!(empty["p50_ms"].is_null());assert!(empty["min_ms"].is_null());assert_eq!(empty["unknown_calls"],2);
    }
    #[test]
    fn first_text_metadata_is_integer_bounded_and_optional() {
        assert_eq!(public_usage(&json!({"first_text_ms":0}))["first_text_ms"],0);
        assert_eq!(public_usage(&json!({"first_text_ms":17}))["first_text_ms"],17);
        for value in [json!(-1),json!(true),json!(1.5),json!("private")] {assert!(public_usage(&json!({"first_text_ms":value}))["first_text_ms"].is_null());assert!(serde_json::from_value::<ExternalUsage>(json!({"first_text_ms":value})).is_err());}
        assert!(public_usage(&json!({}))["first_text_ms"].is_null());
    }
    #[test]
    fn guardrail_policy_and_outcome_filters_require_the_same_span() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();
        let store=Store::new(&data).unwrap();let trace=store.begin("session","project").unwrap();
        let a="a".repeat(64);let b="b".repeat(64);
        let mut observed=trace.span("tool",&format!("guardrail.input.observe.fail.{a}"),None);observed.finish("completed",&json!({}));
        let mut blocked=trace.span("tool",&format!("guardrail.output.block.fail.{b}"),None);blocked.finish("failed",&json!({}));
        assert_eq!(store.list(&Search{guardrail:Some(GuardrailFilter::Blocked),guardrail_policy_sha256:Some(a.clone()),..Default::default()}).unwrap()["total"],0);
        assert_eq!(store.list(&Search{guardrail:Some(GuardrailFilter::Blocked),guardrail_policy_sha256:Some(b),..Default::default()}).unwrap()["total"],1);
        assert_eq!(store.list(&Search{guardrail_policy_sha256:Some(a),..Default::default()}).unwrap()["total"],1);
        for hash in ["bad".into(),"A".repeat(64),"a".repeat(65)] {assert!(store.list(&Search{guardrail_policy_sha256:Some(hash),..Default::default()}).is_err());}
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn guardrail_policy_groups_are_bounded_complete_and_order_independent() {
        let make=|index:usize|->Span { serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_tool","name":format!("guardrail.input.block.fail.{index:064x}"),"status":"failed","started_ms":0,"duration_ms":0,"usage":{}})).unwrap() };
        let mut ascending=GuardrailSummary::default();let mut descending=GuardrailSummary::default();
        for _ in 0..2 {
            for index in 0..105 { ascending.observe(&make(index)); }
            for index in (0..105).rev() { descending.observe(&make(index)); }
        }
        let result=ascending.receipt();assert_eq!(result,descending.receipt());
        assert_eq!(result["checks"],210);assert_eq!(result["blocked"],210);assert_eq!(result["policy_groups_truncated"],true);
        let groups=result["policies"].as_array().unwrap();assert_eq!(groups.len(),100);
        assert!(groups.iter().all(|row|row["checks"]==2 && row["blocked"]==2));
        assert_eq!(groups[0]["policy_sha256"],format!("{:064x}",0));assert_eq!(groups[99]["policy_sha256"],format!("{:064x}",99));
    }
    #[test]
    fn guardrail_summary_separates_reported_outcomes_and_incomplete_checks() {
        let mut totals=GuardrailTotals::default();
        for (stage,action,outcome,status) in [("input","block","pass","completed"),("output","observe","fail","completed"),("output","block","fail","failed"),("input","block","fail","running"),("input","block","pass","failed")] {
            let span:Span=serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_tool","name":format!("guardrail.{stage}.{action}.{outcome}.{}","a".repeat(64)),"status":status,"started_ms":0,"duration_ms":0,"usage":{}})).unwrap();
            totals.observe(&span);
        }
        let receipt=totals.receipt();assert_eq!(receipt["checks"],5);assert_eq!(receipt["passed"],1);assert_eq!(receipt["violations"],2);assert_eq!(receipt["blocked"],1);assert_eq!(receipt["incomplete_or_inconsistent"],2);assert_eq!(receipt["input_checks"],3);assert_eq!(receipt["output_checks"],2);
    }
    #[test]
    fn guardrail_filters_validate_fingerprints_and_filter_before_pagination() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();
        let store=Store::new(&data).unwrap();
        for (name,status) in [(format!("guardrail.input.block.fail.{}","a".repeat(64)),"failed"),(format!("guardrail.output.observe.fail.{}","b".repeat(64)),"completed"),("guardrail.input.block.fail.invalid".into(),"failed")] {
            let trace=store.begin("session","project").unwrap();let mut span=trace.span("tool",&name,None);span.finish(status,&json!({}));
        }
        assert_eq!(store.summary(&SummarySearch::default()).unwrap()["guardrails"]["checks"],2);
        assert_eq!(store.summary(&SummarySearch{project_id:Some("other".into()),..Default::default()}).unwrap()["guardrails"]["checks"],0);
        for (filter,count) in [(GuardrailFilter::Any,2),(GuardrailFilter::Failed,2),(GuardrailFilter::Blocked,1)] {
            let page=store.list(&Search{guardrail:Some(filter),limit:Some(1),..Default::default()}).unwrap();
            assert_eq!(page["total"],count);assert_eq!(page["traces"].as_array().unwrap().len(),1);
        }
        assert_eq!(store.list(&Search{guardrail:Some(GuardrailFilter::Any),project_id:Some("other".into()),..Default::default()}).unwrap()["total"],0);
        assert!(serde_json::from_value::<Search>(json!({"guardrail":"unknown"})).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn removal_preserves_bytes_filters_totals_and_reopens_after_restart() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();
        let store=Store::new(&data).unwrap();let trace=store.begin("session","project").unwrap();let mut root=trace.span("model","m",None);
        assert!(store.change_visibility(&trace.id(),true).is_err());root.finish("completed",&json!({"input_tokens":7}));
        let path=store.root.join(format!("{}.json",trace.id()));let original=std::fs::read(&path).unwrap();
        assert_eq!(store.change_visibility(&trace.id(),true).unwrap()["changed"],true);
        assert_eq!(store.change_visibility(&trace.id(),true).unwrap()["changed"],false);
        assert_eq!(store.summary(&SummarySearch::default()).unwrap()["trace_count"],0);
        assert_eq!(store.list(&Search{removed:Some(true),..Default::default()}).unwrap()["traces"].as_array().unwrap().len(),1);
        let reopened=Store::new(&data).unwrap();assert!(reopened.validate_target(&trace.id(),None).is_err());
        assert_eq!(std::fs::read(&path).unwrap(),original);
        reopened.change_visibility(&trace.id(),false).unwrap();reopened.validate_target(&trace.id(),Some(0)).unwrap();
        assert_eq!(reopened.summary(&SummarySearch::default()).unwrap()["input_tokens"],7);
        assert_eq!(std::fs::read(&path).unwrap(),original);std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn keyed_external_retries_are_atomic_project_scoped_and_conflict_checked() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();
        let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let request=json!({"project_id":"p","correlation_id":"client","idempotency_key":"receipt-key","started_ms":100,
            "spans":[{"parent_id":null,"kind":"agent","name":"pipeline","status":"completed","started_ms":100,"duration_ms":20}]});
        let threads:Vec<_>=(0..8).map(|_|{let store=store.clone();let request=request.clone();std::thread::spawn(move||store.ingest(serde_json::from_value(request).unwrap()).unwrap())}).collect();
        let receipts:Vec<_>=threads.into_iter().map(|thread|thread.join().unwrap()).collect();
        assert!(receipts.iter().all(|receipt|receipt["id"]==receipts[0]["id"]));
        assert_eq!(receipts.iter().filter(|receipt|receipt["deduplicated"]==false).count(),1);
        let mut conflicting=request.clone();conflicting["spans"][0]["duration_ms"]=json!(21);
        assert!(store.ingest(serde_json::from_value(conflicting).unwrap()).unwrap_err().to_string().contains("idempotency conflict"));
        let mut other=request.clone();other["project_id"]=json!("other");
        assert_ne!(store.ingest(serde_json::from_value(other).unwrap()).unwrap()["id"],receipts[0]["id"]);
        assert_eq!(std::fs::read_dir(store.root.as_ref()).unwrap().count(),2);
        let stored=std::fs::read_to_string(store.root.join(format!("{}.json",receipts[0]["id"].as_str().unwrap()))).unwrap();
        assert!(!stored.contains("receipt-key"));
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn external_tree_bounds_and_immutable_metadata_receipts() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let input=json!({"project_id":"p","correlation_id":"client-1","started_ms":100,
            "spans":[{"parent_id":null,"kind":"agent","name":"pipeline","status":"completed","started_ms":100,"duration_ms":50},
            {"parent_id":0,"kind":"model","name":"model-v1","status":"completed","started_ms":110,"duration_ms":20,
             "usage":{"input_tokens":3,"output_tokens":4,"cost":0.1,"cost_currency":"USD"}}]});
        let receipt=store.ingest(serde_json::from_value(input.clone()).unwrap()).unwrap();
        let second=store.ingest(serde_json::from_value(input.clone()).unwrap()).unwrap();assert_ne!(receipt["id"],second["id"]);
        let record=store.read_record(receipt["id"].as_str().unwrap()).unwrap();assert_eq!(record.spans[1].kind,"external_model");
        let summary=store.summary(&SummarySearch{project_id:Some("p".into()),..Default::default()}).unwrap();assert_eq!(summary["external_model_calls"],2);
        for (field,value) in [("parent_id",json!(1)),("started_ms",json!(99)),("duration_ms",json!(u64::MAX)),("status",json!("running")),("name",json!("private prompt text"))] {
            let mut invalid=input.clone();invalid["spans"][1][field]=value;
            assert!(store.ingest(serde_json::from_value(invalid).unwrap()).is_err());
        }
        let mut private=input.clone();private["spans"][1]["usage"]["api_key"]=json!("secret");assert!(serde_json::from_value::<ExternalTrace>(private).is_err());
        let mut future=input.clone();future["spans"][1]["started_ms"]=json!(151);future["spans"][1]["duration_ms"]=Value::Null;
        assert!(store.ingest(serde_json::from_value(future).unwrap()).is_err());
        assert_eq!(std::fs::read_dir(store.root.as_ref()).unwrap().count(),2);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn conversation_summary_separates_projects_and_bounds_display_only() {
        let data=std::env::temp_dir().join(crate::evaluation::new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        for index in 0..101 {
            let trace=store.begin(&format!("session-{index}"),"project").unwrap();
            trace.span("turn","agent",None).finish("completed",&Value::Null);
        }
        let other=store.begin("session-0","other").unwrap();other.span("turn","agent",None).finish("failed",&Value::Null);
        let result=store.summary(&SummarySearch::default()).unwrap();
        assert_eq!(result["conversation_count"],102);assert_eq!(result["conversations"].as_array().unwrap().len(),100);
        assert_eq!(result["conversations_truncated"],true);assert_eq!(result["trace_count"],102);
        let second=store.summary(&SummarySearch{conversation_offset:Some(100),..Default::default()}).unwrap();
        assert_eq!(second["conversations"].as_array().unwrap().len(),2);assert_eq!(second["conversation_has_more"],false);
        assert_eq!(second["trace_count"],result["trace_count"]);
        let ids:std::collections::HashSet<_>=result["conversations"].as_array().unwrap().iter().chain(second["conversations"].as_array().unwrap()).map(|row|(row["project_id"].as_str().unwrap(),row["session_id"].as_str().unwrap())).collect();assert_eq!(ids.len(),102);
        for (offset,limit) in [(10_001,1),(0,0),(0,101)] {assert!(store.summary(&SummarySearch{conversation_offset:Some(offset),conversation_limit:Some(limit),..Default::default()}).is_err());}
        let failed=store.summary(&SummarySearch{status:Some("failed".into()),..Default::default()}).unwrap();
        assert_eq!(failed["trace_count"],1);assert_eq!(failed["conversation_count"],1);assert_eq!(failed["conversations"][0]["project_id"],"other");
        assert_eq!(failed["status"],"failed");
        assert!(store.summary(&SummarySearch{status:Some("INVALID".into()),..Default::default()}).is_err());
        let scoped=store.summary(&SummarySearch{session_id:Some("session-0".into()),..Default::default()}).unwrap();
        assert_eq!(scoped["conversation_count"],2);assert_eq!(scoped["trace_count"],2);
        let project=store.summary(&SummarySearch{project_id:Some("other".into()),..Default::default()}).unwrap();
        assert_eq!(project["conversations"][0]["trace_statuses"]["failed"],1);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn summary_counts_model_leaves_once_and_keeps_unknown_currency_separate(){
        let data=std::env::temp_dir().join(format!("allpaka-summary-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let trace=store.begin("session","project").unwrap();let mut root=trace.span("turn","agent",None);
        let mut a=trace.span("model","a",Some(root.id()));a.finish("completed",&json!({"prompt_tokens":3,"input_tokens":100,"completion_tokens":4,"cost":0.25,"cost_currency":"USD"}));
        let mut b=trace.span("model","b",Some(root.id()));b.finish("completed",&json!({"input_tokens":5,"output_tokens":6,"cost":0.5}));
        let mut c=trace.span("model","c",Some(root.id()));c.finish("failed",&Value::Null);
        root.finish("completed",&json!({"prompt_tokens":999,"cost":999,"cost_currency":"USD"}));
        let result=store.summary(&SummarySearch{project_id:Some("project".into()),session_id:None,since_ms:None,until_ms:None,..Default::default()}).unwrap();
        assert_eq!(result["trace_count"],1);assert_eq!(result["model_calls"],3);assert_eq!(result["input_tokens"],8);assert_eq!(result["output_tokens"],10);assert_eq!(result["input_tokens_unknown_calls"],1);assert_eq!(result["cost_unknown_calls"],1);assert_eq!(result["cost_currency_unknown_calls"],1);assert_eq!(result["reported_cost_by_currency"]["USD"],0.25);
        assert_eq!(result["conversation_count"],1);
        let conversation=&result["conversations"][0];
        assert_eq!(conversation["project_id"],"project");assert_eq!(conversation["session_id"],"session");
        assert_eq!(conversation["trace_count"],1);assert_eq!(conversation["model_usage"]["calls"],3);
        assert_eq!(conversation["model_usage"]["input_tokens"],8);assert_eq!(conversation["model_usage"]["input_tokens_unknown_calls"],1);
        assert_eq!(conversation["model_usage"]["reported_cost_by_currency"]["USD"],0.25);
        assert_eq!(conversation["model_usage"]["cost_currency_unknown_calls"],1);
        let empty=store.summary(&SummarySearch{project_id:Some("absent".into()),session_id:None,since_ms:None,until_ms:None,..Default::default()}).unwrap();assert_eq!(empty["model_calls"],0);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn restart_recovery_keeps_completed_leaves_and_unknown_durations() {
        let data = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let store = Store::new(&data).unwrap();
        let record = Record {
            id: "trace-orphan".into(),
            session_id: "session".into(),
            project_id: "project".into(),
            started_ms: 1,
            status: "running".into(),
            recovered_ms: None,
            spans: vec![
                Span {
                    id: 0,
                    parent_id: None,
                    kind: "turn".into(),
                    name: "agent".into(),
                    status: "running".into(),
                    started_ms: 1,
                    duration_ms: None,
                    usage: Value::Null,
                    linked_trace_id: None,
                provider_id: None,
                },
                Span {
                    id: 1,
                    parent_id: Some(0),
                    kind: "model".into(),
                    name: "model".into(),
                    status: "completed".into(),
                    started_ms: 2,
                    duration_ms: Some(3),
                    usage: json!({"completion_tokens":4}),
                    linked_trace_id: None,
                provider_id: None,
                },
            ],
        };
        atomic_write(&store.root.join("trace-orphan.json"), &record).unwrap();
        assert_eq!(store.recover().unwrap(), 1);
        assert_eq!(store.recover().unwrap(), 0);
        let rows = store.list(&Search::default()).unwrap();
        let recovered = &rows["traces"][0];
        assert_eq!(recovered["status"], "interrupted");
        assert!(recovered["recovered_ms"].as_u64().is_some());
        assert_eq!(recovered["spans"][0]["status"], "interrupted");
        assert!(recovered["spans"][0]["duration_ms"].is_null());
        assert_eq!(recovered["spans"][1]["status"], "completed");
        assert_eq!(recovered["spans"][1]["usage"]["completion_tokens"], 4);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn trace_tree_persists_status_and_whitelisted_usage_and_drop_interrupts() {
        let data =
            std::env::temp_dir().join(format!("allpaka-trace-{}-{}", std::process::id(), now()));
        std::fs::create_dir(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let store = Store::new(&data).unwrap();
        let trace = store.begin("session", "project").unwrap();
        {
            let mut turn = trace.span("turn", "agent", None);
            let mut model = trace.span("model", "mock", Some(turn.id()));
            model.finish("completed",&json!({"prompt_tokens":3,"completion_tokens":4,"cost":0.1,"api_key":"secret","system":"private"}));
            turn.finish("completed", &Value::Null);
        }
        {
            let trace = store.begin("interrupted-session", "project").unwrap();
            let _turn = trace.span("turn", "agent", None);
        }
        let rows = store.list(&Search::default()).unwrap();
        assert_eq!(rows["total"], 2);
        let text = rows.to_string();
        assert!(!text.contains("secret"));
        assert!(!text.contains("private"));
        let rows = store
            .list(&Search {
                session_id: Some("session".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows["traces"][0]["status"], "completed");
        assert_eq!(rows["traces"][0]["spans"][1]["parent_id"], 0);
        assert_eq!(rows["traces"][0]["spans"][1]["usage"]["cost"], 0.1);
        let rows = store
            .list(&Search {
                session_id: Some("interrupted-session".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows["traces"][0]["status"], "interrupted");
        assert_eq!(
            Store::new(&data).unwrap().list(&Search::default()).unwrap()["total"],
            2
        );
        std::fs::remove_dir_all(data).unwrap();
    }
}
