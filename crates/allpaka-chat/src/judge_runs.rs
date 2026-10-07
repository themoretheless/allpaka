//! Durable immutable event receipts for cancellable native judge batches.
use crate::{App, judge};
use anyhow::{bail,Context,Result};
use axum::{extract::{Path,Query,State},http::StatusCode,Json};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::HashMap,path::{Path as FsPath,PathBuf},sync::{Arc,Mutex}};
use tokio::sync::{watch,Semaphore};
#[derive(Clone)]
pub(crate) struct Manager { root:Arc<PathBuf>,active:Arc<Mutex<HashMap<String,watch::Sender<bool>>>>,capacity:Arc<Semaphore> }
fn read(path:&FsPath)->Result<Value>{
    use std::io::Read;
    let meta=std::fs::symlink_metadata(path)?;if !meta.is_file()||meta.len()>64000{bail!("Invalid judge run event");}
    let mut bytes=Vec::new();std::fs::File::open(path)?.take(64001).read_to_end(&mut bytes)?;
    if bytes.len()>64000{bail!("Oversize judge run event");}
    Ok(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)
}
impl Manager {
    pub(crate) fn new(data:&FsPath)->Result<Self>{
        let root=data.join("evaluation/judge_runs");std::fs::create_dir_all(&root)?;let root=root.canonicalize()?;
        if !root.starts_with(data){bail!("Judge runs escape Studio data");}
        Ok(Self{root:Arc::new(root),active:Default::default(),capacity:Arc::new(Semaphore::new(2))})
    }
    fn directory(&self,id:&str)->Result<PathBuf>{
        if id.is_empty()||id.len()>80||!id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'){bail!("Invalid judge run ID");}
        let directory=self.root.join(id);if directory.exists()&&!std::fs::symlink_metadata(&directory)?.is_dir(){bail!("Invalid judge run directory");}Ok(directory)
    }
    pub(crate) fn recover(&self)->Result<()>{
        for (index,entry) in std::fs::read_dir(self.root.as_ref())?.enumerate(){
            if index>=1000{bail!("Judge recovery scan limit reached");}let entry=entry?;
            if !entry.file_type()?.is_dir(){bail!("Invalid judge run entry");}
            let directory=entry.path();
            if directory.join("initial.json").exists()&&!directory.join("terminal.json").exists(){
                crate::evaluation::commit_new(&directory.join("terminal.json"),&json!({"status":"interrupted","mean_score":null,"reason":"server_restarted","automatic_replay":false}))?;
            }
        }Ok(())
    }
    fn load(&self,id:&str)->Result<Value>{
        let directory=self.directory(id)?;let mut initial=read(&directory.join("initial.json"))?;
        let count=initial["sample_count"].as_u64().filter(|n|*n<=200).context("Invalid judge run sample count")?;
        let mut items=Vec::new();for index in 0..count{let path=directory.join(format!("item-{index:04}.json"));if !path.exists(){break;}items.push(read(&path)?);}
        let terminal=if directory.join("terminal.json").exists(){read(&directory.join("terminal.json"))?}else{json!({"status":if self.active.lock().unwrap().contains_key(id){"running"}else{"interrupted"},"mean_score":null})};
        let mut terminal=terminal;
        if self.active.lock().unwrap().contains_key(id) {terminal["status"]=json!("running");terminal["mean_score"]=Value::Null;}
        initial["items"]=json!(items);initial["status"]=terminal["status"].clone();initial["mean_score"]=terminal["mean_score"].clone();initial["terminal"]=terminal;Ok(initial)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Start { plan_id:String }
#[derive(Deserialize)]
pub(crate) struct Catalog { project_id:String, #[serde(default)] offset:usize, limit:Option<usize> }
pub(crate) async fn list(State(app):State<App>,Query(query):Query<Catalog>)->crate::ApiResult<Value>{
    let limit=query.limit.unwrap_or(20);
    if !(1..=20).contains(&limit)||query.offset>1000{return Err(crate::error(StatusCode::BAD_REQUEST,"Invalid judge catalog page"));}
    let ids=(||->Result<Vec<String>>{
        let mut ids=Vec::new();
        for (index,entry) in std::fs::read_dir(app.judge_runs.root.as_ref())?.enumerate(){
            if index>=1000{bail!("Judge catalog scan limit reached");}
            let entry=entry?;if !entry.file_type()?.is_dir(){bail!("Invalid judge run entry");}
            let id=entry.file_name().into_string().map_err(|_|anyhow::anyhow!("Invalid judge run ID"))?;
            let directory=app.judge_runs.directory(&id)?;
            let initial=read(&directory.join("initial.json"))?;
            if initial["id"]!=id||initial["kind"]!="judge_run"{bail!("Judge run identity mismatch");}
            if initial["project_id"]==query.project_id{ids.push(id);}
        }
        ids.sort_by(|a,b|b.cmp(a));Ok(ids)
    })().map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))?;
    let total=ids.len();let mut runs=Vec::new();
    for id in ids.into_iter().skip(query.offset).take(limit){
        let run=get(State(app.clone()),Path(id)).await?.0;
        runs.push(json!({"id":run["id"],"project_id":run["project_id"],"plan_id":run["plan_id"],"trace_id":run["trace_id"],"status":run["status"],"sample_count":run["sample_count"],"completed":run["items"].as_array().map(|items|items.len()),"mean_score":run["mean_score"]}));
    }
    Ok(Json(json!({"runs":runs,"total":total,"offset":query.offset,"limit":limit})))
}
pub(crate) async fn start(State(app):State<App>,Json(input):Json<Start>)->crate::ApiResult<Value>{
    let plan=judge::read_plan(State(app.clone()),Path(input.plan_id)).await?.0;
    let permit=app.judge_runs.capacity.clone().try_acquire_owned().map_err(|_|crate::error(StatusCode::TOO_MANY_REQUESTS,"Judge batch capacity reached"))?;
    let prepared=(||->Result<_>{
        let settings:crate::types::Settings=serde_json::from_value(plan["settings"].clone())?;crate::validate_settings(&settings,&app)?;
        let samples=plan["samples"].as_array().context("Missing plan samples")?;
        let id=crate::evaluation::new_id();let directory=app.judge_runs.directory(&id)?;std::fs::create_dir(&directory)?;
        let trace=app.observability.begin(&format!("judge-run-{id}"),settings.project_id.as_str())?;
        let root=trace.span("judge_batch","rubric batch",None);
        let initial=json!({"id":id,"kind":"judge_run","trace_id":trace.id(),"plan_id":plan["id"],"plan_sha256":plan["plan_sha256"],"project_id":plan["project_id"],"dataset_id":plan["dataset_id"],"dataset_version":plan["dataset_version"],"dataset_sha256":plan["dataset_sha256"],"sample_count":samples.len()});
        crate::evaluation::commit_new(&directory.join("initial.json"),&initial)?;Ok((id,directory,initial,trace,root))
    })().map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))?;
    let (sender,mut cancellation)=watch::channel(false);app.judge_runs.active.lock().unwrap().insert(prepared.0.clone(),sender);
    let response=json!({"id":prepared.0,"status":"running","plan_id":plan["id"]});
    tokio::spawn(async move{
        let _permit=permit;let mut root=prepared.4;let mut status="completed";let mut total=0.0;let mut completed=0usize;
        for (index,sample) in plan["samples"].as_array().unwrap().iter().enumerate(){
            if *cancellation.borrow(){status="cancelled";break;}
            let mut item_span=prepared.3.span("judge_item",&format!("sample {index}"),Some(root.id()));
            let request=json!({"settings":plan["settings"],"rubric":plan["rubric"],"input":sample["input"],"output":sample["output"],"reference":sample["reference"],"dataset_ref":{"dataset_id":plan["dataset_id"],"dataset_version":plan["dataset_version"],"sample_id":sample["sample_id"]}});
            let result=match serde_json::from_value(request){Ok(request)=>{
                tokio::select!{biased;result=judge::judge(State(app.clone()),Json(request))=>Some(result),_=cancellation.changed()=>None}
            },Err(_)=>{item_span.finish("failed",&Value::Null);status="failed";break;}};
            match result{
                Some(Ok(Json(receipt)))=>{
                    let item=json!({"sample_id":sample["sample_id"],"receipt_id":receipt["id"],"receipt_sha256":receipt["receipt_sha256"],"score":receipt["score"],"trace_id":receipt["trace_id"]});
                    if crate::evaluation::commit_new(&prepared.1.join(format!("item-{index:04}.json")),&item).is_err(){item_span.finish("failed",&Value::Null);status="failed";break;}
                    if let Some(trace_id)=receipt["trace_id"].as_str(){item_span.link_trace(trace_id);}
                    item_span.finish("completed",&Value::Null);
                    total+=receipt["score"].as_f64().unwrap();completed+=1;
                },None=>{item_span.finish("interrupted",&Value::Null);status="cancelled";break;},Some(Err(_))=>{item_span.finish("failed",&Value::Null);status="failed";break;}
            }
        }
        let terminal=json!({"status":status,"completed":completed,"mean_score":if status=="completed"&&completed>0{Some(total/completed as f64)}else{None},"automatic_replay":false});
        let saved=crate::evaluation::commit_new(&prepared.1.join("terminal.json"),&terminal);
        root.finish(if saved.is_err(){"failed"}else if status=="cancelled"{"interrupted"}else{status},&Value::Null);
        app.judge_runs.active.lock().unwrap().remove(&prepared.0);
    });Ok(Json(response))
}
pub(crate) async fn get(State(app):State<App>,Path(id):Path<String>)->crate::ApiResult<Value>{
    let run=app.judge_runs.load(&id).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))?;
    let plan_id=run["plan_id"].as_str().ok_or_else(||crate::error(StatusCode::BAD_REQUEST,"Missing judge plan"))?;
    let plan=judge::read_plan(State(app.clone()),Path(plan_id.into())).await?.0;
    if run["id"]!=id || run["plan_sha256"]!=plan["plan_sha256"] || run["sample_count"]!=plan["samples"].as_array().map(|s|s.len()).unwrap_or(0){return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run source mismatch"));}
    for field in ["project_id","dataset_id","dataset_version","dataset_sha256"]{
        if run[field]!=plan[field]{return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run dataset mismatch"));}
    }
    let mut total=0.0;
    for (index,item) in run["items"].as_array().unwrap().iter().enumerate(){
        let receipt_id=item["receipt_id"].as_str().ok_or_else(||crate::error(StatusCode::BAD_REQUEST,"Missing judge receipt"))?;
        let receipt=judge::read(State(app.clone()),Path(receipt_id.into())).await?.0;
        if receipt["settings"]!=plan["settings"]||receipt["project_id"]!=plan["project_id"]||receipt["input"]!=plan["samples"][index]["input"]||receipt["reference"]!=plan["samples"][index]["reference"]{return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run evaluator or source mismatch"));}
        let source=json!({"dataset_id":plan["dataset_id"],"dataset_version":plan["dataset_version"],"dataset_sha256":plan["dataset_sha256"],"sample_id":plan["samples"][index]["sample_id"]});
        if receipt["dataset_ref"]!=source||item["trace_id"]!=receipt["trace_id"]{return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run receipt source mismatch"));}
        if item["sample_id"]!=plan["samples"][index]["sample_id"] || item["receipt_sha256"]!=receipt["receipt_sha256"] || item["score"]!=receipt["score"] || receipt["output"]!=plan["samples"][index]["output"] || receipt["rubric"]!=plan["rubric"]{return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run score mismatch"));}
        total+=receipt["score"].as_f64().unwrap();
    }
    if run["status"]=="completed"&&(run["items"].as_array().unwrap().len()!=plan["samples"].as_array().unwrap().len()||run["mean_score"]!=json!(total/plan["samples"].as_array().unwrap().len() as f64)){return Err(crate::error(StatusCode::BAD_REQUEST,"Judge run completion mismatch"));}
    Ok(Json(run))
}
pub(crate) async fn cancel(State(app):State<App>,Path(id):Path<String>)->crate::ApiResult<Value>{
    app.judge_runs.load(&id).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))?;
    if let Some(sender)=app.judge_runs.active.lock().unwrap().get(&id){let _=sender.send(true);}
    Ok(Json(json!({"id":id,"cancel_requested":true})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Compare { baseline_id:String,candidate_id:String }
pub(crate) async fn compare(State(app):State<App>,Json(input):Json<Compare>)->crate::ApiResult<Value>{
    let baseline=get(State(app.clone()),Path(input.baseline_id)).await?.0;
    let candidate=get(State(app.clone()),Path(input.candidate_id)).await?.0;
    let left=judge::read_plan(State(app.clone()),Path(baseline["plan_id"].as_str().unwrap().into())).await?.0;
    let right=judge::read_plan(State(app.clone()),Path(candidate["plan_id"].as_str().unwrap().into())).await?.0;
    let result=(||->Result<Value>{
        if baseline["status"]!="completed"||candidate["status"]!="completed"{bail!("Judge comparisons require completed batches");}
        for field in ["project_id","dataset_id","dataset_version","dataset_sha256","rubric","rubric_snapshot","judge_preset","settings"]{
            if left[field]!=right[field]{bail!("Judge comparison requires identical dataset, rubric and judge settings");}
        }
        let a=baseline["items"].as_array().context("Missing baseline scores")?;
        let b=candidate["items"].as_array().context("Missing candidate scores")?;
        if a.is_empty()||a.len()!=b.len(){bail!("Judge comparison coverage mismatch");}
        let mut improvements=0;let mut regressions=0;let mut pairs=Vec::new();
        for (a,b) in a.iter().zip(b){
            if a["sample_id"]!=b["sample_id"]{bail!("Judge comparison sample mismatch");}
            let before=a["score"].as_f64().context("Missing baseline score")?;
            let after=b["score"].as_f64().context("Missing candidate score")?;
            if after>before{improvements+=1;}if after<before{regressions+=1;}
            pairs.push(json!({"sample_id":a["sample_id"],"baseline":before,"candidate":after,"delta":after-before,"baseline_receipt_id":a["receipt_id"],"candidate_receipt_id":b["receipt_id"]}));
        }
        let id=crate::evaluation::new_id();
        let receipt=json!({"id":id,"kind":"judge_comparison","baseline_id":baseline["id"],"candidate_id":candidate["id"],"baseline_plan_sha256":left["plan_sha256"],"candidate_plan_sha256":right["plan_sha256"],"project_id":left["project_id"],"dataset_id":left["dataset_id"],"dataset_version":left["dataset_version"],"dataset_sha256":left["dataset_sha256"],"eligible":regressions==0&&improvements>0,"improvements":improvements,"regressions":regressions,"pairs":pairs,"reason":if regressions>0{"paired_regression"}else if improvements==0{"no_improvement"}else{"strict_paired_improvement"},"observational_only":true,"provider_calls":0,"automatic_promotion":false});
        let directory=app.data.join("evaluation/judge_comparisons");std::fs::create_dir_all(&directory)?;
        if !directory.canonicalize()?.starts_with(app.data.as_ref()){bail!("Judge comparison storage escapes Studio data");}
        crate::evaluation::commit_new(&directory.join(format!("{id}.json")),&receipt)?;Ok(receipt)
    })();result.map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
