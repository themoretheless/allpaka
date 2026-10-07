//! Versioned evaluation datasets. Revisions are immutable and content-addressed.
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    io::{Read, Write},
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_BYTES: usize = 8 * 1024 * 1024;
#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    writes: Arc<Mutex<()>>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sample {
    pub id: String,
    pub input: String,
    #[serde(default)]
    pub expected_output: Option<String>,
    #[serde(default)]
    pub contexts: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Origin { pub id:String, pub version:u64, pub sha256:String }
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub version: u64,
    pub sha256: String,
    pub samples: Vec<Sample>,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub origin: Option<Origin>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveDataset {
    id: Option<String>,
    project_id: String,
    name: String,
    base_version: u64,
    samples: Vec<Sample>,
    #[serde(default)]
    origin: Option<Origin>,
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
pub(crate) fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "{:x}-{:x}-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
fn digest(snapshot: &Snapshot) -> Result<String> {
    let fields = (
        &snapshot.id,
        &snapshot.project_id,
        &snapshot.name,
        snapshot.version,
        &snapshot.samples,
    );
    let bytes=match &snapshot.origin {Some(origin)=>serde_json::to_vec(&(fields,origin))?,None=>serde_json::to_vec(&fields)?};
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
fn validate(snapshot: &Snapshot) -> Result<()> {
    if !valid_id(&snapshot.id) || !valid_id(&snapshot.project_id) || snapshot.version == 0 {
        bail!("Invalid dataset identity/version");
    }
    if let Some(origin)=&snapshot.origin {
        if !valid_id(&origin.id)||origin.id==snapshot.id||origin.version==0||origin.sha256.len()!=64||!origin.sha256.bytes().all(|b|b.is_ascii_hexdigit()){bail!("Invalid dataset origin");}
    }
    if snapshot.name.trim().is_empty()
        || snapshot.name.len() > 200
        || snapshot.samples.is_empty()
        || snapshot.samples.len() > 2000
    {
        bail!("Dataset needs a name and 1–2000 samples");
    }
    let mut ids = HashSet::new();
    for sample in &snapshot.samples {
        if !valid_id(&sample.id) || !ids.insert(&sample.id) {
            bail!("Sample IDs must be valid and unique");
        }
        if sample.input.trim().is_empty()
            || sample.input.len() > 64 * 1024
            || sample
                .expected_output
                .as_ref()
                .is_some_and(|s| s.len() > 64 * 1024)
            || sample.contexts.len() > 50
            || sample.contexts.iter().any(|s| s.len() > 64 * 1024)
            || serde_json::to_vec(&sample.metadata)?.len() > 64 * 1024
        {
            bail!("Sample content exceeds its bounds");
        }
    }
    if serde_json::to_vec(snapshot)?.len() > MAX_BYTES {
        bail!("Dataset exceeds 8 MiB");
    }
    Ok(())
}
fn read<T: serde::de::DeserializeOwned>(path: &FsPath) -> Result<T> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES as u64 {
        bail!("Invalid evaluation file");
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_BYTES {
        bail!("Oversize evaluation file");
    }
    Ok(serde_json::from_slice(&bytes)?)
}
pub(crate) fn commit_new(path: &FsPath, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_BYTES {
        bail!("Evaluation file exceeds 8 MiB");
    }
    let temp = path.with_extension(format!("{}.tmp", new_id()));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        // Unlike rename, linking cannot overwrite an already committed revision.
        std::fs::hard_link(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(temp);
    result
}
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Lifecycle { revision:u64, archived:bool }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LifecycleInput { project_id:String, base_version:u64, base_revision:u64, archived:bool }
impl Store {
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("evaluation/datasets");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if !root.starts_with(data) {
            bail!("Evaluation storage escapes Studio data directory");
        }
        Ok(Self {
            root: Arc::new(root),
            writes: Default::default(),
        })
    }
    fn directory(&self, id: &str) -> Result<PathBuf> {
        if !valid_id(id) {
            bail!("Invalid dataset ID");
        }
        let path = self.root.join(id);
        if path.exists() {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_dir() {
                bail!("Invalid dataset directory");
            }
        }
        Ok(path)
    }
    fn latest(&self, id: &str) -> Result<u64> {
        let directory = self.directory(id)?;
        if !directory.exists() {
            return Ok(0);
        }
        let mut latest = 0;
        let mut count = 0;
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().is_some_and(|s| s == "json") {
                let version = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .context("Invalid dataset revision filename")?;
                latest = latest.max(version);
                count += 1;
                if count > 1000 {
                    bail!("Dataset revision limit reached");
                }
            }
        }
        Ok(latest)
    }
    pub(crate) fn load(&self, id: &str, version: Option<u64>) -> Result<Snapshot> {
        let version = match version {
            Some(version) => version,
            None => self.latest(id)?,
        };
        if version == 0 {
            bail!("Dataset not found");
        }
        let snapshot: Snapshot = read(&self.directory(id)?.join(format!("{version:020}.json")))?;
        validate(&snapshot)?;
        if snapshot.id != id || snapshot.version != version || snapshot.sha256 != digest(&snapshot)?
        {
            bail!("Dataset snapshot integrity check failed");
        }
        Ok(snapshot)
    }
    fn versions(&self,id:&str,project:&str,offset:usize,limit:usize)->Result<Value>{
        if offset>1000||!(1..=100).contains(&limit){bail!("Invalid dataset version pagination");}
        let latest=self.latest(id)?;if latest==0||latest>1000{bail!("Invalid dataset version catalog");}
        let directory=self.directory(id)?;let mut bytes=std::fs::symlink_metadata(directory.join(format!("{latest:020}.json")))?.len();
        let current=self.load(id,Some(latest))?;if current.project_id!=project{bail!("Dataset belongs to another project");}
        let mut rows=Vec::new();
        for version in (1..=latest).rev().skip(offset).take(limit){
            bytes=bytes.checked_add(std::fs::symlink_metadata(directory.join(format!("{version:020}.json")))?.len()).context("Version scan overflow")?;
            if bytes>64*1024*1024{bail!("Version catalog exceeds 64 MiB; reduce page size");}
            let snapshot=self.load(id,Some(version))?;if snapshot.project_id!=project{bail!("Dataset version project mismatch");}
            rows.push(json!({"version":version,"name":snapshot.name,"sha256":snapshot.sha256,"samples":snapshot.samples.len()}));
        }
        let lifecycle=self.lifecycle(id)?;
        Ok(json!({"id":id,"project_id":project,"latest_version":latest,"archived":lifecycle.archived,"lifecycle_revision":lifecycle.revision,"versions":rows,"total":latest,"offset":offset,"limit":limit,"has_more":offset+rows.len()<latest as usize,"order":"version_desc","provider_calls":0}))
    }
    fn compare(&self,id:&str,query:&CompareQuery)->Result<Value>{
        let offset=query.offset.unwrap_or(0);let limit=query.limit.unwrap_or(100);
        if !(1..=1000).contains(&query.from_version)||!(1..=1000).contains(&query.to_version)||offset>4000||!(1..=100).contains(&limit){bail!("Invalid dataset comparison bounds");}
        let before=self.load(id,Some(query.from_version))?;let after=self.load(id,Some(query.to_version))?;
        if before.project_id!=query.project_id||after.project_id!=query.project_id{bail!("Dataset comparison project mismatch");}
        let old:BTreeMap<_,_>=before.samples.iter().map(|s|(s.id.as_str(),s)).collect();let new:BTreeMap<_,_>=after.samples.iter().map(|s|(s.id.as_str(),s)).collect();
        let ids:std::collections::BTreeSet<_>=old.keys().chain(new.keys()).copied().collect();
        let mut changes=Vec::new();let(mut added,mut removed,mut changed,mut unchanged)=(0,0,0,0);
        for sample_id in ids {
            let(kind,fields)=match (old.get(sample_id),new.get(sample_id)){
                (None,Some(_))=>{added+=1;("added",Vec::new())},
                (Some(_),None)=>{removed+=1;("removed",Vec::new())},
                (Some(a),Some(b))=>{let mut fields=Vec::new();if a.input!=b.input{fields.push("input");}if a.expected_output!=b.expected_output{fields.push("expected_output");}if a.contexts!=b.contexts{fields.push("contexts");}if a.metadata!=b.metadata{fields.push("metadata");}if fields.is_empty(){unchanged+=1;continue;}changed+=1;("changed",fields)},
                _=>unreachable!(),
            };
            changes.push(json!({"sample_id":sample_id,"kind":kind,"fields":fields}));
        }
        let total=changes.len();let changes:Vec<_>=changes.into_iter().skip(offset).take(limit).collect();
        Ok(json!({"id":id,"project_id":query.project_id,"from":{"version":before.version,"sha256":before.sha256,"name":before.name,"samples":before.samples.len()},"to":{"version":after.version,"sha256":after.sha256,"name":after.name,"samples":after.samples.len()},"name_changed":before.name!=after.name,"counts":{"added":added,"removed":removed,"changed":changed,"unchanged":unchanged},"changes":changes,"total":total,"offset":offset,"limit":limit,"has_more":offset+changes.len()<total,"order":"sample_id_asc","provider_calls":0}))
    }
    fn lifecycle(&self,id:&str)->Result<Lifecycle>{
        let path=self.directory(id)?.join("_state");
        if path.exists(){read(&path)}else{Ok(Lifecycle::default())}
    }
    fn change_lifecycle(&self,id:&str,input:LifecycleInput)->Result<Value>{
        let _writer=self.writes.lock().unwrap();
        let snapshot=self.load(id,None)?;
        if snapshot.project_id!=input.project_id {bail!("Dataset belongs to another project");}
        let state=self.lifecycle(id)?;
        if snapshot.version!=input.base_version || state.revision!=input.base_revision {bail!("Dataset lifecycle conflict; reload current metadata");}
        let state=Lifecycle{revision:state.revision.checked_add(1).context("Dataset lifecycle revision exhausted")?,archived:input.archived};
        let directory=self.directory(id)?;let path=directory.join("_state");let temp=directory.join(format!("{}.tmp",new_id()));
        let result=(||->Result<()> {
            let mut options=std::fs::OpenOptions::new();options.write(true).create_new(true);
            #[cfg(unix)] {use std::os::unix::fs::OpenOptionsExt;options.mode(0o600);}
            let mut file=options.open(&temp)?;file.write_all(&serde_json::to_vec(&state)?)?;file.sync_all()?;
            std::fs::rename(&temp,&path)?;
            #[cfg(unix)] std::fs::File::open(&directory)?.sync_all()?;
            Ok(())
        })();let _=std::fs::remove_file(&temp);result?;
        Ok(json!({"id":id,"project_id":snapshot.project_id,"version":snapshot.version,"sha256":snapshot.sha256,"lifecycle_revision":state.revision,"archived":state.archived,"snapshots_retained":true}))
    }
    pub(crate) fn draft_single(project:&str,name:&str,sample:Sample)->Result<Snapshot>{
        let mut snapshot=Snapshot{id:new_id(),project_id:project.into(),name:name.into(),version:1,sha256:String::new(),samples:vec![sample],origin:None};validate(&snapshot)?;snapshot.sha256=digest(&snapshot)?;Ok(snapshot)
    }
    pub(crate) fn commit_single(&self,snapshot:Snapshot)->Result<Snapshot>{
        if snapshot.version!=1||snapshot.samples.len()!=1||snapshot.origin.is_some(){bail!("Invalid single-input draft");}
        self.save(SaveDataset{id:Some(snapshot.id),project_id:snapshot.project_id,name:snapshot.name,base_version:0,samples:snapshot.samples,origin:None})
    }
    fn save(&self, input: SaveDataset) -> Result<Snapshot> {
        let _writer = self.writes.lock().unwrap();
        let id = input.id.unwrap_or_else(new_id);
        let latest = self.latest(&id)?;
        if self.lifecycle(&id)?.archived {bail!("Archived dataset cannot be edited; restore it first");}
        if latest != input.base_version {
            bail!(
                "Dataset version conflict: expected {latest}, received {}",
                input.base_version
            );
        }
        if latest >= 1000 {
            bail!("Dataset revision limit reached");
        }
        let previous=if latest>0 {Some(self.load(&id,Some(latest))?)}else{None};
        if previous.as_ref().is_some_and(|p|p.project_id!=input.project_id){bail!("Dataset cannot move between projects");}
        let origin=match previous {
            Some(previous)=>{if input.origin.as_ref().is_some_and(|o|Some(o)!=previous.origin.as_ref()){bail!("Dataset origin cannot change");}previous.origin},
            None=>{if let Some(origin)=&input.origin {let parent=self.load(&origin.id,Some(origin.version))?;if parent.project_id!=input.project_id||parent.sha256!=origin.sha256 {bail!("Dataset origin project/hash mismatch");}}input.origin}
        };
        let mut snapshot = Snapshot {
            id,
            project_id: input.project_id,
            name: input.name,
            version: latest + 1,
            sha256: String::new(),
            samples: input.samples,
            origin,
        };
        validate(&snapshot)?;
        snapshot.sha256 = digest(&snapshot)?;
        let directory = self.directory(&snapshot.id)?;
        std::fs::create_dir_all(&directory)?;
        commit_new(
            &directory.join(format!("{:020}.json", snapshot.version)),
            &snapshot,
        )?;
        Ok(snapshot)
    }
    fn list(&self, project: &str) -> Result<Value> { self.list_state(project,false) }
    fn list_state(&self, project: &str, archived:bool) -> Result<Value> { self.list_page(project,archived,None,None) }
    fn list_page(&self,project:&str,archived:bool,offset:Option<usize>,limit:Option<usize>)->Result<Value>{self.list_query(project,archived,offset,limit,None)}
    fn list_query(&self,project:&str,archived:bool,offset:Option<usize>,limit:Option<usize>,query:Option<&str>)->Result<Value>{
        if query.is_some_and(|query|query.chars().count()>200||query.len()>800){bail!("Dataset search exceeds 200 characters");}
        let query=query.unwrap_or("").trim().to_lowercase();
        let paged=offset.is_some()||limit.is_some();let offset=offset.unwrap_or(0);let limit=limit.unwrap_or(if paged {100}else{2000});
        if offset>2000||limit==0||(paged&&limit>100){bail!("Invalid dataset catalog pagination");}
        let mut rows = Vec::new();
        let mut visited = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            visited += 1;
            if visited > 2000 {
                bail!("Dataset catalog scan limit reached");
            }
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().to_string();
            if self.latest(&id)? == 0 {
                continue;
            }
            let snapshot = self.load(&id, None)?;
            let lifecycle=self.lifecycle(&id)?;
            if snapshot.project_id == project && lifecycle.archived==archived {
                rows.push(json!({"id":snapshot.id,"name":snapshot.name,"project_id":snapshot.project_id,"version":snapshot.version,"sha256":snapshot.sha256,"samples":snapshot.samples.len(),"archived":lifecycle.archived,"lifecycle_revision":lifecycle.revision}));
            }
        }
        rows.retain(|row|query.is_empty()||row["name"].as_str().unwrap_or("").to_lowercase().contains(&query)||row["id"].as_str().unwrap_or("").to_lowercase().contains(&query));
        rows.sort_by(|a,b|a["name"].as_str().cmp(&b["name"].as_str()).then_with(||a["id"].as_str().cmp(&b["id"].as_str())));
        let total=rows.len();let rows:Vec<_>=rows.into_iter().skip(offset).take(limit).collect();
        Ok(json!({"datasets":rows,"total":total,"offset":offset,"limit":limit,"has_more":offset+rows.len()<total}))
    }
}
#[derive(Deserialize)]
pub(crate) struct ListQuery {
    q:Option<String>,
    offset:Option<usize>,
    limit:Option<usize>,
    #[serde(default)]
    archived:bool,
    project_id: String,
}
pub(crate) fn project_exists(app: &crate::App, id: &str) -> Result<()> {
    if !app.projects.lock().unwrap().iter().any(|p| p.id == id) {
        bail!("Project not found");
    }
    Ok(())
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<ListQuery>,
) -> crate::ApiResult<Value> {
    project_exists(&app, &query.project_id)
        .and_then(|_| app.evaluation.list_query(&query.project_id,query.archived,query.offset,query.limit,query.q.as_deref()))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn lifecycle(State(app):State<crate::App>,Path(id):Path<String>,Json(input):Json<LifecycleInput>)->crate::ApiResult<Value>{
    project_exists(&app,&input.project_id).and_then(|_|app.evaluation.change_lifecycle(&id,input)).map(Json).map_err(|e|crate::error(if e.to_string().starts_with("Dataset lifecycle conflict"){StatusCode::CONFLICT}else{StatusCode::BAD_REQUEST},e))
}
pub(crate) async fn save(
    State(app): State<crate::App>,
    Json(input): Json<SaveDataset>,
) -> crate::ApiResult<Value> {
    project_exists(&app, &input.project_id)
        .and_then(|_| app.evaluation.save(input))
        .and_then(|s| Ok(serde_json::to_value(s)?))
        .map(Json)
        .map_err(|e| {
            let code = if e.to_string().starts_with("Dataset version conflict") {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_REQUEST
            };
            crate::error(code, e)
        })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VersionsQuery { project_id:String,offset:Option<usize>,limit:Option<usize> }
pub(crate) async fn versions(State(app):State<crate::App>,Path(id):Path<String>,Query(query):Query<VersionsQuery>)->crate::ApiResult<Value>{
    project_exists(&app,&query.project_id).and_then(|_|app.evaluation.versions(&id,&query.project_id,query.offset.unwrap_or(0),query.limit.unwrap_or(20))).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompareQuery {project_id:String,from_version:u64,to_version:u64,offset:Option<usize>,limit:Option<usize>}
pub(crate) async fn compare(State(app):State<crate::App>,Path(id):Path<String>,Query(query):Query<CompareQuery>)->crate::ApiResult<Value>{
    project_exists(&app,&query.project_id).and_then(|_|app.evaluation.compare(&id,&query)).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn snapshot(
    State(app): State<crate::App>,
    Path((id, version)): Path<(String, u64)>,
) -> crate::ApiResult<Value> {
    app.evaluation
        .load(&id, Some(version))
        .and_then(|s| Ok(serde_json::to_value(s)?))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn input(id: Option<String>, version: u64, text: &str) -> SaveDataset {
        SaveDataset {
            id,
            project_id: "default".into(),
            name: "Examples".into(),
            base_version: version,
            origin: None,
            samples: vec![Sample {
                id: "a".into(),
                input: text.into(),
                expected_output: Some("expected".into()),
                contexts: vec!["evidence".into()],
                metadata: BTreeMap::new(),
            }],
        }
    }
    #[test]
    fn version_comparison_tracks_all_sample_fields_and_direction(){
        let root=std::env::temp_dir().join(new_id());std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::new(&root).unwrap();
        let mut first=input(None,0,"before");let unchanged=Sample{id:"same".into(),..first.samples[0].clone()};let removed=Sample{id:"removed".into(),..unchanged.clone()};first.samples.extend([unchanged,removed]);let first=store.save(first).unwrap();
        let mut second=input(Some(first.id.clone()),1,"after");second.name="Renamed".into();second.samples[0].expected_output=None;second.samples[0].contexts.clear();second.samples[0].metadata.insert("tag".into(),json!(1));second.samples.push(first.samples[1].clone());second.samples.push(Sample{id:"added".into(),..second.samples[0].clone()});let second=store.save(second).unwrap();
        let mut query=CompareQuery{project_id:"default".into(),from_version:1,to_version:2,offset:Some(0),limit:Some(2)};let diff=store.compare(&first.id,&query).unwrap();assert_eq!(diff["counts"],json!({"added":1,"removed":1,"changed":1,"unchanged":1}));assert_eq!(diff["changes"][0]["fields"],json!(["input","expected_output","contexts","metadata"]));assert_eq!(diff["from"]["sha256"],first.sha256);assert_eq!(diff["to"]["sha256"],second.sha256);assert_eq!(diff["has_more"],true);assert_eq!(diff["name_changed"],true);
        query.offset=Some(2);assert_eq!(store.compare(&first.id,&query).unwrap()["changes"][0]["sample_id"],"removed");query.from_version=2;query.to_version=1;query.offset=Some(0);assert_eq!(store.compare(&first.id,&query).unwrap()["changes"][1]["kind"],"removed");
        query.to_version=2;assert_eq!(store.compare(&first.id,&query).unwrap()["total"],0);query.project_id="foreign".into();assert!(store.compare(&first.id,&query).is_err());query.project_id="default".into();query.limit=Some(101);assert!(store.compare(&first.id,&query).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn dataset_variants_pin_immutable_ancestry(){
        let root=std::env::temp_dir().join(new_id());std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::new(&root).unwrap();
        let source=store.save(input(None,0,"original")).unwrap();let source_bytes=std::fs::read(store.directory(&source.id).unwrap().join("00000000000000000001.json")).unwrap();
        let origin=Origin{id:source.id.clone(),version:1,sha256:source.sha256.clone()};
        let mut draft=input(None,0,"variant");draft.origin=Some(origin.clone());let variant=store.save(draft).unwrap();assert_ne!(variant.id,source.id);assert!(variant.origin.as_ref()==Some(&origin));
        store.save(input(Some(source.id.clone()),1,"source changed")).unwrap();
        let next=store.save(input(Some(variant.id.clone()),1,"variant changed")).unwrap();assert!(next.origin.as_ref()==Some(&origin));assert_eq!(store.load(&source.id,Some(1)).unwrap().sha256,source.sha256);
        let mut bad=input(None,0,"bad hash");bad.origin=Some(Origin{sha256:"0".repeat(64),..origin.clone()});assert!(store.save(bad).is_err());
        let mut bad=input(Some(variant.id.clone()),2,"changed origin");bad.origin=Some(Origin{version:2,..origin.clone()});assert!(store.save(bad).is_err());
        let mut bad=input(None,0,"foreign");bad.project_id="other".into();bad.origin=Some(origin);assert!(store.save(bad).is_err());
        assert_eq!(std::fs::read(store.directory(&source.id).unwrap().join("00000000000000000001.json")).unwrap(),source_bytes);
        let path=store.directory(&variant.id).unwrap().join("00000000000000000001.json");let mut tampered:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();tampered["origin"]["version"]=json!(2);std::fs::write(path,serde_json::to_vec(&tampered).unwrap()).unwrap();assert!(store.load(&variant.id,Some(1)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn version_catalog_verifies_pins_and_pages_newest_first(){
        let data=std::env::temp_dir().join(new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let first=store.save(input(None,0,"original")).unwrap();let second=store.save(input(Some(first.id.clone()),1,"changed")).unwrap();
        let page=store.versions(&first.id,"default",0,1).unwrap();assert_eq!(page["versions"][0]["sha256"],second.sha256);assert_eq!(page["has_more"],true);
        assert_eq!(store.versions(&first.id,"default",1,1).unwrap()["versions"][0]["sha256"],first.sha256);
        assert!(store.versions(&first.id,"other",0,20).is_err());assert!(store.versions(&first.id,"default",0,101).is_err());
        let path=store.directory(&first.id).unwrap().join("00000000000000000001.json");let mut altered=first;altered.name="tampered".into();std::fs::write(path,serde_json::to_vec(&altered).unwrap()).unwrap();assert!(store.versions(&second.id,"default",0,20).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn dataset_catalog_pages_cover_equal_names_without_duplicates(){
        let data=std::env::temp_dir().join(new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        for _ in 0..25 {store.save(input(None,0,"sample")).unwrap();}
        let first=store.list_page("default",false,Some(0),Some(20)).unwrap();let second=store.list_page("default",false,Some(20),Some(20)).unwrap();
        assert_eq!(first["total"],25);assert_eq!(second["datasets"].as_array().unwrap().len(),5);assert_eq!(second["has_more"],false);
        assert_eq!(store.list_query("default",false,None,None,Some("EXAMPLES")).unwrap()["total"],25);
        let id=first["datasets"][0]["id"].as_str().unwrap();assert_eq!(store.list_query("default",false,None,None,Some(id)).unwrap()["total"],1);
        assert_eq!(store.list_query("default",false,None,None,Some("missing")).unwrap()["total"],0);
        assert!(store.list_query("default",false,None,None,Some(&"x".repeat(201))).is_err());
        let ids:HashSet<_>=first["datasets"].as_array().unwrap().iter().chain(second["datasets"].as_array().unwrap()).map(|row|row["id"].as_str().unwrap()).collect();assert_eq!(ids.len(),25);
        assert!(store.list_page("default",false,None,Some(101)).is_err());assert_eq!(store.list("default").unwrap()["datasets"].as_array().unwrap().len(),25);
        let mut unicode=input(None,0,"sample");unicode.name="Набор Проверки".into();store.save(unicode).unwrap();assert_eq!(store.list_query("default",false,None,None,Some("ПРОВЕРКИ")).unwrap()["total"],1);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn archive_retains_snapshots_and_rejects_stale_changes(){
        let data=std::env::temp_dir().join(new_id());std::fs::create_dir_all(&data).unwrap();let data=data.canonicalize().unwrap();let store=Store::new(&data).unwrap();
        let snapshot=store.save(input(None,0,"original")).unwrap();
        let change=|revision,archived|LifecycleInput{project_id:"default".into(),base_version:1,base_revision:revision,archived};
        store.change_lifecycle(&snapshot.id,change(0,true)).unwrap();
        assert!(store.list("default").unwrap()["datasets"].as_array().unwrap().is_empty());
        assert_eq!(store.list_state("default",true).unwrap()["datasets"][0]["lifecycle_revision"],1);
        assert_eq!(store.load(&snapshot.id,Some(1)).unwrap().sha256,snapshot.sha256);
        assert!(store.save(input(Some(snapshot.id.clone()),1,"modified")).is_err());
        assert!(store.change_lifecycle(&snapshot.id,change(0,false)).is_err());
        let recovered=Store::new(&data).unwrap();recovered.change_lifecycle(&snapshot.id,change(1,false)).unwrap();
        assert_eq!(recovered.list("default").unwrap()["datasets"][0]["sha256"],snapshot.sha256);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn immutable_revisions_conflicts_integrity_and_restart() {
        let root = std::env::temp_dir().join(format!("allpaka-datasets-{}", new_id()));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let store = Store::new(&root).unwrap();
        let first = store.save(input(None, 0, "first")).unwrap();
        let second = store
            .save(input(Some(first.id.clone()), 1, "second"))
            .unwrap();
        assert_ne!(first.sha256, second.sha256);
        assert_eq!(
            store.load(&first.id, Some(1)).unwrap().samples[0].input,
            "first"
        );
        assert!(store
            .save(input(Some(first.id.clone()), 1, "stale"))
            .is_err());
        assert_eq!(
            Store::new(&root)
                .unwrap()
                .load(&first.id, None)
                .unwrap()
                .version,
            2
        );
        assert_eq!(store.list("other").unwrap()["datasets"], json!([]));
        let file = store
            .directory(&first.id)
            .unwrap()
            .join("00000000000000000001.json");
        let mut corrupt = first.clone();
        corrupt.samples[0].input = "changed externally".into();
        std::fs::write(&file, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(store.load(&first.id, Some(1)).is_err());
        assert!(store.load("../escape", None).is_err());
        let mut duplicates = input(None, 0, "text");
        duplicates.samples.push(duplicates.samples[0].clone());
        assert!(store.save(duplicates).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn concurrent_updates_never_overwrite_a_revision() {
        let root = std::env::temp_dir().join(format!("allpaka-dataset-race-{}", new_id()));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let store = Store::new(&root).unwrap();
        let first = store.save(input(None, 0, "first")).unwrap();
        let successes = std::thread::scope(|scope| {
            let tasks: Vec<_> = (0..8)
                .map(|i| {
                    let store = &store;
                    let id = first.id.clone();
                    scope.spawn(move || {
                        store
                            .save(input(Some(id), 1, &format!("update {i}")))
                            .is_ok()
                    })
                })
                .collect();
            tasks
                .into_iter()
                .map(|t| usize::from(t.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(successes, 1);
        assert_eq!(store.load(&first.id, None).unwrap().version, 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
