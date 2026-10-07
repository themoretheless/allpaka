//! Immutable prompt versions pinned by evaluation runs.
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
    io::Read,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
};

const MAX_BYTES: usize = 64 * 1024;
#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    writes: Arc<Mutex<()>>,
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Origin {
    pub id: String,
    pub version: u64,
    pub sha256: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TemplateMessage {pub(crate) role:String,pub(crate) content:String}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub version: u64,
    pub sha256: String,
    pub template: String,
    pub system: String,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub messages:Option<Vec<TemplateMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SavePrompt {
    id: Option<String>,
    project_id: String,
    name: String,
    base_version: u64,
    #[serde(default)]
    template: String,
    #[serde(default)]
    messages:Option<Vec<TemplateMessage>>,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default)]
    origin: Option<Origin>,
}
pub(crate) fn default_system() -> String {
    "Answer the supplied evaluation sample.".into()
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
use crate::evaluation::{commit_new, new_id, project_exists};
fn digest(snapshot: &Snapshot) -> Result<String> {
    let fields = (
        &snapshot.id,
        &snapshot.project_id,
        &snapshot.name,
        snapshot.version,
        &snapshot.template,
        &snapshot.system,
    );
    // Keep hashes of pre-lineage snapshots stable.
    let bytes = if let Some(messages)=&snapshot.messages {serde_json::to_vec(&(fields,&snapshot.origin,messages))?}else{match &snapshot.origin {
        Some(origin) => serde_json::to_vec(&(fields, origin))?,
        None => serde_json::to_vec(&fields)?,
    }};
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
fn validate(snapshot: &Snapshot) -> Result<()> {
    if !valid_id(&snapshot.id) || !valid_id(&snapshot.project_id) || snapshot.version == 0 {
        bail!("Invalid prompt identity/version");
    }
    if let Some(origin) = &snapshot.origin {
        if !valid_id(&origin.id)
            || origin.id == snapshot.id
            || origin.version == 0
            || origin.sha256.len() != 64
            || !origin.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("Invalid prompt origin");
        }
    }
    if snapshot.name.trim().is_empty()
        || snapshot.name.len() > 200
        || snapshot.template.len() > 16000
        || (snapshot.messages.is_none()&&!snapshot.template.contains("{{input}}"))
        || snapshot.system.trim().is_empty()
        || snapshot.system.len() > 16000
    {
        bail!("Prompt needs a name, system instruction and a bounded template containing {{{{input}}}}");
    }
    if let Some(messages)=&snapshot.messages {
        if !snapshot.template.is_empty()||messages.is_empty()||messages.len()>15||messages.len()%2==0||messages.iter().enumerate().any(|(i,m)|m.role!=if i%2==0 {"user"}else{"assistant"}||m.content.trim().is_empty()||m.content.len()>16000)||messages.iter().map(|m|m.content.len()).sum::<usize>()>64000||!messages.last().unwrap().content.contains("{{input}}") {bail!("Chat prompt needs alternating User/Assistant messages ending with User containing {{{{input}}}}, at most 15 messages / 64 KiB");}
    }
    Ok(())
}
pub(crate) fn verify(snapshot: &Snapshot) -> Result<()> {
    validate(snapshot)?;
    if snapshot.sha256 != digest(snapshot)? {
        bail!("Prompt snapshot integrity check failed");
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
impl Store {
    pub(crate) fn new(data: &FsPath) -> Result<Self> {
        let root = data.join("evaluation/prompts");
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
            bail!("Invalid prompt ID");
        }
        let path = self.root.join(id);
        if path.exists() {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_dir() {
                bail!("Invalid prompt directory");
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
                    .context("Invalid prompt revision filename")?;
                latest = latest.max(version);
                count += 1;
                if count > 1000 {
                    bail!("Prompt revision limit reached");
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
            bail!("Prompt not found");
        }
        let snapshot: Snapshot = read(&self.directory(id)?.join(format!("{version:020}.json")))?;
        validate(&snapshot)?;
        if snapshot.id != id || snapshot.version != version || snapshot.sha256 != digest(&snapshot)?
        {
            bail!("Prompt snapshot integrity check failed");
        }
        Ok(snapshot)
    }
    fn save(&self, input: SavePrompt) -> Result<Snapshot> {
        let _writer = self.writes.lock().unwrap();
        let id = input.id.unwrap_or_else(new_id);
        let latest = self.latest(&id)?;
        if latest != input.base_version {
            bail!(
                "Prompt version conflict: expected {latest}, received {}",
                input.base_version
            );
        }
        if latest >= 1000 {
            bail!("Prompt revision limit reached");
        }
        let previous = if latest > 0 {
            Some(self.load(&id, Some(latest))?)
        } else {
            None
        };
        if previous
            .as_ref()
            .is_some_and(|p| p.project_id != input.project_id)
        {
            bail!("Prompt cannot move between projects");
        }
        let origin = match previous {
            Some(previous) => {
                if input
                    .origin
                    .as_ref()
                    .is_some_and(|o| Some(o) != previous.origin.as_ref())
                {
                    bail!("Prompt origin cannot change");
                }
                previous.origin
            }
            None => {
                if let Some(origin) = &input.origin {
                    let parent = self.load(&origin.id, Some(origin.version))?;
                    if parent.project_id != input.project_id || parent.sha256 != origin.sha256 {
                        bail!("Prompt origin project/hash mismatch");
                    }
                }
                input.origin
            }
        };
        let mut snapshot = Snapshot {
            id,
            project_id: input.project_id,
            name: input.name,
            version: latest + 1,
            sha256: String::new(),
            template: input.template,
            system: input.system,
            messages:input.messages,
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
    fn history(&self, id: &str, offset: usize, limit: usize) -> Result<Value> {
        if limit == 0 || limit > 100 || offset > 1000 {
            bail!("Invalid prompt history page");
        }
        let _writer = self.writes.lock().unwrap();
        let directory = self.directory(id)?;
        let mut versions = Vec::new();
        for (visited, entry) in std::fs::read_dir(directory)?.enumerate() {
            if visited >= 2000 {
                bail!("Prompt history scan limit reached");
            }
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|s| s == "json") {
                let version = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .context("Invalid prompt revision filename")?;
                if version == 0 || versions.len() >= 1000 {
                    bail!("Invalid prompt revision catalog");
                }
                versions.push(version);
            }
        }
        versions.sort_unstable_by(|a, b| b.cmp(a));
        let total = versions.len();
        if total == 0 {
            bail!("Prompt not found");
        }
        let mut rows = Vec::new();
        for version in versions.into_iter().skip(offset).take(limit) {
            let prompt = self.load(id, Some(version))?;
            rows.push(json!({"id":prompt.id,"name":prompt.name,"version":prompt.version,"sha256":prompt.sha256,"origin":prompt.origin}));
        }
        Ok(json!({"versions":rows,"total":total,"offset":offset,"limit":limit}))
    }
    fn list(&self, project: &str) -> Result<Value> {
        let mut rows = Vec::new();
        let mut visited = 0;
        for entry in std::fs::read_dir(self.root.as_ref())? {
            let entry = entry?;
            visited += 1;
            if visited > 2000 {
                bail!("Prompt catalog scan limit reached");
            }
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().to_string();
            if self.latest(&id)? == 0 {
                continue;
            }
            let snapshot = self.load(&id, None)?;
            if snapshot.project_id == project {
                rows.push(json!({"id":snapshot.id,"name":snapshot.name,"project_id":snapshot.project_id,"version":snapshot.version,"sha256":snapshot.sha256}));
            }
        }
        rows.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_owned());
        Ok(json!({"prompts":rows}))
    }
}
#[derive(Deserialize)]
pub(crate) struct ListQuery {
    project_id: String,
}
pub(crate) async fn list(
    State(app): State<crate::App>,
    Query(query): Query<ListQuery>,
) -> crate::ApiResult<Value> {
    project_exists(&app, &query.project_id)
        .and_then(|_| app.prompts.list(&query.project_id))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn save(
    State(app): State<crate::App>,
    Json(input): Json<SavePrompt>,
) -> crate::ApiResult<Value> {
    project_exists(&app, &input.project_id)
        .and_then(|_| app.prompts.save(input))
        .and_then(|s| Ok(serde_json::to_value(s)?))
        .map(Json)
        .map_err(|e| {
            let code = if e.to_string().starts_with("Prompt version conflict") {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_REQUEST
            };
            crate::error(code, e)
        })
}
pub(crate) async fn snapshot(
    State(app): State<crate::App>,
    Path((id, version)): Path<(String, u64)>,
) -> crate::ApiResult<Value> {
    app.prompts
        .load(&id, Some(version))
        .and_then(|s| Ok(serde_json::to_value(s)?))
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryQuery {
    #[serde(default)]
    offset: usize,
    #[serde(default = "history_limit")]
    limit: usize,
}
fn history_limit() -> usize {
    50
}
pub(crate) async fn history(
    State(app): State<crate::App>,
    Path(id): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> crate::ApiResult<Value> {
    app.prompts
        .history(&id, query.offset, query.limit)
        .map(Json)
        .map_err(|e| crate::error(StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreviewInput {pub(crate) project_id:String,pub(crate) input:String,#[serde(default)]pub(crate) contexts:Vec<String>}
pub(crate) fn preview_receipt(prompt:&Snapshot,input:&PreviewInput)->Result<Value>{
    verify(prompt)?;
    if prompt.project_id!=input.project_id {bail!("Prompt preview project mismatch");}
    if input.input.trim().is_empty()||input.input.len()>64*1024||input.contexts.len()>50||input.contexts.iter().any(|s|s.len()>64*1024){bail!("Prompt preview input exceeds bounds");}
    let rendered=render_messages(prompt,&input.input,&input.contexts)?;
    let mut messages=vec![json!({"role":"system","content":prompt.system})];messages.extend(rendered.into_iter().map(|m|json!({"role":m.role,"content":m.content})));
    Ok(json!({"kind":"prompt_preview","project_id":prompt.project_id,"prompt_id":prompt.id,"prompt_version":prompt.version,"prompt_sha256":prompt.sha256,"messages":messages,"provider_calls":0,"saved":false}))
}
pub(crate) fn render_messages(prompt:&Snapshot,input:&str,contexts:&[String])->Result<Vec<TemplateMessage>>{
    let contexts_len=contexts.iter().map(String::len).sum::<usize>()+contexts.len().saturating_sub(1)*2;
    if contexts_len>1024*1024 {bail!("Prompt preview context exceeds 1 MiB");}
    let contexts=contexts.join("\n\n");let legacy=vec![TemplateMessage{role:"user".into(),content:prompt.template.clone()}];let templates=prompt.messages.as_ref().unwrap_or(&legacy);let mut result=Vec::new();let mut total=0usize;
    for message in templates {
        let mut rendered=String::new();
        for(i,part)in message.content.split("{{input}}").enumerate(){
            if i>0 {if total+rendered.len()+input.len()>1024*1024 {bail!("Rendered prompt exceeds 1 MiB");}rendered.push_str(input);}
            for(j,text)in part.split("{{contexts}}").enumerate(){let extra=text.len()+if j>0{contexts.len()}else{0};if total+rendered.len()+extra>1024*1024 {bail!("Rendered prompt exceeds 1 MiB");}if j>0 {rendered.push_str(&contexts);}rendered.push_str(text);}
        }
        total+=rendered.len();result.push(TemplateMessage{role:message.role.clone(),content:rendered});
    }
    Ok(result)
}

pub(crate) async fn preview(State(app):State<crate::App>,Path((id,version)):Path<(String,u64)>,Json(input):Json<PreviewInput>)->crate::ApiResult<Value>{
    project_exists(&app,&input.project_id).and_then(|_|app.prompts.load(&id,Some(version))).and_then(|prompt|preview_receipt(&prompt,&input)).map(Json).map_err(|e|crate::error(StatusCode::BAD_REQUEST,e))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_templates_bind_roles_content_and_literal_rendering(){
        let mut prompt=Snapshot{id:"prompt".into(),project_id:"default".into(),name:"Chat".into(),version:1,sha256:String::new(),template:String::new(),system:"System".into(),messages:Some(vec![TemplateMessage{role:"user".into(),content:"Example".into()},TemplateMessage{role:"assistant".into(),content:"Example answer".into()},TemplateMessage{role:"user".into(),content:"{{input}} / {{contexts}}".into()}]),origin:None};prompt.sha256=digest(&prompt).unwrap();verify(&prompt).unwrap();
        let receipt=preview_receipt(&prompt,&PreviewInput{project_id:"default".into(),input:"Literal {{contexts}}".into(),contexts:vec!["Context {{input}}".into()]}).unwrap();assert_eq!(receipt["messages"].as_array().unwrap().len(),4);assert_eq!(receipt["messages"][2]["role"],"assistant");assert_eq!(receipt["messages"][3]["content"],"Literal {{contexts}} / Context {{input}}");
        prompt.messages.as_mut().unwrap()[1].content="Changed answer".into();assert!(verify(&prompt).is_err());prompt.sha256=digest(&prompt).unwrap();verify(&prompt).unwrap();prompt.messages.as_mut().unwrap()[1].role="system".into();assert!(validate(&prompt).is_err());prompt.messages.as_mut().unwrap()[1].role="assistant".into();prompt.template="{{input}}".into();assert!(validate(&prompt).is_err());prompt.template.clear();prompt.messages.as_mut().unwrap().pop();assert!(validate(&prompt).is_err());
    }
    #[test]
    fn preview_pins_messages_preserves_literals_and_bounds_expansion(){
        let mut prompt=Snapshot{id:"prompt".into(),project_id:"default".into(),name:"Preview".into(),version:1,sha256:String::new(),template:"Q: {{input}} C: {{contexts}} Q2: {{input}}".into(),system:"System {{input}}".into(),messages:None,origin:None};prompt.sha256=digest(&prompt).unwrap();
        let mut input=PreviewInput{project_id:"default".into(),input:"literal {{contexts}}".into(),contexts:vec!["one".into(),"two {{input}}".into()]};let receipt=preview_receipt(&prompt,&input).unwrap();assert_eq!(receipt["messages"][0]["content"],"System {{input}}");assert_eq!(receipt["messages"][1]["content"],"Q: literal {{contexts}} C: one\n\ntwo {{input}} Q2: literal {{contexts}}");assert_eq!(receipt["prompt_sha256"],prompt.sha256);assert_eq!(receipt["provider_calls"],0);
        input.project_id="foreign".into();assert!(preview_receipt(&prompt,&input).is_err());input.project_id="default".into();input.input="x".repeat(64*1024);prompt.template="{{input}}".repeat(17);prompt.sha256=digest(&prompt).unwrap();assert!(preview_receipt(&prompt,&input).is_err());prompt.sha256="0".repeat(64);assert!(preview_receipt(&prompt,&input).is_err());
    }
    fn input(id: Option<String>, version: u64, text: &str) -> SavePrompt {
        SavePrompt {
            id,
            project_id: "default".into(),
            name: "Examples".into(),
            base_version: version,
            messages:None,
            template: format!("{text} {{{{input}}}}"),
            system: default_system(),
            origin: None,
        }
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
        let page = store.history(&first.id, 0, 1).unwrap();
        assert_eq!(page["total"], 2);
        assert_eq!(page["versions"][0]["version"], 2);
        assert_eq!(
            store.history(&first.id, 1, 1).unwrap()["versions"][0]["sha256"],
            first.sha256
        );
        assert_eq!(
            store.history(&first.id, 2, 1).unwrap()["versions"],
            json!([])
        );
        assert!(store.history(&first.id, 0, 101).is_err());
        assert!(store.history("../escape", 0, 1).is_err());
        assert_eq!(
            store.load(&first.id, Some(1)).unwrap().template,
            "first {{input}}"
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
        assert_eq!(store.list("other").unwrap()["prompts"], json!([]));
        let file = store
            .directory(&first.id)
            .unwrap()
            .join("00000000000000000001.json");
        let mut corrupt = first.clone();
        corrupt.template = "changed externally {{input}}".into();
        std::fs::write(&file, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(store.load(&first.id, Some(1)).is_err());
        assert!(store.load("../escape", None).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn variants_pin_parent_and_keep_ancestry_across_updates() {
        let root = std::env::temp_dir().join(format!("allpaka-prompt-origin-{}", new_id()));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let store = Store::new(&root).unwrap();
        let parent = store.save(input(None, 0, "parent")).unwrap();
        let origin = Origin {
            id: parent.id.clone(),
            version: 1,
            sha256: parent.sha256.clone(),
        };
        let mut child = input(None, 0, "variant");
        child.origin = Some(origin.clone());
        let child = store.save(child).unwrap();
        assert!(child.origin == Some(origin.clone()));
        store
            .save(input(Some(parent.id.clone()), 1, "parent changed"))
            .unwrap();
        let update = store
            .save(input(Some(child.id.clone()), 1, "variant changed"))
            .unwrap();
        assert!(update.origin == Some(origin.clone()));
        let mut bad = input(None, 0, "bad");
        bad.origin = Some(Origin {
            sha256: "0".repeat(64),
            ..origin.clone()
        });
        assert!(store.save(bad).is_err());
        let mut changed = input(Some(child.id.clone()), 2, "changed ancestry");
        changed.origin = Some(Origin {
            version: 2,
            ..origin.clone()
        });
        assert!(store.save(changed).is_err());
        let mut foreign = input(None, 0, "foreign");
        foreign.project_id = "other".into();
        foreign.origin = Some(origin);
        assert!(store.save(foreign).is_err());
        let mut corrupt = update.clone();
        corrupt.origin.as_mut().unwrap().version = 2;
        assert!(verify(&corrupt).is_err());
        assert!(
            Store::new(&root)
                .unwrap()
                .load(&child.id, Some(2))
                .unwrap()
                .origin
                == update.origin
        );
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
