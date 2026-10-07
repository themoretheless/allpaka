//! Durable explicit model quality jobs. Claimed requests are never automatically replayed.
use crate::{
    types::{Mode, Settings},
    App,
};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
static WRITES: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn hash(value: &Value) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
fn valid_hash(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn read(path: &Path) -> Result<Value> {
    read_bound(path, 64000)
}
fn read_bound(path: &Path, limit: usize) -> Result<Value> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        bail!("Invalid quality job packet");
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("Quality job packet exceeds bound");
    }
    Ok(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Input {
    pub(crate) source_sha256: String,
    pub(crate) settings: Settings,
    pub(crate) rubric: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rule_pin: Option<Value>,
}
fn root(data: &Path) -> Result<PathBuf> {
    let directory = data.join("evaluation/quality-jobs");
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    if !directory.starts_with(data) {
        bail!("Quality jobs escape data");
    }
    Ok(directory)
}
fn directory(data: &Path, id: &str) -> Result<PathBuf> {
    if !valid_hash(id) {
        bail!("Invalid quality job ID");
    }
    let root = root(data)?;
    let directory = root.join(id);
    if directory.exists()
        && (!std::fs::symlink_metadata(&directory)?.file_type().is_dir()
            || !directory.canonicalize()?.starts_with(&root))
    {
        bail!("Invalid quality job directory");
    }
    Ok(directory)
}
fn plan(data: &Path, id: &str) -> Result<(Value, Input)> {
    let packet = read(&directory(data, id)?.join("plan.json"))?;
    if packet.as_object().map(|o| o.len()) != Some(4)
        || packet["kind"] != "online_quality_job"
        || packet["schema_version"] != 1
        || packet["id"] != id
        || hash(&packet["request"])? != id
    {
        bail!("Quality job plan integrity mismatch");
    }
    let input: Input = serde_json::from_value(packet["request"].clone())?;
    if !valid_hash(&input.source_sha256)
        || input.rubric.trim().is_empty()
        || input.rubric.len() > 16000
        || input.settings.mode != Mode::Chat
        || input.settings.allow_writes
    {
        bail!("Invalid quality job plan");
    }
    Ok((packet, input))
}
fn result(data: &Path, id: &str) -> Result<Option<Value>> {
    let directory = directory(data, id)?;
    let path = directory.join("result.json");
    if !path.try_exists()? {
        return Ok(None);
    }
    let packet = read(&path)?;
    let mut payload = packet.clone();
    let digest = payload
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Invalid result"))?
        .remove("result_sha256")
        .ok_or_else(|| anyhow::anyhow!("Missing result hash"))?;
    if payload.as_object().map(|o| o.len()) != Some(7)
        || digest != hash(&payload)?
        || packet["kind"] != "online_quality_job_result"
        || packet["schema_version"] != 1
        || packet["id"] != id
        || !matches!(
            packet["status"].as_str(),
            Some("completed" | "failed" | "interrupted")
        )
        || packet["automatic_execution"] != true
    {
        bail!("Quality job result integrity mismatch");
    }
    let claim = read(&directory.join("claim.json"))?;
    if claim != json!({"kind":"online_quality_job_claim","id":id,"schema_version":1}) {
        bail!("Invalid quality job claim");
    }
    if packet["status"] == "completed" {
        if !packet["judge_id"].as_str().is_some_and(|v| {
            !v.is_empty()
                && v.len() <= 80
                && v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        }) || !packet["judge_receipt_sha256"]
            .as_str()
            .is_some_and(valid_hash)
        {
            bail!("Missing quality verdict evidence");
        }
        let judge_id = packet["judge_id"].as_str().unwrap();
        let path = data
            .join("evaluation/judges")
            .join(format!("{judge_id}.json"));
        if !path.canonicalize()?.starts_with(data) {
            bail!("Quality verdict escapes data");
        }
        let verdict = read_bound(&path, 1024 * 1024)?;
        let mut payload = verdict.clone();
        let digest = payload
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("Invalid linked verdict"))?
            .remove("receipt_sha256")
            .ok_or_else(|| anyhow::anyhow!("Missing linked verdict hash"))?;
        let (_, request) = plan(data, id)?;
        if digest != hash(&payload)?
            || verdict["receipt_sha256"] != packet["judge_receipt_sha256"]
            || verdict["kind"] != "llm_judge"
            || verdict["id"] != judge_id
            || verdict["project_id"] != request.settings.project_id
            || verdict["quality_source_sha256"] != request.source_sha256
            || verdict["rubric"] != request.rubric
        {
            bail!("Linked quality verdict mismatch");
        }
    } else if !packet["judge_id"].is_null() || !packet["judge_receipt_sha256"].is_null() {
        bail!("Invalid failed quality evidence");
    }
    Ok(Some(packet))
}
fn finish(data: &Path, id: &str, status: &str, verdict: Option<&Value>) -> Result<()> {
    let mut packet = json!({"kind":"online_quality_job_result","schema_version":1,"id":id,"status":status,"judge_id":verdict.map(|v|v["id"].clone()),"judge_receipt_sha256":verdict.map(|v|v["receipt_sha256"].clone()),"automatic_execution":true});
    packet["result_sha256"] = json!(hash(&packet)?);
    crate::evaluation::commit_new(&directory(data, id)?.join("result.json"), &packet)
}
fn ids(data: &Path) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(root(data)?)? {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !entry.file_type()?.is_dir() || !valid_hash(&id) {
            bail!("Invalid quality job inventory");
        }
        ids.push(id);
        if ids.len() > 1000 {
            bail!("Quality job scan exceeds limit");
        }
    }
    ids.sort();
    Ok(ids)
}
pub(crate) fn recover(data: &Path) -> Result<()> {
    for id in ids(data)? {
        plan(data, &id)?;
        if result(data, &id)?.is_none() && directory(data, &id)?.join("claim.json").try_exists()? {
            let claim = read(&directory(data, &id)?.join("claim.json"))?;
            if claim != json!({"kind":"online_quality_job_claim","id":id,"schema_version":1}) {
                bail!("Invalid recovered claim");
            }
            finish(data, &id, "interrupted", None)?;
        }
    }
    Ok(())
}
fn state(data: &Path, id: &str) -> Result<(String, Option<Value>)> {
    let terminal = result(data, id)?;
    if let Some(packet) = &terminal {
        return Ok((packet["status"].as_str().unwrap().into(), terminal));
    }
    let claim = directory(data, id)?.join("claim.json");
    if claim.try_exists()? {
        if read(&claim)? != json!({"kind":"online_quality_job_claim","id":id,"schema_version":1}) {
            bail!("Invalid pending quality claim");
        }
        Ok(("running".into(), None))
    } else {
        Ok(("pending".into(), None))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Page {
    project_id: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "page_limit")]
    limit: usize,
}
fn page_limit() -> usize {
    20
}
fn catalog(data: &Path, project: &str, offset: usize, limit: usize) -> Result<Value> {
    if offset > 1000 || !(1..=100).contains(&limit) {
        bail!("Invalid quality job page");
    }
    let mut rows = Vec::new();
    for id in ids(data)? {
        let (_, request) = plan(data, &id)?;
        let (status, result) = state(data, &id)?;
        if request.settings.project_id != project {
            continue;
        }
        rows.push(json!({"id":id,"project_id":project,"source_sha256":request.source_sha256,"provider":request.settings.provider,"model":request.settings.model,"status":status,"result":result}));
    }
    let total = rows.len();
    let jobs = rows
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();
    Ok(
        json!({"kind":"online_quality_job_catalog","project_id":project,"offset":offset,"limit":limit,"total":total,"has_more":offset+jobs.len()<total,"order":"id_ascending","jobs":jobs,"provider_calls":0,"automatic_execution":false}),
    )
}
pub(crate) async fn list_api(
    axum::extract::State(app): axum::extract::State<App>,
    axum::extract::Query(page): axum::extract::Query<Page>,
) -> crate::ApiResult<Value> {
    let outcome = (|| -> Result<_> {
        crate::memory::project_exists(&app, &page.project_id)?;
        catalog(&app.data, &page.project_id, page.offset, page.limit)
    })();
    outcome
        .map(axum::Json)
        .map_err(|e| crate::error(axum::http::StatusCode::BAD_REQUEST, e))
}
pub(crate) async fn get_api(
    axum::extract::State(app): axum::extract::State<App>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> crate::ApiResult<Value> {
    let outcome = (|| -> Result<_> {
        let (packet, _) = plan(&app.data, &id)?;
        let (status, result) = state(&app.data, &id)?;
        Ok(
            json!({"kind":"online_quality_job_status","id":id,"status":status,"plan":packet,"result":result}),
        )
    })();
    outcome.map(axum::Json).map_err(|_| {
        crate::error(
            axum::http::StatusCode::NOT_FOUND,
            "Quality job unavailable or invalid",
        )
    })
}
pub(crate) async fn submit_api(
    axum::extract::State(app): axum::extract::State<App>,
    axum::Json(input): axum::Json<Input>,
) -> crate::ApiResult<Value> {
    let outcome =
        enqueue(&app, input).map_err(|e| crate::error(axum::http::StatusCode::BAD_REQUEST, e))?;
    get_api(axum::extract::State(app), axum::extract::Path(outcome)).await
}
pub(crate) fn enqueue(app: &App, input: Input) -> Result<String> {
    crate::validate_settings(&input.settings, app)?;
    crate::model_evaluators::verify_job(&app.data, &input)?;
    if input.settings.mode != Mode::Chat
        || input.settings.allow_writes
        || input.rubric.trim().is_empty()
        || input.rubric.len() > 16000
    {
        bail!("Quality jobs require bounded rubric and Chat without writes");
    }
    crate::online_sources::execution_source(
        &app.data,
        &input.source_sha256,
        &input.settings.project_id,
    )?;
    let request = serde_json::to_value(input)?;
    if serde_json::to_vec(&request)?.len() > 60000 {
        bail!("Quality job request exceeds bound");
    }
    let id = hash(&request)?;
    let _lock = WRITES
        .lock()
        .map_err(|_| anyhow::anyhow!("Quality job lock unavailable"))?;
    let directory = directory(&app.data, &id)?;
    if directory.join("plan.json").try_exists()? {
        let (existing, _) = plan(&app.data, &id)?;
        if existing["request"] != request {
            bail!("Quality job identity conflict");
        }
        return Ok(id);
    }
    if ids(&app.data)?.len() >= 1000 {
        bail!("Quality job capacity reached");
    }
    std::fs::create_dir(&directory)?;
    let packet = json!({"kind":"online_quality_job","schema_version":1,"id":id,"request":request});
    if let Err(error) = crate::evaluation::commit_new(&directory.join("plan.json"), &packet) {
        let _ = std::fs::remove_dir(&directory);
        return Err(error);
    }
    Ok(id)
}
pub(crate) struct Worker(tokio::task::JoinHandle<()>);
impl Drop for Worker {
    fn drop(&mut self) {
        self.0.abort();
    }
}
fn claim_one(data: &Path) -> Result<Option<(String, Input)>> {
    let _lock = WRITES
        .lock()
        .map_err(|_| anyhow::anyhow!("Quality job lock unavailable"))?;
    for id in ids(data)? {
        let (_, input) = plan(data, &id)?;
        if result(data, &id)?.is_some() {
            continue;
        }
        let claim = directory(data, &id)?.join("claim.json");
        if claim.exists() {
            continue;
        }
        crate::evaluation::commit_new(
            &claim,
            &json!({"kind":"online_quality_job_claim","id":id,"schema_version":1}),
        )?;
        return Ok(Some((id, input)));
    }
    Ok(None)
}
pub(crate) fn spawn(app: App) -> Worker {
    Worker(tokio::spawn(async move {
        let mut last_error = None;
        let mut admission_error = None;
        let mut done = std::collections::HashSet::new();
        let mut cursor = 0;
        loop {
            let error = crate::model_evaluators::repair_sources(&app, &mut done, &mut cursor)
                .err()
                .map(|e| e.to_string());
            if error != admission_error {
                if let Some(message) = &error {
                    eprintln!("Model source recovery failed: {message}");
                }
                admission_error = error;
            }

            let outcome = async {
                if let Some((id, input)) = claim_one(&app.data)? {
                    if crate::model_evaluators::verify_job(app.data.as_ref(), &input).is_err() {
                        finish(&app.data, &id, "failed", None)?;
                    } else {
                        let request = crate::judge::SourceJudge {
                            settings: input.settings,
                            rubric: input.rubric,
                        };
                        let verdict = crate::judge::judge_source(
                            axum::extract::State(app.clone()),
                            axum::extract::Path(input.source_sha256),
                            axum::Json(request),
                        )
                        .await;
                        match verdict {
                            Ok(verdict) => finish(&app.data, &id, "completed", Some(&verdict.0))?,
                            Err(_) => finish(&app.data, &id, "failed", None)?,
                        }
                    }
                }
                Ok::<(), anyhow::Error>(())
            }
            .await;
            let error = outcome.err().map(|e| e.to_string());
            if error != last_error {
                if let Some(message) = &error {
                    eprintln!("Model quality worker failed: {message}");
                }
                last_error = error;
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn claims_are_once_only_and_restart_retains_interruption_without_replay() {
        let data = std::env::temp_dir().join(format!(
            "allpaka-quality-jobs-{}",
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let input:Input=serde_json::from_value(json!({"source_sha256":"a".repeat(64),"settings":{"provider":"local","model":"mock"},"rubric":"Quality"})).unwrap();
        let request = serde_json::to_value(input).unwrap();
        let id = hash(&request).unwrap();
        let directory = directory(&data, &id).unwrap();
        std::fs::create_dir(&directory).unwrap();
        crate::evaluation::commit_new(
            &directory.join("plan.json"),
            &json!({"kind":"online_quality_job","schema_version":1,"id":id,"request":request}),
        )
        .unwrap();
        recover(&data).unwrap();
        assert_eq!(
            catalog(&data, "default", 0, 1).unwrap()["jobs"][0]["status"],
            "pending"
        );
        assert_eq!(catalog(&data, "foreign", 0, 1).unwrap()["total"], 0);
        assert!(catalog(&data, "default", 0, 0).is_err());
        assert!(result(&data, &id).unwrap().is_none());
        assert_eq!(claim_one(&data).unwrap().unwrap().0, id);
        assert!(claim_one(&data).unwrap().is_none());
        recover(&data).unwrap();
        let interrupted = result(&data, &id).unwrap().unwrap();
        assert_eq!(interrupted["status"], "interrupted");
        assert_eq!(
            catalog(&data, "default", 0, 1).unwrap()["jobs"][0]["result"],
            interrupted
        );
        assert!(claim_one(&data).unwrap().is_none());
        let bytes = std::fs::read(directory.join("result.json")).unwrap();
        recover(&data).unwrap();
        assert_eq!(std::fs::read(directory.join("result.json")).unwrap(), bytes);
        let mut corrupt = interrupted;
        corrupt["status"] = json!("completed");
        std::fs::write(
            directory.join("result.json"),
            serde_json::to_vec(&corrupt).unwrap(),
        )
        .unwrap();
        assert!(result(&data, &id).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
}
