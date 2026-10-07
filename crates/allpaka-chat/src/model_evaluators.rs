//! Immutable model evaluator configuration and selected-source job admission.
use crate::{
    types::{Mode, Settings},
    App,
};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
static WRITES: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Definition {
    settings: Settings,
    rubric: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    kind: String,
    schema_version: u64,
    evaluator_sha256: String,
    evaluator_id: String,
    evaluator_version: u64,
    definition: Definition,
}
fn digest(definition: &Definition) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(definition)?)
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
fn directory(data: &Path) -> Result<std::path::PathBuf> {
    let root = data.join("evaluation/model-evaluators");
    std::fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    if !root.starts_with(data) {
        bail!("Model evaluator storage escapes data");
    }
    Ok(root)
}
fn read(data: &Path, hash: &str) -> Result<Snapshot> {
    if !valid_hash(hash) {
        bail!("Invalid model evaluator hash");
    }
    let path = directory(data)?.join(format!("{hash}.json"));
    if !std::fs::symlink_metadata(&path)?.file_type().is_file() {
        bail!("Invalid model evaluator packet");
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(64001)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 64000 {
        bail!("Model evaluator packet exceeds bound");
    }
    let packet: Snapshot =
        serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)?;
    if packet.kind != "online_model_evaluator"
        || packet.schema_version != 1
        || packet.evaluator_sha256 != hash
        || packet.evaluator_id != format!("model_quality.{hash}")
        || packet.evaluator_version != 1
        || digest(&packet.definition)? != hash
        || packet.definition.rubric.trim().is_empty()
        || packet.definition.rubric.len() > 16000
        || packet.definition.settings.mode != Mode::Chat
        || packet.definition.settings.allow_writes
    {
        bail!("Model evaluator integrity mismatch");
    }
    Ok(packet)
}
pub(crate) async fn save_api(
    axum::extract::State(app): axum::extract::State<App>,
    axum::Json(definition): axum::Json<Definition>,
) -> crate::ApiResult<Value> {
    let outcome = (|| -> Result<_> {
        crate::validate_settings(&definition.settings, &app)?;
        if definition.settings.mode != Mode::Chat
            || definition.settings.allow_writes
            || definition.rubric.trim().is_empty()
            || definition.rubric.len() > 16000
        {
            bail!("Invalid model evaluator settings/rubric");
        }
        let hash = digest(&definition)?;
        let packet = Snapshot {
            kind: "online_model_evaluator".into(),
            schema_version: 1,
            evaluator_id: format!("model_quality.{hash}"),
            evaluator_version: 1,
            evaluator_sha256: hash.clone(),
            definition,
        };
        if serde_json::to_vec(&packet)?.len() > 64000 {
            bail!("Model evaluator exceeds write bound");
        }
        let _lock = WRITES
            .lock()
            .map_err(|_| anyhow::anyhow!("Model evaluator lock unavailable"))?;
        let root = directory(&app.data)?;
        let path = root.join(format!("{hash}.json"));
        if path.try_exists()? {
            return Ok(serde_json::to_value(read(&app.data, &hash)?)?);
        }
        let mut count = 0;
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            if entry.path().extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }
            count += 1;
            if count >= 1000 || !entry.file_type()?.is_file() {
                bail!("Model evaluator capacity reached");
            }
        }
        crate::evaluation::commit_new(&path, &packet)?;
        Ok(serde_json::to_value(packet)?)
    })();
    outcome
        .map(axum::Json)
        .map_err(|e| crate::error(axum::http::StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    project_id: String,
}
pub(crate) async fn read_api(
    axum::extract::State(app): axum::extract::State<App>,
    axum::extract::Path(hash): axum::extract::Path<String>,
    axum::extract::Query(scope): axum::extract::Query<Scope>,
) -> crate::ApiResult<Value> {
    let outcome = (|| -> Result<_> {
        crate::memory::project_exists(&app, &scope.project_id)?;
        let packet = read(&app.data, &hash)?;
        if packet.definition.settings.project_id != scope.project_id {
            bail!("Model evaluator project mismatch");
        }
        Ok(serde_json::to_value(packet)?)
    })();
    outcome.map(axum::Json).map_err(|_| {
        crate::error(
            axum::http::StatusCode::NOT_FOUND,
            "Model evaluator unavailable or invalid",
        )
    })
}
pub(crate) fn validate_rule(data: &Path, rule: &crate::online_evaluation::Rule) -> Result<()> {
    if let Some(hash) = rule.evaluator_id.strip_prefix("model_quality.") {
        let config = read(data, hash)?;
        if rule.evaluator_version != 1 || config.definition.settings.project_id != rule.project_id {
            bail!("Model rule configuration mismatch");
        }
    }
    Ok(())
}
pub(crate) fn pin_capture(
    data: &Path,
    hash: &str,
    source: &crate::online_sources::Source,
) -> Result<()> {
    let selection = crate::online_evaluation::selection_pin(
        data,
        &source.project_id,
        &source.trace_id,
        &source.trace_sha256,
    )?;
    let root = data.join("evaluation/source-admissions");
    std::fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    if !root.starts_with(data) {
        bail!("Source admission storage escapes data");
    }
    let packet = json!({"kind":"model_source_admission","schema_version":1,"source_sha256":hash,"project_id":source.project_id,"trace_id":source.trace_id,"trace_sha256":source.trace_sha256,"selection_sha256":selection});
    let path = root.join(format!("{hash}.json"));
    if path.try_exists()? {
        if read_admission(&path)? != packet {
            bail!("Source admission pin mismatch");
        }
        return Ok(());
    }
    let mut count = 0;
    for entry in std::fs::read_dir(&root)? {
        let entry = entry?;
        if entry.path().extension().and_then(|v| v.to_str()) != Some("json") {
            continue;
        }
        count += 1;
        if count >= 1000 || !entry.file_type()?.is_file() {
            bail!("Source admission capacity reached");
        }
    }
    crate::evaluation::commit_new(&path, &packet)
}
fn read_admission(path: &Path) -> Result<Value> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        bail!("Invalid source admission file");
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(8193)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 8192 {
        bail!("Source admission exceeds bound");
    }
    Ok(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)
}
pub(crate) fn repair_candidates(data: &Path) -> Result<Vec<String>> {
    let root = data.join("evaluation/online-sources");
    if !root.try_exists()? {
        return Ok(Vec::new());
    }
    if !root.canonicalize()?.starts_with(data) {
        bail!("Source scan escapes data");
    }
    let mut ids = Vec::new();
    let mut count = 0;
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        count += 1;
        if count > 1000 || !entry.file_type()?.is_file() {
            bail!("Source repair scan exceeds bounds");
        }
        let hash = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid source filename"))?;
        if !valid_hash(hash) {
            bail!("Invalid source hash");
        }
        if data
            .join("evaluation/source-admissions")
            .join(format!("{hash}.json"))
            .try_exists()?
        {
            ids.push(hash.to_owned());
        }
    }
    ids.sort();
    Ok(ids)
}
pub(crate) fn repair_sources(
    app: &App,
    done: &mut std::collections::HashSet<String>,
    cursor: &mut usize,
) -> Result<()> {
    let ids = repair_candidates(&app.data)?
        .into_iter()
        .filter(|id| !done.contains(id))
        .collect::<Vec<_>>();
    if ids.is_empty() {
        *cursor = 0;
        return Ok(());
    }
    let start = *cursor % ids.len();
    let mut error = None;
    for step in 0..ids.len().min(10) {
        let hash = &ids[(start + step) % ids.len()];
        match admit_source(app, hash) {
            Ok(_) => {
                done.insert(hash.clone());
            }
            Err(e) => {
                if error.is_none() {
                    error = Some(e);
                }
            }
        }
    }
    *cursor = (start + 10) % ids.len();
    if let Some(error) = error {
        return Err(error);
    }
    Ok(())
}
pub(crate) fn admit_source(app: &App, source_sha256: &str) -> Result<Vec<String>> {
    let source = crate::online_sources::retained_source(&app.data, source_sha256)?;
    let path = app
        .data
        .join("evaluation/source-admissions")
        .join(format!("{source_sha256}.json"));
    let admission = read_admission(&path)?;
    let selection = crate::online_evaluation::selection_pin(
        &app.data,
        &source.project_id,
        &source.trace_id,
        &source.trace_sha256,
    )?;
    if admission
        != json!({"kind":"model_source_admission","schema_version":1,"source_sha256":source_sha256,"project_id":source.project_id,"trace_id":source.trace_id,"trace_sha256":source.trace_sha256,"selection_sha256":selection})
    {
        bail!("Source admission evidence mismatch");
    }

    let pins = crate::online_evaluation::model_selections(
        &app.data,
        &source.project_id,
        &source.trace_id,
        &source.trace_sha256,
    )?;
    let mut jobs = Vec::new();
    for pin in pins {
        let config = read(
            &app.data,
            pin["evaluator_sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing evaluator hash"))?,
        )?;
        if config.definition.settings.project_id != source.project_id {
            bail!("Selected evaluator project mismatch");
        }
        jobs.push(crate::quality_jobs::enqueue(
            app,
            crate::quality_jobs::Input {
                source_sha256: source_sha256.into(),
                settings: config.definition.settings,
                rubric: config.definition.rubric,
                rule_pin: Some(pin),
            },
        )?);
    }
    Ok(jobs)
}
pub(crate) fn verify_job(data: &Path, input: &crate::quality_jobs::Input) -> Result<()> {
    if let Some(pin) = &input.rule_pin {
        let source = crate::online_sources::execution_source(
            data,
            &input.source_sha256,
            &input.settings.project_id,
        )?;
        let pins = crate::online_evaluation::model_selections(
            data,
            &source.project_id,
            &source.trace_id,
            &source.trace_sha256,
        )?;
        if !pins.contains(pin) {
            bail!("Quality job rule pin mismatch");
        }
        let config = read(
            data,
            pin["evaluator_sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid evaluator pin"))?,
        )?;
        if serde_json::to_value(&config.definition.settings)?
            != serde_json::to_value(&input.settings)?
            || config.definition.rubric != input.rubric
        {
            bail!("Quality job model configuration mismatch");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_model_config_reopens_and_rules_reject_project_or_version_substitution() {
        let data = std::env::temp_dir().join(format!(
            "allpaka-model-config-{}",
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(&data).unwrap();
        let data = data.canonicalize().unwrap();
        let definition: Definition = serde_json::from_value(
            json!({"settings":{"provider":"local","model":"mock"},"rubric":"Check clarity"}),
        )
        .unwrap();
        let hash = digest(&definition).unwrap();
        let snapshot = Snapshot {
            kind: "online_model_evaluator".into(),
            schema_version: 1,
            evaluator_sha256: hash.clone(),
            evaluator_id: format!("model_quality.{hash}"),
            evaluator_version: 1,
            definition,
        };
        let path = directory(&data).unwrap().join(format!("{hash}.json"));
        crate::evaluation::commit_new(&path, &snapshot).unwrap();
        assert_eq!(
            read(&data, &hash).unwrap().definition.rubric,
            "Check clarity"
        );
        let mut rule = crate::online_evaluation::Rule {
            id: "quality".into(),
            project_id: "default".into(),
            evaluator_id: snapshot.evaluator_id,
            evaluator_version: 1,
            sample_rate: 1.0,
            enabled: true,
        };
        validate_rule(&data, &rule).unwrap();
        rule.project_id = "foreign".into();
        assert!(validate_rule(&data, &rule).is_err());
        rule.project_id = "default".into();
        rule.evaluator_version = 2;
        assert!(validate_rule(&data, &rule).is_err());
        let mut corrupt = serde_json::to_value(read(&data, &hash).unwrap()).unwrap();
        corrupt["definition"]["rubric"] = json!("Substituted criterion");
        std::fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(read(&data, &hash).is_err());
        std::fs::remove_dir_all(data).unwrap();
    }
}
