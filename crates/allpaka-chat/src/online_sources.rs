//! Explicit caller-supplied text snapshots for trace-linked quality evaluation.
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
const MAX: usize = 160_000;
static WRITES: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) project_id: String,
    pub(crate) trace_id: String,
    pub(crate) trace_sha256: String,
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) reference: Option<String>,
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn digest(source: &Source) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(source)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
impl Source {
    fn validate(&self) -> Result<()> {
        if !identity(&self.project_id)
            || !identity(&self.trace_id)
            || !sha(&self.trace_sha256)
            || self.input.len() > 16000
            || self.output.trim().is_empty()
            || self.output.len() > 64000
            || self.reference.as_ref().is_some_and(|v| v.len() > 64000)
        {
            bail!("Invalid quality source identity or text bounds");
        }
        Ok(())
    }
    fn verify_trace(&self, data: &Path) -> Result<()> {
        self.validate()?;
        if crate::observability::Store::new(data)?
            .completed_trace_fingerprint(&self.trace_id, &self.project_id)?
            != self.trace_sha256
        {
            bail!("Quality source trace evidence changed");
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    kind: String,
    schema_version: u64,
    source_sha256: String,
    answer_source: String,
    trace_content_verified: bool,
    source: Source,
}
struct Store {
    directory: PathBuf,
}
impl Store {
    fn open(data: &Path) -> Result<Self> {
        let directory = data.join("evaluation/online-sources");
        std::fs::create_dir_all(&directory)?;
        let directory = directory.canonicalize()?;
        if !directory.starts_with(data) {
            bail!("Quality source storage escapes data");
        }
        Ok(Self { directory })
    }
    fn read(&self, hash: &str) -> Result<Snapshot> {
        if !sha(hash) {
            bail!("Invalid quality source hash");
        }
        let path = self.directory.join(format!("{hash}.json"));
        if !std::fs::symlink_metadata(&path)?.file_type().is_file() {
            bail!("Invalid quality source file");
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX {
            bail!("Quality source exceeds read bound");
        }
        let packet: Snapshot =
            serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)?;
        packet.source.validate()?;
        if packet.kind != "online_quality_source"
            || packet.schema_version != 1
            || packet.source_sha256 != hash
            || digest(&packet.source)? != hash
            || packet.answer_source != "caller_supplied"
            || packet.trace_content_verified
        {
            bail!("Quality source integrity mismatch");
        }
        Ok(packet)
    }
    fn save(&self, data: &Path, source: Source) -> Result<Snapshot> {
        source.verify_trace(data)?;
        let hash = digest(&source)?;
        let packet = Snapshot {
            kind: "online_quality_source".into(),
            schema_version: 1,
            source_sha256: hash.clone(),
            answer_source: "caller_supplied".into(),
            trace_content_verified: false,
            source,
        };
        let size = serde_json::to_vec(&packet)?.len();
        if size > MAX {
            bail!("Quality source exceeds write bound");
        }
        let _lock = WRITES
            .lock()
            .map_err(|_| anyhow::anyhow!("Quality source storage lock unavailable"))?;
        let path = self.directory.join(format!("{hash}.json"));
        if path.try_exists()? {
            let stored = self.read(&hash)?;
            crate::model_evaluators::pin_capture(data, &hash, &stored.source)?;
            return Ok(stored);
        }
        let (mut count, mut total) = (0usize, 0usize);
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let metadata = entry.metadata()?;
            if !entry.file_type()?.is_file() || metadata.len() > MAX as u64 {
                bail!("Invalid retained quality source");
            }
            count += 1;
            total = total
                .checked_add(metadata.len() as usize)
                .ok_or_else(|| anyhow::anyhow!("Quality source size overflow"))?;
            if count >= 1000 || total + size > 16 * 1024 * 1024 {
                bail!("Quality source capacity reached");
            }
        }
        packet.source.verify_trace(data)?;
        crate::model_evaluators::pin_capture(data, &hash, &packet.source)?;
        crate::evaluation::commit_new(&path, &packet)?;
        Ok(packet)
    }
}
pub(crate) fn retained_source(data: &Path, hash: &str) -> Result<Source> {
    Ok(Store::open(data)?.read(hash)?.source)
}
pub(crate) fn execution_source(data: &Path, hash: &str, project: &str) -> Result<Source> {
    let packet = Store::open(data)?.read(hash)?;
    if packet.source.project_id != project {
        bail!("Quality source project mismatch");
    }
    packet.source.verify_trace(data)?;
    Ok(packet.source)
}
pub(crate) async fn save_api(
    axum::extract::State(app): axum::extract::State<crate::App>,
    body: axum::body::Bytes,
) -> crate::ApiResult<serde_json::Value> {
    let result = (|| -> Result<_> {
        if body.len() > MAX {
            bail!("Quality source request exceeds bound");
        }
        let source: Source =
            serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&body)?)?)?;
        crate::memory::project_exists(&app, &source.project_id)?;
        Ok(serde_json::to_value(
            Store::open(&app.data)?.save(&app.data, source)?,
        )?)
    })();
    if let Ok(packet) = &result {
        if let Some(hash) = packet["source_sha256"].as_str() {
            if let Err(error) = crate::model_evaluators::admit_source(&app, hash) {
                eprintln!("Model rule source admission failed: {error}");
            }
        }
    }
    result
        .map(axum::Json)
        .map_err(|e| crate::error(axum::http::StatusCode::BAD_REQUEST, e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    project_id: String,
}
pub(crate) async fn read_api(
    axum::extract::State(app): axum::extract::State<crate::App>,
    axum::extract::Path(hash): axum::extract::Path<String>,
    axum::extract::Query(scope): axum::extract::Query<Scope>,
) -> crate::ApiResult<serde_json::Value> {
    let result = (|| -> Result<_> {
        crate::memory::project_exists(&app, &scope.project_id)?;
        let packet = Store::open(&app.data)?.read(&hash)?;
        if packet.source.project_id != scope.project_id {
            bail!("Quality source unavailable");
        }
        Ok(serde_json::to_value(packet)?)
    })();
    result.map(axum::Json).map_err(|_| {
        crate::error(
            axum::http::StatusCode::NOT_FOUND,
            "Quality source unavailable or invalid",
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_explicit_source_reopens_rejects_tampering_and_changed_trace() {
        let root = std::env::temp_dir().join(format!(
            "allpaka-quality-source-{}",
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let traces = crate::observability::Store::new(&root).unwrap();
        let trace = traces.begin("source", "default").unwrap();
        trace
            .span("root", "agent", None)
            .finish("completed", &serde_json::Value::Null);
        let source = Source {
            project_id: "default".into(),
            trace_id: trace.id(),
            trace_sha256: traces
                .completed_trace_fingerprint(&trace.id(), "default")
                .unwrap(),
            input: "question".into(),
            output: "answer".into(),
            reference: None,
        };
        let store = Store::open(&root).unwrap();
        let saved = store.save(&root, source.clone()).unwrap();
        assert_eq!(
            store.save(&root, source.clone()).unwrap().source_sha256,
            saved.source_sha256
        );
        assert_eq!(std::fs::read_dir(&store.directory).unwrap().count(), 1);
        assert_eq!(
            Store::open(&root)
                .unwrap()
                .read(&saved.source_sha256)
                .unwrap()
                .source
                .output,
            "answer"
        );
        let path = store
            .directory
            .join(format!("{}.json", saved.source_sha256));
        let original = std::fs::read(&path).unwrap();
        let mut corrupt: serde_json::Value = serde_json::from_slice(&original).unwrap();
        corrupt["source"]["output"] = serde_json::json!("substituted");
        std::fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(store.read(&saved.source_sha256).is_err());
        std::fs::write(&path, original).unwrap();
        trace
            .span("changed", "tool", Some(0))
            .finish("completed", &serde_json::Value::Null);
        assert!(execution_source(&root, &saved.source_sha256, "default").is_err());
        assert!(store.save(&root, source).is_err());
        assert!(store.read(&saved.source_sha256).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }
}
