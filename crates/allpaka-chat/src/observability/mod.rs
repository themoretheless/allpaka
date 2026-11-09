//! Native, metadata-only trace storage. No prompts, tool inputs or outputs are stored.
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

mod api;
mod guardrails;
mod ingest;
mod model;
mod receipts;
mod store;
mod trace;

pub(crate) use api::*;
use guardrails::*;
pub(crate) use ingest::*;
use model::*;
pub(crate) use receipts::*;
pub(crate) use trace::*;

type TraceCache = Arc<Mutex<std::collections::HashMap<String, (u64, Option<SystemTime>, Record)>>>;

#[derive(Clone)]
pub(crate) struct Store {
    root: Arc<PathBuf>,
    lifecycle: Arc<Mutex<()>>,
    cache: TraceCache,
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
        if record.status != "running" {
            file.sync_all()?;
        }
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(temp);
    result
}
fn update_cache(cache: &TraceCache, path: &std::path::Path, record: &Record) {
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        cache.lock().unwrap().insert(
            record.id.clone(),
            (meta.len(), meta.modified().ok(), record.clone()),
        );
    }
}
#[cfg(test)]
mod tests;
