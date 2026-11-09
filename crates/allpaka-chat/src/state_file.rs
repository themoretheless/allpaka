//! Atomic read/write for the `<name>.state` JSON lists Studio owns.
//!
//! Every state file is a small JSON array written as a temp file plus a
//! rename, so a crash mid-write leaves the previous file intact. Reads are
//! capped: a truncated or hand-pasted oversized file is rejected before it
//! is parsed.

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

const MAX_STATE_BYTES: usize = 65536;

pub fn load_list<T>(
    data: &Path,
    file: &str,
    label: &str,
    validate: impl Fn(&T) -> Result<()>,
) -> Result<Vec<T>>
where
    T: DeserializeOwned,
{
    let path = data.join(format!("{file}.state"));
    if !path.exists() && std::fs::symlink_metadata(&path).is_err() {
        return Ok(Vec::new());
    }
    let bytes = crate::storage::read_bounded_bytes(&path, MAX_STATE_BYTES)
        .map_err(|_| anyhow::anyhow!("Invalid {label} file"))?;
    let list: Vec<T> =
        serde_json::from_slice(&bytes).with_context(|| format!("Invalid {label} file"))?;
    for item in &list {
        validate(item)?;
    }
    Ok(list)
}

pub fn save_list<T: Serialize + ?Sized>(data: &Path, file: &str, list: &T) -> Result<()> {
    let path = data.join(format!("{file}.state"));
    let bytes = serde_json::to_vec(list)?;
    crate::storage::atomic_write_bytes(&path, &bytes)
}
