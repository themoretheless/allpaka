//! Atomic read/write for the `<name>.state` JSON lists Studio owns.
//!
//! Every state file is a small JSON array written as a temp file plus a
//! rename, so a crash mid-write leaves the previous file intact. Reads are
//! capped: a truncated or hand-pasted oversized file is rejected before it
//! is parsed.

use anyhow::{bail, Context, Result};
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
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() > MAX_STATE_BYTES {
        bail!("Invalid {label} file");
    }
    let list: Vec<T> = serde_json::from_slice(&bytes).context(format!("Invalid {label} file"))?;
    for item in &list {
        validate(item)?;
    }
    Ok(list)
}

pub fn save_list<T: Serialize + ?Sized>(data: &Path, file: &str, list: &T) -> Result<()> {
    let path = data.join(format!("{file}.state"));
    let temp = data.join(format!("{file}.tmp"));
    std::fs::write(&temp, serde_json::to_vec(list)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
}
