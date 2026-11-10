//! Atomic read/write for the `<name>.state` JSON lists Studio owns.
//!
//! Every state file is a small JSON array written as a temp file plus a
//! rename, so a crash mid-write leaves the previous file intact. Reads are
//! capped: a truncated or hand-pasted oversized file is rejected before it
//! is parsed.
//!
//! The rename makes each write atomic, but a load, change and save from two
//! handlers at once would still lose one of the changes. `update_list` holds
//! a per-file lock across the whole read-modify-write; the data directory
//! itself is owned by one Studio process (`data_lock`).

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

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

fn file_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut locks = LOCKS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    locks.entry(path.to_path_buf()).or_default().clone()
}

/// Load, change and save a list under one lock, so concurrent updates of
/// the same file cannot overwrite each other. `change` may fail to abort
/// without writing; its value is returned.
pub fn update_list<T, R>(
    data: &Path,
    file: &str,
    label: &str,
    validate: impl Fn(&T) -> Result<()>,
    change: impl FnOnce(&mut Vec<T>) -> Result<R>,
) -> Result<R>
where
    T: DeserializeOwned + Serialize,
{
    let lock = file_lock(&data.join(format!("{file}.state")));
    let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = load_list(data, file, label, validate)?;
    let out = change(&mut list)?;
    save_list(data, file, &list)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "allpaka-state-file-{tag}-{}-{}",
            std::process::id(),
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_is_an_empty_list_and_round_trips() {
        let dir = temp_dir("roundtrip");
        let empty: Vec<String> = load_list(&dir, "names", "name", |_| Ok(())).unwrap();
        assert!(empty.is_empty());
        save_list(&dir, "names", &["a".to_string(), "b".to_string()]).unwrap();
        let back: Vec<String> = load_list(&dir, "names", "name", |_| Ok(())).unwrap();
        assert_eq!(back, ["a", "b"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn oversized_or_invalid_files_are_rejected() {
        let dir = temp_dir("reject");
        std::fs::write(dir.join("big.state"), vec![b' '; MAX_STATE_BYTES + 1]).unwrap();
        let big = load_list::<String>(&dir, "big", "big thing", |_| Ok(()));
        assert_eq!(big.unwrap_err().to_string(), "Invalid big thing file");
        save_list(&dir, "bad", &["x".to_string()]).unwrap();
        let bad = load_list::<String>(&dir, "bad", "bad", |_| anyhow::bail!("nope"));
        assert_eq!(bad.unwrap_err().to_string(), "nope");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// B7: concurrent read-modify-write cycles keep every change.
    #[test]
    fn concurrent_updates_lose_nothing() {
        let dir = temp_dir("race");
        std::thread::scope(|s| {
            for i in 0..16u32 {
                let dir = &dir;
                s.spawn(move || {
                    update_list(
                        dir,
                        "items",
                        "item",
                        |_: &u32| Ok(()),
                        |list| {
                            std::thread::yield_now();
                            list.push(i);
                            Ok(())
                        },
                    )
                    .unwrap();
                });
            }
        });
        let mut list: Vec<u32> = load_list(&dir, "items", "item", |_| Ok(())).unwrap();
        list.sort();
        assert_eq!(list, (0..16).collect::<Vec<_>>());
        let kept: Result<()> = update_list(
            &dir,
            "items",
            "item",
            |_: &u32| Ok(()),
            |list| {
                list.clear();
                anyhow::bail!("abort")
            },
        );
        assert!(kept.is_err());
        let list: Vec<u32> = load_list(&dir, "items", "item", |_| Ok(())).unwrap();
        assert_eq!(list.len(), 16, "a failed change writes nothing");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
