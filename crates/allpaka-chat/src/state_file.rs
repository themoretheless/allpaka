//! Atomic, lock-serialised read/modify/write for the `<name>.state` JSON lists Studio owns.
//!
//! Every state file is a small JSON array written as a temp file plus a
//! rename, so a crash mid-write leaves the previous file intact. Reads are
//! capped: a truncated or hand-pasted oversized file is rejected before it
//! is parsed. Read-modify-write mutations hold both an in-process per-path
//! mutex and an OS advisory lock (`<name>.state.lock`) so neither concurrent
//! threads nor helper processes can lose updates.

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, OnceLock},
};

const MAX_STATE_BYTES: usize = 65536;

struct StateGuard {
    _thread: MutexGuard<'static, ()>,
    file: File,
}

impl Drop for StateGuard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

fn lock_state_path(state_path: &Path) -> Result<StateGuard> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, &'static Mutex<()>>>> = OnceLock::new();
    if let Some(parent) = state_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let canonical_parent = state_path
        .parent()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| state_path.parent().unwrap_or(Path::new(".")).to_path_buf());
    let key = canonical_parent.join(state_path.file_name().unwrap_or_default());

    let mutex: &'static Mutex<()> = {
        let mut map = LOCKS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.entry(key)
            .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))))
    };
    let thread_guard = mutex.lock().unwrap_or_else(|e| e.into_inner());

    let lock_path = state_path.with_extension("state.lock");
    if std::fs::symlink_metadata(&lock_path).is_ok_and(|m| !m.is_file()) {
        bail!("Invalid state lock file");
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)?;
    file.lock_exclusive()
        .with_context(|| format!("Locking {}", lock_path.display()))?;

    Ok(StateGuard {
        _thread: thread_guard,
        file,
    })
}

fn load_unlocked<T>(path: &Path, label: &str, validate: impl Fn(&T) -> Result<()>) -> Result<Vec<T>>
where
    T: DeserializeOwned,
{
    if !path.exists() && std::fs::symlink_metadata(path).is_err() {
        return Ok(Vec::new());
    }
    let bytes = crate::storage::read_bounded_bytes(path, MAX_STATE_BYTES)
        .map_err(|_| anyhow::anyhow!("Invalid {label} file"))?;
    let list: Vec<T> =
        serde_json::from_slice(&bytes).with_context(|| format!("Invalid {label} file"))?;
    for item in &list {
        validate(item)?;
    }
    Ok(list)
}

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
    load_unlocked(&path, label, validate)
}

pub fn save_list<T: Serialize + ?Sized>(data: &Path, file: &str, list: &T) -> Result<()> {
    let path = data.join(format!("{file}.state"));
    let _guard = lock_state_path(&path)?;
    let bytes = serde_json::to_vec(list)?;
    crate::storage::atomic_write_bytes(&path, &bytes)
}

/// Atomically load, mutate, and persist `<file>.state` under an exclusive lock
/// so concurrent writers cannot overwrite each other's changes.
pub fn update_list<T, R>(
    data: &Path,
    file: &str,
    label: &str,
    validate: impl Fn(&T) -> Result<()>,
    mutate: impl FnOnce(&mut Vec<T>) -> Result<R>,
) -> Result<R>
where
    T: DeserializeOwned + Serialize,
{
    let path = data.join(format!("{file}.state"));
    let _guard = lock_state_path(&path)?;
    let mut list = load_unlocked(&path, label, validate)?;
    let out = mutate(&mut list)?;
    let bytes = serde_json::to_vec(&list)?;
    crate::storage::atomic_write_bytes(&path, &bytes)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::sync::{Arc, Barrier};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Item {
        id: String,
        value: usize,
    }

    #[test]
    fn concurrent_update_list_preserves_every_writer() {
        let dir =
            std::env::temp_dir().join(format!("allpaka-state-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let workers = 16usize;
        let rounds = 8usize;
        let barrier = Arc::new(Barrier::new(workers));

        std::thread::scope(|s| {
            for w in 0..workers {
                let data = dir.as_path();
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    barrier.wait();
                    for r in 0..rounds {
                        let id = format!("w{w}-r{r}");
                        update_list(
                            data,
                            "concurrent",
                            "item",
                            |item: &Item| {
                                anyhow::ensure!(!item.id.is_empty(), "empty id");
                                Ok(())
                            },
                            |list| {
                                list.push(Item {
                                    id,
                                    value: w * 100 + r,
                                });
                                Ok(())
                            },
                        )
                        .unwrap();
                    }
                });
            }
        });

        let final_list: Vec<Item> = load_list(&dir, "concurrent", "item", |_| Ok(())).unwrap();
        assert_eq!(final_list.len(), workers * rounds);
        for w in 0..workers {
            for r in 0..rounds {
                let expected = Item {
                    id: format!("w{w}-r{r}"),
                    value: w * 100 + r,
                };
                assert!(
                    final_list.contains(&expected),
                    "missing {expected:?} after concurrent updates"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_or_invalid_state_file_is_rejected() {
        let dir =
            std::env::temp_dir().join(format!("allpaka-state-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.state");
        std::fs::write(&path, vec![b'['; MAX_STATE_BYTES + 1]).unwrap();
        let err = load_list::<Item>(&dir, "broken", "widget", |_| Ok(())).unwrap_err();
        assert!(err.to_string().contains("Invalid widget file"));

        std::fs::write(&path, b"[{\"id\":\"\",\"value\":1}]").unwrap();
        let err = load_list::<Item>(&dir, "broken", "widget", |item| {
            anyhow::ensure!(!item.id.is_empty(), "id required");
            Ok(())
        })
        .unwrap_err();
        assert!(err.to_string().contains("id required"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
