//! Undo of an agent's file write, checked against what is on disk.

use crate::types::FileCheckpoint;
use std::path::Path;

/// Put back what `checkpoint` recorded, and say what that was: `"previous
/// content"` or `"removed"` (the tool had created the file).
///
/// Refused, with the reason, unless the file still holds exactly the bytes the
/// tool wrote: a change made since, by a person or another program, is not
/// overwritten. After restoring, the file is read back and must match.
pub(crate) fn restore(checkpoint: &FileCheckpoint) -> Result<&'static str, String> {
    let path = Path::new(&checkpoint.path);
    let current = std::fs::read(path).map_err(|error| {
        format!(
            "{} cannot be read ({error}); it was moved or removed since the change",
            checkpoint.path
        )
    })?;
    if crate::storage::hex_sha256(&current) != checkpoint.after_sha256 {
        return Err(format!(
            "{} has changed since the agent wrote it; undo would discard that change",
            checkpoint.path
        ));
    }
    match &checkpoint.before {
        Some(before) => {
            crate::tools::write_verified(path, before).map_err(|error| error.to_string())?;
            Ok("previous content")
        }
        None => {
            std::fs::remove_file(path).map_err(|error| error.to_string())?;
            if path.exists() {
                return Err(format!("{} is still there after removal", checkpoint.path));
            }
            Ok("removed")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::restore;
    use crate::types::FileCheckpoint;

    fn record(path: &std::path::Path, before: Option<&str>, after: &str) -> FileCheckpoint {
        FileCheckpoint {
            path: path.to_string_lossy().into_owned(),
            before: before.map(str::to_owned),
            after_sha256: crate::storage::hex_sha256(after.as_bytes()),
            tool: "edit_file".into(),
            recorded_ms: 0,
        }
    }

    fn dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("allpaka-undo-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_edit_is_put_back_byte_for_byte() {
        let dir = dir();
        let path = dir.join("a.txt");
        std::fs::write(&path, "after\r\n").unwrap();
        assert_eq!(
            restore(&record(&path, Some("before\n\u{e9}"), "after\r\n")),
            Ok("previous content")
        );
        assert_eq!(std::fs::read(&path).unwrap(), "before\n\u{e9}".as_bytes());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_created_file_is_removed() {
        let dir = dir();
        let path = dir.join("new.txt");
        std::fs::write(&path, "made by the agent").unwrap();
        assert_eq!(
            restore(&record(&path, None, "made by the agent")),
            Ok("removed")
        );
        assert!(!path.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_change_made_since_is_not_overwritten() {
        let dir = dir();
        let path = dir.join("b.txt");
        std::fs::write(&path, "after, then edited by hand").unwrap();
        let refused = restore(&record(&path, Some("before"), "after")).unwrap_err();
        assert!(refused.contains("changed since"), "{refused}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "after, then edited by hand"
        );
        let gone = restore(&record(&dir.join("missing.txt"), Some("x"), "y")).unwrap_err();
        assert!(gone.contains("cannot be read"), "{gone}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
