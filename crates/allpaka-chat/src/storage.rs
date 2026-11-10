//! Shared storage, validation, and hex digest primitives for Studio state and artifacts.
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
};

#[inline]
pub fn format_hex(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for &b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[inline]
pub fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format_hex(&digest)
}

#[inline]
pub fn hex_sha256_hasher(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    format_hex(&digest)
}

/// Milliseconds since the Unix epoch, saturating; zero if the clock is before it.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

#[inline]
/// The one rule for named identifiers (B9, PLAN 3.4): `[A-Za-z0-9_-]`, 1 to
/// 100 bytes. It accepts every id any store accepted before. Generated ids
/// use the stricter [`valid_hex_id`].
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 100
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[inline]
/// Generated identifiers (sessions, projects, jobs): hex and `-`, 1 to 80 bytes.
pub fn valid_hex_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')
}

#[inline]
pub fn valid_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn read_bounded_bytes(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(not(unix))]
    {
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() {
            bail!("Refusing to read symlink");
        }
    }
    let file = options.open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() > limit as u64 {
        bail!("Invalid or oversize file");
    }
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("File exceeds read bound");
    }
    Ok(bytes)
}

pub fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_bytes_with_sync(path, bytes, false)
}

pub fn atomic_write_bytes_with_sync(path: &Path, bytes: &[u8], sync_parent: bool) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
    let result = (|| -> Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // Windows refuses to rename a file that still has an open handle.
        drop(file);
        std::fs::rename(&temp, path)?;
        if sync_parent {
            #[cfg(unix)]
            if let Some(parent) = path.parent() {
                std::fs::File::open(parent)?.sync_all()?;
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&temp);
    result
}

pub fn atomic_create_new_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
    let result = (|| -> Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // Close before linking so the temp name can be removed on Windows.
        drop(file);
        std::fs::hard_link(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PLAN 3.4 / B9: two identifier rules. Named ids (plugins, feedback,
    /// review queues, storage) share valid_id; generated ids (sessions,
    /// projects, matrix jobs) share valid_hex_id. Each accepts every id the
    /// per-store rules it replaced accepted.
    #[test]
    fn identifier_rules() {
        let long80 = "a".repeat(80);
        let long81 = "a".repeat(81);
        assert!(valid_id("Ab_9-z") && valid_id(&"a".repeat(100)));
        assert!(!valid_id("") && !valid_id(&"a".repeat(101)));
        assert!(!valid_id("a.b") && !valid_id("a/b"));
        assert!(valid_hex_id("0a-F9") && valid_hex_id(&long80));
        assert!(!valid_hex_id("g") && !valid_hex_id("a_b") && !valid_hex_id(&long81));
        let named = crate::plugins::valid_id;
        assert!(named("Ab_9-z") && named(&"a".repeat(100)) && !named("a.b"));
        let generated = crate::api::valid_id;
        assert!(generated("0a-f") && generated(&long80));
        assert!(!generated("a_b") && !generated(&long81) && !generated(""));
    }

    #[test]
    fn hex_and_validation_primitives_match_spec() {
        let h = hex_sha256(b"hello");
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert!(valid_sha256_hex(&h));
        assert!(!valid_sha256_hex("short"));
        assert!(!valid_sha256_hex(&h.to_uppercase()));
        assert!(valid_id("valid-id_123"));
        assert!(!valid_id(""));
        assert!(!valid_id("../bad"));
        assert!(valid_hex_id("abc-123"));
        assert!(!valid_hex_id("xyz"));
    }
}
