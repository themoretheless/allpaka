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

/// Rename `temp` over `path`. On Windows the rename is refused while another
/// process holds `path` open without delete sharing (a reader, an indexer, an
/// antivirus scan); such holds are brief, so retry for up to half a second
/// instead of losing the write.
fn rename_over(temp: &Path, path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        let mut attempt = 0;
        loop {
            match std::fs::rename(temp, path) {
                Err(error)
                    if attempt < 50 && matches!(error.raw_os_error(), Some(5) | Some(32)) =>
                {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                other => return other,
            }
        }
    }
    #[cfg(not(windows))]
    std::fs::rename(temp, path)
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
        rename_over(&temp, path)?;
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

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode arbitrary bytes using standard RFC 4648 base64 with `=` padding.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[(n >> 18) & 0x3f] as char);
        out.push(B64_ALPHABET[(n >> 12) & 0x3f] as char);
        if chunk.len() > 1 {
            out.push(B64_ALPHABET[(n >> 6) & 0x3f] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ALPHABET[n & 0x3f] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode standard RFC 4648 base64 (ignoring ASCII whitespace).
pub fn base64_decode(input: &str) -> Result<Vec<u8>> {
    let compact: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if compact.is_empty() {
        return Ok(Vec::new());
    }
    if compact.len() % 4 != 0 {
        anyhow::bail!("invalid base64 length");
    }
    fn val(byte: u8) -> Result<u8> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => anyhow::bail!("invalid base64 byte"),
        }
    }
    let mut out = Vec::with_capacity(compact.len() / 4 * 3);
    let chunks = compact.len() / 4;
    for (idx, chunk) in compact.chunks_exact(4).enumerate() {
        let is_last = idx + 1 == chunks;
        if chunk[0] == b'=' || chunk[1] == b'=' {
            anyhow::bail!("invalid base64 padding");
        }
        if !is_last && (chunk[2] == b'=' || chunk[3] == b'=') {
            anyhow::bail!("unexpected mid-stream base64 padding");
        }
        let a = val(chunk[0])? as u32;
        let b = val(chunk[1])? as u32;
        let pad2 = chunk[2] == b'=';
        let pad3 = chunk[3] == b'=';
        if pad2 && !pad3 {
            anyhow::bail!("invalid base64 padding sequence");
        }
        let c = if pad2 { 0 } else { val(chunk[2])? as u32 };
        let d = if pad3 { 0 } else { val(chunk[3])? as u32 };
        let n = (a << 18) | (b << 12) | (c << 6) | d;
        out.push(((n >> 16) & 0xff) as u8);
        if !pad2 {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if !pad3 {
            out.push((n & 0xff) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that briefly holds the destination open must not lose the write.
    #[cfg(windows)]
    #[test]
    fn atomic_write_survives_a_brief_reader_without_delete_sharing() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir =
            std::env::temp_dir().join(format!("allpaka-rename-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("job.json");
        atomic_write_bytes(&path, b"old").unwrap();
        // FILE_SHARE_READ only: what Python's open() and most tools request.
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(&path)
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(held);
        });
        atomic_write_bytes(&path, b"new").unwrap();
        release.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        std::fs::remove_dir_all(&dir).unwrap();
    }

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

    #[test]
    fn base64_roundtrip_handles_padding_and_errors() {
        for sample in [
            b"".as_slice(),
            b"a",
            b"ab",
            b"abc",
            b"hello world\x00\xff",
            &[0u8, 1, 2, 250, 251, 252, 253, 254, 255],
        ] {
            let encoded = base64_encode(sample);
            assert_eq!(base64_decode(&encoded).unwrap(), sample);
        }
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
        assert!(base64_decode("abc").is_err());
        assert!(base64_decode("ab*=").is_err());
        assert!(base64_decode("a=aa").is_err());
        assert!(base64_decode("aa=a").is_err());
    }
}
