//! HTTP/1.1 request framing, JSON responses, and SSE streaming helpers.

use anyhow::{bail, Result};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;

/// Largest accepted JSON request body. Image attachments arrive base64-encoded
/// (Studio allows four images, ~6 MB before encoding, so ~8 MB on the wire), and
/// the previous fixed 4 MiB ceiling rejected them before the handler could
/// answer. Override with `ALLPAKA_MAX_BODY_MIB` (1..1024, default 16).
pub(crate) fn max_body_bytes_from(configured: Option<&str>) -> usize {
    const DEFAULT_MIB: usize = 16;
    let mib = configured
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|mib| (1..=1024).contains(mib))
        .unwrap_or(DEFAULT_MIB);
    mib * 1024 * 1024
}

pub(crate) fn max_body_bytes() -> usize {
    static MAX_BODY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_BODY
        .get_or_init(|| max_body_bytes_from(std::env::var("ALLPAKA_MAX_BODY_MIB").ok().as_deref()))
}

pub(crate) fn read_request(stream: &mut TcpStream) -> Result<(String, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end;
    loop {
        let prev_len = buf.len();
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-request");
        }
        buf.extend_from_slice(&chunk[..n]);
        let scan_start = prev_len.saturating_sub(3);
        if let Some(rel) = find_header_end(&buf[scan_start..]) {
            header_end = scan_start + rel;
            break;
        }
        if buf.len() > 1 << 20 {
            bail!("request headers over 1 MiB");
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]);
    let request_line = head.lines().next().unwrap_or_default().to_string();
    let content_length: usize = head
        .lines()
        .find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        })
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);
    let limit = max_body_bytes();
    anyhow::ensure!(
        content_length <= limit,
        "request body exceeds {} MiB (raise ALLPAKA_MAX_BODY_MIB)",
        limit >> 20
    );

    let already = buf.len() - (header_end + 4);
    let mut body = Vec::with_capacity(content_length.max(already));
    body.extend_from_slice(&buf[header_end + 4..]);
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-body");
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    let body_str = String::from_utf8(body)
        .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned());
    Ok((request_line, body_str))
}

pub(crate) fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub(crate) fn respond(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let text = body.to_string();
    let reason = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

pub(crate) fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
          Connection: close\r\n\r\n",
    )?;
    Ok(())
}

pub(crate) fn write_sse_event(stream: &mut TcpStream, body: &Value) -> Result<()> {
    stream.write_all(format!("data: {body}\n\n").as_bytes())?;
    stream.flush()?;
    Ok(())
}

pub(crate) fn respond_html(stream: &mut TcpStream, page: &str) -> Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_limit_defaults_to_16_mib_and_accepts_overrides() {
        assert_eq!(max_body_bytes_from(None), 16 << 20);
        assert_eq!(max_body_bytes_from(Some(" 32 ")), 32 << 20);
        // Out-of-range or unparsable values fall back to the default.
        assert_eq!(max_body_bytes_from(Some("0")), 16 << 20);
        assert_eq!(max_body_bytes_from(Some("2048")), 16 << 20);
        assert_eq!(max_body_bytes_from(Some("nope")), 16 << 20);
    }
}
