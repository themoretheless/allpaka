//! Web retrieval (`web_fetch`) and search (`web_search`) with strict SSRF protection,
//! DNS IP pinning, manual redirect validation, HTML extraction, and session spill truncation.

use crate::spill;
use anyhow::{bail, Context, Result};
use reqwest::header::LOCATION;
use reqwest::redirect::Policy;
use reqwest::{Client, Url};
use serde_json::{json, Value};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

const MAX_REDIRECTS: usize = 5;
const MAX_FETCH_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_MAX_CHARS: usize = 24 * 1024;
const MAX_ALLOWED_CHARS: usize = 128 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
const USER_AGENT: &str = "allpaka-studio/1.0 (+https://github.com/allpaka)";

pub(crate) fn web_fetch_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a public HTTP/HTTPS web page with SSRF protection, extract readable text (or raw markup), and spill oversized responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "HTTP or HTTPS URL to retrieve" },
                    "max_chars": { "type": "integer", "minimum": 256, "maximum": 131072 },
                    "raw": { "type": "boolean", "description": "Preserve raw text/HTML instead of stripping tags" }
                },
                "required": ["url"],
                "additionalProperties": false
            }
        }
    })
}

pub(crate) fn web_search_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using ALLPAKA_SEARCH_URL (if configured) or public search HTML endpoints and return structured results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "minLength": 1, "maxLength": 500 },
                    "max_results": { "type": "integer", "minimum": 1, "maximum": 20 }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        }
    })
}

fn is_blocked_ipv4(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_unspecified()
        || v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_multicast()
        || v4.is_broadcast()
        || v4.is_documentation()
        || o[0] == 0
        // Carrier-grade NAT 100.64.0.0/10
        || (o[0] == 100 && (o[1] & 0xC0) == 64)
        // IETF protocol assignments 192.0.0.0/24
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)
        // Benchmarking 198.18.0.0/15
        || (o[0] == 198 && (o[1] & 0xFE) == 18)
        // Reserved / Class E 240.0.0.0/4
        || o[0] >= 240
}

fn is_blocked_ipv6(v6: Ipv6Addr) -> bool {
    if v6.is_unspecified() || v6.is_loopback() || v6.is_multicast() {
        return true;
    }
    let seg = v6.segments();
    // Unique local fc00::/7
    if (seg[0] & 0xFE00) == 0xFC00 {
        return true;
    }
    // Link-local fe80::/10
    if (seg[0] & 0xFFC0) == 0xFE80 {
        return true;
    }
    // Site-local fec0::/10 (deprecated)
    if (seg[0] & 0xFFC0) == 0xFEC0 {
        return true;
    }
    // Documentation 2001:db8::/32
    if seg[0] == 0x2001 && seg[1] == 0x0DB8 {
        return true;
    }
    // IPv4-mapped ::ffff:a.b.c.d
    if let Some(v4) = v6.to_ipv4_mapped() {
        return is_blocked_ipv4(v4);
    }
    // IPv4-compatible ::a.b.c.d
    if seg[..6] == [0; 6] {
        let v4 = Ipv4Addr::new(
            (seg[6] >> 8) as u8,
            (seg[6] & 0xFF) as u8,
            (seg[7] >> 8) as u8,
            (seg[7] & 0xFF) as u8,
        );
        return is_blocked_ipv4(v4);
    }
    false
}

pub(crate) fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_ipv4(v4),
        IpAddr::V6(v6) => is_blocked_ipv6(v6),
    }
}

fn unbracket_host(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host)
}

fn is_blocked_hostname(host: &str) -> bool {
    let lower = unbracket_host(host)
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    lower.is_empty()
        || lower == "localhost"
        || lower.ends_with(".localhost")
        || lower.ends_with(".local")
        || lower.ends_with(".internal")
        || lower == "metadata"
        || lower == "metadata.google.internal"
}

pub(crate) fn validate_public_url_syntax(raw_url: &str) -> Result<Url> {
    let url = Url::parse(raw_url.trim()).context("Invalid URL")?;
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        bail!("Only http and https URLs are permitted (got `{scheme}`)");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("URLs with embedded credentials are rejected");
    }
    let host = url.host_str().context("URL has no host")?;
    if is_blocked_hostname(host) {
        bail!("Refusing request to local/internal host `{host}`");
    }
    let clean_host = unbracket_host(host);
    if let Ok(ip) = clean_host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            bail!("Refusing request to blocked IP `{ip}`");
        }
    }
    Ok(url)
}

pub(crate) async fn validate_and_resolve_public_url(
    raw_url: &str,
) -> Result<(Url, Option<(String, SocketAddr)>)> {
    let url = validate_public_url_syntax(raw_url)?;
    let clean_host = unbracket_host(url.host_str().context("URL has no host")?).to_owned();
    let port = url
        .port_or_known_default()
        .context("Unknown port for URL scheme")?;

    if let Ok(ip) = clean_host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            bail!("Refusing request to blocked IP `{ip}`");
        }
        return Ok((url, None));
    }

    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((clean_host.as_str(), port))
        .await
        .with_context(|| format!("DNS resolution failed for `{clean_host}`"))?
        .collect();
    if addrs.is_empty() {
        bail!("DNS resolution returned no addresses for `{clean_host}`");
    }
    for addr in &addrs {
        if is_blocked_ip(addr.ip()) {
            bail!(
                "Refusing request to `{clean_host}`: resolved to blocked IP `{}`",
                addr.ip()
            );
        }
    }
    let pinned = addrs[0];
    Ok((url, Some((clean_host, pinned))))
}

fn build_pinned_client(pin: Option<&(String, SocketAddr)>) -> Result<Client> {
    let mut builder = Client::builder()
        .redirect(Policy::none())
        .timeout(FETCH_TIMEOUT)
        .user_agent(USER_AGENT);
    if let Some((host, addr)) = pin {
        builder = builder.resolve(host, *addr);
    }
    builder.build().context("Failed to construct HTTP client")
}

async fn read_bounded_body(mut resp: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        let remain = limit.saturating_sub(buf.len());
        if remain == 0 {
            break;
        }
        let take = chunk.len().min(remain);
        buf.extend_from_slice(&chunk[..take]);
        if take < chunk.len() {
            break;
        }
    }
    Ok(buf)
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(rel_end) = s[i + 1..].find(';') {
                if rel_end <= 12 {
                    let ent = &s[i + 1..i + 1 + rel_end];
                    let decoded = match ent {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" | "#39" => Some('\''),
                        "nbsp" => Some(' '),
                        _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                            u32::from_str_radix(&ent[2..], 16)
                                .ok()
                                .and_then(char::from_u32)
                        }
                        _ if ent.starts_with('#') => {
                            ent[1..].parse::<u32>().ok().and_then(char::from_u32)
                        }
                        _ => None,
                    };
                    if let Some(ch) = decoded {
                        out.push(ch);
                        i += 1 + rel_end + 1;
                        continue;
                    }
                }
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn strip_tag_block_ci(html: &str, tag: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut pos = 0usize;
    while let Some(rel_start) = lower[pos..].find(&open) {
        let start = pos + rel_start;
        out.push_str(&html[pos..start]);
        let next_after = lower.get(start + open.len()..start + open.len() + 1);
        let valid_tag = matches!(next_after, Some(c) if c == ">" || c == " " || c == "\t" || c == "\n" || c == "\r");
        if !valid_tag {
            out.push_str(&html[start..start + open.len()]);
            pos = start + open.len();
            continue;
        }
        if let Some(rel_end) = lower[start..].find(&close) {
            pos = start + rel_end + close.len();
        } else {
            pos = html.len();
            break;
        }
    }
    out.push_str(&html[pos..]);
    out
}

pub(crate) fn extract_html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let gt = lower[start..].find('>')? + start + 1;
    let end = lower[gt..].find("</title>")? + gt;
    let title = decode_entities(html[gt..end].trim())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

pub(crate) fn html_to_text(html: &str) -> String {
    let mut cleaned = strip_tag_block_ci(html, "script");
    cleaned = strip_tag_block_ci(&cleaned, "style");
    cleaned = strip_tag_block_ci(&cleaned, "noscript");
    cleaned = strip_tag_block_ci(&cleaned, "svg");

    let mut out = String::with_capacity(cleaned.len() / 2);
    let mut in_tag = false;
    let mut tag_buf = String::new();
    for ch in cleaned.chars() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let tag = tag_buf
                    .trim()
                    .trim_start_matches('/')
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(
                    tag.as_str(),
                    "p" | "div"
                        | "br"
                        | "li"
                        | "tr"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "section"
                        | "article"
                        | "blockquote"
                        | "pre"
                ) && !out.ends_with('\n')
                {
                    out.push('\n');
                }
                tag_buf.clear();
            } else if tag_buf.len() < 64 {
                tag_buf.push(ch);
            }
        } else if ch == '<' {
            in_tag = true;
            tag_buf.clear();
        } else {
            out.push(ch);
        }
    }

    let decoded = decode_entities(&out);
    let mut lines = Vec::new();
    for line in decoded.lines() {
        let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() {
            lines.push(collapsed);
        }
    }
    lines.join("\n")
}

async fn fetch_public_bytes(raw_url: &str, accept: &str) -> Result<(Url, u16, String, Vec<u8>)> {
    let mut current = raw_url.to_owned();
    for hop in 0..=MAX_REDIRECTS {
        let (url, pin) = validate_and_resolve_public_url(&current).await?;
        let client = build_pinned_client(pin.as_ref())?;
        let resp = client
            .get(url.clone())
            .header("Accept", accept)
            .send()
            .await
            .with_context(|| format!("HTTP request failed for {url}"))?;
        let status = resp.status();
        if status.is_redirection() {
            if hop == MAX_REDIRECTS {
                bail!("Too many HTTP redirects (max {MAX_REDIRECTS})");
            }
            let loc = resp
                .headers()
                .get(LOCATION)
                .and_then(|v| v.to_str().ok())
                .context("Redirect response missing Location header")?;
            let next_url = url.join(loc).context("Invalid redirect target URL")?;
            current = next_url.to_string();
            continue;
        }
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_owned();
        let bytes = read_bounded_body(resp, MAX_FETCH_BYTES).await?;
        return Ok((url, status.as_u16(), content_type, bytes));
    }
    bail!("Unreachable redirect loop termination");
}

pub(crate) async fn execute_web_fetch(
    data: Option<&Path>,
    session_id: Option<&str>,
    args: &Value,
) -> Result<Value> {
    let raw_url = args["url"].as_str().context("Missing `url` string")?;
    let max_chars = args["max_chars"]
        .as_u64()
        .map(|n| (n as usize).clamp(256, MAX_ALLOWED_CHARS))
        .unwrap_or(DEFAULT_MAX_CHARS);
    let raw_mode = args["raw"].as_bool().unwrap_or(false);

    let (final_url, status, content_type, bytes) = fetch_public_bytes(
        raw_url,
        "text/html,application/xhtml+xml,text/plain,application/json,*/*;q=0.5",
    )
    .await?;
    if status >= 400 {
        bail!("HTTP {status} fetching {final_url}");
    }

    let ct_lower = content_type.to_ascii_lowercase();
    let is_text = ct_lower.starts_with("text/")
        || ct_lower.contains("json")
        || ct_lower.contains("xml")
        || ct_lower.contains("html")
        || ct_lower.contains("javascript")
        || ct_lower.contains("yaml");
    if !is_text && bytes.iter().take(256).filter(|&&b| b == 0).count() > 0 {
        bail!(
            "Refusing non-text binary response `{content_type}` ({} bytes)",
            bytes.len()
        );
    }

    let body_str = String::from_utf8_lossy(&bytes).into_owned();
    let is_html = ct_lower.contains("html") || body_str.trim_start().starts_with('<');
    let title = if is_html {
        extract_html_title(&body_str)
    } else {
        None
    };
    let text = if raw_mode || !is_html {
        body_str
    } else {
        html_to_text(&body_str)
    };

    let (preview, spill_id, truncated, total_bytes) =
        spill::truncate_or_spill(data, session_id, "web_fetch", &text, max_chars);

    Ok(json!({
        "url": final_url.to_string(),
        "status": status,
        "content_type": content_type,
        "title": title,
        "content": preview,
        "truncated": truncated,
        "total_bytes": total_bytes,
        "spill_id": spill_id,
    }))
}

fn extract_attr<'a>(tag: &'a str, attr: &str) -> Option<&'a str> {
    let lower = tag.to_ascii_lowercase();
    for quote in ['"', '\''] {
        let pat = format!("{attr}={quote}");
        if let Some(pos) = lower.find(&pat) {
            let start = pos + pat.len();
            if let Some(end) = tag[start..].find(quote) {
                return Some(&tag[start..start + end]);
            }
        }
    }
    None
}

fn unwrap_ddg_href(href: &str) -> String {
    let decoded = decode_entities(href.trim());
    let full = if decoded.starts_with("//") {
        format!("https:{decoded}")
    } else {
        decoded
    };
    if let Ok(u) = Url::parse(&full) {
        if u.host_str().is_some_and(|h| h.contains("duckduckgo.com")) {
            for (k, v) in u.query_pairs() {
                if k == "uddg" && !v.is_empty() {
                    return v.into_owned();
                }
            }
        }
    }
    full
}

pub(crate) fn parse_duckduckgo_results(html: &str, max_results: usize) -> Vec<Value> {
    let mut results = Vec::new();
    let lower = html.to_ascii_lowercase();
    let mut pos = 0usize;
    while results.len() < max_results {
        let next_a = lower[pos..].find("<a ");
        let Some(rel) = next_a else { break };
        let a_start = pos + rel;
        let Some(gt_rel) = lower[a_start..].find('>') else {
            break;
        };
        let tag_end = a_start + gt_rel;
        let tag = &html[a_start..=tag_end];
        let is_result_link = tag.contains("result__a") || tag.contains("result-link");
        if is_result_link {
            if let Some(close_rel) = lower[tag_end + 1..].find("</a>") {
                let raw_title = &html[tag_end + 1..tag_end + 1 + close_rel];
                let title = html_to_text(raw_title);
                let url = extract_attr(tag, "href")
                    .map(unwrap_ddg_href)
                    .unwrap_or_default();
                let search_window_end = (tag_end + 1 + close_rel + 1200).min(html.len());
                let snippet_slice = &html[tag_end + 1 + close_rel..search_window_end];
                let snippet = extract_first_snippet(snippet_slice);
                if !title.is_empty() && !url.is_empty() {
                    results.push(json!({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }));
                }
                pos = tag_end + 1 + close_rel + 4;
                continue;
            }
        }
        pos = tag_end + 1;
    }
    results
}

fn extract_first_snippet(slice: &str) -> String {
    let lower = slice.to_ascii_lowercase();
    for marker in ["result__snippet", "result-snippet"] {
        if let Some(m) = lower.find(marker) {
            if let Some(gt) = lower[m..].find('>') {
                let start = m + gt + 1;
                for close in ["</a>", "</td>", "</span>", "</div>"] {
                    if let Some(end) = lower[start..].find(close) {
                        return html_to_text(&slice[start..start + end]);
                    }
                }
            }
        }
    }
    String::new()
}

pub(crate) async fn execute_web_search(args: &Value) -> Result<Value> {
    let query = args["query"]
        .as_str()
        .context("Missing `query` string")?
        .trim();
    if query.is_empty() {
        bail!("Search query cannot be empty");
    }
    let max_results = args["max_results"]
        .as_u64()
        .map(|n| (n as usize).clamp(1, 20))
        .unwrap_or(8);

    if let Ok(endpoint) = std::env::var("ALLPAKA_SEARCH_URL") {
        let trimmed = endpoint.trim();
        if !trimmed.is_empty() {
            let mut url = Url::parse(trimmed).context("Invalid ALLPAKA_SEARCH_URL")?;
            url.query_pairs_mut().append_pair("q", query);
            let client = Client::builder()
                .timeout(FETCH_TIMEOUT)
                .user_agent(USER_AGENT)
                .build()?;
            let resp = client
                .get(url)
                .send()
                .await
                .context("Search endpoint request failed")?;
            let status = resp.status();
            let body = read_bounded_body(resp, MAX_FETCH_BYTES).await?;
            if !status.is_success() {
                bail!("Search endpoint returned HTTP {status}");
            }
            if let Ok(parsed) = serde_json::from_slice::<Value>(&body) {
                let items = parsed
                    .get("results")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .take(max_results)
                    .collect::<Vec<_>>();
                return Ok(json!({
                    "ok": true,
                    "source": "configured_endpoint",
                    "query": query,
                    "count": items.len(),
                    "results": items,
                }));
            }
            let text = String::from_utf8_lossy(&body);
            let items = parse_duckduckgo_results(&text, max_results);
            return Ok(json!({
                "ok": true,
                "source": "configured_html",
                "query": query,
                "count": items.len(),
                "results": items,
            }));
        }
    }

    let mut search_url = Url::parse("https://lite.duckduckgo.com/lite/")?;
    search_url.query_pairs_mut().append_pair("q", query);
    match fetch_public_bytes(search_url.as_str(), "text/html").await {
        Ok((_, status, _, bytes)) if status < 400 => {
            let html = String::from_utf8_lossy(&bytes);
            let results = parse_duckduckgo_results(&html, max_results);
            Ok(json!({
                "ok": true,
                "source": "duckduckgo_lite",
                "query": query,
                "count": results.len(),
                "results": results,
            }))
        }
        Ok((_, status, _, _)) => Ok(json!({
            "ok": false,
            "available": false,
            "query": query,
            "error": format!("Search upstream returned HTTP {status}"),
            "results": [],
        })),
        Err(err) => Ok(json!({
            "ok": false,
            "available": false,
            "query": query,
            "error": err.to_string(),
            "results": [],
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssrf_blocks_loopback_private_metadata_and_ipv4_mapped_v6() {
        for blocked in [
            "http://127.0.0.1/",
            "http://10.0.0.5/secret",
            "http://172.16.4.2/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/",
            "http://0.0.0.0:8080/",
            "http://[::1]:8080/",
            "http://[fc00::1]/",
            "http://[fe80::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:169.254.169.254]/",
            "http://localhost/admin",
            "http://metadata.google.internal/",
            "http://service.internal/",
            "file:///etc/passwd",
            "https://user:pass@93.184.216.34/",
        ] {
            assert!(
                validate_public_url_syntax(blocked).is_err(),
                "expected SSRF block for {blocked}"
            );
        }

        assert!(validate_public_url_syntax("https://93.184.216.34/docs").is_ok());
        assert!(validate_public_url_syntax("https://example.org/guide").is_ok());
    }

    #[test]
    fn html_to_text_strips_scripts_and_decodes_entities() {
        let html = r#"<html><head><title>Rust &amp; Allpaka</title><style>body{color:red}</style></head>
        <body><script>alert('xss')</script><h1>Heading &quot;One&quot;</h1><p>Alpha &lt; Beta&#39;s</p></body></html>"#;
        assert_eq!(extract_html_title(html).as_deref(), Some("Rust & Allpaka"));
        let txt = html_to_text(html);
        assert!(!txt.contains("alert"));
        assert!(!txt.contains("color:red"));
        assert!(txt.contains("Heading \"One\""));
        assert!(txt.contains("Alpha < Beta's"));
    }

    #[test]
    fn parse_duckduckgo_lite_and_html_results() {
        let sample = r#"
        <table><tr><td>
          <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Frust" class="result-link">Example Rust</a>
        </td></tr>
        <tr><td class="result-snippet">A systems &amp; harness guide.</td></tr></table>
        "#;
        let parsed = parse_duckduckgo_results(sample, 5);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["title"], "Example Rust");
        assert_eq!(parsed[0]["url"], "https://example.org/rust");
        assert_eq!(parsed[0]["snippet"], "A systems & harness guide.");
    }
}
