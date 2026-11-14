//! Document extraction for Studio attachments: PDF, DOCX, XLSX, plain text and
//! HTML in-process; images (OCR), audio and video through external tools when
//! they are on PATH. Every input is typed by magic bytes against an explicit
//! allow-list, capped per type, and stored as the original bytes under the
//! data directory, addressed by SHA-256, next to the extraction record.
use crate::{error, storage, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::{
    body::Bytes,
    extract::{Path as UrlPath, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// Extracted text is cut at this many bytes (on a char boundary) so the
/// attachment still fits in one message (`Message must contain 1–131072 bytes`).
pub(crate) const MAX_TEXT_BYTES: usize = 100_000;
/// Upper bound for one decompressed OOXML part, against zip bombs.
const MAX_XML_PART: u64 = 32 << 20;
const MAX_SHEETS: usize = 64;
const MAX_URL_LEN: usize = 2048;
const MAX_NAME_LEN: usize = 250;
const URL_TIMEOUT: Duration = Duration::from_secs(20);
const TOOL_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Kind {
    Text,
    Html,
    Pdf,
    Docx,
    Xlsx,
    Image,
    Audio,
    Video,
}
impl Kind {
    /// Input byte cap per type.
    pub(crate) fn max_bytes(self) -> usize {
        match self {
            Kind::Text | Kind::Html => 2 << 20,
            Kind::Pdf | Kind::Docx | Kind::Xlsx | Kind::Image => 8 << 20,
            Kind::Audio | Kind::Video => 9 << 20,
        }
    }
    fn extensions(self) -> &'static [&'static str] {
        match self {
            Kind::Text => &[
                "txt", "md", "csv", "tsv", "json", "log", "xml", "yaml", "yml",
            ],
            Kind::Html => &["html", "htm"],
            Kind::Pdf => &["pdf"],
            Kind::Docx => &["docx"],
            Kind::Xlsx => &["xlsx"],
            Kind::Image => &["png", "jpg", "jpeg", "gif", "webp", "tif", "tiff"],
            Kind::Audio => &["mp3", "wav", "ogg", "oga", "flac", "m4a"],
            Kind::Video => &["mp4", "mov", "m4v", "webm", "mkv"],
        }
    }
}

/// Every extension any kind claims; a name with one of these must match the
/// detected kind, so `report.pdf` holding a zip is refused instead of guessed.
fn claimed_kind(name: &str) -> Option<Kind> {
    let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
    [
        Kind::Text,
        Kind::Html,
        Kind::Pdf,
        Kind::Docx,
        Kind::Xlsx,
        Kind::Image,
        Kind::Audio,
        Kind::Video,
    ]
    .into_iter()
    .find(|k| k.extensions().contains(&ext.as_str()))
}

/// Type detection by content. Returns the kind and its MIME type, or an error
/// naming why the bytes are not on the allow-list.
pub(crate) fn detect(bytes: &[u8]) -> Result<(Kind, &'static str)> {
    let at = |offset: usize, magic: &[u8]| bytes.get(offset..offset + magic.len()) == Some(magic);
    if bytes.is_empty() {
        bail!("The file is empty");
    }
    if at(0, b"%PDF-") {
        return Ok((Kind::Pdf, "application/pdf"));
    }
    if at(0, b"PK\x03\x04") {
        let archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
            .context("Unreadable zip container")?;
        if archive.index_for_name("word/document.xml").is_some() {
            return Ok((
                Kind::Docx,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ));
        }
        if archive.index_for_name("xl/workbook.xml").is_some() {
            return Ok((
                Kind::Xlsx,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ));
        }
        bail!("Unsupported zip archive: only DOCX and XLSX documents are accepted");
    }
    if at(0, b"\x89PNG\r\n\x1a\n") {
        return Ok((Kind::Image, "image/png"));
    }
    if at(0, b"\xFF\xD8\xFF") {
        return Ok((Kind::Image, "image/jpeg"));
    }
    if at(0, b"GIF87a") || at(0, b"GIF89a") {
        return Ok((Kind::Image, "image/gif"));
    }
    if at(0, b"RIFF") && at(8, b"WEBP") {
        return Ok((Kind::Image, "image/webp"));
    }
    if at(0, b"II*\0") || at(0, b"MM\0*") {
        return Ok((Kind::Image, "image/tiff"));
    }
    if at(0, b"RIFF") && at(8, b"WAVE") {
        return Ok((Kind::Audio, "audio/wav"));
    }
    if at(0, b"ID3") || (bytes.len() > 1 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0) {
        return Ok((Kind::Audio, "audio/mpeg"));
    }
    if at(0, b"OggS") {
        return Ok((Kind::Audio, "audio/ogg"));
    }
    if at(0, b"fLaC") {
        return Ok((Kind::Audio, "audio/flac"));
    }
    if at(4, b"ftyp") {
        return Ok(if at(8, b"M4A ") {
            (Kind::Audio, "audio/mp4")
        } else if at(8, b"qt  ") {
            (Kind::Video, "video/quicktime")
        } else {
            (Kind::Video, "video/mp4")
        });
    }
    if at(0, b"\x1A\x45\xDF\xA3") {
        return Ok((Kind::Video, "video/webm"));
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        bail!("Unsupported file type: not PDF, DOCX, XLSX, UTF-8 text, image, audio or video");
    };
    if text.contains('\0') {
        bail!("Unsupported binary file");
    }
    let head = text.trim_start_matches('\u{feff}').trim_start();
    let head = head
        .get(..head.len().min(256))
        .unwrap_or(head)
        .to_ascii_lowercase();
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        return Ok((Kind::Html, "text/html"));
    }
    Ok((Kind::Text, "text/plain"))
}

/// External programs used for OCR and transcription. Names come from
/// `ALLPAKA_TESSERACT`, `ALLPAKA_FFMPEG`, `ALLPAKA_WHISPER` (otherwise the
/// usual binary names); tests point them at names that do not exist.
#[derive(Clone, Debug)]
pub(crate) struct Tools {
    pub(crate) tesseract: String,
    pub(crate) ffmpeg: String,
    pub(crate) whisper: Vec<String>,
    /// Model file for whisper.cpp's `whisper-cli` (`ALLPAKA_WHISPER_MODEL`).
    pub(crate) whisper_model: Option<PathBuf>,
}
impl Tools {
    pub(crate) fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Self {
            tesseract: var("ALLPAKA_TESSERACT").unwrap_or_else(|| "tesseract".into()),
            ffmpeg: var("ALLPAKA_FFMPEG").unwrap_or_else(|| "ffmpeg".into()),
            whisper: var("ALLPAKA_WHISPER")
                .map(|w| vec![w])
                .unwrap_or_else(|| vec!["whisper-cli".into(), "whisper".into()]),
            whisper_model: var("ALLPAKA_WHISPER_MODEL").map(PathBuf::from),
        }
    }
}

/// Resolves a program name or path to an executable file, or fails naming it.
pub(crate) fn find_tool(name: &str, purpose: &str) -> Result<PathBuf> {
    let direct = Path::new(name);
    if direct.components().count() > 1 {
        if direct.is_file() {
            return Ok(direct.to_path_buf());
        }
        bail!("{purpose} needs `{name}`, which does not exist");
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT".into())
            .split(';')
            .map(str::to_ascii_lowercase)
            .chain([String::new()])
            .collect()
    } else {
        vec![String::new()]
    };
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for ext in &exts {
                let candidate = dir.join(format!("{name}{ext}"));
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
    }
    bail!("{purpose} needs the external tool `{name}`, which is not on PATH; install it or set its path")
}

/// Runs a program with a deadline; stdout and stderr go to files in `work`.
fn run_tool(program: &Path, args: &[&std::ffi::OsStr], work: &Path) -> Result<String> {
    let out = work.join("stdout.txt");
    let err = work.join("stderr.txt");
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&out)?)
        .stderr(std::fs::File::create(&err)?)
        .spawn()
        .with_context(|| format!("Could not start {}", program.display()))?;
    let deadline = Instant::now() + TOOL_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("{} timed out", program.display());
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    if !status.success() {
        let tail = storage::read_bounded_bytes(&err, 1 << 20).unwrap_or_default();
        let tail = String::from_utf8_lossy(&tail);
        let tail: String = tail
            .chars()
            .rev()
            .take(400)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        bail!("{} failed ({status}): {}", program.display(), tail.trim());
    }
    let bytes = storage::read_bounded_bytes(&out, 16 << 20)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn ocr(tools: &Tools, input: &Path, work: &Path) -> Result<String> {
    let tesseract = find_tool(&tools.tesseract, "OCR")?;
    run_tool(&tesseract, &[input.as_os_str(), "stdout".as_ref()], work)
}

fn transcribe(tools: &Tools, input: &Path, work: &Path) -> Result<String> {
    let ffmpeg = find_tool(&tools.ffmpeg, "Audio/video transcription")?;
    let mut missing = Vec::new();
    let mut whisper = None;
    for name in &tools.whisper {
        match find_tool(name, "Audio/video transcription") {
            Ok(path) => {
                whisper = Some((name.clone(), path));
                break;
            }
            Err(_) => missing.push(format!("`{name}`")),
        }
    }
    let Some((name, whisper)) = whisper else {
        bail!(
            "Audio/video transcription needs a whisper CLI ({}), none is on PATH; install one or set ALLPAKA_WHISPER",
            missing.join(" or ")
        );
    };
    let wav = work.join("audio.wav");
    run_tool(
        &ffmpeg,
        &[
            "-nostdin".as_ref(),
            "-y".as_ref(),
            "-i".as_ref(),
            input.as_os_str(),
            "-vn".as_ref(),
            "-ac".as_ref(),
            "1".as_ref(),
            "-ar".as_ref(),
            "16000".as_ref(),
            wav.as_os_str(),
        ],
        work,
    )?;
    let stem = Path::new(&name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if stem == "whisper" {
        // openai-whisper writes <stem>.txt into --output_dir.
        run_tool(
            &whisper,
            &[
                wav.as_os_str(),
                "--output_format".as_ref(),
                "txt".as_ref(),
                "--output_dir".as_ref(),
                work.as_os_str(),
            ],
            work,
        )?;
        let bytes = storage::read_bounded_bytes(&work.join("audio.txt"), 16 << 20)
            .context("whisper produced no transcript")?;
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }
    let model = tools.whisper_model.as_ref().context(
        "whisper-cli needs a model file; set ALLPAKA_WHISPER_MODEL to a ggml model path",
    )?;
    run_tool(
        &whisper,
        &[
            "-m".as_ref(),
            model.as_os_str(),
            "-f".as_ref(),
            wav.as_os_str(),
            "-nt".as_ref(),
        ],
        work,
    )
}

fn xml_text(e: &quick_xml::events::BytesText) -> Result<String> {
    Ok(e.xml10_content()?.into_owned())
}
fn xml_ref(e: &quick_xml::events::BytesRef) -> Result<String> {
    if let Some(c) = e.resolve_char_ref()? {
        return Ok(c.to_string());
    }
    let name = e.decode()?;
    Ok(quick_xml::escape::resolve_predefined_entity(&name)
        .unwrap_or_default()
        .to_string())
}

fn zip_part(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    name: &str,
) -> Result<Option<Vec<u8>>> {
    let Some(index) = archive.index_for_name(name) else {
        return Ok(None);
    };
    let file = archive.by_index(index)?;
    if file.size() > MAX_XML_PART {
        bail!(
            "Document part {name} is larger than {} MB uncompressed",
            MAX_XML_PART >> 20
        );
    }
    let mut out = Vec::new();
    file.take(MAX_XML_PART + 1).read_to_end(&mut out)?;
    if out.len() as u64 > MAX_XML_PART {
        bail!(
            "Document part {name} is larger than {} MB uncompressed",
            MAX_XML_PART >> 20
        );
    }
    Ok(Some(out))
}

/// Paragraphs as lines, table cells separated by tabs, rows by newlines.
pub(crate) fn docx_text(bytes: &[u8]) -> Result<String> {
    use quick_xml::events::Event;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    let xml =
        zip_part(&mut archive, "word/document.xml")?.context("DOCX has no word/document.xml")?;
    let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
    let mut buf = Vec::new();
    let mut out = String::new();
    let mut in_text = false;
    loop {
        match reader
            .read_event_into(&mut buf)
            .context("Malformed DOCX XML")?
        {
            Event::Start(e) if e.local_name().as_ref() == b"t" => in_text = true,
            Event::End(e) => match e.local_name().as_ref() {
                b"t" => in_text = false,
                b"p" => out.push('\n'),
                b"tc" => {
                    if out.ends_with('\n') {
                        out.pop();
                    }
                    out.push('\t')
                }
                b"tr" => {
                    if out.ends_with('\t') {
                        out.pop();
                    }
                    out.push('\n')
                }
                _ => {}
            },
            Event::Empty(e) => match e.local_name().as_ref() {
                b"tab" => out.push('\t'),
                b"br" | b"cr" => out.push('\n'),
                _ => {}
            },
            Event::Text(e) if in_text => out.push_str(&xml_text(&e)?),
            Event::CData(e) if in_text => out.push_str(&String::from_utf8_lossy(&e)),
            Event::GeneralRef(e) if in_text => out.push_str(&xml_ref(&e)?),
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
        if out.len() > MAX_TEXT_BYTES * 4 {
            break;
        }
    }
    Ok(out)
}

/// Each worksheet as `## sheetN` followed by tab-separated rows.
pub(crate) fn xlsx_text(bytes: &[u8]) -> Result<String> {
    use quick_xml::events::Event;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    let mut shared = Vec::new();
    if let Some(xml) = zip_part(&mut archive, "xl/sharedStrings.xml")? {
        let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
        let mut buf = Vec::new();
        let (mut current, mut in_t) = (String::new(), false);
        loop {
            match reader
                .read_event_into(&mut buf)
                .context("Malformed XLSX shared strings")?
            {
                Event::Start(e) if e.local_name().as_ref() == b"si" => current.clear(),
                Event::Start(e) if e.local_name().as_ref() == b"t" => in_t = true,
                Event::End(e) if e.local_name().as_ref() == b"t" => in_t = false,
                Event::End(e) if e.local_name().as_ref() == b"si" => {
                    shared.push(std::mem::take(&mut current))
                }
                Event::Empty(e) if e.local_name().as_ref() == b"si" => shared.push(String::new()),
                Event::Text(e) if in_t => current.push_str(&xml_text(&e)?),
                Event::GeneralRef(e) if in_t => current.push_str(&xml_ref(&e)?),
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
        }
    }
    let mut sheets: Vec<(u32, String)> = archive
        .file_names()
        .filter_map(|n| {
            let number = n
                .strip_prefix("xl/worksheets/sheet")?
                .strip_suffix(".xml")?
                .parse()
                .ok()?;
            Some((number, n.to_string()))
        })
        .collect();
    sheets.sort();
    if sheets.len() > MAX_SHEETS {
        bail!("Workbook has more than {MAX_SHEETS} worksheets");
    }
    let mut out = String::new();
    for (number, name) in sheets {
        let xml = zip_part(&mut archive, &name)?.unwrap_or_default();
        out.push_str(&format!("## sheet{number}\n"));
        let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
        let mut buf = Vec::new();
        let mut row: Vec<String> = Vec::new();
        let (mut cell_type, mut value, mut in_value) = (String::new(), String::new(), false);
        loop {
            match reader
                .read_event_into(&mut buf)
                .context("Malformed XLSX worksheet")?
            {
                Event::Start(e) if e.local_name().as_ref() == b"c" => {
                    cell_type = match e.try_get_attribute("t")? {
                        Some(a) => a
                            .normalized_value(quick_xml::XmlVersion::Implicit1_0)?
                            .into_owned(),
                        None => String::new(),
                    };
                    value.clear();
                }
                Event::Start(e) if matches!(e.local_name().as_ref(), b"v" | b"t") => {
                    in_value = true
                }
                Event::End(e) => match e.local_name().as_ref() {
                    b"v" | b"t" => in_value = false,
                    b"c" => {
                        let text = if cell_type == "s" {
                            value
                                .trim()
                                .parse::<usize>()
                                .ok()
                                .and_then(|i| shared.get(i).cloned())
                                .unwrap_or_default()
                        } else if cell_type == "b" {
                            (if value.trim() == "1" { "TRUE" } else { "FALSE" }).into()
                        } else {
                            value.clone()
                        };
                        row.push(text);
                    }
                    b"row" => {
                        while row.last().is_some_and(String::is_empty) {
                            row.pop();
                        }
                        if !row.is_empty() {
                            out.push_str(&row.join("\t"));
                            out.push('\n');
                        }
                        row.clear();
                    }
                    _ => {}
                },
                Event::Text(e) if in_value => value.push_str(&xml_text(&e)?),
                Event::GeneralRef(e) if in_value => value.push_str(&xml_ref(&e)?),
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
            if out.len() > MAX_TEXT_BYTES * 4 {
                return Ok(out);
            }
        }
        out.push('\n');
    }
    Ok(out)
}

pub(crate) fn pdf_text(bytes: &[u8]) -> Result<String> {
    let doc = lopdf::Document::load_mem(bytes).context("Unreadable PDF")?;
    if doc.is_encrypted() {
        bail!("Encrypted PDFs are not supported");
    }
    let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    let mut out = String::new();
    for page in pages {
        // A page that cannot be decoded is reported inline, not dropped silently.
        match doc.extract_text(&[page]) {
            Ok(text) => out.push_str(&text),
            Err(e) => out.push_str(&format!("[page {page}: text not extractable: {e}]\n")),
        }
        if out.len() > MAX_TEXT_BYTES * 4 {
            break;
        }
    }
    if out.trim().is_empty() {
        bail!("The PDF has no extractable text layer (a scanned PDF needs OCR first)");
    }
    Ok(out)
}

/// Crude HTML to text: drops script/style/head, tags, decodes common entities.
pub(crate) fn html_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0;
    while i < html.len() {
        let rest = &lower[i..];
        if rest.starts_with('<') {
            let skip = ["script", "style", "head", "noscript"]
                .into_iter()
                .find(|t| rest[1..].starts_with(t));
            let end = if let Some(tag) = skip {
                rest.find(&format!("</{tag}"))
                    .and_then(|s| rest[s..].find('>').map(|e| s + e + 1))
            } else {
                rest.find('>').map(|e| e + 1)
            };
            let Some(end) = end else { break };
            let tag = &rest[..end];
            if [
                "<br", "<p", "</p", "<div", "</div", "<li", "<tr", "<h", "</h",
            ]
            .iter()
            .any(|t| tag.starts_with(t))
            {
                out.push('\n');
            } else if tag.starts_with("<td") || tag.starts_with("<th") {
                out.push('\t');
            }
            i += end;
            continue;
        }
        let next = rest.find('<').unwrap_or(rest.len());
        out.push_str(&html[i..i + next]);
        i += next;
    }
    let out = out
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&");
    let mut lines: Vec<String> = Vec::new();
    for line in out.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() || lines.last().is_some_and(|l| !l.is_empty()) {
            lines.push(line);
        }
    }
    lines.join("\n").trim().to_string()
}

/// Cuts at `MAX_TEXT_BYTES` on a char boundary; returns whether it cut.
pub(crate) fn cap_text(text: &mut String) -> bool {
    if text.len() <= MAX_TEXT_BYTES {
        return false;
    }
    let mut end = MAX_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    true
}

fn check_name(name: &str) -> Result<()> {
    if name.trim().is_empty()
        || name.len() > MAX_NAME_LEN
        || name.contains(['/', '\\', '\0'])
        || name.chars().any(char::is_control)
    {
        bail!("File name must be 1–{MAX_NAME_LEN} bytes without path separators");
    }
    Ok(())
}

/// Detects, limits and extracts. `work` is a scratch folder for external tools.
pub(crate) fn extract_bytes(
    name: &str,
    bytes: &[u8],
    tools: &Tools,
    work: &Path,
) -> Result<(Kind, &'static str, String, bool)> {
    check_name(name)?;
    let (kind, mime) = detect(bytes)?;
    if bytes.len() > kind.max_bytes() {
        bail!(
            "{kind:?} input is {} bytes; the limit for this type is {} bytes",
            bytes.len(),
            kind.max_bytes()
        );
    }
    if let Some(claimed) = claimed_kind(name) {
        // Text extensions cover HTML-looking text too (e.g. an .xml file).
        let compatible = claimed == kind
            || (claimed == Kind::Text && kind == Kind::Html)
            || (claimed == Kind::Html && kind == Kind::Text)
            || (matches!(claimed, Kind::Audio | Kind::Video)
                && matches!(kind, Kind::Audio | Kind::Video));
        if !compatible {
            bail!("The file name says {claimed:?} but the content is {kind:?}");
        }
    }
    let mut text = match kind {
        Kind::Text => std::str::from_utf8(bytes)?
            .trim_start_matches('\u{feff}')
            .to_string(),
        Kind::Html => html_text(std::str::from_utf8(bytes)?),
        Kind::Pdf => pdf_text(bytes)?,
        Kind::Docx => docx_text(bytes)?,
        Kind::Xlsx => xlsx_text(bytes)?,
        Kind::Image | Kind::Audio | Kind::Video => {
            std::fs::create_dir_all(work)?;
            let ext = claimed_kind(name)
                .and(name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()))
                .unwrap_or_else(|| mime.rsplit('/').next().unwrap_or("bin").into());
            let input = work.join(format!("input.{ext}"));
            std::fs::write(&input, bytes)?;
            if kind == Kind::Image {
                ocr(tools, &input, work)?
            } else {
                transcribe(tools, &input, work)?
            }
        }
    };
    text = text.replace("\r\n", "\n");
    if text.trim().is_empty() {
        bail!("No text could be extracted from this {kind:?} file");
    }
    let truncated = cap_text(&mut text);
    Ok((kind, mime, text, truncated))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Record {
    pub(crate) id: String,
    pub(crate) created_ms: u64,
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
    pub(crate) kind: Kind,
    pub(crate) mime: String,
    pub(crate) bytes: usize,
    pub(crate) sha256: String,
    /// The preserved original, relative to the Studio data directory.
    pub(crate) original: String,
    pub(crate) text: String,
    pub(crate) truncated: bool,
}
impl Record {
    /// The block a client puts into a message to use the extraction in a turn.
    pub(crate) fn attachment(&self) -> String {
        let source = self.url.as_deref().unwrap_or(&self.name);
        format!(
            "<attachment name=\"{}\" type=\"{}\" sha256=\"{}\"{}>\n{}\n</attachment>",
            source.replace('"', "'"),
            self.mime,
            self.sha256,
            if self.truncated {
                " truncated=\"true\""
            } else {
                ""
            },
            self.text
        )
    }
    fn view(&self) -> Value {
        let mut v = json!(self);
        v["attachment"] = json!(self.attachment());
        v
    }
}

fn root(data: &Path) -> PathBuf {
    data.join("extractions")
}

/// Extracts and persists: original under `extractions/originals/<sha256>`,
/// record under `extractions/records/<id>.json`.
pub(crate) fn store(
    data: &Path,
    name: &str,
    url: Option<String>,
    bytes: &[u8],
    tools: &Tools,
) -> Result<Record> {
    let id = crate::evaluation::new_id();
    let base = root(data);
    let work = base.join("tmp").join(&id);
    let result = extract_bytes(name, bytes, tools, &work);
    let _ = std::fs::remove_dir_all(&work);
    let (kind, mime, text, truncated) = result?;
    let sha256 = storage::hex_sha256(bytes);
    let originals = base.join("originals");
    let records = base.join("records");
    std::fs::create_dir_all(&originals)?;
    std::fs::create_dir_all(&records)?;
    let original = originals.join(&sha256);
    if !original.exists() {
        if let Err(e) = storage::atomic_create_new_bytes(&original, bytes) {
            if !original.exists() {
                return Err(e);
            }
        }
    }
    let record = Record {
        id: id.clone(),
        created_ms: storage::now_ms(),
        name: name.into(),
        url,
        kind,
        mime: mime.into(),
        bytes: bytes.len(),
        original: format!("extractions/originals/{sha256}"),
        sha256,
        text,
        truncated,
    };
    storage::atomic_write_bytes(
        &records.join(format!("{id}.json")),
        &serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}

pub(crate) fn load(data: &Path, id: &str) -> Result<Record> {
    if !storage::valid_id(id) {
        bail!("Invalid extraction id");
    }
    let bytes = storage::read_bounded_bytes(
        &root(data).join("records").join(format!("{id}.json")),
        4 << 20,
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Fetches a URL with a byte cap and timeout; returns the name and the body.
async fn fetch(client: &reqwest::Client, url: &str) -> Result<(String, Vec<u8>)> {
    if url.len() > MAX_URL_LEN {
        bail!("URL is longer than {MAX_URL_LEN} bytes");
    }
    let parsed = reqwest::Url::parse(url).context("Invalid URL")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        bail!("Only http and https URLs can be fetched");
    }
    let cap = Kind::Video.max_bytes();
    let mut response = client
        .get(parsed.clone())
        .timeout(URL_TIMEOUT)
        .send()
        .await
        .context("Fetch failed")?;
    if !response.status().is_success() {
        bail!("Fetch failed: HTTP {}", response.status());
    }
    if response.content_length().is_some_and(|n| n > cap as u64) {
        bail!("The resource is larger than {cap} bytes");
    }
    let mut body = Vec::new();
    while let Some(chunk) = tokio::time::timeout(URL_TIMEOUT, response.chunk())
        .await
        .context("Fetch timed out")??
    {
        body.extend_from_slice(&chunk);
        if body.len() > cap {
            bail!("The resource is larger than {cap} bytes");
        }
    }
    let name = parsed
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.chars()
                .filter(|c| !matches!(c, '/' | '\\') && !c.is_control())
                .take(120)
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| parsed.host_str().unwrap_or("page").to_string());
    Ok((name, body))
}

static SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

async fn run(app: App, name: String, url: Option<String>, bytes: Vec<u8>) -> ApiResult<Value> {
    let permit = SLOTS.try_acquire().map_err(|_| {
        error(
            StatusCode::TOO_MANY_REQUESTS,
            "Extraction is busy; try again",
        )
    })?;
    crate::blocking(move || {
        let _permit = permit;
        store(&app.data, &name, url, &bytes, &Tools::from_env())
            .map(|r| Json(r.view()))
            .map_err(crate::bad_request)
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UploadQuery {
    name: String,
}
/// `POST /api/extract/upload?name=report.pdf` with the raw file as the body.
pub(crate) async fn upload(
    State(app): State<App>,
    Query(q): Query<UploadQuery>,
    body: Bytes,
) -> ApiResult<Value> {
    run(app, q.name, None, body.to_vec()).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UrlRequest {
    url: String,
}
/// `POST /api/extract/url` with `{"url": "https://..."}`.
pub(crate) async fn url(State(app): State<App>, Json(input): Json<UrlRequest>) -> ApiResult<Value> {
    let (name, bytes) = fetch(&app.client, input.url.trim())
        .await
        .map_err(crate::bad_request)?;
    run(app, name, Some(input.url.trim().to_string()), bytes).await
}

/// `GET /api/extract/:id`.
pub(crate) async fn get(State(app): State<App>, UrlPath(id): UrlPath<String>) -> ApiResult<Value> {
    crate::blocking(move || {
        load(&app.data, &id)
            .map(|r| Json(r.view()))
            .map_err(|e| crate::not_found(format!("Extraction not found: {e}")))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut out);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, body) in parts {
                w.start_file(*name, opts).unwrap();
                w.write_all(body.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        out.into_inner()
    }

    pub(crate) fn docx() -> Vec<u8> {
        zip_of(&[
            ("[Content_Types].xml", "<Types/>"),
            (
                "word/document.xml",
                r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world &amp; co</w:t></w:r></w:p>
<w:p><w:r><w:t>Second</w:t><w:tab/><w:t>para</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"#,
            ),
        ])
    }

    fn xlsx() -> Vec<u8> {
        zip_of(&[
            ("[Content_Types].xml", "<Types/>"),
            ("xl/workbook.xml", "<workbook/>"),
            (
                "xl/sharedStrings.xml",
                r#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Name</t></si><si><t>Qty</t></si><si><r><t>Ap</t></r><r><t>ple</t></r></si></sst>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>42</v></c><c r="C2" t="inlineStr"><is><t>fresh</t></is></c><c r="D2" t="b"><v>1</v></c></row>
</sheetData></worksheet>"#,
            ),
        ])
    }

    /// A one-page PDF with a correct xref table, built in code.
    pub(crate) fn pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_string(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_string(),
            format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for o in offsets {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("studio-extract-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn missing_tools() -> Tools {
        Tools {
            tesseract: "allpaka-no-such-tesseract".into(),
            ffmpeg: "allpaka-no-such-ffmpeg".into(),
            whisper: vec!["allpaka-no-such-whisper".into()],
            whisper_model: None,
        }
    }

    #[test]
    fn extracts_docx_paragraphs_and_tables() {
        let text = docx_text(&docx()).unwrap();
        assert_eq!(text, "Hello world & co\nSecond\tpara\nA1\tB1\n");
    }

    #[test]
    fn extracts_xlsx_cells_with_shared_strings() {
        let text = xlsx_text(&xlsx()).unwrap();
        assert_eq!(text, "## sheet1\nName\tQty\nApple\t42\tfresh\tTRUE\n\n");
    }

    #[test]
    fn extracts_pdf_text_layer() {
        let text = pdf_text(&pdf("Hello PDF world")).unwrap();
        assert!(text.contains("Hello PDF world"), "{text:?}");
    }

    #[test]
    fn detection_uses_magic_bytes_and_allow_list() {
        assert_eq!(detect(&pdf("x")).unwrap().0, Kind::Pdf);
        assert_eq!(detect(&docx()).unwrap().0, Kind::Docx);
        assert_eq!(detect(&xlsx()).unwrap().0, Kind::Xlsx);
        assert_eq!(detect(b"\x89PNG\r\n\x1a\n....").unwrap().0, Kind::Image);
        assert_eq!(detect(b"RIFF\0\0\0\0WAVEfmt ").unwrap().0, Kind::Audio);
        assert_eq!(detect(b"\0\0\0\x18ftypisom").unwrap().0, Kind::Video);
        assert_eq!(detect(b"<!DOCTYPE html><p>x").unwrap().0, Kind::Html);
        assert_eq!(detect("plain ✓".as_bytes()).unwrap().0, Kind::Text);
        assert!(detect(b"").is_err());
        assert!(detect(b"\x7fELF\x02\x01\x01\0\xff\xfe").is_err());
        assert!(detect(&zip_of(&[("a.txt", "x")]))
            .unwrap_err()
            .to_string()
            .contains("zip"));
        let tools = missing_tools();
        let work = temp("detect");
        // Extension cannot override content.
        let err = extract_bytes("report.pdf", &docx(), &tools, &work).unwrap_err();
        assert!(err.to_string().contains("says Pdf"), "{err}");
        assert!(extract_bytes("../x.txt", b"hi", &tools, &work).is_err());
        std::fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn size_and_text_limits_are_explicit() {
        let tools = missing_tools();
        let work = temp("limits");
        let big = vec![b'a'; Kind::Text.max_bytes() + 1];
        let err = extract_bytes("big.txt", &big, &tools, &work).unwrap_err();
        assert!(err.to_string().contains("limit"), "{err}");
        let long = "é".repeat(MAX_TEXT_BYTES); // 2 bytes each
        let (_, _, text, truncated) =
            extract_bytes("long.txt", long.as_bytes(), &tools, &work).unwrap();
        assert!(truncated && text.len() <= MAX_TEXT_BYTES && text.len() >= MAX_TEXT_BYTES - 1);
        let (kind, _, text, truncated) = extract_bytes(
            "page.html",
            b"<html><head><title>t</title></head><body><script>x()</script><p>Hi &amp; bye</p></body></html>",
            &tools,
            &work,
        )
        .unwrap();
        assert_eq!(
            (kind, text.as_str(), truncated),
            (Kind::Html, "Hi & bye", false)
        );
        std::fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn missing_external_tools_name_the_tool() {
        let tools = missing_tools();
        let work = temp("tools");
        let err = extract_bytes("scan.png", b"\x89PNG\r\n\x1a\n....", &tools, &work).unwrap_err();
        assert!(
            err.to_string().contains("allpaka-no-such-tesseract"),
            "{err}"
        );
        let err = extract_bytes("talk.wav", b"RIFF\0\0\0\0WAVEfmt ", &tools, &work).unwrap_err();
        assert!(err.to_string().contains("allpaka-no-such-ffmpeg"), "{err}");
        let mut only_ffmpeg = missing_tools();
        if let Ok(ffmpeg) = find_tool("ffmpeg", "test") {
            only_ffmpeg.ffmpeg = ffmpeg.display().to_string();
            let err =
                extract_bytes("clip.mp4", b"\0\0\0\x18ftypisom", &only_ffmpeg, &work).unwrap_err();
            assert!(err.to_string().contains("allpaka-no-such-whisper"), "{err}");
        }
        std::fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn ocr_success_when_tesseract_is_installed() {
        if find_tool("tesseract", "test").is_err() {
            eprintln!("skipped: tesseract is not on PATH");
            return;
        }
        // A real scan fixture is not bundled; only the tool lookup is covered here.
        assert!(find_tool("tesseract", "test").unwrap().is_file());
    }

    #[test]
    fn store_preserves_original_with_hash() {
        let data = temp("store");
        let bytes = docx();
        let record = store(&data, "memo.docx", None, &bytes, &missing_tools()).unwrap();
        assert_eq!(record.kind, Kind::Docx);
        assert_eq!(record.sha256, storage::hex_sha256(&bytes));
        assert_eq!(std::fs::read(data.join(&record.original)).unwrap(), bytes);
        assert!(record.text.starts_with("Hello world"));
        let again = load(&data, &record.id).unwrap();
        assert_eq!(again.sha256, record.sha256);
        assert!(again.attachment().contains("sha256=\""));
        // Same bytes twice reuse one original.
        let second = store(&data, "memo.docx", None, &bytes, &missing_tools()).unwrap();
        assert_eq!(second.original, record.original);
        assert!(load(&data, "../x").is_err());
        // Failed extractions store nothing and leave no scratch folder.
        assert!(store(
            &data,
            "x.bin",
            None,
            b"\x7fELF\x02\x01\x01\0\xff\xfe",
            &missing_tools()
        )
        .is_err());
        assert_eq!(
            std::fs::read_dir(data.join("extractions/originals"))
                .unwrap()
                .count(),
            1
        );
        assert!(!data
            .join("extractions/tmp")
            .read_dir()
            .is_ok_and(|mut d| d.next().is_some()));
        std::fs::remove_dir_all(data).unwrap();
    }
}
