//! Pure-Rust document extraction (`extract_document` tool & `/api/documents/extract`)
//! supporting PDF (BT..ET + FlateDecode), DOCX, XLSX, PPTX, ODT/ODS/ODP, and text formats,
//! backed by a built-in bounded RFC 1951 DEFLATE decompressor and ZIP reader (<= 32 MiB cap).

use crate::ext::{get_session, project_for_session, resolve_project_root};
use crate::{bad_request, not_found, spill, storage, tools, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

const MAX_DECOMPRESSED_BYTES: usize = 32 * 1024 * 1024;
const MAX_UPLOAD_BYTES: usize = 8 * 1024 * 1024;
const MAX_ZIP_ENTRIES: usize = 2048;
const DEFAULT_EXTRACT_CHARS: usize = 32 * 1024;
const MAX_EXTRACT_CHARS: usize = 128 * 1024;

pub(crate) fn schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "extract_document",
            "description": "Extract readable text from workspace documents (.pdf, .docx, .xlsx, .pptx, .odt, .ods, .odp, .html, .xml, .md, .txt, .csv) with pure-Rust DEFLATE/ZIP/PDF decoding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative path to the document" },
                    "max_chars": { "type": "integer", "minimum": 256, "maximum": 131072 }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }
    })
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_cnt: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    fn ensure_bits(&mut self, n: u8) -> Result<()> {
        while self.bit_cnt < n {
            if self.byte_pos >= self.data.len() {
                bail!("Unexpected end of DEFLATE stream");
            }
            self.bit_buf |= (self.data[self.byte_pos] as u32) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure_bits(n)?;
        let val = self.bit_buf & ((1u32 << n) - 1);
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt % 8;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_aligned_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.align_byte();
        let buffered_bytes = (self.bit_cnt / 8) as usize;
        self.byte_pos -= buffered_bytes;
        self.bit_buf = 0;
        self.bit_cnt = 0;
        let end = self
            .byte_pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .context("Truncated DEFLATE stored block")?;
        let slice = &self.data[self.byte_pos..end];
        self.byte_pos = end;
        Ok(slice)
    }
}

struct HuffTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &len in lengths {
            if len > 15 {
                bail!("Invalid Huffman code length {len}");
            }
            counts[len as usize] += 1;
        }
        if counts[0] as usize == lengths.len() {
            return Ok(Self {
                counts,
                symbols: Vec::new(),
            });
        }
        let mut left = 1i32;
        for &cnt in counts.iter().skip(1) {
            left <<= 1;
            left -= cnt as i32;
            if left < 0 {
                bail!("Over-subscribed Huffman code tree");
            }
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let non_zero = lengths.iter().filter(|&&l| l > 0).count();
        let mut symbols = vec![0u16; non_zero];
        for (sym, &len) in lengths.iter().enumerate() {
            if len != 0 {
                let idx = offsets[len as usize] as usize;
                offsets[len as usize] += 1;
                symbols[idx] = sym as u16;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        if self.symbols.is_empty() {
            bail!("Attempted to decode from empty Huffman alphabet");
        }
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0usize;
        for len in 1..=15 {
            code |= reader.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                let pos = index + (code - first) as usize;
                return self
                    .symbols
                    .get(pos)
                    .copied()
                    .context("Huffman symbol lookup out of bounds");
            }
            index += count as usize;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        bail!("Invalid Huffman code in DEFLATE stream");
    }
}

fn fixed_trees() -> Result<(HuffTree, HuffTree)> {
    let mut lit_lengths = [0u8; 288];
    lit_lengths[0..=143].fill(8);
    lit_lengths[144..=255].fill(9);
    lit_lengths[256..=279].fill(7);
    lit_lengths[280..=287].fill(8);
    let dist_lengths = [5u8; 32];
    Ok((
        HuffTree::from_lengths(&lit_lengths)?,
        HuffTree::from_lengths(&dist_lengths)?,
    ))
}

fn dynamic_trees(reader: &mut BitReader<'_>) -> Result<(HuffTree, HuffTree)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let nlit = reader.read_bits(5)? as usize + 257;
    let ndist = reader.read_bits(5)? as usize + 1;
    let ncode = reader.read_bits(4)? as usize + 4;
    if nlit > 286 || ndist > 30 {
        bail!("Invalid dynamic Huffman header counts ({nlit}, {ndist})");
    }
    let mut cl_lengths = [0u8; 19];
    for &idx in ORDER.iter().take(ncode) {
        cl_lengths[idx] = reader.read_bits(3)? as u8;
    }
    let cl_tree = HuffTree::from_lengths(&cl_lengths)?;

    let total = nlit + ndist;
    let mut lengths = Vec::with_capacity(total);
    while lengths.len() < total {
        let sym = cl_tree.decode(reader)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths.last().context("Code-length repeat 16 at start")?;
                let rep = reader.read_bits(2)? as usize + 3;
                if lengths.len() + rep > total {
                    bail!("Code-length repeat 16 overflows table");
                }
                lengths.resize(lengths.len() + rep, prev);
            }
            17 => {
                let rep = reader.read_bits(3)? as usize + 3;
                if lengths.len() + rep > total {
                    bail!("Code-length repeat 17 overflows table");
                }
                lengths.resize(lengths.len() + rep, 0);
            }
            18 => {
                let rep = reader.read_bits(7)? as usize + 11;
                if lengths.len() + rep > total {
                    bail!("Code-length repeat 18 overflows table");
                }
                lengths.resize(lengths.len() + rep, 0);
            }
            _ => bail!("Invalid code-length symbol {sym}"),
        }
    }
    Ok((
        HuffTree::from_lengths(&lengths[..nlit])?,
        HuffTree::from_lengths(&lengths[nlit..])?,
    ))
}

/// Pure-Rust bounded RFC 1951 DEFLATE decompressor.
pub(crate) fn inflate_raw(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    let mut reader = BitReader::new(src);
    let mut out = Vec::new();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;
        match btype {
            0 => {
                let hdr = reader.read_aligned_bytes(4)?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]);
                let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
                if len != !nlen {
                    bail!("Corrupt stored DEFLATE block length");
                }
                if out.len() + len as usize > max_out {
                    bail!("Decompressed output exceeds {max_out}-byte safety limit");
                }
                let raw = reader.read_aligned_bytes(len as usize)?;
                out.extend_from_slice(raw);
            }
            1 | 2 => {
                let (lit_tree, dist_tree) = if btype == 1 {
                    fixed_trees()?
                } else {
                    dynamic_trees(&mut reader)?
                };
                loop {
                    let sym = lit_tree.decode(&mut reader)?;
                    if sym < 256 {
                        if out.len() + 1 > max_out {
                            bail!("Decompressed output exceeds {max_out}-byte safety limit");
                        }
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let idx = (sym as usize)
                            .checked_sub(257)
                            .context("Invalid length symbol")?;
                        if idx >= LEN_BASE.len() {
                            bail!("Invalid length symbol {sym}");
                        }
                        let match_len =
                            LEN_BASE[idx] as usize + reader.read_bits(LEN_EXTRA[idx])? as usize;
                        let d_sym = dist_tree.decode(&mut reader)? as usize;
                        if d_sym >= DIST_BASE.len() {
                            bail!("Invalid distance symbol {d_sym}");
                        }
                        let dist = DIST_BASE[d_sym] as usize
                            + reader.read_bits(DIST_EXTRA[d_sym])? as usize;
                        if dist == 0 || dist > out.len() {
                            bail!("DEFLATE back-reference distance out of range");
                        }
                        if out.len() + match_len > max_out {
                            bail!("Decompressed output exceeds {max_out}-byte safety limit");
                        }
                        for _ in 0..match_len {
                            let b = out[out.len() - dist];
                            out.push(b);
                        }
                    }
                }
            }
            _ => bail!("Reserved DEFLATE block type 3"),
        }
        if bfinal == 1 {
            break;
        }
    }
    Ok(out)
}

pub(crate) fn zlib_decompress(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    if src.len() < 2 {
        bail!("Truncated zlib header");
    }
    let cmf = src[0];
    let flg = src[1];
    if ((cmf as u16) * 256 + flg as u16) % 31 != 0 || (cmf & 0x0F) != 8 {
        bail!("Invalid zlib header");
    }
    let mut start = 2usize;
    if (flg & 0x20) != 0 {
        if src.len() < 6 {
            bail!("Truncated zlib preset dictionary");
        }
        start += 4;
    }
    inflate_raw(&src[start..], max_out)
}

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_zip_entries(archive: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    if archive.len() < 22 {
        bail!("Archive too small to be a ZIP file");
    }
    let min_eocd = archive.len().saturating_sub(65535 + 22);
    let mut eocd_pos = None;
    for pos in (min_eocd..=archive.len() - 22).rev() {
        if &archive[pos..pos + 4] == b"PK\x05\x06" {
            let cmt_len = le16(&archive[pos + 20..pos + 22]) as usize;
            if pos + 22 + cmt_len == archive.len() {
                eocd_pos = Some(pos);
                break;
            }
        }
    }
    let eocd = eocd_pos.context("ZIP end-of-central-directory signature not found")?;
    let total_entries = le16(&archive[eocd + 10..eocd + 12]) as usize;
    let cd_size = le32(&archive[eocd + 12..eocd + 16]) as usize;
    let cd_offset = le32(&archive[eocd + 16..eocd + 20]) as usize;
    if total_entries > MAX_ZIP_ENTRIES {
        bail!("ZIP archive exceeds {MAX_ZIP_ENTRIES}-entry limit");
    }
    if cd_offset
        .checked_add(cd_size)
        .is_none_or(|end| end > archive.len())
    {
        bail!("ZIP central directory out of bounds");
    }

    let mut entries = BTreeMap::new();
    let mut cur = cd_offset;
    let mut total_uncompressed = 0usize;

    for _ in 0..total_entries {
        if cur + 46 > archive.len() || &archive[cur..cur + 4] != b"PK\x01\x02" {
            bail!("Corrupt ZIP central directory entry");
        }
        let method = le16(&archive[cur + 10..cur + 12]);
        let comp_size = le32(&archive[cur + 20..cur + 24]) as usize;
        let uncomp_size = le32(&archive[cur + 24..cur + 28]) as usize;
        let name_len = le16(&archive[cur + 28..cur + 30]) as usize;
        let extra_len = le16(&archive[cur + 30..cur + 32]) as usize;
        let cmt_len = le16(&archive[cur + 32..cur + 34]) as usize;
        let local_off = le32(&archive[cur + 42..cur + 46]) as usize;

        if cur + 46 + name_len + extra_len + cmt_len > archive.len() {
            bail!("ZIP central directory entry truncated");
        }
        let name = String::from_utf8_lossy(&archive[cur + 46..cur + 46 + name_len]).into_owned();
        cur += 46 + name_len + extra_len + cmt_len;

        if name.ends_with('/') {
            continue;
        }
        if uncomp_size > MAX_DECOMPRESSED_BYTES
            || total_uncompressed.saturating_add(uncomp_size) > MAX_DECOMPRESSED_BYTES
        {
            bail!("ZIP archive exceeds {MAX_DECOMPRESSED_BYTES}-byte decompression budget");
        }
        if local_off + 30 > archive.len() || &archive[local_off..local_off + 4] != b"PK\x03\x04" {
            bail!("Corrupt ZIP local file header for `{name}`");
        }
        let l_name_len = le16(&archive[local_off + 26..local_off + 28]) as usize;
        let l_extra_len = le16(&archive[local_off + 28..local_off + 30]) as usize;
        let data_start = local_off + 30 + l_name_len + l_extra_len;
        let data_end = data_start
            .checked_add(comp_size)
            .filter(|&e| e <= archive.len())
            .context("ZIP entry payload out of bounds")?;
        let payload = &archive[data_start..data_end];
        let remaining_budget = MAX_DECOMPRESSED_BYTES - total_uncompressed;

        let decompressed = match method {
            0 => {
                if payload.len() > remaining_budget {
                    bail!("ZIP stored entry exceeds decompression limit");
                }
                payload.to_vec()
            }
            8 => inflate_raw(payload, remaining_budget)?,
            other => bail!("Unsupported ZIP compression method {other} in `{name}`"),
        };
        total_uncompressed += decompressed.len();
        entries.insert(name, decompressed);
    }

    Ok(entries)
}

fn decode_xml_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn strip_xml_tags(xml: &str, block_breaks: &[&str], tab_tags: &[&str]) -> String {
    let mut out = String::with_capacity(xml.len() / 2);
    let mut in_tag = false;
    let mut tag_buf = String::new();
    for ch in xml.chars() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let trimmed = tag_buf.trim();
                let name = trimmed
                    .trim_start_matches('/')
                    .trim_end_matches('/')
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                if tab_tags.contains(&name) {
                    out.push('\t');
                } else if trimmed.starts_with('/')
                    && block_breaks.contains(&name)
                    && !out.ends_with('\n')
                {
                    out.push('\n');
                }
                tag_buf.clear();
            } else if tag_buf.len() < 96 {
                tag_buf.push(ch);
            }
        } else if ch == '<' {
            in_tag = true;
            tag_buf.clear();
        } else {
            out.push(ch);
        }
    }
    let decoded = decode_xml_entities(&out);
    decoded
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_xml_tag_texts(xml: &str, open_tag: &str, close_tag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = xml[pos..].find(open_tag) {
        let s = pos + rel;
        let Some(gt) = xml[s..].find('>') else { break };
        let body_start = s + gt + 1;
        let Some(end_rel) = xml[body_start..].find(close_tag) else {
            break;
        };
        out.push(decode_xml_entities(&xml[body_start..body_start + end_rel]));
        pos = body_start + end_rel + close_tag.len();
    }
    out
}

fn extract_docx(entries: &BTreeMap<String, Vec<u8>>) -> String {
    let mut sections = Vec::new();
    let mut keys: Vec<&String> = entries
        .keys()
        .filter(|k| {
            k.as_str() == "word/document.xml"
                || (k.starts_with("word/")
                    && k.ends_with(".xml")
                    && (k.contains("header") || k.contains("footer") || k.contains("footnotes")))
        })
        .collect();
    keys.sort_by_key(|k| (k.as_str() != "word/document.xml", k.as_str()));
    for k in keys {
        if let Some(bytes) = entries.get(k) {
            let xml = String::from_utf8_lossy(bytes);
            let txt = strip_xml_tags(&xml, &["w:p", "w:tr"], &["w:tab", "w:tc"]);
            if !txt.is_empty() {
                sections.push(txt);
            }
        }
    }
    sections.join("\n\n")
}

fn parse_xlsx_shared_strings(bytes: &[u8]) -> Vec<String> {
    let xml = String::from_utf8_lossy(bytes);
    let mut shared = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = xml[pos..].find("<si") {
        let s = pos + rel;
        let Some(end) = xml[s..].find("</si>") else {
            break;
        };
        let block = &xml[s..s + end + 5];
        let pieces = parse_xml_tag_texts(block, "<t", "</t>");
        shared.push(pieces.concat());
        pos = s + end + 5;
    }
    shared
}

fn extract_xlsx(entries: &BTreeMap<String, Vec<u8>>) -> String {
    let shared = entries
        .get("xl/sharedStrings.xml")
        .map(|b| parse_xlsx_shared_strings(b))
        .unwrap_or_default();
    let mut sheets: Vec<&String> = entries
        .keys()
        .filter(|k| k.starts_with("xl/worksheets/sheet") && k.ends_with(".xml"))
        .collect();
    sheets.sort_by_key(|k| natural_number_key(k));

    let mut out = Vec::new();
    for s_key in sheets {
        let xml = String::from_utf8_lossy(&entries[s_key]);
        let mut rows_text = Vec::new();
        let mut r_pos = 0usize;
        while let Some(r_rel) = xml[r_pos..].find("<row") {
            let r_start = r_pos + r_rel;
            let Some(r_end) = xml[r_start..].find("</row>") else {
                break;
            };
            let row_xml = &xml[r_start..r_start + r_end];
            let mut cells = Vec::new();
            let mut c_pos = 0usize;
            while let Some(c_rel) = row_xml[c_pos..].find("<c") {
                let c_start = c_pos + c_rel;
                let Some(c_gt) = row_xml[c_start..].find('>') else {
                    break;
                };
                let c_tag = &row_xml[c_start..=c_start + c_gt];
                let c_end = row_xml[c_start..]
                    .find("</c>")
                    .map(|e| c_start + e + 4)
                    .unwrap_or(c_start + c_gt + 1);
                let cell_xml = &row_xml[c_start..c_end];
                let val = if c_tag.contains("t=\"s\"") {
                    parse_xml_tag_texts(cell_xml, "<v", "</v>")
                        .into_iter()
                        .next()
                        .and_then(|n| n.trim().parse::<usize>().ok())
                        .and_then(|idx| shared.get(idx).cloned())
                        .unwrap_or_default()
                } else if c_tag.contains("t=\"inlineStr\"") {
                    parse_xml_tag_texts(cell_xml, "<t", "</t>").concat()
                } else {
                    parse_xml_tag_texts(cell_xml, "<v", "</v>").concat()
                };
                if !val.is_empty() {
                    cells.push(val);
                }
                c_pos = c_end;
            }
            if !cells.is_empty() {
                rows_text.push(cells.join("\t"));
            }
            r_pos = r_start + r_end + 6;
        }
        if !rows_text.is_empty() {
            out.push(format!("[{s_key}]\n{}", rows_text.join("\n")));
        }
    }
    out.join("\n\n")
}

fn natural_number_key(s: &str) -> (usize, &str) {
    let digits: String = s.chars().filter(char::is_ascii_digit).collect();
    (digits.parse::<usize>().unwrap_or(0), s)
}

fn extract_pptx(entries: &BTreeMap<String, Vec<u8>>) -> String {
    let mut slides: Vec<&String> = entries
        .keys()
        .filter(|k| k.starts_with("ppt/slides/slide") && k.ends_with(".xml"))
        .collect();
    slides.sort_by_key(|k| natural_number_key(k));
    let mut out = Vec::new();
    for (idx, k) in slides.into_iter().enumerate() {
        let xml = String::from_utf8_lossy(&entries[k]);
        let txt = strip_xml_tags(&xml, &["a:p"], &[]);
        if !txt.is_empty() {
            out.push(format!("[Slide {}]\n{}", idx + 1, txt));
        }
    }
    out.join("\n\n")
}

fn extract_odf(entries: &BTreeMap<String, Vec<u8>>) -> String {
    let Some(bytes) = entries.get("content.xml") else {
        return String::new();
    };
    let xml = String::from_utf8_lossy(bytes);
    strip_xml_tags(
        &xml,
        &["text:p", "text:h", "table:table-row"],
        &["table:table-cell", "text:tab"],
    )
}

fn parse_pdf_literal(bytes: &[u8], start: usize) -> (String, usize) {
    let mut out = Vec::new();
    let mut i = start;
    let mut depth = 1usize;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                i += 1;
                match bytes[i] {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0c),
                    b'(' | b')' | b'\\' => out.push(bytes[i]),
                    b'0'..=b'7' => {
                        let mut oct = bytes[i] - b'0';
                        for _ in 0..2 {
                            if i + 1 < bytes.len() && matches!(bytes[i + 1], b'0'..=b'7') {
                                i += 1;
                                oct = oct.wrapping_mul(8).wrapping_add(bytes[i] - b'0');
                            } else {
                                break;
                            }
                        }
                        out.push(oct);
                    }
                    other => out.push(other),
                }
            }
            b'(' => {
                depth += 1;
                out.push(b'(');
            }
            b')' => {
                depth -= 1;
                if depth > 0 {
                    out.push(b')');
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    (decode_pdf_bytes(&out), i)
}

fn parse_pdf_hex(bytes: &[u8], start: usize) -> (String, usize) {
    let mut hex = Vec::new();
    let mut i = start;
    while i < bytes.len() && bytes[i] != b'>' {
        if bytes[i].is_ascii_hexdigit() {
            hex.push(bytes[i]);
        }
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'>' {
        i += 1;
    }
    if hex.len() % 2 == 1 {
        hex.push(b'0');
    }
    let decoded: Vec<u8> = hex
        .chunks_exact(2)
        .filter_map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|s| u8::from_str_radix(s, 16).ok())
        })
        .collect();
    (decode_pdf_bytes(&decoded), i)
}

fn decode_pdf_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xFE, 0xFF]) && bytes.len() >= 4 {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    bytes
        .iter()
        .map(|&b| {
            if b == b'\n' || b == b'\r' || b == b'\t' || (0x20..0x7f).contains(&b) {
                b as char
            } else if b >= 0x80 {
                char::from(b)
            } else {
                ' '
            }
        })
        .collect()
}

fn extract_pdf_text_operators(stream: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    let mut in_text = false;
    while i < stream.len() {
        if stream[i..].starts_with(b"BT")
            && (i == 0 || stream[i - 1].is_ascii_whitespace())
            && stream.get(i + 2).is_none_or(|b| b.is_ascii_whitespace())
        {
            in_text = true;
            i += 2;
            continue;
        }
        if stream[i..].starts_with(b"ET")
            && (i == 0 || stream[i - 1].is_ascii_whitespace())
            && stream.get(i + 2).is_none_or(|b| b.is_ascii_whitespace())
        {
            if in_text && !out.ends_with('\n') && !out.is_empty() {
                out.push('\n');
            }
            in_text = false;
            i += 2;
            continue;
        }
        if !in_text {
            i += 1;
            continue;
        }
        if stream[i] == b'(' {
            let (s, next) = parse_pdf_literal(stream, i + 1);
            out.push_str(&s);
            i = next;
            continue;
        }
        if stream[i] == b'<' && stream.get(i + 1) != Some(&b'<') {
            let (s, next) = parse_pdf_hex(stream, i + 1);
            out.push_str(&s);
            i = next;
            continue;
        }
        if stream[i..].starts_with(b"Td")
            || stream[i..].starts_with(b"TD")
            || stream[i..].starts_with(b"T*")
        {
            if !out.ends_with('\n') && !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn find_bytes(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from + needle.len() > haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|rel| from + rel)
}

fn extract_pdf(bytes: &[u8]) -> Result<String> {
    let mut chunks = Vec::new();
    let mut pos = 0usize;
    let mut total_decoded = 0usize;
    while let Some(s_pos) = find_bytes(bytes, b"stream", pos) {
        let dict_start = s_pos.saturating_sub(512);
        let dict_text = String::from_utf8_lossy(&bytes[dict_start..s_pos]);
        let mut data_start = s_pos + 6;
        if bytes.get(data_start) == Some(&b'\r') && bytes.get(data_start + 1) == Some(&b'\n') {
            data_start += 2;
        } else if matches!(bytes.get(data_start), Some(b'\n' | b'\r')) {
            data_start += 1;
        } else {
            pos = s_pos + 6;
            continue;
        }
        let Some(end_pos) = find_bytes(bytes, b"endstream", data_start) else {
            break;
        };
        let mut raw_stream = &bytes[data_start..end_pos];
        while raw_stream.ends_with(b"\n") || raw_stream.ends_with(b"\r") {
            raw_stream = &raw_stream[..raw_stream.len() - 1];
        }
        let decoded = if dict_text.contains("/FlateDecode") {
            let budget = MAX_DECOMPRESSED_BYTES.saturating_sub(total_decoded);
            zlib_decompress(raw_stream, budget).unwrap_or_default()
        } else {
            raw_stream.to_vec()
        };
        total_decoded += decoded.len();
        if total_decoded > MAX_DECOMPRESSED_BYTES {
            bail!("PDF decompressed streams exceed {MAX_DECOMPRESSED_BYTES}-byte limit");
        }
        let text = extract_pdf_text_operators(&decoded);
        if !text.is_empty() {
            chunks.push(text);
        }
        pos = end_pos + 9;
    }
    if chunks.is_empty() {
        let fallback = extract_pdf_text_operators(bytes);
        if !fallback.is_empty() {
            chunks.push(fallback);
        }
    }
    Ok(chunks.join("\n"))
}

pub(crate) fn extract_document_bytes(ext_or_name: &str, bytes: &[u8]) -> Result<(String, String)> {
    let ext = Path::new(ext_or_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let (format, text) = match ext.as_str() {
        "pdf" => ("pdf", extract_pdf(bytes)?),
        "docx" => ("docx", extract_docx(&read_zip_entries(bytes)?)),
        "xlsx" => ("xlsx", extract_xlsx(&read_zip_entries(bytes)?)),
        "pptx" => ("pptx", extract_pptx(&read_zip_entries(bytes)?)),
        "odt" | "ods" | "odp" => ("odf", extract_odf(&read_zip_entries(bytes)?)),
        "html" | "htm" => (
            "html",
            crate::web_tools::html_to_text(&String::from_utf8_lossy(bytes)),
        ),
        "xml" | "svg" => (
            "xml",
            strip_xml_tags(&String::from_utf8_lossy(bytes), &[], &[]),
        ),
        _ => {
            if bytes.starts_with(b"%PDF-") {
                ("pdf", extract_pdf(bytes)?)
            } else if bytes.starts_with(b"PK\x03\x04") {
                let entries = read_zip_entries(bytes)?;
                if entries.contains_key("word/document.xml") {
                    ("docx", extract_docx(&entries))
                } else if entries.keys().any(|k| k.starts_with("xl/")) {
                    ("xlsx", extract_xlsx(&entries))
                } else if entries.keys().any(|k| k.starts_with("ppt/")) {
                    ("pptx", extract_pptx(&entries))
                } else {
                    ("odf", extract_odf(&entries))
                }
            } else {
                ("text", String::from_utf8_lossy(bytes).into_owned())
            }
        }
    };
    Ok((format.to_owned(), text))
}

pub(crate) async fn execute_tool(
    data: &Path,
    session_id: &str,
    project: &crate::context::Project,
    args: &Value,
) -> Result<Value> {
    let raw_path = args["path"].as_str().context("Missing `path` string")?;
    let max_chars = args["max_chars"]
        .as_u64()
        .map(|n| (n as usize).clamp(256, MAX_EXTRACT_CHARS))
        .unwrap_or(DEFAULT_EXTRACT_CHARS);
    let (root, rel) = resolve_project_root(project, raw_path, false)?;
    let resolved = tools::resolve(&root.path, &rel, false)?;
    let path_str = resolved.display().to_string();
    let meta = std::fs::metadata(&resolved).context("Failed to stat document")?;
    if meta.len() > MAX_DECOMPRESSED_BYTES as u64 {
        bail!("Document exceeds {MAX_DECOMPRESSED_BYTES}-byte limit");
    }

    let path_clone = resolved.clone();
    let (format, text) = tokio::task::spawn_blocking(move || {
        let bytes = storage::read_bounded_bytes(&path_clone, MAX_DECOMPRESSED_BYTES)?;
        extract_document_bytes(&path_clone.to_string_lossy(), &bytes)
    })
    .await
    .context("Document extraction task failed")??;

    let (preview, spill_id, truncated, total_bytes) = spill::truncate_or_spill(
        Some(data),
        Some(session_id),
        "extract_doc",
        &text,
        max_chars,
    );

    Ok(json!({
        "path": path_str,
        "format": format,
        "text": preview,
        "truncated": truncated,
        "total_bytes": total_bytes,
        "spill_id": spill_id,
    }))
}

pub(crate) async fn extract_api(
    State(app): State<App>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    let max_chars = body["max_chars"]
        .as_u64()
        .map(|n| (n as usize).clamp(256, MAX_EXTRACT_CHARS))
        .unwrap_or(DEFAULT_EXTRACT_CHARS);

    if let Some(b64) = body["content_b64"].as_str() {
        let filename = body["filename"].as_str().unwrap_or("upload.txt").to_owned();
        let bytes = storage::base64_decode(b64).map_err(bad_request)?;
        if bytes.len() > MAX_UPLOAD_BYTES {
            return Err(bad_request(format!(
                "Uploaded document exceeds {MAX_UPLOAD_BYTES}-byte limit"
            )));
        }
        let (format, text) =
            tokio::task::spawn_blocking(move || extract_document_bytes(&filename, &bytes))
                .await
                .map_err(bad_request)?
                .map_err(bad_request)?;
        let sid = body["session_id"].as_str();
        let (preview, spill_id, truncated, total_bytes) = spill::truncate_or_spill(
            Some(app.data.as_path()),
            sid,
            "extract_doc",
            &text,
            max_chars,
        );
        return Ok(Json(json!({
            "format": format,
            "text": preview,
            "truncated": truncated,
            "total_bytes": total_bytes,
            "spill_id": spill_id,
        })));
    }

    let sid = body["session_id"]
        .as_str()
        .ok_or_else(|| bad_request("Provide either `content_b64` or (`session_id` + `path`)"))?;
    let path = body["path"]
        .as_str()
        .ok_or_else(|| bad_request("Missing `path`"))?;
    let shared = get_session(&app, sid)?;
    let (_, _, project) = project_for_session(&app, &shared)?;
    if project.roots.is_empty() {
        return Err(not_found("Project has no workspace roots"));
    }
    let res = execute_tool(
        app.data.as_path(),
        sid,
        &project,
        &json!({ "path": path, "max_chars": max_chars }),
    )
    .await
    .map_err(bad_request)?;
    Ok(Json(res))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if (crc & 1) != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn deflate_stored(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(0x01); // BFINAL=1, BTYPE=00
        let len = data.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    fn deflate_fixed_literals(data: &[u8]) -> Vec<u8> {
        let mut bits = 0u64;
        let mut nbits = 0u8;
        let mut out = Vec::new();
        let mut push_bits = |val: u32, count: u8| {
            bits |= (val as u64) << nbits;
            nbits += count;
            while nbits >= 8 {
                out.push((bits & 0xFF) as u8);
                bits >>= 8;
                nbits -= 8;
            }
        };
        let reverse = |mut v: u32, len: u8| -> u32 {
            let mut r = 0u32;
            for _ in 0..len {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            r
        };
        // BFINAL=1, BTYPE=01 -> LSB bits: 1, 1, 0 = 0b011
        push_bits(0b011, 3);
        for &b in data {
            let code = if b <= 143 {
                0x30 + b as u32
            } else {
                0x190 + (b as u32 - 144)
            };
            let len = if b <= 143 { 8 } else { 9 };
            push_bits(reverse(code, len), len);
        }
        // End-of-block symbol 256 -> 7-bit 0000000
        push_bits(0, 7);
        if nbits > 0 {
            out.push(bits as u8);
        }
        out
    }

    fn build_zip(entries: &[(&str, Vec<u8>, u16)]) -> Vec<u8> {
        let mut archive = Vec::new();
        let mut cd = Vec::new();
        for (name, raw, method) in entries {
            let payload = match method {
                0 => raw.clone(),
                8 => deflate_fixed_literals(raw),
                _ => unreachable!(),
            };
            let crc = crc32(raw);
            let local_offset = archive.len() as u32;
            archive.extend_from_slice(b"PK\x03\x04");
            archive.extend_from_slice(&20u16.to_le_bytes());
            archive.extend_from_slice(&0u16.to_le_bytes());
            archive.extend_from_slice(&method.to_le_bytes());
            archive.extend_from_slice(&0u32.to_le_bytes());
            archive.extend_from_slice(&crc.to_le_bytes());
            archive.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            archive.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
            archive.extend_from_slice(&0u16.to_le_bytes());
            archive.extend_from_slice(name.as_bytes());
            archive.extend_from_slice(&payload);

            cd.extend_from_slice(b"PK\x01\x02");
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&method.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&crc.to_le_bytes());
            cd.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&local_offset.to_le_bytes());
            cd.extend_from_slice(name.as_bytes());
        }
        let cd_offset = archive.len() as u32;
        let cd_size = cd.len() as u32;
        archive.extend_from_slice(&cd);
        archive.extend_from_slice(b"PK\x05\x06");
        archive.extend_from_slice(&0u32.to_le_bytes());
        archive.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        archive.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        archive.extend_from_slice(&cd_size.to_le_bytes());
        archive.extend_from_slice(&cd_offset.to_le_bytes());
        archive.extend_from_slice(&0u16.to_le_bytes());
        archive
    }

    #[test]
    fn deflate_stored_fixed_and_budget_limit() {
        let msg = b"Hello Allpaka DEFLATE world!";
        assert_eq!(inflate_raw(&deflate_stored(msg), 1024).unwrap(), msg);
        assert_eq!(
            inflate_raw(&deflate_fixed_literals(msg), 1024).unwrap(),
            msg
        );
        assert!(inflate_raw(&deflate_fixed_literals(msg), 8).is_err());
    }

    #[test]
    fn extracts_compressed_docx_and_xlsx() {
        let doc_xml = br#"<w:document><w:body><w:p><w:r><w:t>First Paragraph</w:t></w:r></w:p><w:p><w:r><w:t>Second &amp; Final</w:t></w:r></w:p></w:body></w:document>"#;
        let docx = build_zip(&[("word/document.xml", doc_xml.to_vec(), 8)]);
        let (fmt, txt) = extract_document_bytes("report.docx", &docx).unwrap();
        assert_eq!(fmt, "docx");
        assert!(txt.contains("First Paragraph"));
        assert!(txt.contains("Second & Final"));

        let shared = br#"<sst><si><t>Metric</t></si><si><t>Score</t></si></sst>"#;
        let sheet1 = br#"<worksheet><sheetData><row><c t="s"><v>0</v></c><c><v>99</v></c></row></sheetData></worksheet>"#;
        let xlsx = build_zip(&[
            ("xl/sharedStrings.xml", shared.to_vec(), 8),
            ("xl/worksheets/sheet1.xml", sheet1.to_vec(), 8),
        ]);
        let (fmt2, txt2) = extract_document_bytes("metrics.xlsx", &xlsx).unwrap();
        assert_eq!(fmt2, "xlsx");
        assert!(txt2.contains("Metric\t99"));
    }

    #[test]
    fn extracts_flate_pdf_text_operators() {
        let content = b"BT /F1 12 Tf 72 720 Td (Hello PDF \\(Flate\\)) Tj ET";
        let deflated = deflate_fixed_literals(content);
        let mut zlib_stream = vec![0x78u8, 0x9C];
        zlib_stream.extend_from_slice(&deflated);
        zlib_stream.extend_from_slice(&[0, 0, 0, 0]);

        let mut pdf = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode >>\nstream\n");
        pdf.extend_from_slice(&zlib_stream);
        pdf.extend_from_slice(b"\nendstream\nendobj\n%%EOF\n");

        let (fmt, txt) = extract_document_bytes("spec.pdf", &pdf).unwrap();
        assert_eq!(fmt, "pdf");
        assert!(txt.contains("Hello PDF (Flate)"));
    }
}
