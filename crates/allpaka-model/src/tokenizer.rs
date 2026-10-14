//! Byte-level BPE, the tokenizer family of every model this engine targets
//! (GPT-2 lineage: Qwen, Llama 3, Mistral's newer releases).
//!
//! Everything comes from the GGUF itself - vocabulary pieces and merge ranks -
//! so there is no external tokenizer file to drift out of sync with the
//! weights.
//!
//! One honest simplification: real byte-level BPE first splits text with a
//! regex pre-tokenizer before merging. This implementation splits on
//! whitespace boundaries only (space attached to the following word, GPT-2
//! style). Merges learned by the reference tokenizer almost never cross the
//! boundaries the regex would have introduced within such a segment, so the
//! ids agree on ordinary text; pathological inputs (long digit runs, mixed
//! scripts) can tokenise differently. The forward pass is indifferent - it
//! sees ids either way - but byte-exact parity with llama.cpp tokenisation is
//! not claimed.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

pub struct Tokenizer {
    pieces: Vec<String>,
    piece_bytes: Vec<Box<[u8]>>,
    ids: HashMap<String, u32>,
    /// Merge rank by "left right" pair; lower rank merges first.
    ranks: HashMap<String, usize>,
    /// GPT-2 byte <-> printable char maps.
    byte_to_char: [char; 256],
    pub eos: Option<u32>,
    pub bos: Option<u32>,
}

impl Tokenizer {
    pub fn from_gguf(f: &allpaka_gguf::GgufFile) -> Result<Self> {
        let pieces: Vec<String> = f
            .vocab_tokens()
            .context("GGUF carries no tokenizer.ggml.tokens; cannot tokenise text")?
            .to_vec();
        let merges = f
            .merges()
            .context("GGUF carries no tokenizer.ggml.merges")?;

        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            ids.insert(p.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            let _ = rule
                .split_once(' ')
                .with_context(|| format!("malformed merge rule {rule:?}"))?;
            ranks.insert(rule.clone(), rank);
        }

        let byte_to_char = byte_to_char_table();
        let mut char_to_byte = [u16::MAX; 512];
        for (b, &c) in byte_to_char.iter().enumerate() {
            char_to_byte[c as usize] = b as u16;
        }

        let mut piece_bytes = Vec::with_capacity(pieces.len());
        let mut buf = Vec::with_capacity(16);
        let mut utf8_tmp = [0u8; 4];
        for piece in &pieces {
            buf.clear();
            for ch in piece.chars() {
                let code = ch as usize;
                if code < char_to_byte.len() && char_to_byte[code] != u16::MAX {
                    buf.push(char_to_byte[code] as u8);
                } else {
                    buf.extend_from_slice(ch.encode_utf8(&mut utf8_tmp).as_bytes());
                }
            }
            piece_bytes.push(buf.as_slice().into());
        }

        Ok(Self {
            eos: f.meta_u32("tokenizer.ggml.eos_token_id"),
            bos: f.meta_u32("tokenizer.ggml.bos_token_id"),
            pieces,
            piece_bytes,
            ids,
            ranks,
            byte_to_char,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// The id of an exact piece, used for special tokens like `<|im_start|>`.
    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).copied()
    }

    /// Encode plain text (no special-token recognition; templates insert
    /// specials by id).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for segment in segments(text) {
            self.encode_segment(&segment, &mut out)?;
        }
        Ok(out)
    }

    fn encode_segment(&self, segment: &str, out: &mut Vec<u32>) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }
        // Build the mapped printable string once and record initial byte-symbol spans.
        let mut mapped = String::with_capacity(segment.len() * 2);
        let n = segment.len();
        let mut starts = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(n);
        let mut next = Vec::with_capacity(n);
        for (idx, b) in segment.bytes().enumerate() {
            let start = mapped.len();
            mapped.push(self.byte_to_char[b as usize]);
            let end = mapped.len();
            starts.push(start);
            ends.push(end);
            next.push(if idx + 1 < n { idx + 1 } else { usize::MAX });
        }

        // Greedy lowest-rank merging over linked contiguous spans in `mapped`:
        // merging adjacent spans `i` and `j = next[i]` simply sets `ends[i] = ends[j]`
        // and `next[i] = next[j]` without any string reallocation or vector shifting!
        let mut pair_buf = String::with_capacity(32);
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, left_idx)
            let mut i = 0;
            while i != usize::MAX {
                let j = next[i];
                if j == usize::MAX {
                    break;
                }
                pair_buf.clear();
                pair_buf.push_str(&mapped[starts[i]..ends[i]]);
                pair_buf.push(' ');
                pair_buf.push_str(&mapped[starts[j]..ends[j]]);
                if let Some(&rank) = self.ranks.get(pair_buf.as_str()) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
                i = j;
            }
            let Some((_, best_i)) = best else { break };
            let best_j = next[best_i];
            ends[best_i] = ends[best_j];
            next[best_i] = next[best_j];
        }

        let mut i = 0;
        while i != usize::MAX {
            let s = &mapped[starts[i]..ends[i]];
            match self.ids.get(s) {
                Some(&id) => out.push(id),
                // A symbol not in the vocabulary decays to per-byte tokens.
                None => {
                    let mut single_buf = [0u8; 4];
                    for ch in s.chars() {
                        let single: &str = ch.encode_utf8(&mut single_buf);
                        let id = self
                            .ids
                            .get(single)
                            .with_context(|| format!("byte piece {single:?} missing from vocab"))?;
                        out.push(*id);
                    }
                }
            }
            i = next[i];
        }
        Ok(())
    }

    /// Append the raw decoded bytes of a single token to `bytes`.
    #[inline]
    pub fn append_token_bytes(&self, token: u32, bytes: &mut Vec<u8>) {
        if let Some(pb) = self.piece_bytes.get(token as usize) {
            bytes.extend_from_slice(pb);
        }
    }

    /// Decode ids to text. Unknown ids render as nothing rather than failing:
    /// decode is for humans, and a hole beats an abort mid-generation.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            if let Some(pb) = self.piece_bytes.get(t as usize) {
                bytes.extend_from_slice(pb);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Round-trip sanity used by loading code: encode must invert decode on
    /// the byte alphabet.
    pub fn self_check(&self) -> Result<()> {
        let text = "hello world";
        let ids = self.encode(text)?;
        if self.decode(&ids) != text {
            bail!("tokenizer failed to round-trip {text:?}");
        }
        Ok(())
    }
}

/// Split text into BPE segments: a run of whitespace is attached to the word
/// that follows it, which is how the GPT-2 alphabet expects spaces to travel.
fn segments(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if in_word && !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            in_word = false;
            current.push(ch);
        } else {
            in_word = true;
            current.push(ch);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// The GPT-2 byte-to-printable-character bijection: printable ASCII and two
/// Latin-1 ranges map to themselves, everything else is displaced to
/// 0x100 + n in the order encountered.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let keep = |b: u8| {
        (b'!'..=b'~').contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b)
    };
    let mut n = 0u32;
    for b in 0u16..256 {
        let b = b as u8;
        table[b as usize] = if keep(b) {
            b as char
        } else {
            let c = char::from_u32(0x100 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_table_is_a_bijection() {
        let table = byte_to_char_table();
        let mut seen = std::collections::HashSet::new();
        for c in table {
            assert!(seen.insert(c), "duplicate mapping for {c:?}");
        }
        // Spot checks against the reference table.
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(table[b' ' as usize], '\u{120}'); // 'Ġ'
        assert_eq!(table[b'\n' as usize], '\u{10a}'); // 'Ċ'
    }

    #[test]
    fn whitespace_travels_with_the_following_word() {
        assert_eq!(segments("a b  c"), vec!["a", " b", "  c"]);
        assert_eq!(segments("  lead"), vec!["  lead"]);
        assert_eq!(segments("tail "), vec!["tail", " "]);
    }
}
