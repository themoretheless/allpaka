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
    ids: HashMap<String, u32>,
    /// Merge rank by "left right" pair; lower rank merges first.
    ranks: HashMap<(String, String), usize>,
    /// GPT-2 byte <-> printable char maps.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos: Option<u32>,
    pub bos: Option<u32>,
}

impl Tokenizer {
    pub fn from_gguf(f: &allpaka_gguf::GgufFile) -> Result<Self> {
        let pieces: Vec<String> = f
            .vocab_tokens()
            .context("GGUF carries no tokenizer.ggml.tokens; cannot tokenise text")?
            .to_vec();
        let merges = f.merges().context("GGUF carries no tokenizer.ggml.merges")?;

        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            ids.insert(p.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            let (l, r) = rule
                .split_once(' ')
                .with_context(|| format!("malformed merge rule {rule:?}"))?;
            ranks.insert((l.to_string(), r.to_string()), rank);
        }

        let byte_to_char = byte_to_char_table();
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &c) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(c, b as u8);
        }

        Ok(Self {
            eos: f.meta_u32("tokenizer.ggml.eos_token_id"),
            bos: f.meta_u32("tokenizer.ggml.bos_token_id"),
            pieces,
            ids,
            ranks,
            byte_to_char,
            char_to_byte,
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
        // Bytes to the printable alphabet, one symbol per byte.
        let mut symbols: Vec<String> = segment
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();

        // Greedy lowest-rank merging until no rule applies.
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.ranks.get(&key) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        for s in &symbols {
            match self.ids.get(s) {
                Some(&id) => out.push(id),
                // A symbol not in the vocabulary decays to per-byte tokens.
                None => {
                    for ch in s.chars() {
                        let single = ch.to_string();
                        let id = self
                            .ids
                            .get(&single)
                            .with_context(|| format!("byte piece {single:?} missing from vocab"))?;
                        out.push(*id);
                    }
                }
            }
        }
        Ok(())
    }

    /// Decode ids to text. Unknown ids render as nothing rather than failing:
    /// decode is for humans, and a hole beats an abort mid-generation.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            let Some(piece) = self.pieces.get(t as usize) else { continue };
            for ch in piece.chars() {
                match self.char_to_byte.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // Specials and anything outside the byte alphabet pass
                    // through as UTF-8.
                    None => bytes.extend_from_slice(ch.to_string().as_bytes()),
                }
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
