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
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub struct Tokenizer {
    pieces: Vec<String>,
    piece_bytes: Vec<Box<[u8]>>,
    ids: HashMap<String, u32>,
    /// Merge table indexed by `(left_sym, right_sym) -> (rank, merged_sym)`.
    /// Eliminates all per-step string formatting and hashing during BPE encoding.
    merges: HashMap<(u32, u32), (usize, u32)>,
    /// Precomputed symbol ID for each raw byte `0..256` after `BYTE_TO_CHAR` mapping.
    byte_sym: [u32; 256],
    /// Precomputed vocabulary token ID for each raw byte `0..256` (`u32::MAX` if absent).
    byte_vocab_id: [u32; 256],
    pub eos: Option<u32>,
    pub bos: Option<u32>,
}

const BYTE_TO_CHAR: [char; 256] = byte_to_char_table();
const CHAR_TO_BYTE: [u16; 512] = char_to_byte_table();

impl Tokenizer {
    pub fn from_gguf(f: &allpaka_gguf::GgufFile) -> Result<Self> {
        let pieces: Vec<String> = f
            .vocab_tokens()
            .context("GGUF carries no tokenizer.ggml.tokens; cannot tokenise text")?
            .to_vec();
        let merge_rules = f
            .merges()
            .context("GGUF carries no tokenizer.ggml.merges")?;

        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            ids.insert(p.clone(), i as u32);
        }

        // Symbol table: `0..pieces.len()` match vocabulary token IDs directly;
        // any intermediate merge pieces not in `pieces` receive IDs `>= pieces.len()`.
        let mut sym_ids: HashMap<String, u32> = ids.clone();
        let mut next_sym = pieces.len() as u32;
        let mut intern_sym = |s: &str, sym_ids: &mut HashMap<String, u32>| -> u32 {
            if let Some(&id) = sym_ids.get(s) {
                id
            } else {
                let id = next_sym;
                next_sym += 1;
                sym_ids.insert(s.to_owned(), id);
                id
            }
        };

        let mut byte_sym = [0u32; 256];
        let mut byte_vocab_id = [u32::MAX; 256];
        let mut single_buf = [0u8; 4];
        for b in 0..256 {
            let ch = BYTE_TO_CHAR[b];
            let s: &str = ch.encode_utf8(&mut single_buf);
            byte_sym[b] = intern_sym(s, &mut sym_ids);
            if let Some(&vid) = ids.get(s) {
                byte_vocab_id[b] = vid;
            }
        }

        let mut merges = HashMap::with_capacity(merge_rules.len());
        let mut merged_buf = String::with_capacity(32);
        for (rank, rule) in merge_rules.iter().enumerate() {
            let (left, right) = rule
                .split_once(' ')
                .with_context(|| format!("malformed merge rule {rule:?}"))?;
            let l_sym = intern_sym(left, &mut sym_ids);
            let r_sym = intern_sym(right, &mut sym_ids);
            merged_buf.clear();
            merged_buf.push_str(left);
            merged_buf.push_str(right);
            let m_sym = intern_sym(&merged_buf, &mut sym_ids);
            merges.entry((l_sym, r_sym)).or_insert((rank, m_sym));
        }

        let char_to_byte = &CHAR_TO_BYTE;
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
            merges,
            byte_sym,
            byte_vocab_id,
        })
    }

    #[inline]
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
            self.encode_segment(segment, &mut out)?;
        }
        Ok(out)
    }

    fn encode_segment(&self, segment: &str, out: &mut Vec<u32>) -> Result<()> {
        let raw = segment.as_bytes();
        let n = raw.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            let b = raw[0] as usize;
            let vid = self.byte_vocab_id[b];
            if vid == u32::MAX {
                bail!("byte piece {:?} missing from vocab", BYTE_TO_CHAR[b]);
            }
            out.push(vid);
            return Ok(());
        }

        let mut syms: Vec<u32> = raw.iter().map(|&b| self.byte_sym[b as usize]).collect();
        let mut prev: Vec<usize> = (0..n)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut next: Vec<usize> = (0..n)
            .map(|i| if i + 1 < n { i + 1 } else { usize::MAX })
            .collect();
        let mut span_end: Vec<usize> = (1..=n).collect();

        // Priority queue of candidate merges: (Reverse(rank), Reverse(left_pos), left_pos, right_pos, left_sym, right_sym)
        let mut heap = BinaryHeap::with_capacity(n);
        for i in 0..(n - 1) {
            let l_sym = syms[i];
            let r_sym = syms[i + 1];
            if let Some(&(rank, _)) = self.merges.get(&(l_sym, r_sym)) {
                heap.push((Reverse(rank), Reverse(i), i, i + 1, l_sym, r_sym));
            }
        }

        while let Some((Reverse(_rank), Reverse(_), i, j, l_sym, r_sym)) = heap.pop() {
            if next[i] != j || syms[i] != l_sym || syms[j] != r_sym {
                continue;
            }
            let &(_, merged_sym) = self
                .merges
                .get(&(l_sym, r_sym))
                .expect("validated merge pair must exist");
            syms[i] = merged_sym;
            syms[j] = u32::MAX;
            span_end[i] = span_end[j];
            let nx = next[j];
            next[i] = nx;
            if nx != usize::MAX {
                prev[nx] = i;
            }

            let p = prev[i];
            if p != usize::MAX {
                let pl_sym = syms[p];
                if let Some(&(r, _)) = self.merges.get(&(pl_sym, merged_sym)) {
                    heap.push((Reverse(r), Reverse(p), p, i, pl_sym, merged_sym));
                }
            }
            if nx != usize::MAX {
                let nr_sym = syms[nx];
                if let Some(&(r, _)) = self.merges.get(&(merged_sym, nr_sym)) {
                    heap.push((Reverse(r), Reverse(i), i, nx, merged_sym, nr_sym));
                }
            }
        }

        let vocab_len = self.pieces.len() as u32;
        let mut i = 0;
        while i != usize::MAX {
            let sym = syms[i];
            if sym < vocab_len {
                out.push(sym);
            } else {
                for &b in &raw[i..span_end[i]] {
                    let vid = self.byte_vocab_id[b as usize];
                    if vid == u32::MAX {
                        bail!(
                            "byte piece {:?} missing from vocab",
                            BYTE_TO_CHAR[b as usize]
                        );
                    }
                    out.push(vid);
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
        match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        }
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
fn segments(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut seg_start = 0;
    let mut in_word = false;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_word && idx > seg_start {
                out.push(&text[seg_start..idx]);
                seg_start = idx;
            }
            in_word = false;
        } else {
            in_word = true;
        }
    }
    if seg_start < text.len() {
        out.push(&text[seg_start..]);
    }
    out
}

/// The GPT-2 byte-to-printable-character bijection: printable ASCII and two
/// Latin-1 ranges map to themselves, everything else is displaced to
/// 0x100 + n in the order encountered.
const fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut b_idx = 0usize;
    while b_idx < 256 {
        let b = b_idx as u8;
        let keep = (b >= b'!' && b <= b'~') || (b >= 0xa1 && b <= 0xac) || b >= 0xae;
        table[b_idx] = if keep {
            b as char
        } else {
            let c = char::from_u32(0x100 + n).unwrap();
            n += 1;
            c
        };
        b_idx += 1;
    }
    table
}

const fn char_to_byte_table() -> [u16; 512] {
    let mut char_to_byte = [u16::MAX; 512];
    let mut b = 0usize;
    while b < 256 {
        let c = BYTE_TO_CHAR[b] as usize;
        char_to_byte[c] = b as u16;
        b += 1;
    }
    char_to_byte
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
