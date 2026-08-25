//! GGUF header parsing: the KV section and the tensor table.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use crate::tensors::{GgmlType, TensorInfo};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
/// Section alignment when the file does not carry `general.alignment`.
const DEFAULT_ALIGNMENT: u64 = 32;

/// The subset of GGUF metadata the planner needs.
#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    /// Key/value heads. Equals `head_count` unless the model uses grouped
    /// query attention, in which case it is smaller and the KV cache shrinks
    /// proportionally.
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    /// Size of the file on disk, which is the real weight footprint at this
    /// quantisation. Far more trustworthy than estimating from parameter count.
    pub file_bytes: u64,
    /// Total elements across every tensor: the parameter count, measured from
    /// the tensor shapes rather than trusted from a filename. Prefill cost is
    /// FLOPs, and FLOPs are parameters, not bytes.
    pub param_count: u64,
    /// Experts per layer, 0 for a dense model.
    pub expert_count: u32,
    /// Experts actually evaluated per token.
    pub expert_used_count: u32,
    /// Share of all model elements that live in expert tensors.
    ///
    /// Measured by summing tensor shapes from the GGUF header rather than
    /// guessed from the architecture. Element counts are used rather than
    /// bytes so that no quantisation block-size table is needed; that is exact
    /// when every tensor shares a quantisation and slightly off when attention
    /// is kept at higher precision than the experts.
    pub expert_element_fraction: f64,
}

impl GgufInfo {
    /// Fraction of the weights that must be read to decode one token.
    ///
    /// A dense model reads everything, so this is 1.0. A mixture of experts
    /// reads its attention and shared layers in full but touches only the
    /// experts the router selected, which is what makes a huge MoE decode at
    /// the speed of a far smaller dense model. Capacity still needs the whole
    /// file resident - only the streaming cost shrinks.
    pub fn active_weight_fraction(&self) -> f64 {
        if self.expert_count == 0 || self.expert_used_count >= self.expert_count {
            return 1.0;
        }
        let used = self.expert_used_count as f64 / self.expert_count as f64;
        let f = self.expert_element_fraction.clamp(0.0, 1.0);
        (1.0 - f) + f * used
    }

    /// KV cache bytes for one token in one layer, at the given cache dtype size
    /// in bytes (2 for f16, 1 for q8_0).
    pub fn kv_bytes_per_token_per_layer(&self, cache_dtype_bytes: u64) -> u64 {
        let per_head = self.key_length as u64 + self.value_length as u64;
        per_head * self.head_count_kv as u64 * cache_dtype_bytes
    }
}

/// A GGUF metadata value. Numeric variants are collapsed to `u64`/`f64`
/// because every field this tool reads is a small unsigned integer.
pub(crate) enum Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    /// Bools land here as 0/1 (GGUF type 7 shares the Int reading path).
    Consumed,
    Str(String),
    /// A string array the caller asked to keep (vocabulary, merges).
    StrArray(Vec<String>),
    /// Arrays are otherwise skipped rather than materialised - most are large
    /// and nothing reads them.
    Skipped,
}

impl Value {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Value::Uint(v) => u32::try_from(*v).ok(),
            Value::Int(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }
}

/// Everything the header holds, parsed in one pass.
pub(crate) struct Header {
    pub architecture: String,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the aligned data section starts. Tensor
    /// offsets in the table are relative to this.
    pub data_offset: u64,
    /// The vocabulary pieces, materialised despite their size because the
    /// tokenizer needs them. Every other array is still skipped.
    pub tokens: Option<Vec<String>>,
    /// BPE merge rules, in rank order.
    pub merges: Option<Vec<String>>,
    fields: Vec<(String, Value)>,
}

impl Header {
    fn get_u32(&self, suffix: &str) -> Option<u32> {
        let full = format!("{}.{suffix}", self.architecture);
        self.field_u32(&full)
    }

    /// Look up a field by its full key, e.g. `qwen3.rope.freq_base`.
    pub(crate) fn field(&self, key: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub(crate) fn field_u32(&self, key: &str) -> Option<u32> {
        self.field(key).and_then(Value::as_u32)
    }

    pub(crate) fn field_f32(&self, key: &str) -> Option<f32> {
        match self.field(key)? {
            Value::Float(v) => Some(*v as f32),
            Value::Uint(v) => Some(*v as f32),
            Value::Int(v) => Some(*v as f32),
            _ => None,
        }
    }
}

pub(crate) fn parse(r: &mut (impl Read + Seek)) -> Result<Header> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).context("reading GGUF magic")?;
    if &magic != MAGIC {
        bail!("not a GGUF file (bad magic)");
    }

    let version = read_u32(r)?;
    if version < 2 {
        bail!("GGUF version {version} is not supported; version 2 or later is required");
    }
    let tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;

    let mut arch: Option<String> = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut fields: Vec<(String, Value)> = Vec::new();
    let mut tokens: Option<Vec<String>> = None;
    let mut merges: Option<Vec<String>> = None;

    for i in 0..kv_count {
        let key = read_string(r).with_context(|| format!("reading key {i}"))?;
        let keep_array = key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges";
        let value = read_value(r, keep_array)
            .with_context(|| format!("reading value for {key:?}"))?;
        if let Value::StrArray(items) = value {
            if key == "tokenizer.ggml.tokens" {
                tokens = Some(items);
            } else {
                merges = Some(items);
            }
            continue;
        }
        if key == "general.architecture" {
            if let Value::Str(s) = &value {
                arch = Some(s.clone());
            }
        }
        if key == "general.alignment" {
            if let Value::Uint(v) = &value {
                if *v > 0 {
                    alignment = *v;
                }
            }
        }
        fields.push((key, value));
    }

    // Later parts of a multi-file split carry no metadata kv at all; the
    // first part's architecture is the one callers see.
    let architecture = arch.unwrap_or_default();

    let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
    for i in 0..tensor_count {
        let name = read_string(r).with_context(|| format!("reading tensor name {i}"))?;
        let n_dims = read_u32(r)?;
        if n_dims > 4 {
            bail!("tensor {name:?} claims {n_dims} dimensions; GGUF allows at most 4");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let type_id = read_u32(r)?;
        let offset = read_u64(r)?;
        tensors.push(TensorInfo { name, dims, ggml_type: GgmlType::from_id(type_id), offset, part: 0 });
    }

    // The data section starts at the next aligned position after the header.
    let here = r.stream_position()?;
    let data_offset = here.div_ceil(alignment) * alignment;

    Ok(Header { architecture, tensors, data_offset, tokens, merges, fields })
}

/// Read the planner's view of a model. Only the header is touched.
pub fn read(path: &Path) -> Result<GgufInfo> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let header = parse(&mut r).with_context(|| format!("parsing {}", path.display()))?;

    let block_count = header
        .get_u32("block_count")
        .with_context(|| format!("GGUF file has no {}.block_count", header.architecture))?;
    let embedding_length = header
        .get_u32("embedding_length")
        .with_context(|| format!("GGUF file has no {}.embedding_length", header.architecture))?;
    let head_count = header.get_u32("attention.head_count").unwrap_or(0);
    let head_count_kv = header.get_u32("attention.head_count_kv").unwrap_or(head_count);

    // Most models omit key_length/value_length and imply embedding / heads.
    let implied = if head_count > 0 { embedding_length / head_count } else { 0 };
    let key_length = header.get_u32("attention.key_length").unwrap_or(implied);
    let value_length = header.get_u32("attention.value_length").unwrap_or(implied);

    let expert_count = header.get_u32("expert_count").unwrap_or(0);
    let expert_used_count = header.get_u32("expert_used_count").unwrap_or(0);

    // Parameters and the expert share both come from the tensor shapes.
    // llama.cpp names the fused expert weights `..._exps.weight`; the router
    // itself (`ffn_gate_inp`) is tiny and runs for every token, so it is
    // deliberately not counted as expert weight.
    let mut total: u128 = 0;
    let mut expert: u128 = 0;
    for t in &header.tensors {
        total += t.elements() as u128;
        if t.name.contains("_exps") {
            expert += t.elements() as u128;
        }
    }
    let expert_element_fraction = if total > 0 { expert as f64 / total as f64 } else { 0.0 };

    Ok(GgufInfo {
        architecture: header.architecture,
        block_count,
        embedding_length,
        head_count_kv,
        key_length,
        value_length,
        file_bytes,
        param_count: u64::try_from(total).unwrap_or(u64::MAX),
        expert_count,
        expert_used_count,
        expert_element_fraction,
    })
}

fn read_value(r: &mut (impl Read + Seek), keep_array: bool) -> Result<Value> {
    let kind = read_u32(r)?;
    Ok(match kind {
        0 => Value::Uint(read_exact_n::<1>(r)?[0] as u64),
        1 => Value::Int(read_exact_n::<1>(r)?[0] as i8 as i64),
        2 => Value::Uint(u16::from_le_bytes(read_exact_n::<2>(r)?) as u64),
        3 => Value::Int(i16::from_le_bytes(read_exact_n::<2>(r)?) as i64),
        4 => Value::Uint(read_u32(r)? as u64),
        5 => Value::Int(i32::from_le_bytes(read_exact_n::<4>(r)?) as i64),
        6 => Value::Float(f32::from_le_bytes(read_exact_n::<4>(r)?) as f64),
        7 => Value::Int(read_exact_n::<1>(r)?[0] as i64),
        8 => Value::Str(read_string(r)?),
        9 if keep_array => {
            let elem_type = read_u32(r)?;
            if elem_type != 8 {
                bail!("expected a string array, found element type {elem_type}");
            }
            let len = read_u64(r)?;
            if len > 4 << 20 {
                bail!("implausible vocabulary size {len}");
            }
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(read_string(r)?);
            }
            Value::StrArray(items)
        }
        9 => {
            skip_array(r)?;
            Value::Skipped
        }
        10 => Value::Uint(read_u64(r)?),
        11 => Value::Int(i64::from_le_bytes(read_exact_n::<8>(r)?)),
        12 => Value::Float(f64::from_le_bytes(read_exact_n::<8>(r)?)),
        other => bail!("unknown GGUF value type {other}"),
    })
}

/// Advance past an array without allocating it.
fn skip_array(r: &mut (impl Read + Seek)) -> Result<()> {
    let elem_type = read_u32(r)?;
    let len = read_u64(r)?;
    // Fixed-width elements can be skipped with a single seek.
    let width = match elem_type {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    };
    match width {
        Some(w) => {
            r.seek(SeekFrom::Current((len * w) as i64))?;
        }
        None if elem_type == 8 => {
            // Strings are length-prefixed, so each must be stepped over in turn.
            for _ in 0..len {
                let n = read_u64(r)?;
                r.seek(SeekFrom::Current(n as i64))?;
            }
        }
        None if elem_type == 9 => {
            for _ in 0..len {
                skip_array(r)?;
            }
        }
        None => bail!("unknown GGUF array element type {elem_type}"),
    }
    Ok(())
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)?;
    if len > 1 << 20 {
        bail!("implausible GGUF string length {len}");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact_n::<4>(r)?))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact_n::<8>(r)?))
}

fn read_exact_n<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GGUF header byte by byte so the parser is checked against the
    /// spec rather than against itself.
    struct Builder {
        kvs: Vec<u8>,
        count: u64,
    }

    impl Builder {
        fn new() -> Self {
            Self { kvs: Vec::new(), count: 0 }
        }

        fn key(&mut self, k: &str) {
            self.kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(k.as_bytes());
        }

        fn string(mut self, k: &str, v: &str) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            self.kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(v.as_bytes());
            self.count += 1;
            self
        }

        fn u32(mut self, k: &str, v: u32) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&v.to_le_bytes());
            self.count += 1;
            self
        }

        /// A vocabulary-shaped array of strings, which the parser must step
        /// over without reading into memory.
        fn string_array(mut self, k: &str, items: &[&str]) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&8u32.to_le_bytes()); // element type: string
            self.kvs.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                self.kvs.extend_from_slice(&(s.len() as u64).to_le_bytes());
                self.kvs.extend_from_slice(s.as_bytes());
            }
            self.count += 1;
            self
        }

        fn u32_array(mut self, k: &str, items: &[u32]) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for v in items {
                self.kvs.extend_from_slice(&v.to_le_bytes());
            }
            self.count += 1;
            self
        }

        fn finish(self) -> Vec<u8> {
            self.finish_with_tensors(&[])
        }

        /// `tensors` is a list of (name, element count).
        fn finish_with_tensors(self, tensors: &[(&str, u64)]) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&self.count.to_le_bytes());
            out.extend_from_slice(&self.kvs);
            for (name, elements) in tensors {
                out.extend_from_slice(&(name.len() as u64).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
                out.extend_from_slice(&elements.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes()); // type
                out.extend_from_slice(&0u64.to_le_bytes()); // offset
            }
            out
        }
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("allpaka-test-{name}.gguf"));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn qwen_like() -> Builder {
        Builder::new()
            .string("general.architecture", "qwen3")
            .string_array("tokenizer.ggml.tokens", &["hello", "world", "<|im_start|>"])
            .u32("qwen3.block_count", 64)
            .u32("qwen3.embedding_length", 5120)
            .u32("qwen3.attention.head_count", 40)
            .u32("qwen3.attention.head_count_kv", 8)
            .u32_array("qwen3.some.list", &[1, 2, 3, 4])
    }

    #[test]
    fn reads_architecture_fields_past_skipped_arrays() {
        let path = write_temp("basic", &qwen_like().finish());
        let info = read(&path).unwrap();
        assert_eq!(info.architecture, "qwen3");
        assert_eq!(info.block_count, 64);
        assert_eq!(info.embedding_length, 5120);
        assert_eq!(info.head_count_kv, 8);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn head_dim_is_implied_when_key_length_is_absent() {
        let path = write_temp("implied", &qwen_like().finish());
        let info = read(&path).unwrap();
        assert_eq!(info.key_length, 5120 / 40);
        assert_eq!(info.value_length, 128);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn explicit_key_length_wins_over_the_implied_one() {
        let path = write_temp("explicit", &qwen_like().u32("qwen3.attention.key_length", 256).finish());
        let info = read(&path).unwrap();
        assert_eq!(info.key_length, 256);
        assert_eq!(info.value_length, 128); // still implied
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn grouped_query_attention_shrinks_the_kv_cache() {
        let path = write_temp("gqa", &qwen_like().finish());
        let info = read(&path).unwrap();
        // 8 kv heads x (128 + 128) dims x 2 bytes
        assert_eq!(info.kv_bytes_per_token_per_layer(2), 8 * 256 * 2);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn file_size_is_reported_as_the_weight_footprint() {
        let bytes = qwen_like().finish();
        let path = write_temp("size", &bytes);
        let info = read(&path).unwrap();
        assert_eq!(info.file_bytes, bytes.len() as u64);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_dense_model_reads_all_of_its_weights() {
        let path = write_temp("dense", &qwen_like().finish());
        let info = read(&path).unwrap();
        assert_eq!(info.expert_count, 0);
        assert_eq!(info.active_weight_fraction(), 1.0);
        std::fs::remove_file(path).ok();
    }

    /// The measurement that matters for MoE: how much of the file is expert
    /// weight, taken from the tensor shapes rather than assumed.
    #[test]
    fn a_mixture_of_experts_reads_only_the_routed_share() {
        // 90% of elements in experts, 8 experts, 2 used per token.
        let bytes = qwen_like()
            .u32("qwen3.expert_count", 8)
            .u32("qwen3.expert_used_count", 2)
            .finish_with_tensors(&[
                ("blk.0.ffn_gate_exps.weight", 900),
                ("blk.0.attn_q.weight", 100),
            ]);
        let path = write_temp("moe", &bytes);
        let info = read(&path).unwrap();
        assert_eq!(info.expert_count, 8);
        assert_eq!(info.expert_used_count, 2);
        assert!((info.expert_element_fraction - 0.9).abs() < 1e-9);
        // 0.1 dense + 0.9 * (2/8) = 0.325
        assert!((info.active_weight_fraction() - 0.325).abs() < 1e-9, "{}", info.active_weight_fraction());
        std::fs::remove_file(path).ok();
    }

    /// The router itself runs for every token, so it is not expert weight.
    #[test]
    fn the_router_is_not_counted_as_expert_weight() {
        let bytes = qwen_like()
            .u32("qwen3.expert_count", 8)
            .u32("qwen3.expert_used_count", 2)
            .finish_with_tensors(&[
                ("blk.0.ffn_gate_exps.weight", 800),
                ("blk.0.ffn_gate_inp.weight", 200),
            ]);
        let path = write_temp("router", &bytes);
        let info = read(&path).unwrap();
        assert!((info.expert_element_fraction - 0.8).abs() < 1e-9);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn using_every_expert_is_the_same_as_being_dense() {
        let bytes = qwen_like()
            .u32("qwen3.expert_count", 8)
            .u32("qwen3.expert_used_count", 8)
            .finish_with_tensors(&[("blk.0.ffn_gate_exps.weight", 1000)]);
        let path = write_temp("allexperts", &bytes);
        assert_eq!(read(&path).unwrap().active_weight_fraction(), 1.0);
        std::fs::remove_file(path).ok();
    }

    /// Parameters come from the tensor shapes, dense models included.
    #[test]
    fn parameter_count_is_the_sum_of_tensor_elements() {
        let bytes = qwen_like().finish_with_tensors(&[
            ("blk.0.attn_q.weight", 700),
            ("blk.0.ffn_up.weight", 300),
        ]);
        let path = write_temp("params", &bytes);
        assert_eq!(read(&path).unwrap().param_count, 1000);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_non_gguf_file_is_rejected() {
        let path = write_temp("notgguf", b"this is a text file");
        assert!(read(&path).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_model_without_block_count_is_rejected() {
        let bytes = Builder::new().string("general.architecture", "qwen3").finish();
        let path = write_temp("noblocks", &bytes);
        let err = read(&path).unwrap_err().to_string();
        assert!(format!("{err:#}").contains("block_count") || err.contains("block_count"));
        std::fs::remove_file(path).ok();
    }
}
