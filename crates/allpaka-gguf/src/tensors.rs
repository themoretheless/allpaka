//! Access to tensor data: the table, the mmap, and bounds-checked views.

use crate::metadata::{parse, Header};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A GGML tensor dtype, with the block geometry needed to size its data.
///
/// Only the types this project actually plans to serve are decoded; everything
/// else is carried as `Other` so the table still parses and the error surfaces
/// at the moment someone asks for the bytes, naming the tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            6 => GgmlType::Q5_0,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            other => GgmlType::Other(other),
        }
    }

    /// Elements per quantisation block.
    pub fn block_elements(self) -> Option<u64> {
        match self {
            GgmlType::F32 | GgmlType::F16 => Some(1),
            GgmlType::Q5_0 | GgmlType::Q8_0 => Some(32),
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
                Some(256)
            }
            GgmlType::Other(_) => None,
        }
    }

    /// Bytes per quantisation block.
    pub fn block_bytes(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 => Some(2),
            // f16 scale + 32 int8 values.
            GgmlType::Q8_0 => Some(34),
            // f16 scale + 32 high bits + 16 nibble bytes.
            GgmlType::Q5_0 => Some(22),
            // 16 scale bytes + 64 quant bytes + d + dmin.
            GgmlType::Q2K => Some(84),
            // 32 high-bit mask + 64 quant bytes + 12 packed scales + d.
            GgmlType::Q3K => Some(110),
            // d + dmin + 12 packed scales + 128 quant bytes.
            GgmlType::Q4K => Some(144),
            // d + dmin + 12 packed scales + 32 high-bit mask + 128 quant bytes.
            GgmlType::Q5K => Some(176),
            // 128 low bytes + 64 high bytes + 16 scales + d.
            GgmlType::Q6K => Some(210),
            GgmlType::Other(_) => None,
        }
    }
}

/// One row of the tensor table.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: dims[0] is the fastest-varying (row) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
    /// Which file of a multi-part GGUF split holds this tensor (0 for
    /// single-file models).
    pub part: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Bytes this tensor occupies on disk, or an error for a dtype this crate
    /// does not know the geometry of.
    pub fn byte_size(&self) -> Result<u64> {
        let (be, bb) = match (self.ggml_type.block_elements(), self.ggml_type.block_bytes()) {
            (Some(be), Some(bb)) => (be, bb),
            _ => bail!(
                "tensor {:?} has unsupported ggml type {:?}",
                self.name,
                self.ggml_type
            ),
        };
        let elements = self.elements();
        if elements % be != 0 {
            bail!(
                "tensor {:?}: {} elements is not a whole number of {}-element blocks",
                self.name,
                elements,
                be
            );
        }
        Ok(elements / be * bb)
    }
}

/// A GGUF file with its data section mapped into memory.
///
/// The mmap is the whole point: a 90 GiB model must not be copied to be read,
/// and the OS page cache shares the mapping between every consumer. Dequantised
/// copies are made per tensor, on request, by the caller who knows how long it
/// needs them.
pub struct GgufFile {
    header: Header,
    /// One mapping per split part; a single-file model has exactly one.
    mmaps: Vec<Mmap>,
    data_offsets: Vec<u64>,
    merged: Vec<TensorInfo>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut mmaps = Vec::new();
        let mut data_offsets = Vec::new();
        let mut merged = Vec::new();
        let mut header = None;
        for (pi, p) in split_paths(path)?.into_iter().enumerate() {
            let file = File::open(&p).with_context(|| format!("opening {}", p.display()))?;
            let h = parse(&mut BufReader::new(&file))
                .with_context(|| format!("parsing {}", p.display()))?;
            // Safety: the map is read-only; a concurrent writer to the file would
            // be undefined behaviour, which is the standard mmap trade every
            // GGUF-serving runtime makes.
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("memory-mapping {}", p.display()))?;
            data_offsets.push(h.data_offset);
            for mut t in h.tensors.clone() {
                t.part = pi;
                merged.push(t);
            }
            if pi == 0 {
                header = Some(h);
            }
            mmaps.push(mmap);
        }
        let header = header.context("no split parts")?;
        Ok(Self { header, mmaps, data_offsets, merged })
    }

    pub fn architecture(&self) -> &str {
        &self.header.architecture
    }

    /// The first part as mapped memory, for backends that want to share the
    /// pages with an accelerator rather than copy tensors out.
    pub fn mapping(&self) -> &[u8] {
        &self.mmaps[0]
    }

    /// Every split part's mapping, first part included.
    pub fn mappings(&self) -> impl Iterator<Item = &[u8]> {
        self.mmaps.iter().map(|m| &m[..])
    }

    /// A metadata field by full key, e.g. `"qwen3.attention.head_count"`.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.header.field_u32(key)
    }

    /// A float metadata field by full key, e.g. `"qwen3.rope.freq_base"`.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.header.field_f32(key)
    }

    /// A boolean metadata field by full key (GGUF bools read as 0/1 ints).
    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        self.header.field_u32(key).map(|v| v != 0)
    }

    /// A small u32 array metadata field by full key.
    pub fn meta_u32_arr(&self, key: &str) -> Option<&[u32]> {
        self.header.field_u32_arr(key)
    }

    /// A string metadata field by full key, e.g. `"tokenizer.chat_template"`.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.header.field_str(key)
    }

    /// The vocabulary pieces, when the file carries a tokenizer.
    pub fn vocab_tokens(&self) -> Option<&[String]> {
        self.header.tokens.as_deref()
    }

    /// BPE merge rules in rank order, when the file carries them.
    pub fn merges(&self) -> Option<&[String]> {
        self.header.merges.as_deref()
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.merged
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.merged.iter().find(|t| t.name == name)
    }

    /// The raw bytes of one tensor, straight out of the mapping.
    pub fn data(&self, t: &TensorInfo) -> Result<&[u8]> {
        let start = self
            .data_offsets
            .get(t.part)
            .with_context(|| format!("tensor {:?}: bad split part", t.name))?
            .checked_add(t.offset)
            .with_context(|| format!("tensor {:?}: offset overflows", t.name))?;
        let len = t.byte_size()?;
        let end = start
            .checked_add(len)
            .with_context(|| format!("tensor {:?}: size overflows", t.name))?;
        let mmap = &self.mmaps[t.part];
        if end > mmap.len() as u64 {
            bail!(
                "tensor {:?} claims bytes {start}..{end}, but the file has only {}",
                t.name,
                mmap.len()
            );
        }
        Ok(&mmap[start as usize..end as usize])
    }

    /// One tensor dequantised to f32, allocated fresh.
    pub fn dequant(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        crate::dequant::dequant(t.ggml_type, self.data(t)?, t.elements() as usize)
            .with_context(|| format!("dequantising tensor {:?}", t.name))
    }
}

/// Expand a possibly-split GGUF path (`name-00001-of-00002.gguf`) into every
/// part's path, in order. A plain path yields just itself.
fn split_paths(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    // The split suffix is exactly `-0000N-of-0000M` (15 chars) at the end.
    let Some(suffix_at) = stem.len().checked_sub(15) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let suffix = &stem[suffix_at..];
    let Some(body) = suffix.strip_prefix('-') else {
        return Ok(vec![path.to_path_buf()]);
    };
    let (a, b) = match body.split_once("-of-") {
        Some(p) => p,
        None => return Ok(vec![path.to_path_buf()]),
    };
    let (Ok(first), Ok(total)) = (a.parse::<u32>(), b.parse::<u32>()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    if total <= 1 || first != 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let base = &stem[..suffix_at];
    let mut out = Vec::with_capacity(total as usize);
    for i in 1..=total {
        let p = path.with_file_name(format!("{base}-{i:05}-of-{total:05}.gguf"));
        if !p.exists() {
            bail!("split GGUF part missing: {}", p.display());
        }
        out.push(p);
    }
    Ok(out)
}
