//! Access to tensor data: the table, the mmap, and bounds-checked views.

use crate::error::{GgufError, Result};
use crate::metadata::{parse, Header};
pub use allpaka_dtype::GgmlType;
use memmap2::Mmap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

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
        self.dims
            .iter()
            .copied()
            .try_fold(1u64, u64::checked_mul)
            .unwrap_or(u64::MAX)
    }

    /// Bytes this tensor occupies on disk, or an error for a dtype this crate
    /// does not know the geometry of.
    pub fn byte_size(&self) -> Result<u64> {
        let (be, bb) = match (
            self.ggml_type.block_elements(),
            self.ggml_type.block_bytes(),
        ) {
            (Some(be), Some(bb)) => (be, bb),
            _ => {
                return Err(GgufError::unsupported_tensor_type(
                    &self.name,
                    self.ggml_type,
                ))
            }
        };
        let elements = self
            .dims
            .iter()
            .copied()
            .try_fold(1u64, u64::checked_mul)
            .ok_or_else(|| GgufError::Tensor {
                name: self.name.clone(),
                detail: "element count overflows u64".to_string(),
            })?;
        if elements % be != 0 {
            return Err(GgufError::Tensor {
                name: self.name.clone(),
                detail: format!("{elements} elements is not a whole number of {be}-element blocks"),
            });
        }
        (elements / be)
            .checked_mul(bb)
            .ok_or_else(|| GgufError::Tensor {
                name: self.name.clone(),
                detail: "byte size overflows u64".to_string(),
            })
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
    tensor_index: std::collections::HashMap<String, usize>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut mmaps = Vec::new();
        let mut data_offsets = Vec::new();
        let mut merged = Vec::new();
        let mut header = None;
        for (pi, p) in split_paths(path)?.into_iter().enumerate() {
            let file = File::open(&p)
                .map_err(|e| GgufError::io_ctx(format!("opening {}", p.display()), e))?;
            let mut h = parse(&mut BufReader::new(&file))
                .map_err(|e| e.with_context(format!("parsing {}", p.display())))?;
            // Safety: the map is read-only; a concurrent writer to the file would
            // be undefined behaviour, which is the standard mmap trade every
            // GGUF-serving runtime makes.
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| GgufError::io_ctx(format!("memory-mapping {}", p.display()), e))?;
            data_offsets.push(h.data_offset);
            for mut t in std::mem::take(&mut h.tensors) {
                t.part = pi;
                merged.push(t);
            }
            if pi == 0 {
                header = Some(h);
            }
            mmaps.push(mmap);
        }
        let header = header.ok_or_else(|| GgufError::Format("no split parts".to_string()))?;
        let mut tensor_index = std::collections::HashMap::with_capacity(merged.len());
        for (idx, t) in merged.iter().enumerate() {
            tensor_index.entry(t.name.clone()).or_insert(idx);
        }
        Ok(Self {
            header,
            mmaps,
            data_offsets,
            merged,
            tensor_index,
        })
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
        self.tensor_index.get(name).map(|&idx| &self.merged[idx])
    }

    /// The raw bytes of one tensor, straight out of the mapping.
    pub fn data(&self, t: &TensorInfo) -> Result<&[u8]> {
        let start = self
            .data_offsets
            .get(t.part)
            .ok_or_else(|| GgufError::Tensor {
                name: t.name.clone(),
                detail: "bad split part".to_string(),
            })?
            .checked_add(t.offset)
            .ok_or_else(|| GgufError::Tensor {
                name: t.name.clone(),
                detail: "offset overflows".to_string(),
            })?;
        let len = t.byte_size()?;
        let end = start.checked_add(len).ok_or_else(|| GgufError::Tensor {
            name: t.name.clone(),
            detail: "size overflows".to_string(),
        })?;
        let mmap = &self.mmaps[t.part];
        if end > mmap.len() as u64 {
            return Err(GgufError::Tensor {
                name: t.name.clone(),
                detail: format!(
                    "claims bytes {start}..{end}, but the file has only {}",
                    mmap.len()
                ),
            });
        }
        Ok(&mmap[start as usize..end as usize])
    }

    /// Page-aligned slices covering only selected tensors. Adjacent or
    /// overlapping page ranges are merged, independently for each split part.
    /// 16 KiB alignment satisfies the Apple backend and is also valid on
    /// platforms with 4 KiB pages. No tensor bytes are copied or read here.
    pub fn selected_mappings(&self, select: impl Fn(&TensorInfo) -> bool) -> Result<Vec<&[u8]>> {
        const PAGE: usize = 16384;
        let mut ranges = Vec::new();
        for tensor in self.merged.iter().filter(|t| select(t)) {
            let data = self.data(tensor)?;
            let mapping = &self.mmaps[tensor.part];
            let offset = data.as_ptr() as usize - mapping.as_ptr() as usize;
            let start = offset / PAGE * PAGE;
            let end = (offset + data.len())
                .div_ceil(PAGE)
                .saturating_mul(PAGE)
                .min(mapping.len());
            if start < end {
                ranges.push((tensor.part, start, end));
            }
        }
        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize, usize)> = Vec::new();
        for (part, start, end) in ranges {
            if let Some(last) = merged.last_mut() {
                if last.0 == part && start <= last.2 {
                    last.2 = last.2.max(end);
                    continue;
                }
            }
            merged.push((part, start, end));
        }
        Ok(merged
            .into_iter()
            .map(|(part, start, end)| &self.mmaps[part][start..end])
            .collect())
    }

    /// One tensor dequantised to f32, allocated fresh.
    pub fn dequant(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        crate::dequant::dequant(t.ggml_type, self.data(t)?, t.elements() as usize).map_err(|e| {
            match e {
                GgufError::Dequant { detail, .. } => GgufError::Dequant {
                    tensor: Some(t.name.clone()),
                    detail,
                },
                other => GgufError::Dequant {
                    tensor: Some(t.name.clone()),
                    detail: other.to_string(),
                },
            }
        })
    }
}

/// Cryptographic identity of all GGUF parts in order. Bounded buffered reads
/// avoid faulting every mapped weight page into the stage's address space.
pub fn fingerprint(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let parts = split_paths(path)?;
    let mut digest = Sha256::new();
    digest.update(b"allpaka-gguf-v1");
    digest.update((parts.len() as u64).to_le_bytes());
    let mut buffer = vec![0u8; 1024 * 1024];
    for part in parts {
        let mut file = File::open(&part)?;
        let expected = file.metadata()?.len();
        digest.update(expected.to_le_bytes());
        let mut read = 0u64;
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
            read += n as u64;
        }
        if read != expected {
            return Err(GgufError::Format(
                "GGUF part changed while computing identity".to_string(),
            ));
        }
    }
    use std::fmt::Write;
    let hash = digest.finalize();
    let mut out = String::with_capacity(64);
    for byte in hash {
        let _ = write!(&mut out, "{byte:02x}");
    }
    Ok(out)
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
            return Err(GgufError::MissingSplitPart(p));
        }
        out.push(p);
    }
    Ok(out)
}
