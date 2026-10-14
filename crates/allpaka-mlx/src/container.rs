//! Container layer: read a list of safetensors files from an MLX checkpoint.
//!
//! Supports both single-file index (one `model.safetensors`) and sharded layout
//! (`model-00001-of-XXXXX.safetensors`), plus the optional `*.safetensors.index.json`.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub offset: u64,  // absolute byte offset within its shard mmap
    pub bytes: usize, // size in bytes
}

/// An MLX safetensors container analogous to `GgufFile`. Holds mmaps of all
/// shards and a combined tensor table.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensors: Vec<TensorInfo>,
    mmaps: Vec<memmap2::Mmap>,
    #[allow(dead_code)]
    data_starts: Vec<u64>,
}

/// Index.json maps each tensor name to the file it lives in and its byte range.
struct ShardedIndex {
    map: HashMap<String, (String, u64, u64)>, // name -> (filename, offset, end)
}

impl ShardedIndex {
    fn parse(json_str: &str) -> anyhow::Result<Self> {
        let obj: serde_json::Value = serde_json::from_str(json_str)?;
        let weight_map = obj
            .get("weight_map")
            .ok_or_else(|| anyhow::anyhow!("no weight_map"))?;
        let mut map = HashMap::new();
        if let serde_json::Value::Object(o) = weight_map {
            for (name, value) in o.iter() {
                let filename = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => continue,
                };
                map.insert(name.clone(), (filename, 0, 0));
            }
        }
        Ok(Self { map })
    }
}

pub fn load<P>(root: P) -> anyhow::Result<MlxSafetensors>
where
    P: AsRef<Path>,
{
    let root = root.as_ref();
    // Look for index.json first (sharded layout)
    let index_path = root.join("model.safetensors.index.json");
    let shard_files: Vec<String> = if index_path.is_file() {
        let idx_txt = std::fs::read_to_string(&index_path)?;
        let index = ShardedIndex::parse(&idx_txt)?;
        let mut files: Vec<String> = index.map.values().map(|(f, _, _)| f.clone()).collect();
        files.sort();
        files.dedup();
        if files.is_empty() {
            let pattern = root.join("model-*.safetensors").to_string_lossy().to_string();
            glob_pattern_to_filenames(&pattern)
        } else {
            files
        }
    } else {
        vec!["model.safetensors".into()]
    };
    let mut tensors = Vec::new();
    let mut mmaps = Vec::new();
    let mut data_starts = Vec::new();
    for (i, fname) in shard_files.iter().enumerate() {
        let fpath = root.join(fname);
        let mm = mmap_shard(&fpath)?;
        let info = parse_safetensors_header(&mm)?;
        let start = info.data_start as u64;
        mmaps.push(mm);
        data_starts.push(start);
        for mut t in info.tensors {
            t.offset = t
                .offset
                .checked_add(start)
                .ok_or_else(|| anyhow::anyhow!("tensor offset overflow for {}", t.name))?;
            t.name = format!("{}/{}/{}", i, fname, t.name);
            tensors.push(t);
        }
    }
    Ok(MlxSafetensors {
        tensors,
        mmaps,
        data_starts,
    })
}

fn parse_safetensors_header(mm: &Mmap) -> anyhow::Result<FileInfo> {
    let len = mm.len() as u64;
    if len < 8 {
        return Err(anyhow::anyhow!("empty safetensors"));
    }
    let n_bytes = u64::from_le_bytes(mm[0..8].try_into()?);
    let h_start = 8u64;
    let h_end = h_start
        .checked_add(n_bytes)
        .ok_or_else(|| anyhow::anyhow!("header size overflow"))?;
    if h_end > len {
        return Err(anyhow::anyhow!("header exceeds mmap"));
    }
    let hdr_txt = std::str::from_utf8(&mm[h_start as usize..h_end as usize])?;
    let hdr: serde_json::Value = serde_json::from_str(hdr_txt)?;
    let hdr_obj = hdr
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header must be a JSON object"))?;
    let empty_map = serde_json::Map::new();
    let meta = hdr_obj
        .get("__metadata__")
        .and_then(|v| v.as_object())
        .unwrap_or(&empty_map);
    let mut tensors = Vec::new();
    for (name, spec) in hdr_obj.iter() {
        if name == "__metadata__" {
            continue;
        }
        if let Some(spec) = spec.as_object() {
            let dtype = spec
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let shape = spec
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
                .unwrap_or_default();
            let offsets = spec.get("data_offsets").and_then(|v| v.as_array());
            let offset = offsets
                .and_then(|a| a.first().and_then(|v| v.as_u64()))
                .unwrap_or(0);
            let end_offset = offsets
                .and_then(|a| a.get(1).and_then(|v| v.as_u64()))
                .unwrap_or(offset);
            let nbytes = end_offset.saturating_sub(offset) as usize;
            tensors.push(TensorInfo {
                name: name.clone(),
                shape,
                dtype,
                offset,
                bytes: nbytes,
            });
        }
    }
    // Data section starts after header + alignment padding (default 32)
    let mut hdr_sz = usize::try_from(h_end).map_err(|_| anyhow::anyhow!("header too large"))?;
    let align: usize = meta
        .get("alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32)
        .max(1) as usize;
    hdr_sz = hdr_sz
        .div_ceil(align)
        .checked_mul(align)
        .ok_or_else(|| anyhow::anyhow!("header alignment overflow"))?;
    Ok(FileInfo {
        tensors,
        data_start: hdr_sz,
    })
}

fn mmap_shard<P>(path: P) -> anyhow::Result<Mmap>
where
    P: AsRef<Path>,
{
    let f = std::fs::File::open(path.as_ref())?;
    unsafe { memmap2::Mmap::map(&f) }
        .map_err(|e| anyhow::anyhow!("mmap fail {}: {}", path.as_ref().display(), e))
}

fn glob_pattern_to_filenames(pattern: &str) -> Vec<String> {
    use glob::glob;
    glob(pattern)
        .expect("invalid pattern")
        .filter_map(|r| r.ok())
        .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
        .collect()
}

struct FileInfo {
    tensors: Vec<TensorInfo>,
    data_start: usize,
}

pub fn tensor_data<'a>(f: &'a MlxSafetensors, t: &TensorInfo) -> anyhow::Result<&'a [u8]> {
    if f.mmaps.is_empty() {
        return Err(anyhow::anyhow!("no mmaps loaded in container"));
    }
    let part: usize = t.name.split('/').next().unwrap_or("0").parse().unwrap_or(0);
    let shard_idx = part.min(f.mmaps.len() - 1);
    let mm = &f.mmaps[shard_idx];
    let off = usize::try_from(t.offset)
        .map_err(|_| anyhow::anyhow!("tensor {} offset overflow", t.name))?;
    let end = off
        .checked_add(t.bytes)
        .ok_or_else(|| anyhow::anyhow!("tensor {} size overflow", t.name))?;
    if end > mm.len() {
        return Err(anyhow::anyhow!("tensor {} out of bounds", t.name));
    }
    Ok(&mm[off..end])
}

impl MlxSafetensors {
    /// Get a tensor by name from the loaded safetensors files.
    /// Matches either the full prefixed name (`"{shard}/{file}/{name}"`) or the raw tensor name.
    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| {
            t.name == name
                || t.name
                    .splitn(3, '/')
                    .nth(2)
                    .is_some_and(|raw_name| raw_name == name)
        })
    }

    /// Get total number of tensors loaded.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if no tensors were loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over all loaded tensors.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
}
