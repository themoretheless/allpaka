//! Container layer: read a list of safetensors files from an MLX checkpoint.
//!
//! Supports both single-file index (one `model.safetensors`) and sharded layout
//! (`model-00001-of-XXXXX.safetensors`), plus the optional `*.safetensors.index.json`.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use memmap2::Mmap;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub offset: u64,  // relative to start of this file's data section
    pub bytes: usize, // size in bytes
}

/// An MLX safetensors container analogous to `GgufFile`. Holds mmaps of all
/// shards and a combined tensor table.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensors: Vec<TensorInfo>,
    mmaps: Vec<memmap2::Mmap>,
    data_starts: Vec<u64>,
}

/// Index.json maps each tensor name to the file it lives in and its byte range.
struct ShardedIndex {
    map: HashMap<String, (String, u64, u64)>, // name -> (filename, offset, end)
}

impl ShardedIndex {
    fn parse(json_str: &str) -> anyhow::Result<Self> {
        let obj: serde_json::Value = serde_json::from_str(json_str)?;
        let weight_map = obj.get("weight_map").ok_or_else(|| anyhow::anyhow!("no weight_map"))?;
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
        let _index = ShardedIndex::parse(&idx_txt)?; // validate presence
        // Collect matching shards from the directory
        let pattern = format!("{}{}", root.display(), "/model-*.safetensors");
        glob_pattern_to_paths(&pattern)
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
            // Make offsets absolute relative to file data section
            t.offset += start;
            t.name = format!("{}/{}/{}", i, fname, t.name);
            tensors.push(t);
        }
    }
    Ok(MlxSafetensors { tensors, mmaps, data_starts })
}

fn parse_safetensors_header(mm:  &Mmap) -> anyhow::Result<FileInfo> {
    let len = mm.len() as u64;
    if len < 8 { return Err(anyhow::anyhow!("empty safetensors")); }
    let n_bytes = u64::from_le_bytes(mm[0..8].try_into()?);
    let h_start = 8u64;
    let h_end = 8 + n_bytes;
    if h_end > len { return Err(anyhow::anyhow!("header exceeds mmap")); }
    let hdr_txt = std::str::from_utf8(&mm[h_start as usize..h_end as usize])?;
    let hdr: serde_json::Value = serde_json::from_str(hdr_txt)?;
    // Parse tensor metadata from __metadata__ or top-level keys
    let empty_map = serde_json::Map::new();
    let meta = hdr.get("__metadata__")
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| &empty_map);
    let mut tensors = Vec::new();
    for (name, spec) in hdr.as_object().unwrap().iter() {
        if name == "__metadata__" { continue; }
        if let Some(spec) = spec.as_object() {
            let dtype = spec.get("dtype").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let shape = spec.get("shape")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
                .unwrap_or_default();
            let offset = spec.get("data_offsets")
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(0).and_then(|v| v.as_u64()))
                .unwrap_or(0);
            let nbytes = (offset as isize - (0 as isize)).abs() as usize; // placeholder
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
    let mut hdr_sz = (8 + n_bytes) as usize;
    let align: usize = meta.get("alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    hdr_sz = ((hdr_sz + align - 1) / align) * align;
    Ok(FileInfo { tensors, data_start: hdr_sz })
}

fn mmap_shard<P>(path: P) -> anyhow::Result<Mmap>
where
    P: AsRef<Path>,
{
    let f = std::fs::File::open(path.as_ref())?;
    unsafe { memmap2::Mmap::map(&f) }.map_err(|e| anyhow::anyhow!("mmap fail {}: {}", path.as_ref().display(), e))
}

fn glob_pattern_to_paths(pattern: &str) -> Vec<String> {
    use glob::glob;
    glob(pattern).expect("invalid pattern").filter_map(|r| r.ok()).map(|p| p.to_string_lossy().to_string()).collect()
}

struct FileInfo {
    tensors: Vec<TensorInfo>,
    data_start: usize,
}
impl FileInfo {
    fn data_section_start(&self) -> usize { self.data_start }
}

pub fn tensor_data<'a>(f: &'a MlxSafetensors, t: &TensorInfo) -> anyhow::Result<&'a [u8]> {
    let part: usize = t.name.split('/').nth(0).unwrap_or("0").parse().unwrap_or(0);
    let file_offset = t.offset;
    let end = file_offset + (t.bytes as u64);
    let mm = &f.mmaps[part.min(f.mmaps.len() - 1)];
    if end > mm.len() as u64 {
        return Err(anyhow::anyhow!("tensor {} out of bounds", t.name));
    }
    let off = (file_offset % f.data_starts[part.min(f.data_starts.len()-1)]) as usize;
    let len = t.bytes;
    Ok(&mm[off..off + len])
}

impl MlxSafetensors {
    /// Get a tensor by name from the loaded safetensors files.
    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
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
