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
/// shards and a combined tensor table with O(1) name lookup.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensors: Vec<TensorInfo>,
    tensor_shards: Vec<usize>,
    by_name: HashMap<String, usize>,
    mmaps: Vec<memmap2::Mmap>,
}

/// `index.json` maps each tensor name to the shard filename it lives in.
struct ShardedIndex {
    files: Vec<String>,
}

impl ShardedIndex {
    fn parse(json_str: &str) -> anyhow::Result<Self> {
        let obj: serde_json::Value = serde_json::from_str(json_str)?;
        let weight_map = obj
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("no weight_map object in safetensors index"))?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        files.sort_unstable();
        files.dedup();
        Ok(Self { files })
    }
}

pub fn load<P>(root: P) -> anyhow::Result<MlxSafetensors>
where
    P: AsRef<Path>,
{
    let root = root.as_ref();
    let index_path = root.join("model.safetensors.index.json");
    let shard_files: Vec<String> = if index_path.is_file() {
        let idx_txt = std::fs::read_to_string(&index_path)?;
        let index = ShardedIndex::parse(&idx_txt)?;
        if index.files.is_empty() {
            let pattern = root
                .join("model-*.safetensors")
                .to_string_lossy()
                .to_string();
            glob_pattern_to_filenames(&pattern)
        } else {
            index.files
        }
    } else {
        vec!["model.safetensors".into()]
    };

    // Map and parse shard headers in parallel when multiple shards exist.
    let parsed_shards: Vec<(Mmap, FileInfo)> = if shard_files.len() <= 1 {
        shard_files
            .iter()
            .map(|fname| {
                let mm = mmap_shard(root.join(fname))?;
                let info = parse_safetensors_header(&mm)?;
                Ok((mm, info))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = shard_files
                .iter()
                .map(|fname| {
                    s.spawn(move || -> anyhow::Result<(Mmap, FileInfo)> {
                        let mm = mmap_shard(root.join(fname))?;
                        let info = parse_safetensors_header(&mm)?;
                        Ok((mm, info))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("shard parse thread panicked"))
                .collect::<anyhow::Result<Vec<_>>>()
        })?
    };

    let total_tensors: usize = parsed_shards
        .iter()
        .map(|(_, info)| info.tensors.len())
        .sum();
    let mut tensors = Vec::with_capacity(total_tensors);
    let mut tensor_shards = Vec::with_capacity(total_tensors);
    let mut by_name = HashMap::with_capacity(total_tensors * 2);
    let mut mmaps = Vec::with_capacity(parsed_shards.len());

    for (i, (fname, (mm, info))) in shard_files.iter().zip(parsed_shards).enumerate() {
        let start = info.data_start as u64;
        mmaps.push(mm);
        for mut t in info.tensors {
            t.offset = t
                .offset
                .checked_add(start)
                .ok_or_else(|| anyhow::anyhow!("tensor offset overflow for {}", t.name))?;
            let raw_name = t.name;
            let prefixed_name = format!("{i}/{fname}/{raw_name}");
            t.name = prefixed_name.clone();
            let idx = tensors.len();
            by_name.entry(raw_name).or_insert(idx);
            by_name.insert(prefixed_name, idx);
            tensors.push(t);
            tensor_shards.push(i);
        }
    }

    Ok(MlxSafetensors {
        tensors,
        tensor_shards,
        by_name,
        mmaps,
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
    let mut tensors = Vec::with_capacity(hdr_obj.len());
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
    let shard_idx = f
        .by_name
        .get(&t.name)
        .and_then(|&idx| f.tensor_shards.get(idx).copied())
        .unwrap_or_else(|| t.name.split('/').next().unwrap_or("0").parse().unwrap_or(0))
        .min(f.mmaps.len() - 1);
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
    /// Get a tensor by name in O(1) from the loaded safetensors files.
    /// Matches either the full prefixed name (`"{shard}/{file}/{name}"`) or the raw tensor name.
    #[inline]
    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&idx| &self.tensors[idx])
    }

    /// Get total number of tensors loaded.
    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if no tensors were loaded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over all loaded tensors.
    #[inline]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
}
