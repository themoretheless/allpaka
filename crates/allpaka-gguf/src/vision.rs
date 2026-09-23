//! Census of a vision-projector (`mmproj`) GGUF.
//!
//! The engine has no vision path yet, so this module reports what the file
//! *contains* instead of asserting a model shape. Hard-coding key names would
//! be guessing: projector families differ in which `clip.*` fields they carry
//! and in how the tower is laid out. What is reported here is read out of the
//! file itself:
//!
//! * every `clip.*` metadata field, with its value rendered,
//! * `clip.projector_type`, the one field the loader will branch on,
//! * tensors grouped by their top-level name prefix (`v`, `mm`, `a`, …), with
//!   counts, element totals and dtypes per group.
//!
//! That is exactly the information a loader needs before it can be written, and
//! it stays true for files this crate has never seen. Nothing here claims the
//! engine can run the projector; `allpaka inspect --mmproj` prints the census
//! and says so.

use crate::metadata::parse_file;
use crate::tensors::TensorInfo;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// One metadata field, already rendered for display.
#[derive(Debug, Clone)]
pub struct Field {
    pub key: String,
    pub value: String,
}

/// Tensors sharing a top-level name prefix.
#[derive(Debug, Clone)]
pub struct TensorGroup {
    pub prefix: String,
    pub count: usize,
    pub elements: u128,
    /// Distinct ggml dtypes in this group, with the number of tensors each.
    pub types: Vec<(String, usize)>,
    /// Up to three example names with their shapes, for orientation.
    pub examples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VisionCensus {
    pub path: String,
    pub file_bytes: u64,
    pub architecture: String,
    /// `clip.projector_type`, when the file states it.
    pub projector_type: Option<String>,
    /// Every `clip.*` metadata field found, in file order.
    pub clip_fields: Vec<Field>,
    /// Tensor groups, ordered by name prefix.
    pub tensor_groups: Vec<TensorGroup>,
    pub tensor_count: usize,
}

impl VisionCensus {
    pub fn read(path: &Path) -> Result<Self> {
        let header = parse_file(path)?;
        let file_bytes = std::fs::metadata(path)
            .with_context(|| format!("reading size of {}", path.display()))?
            .len();

        let projector_type = header.field_str("clip.projector_type").map(str::to_string);
        let clip_fields = header
            .fields()
            .filter(|(key, _)| key.starts_with("clip."))
            .map(|(key, value)| Field {
                key: key.to_string(),
                value: value.describe(),
            })
            .collect();

        let tensor_count = header.tensors.len();
        let mut groups: BTreeMap<String, TensorGroup> = BTreeMap::new();
        for t in &header.tensors {
            let prefix = t.name.split('.').next().unwrap_or_default().to_string();
            let group = groups.entry(prefix.clone()).or_insert_with(|| TensorGroup {
                prefix,
                count: 0,
                elements: 0,
                types: Vec::new(),
                examples: Vec::new(),
            });
            group.count += 1;
            group.elements += t.elements() as u128;
            let dtype = format!("{:?}", t.ggml_type);
            match group.types.iter_mut().find(|(name, _)| *name == dtype) {
                Some((_, n)) => *n += 1,
                None => group.types.push((dtype, 1)),
            }
            if group.examples.len() < 3 {
                group.examples.push(format!("{} {}", t.name, shape(t)));
            }
        }

        Ok(Self {
            path: path.display().to_string(),
            file_bytes,
            architecture: header.architecture,
            projector_type,
            clip_fields,
            tensor_groups: groups.into_values().collect(),
            tensor_count,
        })
    }

    /// The group with this top-level prefix, if the file has one.
    ///
    /// llama.cpp mmproj files conventionally name the vision tower `v.`, an
    /// audio tower `a.` and the projector `mm.`. The census reports the
    /// prefixes it actually saw; it does not assume them.
    pub fn group(&self, prefix: &str) -> Option<&TensorGroup> {
        self.tensor_groups.iter().find(|g| g.prefix == prefix)
    }
}

/// Shapes are printed llama.cpp-style: a 1-D tensor as `[n]`, and higher ranks
/// in declaration order, which is what the tensor table carries.
fn shape(t: &TensorInfo) -> String {
    let dims = t
        .dims
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" x ");
    format!("[{}]", dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A tiny GGUF shaped like an mmproj: two string fields, one `clip.*`
    /// integer field, one `v.` tensor and one `mm.` tensor.
    fn clip_gguf() -> Vec<u8> {
        let mut kvs: Vec<u8> = Vec::new();
        let kv_str = |kvs: &mut Vec<u8>, key: &str, value: &str| {
            kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
            kvs.extend_from_slice(key.as_bytes());
            kvs.extend_from_slice(&8u32.to_le_bytes()); // GGUF value type: string
            kvs.extend_from_slice(&(value.len() as u64).to_le_bytes());
            kvs.extend_from_slice(value.as_bytes());
        };
        kv_str(&mut kvs, "general.architecture", "clip");
        kv_str(&mut kvs, "clip.projector_type", "mlp");
        let kv_u32 = |kvs: &mut Vec<u8>, key: &str, value: u32| {
            kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
            kvs.extend_from_slice(key.as_bytes());
            kvs.extend_from_slice(&4u32.to_le_bytes()); // GGUF value type: u32
            kvs.extend_from_slice(&value.to_le_bytes());
        };
        kv_u32(&mut kvs, "clip.vision.patch_size", 14);

        let mut tensors: Vec<u8> = Vec::new();
        let add = |out: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64| {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        };
        add(&mut tensors, "v.patch_embd.weight", &[64, 3, 14, 14], 0, 0);
        add(&mut tensors, "mm.0.weight", &[64, 64], 0, 0);

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes()); // tensors
        out.extend_from_slice(&3u64.to_le_bytes()); // kv pairs
        out.extend_from_slice(&kvs);
        out.extend_from_slice(&tensors);
        out
    }

    /// Each test writes its own file name: `cargo test` runs them in parallel
    /// threads of one process, so a shared path would be a race.
    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("allpaka-vision-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn census_reports_clip_fields_and_tensor_groups() {
        let path = write_temp("mmproj.gguf", &clip_gguf());
        let census = VisionCensus::read(&path).unwrap();

        assert_eq!(census.architecture, "clip");
        assert_eq!(census.projector_type.as_deref(), Some("mlp"));
        assert!(census
            .clip_fields
            .iter()
            .any(|f| f.key == "clip.vision.patch_size" && f.value == "14"));

        assert_eq!(census.tensor_count, 2);
        assert_eq!(census.tensor_groups.len(), 2);

        let vision = census.group("v").expect("vision tower group");
        assert_eq!(vision.count, 1);
        assert_eq!(vision.elements, 64u128 * 3 * 14 * 14);
        assert_eq!(vision.types, vec![("F32".to_string(), 1)]);
        assert_eq!(vision.examples, vec!["v.patch_embd.weight [64 x 3 x 14 x 14]"]);

        let projector = census.group("mm").expect("projector group");
        assert_eq!(projector.count, 1);
        assert!(census.group("a").is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_text_model_has_no_clip_fields() {
        // The census must not pretend a plain model is a projector.
        let mut kvs: Vec<u8> = Vec::new();
        let key = "general.architecture";
        kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
        kvs.extend_from_slice(key.as_bytes());
        kvs.extend_from_slice(&8u32.to_le_bytes());
        let value = "llama";
        kvs.extend_from_slice(&(value.len() as u64).to_le_bytes());
        kvs.extend_from_slice(value.as_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one kv pair
        bytes.extend_from_slice(&kvs);

        let path = write_temp("text-model.gguf", &bytes);
        let census = VisionCensus::read(&path).unwrap();
        assert_eq!(census.architecture, "llama");
        assert!(census.clip_fields.is_empty());
        assert!(census.projector_type.is_none());
        assert_eq!(census.tensor_count, 0);
        let _ = std::fs::remove_file(&path);
    }
}
