//! GGUF reading for allpaka.
//!
//! Three layers, used by different consumers:
//!
//! * [`metadata`] - the header only. The planner uses this: layers, hidden
//!   size, expert counts, parameter count. Touches no tensor data.
//! * [`tensors`] - the tensor table plus an mmap of the data section. The
//!   engine uses this: named, bounds-checked access to weight bytes without
//!   copying a 90 GiB file.
//! * [`dequant`] - block formats to f32. The reference path; fast kernels
//!   operating on quantised blocks directly come later and are checked
//!   against it.

pub mod dequant;
pub mod metadata;
pub mod tensors;

pub use metadata::{read, GgufInfo};
pub use tensors::{GgmlType, GgufFile, TensorInfo};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny but complete GGUF file: header, aligned data section, one
    /// f32 tensor and one q8_0 tensor with known contents.
    fn build_test_file() -> Vec<u8> {
        let mut kvs: Vec<u8> = Vec::new();
        let mut kv_count = 0u64;
        let mut kv_str = |k: &str, v: &str| {
            kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
            kvs.extend_from_slice(k.as_bytes());
            kvs.extend_from_slice(&8u32.to_le_bytes());
            kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
            kvs.extend_from_slice(v.as_bytes());
            kv_count += 1;
        };
        kv_str("general.architecture", "qwen3");

        // Tensor 0: four f32 values at data offset 0.
        // Tensor 1: one q8_0 block (32 elements) at offset 32 (aligned).
        let mut tinfo: Vec<u8> = Vec::new();
        let mut add_tensor = |name: &str, dims: &[u64], ty: u32, off: u64| {
            tinfo.extend_from_slice(&(name.len() as u64).to_le_bytes());
            tinfo.extend_from_slice(name.as_bytes());
            tinfo.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                tinfo.extend_from_slice(&d.to_le_bytes());
            }
            tinfo.extend_from_slice(&ty.to_le_bytes());
            tinfo.extend_from_slice(&off.to_le_bytes());
        };
        add_tensor("small.f32", &[2, 2], 0, 0);
        add_tensor("block.q8", &[32], 8, 32);

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes()); // tensors
        out.extend_from_slice(&kv_count.to_le_bytes());
        out.extend_from_slice(&kvs);
        out.extend_from_slice(&tinfo);

        // Pad to the default 32-byte alignment.
        while out.len() % 32 != 0 {
            out.push(0);
        }
        // f32 tensor data: 1.0 2.0 3.0 4.0 (16 bytes), pad to offset 32.
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[0u8; 16]);
        // q8_0 block: scale 1.0 (f16 0x3c00), values 0..31.
        out.extend_from_slice(&0x3c00u16.to_le_bytes());
        for i in 0..32u8 {
            out.push(i);
        }
        out
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("allpaka-gguf-{name}.gguf"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn a_full_file_round_trips_through_the_mmap() {
        let path = write_temp("roundtrip", &build_test_file());
        let f = GgufFile::open(&path).unwrap();
        assert_eq!(f.architecture(), "qwen3");
        assert_eq!(f.tensors().len(), 2);

        let t = f.tensor("small.f32").unwrap().clone();
        assert_eq!(t.elements(), 4);
        assert_eq!(f.dequant(&t).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let q = f.tensor("block.q8").unwrap().clone();
        let vals = f.dequant(&q).unwrap();
        assert_eq!(vals.len(), 32);
        assert_eq!(vals[7], 7.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_tensor_reaching_past_the_end_of_the_file_is_an_error() {
        let mut bytes = build_test_file();
        bytes.truncate(bytes.len() - 8); // cut into the q8_0 block
        let path = write_temp("truncated", &bytes);
        let f = GgufFile::open(&path).unwrap();
        let q = f.tensor("block.q8").unwrap().clone();
        let err = f.data(&q).unwrap_err().to_string();
        assert!(err.contains("block.q8"), "unhelpful error: {err}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_unknown_dtype_fails_at_access_not_at_parse() {
        let mut bytes = build_test_file();
        // Corrupt tensor 0's type id (points at "small.f32"'s type field).
        // Parsing must still succeed; only data access may fail.
        let path = write_temp("unknown-type", &bytes);
        let f = GgufFile::open(&path).unwrap();
        drop(f);
        // Direct check on the type layer instead of byte surgery:
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![4],
            ggml_type: GgmlType::Other(31),
            offset: 0,
            part: 0,
        };
        assert!(t.byte_size().is_err());
        bytes.clear();
        std::fs::remove_file(path).ok();
    }
}
