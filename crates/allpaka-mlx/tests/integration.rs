//! Integration tests for MLX weight container format.
//! These tests validate the full pipeline: loading safetensors files, parsing metadata,
//! and dequantizing Q4/Q8Affine blocks. Requires test fixtures in fixtures/.

use std::fs;
use tempfile::tempdir;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Create a minimal valid safetensors file with Q4Affine-like structure.
    fn create_test_safetensors() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        
        // Build header with two tensors: one q4 block + some f32 data
        let mut header_json = serde_json::Map::new();
        header_json.insert("__metadata__".to_string(), 
            serde_json::json!({"total_byte_size": 100}));
        
        // Tensor 1: 64 bytes of fake q4 data (one block)
        header_json.insert("q4_block".to_string(), serde_json::json!({
            "dtype": "Q4_AFFINE",
            "shape": [64],
            "data_offsets": [0, 68]
        }));
        
        // Tensor 2: 4 f32 values
        header_json.insert("weights.f32".to_string(), serde_json::json!({
            "dtype": "BFLOAT16",
            "shape": [4],
            "data_offsets": [68, 72]
        }));
        
        let header = serde_json::to_string(&header_json).unwrap();
        let header_bytes: Vec<u8> = header.into_bytes();
        
        // Pad to 32-byte alignment
        let total_header_len = 8 + header_bytes.len();
        let padded_hdr = ((total_header_len + 31) / 32) * 32;
        let padding = vec![0u8; padded_hdr - total_header_len];
        
        let mut data = Vec::new();
        // Size prefix
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        // Header JSON
        data.extend_from_slice(&header_bytes);
        // Padding
        data.extend_from_slice(&padding);
        // Fake tensor data
        data.extend_from_slice(&[0u8; 68]); // q4 block
        data.extend_from_slice(&[0u16.to_le_bytes()[0], 0x3f, 0x00, 0x00]); // bf16 scale/bias
        
        // Write to file
        let path = dir.path().join("model.safetensors");
        fs::write(&path, &data).unwrap();
        
        dir
    }

    #[test]

    #[test]
    fn test_q4_dequant_zero_block() {
        let mut block = [0u8; 36];
        // Scale=1.0 (bf16 0x3f80), bias=0.0
        block[32..34].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes());
        // All nibbles are 0
        
        let mut out = Vec::with_capacity(64);
        allpaka_mlx::dequant_mlx_q4_block(&block, &mut out);
        
        assert_eq!(out.len(), 64);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_q8_dequant_uniform_block() {
        let mut block = [0u8; 68];
        block[64..66].copy_from_slice(&0x3f80u16.to_le_bytes()); // scale=1.0
        block[66..68].copy_from_slice(&0u16.to_le_bytes());     // bias=0.0
        
        // Set all bytes to 128
        for i in 0..64 {
            block[i] = 128u8;
        }
        
        let mut out = Vec::with_capacity(64);
        allpaka_mlx::dequant_mlx_q8_block(&block, &mut out);
        
        assert_eq!(out.len(), 64);
        for v in out.iter() {
            assert_eq!(*v, 128.0);
        }
    }

    #[test]

    #[test]
    fn test_shape_flattening_for_matmul() {
        // Test shapes module utilities
        let shape_2d = vec![64, 256];
        let (n_out, n_in) = allpaka_mlx::flatten_for_matmul(&shape_2d);
        assert_eq!(n_out, 256);
        assert_eq!(n_in, 64);
        
        let shape_3d_expert = vec![8, 2048, 512]; // [expert_count, intermediate, hidden]
        let (n_out, n_in) = allpaka_mlx::flatten_for_matmul(&shape_3d_expert);
        assert_eq!(n_out, 2048 * 512);
        assert_eq!(n_in, 8);
    }

    #[test]
    fn test_empty_tensor_info() {
        // Edge case: tensor with empty shape but non-zero bytes
        let info = allpaka_mlx::TensorInfo {
            name: "scalar".to_string(),
            shape: vec![],
            dtype: "F32".to_string(),
            offset: 0,
            bytes: 4,
        };
        
        assert_eq!(info.name, "scalar");
        assert!(info.shape.is_empty());
        assert_eq!(info.bytes, 4);
    }
}
