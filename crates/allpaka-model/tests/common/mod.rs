//! Shared fixtures for the end-to-end tests: a minimal GGUF writer and the
//! tiny deterministic models built with it.

#![allow(dead_code)]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;
use std::io::Write;

/// Deterministic pseudo-random weights: an LCG, so the test model is the
/// same on every run and platform.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to roughly [-0.1, 0.1]: small weights keep activations tame
        // through two layers without normal init math.
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    }
}

pub struct FileBuilder {
    kvs: Vec<u8>,
    kv_count: u64,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl FileBuilder {
    pub fn new() -> Self {
        Self {
            kvs: Vec::new(),
            kv_count: 0,
            tensors: Vec::new(),
        }
    }

    pub fn key(&mut self, k: &str) {
        self.kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
        self.kvs.extend_from_slice(k.as_bytes());
    }

    pub fn str_kv(mut self, k: &str, v: &str) -> Self {
        self.key(k);
        self.kvs.extend_from_slice(&8u32.to_le_bytes());
        self.kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
        self.kvs.extend_from_slice(v.as_bytes());
        self.kv_count += 1;
        self
    }

    pub fn u32_kv(mut self, k: &str, v: u32) -> Self {
        self.key(k);
        self.kvs.extend_from_slice(&4u32.to_le_bytes());
        self.kvs.extend_from_slice(&v.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub fn f32_kv(mut self, k: &str, v: f32) -> Self {
        self.key(k);
        self.kvs.extend_from_slice(&6u32.to_le_bytes());
        self.kvs.extend_from_slice(&v.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// dims in GGUF order: dims[0] is the contiguous (input) dimension.
    pub fn tensor(mut self, name: &str, dims: &[u64], data: Vec<f32>) -> Self {
        assert_eq!(dims.iter().product::<u64>(), data.len() as u64);
        let mut raw = Vec::with_capacity(data.len() * 4);
        for v in data {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors.push((name.into(), dims.to_vec(), 0u32, raw));
        self
    }

    pub fn tensor_q8_0(mut self, name: &str, dims: &[u64], data: Vec<f32>) -> Self {
        assert_eq!(dims.iter().product::<u64>(), data.len() as u64);
        assert_eq!(data.len() % 32, 0);
        let mut raw = Vec::with_capacity(data.len() / 32 * 34);
        let scale_f16 = 0x211Fu16.to_le_bytes(); // ~0.01 in f16
        for chunk in data.chunks_exact(32) {
            raw.extend_from_slice(&scale_f16);
            for &v in chunk {
                let q = (v * 100.0).round().clamp(-127.0, 127.0) as i8;
                raw.push(q as u8);
            }
        }
        self.tensors.push((name.into(), dims.to_vec(), 8u32, raw));
        self
    }

    pub fn build(self) -> Vec<u8> {
        let mut info = Vec::new();
        let mut data = Vec::new();
        for (name, dims, ty, raw) in &self.tensors {
            // Each tensor starts 32-byte aligned inside the data section.
            while data.len() % 32 != 0 {
                data.push(0u8);
            }
            info.extend_from_slice(&(name.len() as u64).to_le_bytes());
            info.extend_from_slice(name.as_bytes());
            info.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                info.extend_from_slice(&d.to_le_bytes());
            }
            info.extend_from_slice(&ty.to_le_bytes());
            info.extend_from_slice(&(data.len() as u64).to_le_bytes());
            data.extend_from_slice(raw);
        }

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kvs);
        out.extend_from_slice(&info);
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&data);
        out
    }
}

pub const HIDDEN: u64 = 8;

pub const HEADS: u64 = 2;

pub const KV_HEADS: u64 = 1;

pub const HEAD_DIM: u64 = 4;

pub const FFN: u64 = 16;

pub const VOCAB: u64 = 11;

pub const LAYERS: u64 = 2;

/// A complete 2-layer llama-shaped model with deterministic weights.
pub fn tiny_model(arch: &str, qk_norm: bool) -> Vec<u8> {
    let mut rng = Rng(42);
    let mut mat =
        |rows: u64, cols: u64| -> Vec<f32> { (0..rows * cols).map(|_| rng.next_f32()).collect() };
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;

    let mut b = FileBuilder::new()
        .str_kv("general.architecture", arch)
        .u32_kv(&format!("{arch}.block_count"), LAYERS as u32)
        .u32_kv(&format!("{arch}.embedding_length"), HIDDEN as u32)
        .u32_kv(&format!("{arch}.attention.head_count"), HEADS as u32)
        .u32_kv(&format!("{arch}.attention.head_count_kv"), KV_HEADS as u32)
        .u32_kv(&format!("{arch}.attention.key_length"), HEAD_DIM as u32)
        .u32_kv(&format!("{arch}.feed_forward_length"), FFN as u32)
        .f32_kv(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
        .f32_kv(&format!("{arch}.rope.freq_base"), 10000.0)
        .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
        .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize])
        .tensor("output.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN));

    for i in 0..LAYERS {
        let n = |p: &str| format!("blk.{i}.{p}.weight");
        b = b
            .tensor(&n("attn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("attn_q"), &[HIDDEN, q_dim], mat(q_dim, HIDDEN))
            .tensor(&n("attn_k"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_v"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_output"), &[q_dim, HIDDEN], mat(HIDDEN, q_dim))
            .tensor(&n("ffn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("ffn_gate"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
            .tensor(&n("ffn_up"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
            .tensor(&n("ffn_down"), &[FFN, HIDDEN], mat(HIDDEN, FFN));
        if qk_norm {
            b = b
                .tensor(&n("attn_q_norm"), &[HEAD_DIM], vec![1.0; HEAD_DIM as usize])
                .tensor(&n("attn_k_norm"), &[HEAD_DIM], vec![1.0; HEAD_DIM as usize]);
        }
    }
    b.build()
}

/// Write `bytes` to a fresh temp file. The name carries the process id and a
/// per-process counter: test binaries of other worktrees, or of this one run
/// concurrently, would otherwise overwrite a file another process still has
/// mapped (on Windows: os error 1224, "user-mapped section open").
pub fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("allpaka-model-{name}-{pid}-{n}.gguf"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(bytes).unwrap();
    path
}

pub fn logits_for(f: &GgufFile, tokens: &[u32]) -> Vec<Vec<f32>> {
    let model = Model::load(f).unwrap();
    let mut s = model.new_session(16);
    tokens
        .iter()
        .map(|&t| model.forward(t, &mut s).unwrap())
        .collect()
}

pub const EXPERTS: u64 = 4;

pub const USED: u64 = 2;

pub const EXPERT_FFN: u64 = 8;

/// A 1-layer qwen3moe-shaped model. `poison` fills one expert's weights
/// with huge values, to prove routing either avoids or hits it.
pub fn tiny_moe(poisoned_expert: Option<usize>, router_bias_expert: usize) -> Vec<u8> {
    let mut rng = Rng(1234);
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let arch = "qwen3moe";

    // Stacked expert tensors: expert-major, each [n_out, n_in] row-major.
    let mut stack = |n_out: u64, n_in: u64, poison: Option<usize>| -> Vec<f32> {
        let mut v = Vec::with_capacity((EXPERTS * n_out * n_in) as usize);
        for e in 0..EXPERTS {
            for _ in 0..n_out * n_in {
                let w = rng.next_f32();
                v.push(if Some(e as usize) == poison { 1e6 } else { w });
            }
        }
        v
    };
    // A router that strongly prefers one expert regardless of input: its
    // row is large-positive on every channel... but sign depends on x, so
    // instead make every OTHER row zero. Zero rows tie; softmax then puts
    // the biased row on top for any x where its dot is positive, and the
    // test drives it with both signs via a symmetric second expert.
    let mut router = vec![0f32; (EXPERTS * HIDDEN) as usize];
    for c in 0..HIDDEN as usize {
        // Biased expert row +w, its mirror row -w: exactly one of the two
        // beats the zero rows for any nonzero x, and both beat ties only
        // through softmax order. Top-2 therefore always contains the pair.
        router[router_bias_expert * HIDDEN as usize + c] = 1.0;
        let mirror = (router_bias_expert + 1) % EXPERTS as usize;
        router[mirror * HIDDEN as usize + c] = -1.0;
    }

    // Materialise the expert stacks before `mat` also borrows the rng.
    let gate_stack = stack(EXPERT_FFN, HIDDEN, poisoned_expert);
    let up_stack = stack(EXPERT_FFN, HIDDEN, poisoned_expert);
    let down_stack = stack(HIDDEN, EXPERT_FFN, poisoned_expert);
    drop(stack);

    let mut mat =
        |rows: u64, cols: u64| -> Vec<f32> { (0..rows * cols).map(|_| rng.next_f32()).collect() };
    let mut b = FileBuilder::new()
        .str_kv("general.architecture", arch)
        .u32_kv(&format!("{arch}.block_count"), 1)
        .u32_kv(&format!("{arch}.embedding_length"), HIDDEN as u32)
        .u32_kv(&format!("{arch}.attention.head_count"), HEADS as u32)
        .u32_kv(&format!("{arch}.attention.head_count_kv"), KV_HEADS as u32)
        .u32_kv(&format!("{arch}.attention.key_length"), HEAD_DIM as u32)
        .u32_kv(&format!("{arch}.feed_forward_length"), FFN as u32)
        .u32_kv(&format!("{arch}.expert_count"), EXPERTS as u32)
        .u32_kv(&format!("{arch}.expert_used_count"), USED as u32)
        .u32_kv(
            &format!("{arch}.expert_feed_forward_length"),
            EXPERT_FFN as u32,
        )
        .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
        .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize])
        .tensor("output.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN));
    let n = |p: &str| format!("blk.0.{p}.weight");
    b = b
        .tensor(&n("attn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
        .tensor(&n("attn_q"), &[HIDDEN, q_dim], mat(q_dim, HIDDEN))
        .tensor(&n("attn_k"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
        .tensor(&n("attn_v"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
        .tensor(&n("attn_output"), &[q_dim, HIDDEN], mat(HIDDEN, q_dim))
        .tensor(&n("ffn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
        .tensor(&n("ffn_gate_inp"), &[HIDDEN, EXPERTS], router)
        .tensor(
            &n("ffn_gate_exps"),
            &[HIDDEN, EXPERT_FFN, EXPERTS],
            gate_stack,
        )
        .tensor(&n("ffn_up_exps"), &[HIDDEN, EXPERT_FFN, EXPERTS], up_stack)
        .tensor(
            &n("ffn_down_exps"),
            &[EXPERT_FFN, HIDDEN, EXPERTS],
            down_stack,
        );
    b.build()
}
