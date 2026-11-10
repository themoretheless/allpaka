//! GPU-vs-CPU parity for `RopeStyle::Norm` (Llama / Mistral adjacent-pair RoPE)
//! across every Metal GPU execution path:
//! 1. Fused prefill (`forward_batch` with `attend_mm_fq`),
//! 2. Batched `qk_prep_batch` prefill (`ALLPAKA_PF_FQK=0`),
//! 3. Whole-token decode (`forward`, verifying `decode_path_stats` succeeds),
//! 4. On-device greedy continuation chain (`forward_greedy_n`),
//! 5. Multi-row speculative verification (`verify_tokens`),
//! 6. Per-layer `attn_block` (`ALLPAKA_NO_TOKENBUF=1` without `ALLPAKA_CPU_ATTN`).

#![cfg(target_os = "macos")]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;
use std::io::Write;
use std::path::PathBuf;

const HIDDEN: usize = 256;
const HEAD_DIM: usize = 128;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const FFN_HIDDEN: usize = 512;
const VOCAB: usize = 64;
const N_LAYERS: usize = 2;

fn align32(n: usize) -> usize {
    (n + 31) & !31
}

fn next_u32(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

/// Deterministic Q8_0 payload for `[n_in, n_out]` (row-major `n_out` rows of `n_in`).
/// Scale = 2^-7 (f16 `0x2000` = 0.0078125) keeps multi-layer activations crisp and bounded.
fn gen_q8_0(n_out: usize, n_in: usize, seed: &mut u64) -> Vec<u8> {
    assert_eq!(n_in % 32, 0);
    let blocks = n_out * (n_in / 32);
    let mut out = Vec::with_capacity(blocks * 34);
    for _ in 0..blocks {
        out.extend_from_slice(&0x2000u16.to_le_bytes());
        for _ in 0..32 {
            let v = ((next_u32(seed) % 31) as i32 - 15) as i8;
            out.push(v as u8);
        }
    }
    out
}

fn gen_norm_f32(n: usize, seed: &mut u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let r = (next_u32(seed) % 401) as f32 / 1000.0; // 0.0 .. 0.4
        let w = 0.8 + r;
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn build_synthetic_llama_gguf(partial_rope: bool) -> Vec<u8> {
    let mut kvs = Vec::new();
    let mut kv_count = 0u64;

    fn kv_str(kvs: &mut Vec<u8>, kv_count: &mut u64, k: &str, v: &str) {
        kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
        kvs.extend_from_slice(k.as_bytes());
        kvs.extend_from_slice(&8u32.to_le_bytes());
        kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
        kvs.extend_from_slice(v.as_bytes());
        *kv_count += 1;
    }
    fn kv_u32(kvs: &mut Vec<u8>, kv_count: &mut u64, k: &str, v: u32) {
        kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
        kvs.extend_from_slice(k.as_bytes());
        kvs.extend_from_slice(&4u32.to_le_bytes());
        kvs.extend_from_slice(&v.to_le_bytes());
        *kv_count += 1;
    }
    fn kv_f32(kvs: &mut Vec<u8>, kv_count: &mut u64, k: &str, v: f32) {
        kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
        kvs.extend_from_slice(k.as_bytes());
        kvs.extend_from_slice(&6u32.to_le_bytes());
        kvs.extend_from_slice(&v.to_le_bytes());
        *kv_count += 1;
    }

    kv_str(&mut kvs, &mut kv_count, "general.architecture", "llama");
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.block_count",
        N_LAYERS as u32,
    );
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.embedding_length",
        HIDDEN as u32,
    );
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.attention.head_count",
        N_HEADS as u32,
    );
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.attention.head_count_kv",
        N_KV_HEADS as u32,
    );
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.attention.key_length",
        HEAD_DIM as u32,
    );
    kv_u32(
        &mut kvs,
        &mut kv_count,
        "llama.feed_forward_length",
        FFN_HIDDEN as u32,
    );
    kv_f32(
        &mut kvs,
        &mut kv_count,
        "llama.attention.layer_norm_rms_epsilon",
        1e-5,
    );
    kv_f32(&mut kvs, &mut kv_count, "llama.rope.freq_base", 10000.0);
    if partial_rope {
        kv_u32(&mut kvs, &mut kv_count, "llama.rope.dimension_count", 64);
    }

    struct RawTensor {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        bytes: Vec<u8>,
    }

    let mut seed = 0x51a7_a001_2026_1111u64;
    let mut tensors = Vec::new();
    tensors.push(RawTensor {
        name: "token_embd.weight".into(),
        dims: vec![HIDDEN as u64, VOCAB as u64],
        ty: 8, // Q8_0
        bytes: gen_q8_0(VOCAB, HIDDEN, &mut seed),
    });
    tensors.push(RawTensor {
        name: "output_norm.weight".into(),
        dims: vec![HIDDEN as u64],
        ty: 0, // F32
        bytes: gen_norm_f32(HIDDEN, &mut seed),
    });
    tensors.push(RawTensor {
        name: "output.weight".into(),
        dims: vec![HIDDEN as u64, VOCAB as u64],
        ty: 8,
        bytes: gen_q8_0(VOCAB, HIDDEN, &mut seed),
    });

    for l in 0..N_LAYERS {
        tensors.push(RawTensor {
            name: format!("blk.{l}.attn_norm.weight"),
            dims: vec![HIDDEN as u64],
            ty: 0,
            bytes: gen_norm_f32(HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.attn_q.weight"),
            dims: vec![HIDDEN as u64, (N_HEADS * HEAD_DIM) as u64],
            ty: 8,
            bytes: gen_q8_0(N_HEADS * HEAD_DIM, HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.attn_k.weight"),
            dims: vec![HIDDEN as u64, KV_DIM as u64],
            ty: 8,
            bytes: gen_q8_0(KV_DIM, HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.attn_v.weight"),
            dims: vec![HIDDEN as u64, KV_DIM as u64],
            ty: 8,
            bytes: gen_q8_0(KV_DIM, HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.attn_output.weight"),
            dims: vec![(N_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
            ty: 8,
            bytes: gen_q8_0(HIDDEN, N_HEADS * HEAD_DIM, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.ffn_norm.weight"),
            dims: vec![HIDDEN as u64],
            ty: 0,
            bytes: gen_norm_f32(HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.ffn_gate.weight"),
            dims: vec![HIDDEN as u64, FFN_HIDDEN as u64],
            ty: 8,
            bytes: gen_q8_0(FFN_HIDDEN, HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.ffn_up.weight"),
            dims: vec![HIDDEN as u64, FFN_HIDDEN as u64],
            ty: 8,
            bytes: gen_q8_0(FFN_HIDDEN, HIDDEN, &mut seed),
        });
        tensors.push(RawTensor {
            name: format!("blk.{l}.ffn_down.weight"),
            dims: vec![FFN_HIDDEN as u64, HIDDEN as u64],
            ty: 8,
            bytes: gen_q8_0(HIDDEN, FFN_HIDDEN, &mut seed),
        });
    }

    let mut tinfo = Vec::new();
    let mut data_off = 0usize;
    for t in &tensors {
        tinfo.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        tinfo.extend_from_slice(t.name.as_bytes());
        tinfo.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            tinfo.extend_from_slice(&d.to_le_bytes());
        }
        tinfo.extend_from_slice(&t.ty.to_le_bytes());
        tinfo.extend_from_slice(&(data_off as u64).to_le_bytes());
        data_off = align32(data_off + t.bytes.len());
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&kv_count.to_le_bytes());
    out.extend_from_slice(&kvs);
    out.extend_from_slice(&tinfo);
    while out.len() % 32 != 0 {
        out.push(0);
    }
    let data_base = out.len();
    for t in &tensors {
        out.extend_from_slice(&t.bytes);
        while (out.len() - data_base) % 32 != 0 {
            out.push(0);
        }
    }
    out
}

fn write_temp_gguf(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "allpaka-norm-rope-{tag}-{}.gguf",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(bytes).unwrap();
    path
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn reset_env() {
    std::env::remove_var("ALLPAKA_CPU_ATTN");
    std::env::remove_var("ALLPAKA_NO_TOKENBUF");
    std::env::remove_var("ALLPAKA_PF_FQK");
    std::env::remove_var("ALLPAKA_GREEDY_CHAIN");
}

fn verify_llama_gpu_parity(tag: &str, file: &GgufFile) {
    reset_env();
    let model = Model::load(file).unwrap();
    assert_eq!(
        model.config.rope_style,
        allpaka_model::RopeStyle::Norm,
        "expected Llama Norm RoPE"
    );

    // 20 tokens >= MM_MIN_M (16), exercising GPU tile-matmul prefill!
    let prompt: Vec<u32> = (0..20).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect();

    // 1. CPU reference prefill + 4-step decode.
    std::env::set_var("ALLPAKA_CPU_ATTN", "1");
    std::env::set_var("ALLPAKA_NO_TOKENBUF", "1");
    let mut s_cpu = model.new_session(64);
    let pf_cpu = model.forward_batch(&prompt, &mut s_cpu).unwrap();
    let mut cpu_steps = Vec::new();
    let mut next_tok = argmax(&pf_cpu);
    for _ in 0..4 {
        let logits = model.forward(next_tok, &mut s_cpu).unwrap();
        next_tok = argmax(&logits);
        cpu_steps.push(logits);
    }

    // 2. GPU prefill (`attend_mm_fq` default) + whole-token GPU decode (`forward`).
    reset_env();
    let mut s_gpu = model.new_session(64);
    let pf_gpu = model.forward_batch(&prompt, &mut s_gpu).unwrap();
    let pf_diff = max_abs_diff(&pf_cpu, &pf_gpu);
    assert!(
        pf_diff < 0.03,
        "{tag}: fused GPU prefill diverged from CPU (max diff {pf_diff})"
    );
    assert_eq!(argmax(&pf_cpu), argmax(&pf_gpu), "{tag}: prefill argmax");

    let stats_before = allpaka_backend::gpu::decode_path_stats();
    let mut gpu_steps = Vec::new();
    let mut tok = argmax(&pf_gpu);
    for _ in 0..4 {
        let logits = model.forward(tok, &mut s_gpu).unwrap();
        tok = argmax(&logits);
        gpu_steps.push(logits);
    }
    let stats_after = allpaka_backend::gpu::decode_path_stats();
    assert_eq!(
        stats_after.successes - stats_before.successes,
        4,
        "{tag}: expected whole-token GPU decode to engage on all 4 steps"
    );
    for (step, (c, g)) in cpu_steps.iter().zip(&gpu_steps).enumerate() {
        let d = max_abs_diff(c, g);
        assert!(
            d < 0.02,
            "{tag} decode step {step}: GPU whole-token diff {d}"
        );
        assert_eq!(argmax(c), argmax(g), "{tag} decode step {step}: argmax");
    }

    // 3. GPU prefill with `ALLPAKA_PF_FQK=0` (exercises `qk_prep_batch` separately!).
    std::env::set_var("ALLPAKA_PF_FQK", "0");
    let mut s_batch_qk = model.new_session(64);
    let pf_batch_qk = model.forward_batch(&prompt, &mut s_batch_qk).unwrap();
    std::env::remove_var("ALLPAKA_PF_FQK");
    let bqk_diff = max_abs_diff(&pf_cpu, &pf_batch_qk);
    assert!(
        bqk_diff < 0.03,
        "{tag}: qk_prep_batch prefill diverged from CPU (max diff {bqk_diff})"
    );

    // 4. Per-layer GPU `attn_block` (`ALLPAKA_NO_TOKENBUF=1` without `ALLPAKA_CPU_ATTN`).
    std::env::set_var("ALLPAKA_NO_TOKENBUF", "1");
    let mut s_ab = model.new_session(64);
    let pf_ab = model.forward_batch(&prompt, &mut s_ab).unwrap();
    let mut tok_ab = argmax(&pf_ab);
    for (step, c) in cpu_steps.iter().enumerate() {
        let logits = model.forward(tok_ab, &mut s_ab).unwrap();
        tok_ab = argmax(&logits);
        let d = max_abs_diff(c, &logits);
        assert!(d < 0.02, "{tag} attn_block step {step}: diff {d}");
    }
    reset_env();

    // 5. On-device greedy continuation chain (`forward_greedy_n`).
    let seed_tok = argmax(&pf_gpu);
    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "0");
    let mut s_gref = model.new_session(64);
    let _ = model.forward_batch(&prompt, &mut s_gref).unwrap();
    let want_chain = model.forward_greedy_n(seed_tok, &mut s_gref, 4).unwrap();

    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "1");
    let mut s_gchain = model.new_session(64);
    let _ = model.forward_batch(&prompt, &mut s_gchain).unwrap();
    let ch_before = allpaka_backend::gpu::decode_path_stats();
    let got_chain = model.forward_greedy_n(seed_tok, &mut s_gchain, 4).unwrap();
    let ch_after = allpaka_backend::gpu::decode_path_stats();
    assert_eq!(got_chain, want_chain, "{tag}: greedy chain mismatch");
    assert_eq!(ch_after.successes - ch_before.successes, 4);
    reset_env();

    // 6. Speculative multi-token verification (`verify_tokens`).
    let mut s_ver = model.new_session_mtp(64);
    let _ = model.forward_batch_hidden(&prompt, &mut s_ver).unwrap();
    let verify_in = [seed_tok, want_chain[0], want_chain[1], want_chain[2]];
    let (ver_argmax, _ver_hidden) = model
        .verify_tokens(&verify_in, &mut s_ver)
        .unwrap()
        .expect("verify_tokens should engage on Llama Norm RoPE");
    assert_eq!(
        &ver_argmax[..],
        &want_chain[..4],
        "{tag}: verify_tokens row argmax diverged from greedy decode"
    );
}

#[test]
fn llama_norm_rope_gpu_paths_match_cpu_full_and_partial_rotary() {
    let f1 = build_synthetic_llama_gguf(false);
    let f2 = build_synthetic_llama_gguf(true);
    let p1 = write_temp_gguf("full128", &f1);
    let p2 = write_temp_gguf("half64", &f2);
    // Keep both GgufFiles alive until both parity checks complete so macOS
    // cannot recycle the unmapped virtual address range onto the second file
    // while the Metal buffer-window registry still holds the first mapping!
    let g1 = GgufFile::open(&p1).unwrap();
    let g2 = GgufFile::open(&p2).unwrap();
    verify_llama_gpu_parity("full128", &g1);
    verify_llama_gpu_parity("half64", &g2);
    drop(g2);
    drop(g1);
    std::fs::remove_file(p1).ok();
    std::fs::remove_file(p2).ok();
}
