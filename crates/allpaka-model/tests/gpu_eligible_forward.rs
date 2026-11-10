//! End-to-end tests on Neox and MoE models, the shapes that take the GPU
//! path when a backend is attached.
//!
//! They share one process and serialise on `GREEDY_CHAIN_LOCK`: the greedy
//! chain tests toggle `ALLPAKA_GREEDY_CHAIN`, which the others would read
//! mid-run, and the GPU decode counters they assert on are process-global.
//! Keep every test that takes the lock in this file.

mod common;

use allpaka_gguf::GgufFile;
use allpaka_model::{model, Config, Model, RopeStyle};
use common::{logits_for, tiny_model, tiny_moe, write_temp, FileBuilder, Rng, EXPERTS, USED};

#[test]
fn qwen_shaped_files_flip_the_two_switches() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = write_temp("qwen", &tiny_model("qwen3", true));
    let f = GgufFile::open(&path).unwrap();
    let c = Config::from_gguf(&f).unwrap();
    assert_eq!(c.rope_style, RopeStyle::Neox);
    assert!(c.has_qk_norm);
    // And the forward pass runs with the extra norms in place.
    let out = logits_for(&f, &[1, 2, 3]);
    assert!(out[2].iter().all(|v| v.is_finite()));
    std::fs::remove_file(path).ok();
}

/// Batched prefill is an optimisation, not a different computation: the
/// last-token logits must match feeding the same tokens one at a time,
/// and the cache it leaves behind must continue identically.
#[test]
fn forward_batch_matches_sequential_forward() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for (arch, qk) in [("llama", false), ("qwen3", true)] {
        let path = write_temp(&format!("batch-{arch}"), &tiny_model(arch, qk));
        let f = GgufFile::open(&path).unwrap();
        let model = Model::load(&f).unwrap();

        let seq = logits_for(&f, &[3, 1, 4, 1, 5]);

        let mut s = model.new_session(16);
        let batched = model.forward_batch(&[3, 1, 4, 1, 5], &mut s).unwrap();
        let diff = batched
            .iter()
            .zip(&seq[4])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "{arch}: batched logits diverge by {diff}");

        // And decoding continues from the batched cache identically.
        let next_from_batch = model.forward(9, &mut s).unwrap();
        let mut s2 = model.new_session(16);
        for &t in &[3u32, 1, 4, 1, 5] {
            model.forward(t, &mut s2).unwrap();
        }
        let next_from_seq = model.forward(9, &mut s2).unwrap();
        let diff = next_from_batch
            .iter()
            .zip(&next_from_seq)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "{arch}: post-batch decode diverges by {diff}");
        std::fs::remove_file(path).ok();
    }
}

/// The MoE path routes per token inside a batch too.
#[test]
fn forward_batch_matches_sequential_on_a_moe() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = write_temp("batch-moe", &tiny_moe(None, 0));
    let f = GgufFile::open(&path).unwrap();
    let model = Model::load(&f).unwrap();

    let seq = logits_for(&f, &[3, 1, 4]);
    let mut s = model.new_session(16);
    let batched = model.forward_batch(&[3, 1, 4], &mut s).unwrap();
    let diff = batched
        .iter()
        .zip(&seq[2])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 1e-4, "moe batched logits diverge by {diff}");
    std::fs::remove_file(path).ok();
}

#[test]
fn route_picks_the_top_k_and_renormalises() {
    let picked = model::route(&[0.1, 0.5, 0.15, 0.25], 2);
    assert_eq!(picked.len(), 2);
    assert_eq!(picked[0].0, 1);
    assert_eq!(picked[1].0, 3);
    // 0.5 and 0.25 renormalised to sum 1.
    assert!((picked[0].1 - 2.0 / 3.0).abs() < 1e-6);
    assert!((picked[1].1 - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn a_moe_model_produces_finite_deterministic_logits() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = write_temp("moe-runs", &tiny_moe(None, 0));
    let f = GgufFile::open(&path).unwrap();
    let c = Config::from_gguf(&f).unwrap();
    let moe = c.moe.as_ref().expect("config should see the experts");
    assert_eq!(moe.n_expert, EXPERTS as u32);
    assert_eq!(moe.n_used, USED as u32);

    let a = logits_for(&f, &[3, 1, 4]);
    let b = logits_for(&f, &[3, 1, 4]);
    assert_eq!(a, b);
    assert!(a[2].iter().all(|v| v.is_finite()));
    std::fs::remove_file(path).ok();
}

/// The point of routing: experts the router does not pick must not touch
/// the output. Poisoning an unpicked expert with 1e6 weights changes
/// nothing; poisoning a picked one blows the logits up visibly.
#[test]
fn unrouted_experts_do_not_influence_the_output() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let clean = write_temp("moe-clean", &tiny_moe(None, 0));
    // Experts 0 and 1 form the always-picked pair (router rows +-1);
    // experts 2 and 3 have zero router rows and lose every top-2.
    let poison_unused = write_temp("moe-poison-unused", &tiny_moe(Some(3), 0));
    let poison_used = write_temp("moe-poison-used", &tiny_moe(Some(0), 0));

    let a = logits_for(&GgufFile::open(&clean).unwrap(), &[3, 1]);
    let b = logits_for(&GgufFile::open(&poison_unused).unwrap(), &[3, 1]);
    let c = logits_for(&GgufFile::open(&poison_used).unwrap(), &[3, 1]);

    assert_eq!(a, b, "an unrouted expert leaked into the output");
    assert_ne!(a, c, "a routed expert was ignored");
    for path in [clean, poison_unused, poison_used] {
        std::fs::remove_file(path).ok();
    }
}

static GREEDY_CHAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
// The decode-path counters are only asserted on Metal.
#[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
fn gpu_greedy_chain_matches_step_by_step_on_128_head_dim() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(12345);
    let mut mat =
        |rows: u64, cols: u64| -> Vec<f32> { (0..rows * cols).map(|_| rng.next_f32()).collect() };
    let hidden = 128u64;
    let heads = 2u64;
    let kv_heads = 1u64;
    let head_dim = 128u64;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let ffn = 128u64;
    let vocab = 32u64;
    let layers = 2u64;

    let mut b = FileBuilder::new()
        .str_kv("general.architecture", "qwen3")
        .u32_kv("qwen3.block_count", layers as u32)
        .u32_kv("qwen3.embedding_length", hidden as u32)
        .u32_kv("qwen3.attention.head_count", heads as u32)
        .u32_kv("qwen3.attention.head_count_kv", kv_heads as u32)
        .u32_kv("qwen3.attention.key_length", head_dim as u32)
        .u32_kv("qwen3.feed_forward_length", ffn as u32)
        .f32_kv("qwen3.attention.layer_norm_rms_epsilon", 1e-5)
        .f32_kv("qwen3.rope.freq_base", 10000.0)
        .tensor_q8_0("token_embd.weight", &[hidden, vocab], mat(vocab, hidden))
        .tensor("output_norm.weight", &[hidden], vec![1.0; hidden as usize])
        .tensor_q8_0("output.weight", &[hidden, vocab], mat(vocab, hidden));

    for i in 0..layers {
        let n = |p: &str| format!("blk.{i}.{p}.weight");
        b = b
            .tensor(&n("attn_norm"), &[hidden], vec![1.0; hidden as usize])
            .tensor_q8_0(&n("attn_q"), &[hidden, q_dim], mat(q_dim, hidden))
            .tensor_q8_0(&n("attn_k"), &[hidden, kv_dim], mat(kv_dim, hidden))
            .tensor_q8_0(&n("attn_v"), &[hidden, kv_dim], mat(kv_dim, hidden))
            .tensor_q8_0(&n("attn_output"), &[q_dim, hidden], mat(hidden, q_dim))
            .tensor(&n("ffn_norm"), &[hidden], vec![1.0; hidden as usize])
            .tensor_q8_0(&n("ffn_gate"), &[hidden, ffn], mat(ffn, hidden))
            .tensor_q8_0(&n("ffn_up"), &[hidden, ffn], mat(ffn, hidden))
            .tensor_q8_0(&n("ffn_down"), &[ffn, hidden], mat(hidden, ffn));
    }
    let mut raw = b.build();
    while raw.len() % 16384 != 0 {
        raw.push(0);
    }
    let path = write_temp("greedy-chain-128", &raw);
    let f = GgufFile::open(&path).unwrap();
    let model = Model::load(&f).unwrap();
    let prompt = [1u32, 5, 12];

    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "0");
    let mut s1 = model.new_session(32);
    let logits = model.forward_batch(&prompt, &mut s1).unwrap();
    let seed = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap();
    let ref_tokens = model.forward_greedy_n(seed, &mut s1, 4).unwrap();

    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "1");
    let mut s2 = model.new_session(32);
    let _ = model.forward_batch(&prompt, &mut s2).unwrap();
    let gpu_before = allpaka_backend::gpu::stats();
    let before = allpaka_backend::gpu::decode_path_stats();
    let chain_tokens = model.forward_greedy_n(seed, &mut s2, 4).unwrap();
    let after = allpaka_backend::gpu::decode_path_stats();
    let gpu_after = allpaka_backend::gpu::stats();
    let waits = gpu_after.0 - gpu_before.0;

    assert_eq!(
        ref_tokens, chain_tokens,
        "greedy chain diverged from step-by-step"
    );
    #[cfg(target_os = "macos")]
    {
        assert!(
            allpaka_backend::gpu::is_attached(),
            "Metal GPU should be attached and shader library compiled cleanly: {:?}",
            allpaka_backend::gpu::init_error()
        );
        assert_eq!(after.attempts - before.attempts, 4);
        assert_eq!(after.successes - before.successes, 4);
        assert_eq!(after.declines - before.declines, 0);
        assert_eq!(
            waits, 2,
            "expected 1 wait for seed + 1 wait for 3-token continuation chain"
        );
    }
    std::fs::remove_file(path).ok();
}

#[test]
// The decode-path counters are only asserted on Metal.
#[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
fn gpu_moe_greedy_chain_matches_step_by_step() {
    let _guard = GREEDY_CHAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(67890);
    let mut mat =
        |rows: u64, cols: u64| -> Vec<f32> { (0..rows * cols).map(|_| rng.next_f32()).collect() };
    let hidden = 128u64;
    let heads = 2u64;
    let kv_heads = 1u64;
    let head_dim = 128u64;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let ffn = 128u64;
    let experts = 4u64;
    let used = 2u64;
    let expert_ffn = 64u64;
    let vocab = 32u64;
    let layers = 2u64;
    let arch = "qwen3moe";

    let mut b = FileBuilder::new()
        .str_kv("general.architecture", arch)
        .u32_kv(&format!("{arch}.block_count"), layers as u32)
        .u32_kv(&format!("{arch}.embedding_length"), hidden as u32)
        .u32_kv(&format!("{arch}.attention.head_count"), heads as u32)
        .u32_kv(&format!("{arch}.attention.head_count_kv"), kv_heads as u32)
        .u32_kv(&format!("{arch}.attention.key_length"), head_dim as u32)
        .u32_kv(&format!("{arch}.feed_forward_length"), ffn as u32)
        .u32_kv(&format!("{arch}.expert_count"), experts as u32)
        .u32_kv(&format!("{arch}.expert_used_count"), used as u32)
        .u32_kv(
            &format!("{arch}.expert_feed_forward_length"),
            expert_ffn as u32,
        )
        .f32_kv(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
        .f32_kv(&format!("{arch}.rope.freq_base"), 10000.0)
        .tensor_q8_0("token_embd.weight", &[hidden, vocab], mat(vocab, hidden))
        .tensor("output_norm.weight", &[hidden], vec![1.0; hidden as usize])
        .tensor_q8_0("output.weight", &[hidden, vocab], mat(vocab, hidden));

    for i in 0..layers {
        let n = |p: &str| format!("blk.{i}.{p}.weight");
        b = b
            .tensor(&n("attn_norm"), &[hidden], vec![1.0; hidden as usize])
            .tensor_q8_0(&n("attn_q"), &[hidden, q_dim], mat(q_dim, hidden))
            .tensor_q8_0(&n("attn_k"), &[hidden, kv_dim], mat(kv_dim, hidden))
            .tensor_q8_0(&n("attn_v"), &[hidden, kv_dim], mat(kv_dim, hidden))
            .tensor_q8_0(&n("attn_output"), &[q_dim, hidden], mat(hidden, q_dim))
            .tensor(&n("ffn_norm"), &[hidden], vec![1.0; hidden as usize])
            .tensor(&n("ffn_gate_inp"), &[hidden, experts], mat(experts, hidden))
            .tensor_q8_0(
                &n("ffn_gate_exps"),
                &[hidden, expert_ffn, experts],
                mat(experts * expert_ffn, hidden),
            )
            .tensor_q8_0(
                &n("ffn_up_exps"),
                &[hidden, expert_ffn, experts],
                mat(experts * expert_ffn, hidden),
            )
            .tensor_q8_0(
                &n("ffn_down_exps"),
                &[expert_ffn, hidden, experts],
                mat(experts * hidden, expert_ffn),
            );
    }
    let mut raw = b.build();
    while raw.len() % 16384 != 0 {
        raw.push(0);
    }
    let path = write_temp("moe-greedy-chain-128", &raw);
    let f = GgufFile::open(&path).unwrap();
    let model = Model::load(&f).unwrap();
    let prompt = [2u32, 7, 14];

    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "0");
    let mut s1 = model.new_session(32);
    let logits = model.forward_batch(&prompt, &mut s1).unwrap();
    let seed = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap();
    let ref_tokens = model.forward_greedy_n(seed, &mut s1, 4).unwrap();

    std::env::set_var("ALLPAKA_GREEDY_CHAIN", "1");
    let mut s2 = model.new_session(32);
    let _ = model.forward_batch(&prompt, &mut s2).unwrap();
    let gpu_before = allpaka_backend::gpu::stats();
    let before = allpaka_backend::gpu::decode_path_stats();
    let chain_tokens = model.forward_greedy_n(seed, &mut s2, 4).unwrap();
    let after = allpaka_backend::gpu::decode_path_stats();
    let gpu_after = allpaka_backend::gpu::stats();
    let waits = gpu_after.0 - gpu_before.0;

    assert_eq!(
        ref_tokens, chain_tokens,
        "MoE greedy chain diverged from step-by-step"
    );
    #[cfg(target_os = "macos")]
    {
        assert!(
            allpaka_backend::gpu::is_attached(),
            "Metal GPU should be attached: {:?}",
            allpaka_backend::gpu::init_error()
        );
        assert_eq!(after.attempts - before.attempts, 4);
        assert_eq!(after.successes - before.successes, 4);
        assert_eq!(after.declines - before.declines, 0);
        assert_eq!(
            waits, 2,
            "expected 1 wait for seed + 1 wait for 3-token MoE continuation chain"
        );
    }
    std::fs::remove_file(path).ok();
}
