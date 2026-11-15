//! End-to-end invariants of the dense llama-shaped forward pass: config
//! parsing, determinism, causality, the tied head, truncation.

mod common;

use allpaka_gguf::GgufFile;
use allpaka_model::{Config, Model, RopeStyle};
use common::{
    logits_for, tiny_model, write_temp, FileBuilder, Rng, FFN, HEADS, HEAD_DIM, HIDDEN, KV_HEADS,
    VOCAB,
};

#[test]
fn the_config_reads_what_the_file_says() {
    let path = write_temp("config", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let c = Config::from_gguf(&f).unwrap();
    assert_eq!(c.n_layers, 2);
    assert_eq!(c.vocab, VOCAB as u32);
    assert_eq!(c.head_dim, HEAD_DIM as u32);
    assert_eq!(c.rope_style, RopeStyle::Norm);
    assert!(!c.has_qk_norm);
    assert!((c.rms_eps - 1e-5).abs() < 1e-12);
    std::fs::remove_file(path).ok();
}

#[test]
fn logits_are_finite_deterministic_and_vocab_sized() {
    let path = write_temp("determinism", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let a = logits_for(&f, &[3, 1, 4]);
    let b = logits_for(&f, &[3, 1, 4]);
    assert_eq!(a, b);
    assert_eq!(a[2].len(), VOCAB as usize);
    assert!(a[2].iter().all(|v| v.is_finite()));
    std::fs::remove_file(path).ok();
}

/// The current token must see earlier tokens: attention is the only
/// mechanism connecting them, so this fails if the cache or the scores
/// are wired wrong.
#[test]
fn changing_an_earlier_token_changes_the_current_logits() {
    let path = write_temp("attends", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let a = logits_for(&f, &[3, 1]);
    let b = logits_for(&f, &[5, 1]);
    assert_ne!(a[1], b[1], "the second token ignored the first");
    std::fs::remove_file(path).ok();
}

/// And only earlier ones: two sessions sharing a prefix must agree on the
/// prefix's logits no matter what they feed afterwards.
#[test]
fn logits_depend_only_on_the_prefix() {
    let path = write_temp("causal", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let a = logits_for(&f, &[3, 1, 4]);
    let b = logits_for(&f, &[3, 1, 9]);
    assert_eq!(a[0], b[0]);
    assert_eq!(a[1], b[1]);
    assert_ne!(a[2], b[2]);
    std::fs::remove_file(path).ok();
}

/// A model without output.weight ties the head to the embedding and still
/// produces logits.
#[test]
fn a_tied_output_head_falls_back_to_the_embedding() {
    let bytes = tiny_model("llama", false);
    let path = write_temp("tied-src", &bytes);
    // Rebuild without the output tensor by name surgery: cheaper to just
    // build a fresh file lacking it.
    std::fs::remove_file(&path).ok();
    let mut rng = Rng(7);
    let mut mat =
        |rows: u64, cols: u64| -> Vec<f32> { (0..rows * cols).map(|_| rng.next_f32()).collect() };
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let mut b = FileBuilder::new()
        .str_kv("general.architecture", "llama")
        .u32_kv("llama.block_count", 1)
        .u32_kv("llama.embedding_length", HIDDEN as u32)
        .u32_kv("llama.attention.head_count", HEADS as u32)
        .u32_kv("llama.attention.head_count_kv", KV_HEADS as u32)
        .u32_kv("llama.attention.key_length", HEAD_DIM as u32)
        .u32_kv("llama.feed_forward_length", FFN as u32)
        .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
        .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize]);
    let n = |p: &str| format!("blk.0.{p}.weight");
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
    let path = write_temp("tied", &b.build());
    let f = GgufFile::open(&path).unwrap();
    let out = logits_for(&f, &[2]);
    assert_eq!(out[0].len(), VOCAB as usize);
    assert!(out[0].iter().all(|v| v.is_finite()));
    std::fs::remove_file(path).ok();
}

/// Rolling back and replaying must reproduce the original logits exactly:
/// truncation is what chat prefix-reuse stands on.
#[test]
fn truncate_then_replay_matches_the_original_computation() {
    let path = write_temp("truncate", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let model = Model::load(&f).unwrap();

    let mut s = model.new_session(16);
    model.forward(3, &mut s).unwrap();
    let after_two = model.forward(1, &mut s).unwrap();
    let after_three = model.forward(4, &mut s).unwrap();

    // Roll back to just the first token and replay a different second
    // token, then the original one again.
    s.truncate(1);
    let replay_two = model.forward(1, &mut s).unwrap();
    assert_eq!(replay_two, after_two);
    let replay_three = model.forward(4, &mut s).unwrap();
    assert_eq!(replay_three, after_three);
    std::fs::remove_file(path).ok();
}

#[test]
fn an_out_of_vocab_token_is_an_error() {
    let path = write_temp("oov", &tiny_model("llama", false));
    let f = GgufFile::open(&path).unwrap();
    let model = Model::load(&f).unwrap();
    let mut s = model.new_session(4);
    assert!(model.forward(VOCAB as u32, &mut s).is_err());
    std::fs::remove_file(path).ok();
}
