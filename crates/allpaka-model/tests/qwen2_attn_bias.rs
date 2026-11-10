//! The qwen2 (Qwen2.5) q/k/v projection-bias path: detected from the file,
//! a no-op when zero, effective when not, and identical between the batched
//! prefill and the single-token decode.

mod common;

use allpaka_gguf::GgufFile;
use allpaka_model::{Config, Model, RopeStyle};
use common::{logits_for, tiny_model, tiny_qwen2_with_bias, write_temp};

fn open(name: &str, bytes: &[u8]) -> GgufFile {
    let path = write_temp(name, bytes);
    GgufFile::open(&path).unwrap()
}

#[test]
fn qwen2_config_sees_the_bias_and_no_qk_norm() {
    let f = open("qwen2-cfg", &tiny_qwen2_with_bias([0.5; 3]));
    let c = Config::from_gguf(&f).unwrap();
    assert!(c.has_attn_bias);
    assert!(!c.has_qk_norm);
    assert_eq!(c.rope_style, RopeStyle::Neox);
}

#[test]
fn zero_bias_equals_no_bias() {
    let plain = open("qwen2-plain", &tiny_model("qwen2", false));
    let zero = open("qwen2-zero", &tiny_qwen2_with_bias([0.0; 3]));
    let toks = [3, 1, 4, 1, 5];
    assert_eq!(logits_for(&plain, &toks), logits_for(&zero, &toks));
}

/// Each of q, k and v on its own must reach the output: dropping any one
/// bias (the class of bug a GPU path can introduce) fails here.
#[test]
fn each_bias_alone_changes_the_logits() {
    let zero = open("qwen2-zero2", &tiny_qwen2_with_bias([0.0; 3]));
    let toks = [3, 1, 4];
    let a = logits_for(&zero, &toks);
    for (which, scale) in [("q", [2.0, 0.0, 0.0]), ("k", [0.0, 2.0, 0.0]), ("v", [0.0, 0.0, 2.0])] {
        let biased = open(&format!("qwen2-bias-{which}"), &tiny_qwen2_with_bias(scale));
        let b = logits_for(&biased, &toks);
        assert_ne!(a[2], b[2], "the {which} bias was not applied");
    }
}

#[test]
fn biased_prefill_matches_biased_decode() {
    let f = open("qwen2-paths", &tiny_qwen2_with_bias([2.0; 3]));
    let toks = [3u32, 1, 4, 1, 5];
    let step = logits_for(&f, &toks);
    let model = Model::load(&f).unwrap();
    let mut s = model.new_session(16);
    let batch = model.forward_batch(&toks, &mut s).unwrap();
    for (i, (a, b)) in batch.iter().zip(step.last().unwrap()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3 * (1.0 + b.abs()),
            "logit {i}: prefill {a} vs decode {b}"
        );
    }
}
