//! Llama and Mistral rotate adjacent pairs of a head (NORM-style RoPE); the
//! fast paths rotate its two halves (NEOX). The loader reorders the rows of Wq
//! and Wk so that one becomes the other. That must change no logit: the model
//! loaded in the file's order, on the reference path, is the truth.
//!
//! One test, because it flips a process-wide environment variable.

mod common;

use allpaka_gguf::GgufFile;
use allpaka_model::{Model, RopeStyle};
use common::{tiny_model, write_temp};

/// Step decode of `tokens`, then the same tokens as one prefill batch.
fn run(model: &Model, tokens: &[u32]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut session = model.new_session(16);
    let steps = tokens
        .iter()
        .map(|&t| model.forward(t, &mut session).unwrap())
        .collect();
    let mut session = model.new_session(16);
    let batch = model.forward_batch(tokens, &mut session).unwrap();
    (steps, batch)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn reordered_llama_matches_the_file_order() {
    let tokens = [3u32, 1, 4, 1, 5, 9, 2, 6];
    for arch in ["llama", "mistral"] {
        let path = write_temp(&format!("{arch}-rope"), &tiny_model(arch, false));
        let file = GgufFile::open(&path).unwrap();

        // SAFETY (set_var/remove_var): this file holds a single test, so no
        // other thread of the process reads the environment meanwhile.
        unsafe { std::env::set_var("ALLPAKA_ROPE_NORM", "1") };
        let native = Model::load(&file).unwrap();
        assert!(matches!(native.config.rope_style, RopeStyle::Norm));
        let (native_steps, native_batch) = run(&native, &tokens);
        drop(native);

        unsafe { std::env::remove_var("ALLPAKA_ROPE_NORM") };
        let reordered = Model::load(&file).unwrap();
        assert!(
            matches!(reordered.config.rope_style, RopeStyle::Neox),
            "{arch} was not loaded in NEOX order"
        );
        let (steps, batch) = run(&reordered, &tokens);

        // Position 0 rotates by nothing, so only later steps can tell a wrong
        // pairing from a right one; all of them must agree.
        for (i, (a, b)) in native_steps.iter().zip(&steps).enumerate() {
            let d = max_diff(a, b);
            assert!(d < 1e-4, "{arch} step {i}: logits differ by {d}");
        }
        let d = max_diff(&native_batch, &batch);
        assert!(d < 1e-4, "{arch} prefill: logits differ by {d}");
        // And the comparison is not vacuous: later positions depend on RoPE.
        assert!(max_diff(&native_steps[0], &native_steps[7]) > 1e-3);
        drop(reordered);
        let _ = std::fs::remove_file(path);
    }
}
