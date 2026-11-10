//! Long-prompt prefill on CUDA must be deterministic and agree with the
//! token-by-token path.
//!
//! Regression for three bugs that together made 1200-token prompts decode to
//! garbage (and differently on every run) while ~30-token prompts were fine:
//! the ggml flash-attention bridge stored its prefill output transposed, the
//! dual-queue sync skipped the ggml-after-cudarc wait for producers that never
//! marked themselves dirty, and an odd KV length (a 3-token prefix then a
//! 512-token chunk) misaligned the FA mask rows.
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_long_prefill -- --ignored
#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// A 3-token prefix (what a prefix-cache hit on the chat header leaves),
/// then the rest in prefill-sized chunks: chunk KV lengths 515, 1027, ...
fn chunked_prefill(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let mut session = model.new_session(tokens.len() + 1);
    let mut logits = model.forward_batch(&tokens[..3], &mut session).unwrap();
    for chunk in tokens[3..].chunks(Model::prefill_chunk()) {
        logits = model.forward_batch(chunk, &mut session).unwrap();
    }
    assert_eq!(session.pos(), tokens.len());
    logits
}

#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn long_chunked_prefill_is_deterministic_and_matches_decode() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    assert!(
        allpaka_backend::gpu::is_attached(),
        "CUDA backend not attached: {:?}",
        allpaka_backend::gpu::init_error()
    );
    let n = 1100usize;
    let tokens: Vec<u32> = (0..n)
        .map(|i| (i as u32 * 733 + 17) % model.config.vocab.min(30000))
        .collect();

    let a = chunked_prefill(&model, &tokens);
    let b = chunked_prefill(&model, &tokens);
    assert!(a.iter().all(|v| v.is_finite()), "non-finite prefill logits");
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "two identical prefills produced different logits"
    );

    // Reference: one token at a time (decode attention, no ggml FA prefill).
    let mut session = model.new_session(n + 1);
    let mut r = Vec::new();
    for t in &tokens {
        r = model.forward_batch(std::slice::from_ref(t), &mut session).unwrap();
    }
    let cos = cosine(&a, &r);
    eprintln!(
        "prefill argmax {} decode argmax {} cosine {cos:.6}",
        argmax(&a),
        argmax(&r)
    );
    assert!(cos > 0.99, "chunked prefill diverges from decode: cosine {cos}");
    assert_eq!(argmax(&a), argmax(&r));
}
