//! Decode must survive a short prefill that follows an earlier decode.
//!
//! Regression for `allpaka serve` answering the second short request in a row
//! with garbage: the fused decode kept its per-layer norm weights (and q/k/v
//! biases) in a range of the shared `x_arena`, uploaded once per decode graph
//! key, while a prefill chunk of a few tokens wrote its RoPE table into the
//! same range. A chat turn that reuses the cached template prefix and prefills
//! only a handful of new tokens therefore decoded with corrupted norms; a long
//! prompt moved to another flash-attention bucket (a new key, a fresh upload)
//! and hid it.
//!
//! Also here: a replayed prefill graph on a model with q/k/v biases (qwen2)
//! must give the same logits on every replay.
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_decode_after_short_prefill --
//! --ignored
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

fn load() -> (GgufFile, u32) {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let vocab = allpaka_model::Config::from_gguf(&file).unwrap().vocab;
    (file, vocab.min(30000))
}

#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn decode_after_a_short_prefill_on_a_reused_session() {
    let (file, vocab) = load();
    let model = Model::load(&file).unwrap();
    assert!(
        allpaka_backend::gpu::is_attached(),
        "CUDA backend not attached: {:?}",
        allpaka_backend::gpu::init_error()
    );
    let toks = |seed: u32, n: usize| -> Vec<u32> {
        (0..n as u32)
            .map(|i| (i * 733 + seed * 101 + 17) % vocab)
            .collect()
    };
    const KEEP: usize = 5;
    const DECODE: usize = 6;
    let first = toks(1, 40);
    let follow: Vec<u32> = toks(3, DECODE);

    // Request 1: prefill, then decode a few tokens (uploads the decode norms).
    let mut s = model.new_session(256);
    model.forward_batch(&first, &mut s).unwrap();
    for &t in &follow {
        model.forward(t, &mut s).unwrap();
    }

    // Requests 2..: keep a shared prefix, prefill a few new tokens, decode.
    // Every remainder length puts the prefill scratch somewhere else.
    for (round, rest_len) in [12usize, 3, 20, 33, 7].into_iter().enumerate() {
        let rest = toks(10 + round as u32, rest_len);
        s.truncate(KEEP);
        model.forward_batch(&rest, &mut s).unwrap();
        let mut got = Vec::new();
        for &t in &follow {
            got.push(model.forward(t, &mut s).unwrap());
        }

        // Reference: the same context through the batched path alone, which
        // carries its own norms. One prefill per decoded position.
        let mut context: Vec<u32> = first[..KEEP].to_vec();
        context.extend_from_slice(&rest);
        for (i, d) in got.iter().enumerate() {
            context.push(follow[i]);
            let mut r = model.new_session(256);
            let want = model.forward_batch(&context, &mut r).unwrap();
            assert!(d.iter().all(|v| v.is_finite()), "non-finite decode logits");
            let cos = cosine(d, &want);
            eprintln!(
                "round {round} (+{rest_len} tok) decode step {i}: argmax {} vs {} cosine {cos:.6}",
                argmax(d),
                argmax(&want)
            );
            assert!(
                cos > 0.99,
                "decode after a {rest_len}-token prefill diverges at step {i}: cosine {cos}"
            );
        }
    }
}

/// Graph replay of a prefill chunk must not depend on host memory that was
/// only valid while the graph was captured (the qwen2 bias upload was).
#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn replayed_prefill_is_bit_stable() {
    let (file, vocab) = load();
    let model = Model::load(&file).unwrap();
    assert!(allpaka_backend::gpu::is_attached());
    let tokens: Vec<u32> = (0..512u32).map(|i| (i * 733 + 17) % vocab).collect();
    let mut s = model.new_session(1024);
    let mut runs: Vec<Vec<f32>> = Vec::new();
    // Run 0 is uncaptured, run 1 captures, the rest replay. Allocate and
    // free between runs so freed host memory gets reused.
    for i in 0..6 {
        s.truncate(0);
        runs.push(model.forward_batch(&tokens, &mut s).unwrap());
        let churn: Vec<Vec<f32>> = (0..64).map(|k| vec![(i * 64 + k) as f32; 3584]).collect();
        drop(churn);
    }
    for (i, r) in runs.iter().enumerate() {
        assert!(
            r.iter().all(|v| v.is_finite()),
            "non-finite logits, run {i}"
        );
        let cos = cosine(r, &runs[0]);
        eprintln!("run {i}: argmax {} cosine vs run 0 {cos:.6}", argmax(r));
        assert!(cos > 0.999, "prefill run {i} diverges from run 0: {cos}");
    }
    for i in 3..runs.len() {
        assert!(
            runs[i]
                .iter()
                .zip(&runs[2])
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "replay {i} differs from the first replay: cosine {:.6}",
            cosine(&runs[i], &runs[2])
        );
    }
}

/// A chunk of 32..63 tokens has no graph key, so the graph captured for the
/// larger chunk before it survives. It must not change what that graph
/// replays (it used to overwrite the staged RoPE table the graph re-reads).
#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn replay_survives_an_unkeyed_chunk_in_between() {
    let (file, vocab) = load();
    let model = Model::load(&file).unwrap();
    assert!(allpaka_backend::gpu::is_attached());
    let tokens: Vec<u32> = (0..113u32).map(|i| (i * 733 + 17) % vocab).collect();
    let mut s = model.new_session(256);
    let mut runs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    // Per run: 73 tokens at base 0 (keyed: uncaptured, captured, replayed...)
    // then 40 more (unkeyed, runs normally every time).
    for _ in 0..6 {
        s.truncate(0);
        let a = model.forward_batch_hidden(&tokens[..73], &mut s).unwrap();
        let b = model.forward_batch_hidden(&tokens[73..], &mut s).unwrap();
        runs.push((a, b));
    }
    for (i, (a, b)) in runs.iter().enumerate() {
        let (ca, cb) = (cosine(a, &runs[0].0), cosine(b, &runs[0].1));
        eprintln!("run {i}: keyed chunk cosine {ca:.6}, following chunk cosine {cb:.6}");
        assert!(
            ca > 0.9999 && cb > 0.9999,
            "run {i} diverges from the uncaptured run: {ca} / {cb}"
        );
    }
}
