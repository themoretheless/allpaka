//! A session rebuilt in place must not inherit the old one's decode state.
//!
//! `serve` keeps one session per chat and replaces it with a larger one when
//! a prompt outgrows it. The new session sits at the old address with a new
//! device buffer and new per-layer offsets. Decode state used to be keyed by
//! that address, so a request after a regrow replayed graphs and offsets of
//! the cache that no longer existed: garbage output at a fraction of a token
//! per second (a 16k-token request after a 9k-token one on Qwen2.5-7B).
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_session_replaced_in_place -- --ignored
#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_gguf::GgufFile;
use allpaka_model::{Model, Session};

fn prompt(model: &Model, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| (i as u32 * 733 + 17 + salt) % model.config.vocab.min(30000))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// Prefill `tokens`, then decode `n` greedy tokens.
fn generate(model: &Model, session: &mut Session, tokens: &[u32], n: usize) -> Vec<u32> {
    let logits = model.forward_batch(tokens, session).unwrap();
    let first = argmax(&logits);
    let mut out = vec![first];
    out.extend(model.forward_greedy_n(first, session, n).unwrap());
    out
}

#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn a_session_rebuilt_in_place_decodes_like_a_fresh_one() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    assert!(
        allpaka_backend::gpu::is_attached(),
        "CUDA backend not attached: {:?}",
        allpaka_backend::gpu::init_error()
    );
    let (short, long) = (prompt(&model, 260, 0), prompt(&model, 280, 5));
    let new_tokens = 12;

    // What the long prompt gives in a session of its own.
    let mut fresh = model.new_session(400);
    let want = generate(&model, &mut fresh, &long, new_tokens);
    drop(fresh);

    // The same prompt in a session that replaced a smaller one in place. Both
    // contexts fall in the same power-of-two attention slot (512), which is
    // the case the address-keyed state could not tell apart.
    let mut session = model.new_session(300);
    let _ = generate(&model, &mut session, &short, new_tokens);
    session = model.new_session(400);
    let got = generate(&model, &mut session, &long, new_tokens);
    assert_eq!(got, want, "the rebuilt session decoded a different stream");

    // And once more back down, then up: every replacement starts clean.
    session = model.new_session(300);
    let _ = generate(&model, &mut session, &short, new_tokens);
    session = model.new_session(400);
    let again = generate(&model, &mut session, &long, new_tokens);
    assert_eq!(again, want, "the second rebuild decoded a different stream");
}
