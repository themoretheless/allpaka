//! What it costs to alternate decode between two sessions token by token,
//! which a step scheduler does when two requests are in flight.
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_session_switch_cost -- --ignored --nocapture
#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn alternating_two_sessions_costs_little_and_changes_nothing() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    let a_prompt: Vec<u32> = (0..300u32).map(|i| (i * 733 + 17) % 30000).collect();
    let b_len: u32 = std::env::var("SWITCH_B_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let b_prompt: Vec<u32> = (0..b_len).map(|i| (i * 211 + 5) % 30000).collect();
    let n = 96;

    let run_alone = |prompt: &[u32]| {
        let mut s = model.new_session(prompt.len() + n + 2);
        let mut t = argmax(&model.forward_batch(prompt, &mut s).unwrap());
        let started = Instant::now();
        let out: Vec<u32> = (0..n)
            .map(|_| {
                t = model.forward_greedy(t, &mut s).unwrap();
                t
            })
            .collect();
        (out, started.elapsed().as_secs_f64())
    };
    // Warm, then measure each alone.
    let _ = run_alone(&a_prompt);
    let (a_alone, a_secs) = run_alone(&a_prompt);
    let (b_alone, b_secs) = run_alone(&b_prompt);

    // The same two generations, one token each in turn.
    let mut sa = model.new_session(a_prompt.len() + n + 2);
    let mut sb = model.new_session(b_prompt.len() + n + 2);
    let mut ta = argmax(&model.forward_batch(&a_prompt, &mut sa).unwrap());
    let mut tb = argmax(&model.forward_batch(&b_prompt, &mut sb).unwrap());
    let (mut a_out, mut b_out) = (Vec::new(), Vec::new());
    let started = Instant::now();
    let mut per_call = Vec::new();
    for _ in 0..n {
        let t = Instant::now();
        ta = model.forward_greedy(ta, &mut sa).unwrap();
        per_call.push(t.elapsed().as_secs_f64() * 1e3);
        a_out.push(ta);
        let t = Instant::now();
        tb = model.forward_greedy(tb, &mut sb).unwrap();
        per_call.push(t.elapsed().as_secs_f64() * 1e3);
        b_out.push(tb);
    }
    let mut sorted = per_call.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "per call while alternating: min {:.2} median {:.2} p90 {:.2} max {:.2} ms; first ten {:?}",
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 9 / 10],
        sorted[sorted.len() - 1],
        per_call[..10]
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    let both_secs = started.elapsed().as_secs_f64();

    let alone = a_secs + b_secs;
    eprintln!(
        "alone: {:.1} + {:.1} ms = {:.1} ms; alternating: {:.1} ms ({:+.1}%)",
        a_secs * 1e3,
        b_secs * 1e3,
        alone * 1e3,
        both_secs * 1e3,
        (both_secs / alone - 1.0) * 100.0
    );
    assert_eq!(
        a_out, a_alone,
        "session A decoded differently when interleaved"
    );
    assert_eq!(
        b_out, b_alone,
        "session B decoded differently when interleaved"
    );
}
