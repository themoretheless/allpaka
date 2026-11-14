//! Batched decode of several sessions against the same sessions decoded one
//! at a time: the greedy tokens must agree (allowing for the matmul and matvec
//! kernels rounding differently), and the time per step is reported.
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_decode_multi_parity -- --ignored --nocapture
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
fn batched_decode_matches_one_at_a_time() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    assert!(
        model.supports_multi(),
        "model has layers forward_multi does not batch"
    );
    let steps = 64;
    let prompts: Vec<Vec<u32>> = (0..4u32)
        .map(|j| {
            (0..200 + 150 * j)
                .map(|i| (i * (211 + 2 * j) + 5 + j) % 30000)
                .collect()
        })
        .collect();
    let prefilled = |p: &[u32]| {
        let mut s = model.new_session(p.len() + steps + 2);
        let t = argmax(&model.forward_batch(p, &mut s).unwrap());
        (s, t)
    };

    // One at a time, through the ordinary decode path.
    let mut alone = Vec::new();
    let mut alone_secs = 0.0;
    for p in &prompts {
        let (mut s, mut t) = prefilled(p);
        let started = Instant::now();
        let mut out = Vec::new();
        for _ in 0..steps {
            t = argmax(&model.forward(t, &mut s).unwrap());
            out.push(t);
        }
        alone_secs += started.elapsed().as_secs_f64();
        alone.push(out);
    }

    // All four together.
    let (mut sessions, mut tokens): (Vec<_>, Vec<_>) = prompts.iter().map(|p| prefilled(p)).unzip();
    let mut batched = vec![Vec::new(); prompts.len()];
    let started = Instant::now();
    for _ in 0..steps {
        let mut refs: Vec<_> = sessions.iter_mut().collect();
        let logits = model.forward_multi(&tokens, &mut refs).unwrap();
        for (i, l) in logits.iter().enumerate() {
            tokens[i] = argmax(l);
            batched[i].push(tokens[i]);
        }
    }
    let batched_secs = started.elapsed().as_secs_f64();

    let mut agree_prefix = Vec::new();
    for (a, b) in alone.iter().zip(&batched) {
        agree_prefix.push(a.iter().zip(b).take_while(|(x, y)| x == y).count());
    }
    eprintln!(
        "one at a time: {:.2} ms per token; batched x{}: {:.2} ms per step ({:.2} ms per token); \
         tokens agreeing before the first difference: {agree_prefix:?} of {steps}",
        alone_secs * 1e3 / (steps * prompts.len()) as f64,
        prompts.len(),
        batched_secs * 1e3 / steps as f64,
        batched_secs * 1e3 / (steps * prompts.len()) as f64,
    );
    // Greedy chains diverge for good after one near-tie flips; require the
    // opening stretch to agree in every session.
    assert!(agree_prefix.iter().all(|&n| n >= 16), "{agree_prefix:?}");
}
