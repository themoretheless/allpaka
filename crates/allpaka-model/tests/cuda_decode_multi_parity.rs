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

    // One at a time, through the ordinary decode path; keep every step's
    // logits and the token fed.
    let mut alone: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
    let mut alone_secs = 0.0;
    for p in &prompts {
        let (mut s, mut t) = prefilled(p);
        let (mut fed, mut logits) = (Vec::new(), Vec::new());
        let started = Instant::now();
        for _ in 0..steps {
            fed.push(t);
            let l = model.forward(t, &mut s).unwrap();
            t = argmax(&l);
            logits.push(l);
        }
        alone_secs += started.elapsed().as_secs_f64();
        alone.push((fed, logits));
    }

    // All four together, fed the same tokens (teacher forcing): a greedy
    // chain diverges for good after one near-tie flips, which says nothing
    // about the step itself.
    let mut sessions: Vec<_> = prompts.iter().map(|p| prefilled(p).0).collect();
    let mut agree = vec![0usize; prompts.len()];
    let mut max_rel = 0f32;
    let started = Instant::now();
    for step in 0..steps {
        let tokens: Vec<u32> = alone.iter().map(|(fed, _)| fed[step]).collect();
        let mut refs: Vec<_> = sessions.iter_mut().collect();
        let logits = model.forward_multi(&tokens, &mut refs).unwrap();
        for (i, l) in logits.iter().enumerate() {
            let want = &alone[i].1[step];
            agree[i] += (argmax(l) == argmax(want)) as usize;
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = l
                .iter()
                .zip(want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            max_rel = max_rel.max(diff / scale);
            if std::env::var_os("MULTI_TRACE").is_some() && diff / scale > 0.05 {
                eprintln!("step {step} session {i}: {:.3}", diff / scale);
            }
        }
    }
    let batched_secs = started.elapsed().as_secs_f64();
    eprintln!(
        "one at a time: {:.2} ms per token; batched x{}: {:.2} ms per step ({:.2} ms per token);          greedy agreement per session {agree:?} of {steps}; largest logit difference {:.4} of the logit range",
        alone_secs * 1e3 / (steps * prompts.len()) as f64,
        prompts.len(),
        batched_secs * 1e3 / steps as f64,
        batched_secs * 1e3 / (steps * prompts.len()) as f64,
        max_rel,
    );
    assert!(agree.iter().all(|&n| n * 100 >= steps * 95), "{agree:?}");
    // The per-layer reference batch (ALLPAKA_NO_MULTI_FUSED=1) differs from
    // single-session decode by up to 0.195 of the logit range on low-ranked
    // tokens (matmul vs matvec rounding); the fused step by 0.10.
    assert!(max_rel < 0.25, "logits differ by {max_rel}");
}
