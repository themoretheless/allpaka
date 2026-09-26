//! Does the SwiGLU-fusion setting change the answer on a real decode?
//!
//! `ALLPAKA_SWFUSE` picks how the decode FFN's down projection gets its
//! activation: folded in, every down row re-applies `silu(gate)*up` to the raw
//! gate/up it loads; unfolded, one standalone swiglu dispatch finishes the
//! activation and the down reads it. The fold is a scheduling choice, so it must
//! be neither numeric nor invisible - this test checks both halves: the settings
//! have to agree on the logits AND differ on the dispatch count, otherwise it is
//! comparing one pipeline with itself.
//!
//! The shipped default is the unfolded one: the fold costs the 235B's q3_k down
//! 7.7 ms/token (18% of its wall, 10/10 paired signs) and q8_0 nothing - the
//! decision is `docs/benchmarks/2026-09-24-235b-swfuse/results.txt`, landed
//! 2026-09-26.
//!
//! Its own file because the setting is read from the environment where the
//! decode FFN pipelines are built, so no other test may be running here.

#![cfg(target_os = "macos")]

use allpaka_backend::gpu;
use allpaka_gguf::GgufFile;
use allpaka_model::Model;

const MODEL: &str = "../../models/qwen3-0.6b-Q8_0.gguf";

/// Decode 8 tokens with `ALLPAKA_SWFUSE` pinned to `setting`, returning the
/// per-step argmax chain, the last logits, and the dispatches issued for them.
fn decode(setting: &str) -> (Vec<u32>, Vec<f32>, u64) {
    std::env::set_var("ALLPAKA_SWFUSE", setting);
    let f = GgufFile::open(std::path::Path::new(MODEL)).expect("0.6B model present");
    let model = Model::load(&f).unwrap();
    let mut session = model.new_session(64);
    let prompt = [785u32, 6722, 315, 9625, 374];
    let mut logits = model.forward_batch(&prompt, &mut session).unwrap();
    let (mut chain, mut last) = (Vec::new(), Vec::new());
    let before = gpu::stats().1;
    for _ in 0..8 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        chain.push(next);
        logits = model.forward(next, &mut session).unwrap();
        last = logits.clone();
    }
    let dispatches = gpu::stats().1 - before;
    std::env::remove_var("ALLPAKA_SWFUSE");
    (chain, last, dispatches)
}

#[test]
fn swiglu_fold_is_a_scheduling_choice_not_a_numeric_one() {
    if !std::path::Path::new(MODEL).is_file() {
        eprintln!("SKIP: {MODEL} not present (CI runners have no models)");
        return;
    }
    // qwen3-0.6b is dense with a Q8_0 down, and `matvec_q8_0` is in the decode
    // FFN builder's `sw_capable` list, so both settings are live on this model.
    let (folded_chain, folded_logits, d_folded) = decode("1");
    let (plain_chain, plain_logits, d_plain) = decode("0");
    println!("dispatches / 8 tok: fold={d_folded} standalone={d_plain}");
    assert!(
        d_plain > d_folded,
        "both settings issued {d_folded} dispatches, so ALLPAKA_SWFUSE never reached the \
         decode FFN builder and this test would be vacuous"
    );
    assert_eq!(folded_chain, plain_chain, "the argmax chain moved");
    let (mut max_abs, mut worst) = (0f32, 0usize);
    for (i, (f, p)) in folded_logits.iter().zip(&plain_logits).enumerate() {
        let a = (f - p).abs();
        if a > max_abs {
            max_abs = a;
            worst = i;
        }
        assert!(a <= 1e-5 * p.abs().max(1.0), "logit {worst}: fold {f} vs standalone {p}");
    }
    // Both arms are the same GPU path on the same weights, so agreement is
    // exact-or-nothing: a nonzero delta means the fold computes different
    // arithmetic, and a 1e-5-tolerant pass would hide a real drift.
    assert!(
        max_abs == 0.0,
        "the fold should re-associate identical arithmetic, but logit {worst} differs by {max_abs}"
    );
}
