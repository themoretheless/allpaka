//! The qwen35moe GPU decode (tokenbuf: GDN kernels + head_dim-256 attention +
//! 256-expert router) against the CPU reference path, on the real model.
//!
//! The CPU path is the verified reference (llama.cpp parity, see
//! `crates/allpaka-cli/src/verify.rs`); this test is the GPU path's arbiter:
//! the same session decoded twice must produce the same logits, and greedy
//! argmax must agree on every step.

#![cfg(target_os = "macos")]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;

const MODEL: &str = "../../models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

fn decode_logits(gpu: bool) -> Vec<Vec<f32>> {
    if gpu {
        std::env::remove_var("ALLPAKA_NO_TOKENBUF");
    } else {
        std::env::set_var("ALLPAKA_NO_TOKENBUF", "1");
    }
    let f = GgufFile::open(std::path::Path::new(MODEL)).expect("Qwen3.6 model present");
    let model = Model::load(&f).unwrap();
    let mut session = model.new_session(64);
    let prompt = [33963u32, 728, 264, 2716, 1103, 314, 2250, 5839, 7736, 13];
    let mut logits = model.forward_batch(&prompt, &mut session).unwrap();
    let mut out = Vec::new();
    for _ in 0..4 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        logits = model.forward(next, &mut session).unwrap();
        out.push(logits.clone());
    }
    out
}

#[test]
fn gpu_decode_matches_the_cpu_reference() {
    if !std::path::Path::new(MODEL).is_file() {
        eprintln!("SKIP: {MODEL} not present (CI runners have no models)");
        return;
    }
    let gpu = decode_logits(true);
    let cpu = decode_logits(false);
    for (step, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let g_arg = g.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let c_arg = c.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(g_arg, c_arg, "argmax diverged at step {step}");
        let worst = g
            .iter()
            .zip(c)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.1, "step {step}: max logit diff {worst}");
    }
}
