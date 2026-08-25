//! The fused GPU attention block against the step-by-step path, on a real
//! model, over several decoded tokens.
//!
//! `verify` arbitrates prefill against llama.cpp, but decode's fused path
//! (gpu::attn_block: qkv + norm + rope + cache store + attention + wo in one
//! command buffer) is not on the prefill path at all. This is its arbiter:
//! the same session decoded twice must produce the same logits within the
//! noise of rmsnorm's f32-vs-f64 accumulation.

#![cfg(target_os = "macos")]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;

const MODEL: &str = "../../models/qwen3-0.6b-Q8_0.gguf";

fn decode_logits(cpu_attn: bool) -> Vec<Vec<f32>> {
    if cpu_attn {
        std::env::set_var("ALLPAKA_CPU_ATTN", "1");
    } else {
        std::env::remove_var("ALLPAKA_CPU_ATTN");
    }
    let f = GgufFile::open(std::path::Path::new(MODEL)).expect("0.6B model present");
    let model = Model::load(&f).unwrap();
    let mut session = model.new_session(64);
    let prompt = [785u32, 6722, 315, 9625, 374];
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
fn fused_decode_matches_the_step_by_step_path() {
    let fused = decode_logits(false);
    let stepped = decode_logits(true);
    for (step, (f, s)) in fused.iter().zip(&stepped).enumerate() {
        let f_arg = f.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let s_arg = s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(f_arg, s_arg, "argmax diverged at step {step}");
        let worst = f
            .iter()
            .zip(s)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.05, "step {step}: max logit diff {worst}");
    }
}
