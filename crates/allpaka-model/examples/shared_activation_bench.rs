//! Paired-process benchmark: ALLPAKA_PREFILL_ZERO_COPY=0/1; fixed token streams.
//! Prints raw samples and path counters. Requires an actual GGUF model + Metal.
use allpaka_backend::gpu;
use allpaka_gguf::GgufFile;
use allpaka_model::Model;
use anyhow::{ensure, Result};
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let file = GgufFile::open(Path::new(args.get(1).expect("MODEL [PP=512] [ROUNDS=5]")))?;
    let model = Model::load(&file)?;
    ensure!(
        gpu::is_attached(),
        "Metal required: {:?}",
        gpu::init_error()
    );
    let pp: usize = args.get(2).map_or(Ok(512), |s| s.parse())?;
    let rounds: usize = args.get(3).map_or(Ok(5), |s| s.parse())?;
    ensure!(pp > 0 && rounds > 0, "nonzero sizes required");
    let tokens: Vec<u32> = (0..pp)
        .map(|i| (i as u32 * 733 + 17) % model.config.vocab.min(30000))
        .collect();
    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut final_logits = Vec::new();
    let mut generated = Vec::new();
    let stats_before = gpu::shared_activation_stats();
    let gpu_before = gpu::stats();
    for round in 0..=rounds {
        let mut session = model.new_session(pp + 33);
        let start = Instant::now();
        let mut logits = Vec::new();
        for chunk in tokens.chunks(Model::prefill_chunk()) {
            logits = model.forward_batch(chunk, &mut session)?;
        }
        let seconds = start.elapsed().as_secs_f64();
        ensure!(
            !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
            "invalid logits"
        );
        if round > 0 {
            prefill.push(pp as f64 / seconds);
        }
        final_logits = logits.clone();
        // Identical one-token initial context, independent of prefill timing.
        let mut session = model.new_session(33);
        logits = model.forward(17, &mut session)?;
        generated.clear();
        let start = Instant::now();
        for _ in 0..32 {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            generated.push(token);
            logits = model.forward(token, &mut session)?;
        }
        if round > 0 {
            decode.push(32.0 / start.elapsed().as_secs_f64());
        }
    }
    let stats = gpu::shared_activation_stats();
    let gpu_after = gpu::stats();
    println!(
        "{}",
        json!({
            "model": args[1], "pp": pp, "rounds": rounds, "warmup_rounds": 1,
            "zero_copy": std::env::var("ALLPAKA_PREFILL_ZERO_COPY").unwrap_or_else(|_| "1".into()),
            "prefill_tokens_per_second": prefill, "decode_tokens_per_second": decode,
            "generated": generated, "prefill_logits": final_logits,
            "shared_outputs": stats.outputs-stats_before.outputs,
            "handoff_bytes_avoided": stats.handoff_bytes-stats_before.handoff_bytes,
            "projection_bytes_avoided": stats.projection_bytes-stats_before.projection_bytes,
            "gpu_dispatches": gpu_after.1-gpu_before.1,
        })
    );
    Ok(())
}
