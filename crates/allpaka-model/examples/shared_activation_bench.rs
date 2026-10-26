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
    let file = GgufFile::open(Path::new(
        args.get(1)
            .expect("MODEL [PP=512] [ROUNDS=5] [greedy|forced]"),
    ))?;
    let model = Model::load(&file)?;
    ensure!(
        gpu::is_attached(),
        "Metal required: {:?}",
        gpu::init_error()
    );
    let pp: usize = args.get(2).map_or(Ok(512), |s| s.parse())?;
    let rounds: usize = args.get(3).map_or(Ok(5), |s| s.parse())?;
    ensure!(pp > 0 && rounds > 0, "nonzero sizes required");
    let greedy = args.get(4).is_some_and(|s| s == "greedy");
    // Fixed inputs and full logits match scripts/llama-matched-bench.cpp,
    // avoiding a comparison of greedy vs random MoE expert selections.
    let forced = args.get(4).is_some_and(|s| s == "forced");
    let token_at = |i: usize| ((i as u32 + 32) * 733 + 17) % model.config.vocab.min(30000);
    let tokens: Vec<u32> = (0..pp)
        .map(|i| {
            ((i as u32 + if greedy || forced { 32 } else { 0 }) * 733 + 17)
                % model.config.vocab.min(30000)
        })
        .collect();
    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut final_logits = Vec::new();
    let mut decode_final_logits = Vec::new();
    let mut generated = Vec::new();
    let mut fast_path = Vec::new();
    let mut phase_costs = Vec::new();
    let stats_before = gpu::shared_activation_stats();
    let gpu_before = gpu::stats();
    for round in 0..=rounds {
        let mut session = model.new_session(pp + 33);
        let prefill_gpu = gpu::stats();
        let prefill_clock = gpu::gpu_time_stats();
        let start = Instant::now();
        let mut logits = Vec::new();
        for chunk in tokens.chunks(Model::prefill_chunk()) {
            logits = model.forward_batch(chunk, &mut session)?;
        }
        let seconds = start.elapsed().as_secs_f64();
        if round > 0 {
            phase_costs.push(costs("prefill", round, seconds, prefill_gpu, prefill_clock));
        }
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
        logits = if greedy || forced {
            model.forward_batch(&[1], &mut session)?
        } else {
            model.forward(17, &mut session)?
        };
        generated.clear();
        let greedy_seed = greedy.then(|| {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32
        });
        let path_before = gpu::decode_path_stats();
        let decode_gpu = gpu::stats();
        let decode_clock = gpu::gpu_time_stats();
        let start = Instant::now();
        if forced {
            for i in 0..32 {
                logits = model.forward(token_at(i), &mut session)?;
            }
        } else if let Some(seed) = greedy_seed {
            generated = model.forward_greedy_n(seed, &mut session, 32)?;
            ensure!(
                generated.len() == 32 && session.pos() == 33,
                "decode workload mismatch"
            );
        } else {
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
        }
        let seconds = start.elapsed().as_secs_f64();
        if round > 0 {
            phase_costs.push(costs("decode", round, seconds, decode_gpu, decode_clock));
        }
        let path_after = gpu::decode_path_stats();
        let attempts = path_after.attempts - path_before.attempts;
        let successes = path_after.successes - path_before.successes;
        let declines = path_after.declines - path_before.declines;
        if greedy || forced {
            ensure!(
                attempts == 32 && successes == 32 && declines == 0,
                "incomplete GPU decode coverage: {attempts}/{successes}/{declines}"
            );
        }
        fast_path.push(json!({"round": round, "attempts": attempts, "successes": successes, "declines": declines}));
        if round > 0 {
            decode.push(32.0 / seconds);
        }
        if forced {
            ensure!(
                session.pos() == 33 && logits.iter().all(|v| v.is_finite()),
                "invalid forced decode"
            );
            decode_final_logits = logits;
        }
    }
    let stats = gpu::shared_activation_stats();
    let gpu_after = gpu::stats();
    println!(
        "{}",
        json!({
            "model": args[1], "pp": pp, "rounds": rounds, "warmup_rounds": 1,
            "decode_kind": if forced { "forced_full_logits" } else if greedy { "forward_greedy_n" } else { "forward" },
            "decode_context_tokens": 1, "decode_tokens": 32, "kv_format": "f16",
            "hidden": model.config.hidden, "architecture": model.config.architecture,
            "decode_fast_path": fast_path,
            "phase_costs": phase_costs,
            "zero_copy": std::env::var("ALLPAKA_PREFILL_ZERO_COPY").unwrap_or_else(|_| "1".into()),
            "prefill_tokens_per_second": prefill, "decode_tokens_per_second": decode,
            "generated": generated, "prefill_logits": final_logits,
            "decode_final_logits": decode_final_logits,
            "forced_decode_inputs": if forced { (0..32).map(token_at).collect::<Vec<_>>() } else { Vec::new() },
            "shared_outputs": stats.outputs-stats_before.outputs,
            "handoff_bytes_avoided": stats.handoff_bytes-stats_before.handoff_bytes,
            "projection_bytes_avoided": stats.projection_bytes-stats_before.projection_bytes,
            "gpu_dispatches": gpu_after.1-gpu_before.1,
        })
    );
    Ok(())
}

// Encode and wait are CPU wall spans; GPU execution and scheduling can
// overlap them. These columns must not be summed into an additive split.
fn costs(
    phase: &str,
    round: usize,
    seconds: f64,
    before: (u64, u64, u64, u64),
    clock_before: (u64, u64),
) -> serde_json::Value {
    let after = gpu::stats();
    let clock = gpu::gpu_time_stats();
    json!({
        "phase": phase, "round": round, "wall_ms": seconds * 1e3,
        "calls": after.0 - before.0, "dispatches": after.1 - before.1,
        "encode_ms": (after.2 - before.2) as f64 / 1e6,
        "wait_ms": (after.3 - before.3) as f64 / 1e6,
        "gpu_busy_ms": (clock.0 - clock_before.0) as f64 / 1e6,
        "gpu_schedule_ms": (clock.1 - clock_before.1) as f64 / 1e6,
    })
}
