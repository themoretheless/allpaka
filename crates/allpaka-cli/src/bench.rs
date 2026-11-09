//! Engine, memory, and network benchmarks (`allpaka bench`).

mod net;
mod report;

pub use net::{measure, measure_memory, serve};

use crate::benchmark_report::{FastPathStats, Measurement};
use anyhow::{Context, Result};
use report::{
    print_capability_report, report_gpu_clock, report_gpu_split, report_phases, write_engine_report,
};

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

const SPEC_PROMPT_TEXT: &str =
    "The history of computing began long before electronics. Mechanical \
     calculators appeared in the seventeenth century, and by the nineteenth \
     century Charles Babbage had designed a programmable machine. His \
     analytical engine was never completed, but the ideas behind it - a \
     store for numbers, a mill for arithmetic, and cards carrying the \
     program - anticipated the structure of the modern computer in";

/// Benchmark the engine itself on a real model: prefill and decode rates.
///
/// The numbers to watch when optimising: prefill tok/s (compute-bound, loves
/// batching) and decode tok/s (bandwidth-bound, loves fewer GPU waits). Run
/// before and after every change; a speedup that does not show here is
/// imaginary.
pub fn measure_engine(
    model_path: &std::path::Path,
    draft_path: Option<&std::path::Path>,
) -> Result<()> {
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    // Sequential prewarm, or the warmup forward faults tens of GiB in GPU
    // access order and the first numbers measure the SSD, not the engine.
    crate::serve::prewarm(file.mapping());
    for m in file.mappings().skip(1) {
        crate::serve::prewarm(m);
    }
    let model = allpaka_model::Model::load(&file)?;
    let c = &model.config;
    println!(
        "engine bench: {} ({} layers, {}{})",
        model_path
            .file_stem()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default(),
        c.n_layers,
        c.architecture,
        if c.moe.is_some() { ", MoE" } else { "" },
    );
    print_capability_report(&model, &file);

    // A synthetic prompt long enough to amortise chunking, plus warmup.
    // ALLPAKA_BENCH_PP overrides the prompt length (32 warmup + N measured);
    // PP=32 gives a decode measurement from ~zero context, matching how
    // llama-bench's tg numbers are taken.
    let measured_pp: u32 = std::env::var("ALLPAKA_BENCH_PP")
        .unwrap_or_else(|_| "480".into())
        .parse()
        .context("ALLPAKA_BENCH_PP must be an integer")?;
    let decode_tokens: usize = std::env::var("ALLPAKA_BENCH_TG")
        .unwrap_or_else(|_| "32".into())
        .parse()
        .context("ALLPAKA_BENCH_TG must be an integer")?;
    anyhow::ensure!(
        (1..=32768).contains(&measured_pp),
        "PP must be in 1..=32768"
    );
    anyhow::ensure!(
        (1..=32768).contains(&decode_tokens),
        "TG must be in 1..=32768"
    );
    let pp = measured_pp + 32;
    let prompt: Vec<u32> = (0..pp)
        .map(|i| (i * 733 + 17) % c.vocab.min(30000))
        .collect();
    let mut session = model.new_session(prompt.len() + decode_tokens + 1);

    let warm = model.forward_batch(&prompt[..32], &mut session)?;
    drop(warm);
    session = model.new_session(prompt.len() + decode_tokens + 1);
    let warm = model.forward_batch(&prompt[32..], &mut session)?;
    drop(warm);
    // Same KV allocation: second pass captures the prefill graph, timed pass replays it.
    session.truncate(0);
    let warm = model.forward_batch(&prompt[32..], &mut session)?;
    drop(warm);
    session.truncate(0);

    let gpu_pre = allpaka_backend::gpu::stats();
    let clock_pre = allpaka_backend::gpu::gpu_time_stats();
    allpaka_model::profile::reset();
    let t0 = std::time::Instant::now();
    let logits = crate::airbug::span("prefill", || -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for chunk in prompt[32..].chunks(allpaka_model::Model::prefill_chunk()) {
            logits = model.forward_batch(chunk, &mut session)?;
        }
        Ok(logits)
    })?;
    let prefill_secs = t0.elapsed().as_secs_f64();
    let gpu_prefill = allpaka_backend::gpu::stats();
    let prefill_rate = (prompt.len() - 32) as f64 / prefill_secs;
    // A kernel reading garbage (nil buffer, bad offset) shows up here first.
    let bad = logits.iter().filter(|v| !v.is_finite()).count();
    anyhow::ensure!(
        bad == 0 && !logits.is_empty(),
        "benchmark invalid: empty or non-finite logits"
    );
    let argmax = argmax_token(&logits);
    eprintln!("  prefill argmax={argmax}");
    println!(
        "  prefill  {:>4} tok in {prefill_secs:>6.2} s   {prefill_rate:>7.1} tok/s",
        prompt.len() - 32
    );
    crate::airbug::log_rate("prefill", prompt.len() - 32, prefill_secs, prefill_rate);
    report_gpu_split("prefill", gpu_pre, gpu_prefill, prefill_secs);
    report_gpu_clock(
        "prefill",
        clock_pre,
        allpaka_backend::gpu::gpu_time_stats(),
        gpu_pre.3,
        gpu_prefill.3,
    );
    let prefill_phases = report_phases("prefill", prefill_secs, prompt.len() - 32);

    // Decode from near-empty KV, matching llama-bench tg (n_prompt=0, n_depth=0).
    if let Ok(ms) = std::env::var("ALLPAKA_BENCH_BETWEEN_MS") {
        let ms = ms
            .parse::<u64>()
            .context("ALLPAKA_BENCH_BETWEEN_MS must be an integer")?;
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    session = model.new_session(decode_tokens + 4);
    let warm_seed_logits = model.forward_batch(&[1u32], &mut session)?;
    let warm_next = argmax_token(&warm_seed_logits);
    let warm = model.forward_greedy_n(warm_next, &mut session, decode_tokens)?;
    drop(warm);
    session.truncate(0);
    let seed = [1u32];
    let seed_logits = model.forward_batch(&seed, &mut session)?;
    let gpu_before = allpaka_backend::gpu::stats();
    let clock_before = allpaka_backend::gpu::gpu_time_stats();
    allpaka_model::profile::reset();
    allpaka_backend::telemetry::reset_global();
    let decode_stats_before = allpaka_backend::gpu::decode_path_stats();
    let mut decode_inputs = Vec::with_capacity(decode_tokens);
    let next = argmax_token(&seed_logits);
    let t1 = std::time::Instant::now();
    let outs = crate::airbug::span("decode", || {
        model.forward_greedy_n(next, &mut session, decode_tokens)
    })?;
    anyhow::ensure!(
        outs.len() == decode_tokens,
        "forward_greedy_n returned {} tokens, expected {decode_tokens}",
        outs.len()
    );
    decode_inputs.push(next);
    decode_inputs.extend_from_slice(&outs[..decode_tokens.saturating_sub(1)]);
    let decode_secs = t1.elapsed().as_secs_f64();
    let decode_rate = decode_tokens as f64 / decode_secs;
    println!(
        "  decode   {decode_tokens:>4} tok in {decode_secs:>6.2} s   {decode_rate:>7.1} tok/s"
    );
    crate::airbug::log_rate("decode", decode_tokens, decode_secs, decode_rate);
    let decode_stats_after = allpaka_backend::gpu::decode_path_stats();
    let gpu_attempts = decode_stats_after.attempts - decode_stats_before.attempts;
    let gpu_successes = decode_stats_after.successes - decode_stats_before.successes;
    let gpu_declines = decode_stats_after.declines - decode_stats_before.declines;
    anyhow::ensure!(
        allpaka_backend::gpu::is_attached(),
        "benchmark invalid: GPU is not attached; refusing to report CPU fallback as GPU throughput"
    );
    anyhow::ensure!(
        gpu_successes == decode_tokens as u64 && gpu_attempts == gpu_successes && gpu_declines == 0,
        "benchmark invalid: incomplete GPU coverage (attempts={gpu_attempts}, successes={gpu_successes}, declines={gpu_declines})"
    );
    println!(
        "  gpu path decode: attempts={gpu_attempts} successes={gpu_successes} declines={gpu_declines}"
    );
    let gpu_after = allpaka_backend::gpu::stats();
    report_gpu_split("decode", gpu_before, gpu_after, decode_secs);
    report_gpu_clock(
        "decode",
        clock_before,
        allpaka_backend::gpu::gpu_time_stats(),
        gpu_before.3,
        gpu_after.3,
    );

    let decode_phases = report_phases("decode", decode_secs, decode_tokens);
    println!(
        "  context at end: {} tokens (attention cost grows with this)",
        session.pos()
    );

    if let Some(dp) = draft_path {
        bench_draft_speculation(&model, &file, dp, decode_tokens)?;
    } else if model.config.nextn && !std::env::var("ALLPAKA_BENCH_SKIP_MTP").is_ok_and(|v| v == "1")
    {
        bench_mtp_speculation(&model, &file, decode_tokens)?;
    }

    let mut prefill_measurement =
        Measurement::new("prefill", prompt.len() - 32, vec![prefill_rate]);
    prefill_measurement.phases = prefill_phases;
    prefill_measurement.context_tokens = Some(0);
    prefill_measurement.input_tokens = prompt[32..].to_vec();
    let mut decode_measurement = Measurement::new("decode", decode_tokens, vec![decode_rate]);
    decode_measurement.phases = decode_phases;
    decode_measurement.context_tokens = Some(1);
    decode_measurement.input_tokens = decode_inputs;
    decode_measurement.fast_path = FastPathStats {
        attempts: gpu_attempts,
        successes: gpu_successes,
        declines: gpu_declines,
    };
    write_engine_report(model_path, &file, prefill_measurement, decode_measurement)?;
    Ok(())
}

fn bench_draft_speculation(
    model: &allpaka_model::Model<'_>,
    file: &allpaka_gguf::GgufFile,
    draft_path: &std::path::Path,
    decode_tokens: usize,
) -> Result<()> {
    let dfile = allpaka_gguf::GgufFile::open(draft_path)?;
    for m in dfile.mappings() {
        crate::serve::prewarm(m);
    }
    let dmodel = allpaka_model::Model::load(&dfile)?;
    if !allpaka_model::speculate::Speculator::compatible(model, &dmodel) {
        anyhow::bail!(
            "draft vocab {} != target vocab {}",
            dmodel.config.vocab,
            model.config.vocab
        );
    }
    let k: usize = std::env::var("ALLPAKA_DRAFT_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let tok = allpaka_model::Tokenizer::from_gguf(file)?;
    let spec_prompt = tok.encode(SPEC_PROMPT_TEXT)?;
    let mut ts = model.new_session(spec_prompt.len() + 64 + k + 1);
    let plain_logits = {
        let mut logits = Vec::new();
        for chunk in spec_prompt.chunks(allpaka_model::Model::prefill_chunk()) {
            logits = model.forward_batch(chunk, &mut ts)?;
        }
        logits
    };
    let prefill_end = ts.pos();
    let mut plain = Vec::new();
    let mut logits = plain_logits.clone();
    let t_plain = std::time::Instant::now();
    for _ in 0..decode_tokens {
        let next = argmax_token(&logits);
        plain.push(next);
        logits = model.forward(next, &mut ts)?;
    }
    let plain_secs = t_plain.elapsed().as_secs_f64();
    println!(
        "  plain on the same text: {:>4} tok in {plain_secs:>6.2} s   {:>7.1} tok/s",
        plain.len(),
        plain.len() as f64 / plain_secs,
    );

    ts.truncate(prefill_end);
    let mut ds = dmodel.new_session(spec_prompt.len() + 64 + k + 1);
    dmodel.forward_batch(&spec_prompt, &mut ds)?;
    let mut spec = allpaka_model::speculate::Speculator {
        target: model,
        target_session: &mut ts,
        draft: &dmodel,
        draft_session: &mut ds,
        k,
    };
    let mut next = argmax_token(&plain_logits);
    let mut emitted = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let t2 = std::time::Instant::now();
    while emitted.len() < decode_tokens {
        let round = spec.round(next)?;
        emitted.extend_from_slice(&round.emitted);
        drafted += round.drafted;
        accepted += round.accepted;
        next = round.next;
    }
    let spec_secs = t2.elapsed().as_secs_f64();
    emitted.truncate(decode_tokens);

    let rate = emitted.len() as f64 / spec_secs;
    println!(
        "  speculative (k={k}): {:>4} tok in {spec_secs:>6.2} s   {rate:>7.1} tok/s, \
         acceptance {accepted}/{drafted} ({:.0}%)",
        emitted.len(),
        accepted as f64 / drafted.max(1) as f64 * 100.0,
    );
    if emitted == plain {
        println!("  speculative stream matches plain greedy: PASS");
    } else {
        let at = emitted.iter().zip(&plain).position(|(a, b)| a != b);
        anyhow::bail!("speculative stream DIVERGED from plain greedy at {at:?}");
    }
    Ok(())
}

fn bench_mtp_speculation(
    model: &allpaka_model::Model<'_>,
    file: &allpaka_gguf::GgufFile,
    decode_tokens: usize,
) -> Result<()> {
    let k: usize = std::env::var("ALLPAKA_DRAFT_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let tok = allpaka_model::Tokenizer::from_gguf(file)?;
    let spec_prompt = tok.encode(SPEC_PROMPT_TEXT)?;
    let hidden = model.config.hidden as usize;
    let mut ts = model.new_session_mtp(spec_prompt.len() + 64 + k + 1);
    let xs = model.forward_batch_hidden(&spec_prompt, &mut ts)?;
    let plain_logits =
        model.lm_head(&model.output_normed(&xs[(spec_prompt.len() - 1) * hidden..]))?;
    let mut plain = Vec::new();
    let mut logits = plain_logits.clone();
    let t_plain = std::time::Instant::now();
    for _ in 0..decode_tokens {
        let next = argmax_token(&logits);
        if std::env::var_os("ALLPAKA_PLAIN_DEBUG").is_some() {
            let mut top: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!(
                "plain @{}: {} (gap {:.6}, runner {} = {:.6})",
                plain.len(),
                next,
                top[0].1 - top[1].1,
                top[1].0,
                top[1].1 - top[0].1,
            );
        }
        plain.push(next);
        logits = model.forward(next, &mut ts)?;
    }
    let plain_secs = t_plain.elapsed().as_secs_f64();
    println!(
        "  plain on the same text: {:>4} tok in {plain_secs:>6.2} s   {:>7.1} tok/s",
        plain.len(),
        plain.len() as f64 / plain_secs,
    );

    let mut ss = model.new_session_mtp(spec_prompt.len() + 64 + k + 1);
    let xs = model.forward_batch_hidden(&spec_prompt, &mut ss)?;
    let h0 = model.output_normed(&xs[(spec_prompt.len() - 1) * hidden..]);
    let mut spec = allpaka_model::speculate::MtpSpeculator {
        model,
        session: &mut ss,
        k,
        h: h0,
    };
    let mut next = argmax_token(&plain_logits);
    let mut emitted = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let t2 = std::time::Instant::now();
    while emitted.len() < decode_tokens {
        let round = spec.round(next)?;
        emitted.extend_from_slice(&round.emitted);
        drafted += round.drafted;
        accepted += round.accepted;
        next = round.next;
    }
    let spec_secs = t2.elapsed().as_secs_f64();
    emitted.truncate(decode_tokens);

    let rate = emitted.len() as f64 / spec_secs;
    println!(
        "  mtp speculative (k={k}): {:>4} tok in {spec_secs:>6.2} s   {rate:>7.1} tok/s, \
         acceptance {accepted}/{drafted} ({:.0}%)",
        emitted.len(),
        accepted as f64 / drafted.max(1) as f64 * 100.0,
    );
    if emitted == plain {
        println!("  mtp stream matches plain greedy: PASS");
    } else {
        let at = emitted.iter().zip(&plain).position(|(a, b)| a != b);
        if let Some(i) = at {
            eprintln!(
                "  divergence @{}: spec {} vs plain {} (ctx spec {:?} vs plain {:?})",
                i,
                emitted[i],
                plain[i],
                &emitted[i.saturating_sub(4)..i],
                &plain[i.saturating_sub(4)..i],
            );
        }
        anyhow::bail!("mtp stream DIVERGED from plain greedy at {at:?}");
    }
    Ok(())
}
