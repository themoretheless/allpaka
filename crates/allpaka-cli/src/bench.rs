//! Measuring the link between two machines.
//!
//! The planner is only as good as these numbers, and on Wi-Fi the advertised
//! link rate bears no useful relation to what you get. Two things are measured
//! separately because they cost differently:
//!
//! * **Round-trip latency**, paid once per cut per generated token. This is what
//!   makes Wi-Fi a poor medium for pipeline parallelism, and the tail matters
//!   more than the median because a stall is paid rather than averaged.
//! * **Throughput**, paid when shipping prompt activations, which are large but
//!   sent once.

use allpaka_core::Link;
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Instant;

const OP_PING: u8 = 0;
const OP_SINK: u8 = 1;

/// Payload for latency probes. Small enough that serialisation time is
/// negligible, so what is measured is genuinely latency.
const PING_BYTES: usize = 64;
const PING_ROUNDS: usize = 200;
/// Discard the first few round trips: TCP slow start and the Wi-Fi radio
/// waking from power save would otherwise be reported as steady-state latency.
const PING_WARMUP: usize = 20;

const CHUNK_BYTES: usize = 4 << 20;
const THROUGHPUT_BYTES: usize = 64 << 20;

/// Big enough to defeat every cache level; small enough to allocate anywhere.
const MEM_BUFFER_BYTES: usize = 1 << 30;
const MEM_PASSES: usize = 5;

/// Measure how fast this machine's CPU can stream from its own RAM.
///
/// This is the number the `pc-ram` pool runs at - decode on CPU is a straight
/// read of the weights - and the one preset most likely to be wrong: DDR4-2400
/// vs 3600 differ by 50%, and a single-channel machine halves it again. GPU
/// pools cannot be measured from here; use vendor numbers or llama-bench for
/// those.
///
/// All cores read disjoint slices at once, because one core cannot saturate a
/// memory controller; the sum approximates what a multi-threaded inference
/// runtime actually gets.
pub fn measure_memory() -> Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("streaming a {} MiB buffer with {threads} threads ...", MEM_BUFFER_BYTES >> 20);

    // u64 words, initialised so the pages are really committed - reading
    // untouched zero pages measures the kernel, not the RAM.
    let words = MEM_BUFFER_BYTES / 8;
    let buf: Vec<u64> = (0..words as u64).collect();

    let mut best_bytes_per_sec = 0.0f64;
    for _ in 0..MEM_PASSES {
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for chunk in buf.chunks(words.div_ceil(threads)) {
                scope.spawn(move || {
                    let mut acc = 0u64;
                    for w in chunk {
                        acc = acc.wrapping_add(*w);
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        let rate = MEM_BUFFER_BYTES as f64 / t0.elapsed().as_secs_f64();
        best_bytes_per_sec = best_bytes_per_sec.max(rate);
    }

    let gbps = best_bytes_per_sec / 1e9;
    println!("\nread bandwidth: {gbps:.1} GB/s (best of {MEM_PASSES} passes)");
    println!("\nfor the CPU-RAM pool of this machine, set in allpaka.toml:");
    println!("mem_bandwidth_gbps = {gbps:.1}");
    println!(
        "\nThis measures CPU-visible RAM only. For a GPU pool keep the preset's\n\
         vendor number, or back one out of measured llama-bench decode tok/s."
    );
    Ok(())
}

fn print_capability_report(model: &allpaka_model::Model<'_>, file: &allpaka_gguf::GgufFile) {
    const OVERRIDES: &[&str] = &[
        "ALLPAKA_NO_GPU",
        "ALLPAKA_CPU_ATTN",
        "ALLPAKA_NO_TOKENBUF",
        "ALLPAKA_PREFILL_CHUNK",
        "ALLPAKA_GPU_ROUTE",
        "ALLPAKA_DECODE_SERIAL",
        "ALLPAKA_PF_ONEBUF",
        "ALLPAKA_PF_DEFER",
    ];
    let c = &model.config;
    println!("  capabilities:");
    println!("    model: arch={} layers={} moe={}", c.architecture, c.n_layers, c.moe.is_some());
    println!("    metal: attached={}", allpaka_backend::gpu::is_attached());
    println!("    weights: metal-mapped={}", allpaka_backend::gpu::is_attached());
    println!("    kv: page-aligned-f16, checked-at-session-runtime");
    println!("    prefill: gpu-counters-enabled");
    println!("    decode: checked-whole-token-fast-path");
    println!("    tensor-types:");
    let mut census = std::collections::BTreeMap::<String, (usize, u64, bool)>::new();
    for tensor in file.tensors() {
        let name = match tensor.ggml_type {
            allpaka_gguf::GgmlType::Other(id) => format!("Other({id})"),
            ty => format!("{ty:?}"),
        };
        let metal_matmul = matches!(
            tensor.ggml_type,
            allpaka_gguf::GgmlType::F32
                | allpaka_gguf::GgmlType::Q8_0
                | allpaka_gguf::GgmlType::Q2K
                | allpaka_gguf::GgmlType::Q3K
                | allpaka_gguf::GgmlType::Q4K
                | allpaka_gguf::GgmlType::Q5K
                | allpaka_gguf::GgmlType::Q6K
        );
        let entry = census.entry(name).or_insert((0, 0, metal_matmul));
        entry.0 += 1;
        entry.1 += tensor.byte_size().unwrap_or(0);
    }
    for (ty, (count, bytes, metal_matmul)) in census {
        println!(
            "      {ty}: tensors={count} bytes={:.2} GiB matmul-kernel={}",
            bytes as f64 / (1u64 << 30) as f64,
            if metal_matmul { "yes" } else { "no" },
        );
    }
    println!("    overrides:");
    let mut any = false;
    for key in OVERRIDES {
        if let Ok(value) = std::env::var(key) {
            println!("      {key}={value}");
            any = true;
        }
    }
    if !any {
        println!("      none");
    }
}

/// Serve bench requests until the peer disconnects. Runs on the remote machine.
pub fn serve(bind: &str) -> Result<()> {
    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
    println!("allpaka bench server listening on {}", listener.local_addr()?);
    println!("run `allpaka bench --connect <this-host>:<port>` on the other machine");

    for stream in listener.incoming() {
        let stream = stream.context("accepting connection")?;
        let peer = stream.peer_addr()?;
        println!("client connected: {peer}");
        if let Err(e) = handle(stream) {
            // A client that finishes its run and closes the socket is normal,
            // not an error worth aborting the server over.
            println!("client {peer} finished: {e}");
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true)?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let mut header = [0u8; 9];
        if stream.read_exact(&mut header).is_err() {
            return Ok(()); // peer hung up
        }
        let op = header[0];
        let len = u64::from_le_bytes(header[1..9].try_into().unwrap()) as usize;
        if len > CHUNK_BYTES {
            bail!("client asked for a {len} byte frame, over the {CHUNK_BYTES} limit");
        }
        stream.read_exact(&mut buf[..len])?;
        match op {
            OP_PING => stream.write_all(&buf[..len])?,
            OP_SINK => stream.write_all(&[0u8; 8])?,
            other => bail!("unknown op {other}"),
        }
        stream.flush()?;
    }
}

/// Connect to a bench server and measure the path to it.
pub fn measure(addr: &str) -> Result<Link> {
    let target = addr
        .to_socket_addrs()
        .with_context(|| format!("resolving {addr}"))?
        .next()
        .with_context(|| format!("no address found for {addr}"))?;
    let mut stream = TcpStream::connect(target).with_context(|| format!("connecting to {addr}"))?;
    // Without this, Nagle's algorithm batches the small ping frames and the
    // latency measurement reports the delayed-ack timer instead of the network.
    stream.set_nodelay(true)?;

    let mut rtts = Vec::with_capacity(PING_ROUNDS);
    let payload = vec![0u8; PING_BYTES];
    let mut echo = vec![0u8; PING_BYTES];
    for i in 0..PING_ROUNDS {
        let t0 = Instant::now();
        send_frame(&mut stream, OP_PING, &payload)?;
        stream.read_exact(&mut echo)?;
        if i >= PING_WARMUP {
            rtts.push(t0.elapsed().as_secs_f64());
        }
    }
    rtts.sort_by(f64::total_cmp);

    let chunk = vec![0u8; CHUNK_BYTES];
    let chunks = THROUGHPUT_BYTES / CHUNK_BYTES;
    let mut ack = [0u8; 8];
    // One untimed chunk so the congestion window is open before we start.
    send_frame(&mut stream, OP_SINK, &chunk)?;
    stream.read_exact(&mut ack)?;

    let t0 = Instant::now();
    for _ in 0..chunks {
        send_frame(&mut stream, OP_SINK, &chunk)?;
        stream.read_exact(&mut ack)?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let throughput = (chunks * CHUNK_BYTES) as f64 / elapsed;

    Ok(Link {
        throughput_bytes_per_sec: throughput,
        rtt_p50_secs: percentile(&rtts, 0.50),
        rtt_p99_secs: percentile(&rtts, 0.99),
    })
}

fn send_frame(stream: &mut TcpStream, op: u8, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; 9];
    header[0] = op;
    header[1..9].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Benchmark the engine itself on a real model: prefill and decode rates.
///
/// The numbers to watch when optimising: prefill tok/s (compute-bound, loves
/// batching) and decode tok/s (bandwidth-bound, loves fewer GPU waits). Run
/// before and after every change; a speedup that does not show here is
/// imaginary.
pub fn measure_engine(model_path: &std::path::Path, draft_path: Option<&std::path::Path>) -> Result<()> {
    let decode_stats_before = allpaka_backend::gpu::decode_path_stats();
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
        model_path.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(),
        c.n_layers,
        c.architecture,
        if c.moe.is_some() { ", MoE" } else { "" },
    );
    print_capability_report(&model, &file);

    // A synthetic prompt long enough to amortise chunking, plus warmup.
    // ALLPAKA_BENCH_PP overrides the prompt length (32 warmup + N measured);
    // PP=32 gives a decode measurement from ~zero context, matching how
    // llama-bench's tg numbers are taken.
    let pp: u32 = std::env::var("ALLPAKA_BENCH_PP")
        .ok()
        .and_then(|v| v.parse().ok())
        .map_or(512, |n: u32| n.max(32) + 32)
        .min(1 << 15);
    let prompt: Vec<u32> = (0..pp).map(|i| (i * 733 + 17) % c.vocab.min(30000)).collect();
    let mut session = model.new_session(prompt.len() + 64);

    let warm = model.forward_batch(&prompt[..32], &mut session)?;
    drop(warm);

    let gpu_pre = allpaka_backend::gpu::stats();
    let clock_pre = allpaka_backend::gpu::gpu_time_stats();
    allpaka_model::profile::reset();
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    for chunk in prompt[32..].chunks(allpaka_model::Model::prefill_chunk()) {
        logits = model.forward_batch(chunk, &mut session)?;
    }
    let prefill_secs = t0.elapsed().as_secs_f64();
    let gpu_prefill = allpaka_backend::gpu::stats();
    let prefill_rate = (prompt.len() - 32) as f64 / prefill_secs;
    // A kernel reading garbage (nil buffer, bad offset) shows up here first.
    let bad = logits.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        println!("  WARNING: {bad} of {} logits are not finite", logits.len());
    }
    println!(
        "  prefill  {:>4} tok in {prefill_secs:>6.2} s   {prefill_rate:>7.1} tok/s",
        prompt.len() - 32
    );
    // The same split for prefill. This is the number that says whether a slow
    // prefill is the GPU or everything around it: wall time far above GPU wait
    // means the time is going to CPU-side work (attention, norms, routing, the
    // gather and scatter of expert rows), not to the kernels.
    report_gpu_split("prefill", gpu_pre, gpu_prefill, prefill_secs);
    report_gpu_clock(
        "prefill",
        clock_pre,
        allpaka_backend::gpu::gpu_time_stats(),
        gpu_pre.3,
        gpu_prefill.3,
    );
    report_phases("prefill", prefill_secs, prompt.len() - 32);

    let gpu_before = allpaka_backend::gpu::stats();
    let clock_before = allpaka_backend::gpu::gpu_time_stats();
    allpaka_model::profile::reset();
    let t1 = std::time::Instant::now();
    let decode_tokens = 32usize;
    let mut next = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    for _ in 0..decode_tokens {
        next = model.forward_greedy(next, &mut session)?;
    }
    let decode_secs = t1.elapsed().as_secs_f64();
    let decode_rate = decode_tokens as f64 / decode_secs;
    println!(
        "  decode   {decode_tokens:>4} tok in {decode_secs:>6.2} s   {decode_rate:>7.1} tok/s"
    );
    let decode_stats_after = allpaka_backend::gpu::decode_path_stats();
    let gpu_attempts = decode_stats_after.attempts - decode_stats_before.attempts;
    let gpu_successes = decode_stats_after.successes - decode_stats_before.successes;
    let gpu_declines = decode_stats_after.declines - decode_stats_before.declines;
    anyhow::ensure!(
        allpaka_backend::gpu::is_attached(),
        "benchmark invalid: Metal GPU is not attached; refusing to report CPU fallback as GPU throughput"
    );
    anyhow::ensure!(
        gpu_successes > 0,
        "benchmark invalid: whole-token GPU decode never succeeded (attempts={gpu_attempts}, declines={gpu_declines})"
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

    report_phases("decode", decode_secs, decode_tokens);
    println!(
        "  context at end: {} tokens (attention cost grows with this)",
        session.pos()
    );

    // Speculative decode: the same 32 tokens again, drafted by the small
    // model and verified by the target. Greedy verification makes the stream
    // a bit-exact copy of the plain one, so equality IS the correctness
    // check, and the speedup is honest or the run fails loudly.
    if let Some(dp) = draft_path {
        let dfile = allpaka_gguf::GgufFile::open(dp)?;
        for m in dfile.mappings() {
            crate::serve::prewarm(m);
        }
        let dmodel = allpaka_model::Model::load(&dfile)?;
        if !allpaka_model::speculate::Speculator::compatible(&model, &dmodel) {
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

        // Acceptance is a property of TEXT: on the synthetic random prompt
        // above, the two models never agree about how gibberish continues
        // and speculation only measures its own overhead. The speculative
        // section therefore decodes a natural prompt instead.
        let tok = allpaka_model::Tokenizer::from_gguf(&file)?;
        let spec_prompt = tok.encode(
            "The history of computing began long before electronics. Mechanical \
             calculators appeared in the seventeenth century, and by the nineteenth \
             century Charles Babbage had designed a programmable machine. His \
             analytical engine was never completed, but the ideas behind it - a \
             store for numbers, a mill for arithmetic, and cards carrying the \
             program - anticipated the structure of the modern computer in",
        )?;
        let mut ts = model.new_session(spec_prompt.len() + 64 + k + 1);
        let plain_logits = {
            let mut logits = Vec::new();
            for chunk in spec_prompt.chunks(allpaka_model::Model::prefill_chunk()) {
                logits = model.forward_batch(chunk, &mut ts)?;
            }
            logits
        };
        let prefill_end = ts.pos();
        // Plain greedy stream for the equality check, timed for the honest
        // comparison on the SAME text.
        let mut plain = Vec::new();
        let mut logits = plain_logits.clone();
        let t_plain = std::time::Instant::now();
        for _ in 0..decode_tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            plain.push(next);
            logits = model.forward(next, &mut ts)?;
        }
        let plain_secs = t_plain.elapsed().as_secs_f64();
        println!(
            "  plain on the same text: {:>4} tok in {plain_secs:>6.2} s   {:>7.1} tok/s",
            plain.len(),
            plain.len() as f64 / plain_secs,
        );

        // Speculative run from the same state.
        ts.truncate(prefill_end);
        let mut ds = dmodel.new_session(spec_prompt.len() + 64 + k + 1);
        dmodel.forward_batch(&spec_prompt, &mut ds)?;
        let mut spec = allpaka_model::speculate::Speculator {
            target: &model,
            target_session: &mut ts,
            draft: &dmodel,
            draft_session: &mut ds,
            k,
        };
        let mut next = plain_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
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
    } else if model.config.nextn {
        // Native MTP speculation: the draft is the model's own nextn block,
        // no second model. Same harness as the draft path - the emitted
        // stream must be a bit-exact copy of plain greedy.
        let k: usize = std::env::var("ALLPAKA_DRAFT_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let tok = allpaka_model::Tokenizer::from_gguf(&file)?;
        let spec_prompt = tok.encode(
            "The history of computing began long before electronics. Mechanical \
             calculators appeared in the seventeenth century, and by the nineteenth \
             century Charles Babbage had designed a programmable machine. His \
             analytical engine was never completed, but the ideas behind it - a \
             store for numbers, a mill for arithmetic, and cards carrying the \
             program - anticipated the structure of the modern computer in",
        )?;
        let hidden = model.config.hidden as usize;
        let mut ts = model.new_session_mtp(spec_prompt.len() + 64 + k + 1);
        let xs = model.forward_batch_hidden(&spec_prompt, &mut ts)?;
        let plain_logits =
            model.lm_head(&model.output_normed(&xs[(spec_prompt.len() - 1) * hidden..]))?;
        let prefill_end = ts.pos();
        let mut plain = Vec::new();
        let mut logits = plain_logits.clone();
        let t_plain = std::time::Instant::now();
        for _ in 0..decode_tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
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

        // The speculative run from a fresh session (the plain pass advanced
        // the irreversible GDN state; re-prefill instead of rolling back).
        let mut ss = model.new_session_mtp(spec_prompt.len() + 64 + k + 1);
        let xs = model.forward_batch_hidden(&spec_prompt, &mut ss)?;
        let h0 = model.output_normed(&xs[(spec_prompt.len() - 1) * hidden..]);
        let mut spec = allpaka_model::speculate::MtpSpeculator {
            model: &model,
            session: &mut ss,
            k,
            h: h0,
        };
        let mut next = plain_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
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
        let _ = prefill_end;
    }
    Ok(())
}

/// Every phase of a forward pass, biggest first, in ms per token. Phases
/// that submit to the GPU include their wait, so they are read against the
/// GPU line above; the rest is CPU work with nowhere to hide. Resets the
/// counters for the next section.
fn report_phases(what: &str, secs: f64, tokens: usize) {
    let phases = allpaka_model::profile::take();
    allpaka_model::profile::reset();
    let mut rows: Vec<(&str, u64)> =
        allpaka_model::profile::NAMES.iter().copied().zip(phases).collect();
    rows.sort_by_key(|&(_, ns)| std::cmp::Reverse(ns));
    let per_token = |ns: u64| ns as f64 / 1e6 / tokens as f64;
    let accounted: u64 = phases.iter().sum();
    println!("  {what} phases, ms/token:");
    for (name, ns) in rows.iter().filter(|&&(_, ns)| ns > 0) {
        println!(
            "    {name:<18} {:>7.1}  ({:>4.1}%)",
            per_token(*ns),
            *ns as f64 / (secs * 1e9) * 100.0
        );
    }
    println!(
        "    {:<18} {:>7.1}  ({:>4.1}%)",
        "unaccounted",
        secs * 1e3 / tokens as f64 - per_token(accounted),
        (secs * 1e9 - accounted as f64) / (secs * 1e9) * 100.0,
    );
}

/// Where a phase's wall time went on the GPU path: round trips are the fixed
/// cost every command buffer pays, and wait time is the CPU blocked on the
/// GPU. Whatever wall time is left over is CPU work outside the kernels.
fn report_gpu_split(phase: &str, before: (u64, u64, u64, u64), after: (u64, u64, u64, u64), secs: f64) {
    let calls = after.0 - before.0;
    if calls == 0 {
        return;
    }
    let dispatches = after.1 - before.1;
    let encode_ms = (after.2 - before.2) as f64 / 1e6;
    let wait_ms = (after.3 - before.3) as f64 / 1e6;
    let total_ms = secs * 1e3;
    println!(
        "  gpu during {phase}: {calls} waits, {dispatches} dispatches, \
         encode {encode_ms:.0} ms, wait {wait_ms:.0} ms of {total_ms:.0} ms total \
         ({:.0}% outside the GPU)",
        (total_ms - wait_ms - encode_ms).max(0.0) / total_ms * 100.0,
    );
}

/// The wait, split by the GPU's own clock: how much of the blocked time the
/// GPU actually executed, how much the driver spent scheduling, and how much
/// was pure round trip. This is the arbiter between "the kernels are slow"
/// and "the kernels are idle" - the two need opposite fixes.
fn report_gpu_clock(phase: &str, before: (u64, u64), after: (u64, u64), wait_before: u64, wait_after: u64) {
    let busy_ms = (after.0 - before.0) as f64 / 1e6;
    let sched_ms = (after.1 - before.1) as f64 / 1e6;
    let wait_ms = (wait_after - wait_before) as f64 / 1e6;
    if wait_ms <= 0.0 {
        return;
    }
    println!(
        "  gpu clock during {phase}: executing {busy_ms:.0} ms, scheduling {sched_ms:.0} ms, \
         round trips + idle {:.0} ms (of {wait_ms:.0} ms waited)",
        (wait_ms - busy_ms - sched_ms).max(0.0),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole client/server protocol over loopback: what the two-machine
    /// run does, minus the cable. If this passes, a failure between real
    /// machines is the network or a firewall, not the tool.
    #[test]
    fn the_bench_protocol_round_trips_over_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let _ = handle(stream);
        });

        let link = measure(&addr).unwrap();
        server.join().unwrap();

        assert!(link.rtt_p50_secs > 0.0);
        assert!(link.rtt_p99_secs >= link.rtt_p50_secs);
        // Loopback moves gigabytes per second; anything below 100 MB/s means
        // the throughput phase measured something other than the transport.
        assert!(link.throughput_bytes_per_sec > 100e6, "{}", link.throughput_bytes_per_sec);
        // And it is fast: p50 over a millisecond on loopback would mean the
        // Nagle/delayed-ack trap is back.
        assert!(link.rtt_p50_secs < 1e-3, "{}", link.rtt_p50_secs);
    }

    #[test]
    fn percentiles_pick_the_expected_ranks() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 0.50), 50.0);
        assert_eq!(percentile(&v, 0.99), 99.0);
    }

    #[test]
    fn percentile_of_empty_is_infinite_rather_than_panicking() {
        assert!(percentile(&[], 0.5).is_infinite());
    }

    #[test]
    fn percentile_handles_a_single_sample() {
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }
}
