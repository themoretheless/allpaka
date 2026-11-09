//! Capability census, GPU/CPU phase breakdowns, and benchmark JSON report writing.

use crate::benchmark_report::{
    model_fingerprint, BenchmarkMetadata, BenchmarkReport, Measurement, PhaseMetric,
    RegressionPolicy, SCHEMA_VERSION,
};
use anyhow::{Context, Result};
use std::collections::BTreeMap;

pub(super) fn print_capability_report(
    model: &allpaka_model::Model<'_>,
    file: &allpaka_gguf::GgufFile,
) {
    const OVERRIDES: &[&str] = &[
        "ALLPAKA_NO_GPU",
        "ALLPAKA_CPU_ATTN",
        "ALLPAKA_NO_TOKENBUF",
        "ALLPAKA_PREFILL_CHUNK",
        "ALLPAKA_GPU_ROUTE",
        "ALLPAKA_DECODE_SERIAL",
        "ALLPAKA_PF_ONEBUF",
        "ALLPAKA_PF_DEFER",
        "ALLPAKA_MM_PIPE",
        "ALLPAKA_ATTN_SPLIT",
        "ALLPAKA_NORMFLAG",
    ];
    let c = &model.config;
    let model_capability = c.capability();
    println!("  capabilities:");
    println!(
        "    model: arch={} layers={} moe={}",
        c.architecture,
        c.n_layers,
        c.moe.is_some()
    );
    println!(
        "    model-support: {:?} ({})",
        model_capability.support, model_capability.reason
    );
    println!(
        "    profile: {}",
        std::env::var("ALLPAKA_PROFILE").unwrap_or_else(|_| "auto".into())
    );
    let accel = if allpaka_backend::gpu::is_attached() {
        #[cfg(target_os = "macos")]
        {
            "metal"
        }
        #[cfg(not(target_os = "macos"))]
        {
            "cuda"
        }
    } else {
        "none"
    };
    println!(
        "    gpu: backend={accel} attached={}",
        allpaka_backend::gpu::is_attached()
    );
    if let Some(e) = allpaka_backend::gpu::init_error() {
        println!("    gpu init error: {e}");
    }
    println!(
        "    weights: gpu-resident={}",
        allpaka_backend::gpu::is_attached()
    );
    let (weight_windows, residency_set) = allpaka_backend::gpu::residency_status();
    println!("    residency: windows={weight_windows} set={residency_set}");
    println!("    encoders: concurrent, resource-barriers=enabled");
    println!("    kv: page-aligned-f16, checked-at-session-runtime");
    println!("    prefill: gpu-counters-enabled");
    println!("    decode: checked-whole-token-fast-path");
    println!("    tensor-types:");
    let mut census = BTreeMap::<String, (usize, u64, bool)>::new();
    for tensor in file.tensors() {
        let name = match tensor.ggml_type {
            allpaka_gguf::GgmlType::Other(id) => format!("Other({id})"),
            ty => format!("{ty:?}"),
        };
        let metal_matmul = matches!(
            tensor.ggml_type,
            allpaka_gguf::GgmlType::F32
                | allpaka_gguf::GgmlType::Q5_0
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

pub(super) fn write_engine_report(
    model_path: &std::path::Path,
    file: &allpaka_gguf::GgufFile,
    prefill: Measurement,
    decode: Measurement,
) -> Result<()> {
    let mut overrides = BTreeMap::new();
    for (name, value) in std::env::vars().filter(|(name, _)| name.starts_with("ALLPAKA_")) {
        overrides.insert(name, value);
    }
    let generated_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let report = BenchmarkReport {
        schema_version: SCHEMA_VERSION,
        metadata: BenchmarkMetadata {
            generated_at_unix_ms,
            git_commit: option_env!("ALLPAKA_GIT_COMMIT")
                .unwrap_or("unknown")
                .into(),
            model: model_path.display().to_string(),
            model_fingerprint: model_fingerprint(model_path, file),
            device: if allpaka_backend::gpu::is_attached() {
                #[cfg(target_os = "macos")]
                {
                    "metal".into()
                }
                #[cfg(not(target_os = "macos"))]
                {
                    "cuda".into()
                }
            } else {
                "cpu".into()
            },
            runtime_profile: std::env::var("ALLPAKA_PROFILE").unwrap_or_else(|_| "auto".into()),
            resolved_overrides: overrides,
        },
        measurements: vec![prefill, decode],
    };

    if let Some(baseline_path) = std::env::var_os("ALLPAKA_BENCH_BASELINE") {
        let baseline_bytes = std::fs::read(&baseline_path).with_context(|| {
            format!(
                "reading benchmark baseline {}",
                std::path::Path::new(&baseline_path).display()
            )
        })?;
        let baseline: BenchmarkReport = serde_json::from_slice(&baseline_bytes)?;
        anyhow::ensure!(
            baseline.metadata.model_fingerprint == report.metadata.model_fingerprint,
            "benchmark baseline belongs to a different model fingerprint"
        );
        let tolerance = std::env::var("ALLPAKA_BENCH_TOLERANCE_PERCENT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3.0);
        let policy = RegressionPolicy {
            throughput_tolerance_percent: tolerance,
            fail_on_decline: true,
        };
        for candidate in &report.measurements {
            let baseline_measurement = baseline
                .measurements
                .iter()
                .find(|measurement| measurement.name == candidate.name)
                .with_context(|| format!("baseline has no {} measurement", candidate.name))?;
            let verdict = policy.evaluate(baseline_measurement, candidate);
            anyhow::ensure!(
                verdict.passed,
                "{} regression gate failed: {}",
                candidate.name,
                verdict.reasons.join("; ")
            );
            println!(
                "  {} regression gate: PASS ({:+.2}%)",
                candidate.name, verdict.throughput_change_percent
            );
        }
    }

    let default_name = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let report_path = std::env::var_os("ALLPAKA_BENCH_REPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from("target/allpaka-bench").join(format!("{default_name}.json"))
        });
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = report_path.with_extension("tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(&report)?)?;
    std::fs::rename(&temporary, &report_path)?;
    println!("  benchmark report: {}", report_path.display());
    if let (Some(prefill), Some(decode)) = (
        report.measurements.iter().find(|m| m.name == "prefill"),
        report.measurements.iter().find(|m| m.name == "decode"),
    ) {
        if let (Some(&prefill_rate), Some(&decode_rate)) =
            (prefill.samples_tok_s.first(), decode.samples_tok_s.first())
        {
            crate::airbug::save_run(prefill_rate, prefill.tokens, decode_rate, decode.tokens);
        }
    }
    Ok(())
}

/// Every phase of a forward pass, biggest first, in ms per token. Phases
/// that submit to the GPU include their wait, so they are read against the
/// GPU line above; the rest is CPU work with nowhere to hide. Resets the
/// counters for the next section.
pub(super) fn report_phases(what: &str, secs: f64, tokens: usize) -> Vec<PhaseMetric> {
    let (phases, calls) = allpaka_model::profile::take_with_counts();
    allpaka_model::profile::reset();
    let mut rows: Vec<(&str, u64, u64)> = allpaka_model::profile::NAMES
        .iter()
        .copied()
        .zip(phases)
        .zip(calls)
        .map(|((name, ns), calls)| (name, ns, calls))
        .collect();
    for sample in allpaka_backend::telemetry::take_global().samples() {
        let ns = sample.duration.as_nanos().min(u64::MAX as u128) as u64;
        if ns == 0 {
            continue;
        }
        let name = sample.phase.name();
        match rows.iter_mut().find(|(known, _, _)| *known == name) {
            Some((_, known_ns, known_calls)) => {
                *known_ns += ns;
                *known_calls += sample.calls;
            }
            None => rows.push((name, ns, sample.calls)),
        }
    }
    rows.sort_by_key(|&(_, ns, _)| std::cmp::Reverse(ns));
    let per_token = |ns: u64| ns as f64 / 1e6 / tokens as f64;
    let accounted: u64 = rows.iter().map(|(_, ns, _)| *ns).sum();
    println!("  {what} phases, ms/token:");
    for (name, ns, _) in rows.iter().filter(|&&(_, ns, _)| ns > 0) {
        println!(
            "    {name:<18} {:>7.1}  ({:>4.1}%)",
            per_token(*ns),
            *ns as f64 / (secs * 1e9) * 100.0
        );
        crate::airbug::log(&format!("{what} {name} {:.1} ms/tok", per_token(*ns)));
    }
    println!(
        "    {:<18} {:>7.1}  ({:>4.1}%)",
        "unaccounted",
        secs * 1e3 / tokens as f64 - per_token(accounted),
        (secs * 1e9 - accounted as f64) / (secs * 1e9) * 100.0,
    );
    rows.into_iter()
        .filter(|(_, ns, _)| *ns > 0)
        .map(|(name, ns, calls)| PhaseMetric {
            name: name.to_string(),
            milliseconds: ns as f64 / 1e6,
            calls,
        })
        .collect()
}

/// Where a phase's wall time went on the GPU path: round trips are the fixed
/// cost every command buffer pays, and wait time is the CPU blocked on the
/// GPU. Whatever wall time is left over is CPU work outside the kernels.
pub(super) fn report_gpu_split(
    phase: &str,
    before: (u64, u64, u64, u64),
    after: (u64, u64, u64, u64),
    secs: f64,
) {
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
pub(super) fn report_gpu_clock(
    phase: &str,
    before: (u64, u64),
    after: (u64, u64),
    wait_before: u64,
    wait_after: u64,
) {
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
