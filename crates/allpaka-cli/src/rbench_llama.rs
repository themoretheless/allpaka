//! Paired allpaka vs llama.cpp throughput benches via the `rbench` observation
//! schema. Candidate = allpaka, baseline = llama-bench; AB/BA order is balanced
//! across independent process pairs.

use anyhow::{bail, Context, Result};
use rbench::{
    analysis::{self, Decision},
    Availability, Case, Direction, Metric, Observation, Run, Status,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Options {
    pub model: PathBuf,
    pub pp: u32,
    pub tg: u32,
    pub repeats: u32,
    pub threshold_percent: f64,
    pub check: bool,
    /// Discarded allpaka+llama process pairs before measured repeats (GPU clock).
    pub warmup: u32,
    /// Sleep between engines within a pair (ms); reduces thermal bleed.
    pub cooldown_ms: u64,
    pub allpaka_bin: PathBuf,
    pub llama_bench: PathBuf,
    pub out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct AllpakaReport {
    measurements: Vec<AllpakaMeasurement>,
}

#[derive(Debug, Deserialize)]
struct AllpakaMeasurement {
    name: String,
    summary: AllpakaSummary,
    #[serde(default)]
    fast_path: AllpakaFastPath,
}

#[derive(Debug, Deserialize)]
struct AllpakaSummary {
    median: f64,
}

#[derive(Debug, Default, Deserialize)]
struct AllpakaFastPath {
    #[serde(default)]
    attempts: u64,
    #[serde(default)]
    successes: u64,
    #[serde(default)]
    declines: u64,
}

#[derive(Debug, Deserialize)]
struct LlamaRow {
    n_prompt: u32,
    n_gen: u32,
    n_depth: u32,
    #[serde(default)]
    samples_ts: Vec<f64>,
    /// Some builds expose a single avg_ts instead of samples_ts.
    #[serde(default)]
    avg_ts: Option<f64>,
}

fn tok_s_metric() -> Metric {
    Metric {
        id: "tok_s".into(),
        unit: "tok/s".into(),
        scope: "engine throughput; independent process".into(),
        phase: "measurement".into(),
        statistic: "individual process".into(),
        direction: Direction::Higher,
    }
}

fn cases(pp: u32, tg: u32, model: &Path) -> Vec<Case> {
    let model_s = model.display().to_string();
    vec![
        Case {
            id: "prefill".into(),
            contract: BTreeMap::from([
                ("workload".into(), format!("pp{pp}")),
                ("model".into(), model_s.clone()),
                (
                    "limitations".into(),
                    "llama-bench generates its own token stream; MoE routing is not identical".into(),
                ),
            ]),
            metrics: vec![tok_s_metric()],
        },
        Case {
            id: "decode".into(),
            contract: BTreeMap::from([
                ("workload".into(), format!("tg{tg}@pp{pp}")),
                ("model".into(), model_s),
                (
                    "limitations".into(),
                    "KV cache type must match (f16); verify against allpaka capability report".into(),
                ),
            ]),
            metrics: vec![tok_s_metric()],
        },
    ]
}

fn push_obs(
    run: &mut Run,
    case: &str,
    variant: &str,
    process: u32,
    pair: u32,
    sequence: u64,
    tok_s: f64,
) -> Result<()> {
    anyhow::ensure!(tok_s.is_finite() && tok_s > 0.0, "{case}/{variant}: non-positive tok/s");
    run.observations.push(Observation {
        case: case.into(),
        metric: "tok_s".into(),
        variant: variant.into(),
        process,
        pair: Some(pair),
        sequence,
        value: Some(format!("{tok_s}")),
        operations: 1,
        availability: Availability::Available,
    });
    Ok(())
}

fn allpaka_bench_env(cmd: &mut Command, opts: &Options, report_path: &Path) {
    cmd.env("ALLPAKA_BENCH_PP", opts.pp.to_string())
        .env("ALLPAKA_BENCH_TG", opts.tg.to_string())
        .env("ALLPAKA_BENCH_REPORT", report_path)
        // Paired throughput: skip post-measure MTP (heats GPU, adds variance).
        .env("ALLPAKA_BENCH_SKIP_MTP", "1");
    // Prefer max-performance unless the caller already pinned a profile.
    if std::env::var_os("ALLPAKA_PROFILE").is_none() {
        cmd.env("ALLPAKA_PROFILE", "max-performance");
    }
}

fn run_allpaka(opts: &Options, pair: u32, work: &Path) -> Result<(f64, f64)> {
    let report_path = work.join(format!("allpaka-{pair}.json"));
    let log_path = work.join(format!("allpaka-{pair}.log"));
    let mut cmd = Command::new(&opts.allpaka_bin);
    cmd.arg("bench").arg("--engine").arg(&opts.model);
    allpaka_bench_env(&mut cmd, opts, &report_path);
    let status = cmd
        .stdout(std::fs::File::create(&log_path)?)
        .stderr(std::process::Stdio::from(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)?,
        ))
        .status()
        .with_context(|| format!("spawning {}", opts.allpaka_bin.display()))?;
    anyhow::ensure!(status.success(), "allpaka bench failed; see {}", log_path.display());
    let report: AllpakaReport = serde_json::from_slice(
        &std::fs::read(&report_path)
            .with_context(|| format!("reading {}", report_path.display()))?,
    )?;
    let prefill = report
        .measurements
        .iter()
        .find(|m| m.name == "prefill")
        .context("allpaka report missing prefill")?;
    let decode = report
        .measurements
        .iter()
        .find(|m| m.name == "decode")
        .context("allpaka report missing decode")?;
    anyhow::ensure!(
        decode.fast_path.attempts == opts.tg as u64
            && decode.fast_path.successes == opts.tg as u64
            && decode.fast_path.declines == 0,
        "allpaka decode fast-path incomplete: attempts={} successes={} declines={}",
        decode.fast_path.attempts,
        decode.fast_path.successes,
        decode.fast_path.declines
    );
    Ok((prefill.summary.median, decode.summary.median))
}

fn cooldown(opts: &Options) {
    if opts.cooldown_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(opts.cooldown_ms));
    }
}

fn llama_rate(rows: &[LlamaRow], prompt: u32, gen: u32, depth: u32) -> Result<f64> {
    let row = rows
        .iter()
        .find(|r| r.n_prompt == prompt && r.n_gen == gen && r.n_depth == depth)
        .with_context(|| format!("llama-bench row missing pp={prompt} tg={gen} depth={depth}"))?;
    if let Some(v) = row.samples_ts.iter().copied().find(|v| v.is_finite() && *v > 0.0) {
        return Ok(v);
    }
    row.avg_ts
        .filter(|v| v.is_finite() && *v > 0.0)
        .context("llama-bench row has no samples_ts/avg_ts")
}

fn run_llama(opts: &Options, pair: u32, work: &Path) -> Result<(f64, f64)> {
    let pp_json = work.join(format!("llama-pp-{pair}.json"));
    let pp_log = work.join(format!("llama-pp-{pair}.log"));
    let tg_json = work.join(format!("llama-tg-{pair}.json"));
    let tg_log = work.join(format!("llama-tg-{pair}.log"));

    let pp_status = Command::new(&opts.llama_bench)
        .args([
            "-m",
            opts.model.to_str().context("model path is not UTF-8")?,
            "-p",
            &opts.pp.to_string(),
            "-n",
            "0",
            "-d",
            "0",
            "-r",
            "1",
            "-ngl",
            "99",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "-o",
            "json",
        ])
        .stdout(std::fs::File::create(&pp_json)?)
        .stderr(std::fs::File::create(&pp_log)?)
        .status()
        .with_context(|| format!("spawning {}", opts.llama_bench.display()))?;
    anyhow::ensure!(pp_status.success(), "llama-bench prefill failed; see {}", pp_log.display());

    let tg_status = Command::new(&opts.llama_bench)
        .args([
            "-m",
            opts.model.to_str().context("model path is not UTF-8")?,
            "-p",
            "0",
            "-n",
            &opts.tg.to_string(),
            "-d",
            &opts.pp.to_string(),
            "-r",
            "1",
            "-ngl",
            "99",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "-o",
            "json",
        ])
        .stdout(std::fs::File::create(&tg_json)?)
        .stderr(std::fs::File::create(&tg_log)?)
        .status()?;
    anyhow::ensure!(tg_status.success(), "llama-bench decode failed; see {}", tg_log.display());

    let pp_rows: Vec<LlamaRow> = parse_llama_json(&std::fs::read(&pp_json)?)?;
    let tg_rows: Vec<LlamaRow> = parse_llama_json(&std::fs::read(&tg_json)?)?;
    Ok((
        llama_rate(&pp_rows, opts.pp, 0, 0)?,
        llama_rate(&tg_rows, 0, opts.tg, opts.pp)?,
    ))
}

fn parse_llama_json(bytes: &[u8]) -> Result<Vec<LlamaRow>> {
    let value: Value = serde_json::from_slice(bytes).context("llama-bench JSON")?;
    match value {
        Value::Array(items) => Ok(serde_json::from_value(Value::Array(items))?),
        Value::Object(_) => Ok(vec![serde_json::from_value(value)?]),
        _ => bail!("unexpected llama-bench JSON root"),
    }
}

/// Run paired allpaka/llama throughput benches and write an rbench `Run`.
pub fn run(opts: Options) -> Result<()> {
    anyhow::ensure!((1..=32768).contains(&opts.pp), "pp must be in 1..=32768");
    anyhow::ensure!((1..=32768).contains(&opts.tg), "tg must be in 1..=32768");
    anyhow::ensure!((1..=10_000).contains(&opts.repeats), "repeats must be in 1..=10000");
    anyhow::ensure!((0..=16).contains(&opts.warmup), "warmup must be in 0..=16");
    anyhow::ensure!(
        (0.0..100.0).contains(&opts.threshold_percent),
        "threshold must be in 0..100"
    );
    anyhow::ensure!(opts.model.is_file(), "model not found: {}", opts.model.display());
    anyhow::ensure!(
        opts.allpaka_bin.exists(),
        "allpaka binary not found: {}",
        opts.allpaka_bin.display()
    );
    anyhow::ensure!(
        which_exists(&opts.llama_bench),
        "llama-bench not found: {}",
        opts.llama_bench.display()
    );

    std::fs::create_dir_all(&opts.out)?;
    let work = opts.out.join("raw");
    std::fs::create_dir_all(&work)?;

    // Full-file SHA on 50–70 GiB GGUFs blocks rbench for minutes before any
    // pair runs. Prefer size+mtime+inode fingerprint for provenance; override
    // with ALLPAKA_RBENCH_HASH=1 for a real sha256 when needed.
    let model_sha = if std::env::var("ALLPAKA_RBENCH_HASH").is_ok_and(|v| v == "1") {
        rbench::hash_file(&opts.model).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        let meta = std::fs::metadata(&opts.model)
            .with_context(|| format!("stat {}", opts.model.display()))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        #[cfg(unix)]
        let ino = {
            use std::os::unix::fs::MetadataExt;
            meta.ino()
        };
        #[cfg(not(unix))]
        let ino = 0u64;
        format!("meta:{}:{}:{}", meta.len(), mtime, ino)
    };
    let mut run = Run::new();
    run.cases = cases(opts.pp, opts.tg, &opts.model);
    run.provenance.insert("model".into(), opts.model.display().to_string());
    run.provenance.insert("model_sha256".into(), model_sha);
    run.provenance.insert("pp".into(), opts.pp.to_string());
    run.provenance.insert("tg".into(), opts.tg.to_string());
    run.provenance.insert("repeats".into(), opts.repeats.to_string());
    run.provenance.insert("warmup".into(), opts.warmup.to_string());
    run.provenance
        .insert("cooldown_ms".into(), opts.cooldown_ms.to_string());
    run.provenance
        .insert("allpaka_bin".into(), opts.allpaka_bin.display().to_string());
    run.provenance
        .insert("llama_bench".into(), opts.llama_bench.display().to_string());
    run.notes.push(
        "Candidate=allpaka, baseline=llama-bench. Sequential AB/BA pairs; tok/s higher is better. llama-bench owns its token stream so MoE routing is not identical. Warmup pairs discarded; ALLPAKA_BENCH_SKIP_MTP=1."
            .into(),
    );

    for w in 1..=opts.warmup {
        eprintln!("rbench warmup {w}/{} (discarded)", opts.warmup);
        let tag = 10_000 + w;
        let _ = run_allpaka(&opts, tag, &work)?;
        cooldown(&opts);
        let _ = run_llama(&opts, tag, &work)?;
        cooldown(&opts);
    }

    let mut process = 0u32;
    let mut ap_pp = Vec::new();
    let mut ap_tg = Vec::new();
    let mut lp_pp = Vec::new();
    let mut lp_tg = Vec::new();
    for pair in 1..=opts.repeats {
        eprintln!("rbench pair {pair}/{}", opts.repeats);
        let allpaka_first = pair % 2 == 1;
        let (ap, lp) = if allpaka_first {
            let ap = run_allpaka(&opts, pair, &work)?;
            cooldown(&opts);
            let lp = run_llama(&opts, pair, &work)?;
            (ap, lp)
        } else {
            let lp = run_llama(&opts, pair, &work)?;
            cooldown(&opts);
            let ap = run_allpaka(&opts, pair, &work)?;
            (ap, lp)
        };
        ap_pp.push(ap.0);
        ap_tg.push(ap.1);
        lp_pp.push(lp.0);
        lp_tg.push(lp.1);
        // allpaka = candidate
        push_obs(&mut run, "prefill", "candidate", process, pair, 0, ap.0)?;
        push_obs(&mut run, "decode", "candidate", process, pair, 0, ap.1)?;
        process += 1;
        // llama = baseline
        push_obs(&mut run, "prefill", "baseline", process, pair, 0, lp.0)?;
        push_obs(&mut run, "decode", "baseline", process, pair, 0, lp.1)?;
        process += 1;
        eprintln!(
            "  allpaka  prefill {:>8.2}  decode {:>8.2} tok/s",
            ap.0, ap.1
        );
        eprintln!(
            "  llama    prefill {:>8.2}  decode {:>8.2} tok/s",
            lp.0, lp.1
        );
        cooldown(&opts);
    }

    run.status = Status::Complete;
    run.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    let run_path = opts.out.join("run.json");
    rbench::model::write_new(&run_path, &run).map_err(|e| anyhow::anyhow!("{e}"))?;

    let comparisons = analysis::compare(&run, None, opts.threshold_percent, 0.05)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let cmp_path = opts.out.join("comparison.json");
    rbench::model::write_new(&cmp_path, &comparisons).map_err(|e| anyhow::anyhow!("{e}"))?;

    fn max_of(xs: &[f64]) -> Option<f64> {
        xs.iter().copied().filter(|v| v.is_finite()).reduce(f64::max)
    }

    // Legacy-shaped summary for scripts/bench-matrix.sh consumers.
    let mut summary = serde_json::json!({
        "model": opts.model.display().to_string(),
        "model_sha256": run.provenance.get("model_sha256").cloned().unwrap_or_default(),
        "pp": opts.pp,
        "tg": opts.tg,
        "warmup": opts.warmup,
        "cooldown_ms": opts.cooldown_ms,
        "comparison_validated": true,
        "schema": "rbench-llama-v1",
        "limitations": [
            "llama-bench generates its own token stream; MoE routing is not identical",
            "KV precision must be verified against the allpaka capability report",
            "Machine thermal state dominates absolute tok/s; prefer Δ% / decision"
        ],
    });
    for c in &comparisons {
        let key_a = format!("allpaka_{}", c.case);
        let key_l = format!("llama_{}", c.case);
        let (ap_max, lp_max) = match c.case.as_str() {
            "prefill" => (max_of(&ap_pp), max_of(&lp_pp)),
            "decode" => (max_of(&ap_tg), max_of(&lp_tg)),
            _ => (None, None),
        };
        summary[&key_a] = serde_json::json!({
            "median": c.candidate,
            "max": ap_max,
            "unit": "tok/s"
        });
        summary[&key_l] = serde_json::json!({
            "median": c.baseline,
            "max": lp_max,
            "unit": "tok/s"
        });
    }
    let summary_path = opts.out.join("summary.json");
    rbench::model::write_new(&summary_path, &summary).map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("RBENCH_RESULT={}", serde_json::to_string(&run)?);
    println!();
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:<16}",
        "case", "llama", "allpaka", "Δ%", "decision"
    );
    for c in &comparisons {
        println!(
            "{:<10} {:>10} {:>10} {:>10} {:<16}",
            c.case,
            c.baseline
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into()),
            c.candidate
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into()),
            c.change_percent
                .map(|v| format!("{v:+.2}"))
                .unwrap_or_else(|| "-".into()),
            format!("{:?}", c.decision)
        );
    }
    println!();
    println!("artifacts: {}", opts.out.display());
    println!("  run:         {}", run_path.display());
    println!("  comparison:  {}", cmp_path.display());

    let failed = comparisons.iter().any(|c| c.decision == Decision::Regression);
    if failed && opts.check {
        bail!("rbench reported a regression vs llama at ±{}%", opts.threshold_percent);
    }
    if failed {
        eprintln!(
            "note: regression vs llama at ±{}% (pass --check to fail the process)",
            opts.threshold_percent
        );
    }
    Ok(())
}

fn which_exists(path: &Path) -> bool {
    if path.components().count() > 1 {
        return path.exists();
    }
    let Ok(path_env) = std::env::var("PATH") else {
        return path.exists();
    };
    for dir in std::env::split_paths(&path_env) {
        if dir.join(path).is_file() {
            return true;
        }
    }
    path.exists()
}

pub fn default_out_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(format!(".rbench/llama-compare-{stamp}"))
}

pub fn resolve_allpaka_bin(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    if let Ok(p) = std::env::var("ALLPAKA_BIN") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().context("current_exe")?;
    Ok(exe)
}

pub fn resolve_llama_bench(explicit: Option<PathBuf>) -> PathBuf {
    explicit
        .or_else(|| std::env::var_os("LLAMA_BENCH").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("llama-bench"))
}
