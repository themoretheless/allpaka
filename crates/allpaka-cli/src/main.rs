//! allpaka - decide whether splitting a model across your machines is worth it,
//! and if so, where to cut it.

mod airbug;
mod autotune;
mod bench;
mod benchmark_report;
mod client;
mod config;
mod dispatch;
mod explain;
mod model_registry;
mod scheduler;
mod serving_runtime;
use allpaka_gguf as gguf;
mod airbug_llama;
mod pipeline;
mod rag_mcp;
mod report;
mod serve;
mod verify;
mod watch;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "allpaka", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Run a contiguous transformer stage on this machine.
    Stage {
        model: PathBuf,
        #[arg(long)]
        first: usize,
        #[arg(long)]
        end: usize,
        /// Same immutable model identity on every stage.
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "127.0.0.1:9798")]
        bind: std::net::SocketAddr,
        #[arg(long, default_value_t = 4096)]
        context: usize,
    },
    /// Produce an executable stage manifest from the cluster planner.
    PipelinePlan {
        model: PathBuf,
        #[arg(long, default_value = "allpaka.toml")]
        config: PathBuf,
        #[arg(long)]
        model_id: String,
        /// Repeat node-name=host:port for every available worker.
        #[arg(long = "endpoint", required = true)]
        endpoints: Vec<String>,
        #[arg(long)]
        context: Option<u32>,
    },
    /// Expose a distributed model to Studio through a local chat API.
    PipelineServe {
        #[arg(long, required = true, value_delimiter = ',')]
        stages: Vec<String>,
        /// GGUF supplies tokenizer metadata; no weights are loaded here.
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "127.0.0.1:8099")]
        bind: std::net::SocketAddr,
        #[arg(long, default_value_t = 4096)]
        context: usize,
    },
    /// Execute prompt tokens through ordered remote model stages.
    Pipeline {
        #[arg(long, required = true, value_delimiter = ',')]
        stages: Vec<String>,
        #[arg(long, required = true, value_delimiter = ',')]
        tokens: Vec<u32>,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long, default_value_t = 16)]
        generate: usize,
    },
    /// Open the multi-provider chat workspace without loading a local model.
    Studio {
        #[arg(long, default_value = "127.0.0.1:8100")]
        bind: std::net::SocketAddr,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Conversation storage directory (defaults to OS user data directory).
        #[arg(long)]
        data_dir: Option<PathBuf>,
    },
    /// Explain the resolved GPU execution path for one GGUF model.
    Explain {
        #[arg(value_name = "GGUF")]
        engine: PathBuf,
        #[arg(long, default_value = "auto")]
        profile: String,
        #[arg(long)]
        json: bool,
    },
    /// Benchmark runtime profiles and cache the fastest compatible plan.
    Autotune {
        #[arg(value_name = "GGUF")]
        engine: PathBuf,
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
    /// Print an example allpaka.toml to stdout.
    Init,

    /// List the built-in hardware presets.
    Presets,

    /// Read a GGUF file's architecture metadata.
    Inspect {
        /// Path to a .gguf model file.
        model: PathBuf,
        /// Census a vision projector (`mmproj`) instead of a text model: the
        /// `clip.*` metadata and the tensors grouped by top-level name prefix.
        #[arg(long)]
        mmproj: bool,
    },

    /// Measure the network path between two machines.
    ///
    /// Start `allpaka bench --serve` on one machine, then run
    /// `allpaka bench --connect <host>:9797` on the other.
    Bench {
        /// Listen for a benchmarking client.
        #[arg(long, conflicts_with = "connect")]
        serve: bool,
        /// Address to bind when serving.
        #[arg(long, default_value = "0.0.0.0:9797")]
        bind: String,
        /// Address of a bench server to measure against.
        #[arg(long)]
        connect: Option<String>,
        /// Measure this machine's own CPU-RAM read bandwidth instead of a
        /// network path. Run it on each machine; it prints the
        /// mem_bandwidth_gbps line to record.
        #[arg(long, conflicts_with_all = ["serve", "connect"])]
        mem: bool,
        /// Benchmark the inference engine on a model: prefill and decode
        /// rates. Run before and after every optimisation.
        #[arg(long, conflicts_with_all = ["serve", "connect", "mem"])]
        engine: Option<PathBuf>,
        /// Typed runtime profile and overrides from allpaka.toml.
        #[arg(long, requires = "engine")]
        config: Option<PathBuf>,
        /// Draft model for speculative decoding (with --engine): decodes
        /// twice, plain and speculative, and asserts the streams match.
        #[arg(long, requires = "engine")]
        draft: Option<PathBuf>,
    },

    /// Paired allpaka vs llama.cpp throughput via airbug (AB/BA process pairs).
    Airbug {
        /// GGUF model path.
        #[arg(value_name = "GGUF")]
        model: PathBuf,
        /// Prefill tokens (ALLPAKA_BENCH_PP / llama -p).
        #[arg(long, default_value_t = 480)]
        pp: u32,
        /// Decode tokens (ALLPAKA_BENCH_TG / llama -n).
        #[arg(long, default_value_t = 32)]
        tg: u32,
        /// Independent AB/BA process pairs.
        #[arg(long, default_value_t = 5)]
        repeats: u32,
        /// Discarded warmup pairs before measurement (GPU clock settle).
        #[arg(long, default_value_t = 1)]
        warmup: u32,
        /// Sleep between engines within a pair, milliseconds.
        #[arg(long, default_value_t = 1500)]
        cooldown_ms: u64,
        /// Relative regression threshold percent for airbug compare.
        #[arg(long, default_value_t = 5.0)]
        threshold: f64,
        /// allpaka binary (default: this executable / ALLPAKA_BIN).
        #[arg(long)]
        allpaka: Option<PathBuf>,
        /// llama-bench binary (default: llama-bench / LLAMA_BENCH).
        #[arg(long)]
        llama_bench: Option<PathBuf>,
        /// Fail with exit 1 if airbug reports a regression vs llama.
        #[arg(long)]
        check: bool,
        /// Output directory for run.json / comparison.json / raw logs.
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Place several models one per memory pool, one endpoint each.
    ///
    /// The other arrangement: instead of splitting one model across machines,
    /// give each pool its own model and each agent its own endpoint. The
    /// machines then never talk to each other.
    Fleet {
        /// Cluster and agent description. Agents normally live in the
        /// `[[agents]]` section here rather than on the command line.
        #[arg(long, default_value = "allpaka.toml")]
        config: PathBuf,
        /// Ad-hoc model, repeated per agent. Overrides `[[agents]]` entirely,
        /// for trying something without editing the config.
        #[arg(long = "model")]
        models: Vec<PathBuf>,
        /// Context for the ad-hoc models above. Ignored when agents come from
        /// the config, which carries a context per agent.
        #[arg(long)]
        ctx: Option<u32>,
    },

    /// Print the llama-server command for every placed agent.
    ///
    /// Takes the same `[[agents]]` the planner uses, so what launches is what
    /// was planned - the two cannot drift apart.
    Launch {
        /// Cluster and agent description.
        #[arg(long, default_value = "allpaka.toml")]
        config: PathBuf,
    },

    /// Serve a model over an OpenAI-compatible chat API, on our own engine.
    Serve {
        /// Path to a .gguf to serve; repeat --model for multi-model serving.
        #[arg(long, required = true)]
        model: Vec<PathBuf>,
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:8099")]
        bind: String,
        /// Typed runtime profile and overrides from allpaka.toml.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Maximum accepted connections waiting for inference.
        #[arg(long, default_value_t = 256)]
        max_queued: usize,
        /// Maximum requests selected in one model-aware scheduling batch.
        #[arg(long, default_value_t = 16)]
        max_batch: usize,
        /// Maximum aggregate prompt tokens in a scheduling batch.
        #[arg(long, default_value_t = 32768)]
        batch_context_tokens: usize,
        /// Total resident model budget in GiB; zero means unlimited.
        #[arg(long, default_value_t = 0)]
        model_budget_gib: u64,
        /// Shared weights, prefix capacity and KV/SSM admission budget; zero is unlimited.
        /// GPU scratch and process overhead are not included yet.
        #[arg(long, default_value_t = 0)]
        memory_budget_mib: u64,
        /// Prefix-cache budget in MiB.
        #[arg(long, default_value_t = 512)]
        prefix_cache_mib: usize,
    },

    /// Probe a running `allpaka serve`: /health plus /stats.
    Status {
        /// Address of the running server.
        #[arg(long, default_value = client::DEFAULT_ADDR)]
        addr: String,
    },

    /// Watch a running `allpaka serve` as a phase plus conditions.
    ///
    /// Polls /health, /v1/models, /stats and /resources and prints only the
    /// transitions between them, so the output is the lifecycle rather than a
    /// wall of JSON.
    Watch {
        /// Address of the running server.
        #[arg(long, default_value = client::DEFAULT_ADDR)]
        addr: String,
        /// Poll interval, milliseconds. Also caps how long a single probe waits.
        #[arg(long, default_value_t = 500)]
        interval_ms: u64,
        /// One JSON object per transition, for scripts.
        #[arg(long)]
        json: bool,
    },

    /// Send one chat request to a running `allpaka serve`.
    Chat {
        /// The user prompt.
        prompt: String,
        /// Optional system prompt.
        #[arg(long)]
        system: Option<String>,
        /// Hand the server the rag_search/rag_read schemas so its built-in
        /// RAG tool-loop can run.
        #[arg(long)]
        rag: bool,
        /// Generation cap.
        #[arg(long, default_value_t = 800)]
        max_tokens: u32,
        /// Model name sent in the request body.
        #[arg(long, default_value = "qwen3")]
        model_name: String,
        /// Address of the running server.
        #[arg(long, default_value = client::DEFAULT_ADDR)]
        addr: String,
    },

    /// RAG smoke test against a running `allpaka serve`: forces rag_search and
    /// fails unless the tool-loop actually ran.
    RagTest {
        /// Model name sent in the request body.
        #[arg(long, default_value = "qwen3")]
        model_name: String,
        /// Address of the running server.
        #[arg(long, default_value = client::DEFAULT_ADDR)]
        addr: String,
    },

    /// Compare our engine's logits against a running llama-server.
    ///
    /// Start the reference first with the SAME model:
    ///   llama-server -m <model.gguf> --port 8080
    Verify {
        /// Path to the .gguf both engines load.
        #[arg(long)]
        model: PathBuf,
        /// Address of the llama-server reference.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Prompt to compare on.
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        /// How many of the reference's top tokens to compare.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Greedy decode steps to verify after the prefill comparison. The
        /// decode path is a different code path from prefill; 0 skips it.
        #[arg(long, default_value_t = 6)]
        decode: usize,
        /// Maximum acceptable |log-prob| difference. Defaults to 0.15 for a
        /// dense model and 0.5 for a MoE: measured llama.cpp Metal-vs-CPU
        /// disagreement on Qwen3-30B-A3B reaches 0.45 by itself, so a MoE
        /// tolerance below that would fail a correct engine.
        #[arg(long)]
        tol: Option<f64>,
    },

    /// Split one model across the cluster by layers.
    Plan {
        /// Path to a .gguf model file.
        #[arg(long)]
        model: PathBuf,
        /// Cluster description.
        #[arg(long, default_value = "allpaka.toml")]
        config: PathBuf,
        /// Context length to budget KV cache for.
        #[arg(long)]
        ctx: Option<u32>,
        /// Prompt length used to estimate one-off activation transfer.
        #[arg(long)]
        prompt: Option<u32>,
        /// Draft model for speculative decoding. It runs on the head node and
        /// never crosses the network; the full model then verifies several
        /// drafted tokens in one pass, so one round trip buys several tokens.
        #[arg(long)]
        draft: Option<PathBuf>,
        /// How many tokens the draft proposes per cycle.
        #[arg(long, default_value_t = 4)]
        draft_len: u32,
        /// Fraction of drafted tokens accepted. Measure it; do not guess.
        #[arg(long, default_value_t = 0.7)]
        accept: f64,
    },
}

fn main() -> Result<()> {
    if let Ok(profile) = std::env::var("ALLPAKA_PROFILE") {
        let profile: allpaka_backend::profile::RuntimeProfile = profile.parse()?;
        let resolved = profile.resolve_with_env();
        let _ = allpaka_backend::runtime::install(resolved.policy);
    }
    let cli = Cli::parse();
    airbug::install();
    dispatch::run(cli.command)
}
