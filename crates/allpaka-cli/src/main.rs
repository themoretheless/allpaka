//! allpaka - decide whether splitting a model across your machines is worth it,
//! and if so, where to cut it.

mod bench;
mod client;
mod config;
use allpaka_gguf as gguf;
mod rag_mcp;
mod report;
mod serve;
mod verify;

use allpaka_core::fleet::FleetMember;
use allpaka_core::{fleet, plan, presets, replicate, Model, PlanRequest, Speculation, Verdict};
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "allpaka", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print an example allpaka.toml to stdout.
    Init,

    /// List the built-in hardware presets.
    Presets,

    /// Read a GGUF file's architecture metadata.
    Inspect {
        /// Path to a .gguf model file.
        model: PathBuf,
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
        /// Draft model for speculative decoding (with --engine): decodes
        /// twice, plain and speculative, and asserts the streams match.
        #[arg(long, requires = "engine")]
        draft: Option<PathBuf>,
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
        /// Path to the .gguf to serve.
        #[arg(long)]
        model: PathBuf,
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:8099")]
        bind: String,
    },

    /// Probe a running `allpaka serve`: /health plus /stats.
    Status {
        /// Address of the running server.
        #[arg(long, default_value = client::DEFAULT_ADDR)]
        addr: String,
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
    match Cli::parse().command {
        Command::Init => {
            print!("{}", config::EXAMPLE);
            Ok(())
        }
        Command::Presets => {
            report::presets(presets::PRESETS);
            Ok(())
        }
        Command::Inspect { model } => {
            let info = gguf::read(&model)?;
            report::gguf(&model, &info);
            Ok(())
        }
        Command::Bench { serve, bind, connect, mem, engine, draft } => {
            match (engine, mem, serve, connect) {
                (Some(model), _, _, _) => bench::measure_engine(&model, draft.as_deref()),
                (None, true, _, _) => bench::measure_memory(),
                (None, false, true, _) => bench::serve(&bind),
                (None, false, false, Some(addr)) => {
                    println!("measuring path to {addr} ...");
                    let link = bench::measure(&addr)?;
                    report::link(&link);
                    Ok(())
                }
                (None, false, false, None) => {
                    bail!("pass --serve, --connect <host:port>, --mem, or --engine <model>")
                }
            }
        }
        Command::Plan { model, config: config_path, ctx, prompt, draft, draft_len, accept } => {
            let spec = match draft {
                None => None,
                Some(path) => {
                    if !(0.0..=1.0).contains(&accept) {
                        bail!("--accept must be between 0 and 1, got {accept}");
                    }
                    let d = gguf::read(&path)?;
                    Some(Speculation {
                        draft_weight_bytes: d.file_bytes,
                        draft_tokens: draft_len,
                        acceptance_rate: accept,
                    })
                }
            };
            run_plan(&model, &config_path, ctx, prompt, spec)
        }
        Command::Fleet { config: config_path, models, ctx } => {
            run_fleet(&config_path, &models, ctx)
        }
        Command::Launch { config: config_path } => run_launch(&config_path),
        Command::Verify { model, addr, prompt, top, tol, decode } => {
            verify::run(&model, &addr, &prompt, top, tol, decode)
        }
        Command::Serve { model, bind } => serve::run(&model, &bind),
        Command::Status { addr } => client::status(&addr),
        Command::Chat { prompt, system, rag, max_tokens, model_name, addr } => {
            client::chat(&addr, &prompt, system.as_deref(), rag, max_tokens, &model_name)
        }
        Command::RagTest { model_name, addr } => client::rag_test(&addr, &model_name),
    }
}

/// Build the planner's view of a model from a GGUF file on disk.
fn load_model(path: &std::path::Path, kv_cache_dtype_bytes: u64) -> Result<Model> {
    let info = gguf::read(path)?;
    Ok(Model {
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| info.architecture.clone()),
        n_layers: info.block_count,
        hidden_size: info.embedding_length,
        total_weight_bytes: info.file_bytes,
        kv_bytes_per_token_per_layer: info.kv_bytes_per_token_per_layer(kv_cache_dtype_bytes),
        activation_bytes: 2,
        active_weight_fraction: info.active_weight_fraction(),
        param_count: info.param_count,
    })
}

fn run_fleet(
    config_path: &std::path::Path,
    ad_hoc: &[PathBuf],
    ctx: Option<u32>,
) -> Result<()> {
    let cfg = config::Config::load(config_path).with_context(|| {
        format!(
            "loading cluster config; run `allpaka init > {}` to create one",
            config_path.display()
        )
    })?;
    let nodes = cfg.resolve_nodes()?;
    let kv = cfg.defaults.kv_cache_dtype_bytes;

    let members: Vec<FleetMember> = if !ad_hoc.is_empty() {
        ad_hoc
            .iter()
            .map(|p| {
                Ok(FleetMember {
                    model: load_model(p, kv)?,
                    context_tokens: ctx.unwrap_or(cfg.defaults.context_tokens),
                    prompt_tokens: cfg.defaults.prompt_tokens,
                    pin: None,
                    speculation: None,
                })
            })
            .collect::<Result<_>>()?
    } else {
        members_from_config(&cfg, config_path)?
    };

    let plan = fleet(&members, &nodes).map_err(|e| anyhow::anyhow!(e))?;
    let ports: Vec<Option<u16>> = if ad_hoc.is_empty() {
        cfg.agents.iter().map(|a| a.port).collect()
    } else {
        vec![None; members.len()]
    };
    report::fleet(&members, &nodes, &plan, &ports);
    Ok(())
}

/// Build the fleet members declared in `[[agents]]`.
fn members_from_config(
    cfg: &config::Config,
    config_path: &std::path::Path,
) -> Result<Vec<FleetMember>> {
    if cfg.agents.is_empty() {
        bail!(
            "{} declares no [[agents]]; add them there, or pass --model for a one-off",
            config_path.display()
        );
    }
    let kv = cfg.defaults.kv_cache_dtype_bytes;
    cfg.agents
        .iter()
        .map(|a| {
            let mut model = load_model(&a.model, kv)?;
            // The agent's own name is what the operator recognises; the
            // file stem is an implementation detail of where it was saved.
            model.name = a.name.clone();
            let speculation = match &a.draft {
                None => None,
                Some(d) => {
                    if !(0.0..=1.0).contains(&a.accept) {
                        bail!("agent {:?}: accept must be between 0 and 1", a.name);
                    }
                    Some(Speculation {
                        draft_weight_bytes: gguf::read(d)?.file_bytes,
                        draft_tokens: a.draft_len,
                        acceptance_rate: a.accept,
                    })
                }
            };
            Ok(FleetMember {
                model,
                context_tokens: a.ctx.unwrap_or(cfg.defaults.context_tokens),
                prompt_tokens: a.prompt.unwrap_or(cfg.defaults.prompt_tokens),
                pin: cfg.pin_index(a)?,
                speculation,
            })
        })
        .collect()
}

fn run_launch(config_path: &std::path::Path) -> Result<()> {
    let cfg = config::Config::load(config_path).with_context(|| {
        format!(
            "loading cluster config; run `allpaka init > {}` to create one",
            config_path.display()
        )
    })?;
    let nodes = cfg.resolve_nodes()?;
    let members = members_from_config(&cfg, config_path)?;
    let plan = fleet(&members, &nodes).map_err(|e| anyhow::anyhow!(e))?;
    report::launch(&cfg, &nodes, &plan);
    Ok(())
}

fn run_plan(
    model_path: &std::path::Path,
    config_path: &std::path::Path,
    ctx: Option<u32>,
    prompt: Option<u32>,
    speculation: Option<Speculation>,
) -> Result<()> {
    let cfg = config::Config::load(config_path).with_context(|| {
        format!(
            "loading cluster config; run `allpaka init > {}` to create one",
            config_path.display()
        )
    })?;
    let nodes = cfg.resolve_nodes()?;

    let req = PlanRequest {
        context_tokens: ctx.unwrap_or(cfg.defaults.context_tokens),
        prompt_tokens: prompt.unwrap_or(cfg.defaults.prompt_tokens),
        speculation,
    };

    let model = load_model(model_path, cfg.defaults.kv_cache_dtype_bytes)?;

    let fabric = cfg.resolve_fabric()?;
    if fabric.is_empty() {
        report::missing_link(&nodes, &model, &req);
        return Ok(());
    }

    let verdict = plan(&nodes, &model, &fabric, &req);
    report::verdict(&nodes, &model, &req, &fabric, &verdict);

    // Replication is a different question, so it is reported next to the split
    // rather than folded into the verdict.
    let reference = match &verdict {
        Verdict::SplitWins { plan, .. }
        | Verdict::SplitRequired { plan }
        | Verdict::UseSingleNode { plan, .. } => Some(plan),
        Verdict::Infeasible { .. } => None,
    };
    if let Some(reference) = reference {
        report::replication(replicate(&nodes, &model, &req).as_ref(), reference);
    }
    Ok(())
}
