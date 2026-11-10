//! Subcommand dispatch and cluster-planning CLI runners for `allpaka`.

use std::path::{Path, PathBuf};

use allpaka_core::fleet::FleetMember;
use allpaka_core::{fleet, plan, presets, replicate, Model, PlanRequest, Speculation, Verdict};
use anyhow::{bail, Context, Result};

use crate::{
    airbug_llama, autotune, bench, client, config, explain, gguf, pipeline, report, serve,
    serving_runtime, verify, watch, Command,
};

pub(crate) fn run(command: Command) -> Result<()> {
    match command {
        Command::Studio {
            bind,
            workspace,
            data_dir,
        } => allpaka_chat::run(bind, workspace, data_dir),
        Command::Explain {
            engine,
            profile,
            json,
        } => explain::run(&engine, &profile, json),
        Command::Autotune {
            engine,
            cache,
            force,
        } => autotune::run(&engine, cache.as_deref(), force),
        Command::Init => {
            print!("{}", config::EXAMPLE);
            Ok(())
        }
        Command::PipelinePlan {
            model,
            config,
            model_id,
            endpoints,
            context,
        } => pipeline::plan_manifest(&model, &config, &model_id, &endpoints, context),
        Command::PipelineServe {
            stages,
            tokenizer,
            model_id,
            bind,
            context,
        } => pipeline::serve_chat(&stages, &tokenizer, &model_id, bind, context),
        Command::Stage {
            model,
            first,
            end,
            model_id,
            bind,
            context,
        } => pipeline::serve(&model, first, end, model_id, bind, context),
        Command::Pipeline {
            stages,
            tokens,
            context,
            generate,
        } => pipeline::generate(&stages, &tokens, context, generate),
        Command::Presets => {
            report::presets(presets::PRESETS);
            Ok(())
        }
        Command::Inspect { model, mmproj } => {
            if mmproj {
                report::vision_census(&model, &gguf::vision::VisionCensus::read(&model)?);
                Ok(())
            } else {
                let info = gguf::read(&model)?;
                report::gguf(&model, &info);
                Ok(())
            }
        }
        Command::Bench {
            serve,
            bind,
            connect,
            mem,
            engine,
            draft,
            config,
        } => match (engine, mem, serve, connect) {
            (Some(model), _, _, _) => {
                apply_runtime_profile(&model, config.as_deref())?;
                bench::measure_engine(&model, draft.as_deref())
            }
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
        },
        Command::Airbug {
            model,
            pp,
            tg,
            repeats,
            warmup,
            cooldown_ms,
            threshold,
            allpaka,
            llama_bench,
            check,
            out,
        } => airbug_llama::run(airbug_llama::Options {
            model,
            pp,
            tg,
            repeats,
            warmup,
            cooldown_ms,
            threshold_percent: threshold,
            check,
            allpaka_bin: airbug_llama::resolve_allpaka_bin(allpaka)?,
            llama_bench: airbug_llama::resolve_llama_bench(llama_bench),
            out: out.unwrap_or_else(airbug_llama::default_out_dir),
        }),
        Command::Plan {
            model,
            config: config_path,
            ctx,
            prompt,
            draft,
            draft_len,
            accept,
        } => {
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
        Command::Fleet {
            config: config_path,
            models,
            ctx,
        } => run_fleet(&config_path, &models, ctx),
        Command::Launch {
            config: config_path,
        } => run_launch(&config_path),
        Command::Verify {
            model,
            addr,
            prompt,
            top,
            tol,
            decode,
        } => verify::run(&model, &addr, &prompt, top, tol, decode),
        Command::Serve {
            model,
            bind,
            config,
            max_queued,
            max_batch,
            batch_context_tokens,
            model_budget_gib,
            memory_budget_mib,
            prefix_cache_mib,
        } => {
            if let Some(primary) = model.first() {
                apply_runtime_profile(primary, config.as_deref())?;
            } else {
                let runtime = config.as_deref().map(config::Config::load).transpose()?;
                runtime
                    .as_ref()
                    .map_or_else(config::RuntimeConfig::default, |c| c.runtime.clone())
                    .apply();
            }
            serve::run(
                &model,
                &bind,
                serving_runtime::ServingLimits {
                    max_queued,
                    max_batch,
                    max_batch_context_tokens: batch_context_tokens,
                    model_budget_bytes: if model_budget_gib == 0 {
                        u64::MAX
                    } else {
                        model_budget_gib.saturating_mul(1 << 30)
                    },
                    prefix_budget_bytes: prefix_cache_mib.saturating_mul(1 << 20),
                    memory_budget_bytes: if memory_budget_mib == 0 {
                        u64::MAX
                    } else {
                        memory_budget_mib
                            .checked_mul(1 << 20)
                            .context("memory budget overflow")?
                    },
                },
            )
        }
        Command::Status { addr } => client::status(&addr),
        Command::Watch {
            addr,
            interval_ms,
            json,
        } => watch::watch(&addr, std::time::Duration::from_millis(interval_ms), json),
        Command::Chat {
            prompt,
            system,
            rag,
            max_tokens,
            model_name,
            addr,
        } => client::chat(
            &addr,
            &prompt,
            system.as_deref(),
            rag,
            max_tokens,
            &model_name,
        ),
        Command::RagTest { model_name, addr } => client::rag_test(&addr, &model_name),
    }
}

fn apply_runtime_profile(primary_model: &Path, config_path: Option<&Path>) -> Result<()> {
    if config_path.is_none() {
        autotune::apply_cached(primary_model, None)?;
    }
    let runtime = config_path.map(config::Config::load).transpose()?;
    runtime
        .as_ref()
        .map_or_else(config::RuntimeConfig::default, |c| c.runtime.clone())
        .apply();
    Ok(())
}

/// Build the planner's view of a model from a GGUF file on disk.
pub(crate) fn load_model(path: &Path, kv_cache_dtype_bytes: u64) -> Result<Model> {
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

fn run_fleet(config_path: &Path, ad_hoc: &[PathBuf], ctx: Option<u32>) -> Result<()> {
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

    let plan = fleet(&members, &nodes)?;
    let ports: Vec<Option<u16>> = if ad_hoc.is_empty() {
        cfg.agents.iter().map(|a| a.port).collect()
    } else {
        vec![None; members.len()]
    };
    report::fleet(&members, &nodes, &plan, &ports);
    Ok(())
}

/// Build the fleet members declared in `[[agents]]`.
fn members_from_config(cfg: &config::Config, config_path: &Path) -> Result<Vec<FleetMember>> {
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

fn run_launch(config_path: &Path) -> Result<()> {
    let cfg = config::Config::load(config_path).with_context(|| {
        format!(
            "loading cluster config; run `allpaka init > {}` to create one",
            config_path.display()
        )
    })?;
    let nodes = cfg.resolve_nodes()?;
    let members = members_from_config(&cfg, config_path)?;
    let plan = fleet(&members, &nodes)?;
    report::launch(&cfg, &nodes, &plan);
    Ok(())
}

fn run_plan(
    model_path: &Path,
    config_path: &Path,
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
