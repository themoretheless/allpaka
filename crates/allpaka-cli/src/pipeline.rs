use allpaka_model::pipeline::{self, Pipeline, StageInfo};
use anyhow::{bail, Context, Result};
use std::{
    net::{SocketAddr, TcpListener},
    path::Path,
};
fn key() -> Result<String> {
    let key = std::env::var("ALLPAKA_CLUSTER_KEY")
        .context("set ALLPAKA_CLUSTER_KEY on coordinator and workers")?;
    if key.len() < 16 {
        bail!("ALLPAKA_CLUSTER_KEY must contain at least 16 bytes");
    }
    Ok(key)
}
pub fn serve(
    path: &Path,
    first: usize,
    end: usize,
    model_id: String,
    bind: SocketAddr,
    context: usize,
) -> Result<()> {
    let key = key()?;
    if model_id.trim().is_empty() || context == 0 {
        bail!("model identity and positive context required");
    }
    let file = allpaka_gguf::GgufFile::open(path)?;
    let model = allpaka_model::Model::load_stage(&file, first, end)?;
    if first >= end || end > model.config.n_layers as usize {
        bail!("invalid stage range");
    }
    let info = StageInfo {
        fingerprint: allpaka_gguf::fingerprint(path)?,
        model_id,
        first,
        end,
        layers: model.config.n_layers as usize,
        hidden: model.config.hidden as usize,
        vocab: model.config.vocab,
    };
    let listener = TcpListener::bind(bind)?;
    println!(
        "Stage {first}..{end} listening on {}",
        listener.local_addr()?
    );
    // One active session bounds cache allocations; each connection owns state.
    for stream in listener.incoming() {
        if let Err(error) = pipeline::serve_connection(stream?, &model, info.clone(), &key, context)
        {
            eprintln!("Stage connection ended: {error}");
        }
    }
    Ok(())
}
pub fn generate(
    addresses: &[String],
    tokens: &[u32],
    context: usize,
    generate: usize,
) -> Result<()> {
    if tokens.is_empty()
        || tokens
            .len()
            .checked_add(generate)
            .is_none_or(|n| n > context)
    {
        bail!("prompt and generation must fit context");
    }
    let mut pipeline = Pipeline::connect(addresses, &key()?, context)?;
    let mut logits = Vec::new();
    for &token in tokens {
        logits = pipeline.forward(token)?;
    }
    let mut output = Vec::new();
    for index in 0..generate {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("empty logits")?
            .0 as u32;
        output.push(token);
        if index + 1 < generate {
            logits = pipeline.forward(token)?;
        }
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// Local API gateway; model stages retain all weight and cache allocations.
pub fn serve_chat(
    addresses: &[String],
    tokenizer_path: &Path,
    model_id: &str,
    bind: SocketAddr,
    context: usize,
) -> Result<()> {
    use crate::serve::{self, Template};
    use serde_json::{json, Value};
    use std::{io::Write, time::Duration};
    if !bind.ip().is_loopback() || context == 0 || model_id.is_empty() {
        bail!("pipeline gateway requires loopback, a model ID and positive context");
    }
    let key = key()?;
    let file = allpaka_gguf::GgufFile::open(tokenizer_path)?;
    let fingerprint = allpaka_gguf::fingerprint(tokenizer_path)?;
    let tokenizer = allpaka_model::Tokenizer::from_gguf(&file)?;
    let mut template = Template::detect(&tokenizer)?;
    if let Template::ChatMl { force_think, .. } = &mut template {
        *force_think = file
            .meta_str("tokenizer.chat_template")
            .is_some_and(|value| value.contains("<think>"));
    }
    let stop = template.stop_tokens(&tokenizer);
    let listener = TcpListener::bind(bind)?;
    println!("Distributed model {model_id} API at http://{bind}/v1");
    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        let mut stream_started = false;
        let result: Result<()> = (|| {
            let (line, body) = serve::read_request(&mut stream)?;
            if line.starts_with("GET /v1/models ") {
                return serve::respond(
                    &mut stream,
                    200,
                    &json!({"object":"list","data":[{"id":model_id,"object":"model","owned_by":"allpaka"}]}),
                );
            }
            if !line.starts_with("POST /v1/chat/completions ") {
                return serve::respond(&mut stream, 404, &json!({"error":"unknown endpoint"}));
            }
            let request: Value = serde_json::from_str(&body)?;
            if request["model"].as_str() != Some(model_id) {
                bail!("unknown distributed model");
            }
            let raw = request["messages"]
                .as_array()
                .context("messages array required")?;
            if raw.is_empty() {
                bail!("messages must not be empty");
            }
            if serve::image_part_count(raw) > 0 {
                bail!("distributed gateway does not support image input");
            }
            if request["temperature"]
                .as_f64()
                .is_some_and(|value| value != 0.0)
            {
                bail!("distributed gateway currently supports greedy sampling (temperature 0)");
            }
            let tool_schemas = request["tools"].as_array();
            let messages = serve::render_messages(raw, tool_schemas.map(Vec::as_slice));
            let prompt = template.prompt(&tokenizer, &messages)?;
            let max_tokens = request["max_tokens"].as_u64().unwrap_or(512).min(8192) as usize;
            if prompt.is_empty()
                || prompt
                    .len()
                    .checked_add(max_tokens)
                    .is_none_or(|n| n > context)
            {
                bail!("request exceeds distributed context");
            }
            let mut pipeline = Pipeline::connect(addresses, &key, context)?;
            if pipeline.info().model_id != model_id
                || pipeline.info().fingerprint != fingerprint
                || pipeline.info().vocab as usize != tokenizer.vocab_size()
            {
                bail!("tokenizer gateway and pipeline stages identify different models");
            }
            let mut logits = Vec::new();
            for token in &prompt {
                logits = pipeline.forward(*token)?;
            }
            let streaming = request["stream"].as_bool().unwrap_or(false);
            let mut emitted = String::new();
            if streaming {
                serve::write_sse_headers(&mut stream)?;
                stream_started = true;
            }
            let mut generated = Vec::new();
            let mut finish = "length";
            for index in 0..max_tokens {
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("empty logits")?
                    .0 as u32;
                if stop.contains(&token) {
                    finish = "stop";
                    break;
                }
                generated.push(token);
                if streaming {
                    let full = format!(
                        "{}{}",
                        template.think_prefix(),
                        tokenizer.decode(&generated)
                    );
                    let complete = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
                    let complete = if tool_schemas.is_some() {
                        &complete[..serve::stream_safe_len(complete)]
                    } else {
                        complete
                    };
                    if let Some(delta) = complete.strip_prefix(&emitted) {
                        if !delta.is_empty() {
                            serve::write_sse_event(
                                &mut stream,
                                &json!({"id":"allpaka-pipeline","object":"chat.completion.chunk","model":model_id,"choices":[{"index":0,"delta":{"content":delta},"finish_reason":null}]}),
                            )?;
                            emitted = complete.to_owned();
                        }
                    }
                }
                if index + 1 < max_tokens {
                    logits = pipeline.forward(token)?;
                }
            }
            let text = format!(
                "{}{}",
                template.think_prefix(),
                tokenizer.decode(&generated)
            );
            let (content, calls) = if tool_schemas.is_some() {
                serve::parse_tool_calls(&text)
            } else {
                (text, Vec::new())
            };
            if !calls.is_empty() {
                finish = "tool_calls";
            }
            let usage = json!({"prompt_tokens":prompt.len(),"completion_tokens":generated.len(),"total_tokens":prompt.len()+generated.len()});
            if streaming {
                let remaining = content
                    .strip_prefix(&emitted)
                    .context("streamed content changed during decoding")?;
                let mut delta = json!({"role":"assistant","content":remaining});
                if !calls.is_empty() {
                    delta["tool_calls"] = json!(calls
                        .iter()
                        .enumerate()
                        .map(|(index, call)| {
                            let mut call = call.clone();
                            call["index"] = json!(index);
                            call
                        })
                        .collect::<Vec<_>>());
                }
                serve::write_sse_event(
                    &mut stream,
                    &json!({"id":"allpaka-pipeline","object":"chat.completion.chunk","model":model_id,"choices":[{"index":0,"delta":delta,"finish_reason":null}]}),
                )?;
                serve::write_sse_event(
                    &mut stream,
                    &json!({"id":"allpaka-pipeline","object":"chat.completion.chunk","model":model_id,"choices":[{"index":0,"delta":{},"finish_reason":finish}],"usage":usage}),
                )?;
                stream.write_all(b"data: [DONE]\n\n")?;
                Ok(())
            } else {
                let mut message = json!({"role":"assistant","content":content});
                if !calls.is_empty() {
                    message["tool_calls"] = json!(calls);
                }
                serve::respond(
                    &mut stream,
                    200,
                    &json!({"id":"allpaka-pipeline","object":"chat.completion","model":model_id,"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":usage}),
                )
            }
        })();
        if let Err(error) = result {
            if stream_started {
                // Never send a successful finish event after a partial failure.
                let _ = serve::write_sse_event(&mut stream, &json!({"error":error.to_string()}));
            } else {
                let _ = serve::respond(&mut stream, 400, &json!({"error":error.to_string()}));
            }
        }
    }
    Ok(())
}

/// Emit structured arguments, preserving paths and avoiding shell interpolation.
pub fn plan_manifest(
    model_path: &Path,
    config_path: &Path,
    model_id: &str,
    endpoints: &[String],
    context: Option<u32>,
) -> Result<()> {
    use allpaka_core::{plan, PlanRequest, Verdict};
    use serde_json::json;
    use std::collections::HashMap;
    let config = crate::config::Config::load(config_path)?;
    let nodes = config.resolve_nodes()?;
    let model = crate::load_model(model_path, 2)?;
    let context = context.unwrap_or(config.defaults.context_tokens);
    let request = PlanRequest {
        context_tokens: context,
        prompt_tokens: config.defaults.prompt_tokens,
        speculation: None,
    };
    let fabric = config.resolve_fabric()?;
    let verdict = plan(&nodes, &model, &fabric, &request);
    let plan = match verdict {
        Verdict::SplitWins { plan, .. }
        | Verdict::SplitRequired { plan }
        | Verdict::UseSingleNode { plan, .. } => plan,
        Verdict::Infeasible { .. } => bail!("cluster cannot fit this model and context"),
    };
    let mut addresses = HashMap::new();
    for endpoint in endpoints {
        let (name, address) = endpoint
            .split_once('=')
            .context("endpoint must be node-name=host:port")?;
        if !nodes.iter().any(|node| node.name == name) {
            bail!("unknown endpoint node {name}");
        }
        use std::net::ToSocketAddrs;
        if address.to_socket_addrs()?.next().is_none() {
            bail!("empty worker address");
        }
        if addresses.insert(name, address).is_some() {
            bail!("duplicate endpoint node {name}");
        }
    }
    let mut stages = Vec::new();
    let mut ordered = Vec::new();
    for stage in &plan.stages {
        let address = addresses
            .get(stage.node_name.as_str())
            .with_context(|| format!("missing endpoint for {}", stage.node_name))?;
        let end = stage.first_layer + stage.layer_count;
        let port = address
            .rsplit_once(':')
            .context("endpoint needs a port")?
            .1
            .parse::<u16>()?;
        if port == 0 {
            bail!("endpoint port must be nonzero");
        }
        let bind = format!("0.0.0.0:{port}");
        ordered.push(*address);
        stages.push(json!({"node":stage.node_name,"endpoint":address,"first":stage.first_layer,"end":end,
            "argv":["allpaka","stage",model_path.to_string_lossy(),"--first",stage.first_layer.to_string(),"--end",end.to_string(),"--model-id",model_id,"--context",context.to_string(),"--bind",bind]}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"model_id":model_id,"context":context,"stages":stages,
        "gateway_argv":["allpaka","pipeline-serve","--stages",ordered.join(","),"--tokenizer",model_path.to_string_lossy(),"--model-id",model_id,"--context",context.to_string()]})
        )?
    );
    Ok(())
}
