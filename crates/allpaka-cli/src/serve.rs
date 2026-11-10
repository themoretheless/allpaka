//! An OpenAI-compatible chat endpoint over the engine.
//!
//! `POST /v1/chat/completions` with the usual `{model, messages, max_tokens,
//! temperature}` body; non-streaming, one request at a time. The point is that
//! an agent framework pointed at this URL cannot tell it is not llama-server -
//! same route, same response shape.

mod http;
mod inference;
mod ingress;
mod launch;
mod memory;
mod messages;
mod predict;
mod rag_tools;
mod request_control;
mod resources;
mod router;
mod sampler;
mod session;
mod template;

pub(crate) use http::{read_request, respond, respond_html, write_sse_event, write_sse_headers};
pub(crate) use messages::{
    append_valid_utf8, image_part_count, parse_tool_calls, render_messages, stream_safe_len,
};
pub(crate) use predict::{
    extract_prediction_text, parse_prompt_lookup_k, predict_followup_prompts,
    predict_prompt_continuation,
};
pub(crate) use sampler::Sampler;
pub(crate) use session::{
    exact_prefix_snapshot, model_prefix_namespace, prefix_key, ChatState, ModelService, PrefixSnap,
};
pub(crate) use template::Template;

use inference::*;
use rag_tools::{rag_default_tool_schemas, run_rag_tool, RagTools};
use request_control::{Interrupted, Registry as RequestRegistry, RequestControl};
use router::*;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::serving_runtime::{ServingLimits, ServingRuntime};
use allpaka_model::prefix_cache::PrefixCache;
use allpaka_model::{Model, Tokenizer};

/// Generation cap the chat field is allowed to ask for. The session is this
/// plus room for the prompt. KV is allocated up front, so a 512000-token
/// request on a 32B model is on the order of 100 GiB.
const MAX_NEW_TOKENS: usize = 512_000;
const SESSION_TOKENS: usize = MAX_NEW_TOKENS + 32_768;

struct CompletionResult {
    content: String,
    tool_calls: Vec<Value>,
    finish: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill_secs: f64,
    decode_secs: f64,
}

use self::router::ModelRoute;

struct QueuedConnection {
    stream: TcpStream,
    route: router::QueuedRoute,
    request: Value,
    control: RequestControl,
}

type ServerRuntime = ServingRuntime<Mutex<ModelService>, QueuedConnection, PrefixSnap>;

struct InferenceContext<'a> {
    model: &'a Model<'static>,
    tok: &'a Tokenizer,
    template: &'a Template,
    chat: &'a mut ChatState,
    prefixes: &'a mut PrefixCache<PrefixSnap>,
    prefix_namespace: u32,
    control: &'a RequestControl,
}

struct GenerationRequest<'a> {
    messages: &'a [(String, String)],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f64,
    parse_tools: bool,
    prediction_tokens: &'a [u32],
    prompt_lookup_k: usize,
    stream: Option<&'a mut TcpStream>,
    emit_done: bool,
}

struct ServeRequestContext<'a> {
    model_name: &'a str,
    inference: InferenceContext<'a>,
    rag: &'a RagTools,
}

struct PrefillOutcome {
    common: usize,
    prefill_secs: f64,
    cached_logits: Option<Arc<[f32]>>,
}

pub fn run(model_paths: &[std::path::PathBuf], bind: &str, limits: ServingLimits) -> Result<()> {
    anyhow::ensure!(!model_paths.is_empty(), "at least one --model is required");
    anyhow::ensure!(limits.max_queued > 0, "--max-queued must be positive");
    let mut planned_bytes = Vec::with_capacity(model_paths.len());
    for path in model_paths {
        let file = allpaka_gguf::GgufFile::open(path)?;
        planned_bytes.push(file.mappings().map(|m| m.len() as u64).sum::<u64>());
    }
    let total_bytes = planned_bytes.iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .context("resident model byte total overflow")
    })?;
    anyhow::ensure!(
        total_bytes <= limits.model_budget_bytes,
        "models require {:.1} GiB but --model-budget-gib allows {:.1} GiB",
        total_bytes as f64 / (1u64 << 30) as f64,
        limits.model_budget_bytes as f64 / (1u64 << 30) as f64,
    );

    let memory = memory::Budget::new(limits.memory_budget_bytes);
    let _weights = memory.reserve(total_bytes)?;
    // Reserving the whole cache ceiling also covers externally pinned entries.
    let _prefixes = memory.reserve(u64::try_from(limits.prefix_budget_bytes)?)?;
    let mut runtime = ServerRuntime::new(limits);
    let mut default_model = String::new();
    for (path, bytes) in model_paths.iter().zip(planned_bytes) {
        let started = std::time::Instant::now();
        let (name, service) =
            crate::airbug::span("load_model", || load_model_service(path, &memory))?;
        let load_secs = started.elapsed().as_secs_f64();
        crate::airbug::histogram("model_load_s", load_secs);
        crate::airbug::gauge("model_bytes", bytes as f64);
        crate::airbug::gauge("model_layers", service.model.config.n_layers as f64);
        crate::airbug::event(
            "startup",
            &format!("{name} loaded in {load_secs:.1}s, {bytes} bytes"),
        );
        if default_model.is_empty() {
            default_model.clone_from(&name);
        }
        runtime
            .install_model(name.clone(), Mutex::new(service), bytes)
            .map_err(|error| anyhow::anyhow!("installing model {name}: {error:?}"))?;
    }
    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
    launch::remember_bind(bind);
    resources::start();
    println!(
        "allpaka serve: {} on http://{bind}/v1/chat/completions",
        runtime.model_names().join(", ")
    );
    println!(
        "  scheduler: queued={} batch={} context={} tokens; resident {:.1} GiB; prefix cache {:.1} MiB",
        limits.max_queued,
        limits.max_batch,
        limits.max_batch_context_tokens,
        runtime.resident_model_bytes() as f64 / (1u64 << 30) as f64,
        limits.prefix_budget_bytes as f64 / (1u64 << 20) as f64,
    );
    let rag_tools = RagTools::load();
    let incoming = ingress::start(listener, default_model, limits);

    loop {
        let first = incoming.recv().context("HTTP ingress stopped")?;
        let mut ready = vec![first];
        while ready.len() < limits.max_queued {
            match incoming.try_recv() {
                Ok(request) => ready.push(request),
                Err(_) => break,
            }
        }

        for (model_name, context_tokens, connection) in ready {
            let mut rejection_stream = connection.stream.try_clone()?;
            if let Err(error) = runtime.enqueue(model_name, context_tokens, connection) {
                respond(
                    &mut rejection_stream,
                    429,
                    &json!({"error":format!("scheduler: {error:?}")}),
                )?;
            }
        }

        while runtime.queued() > 0 {
            for scheduled in runtime.next_batch() {
                if let Err(error) = dispatch_connection(
                    scheduled.model,
                    scheduled.payload,
                    &mut runtime,
                    &rag_tools,
                ) {
                    println!("request failed: {error:#}");
                }
            }
        }
    }
}

fn load_model_service(
    model_path: &Path,
    memory: &memory::Budget,
) -> Result<(String, ModelService)> {
    // A serving model borrows mmap-backed tensor bytes for the process
    // lifetime. Leaking the owner is deliberate: Metal bytes-no-copy buffers
    // must remain valid until server shutdown, when the OS reclaims mappings.
    let file: &'static allpaka_gguf::GgufFile =
        Box::leak(Box::new(allpaka_gguf::GgufFile::open(model_path)?));
    for mapping in file.mappings() {
        prewarm(mapping);
    }
    let model = Model::load(file)?;
    let tokenizer = Tokenizer::from_gguf(file)?;
    tokenizer.self_check()?;
    let force_think = file
        .meta_str("tokenizer.chat_template")
        .is_some_and(|value| value.contains("<think>"));
    let template_name = match Template::detect(&tokenizer)? {
        Template::ChatMl { .. } => "chatml",
        Template::Llama3 { .. } => "llama3",
        Template::Glm { .. } => "glm",
    };
    let name = model_path
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "allpaka".into());
    let chat = ChatState::new(&model, memory, 1)?;
    println!(
        "  model {name}: arch {}, vocab {}, template {}",
        model.config.architecture,
        tokenizer.vocab_size(),
        template_name,
    );
    Ok((
        name,
        ModelService {
            model,
            tokenizer,
            force_think,
            chat,
            sessions: std::collections::HashMap::new(),
        },
    ))
}

/// Touch every page of the weight mapping once, sequentially.
///
/// Without this the first request faults the whole file in access order
/// during inference - seconds of stutter billed to the first user. One
/// sequential pass at SSD streaming speed moves that cost to startup.
pub fn prewarm(mapping: &[u8]) {
    const PAGE: usize = 16384;
    let (gib, secs) = crate::airbug::span("prewarm", || {
        let t0 = std::time::Instant::now();
        let mut acc = 0u8;
        let mut i = 0;
        while i < mapping.len() {
            acc ^= mapping[i];
            i += PAGE;
        }
        std::hint::black_box(acc);
        let gib = mapping.len() as f64 / (1u64 << 30) as f64;
        let secs = t0.elapsed().as_secs_f64();
        println!("prewarmed {gib:.1} GiB of weights in {secs:.1}s");
        (gib, secs)
    });
    crate::airbug::histogram("prewarm_s", secs);
    crate::airbug::gauge("weights_gib", gib);
}

fn handle(
    mut stream: TcpStream,
    route: ModelRoute,
    req: Value,
    mut context: ServeRequestContext<'_>,
) -> Result<()> {
    let model_name = context.model_name;
    let rag = context.rag;
    // Answered after the model lock is taken, so `/stats` reflects the live
    // context and queues behind an in-flight generation. `/resources` is
    // served before the queue and reports the context as of the last
    // completed generation -- the two endpoints share `context_used` and
    // `context_capacity` with different freshness, which is why the chat
    // page reads each pair from one of them only.
    match route {
        ModelRoute::Stats => {
            return respond(
                &mut stream,
                200,
                &json!({
                    "model": model_name,
                    "architecture": context.inference.model.config.architecture,
                    "context_used": context.inference.chat.tokens.len(),
                    "context_capacity": context.inference.chat.session.capacity(),
                    "n_layers": context.inference.model.config.n_layers,
                    "moe": context.inference.model.config.moe.is_some(),
                    "prefix_cache_entries": context.inference.prefixes.len(),
                    "prefix_cache_bytes": context.inference.prefixes.resident_bytes(),
                    "prefix_cache_allocated_bytes": context.inference.prefixes.allocated_bytes(),
                    "prefix_cache_hits": context.inference.prefixes.hits(),
                    "prefix_cache_misses": context.inference.prefixes.misses(),
                    "prefix_cache_reused_tokens": context.inference.prefixes.reused_tokens(),
                    "memory_admission": memory::snapshot(&context.inference.chat.memory.budget()),
                    "prefix_cache_pinned_bytes": context.inference.prefixes.pinned_bytes(),
                    "batching_mode": "model-aware-admission",
                    "kernel_batching": false,
                    "gpu_fallback": "runtime-policy-controlled",
                }),
            );
        }
        ModelRoute::ChatPage => return respond_html(&mut stream, CHAT_PAGE),
        ModelRoute::CorsOptions => return respond(&mut stream, 200, &json!({})),
        ModelRoute::PromptPredict => {
            let draft = req["draft"].as_str().unwrap_or("");
            let max_k = req["max_tokens"].as_u64().unwrap_or(12).clamp(1, 64) as usize;
            let raw_messages = req["messages"].as_array().map_or(&[][..], Vec::as_slice);
            let completion = predict_prompt_continuation(
                context.inference.tok,
                &context.inference.chat.tokens,
                raw_messages,
                draft,
                max_k,
            );
            let suggestions = predict_followup_prompts(raw_messages);
            return respond(
                &mut stream,
                200,
                &json!({
                    "completion": completion,
                    "suggestions": suggestions,
                }),
            );
        }
        ModelRoute::PromptPrewarm => {
            let raw_messages = req["messages"].as_array().map_or(&[][..], Vec::as_slice);
            let draft = req["draft"].as_str().unwrap_or("").trim();
            let reserve_tokens = req["max_tokens"].as_u64().unwrap_or(1024) as usize;
            let mut turns = render_messages(raw_messages, None);
            if !draft.is_empty() {
                turns.push(("user".into(), draft.to_string()));
            }
            if turns.is_empty() {
                return respond(
                    &mut stream,
                    200,
                    &json!({"prewarmed_tokens": 0, "cached_tokens": 0}),
                );
            }
            let (prewarmed, cached) =
                prewarm_prompt(&mut context.inference, &turns, reserve_tokens)?;
            return respond(
                &mut stream,
                200,
                &json!({
                    "prewarmed_tokens": prewarmed,
                    "cached_tokens": cached,
                    "context_used": context.inference.chat.tokens.len(),
                }),
            );
        }
        ModelRoute::Unknown => {
            return respond(&mut stream, 404, &json!({"error": "unknown route"}));
        }
        ModelRoute::ChatCompletions => {}
    }

    let Some(raw_messages) = req["messages"].as_array() else {
        return respond(&mut stream, 400, &json!({"error": "messages[] required"}));
    };
    let images = image_part_count(raw_messages);
    if images > 0 {
        // The engine has no vision projector, so an image could only be
        // ignored. Fail closed and say so: silently dropping the picture (and
        // the text that travelled with it) is worse than an explicit refusal.
        return respond(
            &mut stream,
            400,
            &json!({
                "error": format!(
                    "no image support: the request carries {images} image part(s) and this \
                     engine has no vision projector. Send text only, or point the client at a \
                     vision-capable provider."
                ),
                "code": "images_unsupported",
            }),
        );
    }
    let tool_specs: Option<Vec<Value>> = match req["tools"].as_array() {
        Some(t) if !t.is_empty() => Some(t.to_vec()),
        _ if rag.cfg.enabled && rag.cfg.inject_tools_when_missing => {
            Some(rag_default_tool_schemas())
        }
        _ => None,
    };
    let tools = tool_specs.as_deref();
    if tools.is_some() && matches!(context.inference.template, Template::Llama3 { .. }) {
        // The Hermes convention below is what Qwen (ChatML) was trained on;
        // pretending Llama 3 understands it would fail silently.
        return respond(
            &mut stream,
            400,
            &json!({"error": "tools are only supported with the ChatML template"}),
        );
    }
    let messages = render_messages(raw_messages, tools);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.7);
    let top_p = req["top_p"].as_f64().unwrap_or(0.95);
    let repeat_penalty = req["repeat_penalty"].as_f64().unwrap_or(1.1);
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let prediction_text = extract_prediction_text(&req["prediction"]);
    let prediction_tokens = if prediction_text.is_empty() {
        Vec::new()
    } else {
        context
            .inference
            .tok
            .encode(&prediction_text)
            .unwrap_or_default()
    };
    let greedy = temperature <= 0.0 && repeat_penalty == 1.0;
    let prompt_lookup_k = parse_prompt_lookup_k(
        &req["prompt_lookup"],
        greedy || !prediction_tokens.is_empty(),
    );

    if streaming && (tools.is_none() || !rag.cfg.enabled) {
        run_inference(
            &mut context.inference,
            GenerationRequest {
                messages: &messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                parse_tools: false,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: Some(&mut stream),
                emit_done: true,
            },
        )?;
        return Ok(());
    }

    let completion = if tools.is_none() {
        run_inference(
            &mut context.inference,
            GenerationRequest {
                messages: &messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                parse_tools: false,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: if streaming { Some(&mut stream) } else { None },
                emit_done: !streaming,
            },
        )?
    } else if !rag.cfg.enabled {
        run_inference(
            &mut context.inference,
            GenerationRequest {
                messages: &messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                parse_tools: true,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: if streaming { Some(&mut stream) } else { None },
                emit_done: !streaming,
            },
        )?
    } else {
        let mut conversation = raw_messages.to_vec();
        let mut final_outcome: Option<CompletionResult> = None;
        for round in 0..rag.cfg.max_tool_rounds {
            let round_messages = render_messages(&conversation, tools);
            let mut outcome = run_inference(
                &mut context.inference,
                GenerationRequest {
                    messages: &round_messages,
                    max_tokens,
                    temperature,
                    top_p,
                    repeat_penalty,
                    parse_tools: true,
                    prediction_tokens: &prediction_tokens,
                    prompt_lookup_k,
                    stream: if streaming { Some(&mut stream) } else { None },
                    emit_done: !streaming,
                },
            )?;
            if outcome.tool_calls.is_empty() {
                final_outcome = Some(outcome);
                break;
            }

            conversation.push(json!({
                "role": "assistant",
                "content": outcome.content.clone(),
                "tool_calls": outcome.tool_calls.clone(),
            }));

            for (i, call) in outcome.tool_calls.iter().enumerate() {
                let result = run_rag_tool(call, rag)?;
                let call_id = call["id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("call_{i}"));
                conversation.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }));
            }

            if round + 1 == rag.cfg.max_tool_rounds {
                outcome.tool_calls.clear();
                outcome.content = format!(
                    "tool loop exhausted after {} rounds. Last tool output could not be resolved into a final answer.",
                    rag.cfg.max_tool_rounds
                );
                final_outcome = Some(outcome);
                break;
            }
        }
        final_outcome.context("tool loop completed without output")?
    };

    if streaming && tools.is_some() && rag.cfg.enabled {
        emit_stream_completion(&mut stream, &completion, false, &completion.finish)?;
        return Ok(());
    }

    respond(
        &mut stream,
        200,
        &json!({
            "id": "chatcmpl-allpaka",
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": Value::String(completion.content.clone()),
                    "tool_calls": if completion.tool_calls.is_empty() {
                        Value::Null
                    } else {
                        Value::Array(completion.tool_calls)
                    },
                },
                "finish_reason": completion.finish,
            }],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": completion.cached_tokens},
            },
        }),
    )
}

/// The built-in chat page: zero dependencies, talks to this same server.
/// Lives as a real .html file so editors and grep see it; embedded at compile
/// time, so the served bytes are unchanged.
const CHAT_PAGE: &str = include_str!("../assets/chat.html");
