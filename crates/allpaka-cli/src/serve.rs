//! An OpenAI-compatible chat endpoint over the engine.
//!
//! `POST /v1/chat/completions` with the usual `{model, messages, max_tokens,
//! temperature}` body; non-streaming, one request at a time. The point is that
//! an agent framework pointed at this URL cannot tell it is not llama-server -
//! same route, same response shape.

mod http;
mod ingress;
mod launch;
mod memory;
mod messages;
mod predict;
mod rag_tools;
mod request_control;
mod resources;
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

use rag_tools::{rag_default_tool_schemas, run_rag_tool, RagTools};
use request_control::{Interrupted, Registry as RequestRegistry, RequestControl};

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

struct QueuedConnection {
    stream: TcpStream,
    request_line: String,
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

/// Shared prefix/SSM session alignment, chunked `forward_batch` prefill, and
/// `PrefixCache` snapshot insertion used by both `prewarm_prompt` and `run_inference`.
fn prefill_prompt(
    context: &mut InferenceContext<'_>,
    prompt: &[u32],
    max_tokens: usize,
    need_logits: bool,
) -> Result<PrefillOutcome> {
    let model = context.model;
    let chat = &mut *context.chat;
    let prefixes = &mut *context.prefixes;
    let prefix_namespace = context.prefix_namespace;

    let mut common = chat
        .tokens
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| **a == **b)
        .count();
    let mut logits: Vec<f32> = Vec::new();
    let mut cached_logits: Option<Arc<[f32]>> = None;
    let mut skip_prefill = false;

    if prompt.len() + max_tokens + 1 > chat.session.capacity() {
        chat.reset(model, prompt.len() + max_tokens + 1)?;
        chat.tokens.clear();
        common = 0;
    } else if chat.session.ssm.is_some() {
        // The gated-delta-net state cannot be rolled back by truncating
        // like the KV cache: a shared prefix is only reusable through the
        // snapshot taken at that prefix's end. Without one, replay from
        // scratch (slow, but correct).
        if common < chat.tokens.len() {
            match exact_prefix_snapshot(prefixes, prefix_namespace, &prompt[..common]) {
                Some(snap) => {
                    let state = snap
                        .ssm_state
                        .as_ref()
                        .expect("SSM prefix snapshot must contain recurrent state");
                    chat.session
                        .ssm
                        .as_mut()
                        .expect("ssm session")
                        .restore(state);
                    chat.session.truncate(common);
                    chat.tokens.truncate(common);
                    if common == prompt.len() {
                        cached_logits = Some(Arc::clone(&snap.logits));
                        skip_prefill = true;
                    }
                }
                None => {
                    chat.reset(model, prompt.len() + max_tokens + 1)?;
                    chat.tokens.clear();
                    common = 0;
                }
            }
        } else if common == prompt.len() {
            match exact_prefix_snapshot(prefixes, prefix_namespace, prompt) {
                Some(snap) => {
                    cached_logits = Some(Arc::clone(&snap.logits));
                    skip_prefill = true;
                }
                None if !need_logits => {
                    return Ok(PrefillOutcome {
                        common,
                        prefill_secs: 0.0,
                        cached_logits: None,
                    });
                }
                None => {
                    chat.reset(model, prompt.len() + max_tokens + 1)?;
                    chat.tokens.clear();
                    common = 0;
                }
            }
        }
    } else {
        if common == prompt.len() {
            if let Some(snap) = exact_prefix_snapshot(prefixes, prefix_namespace, prompt) {
                cached_logits = Some(Arc::clone(&snap.logits));
                skip_prefill = true;
            } else {
                common = common.saturating_sub(1);
            }
        }
        chat.session.truncate(common);
        chat.tokens.truncate(common);
    }

    let prefill_chunk = allpaka_model::Model::prefill_chunk();
    let t0 = std::time::Instant::now();
    crate::airbug::span("prefill", || -> Result<()> {
        if !skip_prefill {
            for chunk in prompt[common..].chunks(prefill_chunk) {
                context.control.check()?;
                logits = model.forward_batch(chunk, &mut chat.session)?;
                chat.tokens.extend_from_slice(chunk);
            }
        }
        Ok(())
    })?;
    let prefill_secs = t0.elapsed().as_secs_f64();

    if !logits.is_empty() {
        let ssm_state = chat.session.ssm.as_ref().map(|ssm| ssm.snapshot());
        let shared_logits: Arc<[f32]> = Arc::from(std::mem::take(&mut logits));
        let value_bytes = shared_logits.len() * size_of::<f32>()
            + ssm_state
                .as_ref()
                .map_or(0, |state| state.len() * size_of::<f32>());
        cached_logits = Some(Arc::clone(&shared_logits));
        prefixes.insert(
            prefix_key(prefix_namespace, prompt),
            PrefixSnap {
                tokens: prompt.len(),
                ssm_state,
                logits: shared_logits,
            },
            value_bytes,
        );
    }

    Ok(PrefillOutcome {
        common,
        prefill_secs,
        cached_logits,
    })
}

/// Speculatively prefill `messages` into the active session and `PrefixCache`
/// while the user is still typing, reducing Time-To-First-Token (TTFT) on submit.
fn prewarm_prompt(
    context: &mut InferenceContext<'_>,
    messages: &[(String, String)],
    reserve_tokens: usize,
) -> Result<(usize, usize)> {
    context.control.check()?;
    let prompt = context.template.prompt(context.tok, messages)?;
    let max_tokens = reserve_tokens.min(MAX_NEW_TOKENS);
    if prompt.is_empty()
        || prompt.len().saturating_add(max_tokens).saturating_add(1) > SESSION_TOKENS
    {
        return Ok((0, 0));
    }

    let outcome = prefill_prompt(context, &prompt, max_tokens, false)?;
    Ok((prompt.len().saturating_sub(outcome.common), outcome.common))
}

fn emit_incremental_sse(
    stream: &mut TcpStream,
    raw_bytes: &[u8],
    decoded_bytes: &mut usize,
    complete_buf: &mut String,
    emitted_len: &mut usize,
    parse_tools: bool,
    tokens_generated: usize,
) -> Result<()> {
    append_valid_utf8(raw_bytes, decoded_bytes, complete_buf, 0);
    let complete = if parse_tools {
        &complete_buf[..stream_safe_len(complete_buf)]
    } else {
        complete_buf.as_str()
    };
    if complete.len() > *emitted_len {
        let delta = &complete[*emitted_len..];
        let chunk = json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": delta}}],
            "tokens_generated": tokens_generated,
        });
        write_sse_event(stream, &chunk)?;
        *emitted_len = complete.len();
    }
    Ok(())
}

fn run_inference(
    context: &mut InferenceContext<'_>,
    request: GenerationRequest<'_>,
) -> Result<CompletionResult> {
    context.control.check()?;
    let GenerationRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        repeat_penalty,
        parse_tools,
        prediction_tokens,
        prompt_lookup_k,
        mut stream,
        emit_done,
    } = request;
    let prompt = context.template.prompt(context.tok, messages)?;
    anyhow::ensure!(
        max_tokens <= MAX_NEW_TOKENS
            && prompt.len().saturating_add(max_tokens).saturating_add(1) <= SESSION_TOKENS,
        "request exceeds the {MAX_NEW_TOKENS}-token generation limit"
    );
    let stops = context.template.stop_tokens(context.tok);
    let mut sampler = Sampler::new(temperature, top_p, repeat_penalty);
    // Greedy without a repeat penalty can take the on-GPU argmax: one word
    // read back per token instead of the whole vocabulary.
    let greedy = temperature <= 0.0 && repeat_penalty == 1.0;

    let PrefillOutcome {
        common,
        prefill_secs,
        mut cached_logits,
    } = prefill_prompt(context, &prompt, max_tokens, true)?;
    let mut logits: Vec<f32> = Vec::new();

    let model = context.model;
    let tok = context.tok;
    let template = context.template;
    let chat = &mut *context.chat;

    let streaming = stream.is_some();
    let think_prefix = template.think_prefix();
    if streaming {
        write_sse_headers(stream.as_mut().unwrap())?;
        context.control.start_stream();
        // The primed `<think>\n` is part of the reply text but not of the
        // generated token stream; send it ahead of the first real delta.
        if !think_prefix.is_empty() {
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": think_prefix}}],
                }),
            )?;
        }
    }

    let t1 = std::time::Instant::now();
    let mut generated = Vec::with_capacity(max_tokens);
    let mut raw_bytes = Vec::with_capacity(max_tokens * 4);
    let mut complete_buf = String::with_capacity(if streaming { max_tokens * 4 } else { 0 });
    let mut decoded_bytes = 0usize;
    let mut emitted_len = 0usize;
    let mut finish = "length";
    let mut greedy_next: Option<u32> = None;
    let use_pld = prompt_lookup_k > 0;
    // Draft length follows acceptance; at k=0 the step below decodes plainly
    // and only advances the controller's cooldown.
    let mut adapt = allpaka_model::speculate::AdaptiveK::new(prompt_lookup_k);
    crate::airbug::span("decode", || -> Result<()> {
        while generated.len() < max_tokens {
            context.control.check()?;
            let next = match greedy_next.take() {
                Some(t) => t,
                None => {
                    let active_logits = cached_logits.take();
                    let slice = active_logits.as_deref().unwrap_or(&logits);
                    sampler.pick(slice, &chat.tokens)
                }
            };
            if stops.contains(&next) {
                finish = "stop";
                break;
            }
            let remaining = max_tokens - generated.len();
            let k = adapt.k().min(remaining.saturating_sub(1));
            if use_pld && k == 0 {
                adapt.record(0, 0);
            }
            if use_pld && k > 0 {
                let mut spec = allpaka_model::speculate::PromptLookupSpeculator {
                    model,
                    session: &mut chat.session,
                    k,
                    ngram_min: if prediction_tokens.is_empty() { 2 } else { 1 },
                    ngram_max: 4,
                };
                let round = if greedy {
                    spec.round(&chat.tokens, prediction_tokens, next)?
                } else {
                    let (r, committed_logits) = spec.round_with_sampler(
                        &chat.tokens,
                        prediction_tokens,
                        next,
                        |l, rec| sampler.pick(l, rec),
                    )?;
                    logits = committed_logits;
                    r
                };
                adapt.record(round.drafted, round.accepted);
                let mut hit_stop = false;
                let mut accepted_before_stop = 0usize;
                for &emitted_tok in &round.emitted {
                    if stops.contains(&emitted_tok) {
                        finish = "stop";
                        hit_stop = true;
                        break;
                    }
                    accepted_before_stop += 1;
                    generated.push(emitted_tok);
                    tok.append_token_bytes(emitted_tok, &mut raw_bytes);
                    chat.tokens.push(emitted_tok);
                    if generated.len() >= max_tokens {
                        break;
                    }
                }
                if accepted_before_stop < round.emitted.len() {
                    let valid_pos = chat.tokens.len();
                    if chat.session.ssm.is_none() {
                        chat.session.truncate(valid_pos);
                    }
                }
                if let Some(stream) = stream.as_mut() {
                    emit_incremental_sse(
                        stream,
                        &raw_bytes,
                        &mut decoded_bytes,
                        &mut complete_buf,
                        &mut emitted_len,
                        parse_tools,
                        generated.len(),
                    )?;
                }
                if hit_stop || generated.len() >= max_tokens {
                    break;
                }
                greedy_next = Some(round.next);
                continue;
            }
            generated.push(next);
            tok.append_token_bytes(next, &mut raw_bytes);
            if let Some(stream) = stream.as_mut() {
                emit_incremental_sse(
                    stream,
                    &raw_bytes,
                    &mut decoded_bytes,
                    &mut complete_buf,
                    &mut emitted_len,
                    parse_tools,
                    generated.len(),
                )?;
            }
            if greedy {
                greedy_next = Some(model.forward_greedy(next, &mut chat.session)?);
            } else {
                logits = model.forward(next, &mut chat.session)?;
            }
            chat.tokens.push(next);
        }
        Ok(())
    })?;
    let decode_secs = t1.elapsed().as_secs_f64();

    let text = format!("{think_prefix}{}", String::from_utf8_lossy(&raw_bytes));
    let (content, tool_calls) = if parse_tools {
        parse_tool_calls(&text)
    } else {
        (text.clone(), Vec::new())
    };
    if !tool_calls.is_empty() && finish == "stop" {
        finish = "tool_calls";
    }
    println!(
        "  {} new + {common} cached prompt tok in {prefill_secs:.1}s, \
         {} generated in {decode_secs:.1}s ({:.1} tok/s)",
        prompt.len() - common,
        generated.len(),
        generated.len() as f64 / decode_secs.max(1e-9),
    );
    let new_prompt = prompt.len() - common;
    if new_prompt > 0 {
        crate::airbug::log_rate(
            "prefill",
            new_prompt,
            prefill_secs,
            new_prompt as f64 / prefill_secs.max(1e-9),
        );
    }
    if !generated.is_empty() {
        crate::airbug::log_rate(
            "decode",
            generated.len(),
            decode_secs,
            generated.len() as f64 / decode_secs.max(1e-9),
        );
    }
    resources::note_generation(
        if new_prompt > 0 {
            new_prompt as f64 / prefill_secs.max(1e-9)
        } else {
            0.0
        },
        if generated.is_empty() {
            0.0
        } else {
            generated.len() as f64 / decode_secs.max(1e-9)
        },
        chat.tokens.len(),
        chat.session.capacity(),
    );

    if streaming {
        // `complete_buf[..emitted_len]` tracks the body only; the think prefix went out first.
        if let Some(delta) =
            content[think_prefix.len()..].strip_prefix(&complete_buf[..emitted_len])
        {
            if !delta.is_empty() {
                write_sse_event(
                    stream.as_mut().unwrap(),
                    &json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta}}],
                    }),
                )?;
            }
        }
        if !tool_calls.is_empty() {
            let calls_delta: Vec<Value> = tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut c = c.clone();
                    c["index"] = json!(i);
                    c
                })
                .collect();
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"tool_calls": calls_delta}}],
                }),
            )?;
        }
        if emit_done {
            write_sse_event(
                stream.as_mut().unwrap(),
                &json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": prompt.len(),
                        "completion_tokens": generated.len(),
                        "total_tokens": prompt.len() + generated.len(),
                        "prompt_tokens_details": {"cached_tokens": common},
                    },
                    "timing": {
                        "prefill_secs": prefill_secs,
                        "decode_secs": decode_secs,
                        "tokens_per_sec": generated.len() as f64 / decode_secs.max(1e-9),
                        "cached_tokens": common,
                        "context_used": chat.tokens.len(),
                        "context_capacity": chat.session.capacity(),
                    },
                }),
            )?;
            stream.as_mut().unwrap().write_all(b"data: [DONE]\n\n")?;
        }
    }

    Ok(CompletionResult {
        content,
        tool_calls,
        finish: finish.to_string(),
        prompt_tokens: prompt.len(),
        completion_tokens: generated.len(),
        cached_tokens: common,
        prefill_secs,
        decode_secs,
    })
}

fn emit_stream_completion(
    stream: &mut TcpStream,
    completion: &CompletionResult,
    emit_content: bool,
    finish: &str,
) -> Result<()> {
    if emit_content && !completion.content.is_empty() {
        write_sse_event(
            stream,
            &json!({
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": completion.content}}],
            }),
        )?;
    }
    write_sse_event(
        stream,
        &json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
            "timing": {
                "prefill_secs": completion.prefill_secs,
                "decode_secs": completion.decode_secs,
                "tokens_per_sec": completion.completion_tokens as f64
                    / completion.decode_secs.max(1e-9),
                "cached_tokens": completion.cached_tokens,
            },
        }),
    )?;
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
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

/// Answers every request that does not need a loaded model. Which routes are in
/// this set is a contract, not a cache: `/health` and `/resources` are the only
/// probes that answer while the model lock is held, and `watch::classify` uses
/// exactly that asymmetry to tell an outage from a busy engine. Moving a
/// post-lock route up here would make every probe answer pre-lock and turn
/// "held by the generation" into a silent "Ready".
fn prepare_connection(
    mut stream: TcpStream,
    default_model: &str,
    max_context_tokens: usize,
    registry: &Arc<RequestRegistry>,
) -> Result<Option<(String, usize, QueuedConnection)>> {
    let (request_line, body) = read_request(&mut stream)?;
    // Exact request line: a prefix match here used to answer 200 for
    // `/healthz` and anything else starting with `/health`.
    if request_line == "GET /health HTTP/1.1" {
        respond(&mut stream, 200, &json!({"status":"ok"}))?;
        return Ok(None);
    }
    if request_line.starts_with("GET /resources ") {
        respond(&mut stream, 200, &resources::snapshot())?;
        return Ok(None);
    }
    if let Some(route) = request_line.strip_prefix("POST /v1/requests/") {
        if let Some(id) = route.strip_suffix("/cancel HTTP/1.1") {
            let cancelled = registry.cancel(id);
            respond(
                &mut stream,
                if cancelled { 200 } else { 404 },
                &json!({"cancelled":cancelled}),
            )?;
            return Ok(None);
        }
    }
    let body_len = body.len();
    let request: Value = if body.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&body)?
    };
    if let Some(id) = request.get("session_id") {
        anyhow::ensure!(
            id.as_str()
                .is_some_and(|id| !id.is_empty() && id.len() <= 128),
            "invalid session_id"
        );
    }
    if let Some(id) = request.get("request_id") {
        anyhow::ensure!(id.is_string(), "request_id must be a string");
    }
    if let Some(tokens) = request.get("max_tokens") {
        anyhow::ensure!(
            tokens
                .as_u64()
                .is_some_and(|n| n > 0 && n <= MAX_NEW_TOKENS as u64),
            "max_tokens must be in 1..={MAX_NEW_TOKENS}"
        );
    }
    let timeout_ms = match request.get("timeout_ms") {
        None => 120_000,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=600_000).contains(n))
            .context("timeout_ms must be in 1..=600000")?,
    };
    let control = registry.register(
        request["request_id"].as_str(),
        std::time::Duration::from_millis(timeout_ms),
    )?;
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(
        timeout_ms.max(30_000),
    )));
    let requested = request["model"]
        .as_str()
        .map(ToOwned::to_owned)
        .filter(|model| model != "allpaka")
        .unwrap_or_else(|| default_model.to_string());
    let context_tokens = body_len.div_ceil(4).clamp(1, max_context_tokens.max(1));
    Ok(Some((
        requested,
        context_tokens,
        QueuedConnection {
            stream,
            request_line,
            request,
            control,
        },
    )))
}

fn resolve_model_name(requested: &str, available: &[String]) -> Option<String> {
    if available.iter().any(|name| name == requested) {
        return Some(requested.to_owned());
    }
    let mut matches = available
        .iter()
        .filter(|name| name.starts_with(requested))
        .cloned();
    let candidate = matches.next()?;
    matches.next().is_none().then_some(candidate)
}

#[cfg(test)]
mod model_resolution_tests {
    use super::resolve_model_name;

    #[test]
    fn aliases_must_be_unique_and_exact_names_win() {
        let names = vec!["qwen3".into(), "qwen3-0.6b".into(), "llama-3".into()];

        assert_eq!(
            resolve_model_name("qwen3", &names).as_deref(),
            Some("qwen3")
        );
        assert_eq!(
            resolve_model_name("qwen3-0", &names).as_deref(),
            Some("qwen3-0.6b")
        );
        assert_eq!(resolve_model_name("qwen", &names), None);
        assert_eq!(resolve_model_name("mistral", &names), None);
    }
}

fn dispatch_connection(
    model_name: String,
    connection: QueuedConnection,
    runtime: &mut ServerRuntime,
    rag: &RagTools,
) -> Result<()> {
    if connection.request_line.starts_with("GET /v1/catalog") {
        return respond(
            &mut connection.stream.try_clone()?,
            200,
            &json!({"object": "list", "data": launch::list_models()}),
        );
    }
    if connection.request_line.starts_with("POST /v1/launch") {
        let path = connection.request["path"].as_str().unwrap_or("");
        let mut stream = connection.stream;
        match launch::schedule(path, "") {
            Ok(()) => {
                respond(&mut stream, 200, &json!({"status": "restarting"}))?;
                let _ = stream.flush();
                println!("relaunch: {path}");
                std::process::exit(0);
            }
            Err(error) => {
                return respond(&mut stream, 400, &json!({"error": error.to_string()}));
            }
        }
    }
    if connection.request_line.starts_with("GET /v1/models") {
        let data = runtime
            .model_names()
            .into_iter()
            .map(|id| json!({"id": id, "object": "model", "owned_by": "allpaka"}))
            .collect::<Vec<_>>();
        return respond(
            &mut connection.stream.try_clone()?,
            200,
            &json!({"object": "list", "data": data}),
        );
    }
    let available_models = runtime.model_names();
    let Some(model_name) = resolve_model_name(&model_name, &available_models) else {
        let mut stream = connection.stream;
        return respond(
            &mut stream,
            404,
            &json!({"error": format!("unknown model: {model_name}")}),
        );
    };
    let Some(service) = runtime.model(&model_name) else {
        anyhow::bail!("resolved model disappeared from registry: {model_name}");
    };
    let mut service = service
        .lock()
        .map_err(|_| anyhow::anyhow!("model service lock poisoned: {model_name}"))?;
    let is_chat = connection
        .request_line
        .starts_with("POST /v1/chat/completions")
        || connection
            .request_line
            .starts_with("POST /v1/prompt/prewarm");
    let session_id = connection
        .request
        .get("session_id")
        .map(|value| {
            value
                .as_str()
                .filter(|id| !id.is_empty() && id.len() <= 128)
                .map(str::to_owned)
                .context("session_id must be a non-empty string of at most 128 bytes")
        })
        .transpose()?;
    let now = std::time::Instant::now();
    service
        .sessions
        .retain(|_, (_, used)| now.duration_since(*used).as_secs() < 600);

    let ModelService {
        model,
        tokenizer,
        force_think,
        chat: default_chat,
        sessions,
    } = &mut *service;

    // Checkout a named session if `session_id` was provided on a chat/prewarm
    // route; otherwise reuse `default_chat` across non-sessioned turns without
    // destroying its KV cache or allocating dummy 1-token sessions.
    let mut checked_out_session: Option<(String, ChatState, bool)> = None;
    if is_chat {
        if let Some(id) = session_id {
            if !sessions.contains_key(&id) && sessions.len() >= 4 {
                let mut stream = connection.stream;
                return respond(
                    &mut stream,
                    429,
                    &json!({"error":"resident session capacity exhausted"}),
                );
            }
            match sessions.remove(&id) {
                Some((session_chat, _)) => {
                    checked_out_session = Some((id, session_chat, true));
                }
                None => match ChatState::new(model, &default_chat.memory.budget(), 1) {
                    Ok(session_chat) => {
                        checked_out_session = Some((id, session_chat, false));
                    }
                    Err(error) => {
                        return respond(
                            &mut connection.stream.try_clone()?,
                            429,
                            &json!({"error":error.to_string()}),
                        );
                    }
                },
            }
        }
    }

    let active_chat: &mut ChatState = match checked_out_session.as_mut() {
        Some((_, session_chat, _)) => session_chat,
        None => default_chat,
    };

    let mut template = Template::detect(tokenizer)?;
    if let Template::ChatMl {
        force_think: enabled,
        ..
    } = &mut template
    {
        *enabled = *force_think;
    }
    let prefix_namespace = model_prefix_namespace(&model_name);
    let prefixes = runtime.prefixes();
    let mut error_stream = connection.stream.try_clone()?;
    let outcome = handle(
        connection.stream,
        connection.request_line,
        connection.request,
        ServeRequestContext {
            model_name: &model_name,
            inference: InferenceContext {
                model,
                tok: tokenizer,
                template: &template,
                chat: active_chat,
                prefixes,
                prefix_namespace,
                control: &connection.control,
            },
            rag,
        },
    );
    let succeeded = outcome.is_ok();
    if let Err(error) = outcome {
        let status = if error.downcast_ref::<Interrupted>().is_some() {
            408
        } else if error.downcast_ref::<memory::Exhausted>().is_some() {
            429
        } else {
            500
        };
        let body = json!({"error":error.to_string(), "request_id":connection.control.id});
        if connection.control.streaming() {
            write_sse_event(&mut error_stream, &body)?;
            error_stream.write_all(b"data: [DONE]\n\n")?;
        } else {
            respond(&mut error_stream, status, &body)?;
        }
    }
    if is_chat {
        if !succeeded {
            active_chat.sanitize_after_failure(model)?;
        }
        if let Some((id, session_chat, had_existing_session)) = checked_out_session {
            if succeeded || had_existing_session {
                sessions.insert(id, (session_chat, std::time::Instant::now()));
            }
        }
    }
    Ok(())
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
    request_line: String,
    req: Value,
    mut context: ServeRequestContext<'_>,
) -> Result<()> {
    let model_name = context.model_name;
    let rag = context.rag;
    // Answered after the model lock is taken, so this reflects the live
    // context and queues behind an in-flight generation. `/resources` is
    // served before the queue and reports the context as of the last
    // completed generation -- the two endpoints share `context_used` and
    // `context_capacity` with different freshness, which is why the chat
    // page reads each pair from one of them only.
    //
    // Prefix match, deliberately: `/stats?ts=...` is a normal request from a
    // polling client. Tightening it to an exact line is a separate call.
    if request_line.starts_with("GET /stats") {
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
    if request_line.starts_with("GET / ") || request_line.starts_with("GET /index") {
        return respond_html(&mut stream, CHAT_PAGE);
    }
    if request_line.starts_with("OPTIONS ") {
        return respond(&mut stream, 200, &json!({}));
    }
    if request_line.starts_with("POST /v1/prompt/predict") {
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
    if request_line.starts_with("POST /v1/prompt/prewarm") {
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
        let (prewarmed, cached) = prewarm_prompt(&mut context.inference, &turns, reserve_tokens)?;
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
    if !request_line.starts_with("POST /v1/chat/completions") {
        return respond(&mut stream, 404, &json!({"error": "unknown route"}));
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
