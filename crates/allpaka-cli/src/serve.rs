//! An OpenAI-compatible chat endpoint over the engine.
//!
//! `POST /v1/chat/completions` with the usual `{model, messages, max_tokens,
//! temperature}` body; non-streaming, one request at a time. The point is that
//! an agent framework pointed at this URL cannot tell it is not llama-server -
//! same route, same response shape.

mod handlers;
mod http;
mod inference;
mod ingress;
mod interleave;
mod launch;
mod memory;
mod messages;
mod predict;
mod preemptible;
mod rag_tools;
mod request_control;
mod resources;
#[cfg(test)]
mod route_tests;
mod router;
mod routes;
mod sampler;
mod session;
mod template;
mod tool_grammar;
mod tool_schema;

pub(crate) use http::{read_request, respond, respond_html, write_sse_event, write_sse_headers};
pub(crate) use messages::{
    append_valid_utf8, image_part_count, parse_tool_calls, parse_tool_calls_checked,
    render_messages, stream_safe_len,
};
pub(crate) use predict::{
    extract_prediction_text, parse_prompt_lookup_k, predict_followup_prompts,
    predict_prompt_continuation,
};
pub(crate) use sampler::Sampler;
pub(crate) use session::{
    check_in_named, check_out_named, exact_prefix_snapshot, model_prefix_namespace, named_in_use,
    prefix_key, ChatState, ModelService, NamedSessions, PrefixSnap,
};
pub(crate) use template::Template;

use inference::*;
use rag_tools::{rag_default_tool_schemas, run_rag_tool, RagTools};
use request_control::{Interrupted, Registry as RequestRegistry, RequestControl};
use router::*;
use routes::{Matched, Route};
use tool_grammar::{ToolChoice, ToolGrammar};

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
    /// Tool calls the model wrote that the request's tool declarations do not
    /// allow: `{name, reason}` each. Their text stays in `content`.
    tool_rejections: Vec<Value>,
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
    /// Called between steps (a prefill chunk, a decode token) of this
    /// generation; may serve a waiting short request in the meantime.
    interrupt: Option<Interrupt<'a>>,
}

/// What a generation hands the step hook: the model it is running on, so a
/// request served in between uses the same weights.
/// The last argument says the call comes from the decode loop, whose steps
/// take the turn through `interleave::decode_step` instead.
type Interrupt<'a> =
    &'a dyn Fn(&Model<'static>, &Tokenizer, &Template, &memory::Budget, usize, bool);

struct GenerationRequest<'a> {
    messages: &'a [(String, String)],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f64,
    parse_tools: bool,
    /// The request's tool declarations; parsed calls are checked against them.
    tool_specs: Option<&'a [Value]>,
    /// Whether the reply may, or must, be a tool call.
    tool_choice: tool_grammar::ToolChoice,
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
                // A chat request that arrives while this one generates is
                // served between its steps instead of after it (roadmap A3).
                let pending = std::cell::RefCell::new(Vec::new());
                let serving = scheduled.model.clone();
                // What a worker would need, taken before dispatch locks the service.
                let shared = runtime.model(&serving).map(|service| {
                    let service = service.lock().unwrap();
                    interleave::Shared {
                        name: serving.clone(),
                        model: Arc::clone(&service.model),
                        tokenizer: Arc::clone(&service.tokenizer),
                        force_think: service.force_think,
                        budget: service.chat.memory.budget(),
                        sessions: Arc::clone(&service.sessions),
                    }
                });
                // A worker generation may already run (started during an
                // earlier request): take turns with it from the first step.
                // The gate is this thread's (`interleave::current`): the
                // decode loop batches its steps through it.
                let gate = interleave::active();
                if let Some(gate) = &gate {
                    gate.enter();
                }
                interleave::set_turn(gate);
                let interrupt = |model: &Model<'static>,
                                 tok: &Tokenizer,
                                 template: &Template,
                                 budget: &memory::Budget,
                                 running_tokens: usize,
                                 in_decode: bool| {
                    if !in_decode {
                        interleave::pass_turn();
                    }
                    while let Ok((name, context_tokens, connection)) = incoming.try_recv() {
                        if preemptible::eligible(
                            &name,
                            &serving,
                            &connection,
                            context_tokens,
                            running_tokens,
                        ) {
                            preemptible::serve(
                                &name, connection, model, tok, template, budget, &rag_tools, None,
                                None,
                            );
                        } else if let Some(shared) = shared
                            .as_ref()
                            .filter(|s| interleave::eligible(&name, s, &connection))
                        {
                            interleave::set_turn(Some(interleave::spawn(
                                shared.clone(),
                                connection,
                            )));
                        } else {
                            pending
                                .borrow_mut()
                                .push((name, context_tokens, connection));
                        }
                    }
                };
                if let Err(error) = dispatch_connection(
                    scheduled.model,
                    scheduled.payload,
                    &mut runtime,
                    &rag_tools,
                    Some(&interrupt),
                ) {
                    println!("request failed: {error:#}");
                }
                if let Some(gate) = interleave::current() {
                    gate.leave();
                }
                interleave::set_turn(None);
                for (name, context_tokens, connection) in pending.into_inner() {
                    let mut rejection_stream = connection.stream.try_clone()?;
                    if let Err(error) = runtime.enqueue(name, context_tokens, connection) {
                        respond(
                            &mut rejection_stream,
                            429,
                            &json!({"error":format!("scheduler: {error:?}")}),
                        )?;
                    }
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
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            force_think,
            chat,
            sessions: Arc::default(),
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

/// Answers a model-stage route under the model lock. Routes answered by an
/// earlier stage never get here; any other line, matched or not, gets the
/// "unknown route" 404.
fn handle(
    mut stream: TcpStream,
    route: Option<Route>,
    req: Value,
    mut context: ServeRequestContext<'_>,
) -> Result<()> {
    match route {
        Some(Route::Stats) => handlers::stats(&mut stream, &context),
        Some(Route::ChatPage) => respond_html(&mut stream, CHAT_PAGE),
        Some(Route::Options) => respond(&mut stream, 200, &json!({})),
        Some(Route::PromptPredict) => handlers::prompt_predict(&mut stream, &req, &context),
        Some(Route::PromptPrewarm) => handlers::prompt_prewarm(&mut stream, &req, &mut context),
        Some(Route::ChatCompletions) => handlers::chat_completions(&mut stream, &req, &mut context),
        _ => respond(&mut stream, 404, &json!({"error": "unknown route"})),
    }
}

/// The built-in chat page: zero dependencies, talks to this same server.
/// Lives as a real .html file so editors and grep see it; embedded at compile
/// time, so the served bytes are unchanged.
const CHAT_PAGE: &str = include_str!("../assets/chat.html");
