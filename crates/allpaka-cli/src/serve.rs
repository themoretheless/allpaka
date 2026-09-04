//! An OpenAI-compatible chat endpoint over the engine.
//!
//! `POST /v1/chat/completions` with the usual `{model, messages, max_tokens,
//! temperature}` body; non-streaming, one request at a time. The point is that
//! an agent framework pointed at this URL cannot tell it is not llama-server -
//! same route, same response shape.

mod rag_tools;
mod request_control;
mod ingress;
use request_control::{Interrupted, Registry as RequestRegistry, RequestControl};
use rag_tools::{rag_default_tool_schemas, run_rag_tool, RagTools};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use allpaka_model::prefix_cache::PrefixCache;
use allpaka_model::{Model, Session, Tokenizer};
use crate::serving_runtime::{ServingLimits, ServingRuntime};

/// The KV cache carried across requests.
///
/// A chat regenerates the same prompt prefix on every turn: the whole prior
/// conversation, byte for byte, plus one new user message. Keeping the session
/// and the token list lets each request pay only for what is actually new -
/// without this, turn N re-runs the forward pass over every earlier turn and
/// the chat slows down linearly with its own length.
struct ChatState {
    session: Session,
    tokens: Vec<u32>,
}

/// Byte-budgeted prompt-boundary snapshot shared by requests for one model.
struct PrefixSnap {
    /// Token count of the prompt this state was captured after.
    tokens: usize,
    /// `SsmCache::snapshot()` at the prompt's end for hybrid models.
    ssm_state: Option<Vec<f32>>,
    /// Last-position logits let an exact repeated prompt skip all prefill.
    logits: Vec<f32>,
}

/// Context the persistent session is sized for. f32 KV cache is heavy
/// (~0.4 MB per token on Qwen3-30B), so this is a deliberate budget, not the
/// model's maximum.
const SESSION_TOKENS: usize = 16384;

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

struct ModelService {
    model: Model<'static>,
    tokenizer: Tokenizer,
    force_think: bool,
    chat: ChatState,
    sessions: std::collections::HashMap<String, (ChatState, std::time::Instant)>,
}

struct QueuedConnection {
    stream: TcpStream,
    request_line: String,
    body: String,
    control: RequestControl,
}

type ServerRuntime = ServingRuntime<Mutex<ModelService>, QueuedConnection, PrefixSnap>;

fn model_prefix_namespace(model: &str) -> u32 {
    model
        .bytes()
        .fold(2_166_136_261u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(16_777_619)
        })
}

fn prefix_key(namespace: u32, tokens: &[u32]) -> Vec<u32> {
    let mut key = Vec::with_capacity(tokens.len() + 1);
    key.push(namespace);
    key.extend_from_slice(tokens);
    key
}

fn exact_prefix_snapshot(
    cache: &mut PrefixCache<PrefixSnap>,
    namespace: u32,
    tokens: &[u32],
) -> Option<Arc<PrefixSnap>> {
    let key = prefix_key(namespace, tokens);
    let hit = cache.longest_prefix(&key)?;
    (hit.matched_tokens == key.len() && hit.value.tokens == tokens.len()).then_some(hit.value)
}

#[cfg(test)]
mod prefix_snapshot_tests {
    use super::{
        exact_prefix_snapshot, model_prefix_namespace, prefix_key, PrefixCache, PrefixSnap,
    };

    #[test]
    fn exact_hits_are_model_namespaced() {
        let mut cache = PrefixCache::new(1024);
        let first = model_prefix_namespace("first");
        let second = model_prefix_namespace("second");
        cache.insert(
            prefix_key(first, &[1, 2, 3]),
            PrefixSnap {
                tokens: 3,
                ssm_state: None,
                logits: vec![4.0],
            },
            4,
        );

        assert_eq!(
            exact_prefix_snapshot(&mut cache, first, &[1, 2, 3])
                .unwrap()
                .logits,
            vec![4.0]
        );
        assert!(exact_prefix_snapshot(&mut cache, first, &[1, 2]).is_none());
        assert!(exact_prefix_snapshot(&mut cache, second, &[1, 2, 3]).is_none());
    }
}

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
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    parse_tools: bool,
    stream: Option<&'a mut TcpStream>,
    emit_done: bool,
}

struct ServeRequestContext<'a> {
    model_name: &'a str,
    inference: InferenceContext<'a>,
    rag: &'a RagTools,
}

fn run_inference(
    context: &mut InferenceContext<'_>,
    request: GenerationRequest<'_>,
) -> Result<CompletionResult> {
    context.control.check()?;
    let model = context.model;
    let tok = context.tok;
    let template = context.template;
    let chat = &mut *context.chat;
    let prefixes = &mut *context.prefixes;
    let prefix_namespace = context.prefix_namespace;
    let GenerationRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        repeat_penalty,
        parse_tools,
        mut stream,
        emit_done,
    } = request;
    let prompt = template.prompt(tok, messages)?;
    anyhow::ensure!(max_tokens <= SESSION_TOKENS && prompt.len().saturating_add(max_tokens).saturating_add(1) <= SESSION_TOKENS,
        "request exceeds the 16384-token session limit");
    let stops = template.stop_tokens(tok);
    let mut sampler = Sampler::new(temperature, top_p, repeat_penalty);
    // Greedy without a repeat penalty can take the on-GPU argmax: one word
    // read back per token instead of the whole vocabulary.
    let greedy = temperature <= 0.0 && repeat_penalty == 1.0;

    let mut common = chat
        .tokens
        .iter()
        .zip(&prompt)
        .take_while(|(a, b)| **a == **b)
        .count();
    let mut logits: Vec<f32> = Vec::new();
    let mut skip_prefill = false;
    if prompt.len() + max_tokens + 1 > chat.session.capacity() {
        chat.session = model.new_session(prompt.len() + max_tokens + 1);
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
                        // The prompt is fully covered: its last logits are
                        // in the snapshot, no prefill at all.
                        logits = snap.logits.clone();
                        skip_prefill = true;
                    }
                }
                None => {
                    chat.session =
                        model.new_session(prompt.len() + max_tokens + 1);
                    chat.tokens.clear();
                    common = 0;
                }
            }
        } else if common == prompt.len() {
            // History IS the prompt: the state already sits at its end and
            // only the last logits are missing (KV-only models replay one
            // token here; the SSM cannot step back). The snapshot has them.
            match exact_prefix_snapshot(prefixes, prefix_namespace, &prompt) {
                Some(snap) => {
                    logits = snap.logits.clone();
                    skip_prefill = true;
                }
                None => {
                    chat.session =
                        model.new_session(prompt.len() + max_tokens + 1);
                    chat.tokens.clear();
                    common = 0;
                }
            }
        }
        // else: common == chat.tokens.len() < prompt.len() - the state is
        // already exactly where the prefix ends; nothing to restore.
    } else {
        if common == prompt.len() {
            if let Some(snap) = exact_prefix_snapshot(prefixes, prefix_namespace, &prompt) {
                logits = snap.logits.clone();
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
    if !skip_prefill {
        for chunk in prompt[common..].chunks(prefill_chunk) {
            context.control.check()?;
            logits = model.forward_batch(chunk, &mut chat.session)?;
            chat.tokens.extend_from_slice(chunk);
        }
    }
    let prefill_secs = t0.elapsed().as_secs_f64();

    // Snapshot before generation, when state is exactly "prompt consumed".
    if !logits.is_empty() {
        let ssm_state = chat.session.ssm.as_ref().map(|ssm| ssm.snapshot());
        let value_bytes = logits.len() * size_of::<f32>()
            + ssm_state
                .as_ref()
                .map_or(0, |state| state.len() * size_of::<f32>());
        prefixes.insert(
            prefix_key(prefix_namespace, &prompt),
            PrefixSnap {
                tokens: prompt.len(),
                ssm_state,
                logits: logits.clone(),
            },
            value_bytes,
        );
    }

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
    let mut generated = Vec::new();
    let mut emitted = String::new();
    let mut finish = "length";
    let mut greedy_next: Option<u32> = None;
    for _ in 0..max_tokens {
        context.control.check()?;
        let next = match greedy_next.take() {
            Some(t) => t,
            None => sampler.pick(&logits, &chat.tokens),
        };
        if stops.contains(&next) {
            finish = "stop";
            break;
        }
        generated.push(next);
        if streaming {
            let full = tok.decode(&generated);
            let complete = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
            let complete = if parse_tools {
                &complete[..stream_safe_len(complete)]
            } else {
                complete
            };
            if let Some(delta) = complete.strip_prefix(emitted.as_str()) {
                if !delta.is_empty() {
                    let chunk = json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta}}],
                        "tokens_generated": generated.len(),
                    });
                    write_sse_event(stream.as_mut().unwrap(), &chunk)?;
                    emitted = complete.to_string();
                }
            }
        }
        if greedy {
            greedy_next = Some(model.forward_greedy(next, &mut chat.session)?);
        } else {
            logits = model.forward(next, &mut chat.session)?;
        }
        chat.tokens.push(next);
    }
    let decode_secs = t1.elapsed().as_secs_f64();

    let text = format!("{think_prefix}{}", tok.decode(&generated));
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

    if streaming {
        // `emitted` tracks the body only; the think prefix went out first.
        if let Some(delta) = content[think_prefix.len()..].strip_prefix(emitted.as_str()) {
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

/// The chat template families this server can format. Decided by which
/// special tokens the vocabulary actually contains.
enum Template {
    /// `<|im_start|>role\n...<|im_end|>\n` - Qwen and much of the ecosystem.
    /// `force_think` mirrors the GGUF chat template of the thinking Qwen
    /// generations: the assistant turn is primed with a literal `<think>\n`
    /// (llama.cpp renders the same prefix from the jinja template).
    ChatMl {
        im_start: u32,
        im_end: u32,
        think: Option<u32>,
        force_think: bool,
    },
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>` - Llama 3.
    Llama3 {
        bos: u32,
        start_header: u32,
        end_header: u32,
        eot: u32,
    },
    /// `[gMASK]<sop><|user|>\n...<|assistant|>\n` - GLM-4.x (ChatGLM).
    Glm {
        gmask: Option<u32>,
        sop: Option<u32>,
        system: u32,
        user: u32,
        assistant: u32,
    },
}

impl Template {
    fn detect(tok: &Tokenizer) -> Result<Self> {
        if let (Some(im_start), Some(im_end)) =
            (tok.piece_id("<|im_start|>"), tok.piece_id("<|im_end|>"))
        {
            return Ok(Template::ChatMl {
                im_start,
                im_end,
                think: tok.piece_id("<think>"),
                force_think: false,
            });
        }
        if let (Some(start_header), Some(end_header), Some(eot)) = (
            tok.piece_id("<|start_header_id|>"),
            tok.piece_id("<|end_header_id|>"),
            tok.piece_id("<|eot_id|>"),
        ) {
            return Ok(Template::Llama3 {
                bos: tok.bos.context("llama3 template needs a BOS token")?,
                start_header,
                end_header,
                eot,
            });
        }
        if let (Some(system), Some(user), Some(assistant)) = (
            tok.piece_id("<|system|>"),
            tok.piece_id("<|user|>"),
            tok.piece_id("<|assistant|>"),
        ) {
            return Ok(Template::Glm {
                gmask: tok.piece_id("[gMASK]"),
                sop: tok.piece_id("<sop>"),
                system,
                user,
                assistant,
            });
        }
        bail!("the vocabulary has neither ChatML nor Llama 3 nor GLM special tokens");
    }

    /// The literal prefix the assistant turn was primed with (`<think>\n`
    /// for the thinking Qwens, empty otherwise). The reply text gets it back
    /// so clients see the complete `<think>...</think>` block.
    fn think_prefix(&self) -> &'static str {
        match self {
            Template::ChatMl {
                force_think: true, ..
            } => "<think>\n",
            _ => "",
        }
    }

    /// Format a conversation into token ids, ending where the assistant
    /// starts writing.
    fn prompt(&self, tok: &Tokenizer, messages: &[(String, String)]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        match self {
            Template::ChatMl {
                im_start,
                im_end,
                think,
                force_think,
            } => {
                for (role, content) in messages {
                    ids.push(*im_start);
                    ids.extend(tok.encode(&format!("{role}\n{content}"))?);
                    ids.push(*im_end);
                    ids.extend(tok.encode("\n")?);
                }
                ids.push(*im_start);
                ids.extend(tok.encode("assistant\n")?);
                if *force_think {
                    // The special token by id, exactly as the jinja template
                    // renders it - plain BPE would split the literal text.
                    match think {
                        Some(t) => {
                            ids.push(*t);
                            ids.extend(tok.encode("\n")?);
                        }
                        None => ids.extend(tok.encode("<think>\n")?),
                    }
                }
            }
            Template::Llama3 {
                bos,
                start_header,
                end_header,
                eot,
            } => {
                ids.push(*bos);
                for (role, content) in messages {
                    ids.push(*start_header);
                    ids.extend(tok.encode(role)?);
                    ids.push(*end_header);
                    ids.extend(tok.encode(&format!("\n\n{content}"))?);
                    ids.push(*eot);
                }
                ids.push(*start_header);
                ids.extend(tok.encode("assistant")?);
                ids.push(*end_header);
                ids.extend(tok.encode("\n\n")?);
            }
            Template::Glm {
                gmask,
                sop,
                system,
                user,
                assistant,
            } => {
                if let Some(g) = gmask {
                    ids.push(*g);
                }
                if let Some(s) = sop {
                    ids.push(*s);
                }
                for (role, content) in messages {
                    let role_id = match role.as_str() {
                        "system" => *system,
                        "assistant" => *assistant,
                        _ => *user,
                    };
                    ids.push(role_id);
                    ids.extend(tok.encode(&format!("\n{content}"))?);
                }
                ids.push(*assistant);
                ids.extend(tok.encode("\n")?);
            }
        }
        Ok(ids)
    }

    /// Tokens that end an assistant turn.
    fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = Vec::new();
        match self {
            Template::ChatMl { im_end, .. } => stops.push(*im_end),
            Template::Llama3 { eot, .. } => stops.push(*eot),
            Template::Glm { user, .. } => stops.push(*user),
        }
        if let Some(eos) = tok.eos {
            stops.push(eos);
        }
        stops
    }
}

/// A deliberately small sampler: greedy at temperature 0, otherwise
/// repetition penalty, temperature scaling and nucleus (top-p) truncation.
struct Sampler {
    temperature: f32,
    top_p: f32,
    /// llama.cpp-style repeat penalty over the recent window: positive
    /// logits of recent tokens are divided by it, negative multiplied.
    /// Without it a heavily quantised model happily locks into "либо-либо-
    /// либо" forever; 1.1 breaks such loops without visibly changing prose.
    repeat_penalty: f32,
    state: u64,
}

/// How many most recent tokens the repeat penalty looks at.
const PENALTY_WINDOW: usize = 256;

impl Sampler {
    fn new(temperature: f32, top_p: f32, repeat_penalty: f32) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed)
            | 1;
        Self {
            temperature,
            top_p,
            repeat_penalty,
            state: seed,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Candidates that survive the top-k preselect. Nucleus sampling never
    /// needs a total order of the vocabulary, and a full 152k-element sort
    /// costs ~3 ms per token, serial with the forward pass.
    const TOP_K: usize = 256;

    /// Sample the next token. `recent` is the context so far; its tail feeds
    /// the repeat penalty.
    fn pick(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        // Penalise recent tokens once each (not per occurrence), before any
        // selection, so a penalised token can fall out of the candidate set.
        let mut adjusted;
        let logits: &[f32] = if self.repeat_penalty != 1.0 && !recent.is_empty() {
            adjusted = logits.to_vec();
            let start = recent.len().saturating_sub(PENALTY_WINDOW);
            let mut window: Vec<u32> = recent[start..].to_vec();
            window.sort_unstable();
            window.dedup();
            for t in window {
                if let Some(v) = adjusted.get_mut(t as usize) {
                    if *v > 0.0 {
                        *v /= self.repeat_penalty;
                    } else {
                        *v *= self.repeat_penalty;
                    }
                }
            }
            &adjusted
        } else {
            logits
        };

        if self.temperature <= 0.0 {
            return (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap_or(0) as u32;
        }
        // O(n) partition to the top candidates, then softmax and nucleus over
        // just those. The probability mass beyond the top 256 tokens is far
        // below any top_p in practical use.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        let kth = Self::TOP_K.min(order.len()) - 1;
        order.select_nth_unstable_by(kth, |&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(kth + 1);
        order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        let mut scaled: Vec<f32> = order
            .iter()
            .map(|&i| logits[i] / self.temperature)
            .collect();
        allpaka_backend::softmax(&mut scaled);

        // Nucleus: keep the smallest prefix whose mass reaches top_p.
        let mut kept = 0usize;
        let mut mass = 0f32;
        for &p in &scaled {
            kept += 1;
            mass += p;
            if mass >= self.top_p {
                break;
            }
        }
        let mut r = self.next_f32() * mass;
        for (&i, &p) in order[..kept].iter().zip(&scaled) {
            r -= p;
            if r <= 0.0 {
                return i as u32;
            }
        }
        order[kept - 1] as u32
    }
}

/// Convert an OpenAI-shape message list (+ optional `tools`) into plain
/// (role, content) turns in the Hermes convention Qwen3 was trained on:
/// tool schemas inside `<tools>...</tools>` in the system turn, assistant
/// calls as `<tool_call>{json}</tool_call>` blocks, tool results as user
/// turns wrapped in `<tool_response>...</tool_response>` (consecutive
/// results merged into one turn, as the reference chat template does).
fn render_messages(raw: &[Value], tools: Option<&[Value]>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut pending_tool: Vec<String> = Vec::new();
    let flush_tools = |out: &mut Vec<(String, String)>, pending: &mut Vec<String>| {
        if !pending.is_empty() {
            let body = pending
                .iter()
                .map(|c| format!("<tool_response>\n{c}\n</tool_response>"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(("user".into(), body));
            pending.clear();
        }
    };
    for m in raw {
        let role = m["role"].as_str().unwrap_or("user");
        let content = m["content"].as_str().unwrap_or_default().to_string();
        if role == "tool" {
            pending_tool.push(content);
            continue;
        }
        flush_tools(&mut out, &mut pending_tool);
        if role == "assistant" {
            let mut text = content;
            if let Some(calls) = m["tool_calls"].as_array() {
                for c in calls {
                    let name = c["function"]["name"].as_str().unwrap_or_default();
                    // Arguments arrive as a JSON-encoded string per the
                    // OpenAI shape; embed them as the object Hermes expects.
                    let args = c["function"]["arguments"].as_str().unwrap_or("{}");
                    text.push_str(&format!(
                        "\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                    ));
                }
            }
            out.push(("assistant".into(), text));
        } else {
            out.push((role.to_string(), content));
        }
    }
    flush_tools(&mut out, &mut pending_tool);

    if let Some(tools) = tools {
        let schemas = tools
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let section = format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>\n{schemas}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
        );
        match out.iter_mut().find(|(role, _)| role == "system") {
            Some((_, content)) => content.push_str(&section),
            None => out.insert(
                0,
                (
                    "system".into(),
                    format!("You are a helpful assistant.{section}"),
                ),
            ),
        }
    }
    out
}

/// Split generated text into visible content and Hermes tool-call blocks,
/// returned in the OpenAI `tool_calls` shape (arguments re-encoded as a JSON
/// string). Malformed blocks stay in the content untouched - the client sees
/// what the model actually wrote instead of a silent drop.
fn parse_tool_calls(text: &str) -> (String, Vec<Value>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        let inner = after[..end].trim();
        let parsed: Option<Value> = serde_json::from_str(inner).ok();
        let valid = parsed
            .as_ref()
            .is_some_and(|v| v["name"].is_string() && !v["arguments"].is_null());
        if valid {
            let v = parsed.unwrap();
            calls.push(json!({
                "id": format!("call_{}", calls.len()),
                "type": "function",
                "function": {
                    "name": v["name"],
                    "arguments": v["arguments"].to_string(),
                },
            }));
            content.push_str(&rest[..start]);
        } else {
            content.push_str(&rest[..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// The longest prefix of `text` that is safe to stream when tool calls may
/// follow: everything up to the first (possibly still incomplete) opening
/// `<tool_call>` tag. A trailing partial match of the tag is held back too.
fn stream_safe_len(text: &str) -> usize {
    const TAG: &str = "<tool_call>";
    if let Some(i) = text.find(TAG) {
        return i;
    }
    let bytes = text.as_bytes();
    let max_tail = (TAG.len() - 1).min(bytes.len());
    for tail in (1..=max_tail).rev() {
        if text.is_char_boundary(bytes.len() - tail)
            && TAG.as_bytes().starts_with(&bytes[bytes.len() - tail..])
        {
            return bytes.len() - tail;
        }
    }
    bytes.len()
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
        total.checked_add(*bytes).context("resident model byte total overflow")
    })?;
    anyhow::ensure!(
        total_bytes <= limits.model_budget_bytes,
        "models require {:.1} GiB but --model-budget-gib allows {:.1} GiB",
        total_bytes as f64 / (1u64 << 30) as f64,
        limits.model_budget_bytes as f64 / (1u64 << 30) as f64,
    );

    let mut runtime = ServerRuntime::new(limits);
    let mut default_model = String::new();
    for (path, bytes) in model_paths.iter().zip(planned_bytes) {
        let (name, service) = load_model_service(path)?;
        if default_model.is_empty() {
            default_model.clone_from(&name);
        }
        runtime
            .install_model(name.clone(), Mutex::new(service), bytes)
            .map_err(|error| anyhow::anyhow!("installing model {name}: {error:?}"))?;
    }
    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
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
                respond(&mut rejection_stream, 429, &json!({"error":format!("scheduler: {error:?}")}))?;
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

fn load_model_service(model_path: &Path) -> Result<(String, ModelService)> {
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
    let chat = ChatState {
        session: model.new_session(1),
        tokens: Vec::new(),
    };
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

fn prepare_connection(
    mut stream: TcpStream,
    default_model: &str,
    max_context_tokens: usize,
    registry: &Arc<RequestRegistry>,
) -> Result<Option<(String, usize, QueuedConnection)>> {
    let (request_line, body) = read_request(&mut stream)?;
    if request_line.starts_with("GET /health ") {
        respond(&mut stream, 200, &json!({"status":"ok"}))?;
        return Ok(None);
    }
    if let Some(route) = request_line.strip_prefix("POST /v1/requests/") {
        if let Some(id) = route.strip_suffix("/cancel HTTP/1.1") {
            let cancelled = registry.cancel(id);
            respond(&mut stream, if cancelled { 200 } else { 404 }, &json!({"cancelled":cancelled}))?;
            return Ok(None);
        }
    }
    let request: Value = if body.is_empty() { json!({}) } else { serde_json::from_str(&body)? };
    if let Some(id) = request.get("session_id") {
        anyhow::ensure!(id.as_str().is_some_and(|id| !id.is_empty() && id.len() <= 128), "invalid session_id");
    }
    if let Some(id) = request.get("request_id") {
        anyhow::ensure!(id.is_string(), "request_id must be a string");
    }
    if let Some(tokens) = request.get("max_tokens") {
        anyhow::ensure!(tokens.as_u64().is_some_and(|n| n > 0 && n < SESSION_TOKENS as u64), "max_tokens must be in 1..16384");
    }
    let timeout_ms = match request.get("timeout_ms") {
        None => 120_000,
        Some(value) => value.as_u64().filter(|n| (1..=600_000).contains(n))
            .context("timeout_ms must be in 1..=600000")?,
    };
    let control = registry.register(request["request_id"].as_str(), std::time::Duration::from_millis(timeout_ms))?;
    let requested = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|request| request["model"].as_str().map(ToOwned::to_owned))
        .filter(|model| model != "allpaka")
        .unwrap_or_else(|| default_model.to_string());
    let context_tokens = body.len().div_ceil(4).clamp(1, max_context_tokens.max(1));
    Ok(Some((
        requested,
        context_tokens,
        QueuedConnection {
            stream,
            request_line,
            body,
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

        assert_eq!(resolve_model_name("qwen3", &names).as_deref(), Some("qwen3"));
        assert_eq!(resolve_model_name("qwen3-0", &names).as_deref(), Some("qwen3-0.6b"));
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
    let is_chat = connection.request_line.starts_with("POST /v1/chat/completions");
    let request: Value = serde_json::from_str(&connection.body).unwrap_or(Value::Null);
    let session_id = request.get("session_id").map(|value| {
        value.as_str().filter(|id| !id.is_empty() && id.len() <= 128)
            .map(str::to_owned).context("session_id must be a non-empty string of at most 128 bytes")
    }).transpose()?;
    let now = std::time::Instant::now();
    service.sessions.retain(|_, (_, used)| now.duration_since(*used).as_secs() < 600);
    if is_chat {
        if session_id.as_ref().is_some_and(|id| !service.sessions.contains_key(id)) && service.sessions.len() >= 4 {
            let mut stream = connection.stream;
            return respond(&mut stream, 429, &json!({"error":"resident session capacity exhausted"}));
        }
        let restored = session_id.as_ref().and_then(|id| service.sessions.remove(id));
        service.chat = restored.map(|(chat, _)| chat).unwrap_or_else(|| ChatState {
            session: service.model.new_session(256), tokens: Vec::new(),
        });
    }
    let ModelService {
        model,
        tokenizer,
        force_think,
        chat,
        ..
    } = &mut *service;
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
        connection.body,
        ServeRequestContext {
            model_name: &model_name,
            inference: InferenceContext {
                model,
                tok: tokenizer,
                template: &template,
                chat,
                prefixes,
                prefix_namespace,
                control: &connection.control,
            },
            rag,
        },
    );
    let succeeded = outcome.is_ok();
    if let Err(error) = outcome {
        let status = if error.downcast_ref::<Interrupted>().is_some() { 408 } else { 500 };
        let body = json!({"error":error.to_string(), "request_id":connection.control.id});
        if connection.control.streaming() {
            write_sse_event(&mut error_stream, &body)?;
            error_stream.write_all(b"data: [DONE]\n\n")?;
        } else {
            respond(&mut error_stream, status, &body)?;
        }
    }
    if is_chat {
        let empty = ChatState { session: service.model.new_session(1), tokens: Vec::new() };
        let chat = std::mem::replace(&mut service.chat, empty);
        if succeeded {
            if let Some(id) = session_id { service.sessions.insert(id, (chat, std::time::Instant::now())); }
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
    let t0 = std::time::Instant::now();
    let mut acc = 0u8;
    let mut i = 0;
    while i < mapping.len() {
        acc ^= mapping[i];
        i += PAGE;
    }
    std::hint::black_box(acc);
    println!(
        "prewarmed {:.1} GiB of weights in {:.1}s",
        mapping.len() as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
}

fn handle(
    mut stream: TcpStream,
    request_line: String,
    body: String,
    mut context: ServeRequestContext<'_>,
) -> Result<()> {
    let model_name = context.model_name;
    let rag = context.rag;
    if request_line.starts_with("GET /health") {
        return respond(&mut stream, 200, &json!({"status": "ok"}));
    }
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
    if !request_line.starts_with("POST /v1/chat/completions") {
        return respond(&mut stream, 404, &json!({"error": "unknown route"}));
    }

    let req: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return respond(
                &mut stream,
                400,
                &json!({"error": format!("bad JSON: {e}")}),
            )
        }
    };
    let Some(raw_messages) = req["messages"].as_array() else {
        return respond(&mut stream, 400, &json!({"error": "messages[] required"}));
    };
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
    let temperature = req["temperature"].as_f64().unwrap_or(0.7) as f32;
    let top_p = req["top_p"].as_f64().unwrap_or(0.95) as f32;
    let repeat_penalty = req["repeat_penalty"].as_f64().unwrap_or(1.1) as f32;
    let streaming = req["stream"].as_bool().unwrap_or(false);

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

    if streaming && !tools.is_none() && rag.cfg.enabled {
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

fn read_request(stream: &mut TcpStream) -> Result<(String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-request");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = find_header_end(&buf) {
            header_end = i;
            break;
        }
        if buf.len() > 1 << 20 {
            bail!("request headers over 1 MiB");
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let request_line = head.lines().next().unwrap_or_default().to_string();
    let content_length: usize = head
        .lines()
        .find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        })
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);
    anyhow::ensure!(content_length <= 4 * 1024 * 1024, "request body exceeds 4 MiB");

    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("connection closed mid-body");
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok((request_line, String::from_utf8_lossy(&body).into_owned()))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn respond(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let text = body.to_string();
    let reason = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
          Connection: close\r\n\r\n",
    )?;
    Ok(())
}

fn write_sse_event(stream: &mut TcpStream, body: &Value) -> Result<()> {
    stream.write_all(format!("data: {body}\n\n").as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn respond_html(stream: &mut TcpStream, page: &str) -> Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_section_lands_in_the_system_turn() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"type":"function","function":{"name":"search","parameters":{}}})];
        let out = render_messages(&msgs, Some(&tools));
        assert_eq!(out[0].0, "system");
        assert!(out[0].1.contains("<tools>"));
        assert!(out[0].1.contains("\"search\""));
        assert_eq!(out[1], ("user".into(), "hi".into()));
        // An existing system message is extended, not duplicated.
        let msgs2 = vec![
            json!({"role":"system","content":"be brief"}),
            msgs[0].clone(),
        ];
        let out2 = render_messages(&msgs2, Some(&tools));
        assert_eq!(out2.len(), 2);
        assert!(out2[0].1.starts_with("be brief"));
        assert!(out2[0].1.contains("<tools>"));
    }

    #[test]
    fn assistant_calls_and_tool_results_round_trip_into_hermes_text() {
        let msgs = vec![
            json!({"role":"user","content":"weather?"}),
            json!({"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]}),
            json!({"role":"tool","tool_call_id":"call_0","content":"{\"temp\":-3}"}),
            json!({"role":"tool","tool_call_id":"call_1","content":"{\"wind\":5}"}),
        ];
        let out = render_messages(&msgs, None);
        assert!(out[1].1.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Oslo\"}}\n</tool_call>"));
        // Both tool results merge into ONE user turn, each in its own wrapper.
        assert_eq!(out[2].0, "user");
        assert_eq!(out[2].1.matches("<tool_response>").count(), 2);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn tool_call_blocks_parse_and_malformed_ones_stay_in_content() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"rag_search\", \"arguments\": {\"q\": \"kv cache\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "rag_search");
        assert_eq!(calls[0]["function"]["arguments"], "{\"q\":\"kv cache\"}");
        // Broken JSON is not silently dropped.
        let (content2, calls2) = parse_tool_calls("<tool_call>not json</tool_call>done");
        assert!(calls2.is_empty());
        assert_eq!(content2, "<tool_call>not json</tool_call>done");
    }

    #[test]
    fn stream_hold_back_covers_partial_opening_tags() {
        assert_eq!(stream_safe_len("hello"), 5);
        assert_eq!(stream_safe_len("hello <tool_call>{"), 6);
        assert_eq!(stream_safe_len("hello <tool_c"), 6);
        assert_eq!(stream_safe_len("<"), 0);
        // A lone '<' that cannot start the tag is NOT held back.
        assert_eq!(stream_safe_len("a < b"), 5);
    }

    #[test]
    fn greedy_pick_is_argmax() {
        let mut s = Sampler::new(0.0, 0.95, 1.0);
        let mut logits = vec![0.0f32; 1000];
        logits[421] = 3.5;
        assert_eq!(s.pick(&logits, &[]), 421);
    }

    /// A dominant logit must win essentially always at moderate temperature,
    /// and the preselect must never return an index outside the vocabulary.
    #[test]
    fn sampled_pick_follows_the_dominant_token() {
        let mut s = Sampler::new(0.7, 0.95, 1.0);
        let mut logits = vec![0.0f32; 152_000];
        logits[123] = 20.0;
        for _ in 0..50 {
            assert_eq!(s.pick(&logits, &[]), 123);
        }
    }

    /// With a flat-ish distribution the sampler must return varied but valid
    /// tokens from the top candidates.
    #[test]
    fn sampled_pick_stays_within_the_top_candidates() {
        let mut s = Sampler::new(1.0, 0.9, 1.0);
        // 512 tokens clearly above the rest; everything sampled must be one.
        let mut logits = vec![-10.0f32; 152_000];
        for v in logits.iter_mut().take(512) {
            *v = 1.0;
        }
        for _ in 0..100 {
            let t = s.pick(&logits, &[]) as usize;
            assert!(t < 512, "picked {t} from outside the plateau");
        }
    }

    /// The repeat penalty must break a tie in favour of the token that has
    /// NOT just been emitted - the "либо-либо-либо" loop breaker.
    #[test]
    fn repeat_penalty_demotes_a_recently_emitted_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.1);
        let mut logits = vec![0.0f32; 1000];
        logits[7] = 3.0; // the looping token
        logits[8] = 2.9; // the almost-as-good alternative
                         // Without history the loop token wins; with itself in the window it
                         // is divided down below the alternative.
        assert_eq!(s.pick(&logits, &[]), 7);
        assert_eq!(s.pick(&logits, &[7]), 8);
        // Negative logits move away from zero instead.
        let mut neg = vec![-1.0f32; 10];
        neg[3] = -0.5;
        neg[4] = -0.52;
        assert_eq!(s.pick(&neg, &[3]), 4);
    }

    /// One occurrence or ten in the window, the penalty applies once: the
    /// classic repeat penalty is presence-based, not frequency-based.
    #[test]
    fn repeat_penalty_applies_once_per_distinct_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.3);
        let mut logits = vec![0.0f32; 100];
        logits[5] = 2.0;
        logits[6] = 1.6;
        // 2.0 / 1.3 = 1.538 < 1.6, so one mention already flips the choice;
        // many mentions must not penalise 6 as well by compounding wrongly.
        assert_eq!(s.pick(&logits, &[5, 5, 5, 5, 5]), 6);
    }
}

/// The built-in chat page: zero dependencies, talks to this same server.
const CHAT_PAGE: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>allpaka chat</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #log { display: flex; flex-direction: column; gap: .6rem; margin-bottom: 1rem; }
  .msg { padding: .6rem .9rem; border-radius: .7rem; white-space: pre-wrap; line-height: 1.4; }
  .user { background: #3b82f6; color: white; align-self: flex-end; max-width: 85%; }
  .bot  { background: rgba(127,127,127,.15); align-self: flex-start; max-width: 85%; }
  .meta { opacity: .5; font-size: .75rem; align-self: flex-start; }
  .thinking { opacity: .55; font-style: italic; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .6rem .8rem; border-radius: .6rem; border: 1px solid rgba(127,127,127,.4); font-size: 1rem; }
  button { padding: .6rem 1.1rem; border-radius: .6rem; border: none; background: #3b82f6; color: white; font-size: 1rem; }
  button:disabled { opacity: .5; }
  h1 { font-size: 1.1rem; opacity: .7; }
  details { opacity: .6; font-size: .85rem; }
  #stats { display: flex; align-items: center; gap: .8rem; font-size: .78rem;
           opacity: .75; margin-bottom: 1rem; flex-wrap: wrap; }
  #ctxbar { width: 160px; height: 6px; border-radius: 3px;
            background: rgba(127,127,127,.25); overflow: hidden; }
  #ctxfill { height: 100%; width: 0%; background: #3b82f6; transition: width .3s; }
</style></head><body>
<h1>allpaka · собственный движок · чат</h1>
<div id="stats">
  <span id="s-model">…</span>
  <span>контекст: <span id="s-ctx">0 / 0</span></span>
  <div id="ctxbar"><div id="ctxfill"></div></div>
  <span id="s-speed"></span>
</div>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Спросите что-нибудь..." autofocus autocomplete="off">
<button id="send">→</button></form>
<script>
const log = document.getElementById('log');
const history = [];
function setStats(used, cap, speed) {
  document.getElementById('s-ctx').textContent =
    `${used.toLocaleString('ru')} / ${cap.toLocaleString('ru')} (${Math.round(used/cap*100)}%)`;
  document.getElementById('ctxfill').style.width = Math.min(100, used/cap*100) + '%';
  if (speed) document.getElementById('s-speed').textContent = speed;
}
fetch('/stats').then(r => r.json()).then(s => {
  document.getElementById('s-model').textContent =
    s.model + (s.moe ? ' (MoE)' : '') + ' · ' + s.n_layers + ' слоёв';
  setStats(s.context_used, s.context_capacity, '');
}).catch(() => {});
function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  d.scrollIntoView();
  return d;
}
let aborter = null;
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q');
  const btn = document.getElementById('send');
  if (aborter) { aborter.abort(); return; } // the button doubles as "stop"
  const text = q.value.trim();
  if (!text) return;
  q.value = '';
  aborter = new AbortController();
  btn.textContent = '⏹';
  add('user', text);
  history.push({role: 'user', content: text});
  const busy = add('bot', '…');
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = 'думаю…';
  log.appendChild(meta);
  const t0 = performance.now();
  let raw = '', tokens = 0, timing = null, stopped = false;
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: aborter.signal,
      body: JSON.stringify({messages: history, max_tokens: 1024, temperature: 0.7, stream: true})
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const events = buf.split('\n\n');
      buf = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        const j = JSON.parse(payload);
        const delta = j.choices && j.choices[0].delta && j.choices[0].delta.content;
        if (delta) raw += delta;
        if (j.tokens_generated) tokens = j.tokens_generated;
        if (j.usage) tokens = j.usage.completion_tokens;
        if (j.timing) timing = j.timing;
        // Render: while the model is inside its think block, stream the
        // thoughts dimmed; once the answer starts, show the answer.
        let reply = raw;
        const m = reply.match(/^<think>([\s\S]*?)(<\/think>\s*|$)/);
        let think = null;
        if (m) { think = m[1].trim(); reply = reply.slice(m[0].length); }
        if (reply) {
          busy.classList.remove('thinking');
          busy.textContent = reply;
        } else if (think) {
          busy.classList.add('thinking');
          busy.textContent = '💭 ' + (think.length > 600 ? '…' + think.slice(-600) : think);
        } else {
          busy.textContent = '…';
        }
        const secs = (performance.now() - t0) / 1000;
        const tps = tokens > 0 ? (tokens / secs).toFixed(1) : '?';
        meta.textContent = `${tokens} токенов · ${secs.toFixed(1)} с · ${tps} tok/s`;
        meta.scrollIntoView();
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      stopped = true;
    } else {
      busy.textContent = 'ошибка: ' + err;
      aborter = null;
      btn.textContent = '→';
      q.focus();
      return;
    }
  }
  // Final render: full reply, or whatever arrived before the stop.
  let reply = raw;
  const m = reply.match(/^<think>([\s\S]*?)(<\/think>\s*|$)/);
  let think = null;
  if (m) { think = m[1].trim(); reply = reply.slice(m[0].length); }
  busy.classList.remove('thinking');
  busy.textContent = reply || (stopped ? '(остановлено на размышлениях)' : '(пусто)');
  if (think) {
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = 'размышления';
    det.appendChild(sum);
    const pre = document.createElement('div');
    pre.textContent = think;
    det.appendChild(pre);
    busy.before(det);
  }
  // Partial replies stay in the history: the server cache holds those tokens
  // too, so the next turn still reuses the prefix.
  history.push({role: 'assistant', content: reply});
  if (timing) {
    meta.textContent = `${tokens} токенов · генерация ${timing.decode_secs.toFixed(1)} с · `
      + `${timing.tokens_per_sec.toFixed(1)} tok/s · промпт ${timing.prefill_secs.toFixed(1)} с`
      + (timing.cached_tokens ? ` (кэш: ${timing.cached_tokens} ток)` : '');
    if (timing.context_used) {
      setStats(timing.context_used, timing.context_capacity,
        timing.tokens_per_sec.toFixed(1) + ' tok/s');
    }
  } else if (stopped) {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    meta.textContent = `${tokens} токенов · ${secs} с · остановлено`;
  }
  aborter = null;
  btn.textContent = '→';
  q.focus();
});
</script></body></html>"#;
