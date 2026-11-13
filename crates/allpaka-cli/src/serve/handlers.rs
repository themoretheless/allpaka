//! Model-stage endpoint handlers: everything answered after the model lock
//! is taken, one function per route.

use super::*;

/// `GET /stats`. Answered after the model lock is taken, so this reflects the
/// live context and queues behind an in-flight generation. `/resources` is
/// served before the queue and reports the context as of the last completed
/// generation -- the two endpoints share `context_used` and
/// `context_capacity` with different freshness, which is why the chat page
/// reads each pair from one of them only.
pub(super) fn stats(stream: &mut TcpStream, context: &ServeRequestContext<'_>) -> Result<()> {
    let model_name = context.model_name;
    respond(
        stream,
        200,
        &json!({
            "model": model_name,
            "architecture": context.inference.model.config.architecture.as_str(),
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
    )
}

/// `POST /v1/prompt/predict`: inline completion and follow-up suggestions for
/// the prompt being typed.
pub(super) fn prompt_predict(
    stream: &mut TcpStream,
    req: &Value,
    context: &ServeRequestContext<'_>,
) -> Result<()> {
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
    respond(
        stream,
        200,
        &json!({
            "completion": completion,
            "suggestions": suggestions,
        }),
    )
}

/// `POST /v1/prompt/prewarm`: prefill the conversation so far, plus the draft,
/// before the user sends it.
pub(super) fn prompt_prewarm(
    stream: &mut TcpStream,
    req: &Value,
    context: &mut ServeRequestContext<'_>,
) -> Result<()> {
    let raw_messages = req["messages"].as_array().map_or(&[][..], Vec::as_slice);
    let draft = req["draft"].as_str().unwrap_or("").trim();
    let reserve_tokens = req["max_tokens"].as_u64().unwrap_or(1024) as usize;
    let mut turns = render_messages(raw_messages, None);
    if !draft.is_empty() {
        turns.push(("user".into(), draft.to_string()));
    }
    if turns.is_empty() {
        return respond(
            stream,
            200,
            &json!({"prewarmed_tokens": 0, "cached_tokens": 0}),
        );
    }
    let (prewarmed, cached) = prewarm_prompt(&mut context.inference, &turns, reserve_tokens)?;
    respond(
        stream,
        200,
        &json!({
            "prewarmed_tokens": prewarmed,
            "cached_tokens": cached,
            "context_used": context.inference.chat.tokens.len(),
        }),
    )
}

/// `POST /v1/chat/completions`, streaming or not, with the optional RAG tool
/// loop.
pub(super) fn chat_completions(
    stream: &mut TcpStream,
    req: &Value,
    context: &mut ServeRequestContext<'_>,
) -> Result<()> {
    let model_name = context.model_name;
    let rag = context.rag;
    let Some(raw_messages) = req["messages"].as_array() else {
        return respond(stream, 400, &json!({"error": "messages[] required"}));
    };
    let images = image_part_count(raw_messages);
    if images > 0 {
        // The engine has no vision projector, so an image could only be
        // ignored. Fail closed and say so: silently dropping the picture (and
        // the text that travelled with it) is worse than an explicit refusal.
        return respond(
            stream,
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
            stream,
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
        // With tools the text is held back from the first `<tool_call>` on and
        // the calls go out parsed and checked, as in the non-streaming reply.
        run_inference(
            &mut context.inference,
            GenerationRequest {
                messages: &messages,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                parse_tools: tools.is_some(),
                tool_specs: tools,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: Some(&mut *stream),
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
                tool_specs: None,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: if streaming { Some(&mut *stream) } else { None },
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
                tool_specs: tools,
                prediction_tokens: &prediction_tokens,
                prompt_lookup_k,
                stream: if streaming { Some(&mut *stream) } else { None },
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
                    tool_specs: tools,
                    prediction_tokens: &prediction_tokens,
                    prompt_lookup_k,
                    stream: if streaming { Some(&mut *stream) } else { None },
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
        emit_stream_completion(stream, &completion, false, &completion.finish)?;
        return Ok(());
    }

    let mut choice = json!({
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
    });
    // Present only when the model wrote a call the request did not allow.
    if !completion.tool_rejections.is_empty() {
        choice["tool_call_rejections"] = Value::Array(completion.tool_rejections);
    }
    respond(
        stream,
        200,
        &json!({
            "id": "chatcmpl-allpaka",
            "object": "chat.completion",
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": completion.cached_tokens},
            },
        }),
    )
}
