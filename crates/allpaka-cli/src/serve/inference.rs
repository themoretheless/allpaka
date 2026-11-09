//! One generation request end to end: prompt prefill with prefix-cache reuse,
//! the sampling loop, and incremental or final delivery of the completion.

use super::*;

/// Shared prefix/SSM session alignment, chunked `forward_batch` prefill, and
/// `PrefixCache` snapshot insertion used by both `prewarm_prompt` and `run_inference`.
pub(super) fn prefill_prompt(
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
pub(super) fn prewarm_prompt(
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

pub(super) fn emit_incremental_sse(
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

pub(super) fn run_inference(
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

pub(super) fn emit_stream_completion(
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
