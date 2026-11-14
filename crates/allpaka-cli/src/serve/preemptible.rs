//! Serving a short chat request between the steps of a long one.
//!
//! Inference has one owner. Without this, a request that arrives while
//! another generates waits for all of it: on Qwen2.5-7B a chat behind an
//! 8 900-token request saw its first token after 8.8 s. Now the running
//! generation calls back after every prefill chunk and decode token; a chat
//! request waiting by then is served to completion in a session of its own,
//! and the long generation resumes. The chat waits for one step, not one
//! request. The long one is delayed by the short one, which is the point.

use super::*;

/// A waiting request is served in between only if it is this many times
/// shorter than the running one: a long request must not stall a short one
/// that happens to be running.
const SHORTER_BY: usize = 4;

/// Whether a waiting request can be served in between: a chat completion for
/// the model already running, with no named session (whose state the running
/// request may be holding), and much less work than the running request.
/// Work is prompt plus requested new tokens; the waiting prompt is estimated
/// from its body size (`context_tokens`), the running one is exact.
pub(super) fn eligible(
    requested: &str,
    serving: &str,
    connection: &QueuedConnection,
    context_tokens: usize,
    running_tokens: usize,
) -> bool {
    let chat = matches!(
        routes::classify(&connection.request_line).map(|m| m.route),
        Some(Route::ChatCompletions)
    );
    let new_tokens = connection.request["max_tokens"].as_u64().unwrap_or(256) as usize;
    let work = context_tokens.saturating_add(new_tokens);
    chat && requested == serving
        && connection.request.get("session_id").is_none()
        && work.saturating_mul(SHORTER_BY) <= running_tokens
}

/// Serve `connection` to completion on `model` in a temporary session that
/// neither reads nor fills the prefix cache, then close it.
pub(super) fn serve(
    model_name: &str,
    connection: QueuedConnection,
    model: &Model<'static>,
    tok: &Tokenizer,
    template: &Template,
    budget: &memory::Budget,
    rag: &RagTools,
) {
    let QueuedConnection {
        stream,
        request,
        control,
        ..
    } = connection;
    let Ok(mut error_stream) = stream.try_clone() else {
        return;
    };
    let mut chat = match ChatState::new(model, budget, 1) {
        Ok(chat) => chat,
        Err(error) => {
            let _ = respond(&mut error_stream, 429, &json!({"error": error.to_string()}));
            router::close_politely(error_stream);
            return;
        }
    };
    let mut no_prefixes = PrefixCache::new(0);
    let outcome = handle(
        stream,
        Some(Route::ChatCompletions),
        request,
        ServeRequestContext {
            model_name,
            inference: InferenceContext {
                model,
                tok,
                template,
                chat: &mut chat,
                prefixes: &mut no_prefixes,
                prefix_namespace: 0,
                control: &control,
                // Served in between; it does not serve others in turn.
                interrupt: None,
            },
            rag,
        },
    );
    if let Err(error) = outcome {
        let status = if error.downcast_ref::<Interrupted>().is_some() {
            408
        } else if error.downcast_ref::<memory::Exhausted>().is_some() {
            429
        } else {
            500
        };
        let body = json!({"error": error.to_string(), "request_id": control.id});
        if control.streaming() {
            let _ = write_sse_event(&mut error_stream, &body);
            let _ = std::io::Write::write_all(&mut error_stream, b"data: [DONE]\n\n");
        } else {
            let _ = respond(&mut error_stream, status, &body);
        }
    }
    router::close_politely(error_stream);
}
