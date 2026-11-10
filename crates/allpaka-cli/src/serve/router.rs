//! Connection routing: classify an HTTP request line against the declarative
//! route table, answer lock-free ingress probes immediately, resolve the target
//! model, and dispatch queued requests to their typed endpoint.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum IngressRoute<'a> {
    Health,
    Resources,
    Cancel(&'a str),
    Queued(QueuedRoute),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueuedRoute {
    Catalog,
    Launch,
    Models,
    Model(ModelRoute),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelRoute {
    Stats,
    ChatPage,
    CorsOptions,
    PromptPredict,
    PromptPrewarm,
    ChatCompletions,
    Unknown,
}

impl ModelRoute {
    #[inline]
    pub(super) const fn uses_chat_session(self) -> bool {
        matches!(self, Self::ChatCompletions | Self::PromptPrewarm)
    }
}

struct RouteEntry {
    method: &'static str,
    path: &'static str,
    route: QueuedRoute,
}

/// Declarative table for all post-ingress routes matched by `(method, path)`
/// after stripping any `?query` suffix.
const STATIC_ROUTES: &[RouteEntry] = &[
    RouteEntry {
        method: "GET",
        path: "/v1/catalog",
        route: QueuedRoute::Catalog,
    },
    RouteEntry {
        method: "POST",
        path: "/v1/launch",
        route: QueuedRoute::Launch,
    },
    RouteEntry {
        method: "GET",
        path: "/v1/models",
        route: QueuedRoute::Models,
    },
    RouteEntry {
        method: "GET",
        path: "/stats",
        route: QueuedRoute::Model(ModelRoute::Stats),
    },
    RouteEntry {
        method: "GET",
        path: "/",
        route: QueuedRoute::Model(ModelRoute::ChatPage),
    },
    RouteEntry {
        method: "GET",
        path: "/index",
        route: QueuedRoute::Model(ModelRoute::ChatPage),
    },
    RouteEntry {
        method: "GET",
        path: "/index.html",
        route: QueuedRoute::Model(ModelRoute::ChatPage),
    },
    RouteEntry {
        method: "POST",
        path: "/v1/prompt/predict",
        route: QueuedRoute::Model(ModelRoute::PromptPredict),
    },
    RouteEntry {
        method: "POST",
        path: "/v1/prompt/prewarm",
        route: QueuedRoute::Model(ModelRoute::PromptPrewarm),
    },
    RouteEntry {
        method: "POST",
        path: "/v1/chat/completions",
        route: QueuedRoute::Model(ModelRoute::ChatCompletions),
    },
];

pub(super) fn classify_request_line(request_line: &str) -> IngressRoute<'_> {
    // Exact request line: a prefix match here used to answer 200 for
    // `/healthz` and anything else starting with `/health`.
    if request_line == "GET /health HTTP/1.1" {
        return IngressRoute::Health;
    }
    if let Some(route) = request_line.strip_prefix("POST /v1/requests/") {
        if let Some(id) = route.strip_suffix("/cancel HTTP/1.1") {
            if !id.is_empty() && !id.contains('/') && !id.contains(' ') {
                return IngressRoute::Cancel(id);
            }
        }
    }

    let mut parts = request_line.split_whitespace();
    let Some(method) = parts.next() else {
        return IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown));
    };
    let Some(target) = parts.next() else {
        return IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown));
    };
    if parts.next().is_none() || parts.next().is_some() {
        return IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown));
    }

    if method == "OPTIONS" {
        return IngressRoute::Queued(QueuedRoute::Model(ModelRoute::CorsOptions));
    }

    let path = target.split_once('?').map_or(target, |(p, _)| p);
    if method == "GET" && path == "/resources" {
        return IngressRoute::Resources;
    }

    for entry in STATIC_ROUTES {
        if entry.method == method && entry.path == path {
            return IngressRoute::Queued(entry.route);
        }
    }
    IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown))
}

/// Answers every request that does not need a loaded model. Which routes are in
/// this set is a contract, not a cache: `/health` and `/resources` are the only
/// probes that answer while the model lock is held, and `watch::classify` uses
/// exactly that asymmetry to tell an outage from a busy engine. Moving a
/// post-lock route up here would make every probe answer pre-lock and turn
/// "held by the generation" into a silent "Ready".
pub(super) fn prepare_connection(
    mut stream: TcpStream,
    default_model: &str,
    max_context_tokens: usize,
    registry: &Arc<RequestRegistry>,
) -> Result<Option<(String, usize, QueuedConnection)>> {
    let (request_line, body) = read_request(&mut stream)?;
    let route = match classify_request_line(&request_line) {
        IngressRoute::Health => {
            respond(&mut stream, 200, &json!({"status":"ok"}))?;
            return Ok(None);
        }
        IngressRoute::Resources => {
            respond(&mut stream, 200, &resources::snapshot())?;
            return Ok(None);
        }
        IngressRoute::Cancel(id) => {
            let cancelled = registry.cancel(id);
            respond(
                &mut stream,
                if cancelled { 200 } else { 404 },
                &json!({"cancelled":cancelled}),
            )?;
            return Ok(None);
        }
        IngressRoute::Queued(queued) => queued,
    };

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
            route,
            request,
            control,
        },
    )))
}

pub(super) fn resolve_model_name(requested: &str, available: &[String]) -> Option<String> {
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

pub(super) fn dispatch_connection(
    model_name: String,
    connection: QueuedConnection,
    runtime: &mut ServerRuntime,
    rag: &RagTools,
) -> Result<()> {
    let model_route = match connection.route {
        QueuedRoute::Catalog => {
            return respond(
                &mut connection.stream.try_clone()?,
                200,
                &json!({"object": "list", "data": launch::list_models()}),
            );
        }
        QueuedRoute::Launch => {
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
        QueuedRoute::Models => {
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
        QueuedRoute::Model(route) => route,
    };

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
    let is_chat = model_route.uses_chat_session();
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
        model_route,
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

#[cfg(test)]
mod model_resolution_tests {
    use super::{classify_request_line, resolve_model_name, IngressRoute, ModelRoute, QueuedRoute};

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

    #[test]
    fn declarative_route_table_preserves_prelock_contract_and_rejects_lookalikes() {
        assert_eq!(
            classify_request_line("GET /health HTTP/1.1"),
            IngressRoute::Health
        );
        assert_eq!(
            classify_request_line("GET /healthz HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown))
        );
        assert_eq!(
            classify_request_line("GET /resources?ts=1 HTTP/1.1"),
            IngressRoute::Resources
        );
        assert_eq!(
            classify_request_line("POST /v1/requests/req-42/cancel HTTP/1.1"),
            IngressRoute::Cancel("req-42")
        );
        assert_eq!(
            classify_request_line("GET /v1/models HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Models)
        );
        assert_eq!(
            classify_request_line("GET /v1/models_extra HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Unknown))
        );
        assert_eq!(
            classify_request_line("GET /stats?ts=1730000000 HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::Stats))
        );
        assert_eq!(
            classify_request_line("GET / HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::ChatPage))
        );
        assert_eq!(
            classify_request_line("GET /index.html HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::ChatPage))
        );
        assert_eq!(
            classify_request_line("OPTIONS /v1/chat/completions HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::CorsOptions))
        );
        assert_eq!(
            classify_request_line("POST /v1/prompt/prewarm HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::PromptPrewarm))
        );
        assert_eq!(
            classify_request_line("POST /v1/chat/completions HTTP/1.1"),
            IngressRoute::Queued(QueuedRoute::Model(ModelRoute::ChatCompletions))
        );
        assert!(ModelRoute::ChatCompletions.uses_chat_session());
        assert!(ModelRoute::PromptPrewarm.uses_chat_session());
        assert!(!ModelRoute::Stats.uses_chat_session());
        assert!(!ModelRoute::PromptPredict.uses_chat_session());
    }
}
