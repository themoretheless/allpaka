//! Remote swarm members execute their model/tool loop against worker-owned roots.
use crate::{
    provider, swarm, tools,
    types::{Message, Mode, Settings, SwarmMember},
    App,
};
use anyhow::{bail, Context, Result};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Duration};

#[derive(Serialize, Deserialize)]
pub(crate) struct Task {
    settings: Settings,
    system: String,
    prompt: String,
    project: String,
}
#[derive(Serialize, Deserialize)]
pub(crate) struct Report {
    pub content: String,
    pub usage: Value,
    pub truncated: bool,
}
fn key() -> Result<String> {
    let key = std::env::var("ALLPAKA_CLUSTER_KEY")
        .context("ALLPAKA_CLUSTER_KEY is required for remote agents")?;
    if key.len() < 16 {
        bail!("cluster key must contain at least 16 bytes");
    }
    Ok(key)
}
pub(crate) fn authorized(headers: &HeaderMap) -> bool {
    key().ok().is_some_and(|key| {
        headers.get("authorization").and_then(|v| v.to_str().ok())
            == Some(format!("Bearer {key}").as_str())
    })
}
pub(crate) fn validate_url(value: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(value)?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        bail!("worker must be an HTTP(S) origin without credentials, path or query");
    }
    Ok(url)
}
pub(crate) async fn dispatch(
    client: &reqwest::Client,
    member: &SwarmMember,
    settings: &Settings,
    system: &str,
    prompt: &str,
) -> Result<Report> {
    let endpoint = validate_url(&member.worker)?.join("api/worker/run")?;
    let task = Task {
        settings: settings.clone(),
        system: system.into(),
        prompt: prompt.into(),
        project: if member.worker_project.is_empty() {
            settings.project_id.clone()
        } else {
            member.worker_project.clone()
        },
    };
    let response = client
        .post(endpoint)
        .bearer_auth(key()?)
        .timeout(Duration::from_secs(300))
        .json(&task)
        .send()
        .await?
        .error_for_status()?;
    if response.content_length().is_some_and(|n| n > 100_000) {
        bail!("worker response too large");
    }
    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if bytes.len() + chunk.len() > 100_000 {
            bail!("worker response too large");
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: Value = serde_json::from_slice(&bytes)?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        bail!("remote worker: {error}");
    }
    Ok(serde_json::from_value(value)?)
}
pub(crate) async fn run(
    State(app): State<App>,
    Json(mut task): Json<Task>,
) -> std::result::Result<axum::response::Response, (StatusCode, Json<Value>)> {
    static LIMIT: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    let _permit = LIMIT
        .get_or_init(|| tokio::sync::Semaphore::new(3))
        .try_acquire()
        .map_err(|_| {
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error":"worker busy"})),
            )
        })?;
    task.settings.mode = Mode::Chat;
    task.settings.allow_writes = false;
    task.settings.max_steps = task.settings.max_steps.clamp(1, 8);
    task.settings.max_output_tokens = task.settings.max_output_tokens.clamp(1, 8192);
    task.settings.json_mode = false;
    if task.prompt.len() > 100_000 || task.system.len() > 100_000 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"worker prompt too large"})),
        ));
    }
    if !app
        .projects
        .lock()
        .unwrap()
        .iter()
        .any(|p| p.id == task.project)
        || !app
            .providers
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.id == task.settings.provider)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"unknown worker project or provider"})),
        ));
    }
    // Execute inside the response body future: Hyper drops this future on
    // disconnect, cancelling model I/O and releasing admission immediately.
    let body = axum::body::Body::from_stream(futures_util::stream::once(async move {
        let _permit = _permit;
        let result = tokio::time::timeout(Duration::from_secs(290), execute(&app, task)).await;
        let value = match result {
            Ok(Ok(report)) => serde_json::to_value(report).unwrap(),
            Ok(Err(error)) => json!({"error":error.to_string()}),
            Err(_) => json!({"error":"worker task timed out"}),
        };
        Ok::<_, std::io::Error>(value.to_string())
    }));
    Ok(axum::response::Response::builder()
        .header("content-type", "application/json")
        .body(body)
        .unwrap())
}
async fn execute(app: &App, task: Task) -> Result<Report> {
    if task.prompt.len() > 100_000 || task.system.len() > 100_000 {
        bail!("worker prompt too large");
    }
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == task.project)
        .cloned()
        .context("unknown worker project")?;
    let provider = app
        .providers
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == task.settings.provider)
        .cloned()
        .context("unknown worker provider")?;
    let schemas = tools::schemas(&task.settings, false);
    let budget = task.settings.swarm.report_bytes.clamp(1000, 24_000);
    let mut history = vec![Message::text("user", task.prompt)];
    let mut report = Report {
        content: String::new(),
        usage: Value::Null,
        truncated: false,
    };
    for step in 0..task.settings.max_steps {
        let (message, usage) = provider::generate(
            &app.client,
            &provider,
            &task.settings,
            &task.system,
            &history,
            &schemas,
            |_, _| {},
        )
        .await?;
        report.usage = swarm::merge_usage(report.usage, usage);
        report.content.push_str(&message.content);
        report.truncated |= message.truncated;
        if report.content.len() > budget {
            let mut end = budget;
            while !report.content.is_char_boundary(end) {
                end -= 1;
            }
            report.content.truncate(end);
            report.truncated = true;
            break;
        }
        if message.tool_calls.is_empty() || message.truncated {
            break;
        }
        if step + 1 == task.settings.max_steps {
            report.truncated = true;
            break;
        }
        history.push(message.clone());
        for call in &message.tool_calls {
            let result = swarm::run_tool(app, &project, &task.settings, call);
            history.push(Message {
                role: "tool".into(),
                content: result.to_string(),
                tool_call_id: call["id"].as_str().map(str::to_owned),
                ..Message::default()
            });
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn authenticated_worker_runs_tools_on_its_own_machine() {
        use axum::{routing::post, Router};
        use std::sync::{Arc, Mutex};
        let previous = std::env::var_os("ALLPAKA_CLUSTER_KEY");
        std::env::set_var("ALLPAKA_CLUSTER_KEY", "fixture-cluster-key-12345");
        let root = std::env::temp_dir().join(format!(
            "allpaka-worker-{}-{}",
            std::process::id(),
            crate::evaluation::new_id()
        ));
        std::fs::create_dir_all(root.join("workspace")).unwrap();
        std::fs::create_dir_all(root.join("history")).unwrap();
        std::fs::write(root.join("workspace/evidence.txt"), "worker-only-evidence").unwrap();
        let model_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let model_address = model_listener.local_addr().unwrap();
        use axum::response::IntoResponse;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Active(Arc<AtomicUsize>);
        impl Drop for Active {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let active = Arc::new(AtomicUsize::new(0));
        let model_active = active.clone();
        let model = tokio::spawn(async move {
            let router = Router::new().route("/v1/chat/completions", post(|State(active): State<Arc<AtomicUsize>>, Json(request): Json<Value>| async move {
                if request["messages"].as_array().unwrap().last().unwrap()["content"] == "WAIT_FOR_CANCEL" {
                    let body = axum::body::Body::from_stream(futures_util::stream::once(async move {
                        active.fetch_add(1, Ordering::SeqCst);
                        let _active = Active(active);
                        futures_util::future::pending::<std::result::Result<String, std::io::Error>>().await
                    }));
                    return axum::response::Response::builder().header("content-type", "text/event-stream").body(body).unwrap();
                }
                let tool_result = request["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool");
                let event = if request["tools"].as_array().is_none_or(|tools| tools.is_empty()) {
                    json!({"choices":[{"delta":{"content":"merged remote evidence"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}})
                } else if let Some(result) = tool_result {
                    assert!(result["content"].as_str().unwrap().contains("worker-only-evidence"));
                    json!({"choices":[{"delta":{"content":"verified remote file"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}})
                } else {
                    json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"read-1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"workspace/evidence.txt\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":1}})
                };
                ([("content-type", "text/event-stream")], format!("data: {event}\n\ndata: [DONE]\n\n")).into_response()
            })).with_state(model_active);
            axum::serve(model_listener, router).await.unwrap();
        });
        let worker_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_address = worker_listener.local_addr().unwrap();
        let data = root.join("history").canonicalize().unwrap();
        let workspace = root.join("workspace").canonicalize().unwrap();
        let mut provider = provider::defaults().remove(0);
        provider.id = "fixture".into();
        provider.base = format!("http://{model_address}/v1");
        provider.key.clear();
        provider.custom = true;
        let app = App {
            projects: Arc::new(Mutex::new(vec![crate::context::Project {
                id: "worker-repo".into(),
                name: "fixture".into(),
                roots: vec![crate::context::Root {
                    alias: "workspace".into(),
                    path: workspace.clone(),
                    writable: true,
                    repository: false,
                }],
                instructions: String::new(),
            }])),
            sessions: Default::default(),
            background: Default::default(),
            observability: crate::observability::Store::new(&data).unwrap(),
            feedback: crate::feedback::Store::new(&data).unwrap(),
            review_queues: crate::review_queues::Store::new(&data).unwrap(),
            evaluation: crate::evaluation::Store::new(&data).unwrap(),
            memory: crate::memory::Store::new(&data).unwrap(),
            judge_runs: crate::judge_runs::Manager::new(&data).unwrap(),
            judges: Arc::new(tokio::sync::Semaphore::new(2)),
            memory_extractions: Arc::new(tokio::sync::Semaphore::new(2)),
            prompts: crate::prompts::Store::new(&data).unwrap(),
            experiments: crate::experiments::Manager::new(&data).unwrap(),
            matrix_jobs: crate::experiments::matrix_jobs::Manager::new(&data).unwrap(),
            providers: Arc::new(Mutex::new(vec![provider])),
            key_mutations: Default::default(),
            client: provider::client().unwrap(),
            workspace: Arc::new(workspace),
            data: Arc::new(data),
            origin: format!("http://{worker_address}"),
            plugins: Default::default(),
        };
        let remote_settings: Settings = serde_json::from_value(json!({
            "provider":"fixture","model":"tiny","project_id":"worker-repo","mode":"swarm",
            "swarm":{"members":[
                {"label":"one","provider":"provider-not-on-coordinator","model":"tiny","worker":"http://127.0.0.1:18100"},
                {"label":"two","provider":"another-remote-provider","model":"tiny","worker":"http://127.0.0.1:18101"}
            ]}
        })).unwrap();
        crate::validate_settings(&remote_settings, &app).unwrap();
        let mut coordinator = app.clone();
        let mut synth = coordinator.providers.lock().unwrap()[0].clone();
        synth.id = "synth-only-on-coordinator".into();
        coordinator.providers = Arc::new(Mutex::new(vec![synth]));
        let worker = tokio::spawn(async move {
            axum::serve(worker_listener, crate::routes(app))
                .await
                .unwrap();
        });
        let client = reqwest::Client::new();
        let unauthorized = client
            .post(format!("http://{worker_address}/api/worker/run"))
            .json(&json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let member = SwarmMember {
            label: "remote".into(),
            provider: "fixture".into(),
            model: "tiny".into(),
            worker: format!("http://{worker_address}"),
            worker_project: "worker-repo".into(),
            ..SwarmMember::default()
        };
        let settings: Settings = serde_json::from_value(json!({"provider":"fixture","model":"tiny","max_steps":2,"allow_writes":true,"mode":"chat"})).unwrap();
        let report = dispatch(
            &client,
            &member,
            &settings,
            "inspect files",
            "read the evidence",
        )
        .await
        .unwrap();
        assert_eq!(report.content, "verified remote file");
        assert_eq!(report.usage["prompt_tokens"], 5);
        assert!(!report.truncated);
        let mut bad_project = member.clone();
        bad_project.worker_project = "unknown-project".into();
        assert!(
            dispatch(&client, &bad_project, &settings, "inspect", "read")
                .await
                .is_err()
        );
        let mut roster = vec![member.clone(), member.clone()];
        roster[0].label = "one".into();
        roster[1].label = "two".into();
        let swarm_settings: Settings = serde_json::from_value(json!({
            "provider":"synth-only-on-coordinator","model":"tiny","project_id":"worker-repo","mode":"swarm",
            "swarm":{"members":roster,"rounds":1,"max_steps_per_member":2}
        })).unwrap();
        crate::validate_settings(&swarm_settings, &coordinator).unwrap();
        let shared: crate::SharedSession = Arc::new(Mutex::new(crate::types::Session::new(
            "remote-wave".into(),
            swarm_settings.clone(),
        )));
        shared
            .lock()
            .unwrap()
            .messages
            .push(Message::text("user", "inspect remote evidence"));
        let project = coordinator.projects.lock().unwrap()[0].clone();
        let trace = coordinator
            .observability
            .begin("remote-wave", "worker-repo")
            .unwrap();
        let mut turn = trace.span("turn", "remote-wave", None);
        swarm::run(
            &coordinator,
            &shared,
            &swarm_settings,
            &project,
            "inspect",
            &trace,
            turn.id(),
        )
        .await
        .unwrap();
        {
            let session = shared.lock().unwrap();
            let message = session.messages.last().unwrap();
            assert_eq!(message.content, "merged remote evidence");
            assert_eq!(message.swarm.len(), 2);
            for report in &message.swarm {
                assert_eq!(report.content, "verified remote file");
                assert_eq!(report.worker, member.worker);
                assert_eq!(report.status, crate::types::MemberStatus::Done);
            }
        }
        let mut changed = swarm_settings.clone();
        changed.swarm.members.clear();
        swarm::retry(
            &coordinator,
            &shared,
            &changed,
            &project,
            "inspect",
            "one",
            &trace,
            turn.id(),
        )
        .await
        .unwrap();
        assert_eq!(
            shared.lock().unwrap().messages.last().unwrap().content,
            "merged remote evidence"
        );
        let mut pending = Vec::new();
        for _ in 0..3 {
            let client = client.clone();
            let member = member.clone();
            let settings = settings.clone();
            pending.push(tokio::spawn(async move {
                dispatch(&client, &member, &settings, "inspect", "WAIT_FOR_CANCEL").await
            }));
        }
        tokio::time::timeout(Duration::from_secs(3), async {
            while active.load(Ordering::SeqCst) != 3 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("three remote model requests must start");
        assert!(
            dispatch(&client, &member, &settings, "inspect", "read")
                .await
                .is_err(),
            "worker must enforce admission"
        );
        for task in pending {
            task.abort();
            let _ = task.await;
        }
        tokio::time::timeout(Duration::from_secs(3), async {
            while active.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("disconnect must cancel upstream model requests");
        assert_eq!(
            dispatch(&client, &member, &settings, "inspect", "read")
                .await
                .unwrap()
                .content,
            "verified remote file"
        );
        turn.finish("completed", &Value::Null);
        worker.abort();
        model.abort();
        let _ = worker.await;
        let _ = model.await;
        std::fs::remove_dir_all(root).unwrap();
        if let Some(value) = previous {
            std::env::set_var("ALLPAKA_CLUSTER_KEY", value);
        } else {
            std::env::remove_var("ALLPAKA_CLUSTER_KEY");
        }
    }

    #[test]
    fn worker_urls_reject_credentials_and_ambiguous_endpoints() {
        assert!(validate_url("http://127.0.0.1:18100").is_ok());
        assert!(validate_url("https://worker.example").is_ok());
        for url in [
            "file:///tmp/x",
            "http://user:secret@host",
            "http://host/api",
            "http://host/?key=secret",
            "http://host/#fragment",
        ] {
            assert!(validate_url(url).is_err(), "{url}");
        }
    }
    #[test]
    fn existing_members_remain_local_and_remote_settings_roundtrip() {
        let member: SwarmMember =
            serde_json::from_value(json!({"label":"a","provider":"local","model":"tiny"})).unwrap();
        assert!(member.worker.is_empty());
        let remote = SwarmMember {
            worker: "http://127.0.0.1:18100".into(),
            worker_project: "repo".into(),
            ..member
        };
        let restored: SwarmMember =
            serde_json::from_value(serde_json::to_value(&remote).unwrap()).unwrap();
        assert_eq!(restored.worker, remote.worker);
        assert_eq!(restored.worker_project, "repo");
    }
}
