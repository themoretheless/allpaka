//! Studio's harness end to end over HTTP, against a local streaming model; no
//! paid API calls.
//!
//! Covers the editable harness (validation, activation, load_skill, hooks), a
//! JSON Propose-Probe-Compose job (probes on copies, adoption), an agentic
//! job over a stream of two tasks with a report-only rubric, a dataset
//! stream, an external executor and the experience library.
//!
//! With `ALLPAKA_CODEX_BIN` and/or `ALLPAKA_OPENCODE_BIN` set, the real Codex
//! and OpenCode CLIs also run as external executors against the same mock.

use axum::{
    extract::State,
    http::Uri,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const SKILL: &str = "---\nname: note\ndescription: the task asks for a note file\n---\nWrite workspace/out.txt with the note, then reread it.\n";
const MEMORY: &str = "In this workspace notes go to workspace/out.txt.";

/// What the mock saw. Handlers cannot fail the test themselves, so broken
/// expectations are collected and asserted at the end.
#[derive(Default)]
struct Seen {
    systems: Vec<String>,
    external: Vec<String>,
    violations: Vec<String>,
}
type Shared = Arc<Mutex<Seen>>;

fn say(text: &str) -> Value {
    json!({"choices":[{"delta":{"content":text},"finish_reason":"stop"}]})
}
fn call(name: &str, args: Value, n: usize) -> Value {
    json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":format!("c{n}"),"type":"function","function":{"name":name,"arguments":args.to_string()}}]},"finish_reason":"tool_calls"}]})
}
fn sse(event: &Value) -> axum::response::Response {
    (
        [("content-type", "text/event-stream")],
        format!("data: {event}\n\ndata: [DONE]\n\n"),
    )
        .into_response()
}

/// Studio's own calls: rollouts, proposers (agentic and JSON), composer,
/// reflector and judge.
fn studio_reply(seen: &mut Seen, body: &Value) -> Value {
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    let system = messages[0]["content"].as_str().unwrap_or("").to_string();
    seen.systems.push(system.clone());
    let tools: Vec<String> = body["tools"]
        .as_array()
        .map(|t| {
            t.iter()
                .filter_map(|t| t["function"]["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let results: Vec<String> = messages
        .iter()
        .filter(|m| m["role"] == "tool")
        .map(|m| m["content"].as_str().unwrap_or("").to_string())
        .collect();
    let last = messages
        .last()
        .and_then(|m| m["content"].as_str())
        .unwrap_or("")
        .to_string();
    let mut expect = |ok: bool, what: &str| {
        if !ok {
            seen.violations.push(what.to_string());
        }
    };
    let n = results.len();
    let proposer =
        system.contains("YOUR DIMENSION") || system.contains("composing THE final harness");
    if proposer && !tools.is_empty() {
        expect(
            !tools
                .iter()
                .any(|t| t == "run_command" || t == "load_skill"),
            "agentic proposers get only file tools",
        );
        if system.contains("YOUR DIMENSION: INV") {
            if n == 0 {
                call(
                    "write_file",
                    json!({"path":"harness/memory.md","content":MEMORY}),
                    n,
                )
            } else {
                say("base1 searched for the target file")
            }
        } else if system.contains("YOUR DIMENSION: DOM") {
            say("KEEP_BASE: nothing domain specific")
        } else if n == 0 {
            call(
                "read_file",
                json!({"path":"context/pool/inv1/harness/memory.md"}),
                n,
            )
        } else if n == 1 {
            expect(
                results[0].contains(MEMORY),
                "composer reads inv1 from the pool",
            );
            call(
                "write_file",
                json!({"path":"harness/memory.md","content":MEMORY}),
                n,
            )
        } else {
            say("kept -> memory.md -> base1 searched 1/1")
        }
    } else if proposer {
        if system.contains("YOUR DIMENSION: INV") {
            say(
                &json!({"files":{"memory.md":MEMORY},"notes":"base1 searched for the target file"})
                    .to_string(),
            )
        } else if system.contains("YOUR DIMENSION: DOM") {
            say(&json!({"files":{},"notes":"KEEP_BASE: nothing domain specific"}).to_string())
        } else {
            expect(
                last.contains("candidate inv1") && last.contains("candidate dom1"),
                "the composer sees both candidates",
            );
            say(
                &json!({"adopt":"inv1","notes":"kept -> memory.md -> base1 searched 1/1"})
                    .to_string(),
            )
        }
    } else if tools.is_empty() {
        if system.contains("complete record") {
            say(&json!({"edits":[{"component":"memory.md","candidates":"inv1","change":"names the note path","observed":"agg1 wrote directly","verdict":"confounded"}],
                        "experience":{"cue":"a task writes a note file","lesson":"state the target path in memory.md","evidence":"inv1 vs base1","confidence":"low: one rollout"}}).to_string())
        } else if system.starts_with("Evaluate the supplied answer") {
            let reference =
                serde_json::from_str::<Value>(&last).unwrap_or_default()["reference"].clone();
            let score = if reference == "a note file exists" {
                1.0
            } else {
                0.75
            };
            say(&json!({"score":score,"reason":"note written"}).to_string())
        } else {
            say("unexpected")
        }
    } else if n == 0 {
        expect(
            tools.iter().any(|t| t == "load_skill")
                && system.contains("# Available skills")
                && system.contains("load the `note` skill"),
            "rollouts list the skill and its trigger",
        );
        call("load_skill", json!({"name":"note"}), n)
    } else if n == 1 {
        expect(
            results[0].contains("Write workspace/out.txt"),
            "load_skill returns the skill",
        );
        call(
            "write_file",
            json!({"path":"workspace/out.txt","content":"note"}),
            n,
        )
    } else {
        expect(
            results[1].contains("[harness hook] Reopen what you wrote."),
            "the post_tool hook note follows write_file",
        );
        say("wrote the note")
    }
}

/// Real Codex (Responses API) and OpenCode (chat completions) runs.
fn external_reply(seen: &mut Seen, body: &Value, responses: bool) -> axum::response::Response {
    seen.external.push(body.to_string());
    let text = "external-real-ok";
    let mut out = String::new();
    if responses {
        let item = json!({"id":"m1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]});
        let events = [
            (
                "response.created",
                json!({"type":"response.created","response":{"id":"r1","object":"response","status":"in_progress","output":[]}}),
            ),
            (
                "response.output_item.added",
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"m1","type":"message","role":"assistant","status":"in_progress","content":[]}}),
            ),
            (
                "response.output_text.delta",
                json!({"type":"response.output_text.delta","item_id":"m1","output_index":0,"content_index":0,"delta":text}),
            ),
            (
                "response.output_item.done",
                json!({"type":"response.output_item.done","output_index":0,"item":item}),
            ),
            (
                "response.completed",
                json!({"type":"response.completed","response":{"id":"r1","object":"response","status":"completed","output":[item],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}),
            ),
        ];
        for (name, data) in events {
            out.push_str(&format!("event: {name}\ndata: {data}\n\n"));
        }
    } else {
        for chunk in [
            json!({"id":"c1","object":"chat.completion.chunk","created":0,"model":"ocmock","choices":[{"index":0,"delta":{"role":"assistant","content":text},"finish_reason":null}]}),
            json!({"id":"c1","object":"chat.completion.chunk","created":0,"model":"ocmock","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
        ] {
            out.push_str(&format!("data: {chunk}\n\n"));
        }
        out.push_str("data: [DONE]\n\n");
    }
    ([("content-type", "text/event-stream")], out).into_response()
}

async fn model(
    State(seen): State<Shared>,
    uri: Uri,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let mut seen = seen.lock().unwrap();
    let responses = uri.path().ends_with("/responses");
    if responses || body["model"] == "ocmock" {
        return external_reply(&mut seen, &body, responses);
    }
    let event = studio_reply(&mut seen, &body);
    sse(&event)
}

struct Studio {
    base: String,
    client: reqwest::Client,
}
impl Studio {
    async fn send(&self, path: &str, body: Option<Value>) -> (u16, Value) {
        let url = format!("{}/api/{path}", self.base);
        let request = match body {
            Some(body) => self
                .client
                .post(url)
                .header("X-Allpaka-Client", "studio")
                .json(&body),
            None => self.client.get(url),
        };
        let response = request.send().await.expect("Studio request");
        let status = response.status().as_u16();
        (status, response.json().await.unwrap_or(Value::Null))
    }
    async fn get(&self, path: &str) -> Value {
        let (status, value) = self.send(path, None).await;
        assert_eq!(status, 200, "GET {path}: {value}");
        value
    }
    async fn post(&self, path: &str, body: Value) -> Value {
        let (status, value) = self.send(path, Some(body)).await;
        assert_eq!(status, 200, "POST {path}: {value}");
        value
    }
    async fn reject(&self, path: &str, body: Value, needle: &str) {
        let (status, value) = self.send(path, Some(body)).await;
        assert_eq!(status, 400, "POST {path} was accepted: {value}");
        let error = value["error"].as_str().unwrap_or("");
        assert!(
            error.contains(needle),
            "POST {path}: {error:?} lacks {needle:?}"
        );
    }
    async fn until(&self, path: &str, done: impl Fn(&Value) -> bool, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let value = self.get(path).await;
            if done(&value) {
                return value;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting on {path}: {value}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    /// Start a refinement job and wait for it; returns the job detail.
    async fn job(&self, body: Value, timeout: Duration) -> Value {
        let id = self.post("harness/jobs", body).await["id"]
            .as_str()
            .unwrap()
            .to_string();
        let detail = self
            .until(
                &format!("harness/jobs/{id}"),
                |d| matches!(d["job"]["status"].as_str(), Some("completed" | "failed")),
                timeout,
            )
            .await;
        assert_eq!(
            detail["job"]["status"], "completed",
            "{}",
            detail["job"]["error"]
        );
        detail
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
fn session_files(data: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(data)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "multi_thread")]
async fn harness_end_to_end() {
    let root: PathBuf =
        std::env::temp_dir().join(format!("allpaka-harness-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (workspace, data) = (root.join("workspace"), root.join("history"));
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(workspace.join("readme.txt"), "fixture").unwrap();

    let seen: Shared = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_port = listener.local_addr().unwrap().port();
    let router = Router::new()
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data":[{"id":"mock"}]})) }),
        )
        .route("/v1/chat/completions", post(model))
        .route("/v1/responses", post(model))
        .with_state(seen.clone());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    // This test is its own process: the environment is Studio's to read.
    std::env::set_var("ALLPAKA_ONLINE_WORKER", "0");
    std::env::set_var("OPENROUTER_API_KEY", "");
    std::env::set_var(
        "ALLPAKA_LOCAL_BASE_URL",
        format!("http://127.0.0.1:{mock_port}/v1"),
    );
    let port = free_port();
    let (bind, ws, dir) = (
        format!("127.0.0.1:{port}").parse().unwrap(),
        workspace.clone(),
        data.clone(),
    );
    std::thread::spawn(move || allpaka_chat::run(bind, ws, Some(dir)).expect("Studio stopped"));

    let studio = Studio {
        base: format!("http://127.0.0.1:{port}"),
        client: reqwest::Client::new(),
    };
    let started = Instant::now();
    let config = format!("{}/api/config", studio.base);
    while !studio
        .client
        .get(&config)
        .send()
        .await
        .is_ok_and(|r| r.status().is_success())
    {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "Studio did not start"
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let project = studio
        .post("projects", json!({"id":"","name":"harness","roots":[{"alias":"workspace","path":workspace,"writable":true}],"instructions":""}))
        .await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let harness_url = format!("harness?project_id={project}");

    // The editable harness: validation, the automatic trigger line, activation.
    let empty = studio.get(&harness_url).await;
    assert!(
        empty["active"].is_null()
            && empty["current"]["tree"] == json!({})
            && empty["current"]["preview"] == ""
    );
    studio
        .reject(
            "harness",
            json!({"project_id":project,"tree":{"notes.txt":"x"}}),
            "not a harness path",
        )
        .await;
    studio
        .reject(
            "harness",
            json!({"project_id":project,"tree":{"skills/note/SKILL.md":"---\nname: other\n---\n"}}),
            "frontmatter name must be `note`",
        )
        .await;
    studio
        .reject("harness", json!({"project_id":project,"tree":{"hooks.json":"{\"pre_tool\":[{\"inject\":\"x\"}]}"}}), "Each pre_tool hook needs exactly one of block, command")
        .await;
    let hooks = json!({"post_tool":[{"matcher":"write_file","inject":"Reopen what you wrote."}]})
        .to_string();
    let saved = studio
        .post("harness", json!({"project_id":project,"tree":{"skills/note/SKILL.md":SKILL,"hooks.json":hooks},"note":"fixture","activate":true}))
        .await;
    assert!(
        saved["current"]["tree"]["guardrails.md"]
            .as_str()
            .unwrap()
            .contains("load the `note` skill"),
        "trigger line is added on save"
    );
    let first = studio.get(&harness_url).await;
    assert_eq!(first["active"], saved["revision"]["id"]);
    assert_eq!(first["revisions"].as_array().unwrap().len(), 1);

    // A real Auto turn uses the skill and sees the hook.
    let settings = json!({"project_id":project,"provider":"local","model":"mock","mode":"auto","max_steps":8,"allow_writes":true});
    let session = studio.post("sessions", settings.clone()).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    studio
        .post(
            &format!("sessions/{session}/actions"),
            json!({"kind":"send","text":"Leave a note","settings":settings}),
        )
        .await;
    let done = studio
        .until(
            &format!("sessions/{session}"),
            |s| matches!(s["status"].as_str(), Some("idle" | "error")),
            Duration::from_secs(20),
        )
        .await;
    assert_eq!(done["status"], "idle", "{}", done["error"]);
    assert_eq!(
        done["messages"].as_array().unwrap().last().unwrap()["content"],
        "wrote the note"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.join("out.txt")).unwrap(),
        "note"
    );
    std::fs::remove_file(workspace.join("out.txt")).unwrap();
    let stored = session_files(&data);

    // One JSON-mode step on the conversation's task.
    let detail = studio
        .job(
            json!({"session_id":session,"k":1,"mode":"parallel","agentic":false}),
            Duration::from_secs(60),
        )
        .await;
    let result = &detail["job"]["results"][0];
    let outcome = &result["outcome"];
    assert_eq!(
        (
            outcome["rollouts"].as_u64(),
            outcome["stages"].as_u64(),
            outcome["chosen"].as_str()
        ),
        (Some(4), Some(3), Some("agg1"))
    );
    assert!(
        outcome["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["delivered"] == true),
        "{}",
        outcome["runs"]
    );
    assert_eq!(detail["diffs"][0]["added"], json!(["memory.md"]));
    assert_eq!(
        detail["active"], first["active"],
        "a job never activates its result"
    );
    assert!(!workspace.join("out.txt").exists(), "probes run on copies");
    assert_eq!(
        session_files(&data),
        stored,
        "probe and proposer sessions are not saved"
    );
    assert!(
        seen.lock()
            .unwrap()
            .systems
            .iter()
            .any(|s| s.contains(&format!("# Long-term memory\n{MEMORY}"))),
        "probes run the candidate harness"
    );
    let job_id = detail["job"]["id"].as_str().unwrap();
    studio
        .post(&format!("harness/jobs/{job_id}/adopt"), json!({}))
        .await;
    let adopted = studio.get(&harness_url).await;
    assert_eq!(adopted["active"], result["revision"]);
    assert_eq!(adopted["current"]["tree"]["memory.md"], MEMORY);

    // Agentic proposers over a stream of two tasks, with a report-only rubric.
    let detail = studio
        .job(json!({"session_id":session,"k":1,"agentic":true,"rubric":"The note file exists.","tasks":["Leave a note","Leave another note"]}), Duration::from_secs(90))
        .await;
    let results = detail["job"]["results"].as_array().unwrap();
    assert_eq!(
        results
            .iter()
            .map(|r| r["task"].clone())
            .collect::<Vec<_>>(),
        [json!("Leave a note"), json!("Leave another note")]
    );
    for r in results {
        let o = &r["outcome"];
        assert_eq!(
            (
                o["rollouts"].as_u64(),
                o["stages"].as_u64(),
                o["chosen"].as_str()
            ),
            (Some(4), Some(3), Some("agg1")),
            "{o}"
        );
        assert_eq!(o["fallbacks"], json!([]));
        assert_eq!(o["scores"], json!({"base1":0.75,"agg1":0.75}));
        assert!(
            r["revision"].is_null(),
            "the composer kept memory.md, which the adopted base already has"
        );
    }
    {
        let seen = seen.lock().unwrap();
        assert!(
            seen.systems.iter().any(|s| s.contains("YOURS TO EDIT"))
                && seen.systems.iter().any(|s| s.contains("HOW TO COMPOSE"))
        );
    }
    assert!(!workspace.join("out.txt").exists());

    // A stream from a dataset: the expected output reaches only the judge.
    let dataset = studio
        .post("evaluation/datasets", json!({"project_id":project,"name":"Notes","base_version":0,"samples":[{"id":"n1","input":"Leave a note","expected_output":"a note file exists"}]}))
        .await;
    let detail = studio
        .job(json!({"session_id":session,"agentic":false,"rubric":"The note file exists.","dataset":{"id":dataset["id"]}}), Duration::from_secs(60))
        .await;
    assert_eq!(detail["job"]["tasks"], json!(["Leave a note"]));
    assert_eq!(detail["job"]["references"], json!(["a note file exists"]));
    assert_eq!(
        detail["job"]["results"][0]["outcome"]["scores"],
        json!({"base1":1.0,"agg1":1.0})
    );
    assert!(!seen
        .lock()
        .unwrap()
        .systems
        .iter()
        .any(
            |s| (s.contains("YOUR DIMENSION") || s.contains("composing"))
                && s.contains("a note file exists")
        ));

    // An external agent CLI executes the rollouts on the materialized harness.
    let detail = studio
        .job(json!({"session_id":session,"agentic":false,"external":{"command":"cat > task.txt; test -f .agents/skills/note/SKILL.md && echo external-ok","layout":"codex"}}), Duration::from_secs(60))
        .await;
    let runs = &detail["job"]["results"][0]["outcome"]["runs"];
    assert!(
        runs.as_array()
            .unwrap()
            .iter()
            .all(|r| r["delivered"] == true),
        "{runs}"
    );
    assert!(!workspace.join("task.txt").exists());

    // Optional: the real Codex and OpenCode CLIs, pointed at the mock.
    let mut expected_library = 5;
    let mut clis = vec![];
    if let Ok(codex) = std::env::var("ALLPAKA_CODEX_BIN") {
        let home = root.join("codex-home");
        std::fs::create_dir_all(&home).unwrap();
        let provider = format!("model_providers.mock={{name=\"mock\",base_url=\"http://127.0.0.1:{mock_port}/v1\",wire_api=\"responses\",env_key=\"MOCK_KEY\"}}");
        clis.push(("codex", format!("CODEX_HOME={} MOCK_KEY=x {codex} exec --skip-git-repo-check -c model_provider=mock -c '{provider}' -m cxmock -", home.display())));
    }
    if let Ok(opencode) = std::env::var("ALLPAKA_OPENCODE_BIN") {
        let home = root.join("oc-home");
        std::fs::create_dir_all(&home).unwrap();
        let config = json!({"provider":{"mock":{"npm":"@ai-sdk/openai-compatible","name":"mock","options":{"baseURL":format!("http://127.0.0.1:{mock_port}/v1"),"apiKey":"x"},"models":{"ocmock":{"name":"ocmock"}}}},"model":"mock/ocmock"});
        let h = home.display();
        clis.push(("opencode", format!("HOME={h} XDG_CONFIG_HOME={h}/c XDG_DATA_HOME={h}/d XDG_CACHE_HOME={h}/x OPENCODE_CONFIG_CONTENT='{config}' {opencode} run")));
    }
    for (layout, command) in clis {
        seen.lock().unwrap().external.clear();
        let detail = studio
            .job(json!({"session_id":session,"agentic":false,"external":{"command":command,"layout":layout}}), Duration::from_secs(300))
            .await;
        let runs = &detail["job"]["results"][0]["outcome"]["runs"];
        assert!(
            runs.as_array()
                .unwrap()
                .iter()
                .all(|r| r["delivered"] == true),
            "{layout}: {runs}"
        );
        let seen = seen.lock().unwrap();
        let with_harness: Vec<&String> = seen
            .external
            .iter()
            .filter(|r| r.contains(MEMORY))
            .collect();
        assert!(
            !with_harness.is_empty(),
            "{layout} never sent the harness to the model"
        );
        assert!(
            with_harness
                .iter()
                .any(|r| r.contains("the task asks for a note")),
            "{layout} did not discover the skill"
        );
        eprintln!(
            "real {layout}: {} model requests, harness and skill seen",
            seen.external.len()
        );
        expected_library += 1;
    }

    // The library, input bounds and switching the harness off.
    let library = studio
        .get(&format!("harness/library?project_id={project}"))
        .await;
    let entries = library["entries"].as_array().unwrap();
    assert_eq!(entries.len(), expected_library);
    assert!(entries
        .iter()
        .all(|e| e["entry"]["cue"] == "a task writes a note file"));
    assert!(entries[0]["experience"]
        .as_str()
        .unwrap()
        .contains("[confounded]"));
    studio
        .reject(
            "harness/jobs",
            json!({"session_id":session,"k":9}),
            "k must be between 1 and 4",
        )
        .await;
    studio
        .reject(
            "harness/jobs",
            json!({"session_id":session,"tasks":["ok","   "]}),
            "at most 20 non-empty tasks",
        )
        .await;
    studio
        .post(
            "harness/activate",
            json!({"project_id":project,"revision":null}),
        )
        .await;
    assert!(studio.get(&harness_url).await["active"].is_null());

    let violations = std::mem::take(&mut seen.lock().unwrap().violations);
    assert!(
        violations.is_empty(),
        "the model saw unexpected requests: {violations:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
