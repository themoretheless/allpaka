//! A whole turn against a fake model: how each stage ends the turn, what
//! the session holds afterwards, what is written to disk and which spans
//! the trace records. These pin `turn` before its stages are split apart.

use crate::harness_refine::tests::{auto_settings, call, fixture, results_of, say, Fixture};
use crate::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn session(settings: Settings, text: &str) -> SharedSession {
    let mut session = Session::new(crate::evaluation::new_id(), settings);
    session.messages.push(Message::text("user", text));
    Arc::new(Mutex::new(session))
}

/// (kind, name, status) of every span in the only trace recorded so far.
fn spans(f: &Fixture) -> Vec<(String, String, String)> {
    let files: Vec<_> = std::fs::read_dir(f.data.join("observability/traces"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let record: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    record["spans"]
        .as_array()
        .unwrap()
        .iter()
        .map(|span| {
            (
                span["kind"].as_str().unwrap().to_string(),
                span["name"].as_str().unwrap().to_string(),
                span["status"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}
fn span(kind: &str, name: &str, status: &str) -> (String, String, String) {
    (kind.into(), name.into(), status.into())
}
fn stored(f: &Fixture, s: &SharedSession) -> Session {
    let id = s.lock().unwrap().id.clone();
    serde_json::from_slice(&std::fs::read(f.data.join(format!("{id}.json"))).unwrap()).unwrap()
}
fn finish(f: Fixture) {
    f.server.abort();
    std::fs::remove_dir_all(&f.root).unwrap();
}

static COMPLETE_SYSTEMS: Mutex<Vec<String>> = Mutex::new(Vec::new());
fn complete_model(request: &Value) -> Value {
    let system = request["messages"][0]["content"].as_str().unwrap_or("");
    COMPLETE_SYSTEMS.lock().unwrap().push(system.to_string());
    let results = results_of(request);
    match results.len() {
        0 => call("read_file", json!({"path":"workspace/readme.txt"})),
        1 => {
            assert!(results[0].contains("hello"), "{}", results[0]);
            call("list_files", json!({"path":"."}))
        }
        _ => {
            assert!(results[1].contains("contexts"), "{}", results[1]);
            say("done")
        }
    }
}

/// Steering becomes a user message, tool results are recorded in order,
/// both assistant messages name the provider and model, and the file on
/// disk matches the session.
#[tokio::test]
async fn a_complete_turn_records_tools_answer_trace_and_file() {
    let f = fixture(complete_model).await;
    let s = session(auto_settings(), "Read the readme");
    s.lock().unwrap().steering.push("be brief".into());
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(matches!(outcome, TurnOutcome::Complete), "{outcome:?}");
    {
        let state = s.lock().unwrap();
        let shape: Vec<_> = state
            .messages
            .iter()
            .map(|m| {
                (
                    m.role.as_str(),
                    m.tool_calls.len(),
                    m.tool_call_id.is_some(),
                )
            })
            .collect();
        assert_eq!(
            shape,
            [
                ("user", 0, false),
                ("user", 0, false),
                ("assistant", 1, false),
                ("tool", 0, true),
                ("assistant", 1, false),
                ("tool", 0, true),
                ("assistant", 0, false),
            ]
        );
        assert_eq!(
            state.messages[1].content,
            "User steering for the current task: be brief"
        );
        assert_eq!(state.messages[6].content, "done");
        for index in [2, 4, 6] {
            assert_eq!(state.messages[index].provider.as_deref(), Some("fixture"));
            assert_eq!(state.messages[index].model.as_deref(), Some("tiny"));
        }
        assert!(state.steering.is_empty());
        assert_eq!(state.step, 3);
    }
    let messages = serde_json::to_value(&s.lock().unwrap().messages).unwrap();
    assert_eq!(
        serde_json::to_value(&stored(&f, &s).messages).unwrap(),
        messages
    );
    let systems = COMPLETE_SYSTEMS.lock().unwrap().clone();
    assert_eq!(systems.len(), 3);
    assert!(systems[0].starts_with("You are allpaka in Auto mode."));
    assert!(systems[0].contains("\nProject: fixture\n"));
    let tools = systems[0]
        .split("Tools actually available for THIS request: ")
        .nth(1)
        .unwrap();
    assert!(tools.contains("read_file") && tools.contains("memory_recall"));
    assert_eq!(
        spans(&f),
        [
            span("turn", "agent", "completed"),
            span("model", "tiny", "completed"),
            span("tool", "read_file", "completed"),
            span("model", "tiny", "completed"),
            span("tool", "list_files", "completed"),
            span("model", "tiny", "completed"),
        ]
    );
    finish(f);
}

fn truncated_model(_: &Value) -> Value {
    json!({"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]})
}

/// A truncated reply ends the turn at once and keeps what arrived.
#[tokio::test]
async fn a_truncated_reply_ends_with_token_limit() {
    let f = fixture(truncated_model).await;
    let s = session(auto_settings(), "Write a lot");
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(matches!(outcome, TurnOutcome::TokenLimit), "{outcome:?}");
    let last = s.lock().unwrap().messages.last().unwrap().clone();
    assert_eq!((last.content.as_str(), last.truncated), ("partial", true));
    assert_eq!(stored(&f, &s).messages.last().unwrap().content, "partial");
    assert_eq!(
        spans(&f),
        [
            span("turn", "agent", "token_limit"),
            span("model", "tiny", "token_limit"),
        ]
    );
    finish(f);
}

fn goal_model(request: &Value) -> Value {
    let system = request["messages"][0]["content"].as_str().unwrap_or("");
    assert!(system.contains("Durable milestone plan revision: 0."));
    say("finished")
}

/// Goal mode without a ready plan asks once to continue, then gives up.
#[tokio::test]
async fn goal_mode_continues_once_then_reports_incomplete() {
    let f = fixture(goal_model).await;
    let mut settings = auto_settings();
    settings.mode = Mode::Goal;
    let s = session(settings, "Reach the goal");
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(
        matches!(outcome, TurnOutcome::GoalIncomplete),
        "{outcome:?}"
    );
    let roles: Vec<_> = s
        .lock()
        .unwrap()
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.starts_with("Goal continuation:")))
        .collect();
    assert_eq!(
        roles,
        [
            ("user".to_string(), false),
            ("assistant".to_string(), false),
            ("user".to_string(), true),
            ("assistant".to_string(), false),
        ]
    );
    assert_eq!(spans(&f)[0], span("turn", "agent", "goal_incomplete"));
    finish(f);
}

fn stop_model(request: &Value) -> Value {
    let last = request["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    if last.starts_with("Pre-completion check from the harness:\nCheck the deliverable.") {
        say("final")
    } else {
        say("draft")
    }
}

/// A stop hook runs once before the turn may complete.
#[tokio::test]
async fn a_stop_hook_asks_once_before_completion() {
    let f = fixture(stop_model).await;
    let tree = harness::Tree::from([(
        "hooks.json".to_string(),
        r#"{"stop":[{"inject":"Check the deliverable."}]}"#.to_string(),
    )]);
    let s = session(auto_settings(), "Deliver");
    s.lock().unwrap().ephemeral = Some(Arc::new(crate::Ephemeral {
        project: f.project.clone(),
        harness: Some(tree),
        system: None,
        nested: false,
        policy: crate::ExecutionPolicy::trusted_unsandboxed(),
    }));
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(matches!(outcome, TurnOutcome::Complete), "{outcome:?}");
    let contents: Vec<_> = s
        .lock()
        .unwrap()
        .messages
        .iter()
        .map(|m| m.content.lines().next().unwrap_or("").to_string())
        .collect();
    assert_eq!(
        contents,
        [
            "Deliver",
            "draft",
            "Pre-completion check from the harness:",
            "final"
        ]
    );
    finish(f);
}

fn failing_model(request: &Value) -> Value {
    assert!(
        results_of(request).is_empty(),
        "no step after the no-progress pause"
    );
    let call = |n: usize| json!({"index":n,"id":format!("c{n}"),"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"workspace/missing.txt\"}"}});
    json!({"choices":[{"delta":{"tool_calls":[call(0),call(1),call(2),call(3)]},"finish_reason":"tool_calls"}]})
}

/// The third identical failure pauses the turn; the rest of the batch is
/// recorded as cancelled, not run.
#[tokio::test]
async fn repeated_tool_failure_pauses_with_no_progress() {
    let f = fixture(failing_model).await;
    let s = session(auto_settings(), "Read a missing file");
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(matches!(outcome, TurnOutcome::NoProgress), "{outcome:?}");
    let results: Vec<Value> = s
        .lock()
        .unwrap()
        .messages
        .iter()
        .filter(|m| m.role == "tool")
        .map(|m| serde_json::from_str(&m.content).unwrap())
        .collect();
    assert_eq!(results.len(), 4);
    assert!(results[..3].iter().all(|r| r["error"].is_string()));
    assert_eq!(
        results[3]["cancelled"],
        "Repeated tool failure paused this turn; this call was not executed"
    );
    let statuses: Vec<_> = spans(&f).into_iter().map(|(_, _, status)| status).collect();
    assert_eq!(
        statuses,
        [
            "no_progress",
            "completed",
            "failed",
            "failed",
            "failed",
            "cancelled"
        ]
    );
    finish(f);
}

fn busy_model(_: &Value) -> Value {
    call("read_file", json!({"path":"workspace/readme.txt"}))
}

/// Running out of steps is an error that names the limit.
#[tokio::test]
async fn the_step_limit_is_an_error() {
    let f = fixture(busy_model).await;
    let mut settings = auto_settings();
    settings.max_steps = 2;
    let s = session(settings, "Keep reading");
    let error = crate::turn(f.app.clone(), s.clone(), None)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Step limit reached (2). Review the result and Resume to continue."
    );
    assert_eq!(s.lock().unwrap().step, 2);
    assert_eq!(spans(&f)[0], span("turn", "agent", "failed"));
    finish(f);
}

static GUARDED_CALLS: AtomicUsize = AtomicUsize::new(0);
fn guarded_model(_: &Value) -> Value {
    GUARDED_CALLS.fetch_add(1, Ordering::SeqCst);
    say("the SECRET")
}
fn secret_model(_: &Value) -> Value {
    say("the SECRET")
}
fn guarded_settings(f: &Fixture, action: &str) -> (Settings, String) {
    let manifest = guardrail_policies::Manifest::create(vec![guardrail_policies::Rule {
        id: "secret".into(),
        kind: guardrail_policies::RuleKind::ForbiddenSubstrings,
        value: json!(["SECRET"]),
    }])
    .unwrap();
    guardrail_policies::Store::open(&f.data)
        .unwrap()
        .put(&manifest)
        .unwrap();
    let hash = manifest.policy_sha256;
    let mut settings = serde_json::to_value(auto_settings()).unwrap();
    settings["guardrails"] =
        json!({"input_policy_sha256":hash,"output_policy_sha256":hash,"action":action});
    (serde_json::from_value(settings).unwrap(), hash)
}

/// A blocked input stops the turn before any model call.
#[tokio::test]
async fn a_blocked_input_never_reaches_the_model() {
    let f = fixture(guarded_model).await;
    let (settings, hash) = guarded_settings(&f, "block");
    let s = session(settings, "share the SECRET");
    let error = crate::turn(f.app.clone(), s.clone(), None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "guardrail_blocked");
    assert_eq!(GUARDED_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(s.lock().unwrap().messages.len(), 1);
    assert_eq!(
        spans(&f),
        [
            span("turn", "agent", "failed"),
            span(
                "tool",
                &format!("guardrail.input.block.fail.{hash}"),
                "failed"
            ),
        ]
    );
    finish(f);
}

/// A blocked output is never shown: the assistant message is emptied and
/// saved that way before the turn fails.
#[tokio::test]
async fn a_blocked_output_is_emptied_and_saved() {
    let f = fixture(secret_model).await;
    let (settings, hash) = guarded_settings(&f, "block");
    let s = session(settings, "hello");
    let error = crate::turn(f.app.clone(), s.clone(), None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "guardrail_blocked");
    {
        let state = s.lock().unwrap();
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[1].role, "assistant");
        assert_eq!(state.messages[1].content, "");
    }
    assert_eq!(stored(&f, &s).messages[1].content, "");
    assert_eq!(
        spans(&f),
        [
            span("turn", "agent", "failed"),
            span(
                "tool",
                &format!("guardrail.input.block.pass.{hash}"),
                "completed"
            ),
            span("model", "tiny", "completed"),
            span(
                "tool",
                &format!("guardrail.output.block.fail.{hash}"),
                "failed"
            ),
        ]
    );
    finish(f);
}

/// In observe mode a failing output is recorded and kept.
#[tokio::test]
async fn an_observed_output_failure_is_recorded_and_kept() {
    let f = fixture(secret_model).await;
    let (settings, hash) = guarded_settings(&f, "observe");
    let s = session(settings, "hello");
    let outcome = crate::turn(f.app.clone(), s.clone(), None).await.unwrap();
    assert!(matches!(outcome, TurnOutcome::Complete), "{outcome:?}");
    assert_eq!(s.lock().unwrap().messages[1].content, "the SECRET");
    assert_eq!(
        spans(&f)[3],
        span(
            "tool",
            &format!("guardrail.output.observe.fail.{hash}"),
            "completed"
        )
    );
    finish(f);
}
