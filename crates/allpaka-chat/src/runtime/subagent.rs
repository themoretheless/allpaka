//! Nested headless turns for delegated subagents.

use crate::*;

/// Boxed so that a turn can contain a (subagent) turn.
pub(crate) fn nested_turn<'a>(
    app: App,
    s: SharedSession,
    trace: &'a observability::Trace,
    parent_span: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TurnOutcome>> + Send + 'a>> {
    Box::pin(turn_traced(app, s, None, trace, parent_span))
}

/// What a subagent inherits from the turn that delegated to it.
pub(crate) struct SubagentParent<'a> {
    pub(crate) settings: &'a Settings,
    pub(crate) project: &'a context::Project,
    pub(crate) tree: &'a harness::Tree,
    pub(crate) policy: ExecutionPolicy,
}

/// Run one harness subagent as a nested headless turn and return its answer.
pub(crate) async fn run_subagent(
    app: &App,
    parent: SubagentParent<'_>,
    args: &Value,
    trace: &observability::Trace,
    parent_span: usize,
) -> Result<Value> {
    let SubagentParent {
        settings,
        project,
        tree,
        policy,
    } = parent;
    let name = args["agent"].as_str().context("agent is required")?;
    let task = args["task"].as_str().context("task is required")?;
    if task.trim().is_empty() || task.len() > 16000 {
        bail!("task must be 1–16000 bytes");
    }
    let agent = harness::subagents(tree)
        .into_iter()
        .find(|a| a.name == name)
        .with_context(|| format!("Unknown subagent `{name}`"))?;
    let mut child = settings.clone();
    child.max_steps = (settings.max_steps / 2).max(2);
    child.allow_writes &= agent.writes;
    if child.mode == Mode::Goal {
        child.mode = Mode::Auto;
    }
    let mut session = Session::new(evaluation::new_id(), child);
    session.ephemeral = Some(Arc::new(Ephemeral {
        project: project.clone(),
        harness: Some(tree.clone()),
        system: Some(format!("You are the `{}` subagent, working on a task handed to you by another agent. Finish it and reply with the final result only; your reply is returned to that agent.\n{}", agent.name, agent.prompt.trim())),
        nested: true,
        policy,
    }));
    session.messages.push(Message::text("user", task));
    let shared: SharedSession = Arc::new(Mutex::new(session));
    let mut span = trace.span("agent", &format!("subagent.{name}"), Some(parent_span));
    let outcome = nested_turn(app.clone(), shared.clone(), trace, span.id()).await;
    span.finish(
        if outcome.is_ok() {
            "completed"
        } else {
            "failed"
        },
        &Value::Null,
    );
    let state = shared.lock().unwrap();
    let answer = state
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
        .map(|m| m.content.clone())
        .unwrap_or_default();
    Ok(json!({
        "agent": name,
        "outcome": match &outcome { Ok(o) => format!("{o:?}"), Err(e) => format!("error: {e:#}") },
        "answer": answer,
    }))
}
