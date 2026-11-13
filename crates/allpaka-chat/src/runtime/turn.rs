//! One model turn: provider call, tool loop, guardrail enforcement, hooks
//! and tracing. `turn_traced` drives the step loop; each stage lives in its
//! own module.

mod guard;
mod model;
mod persist;
mod setup;
mod tool_loop;

use crate::*;
use guard::TurnGuardrails;
use setup::TurnContext;

/// Run one turn. `retry_member` names a swarm member to re-run instead of a full
/// wave; only the Swarm executor understands it, and the actor refuses it for
/// every other mode.
pub(crate) async fn turn(
    app: App,
    s: SharedSession,
    retry_member: Option<String>,
) -> Result<TurnOutcome> {
    let (id, project) = {
        let state = s.lock().unwrap();
        (state.id.clone(), state.settings.project_id.clone())
    };
    let trace = app.observability.begin(&id, &project)?;
    let mut span = trace.span("turn", "agent", None);
    let result = turn_traced(app, s, retry_member, &trace, span.id()).await;
    let status = match &result {
        Ok(TurnOutcome::Complete) => "completed",
        Ok(TurnOutcome::NoProgress) => "no_progress",
        Ok(TurnOutcome::GoalIncomplete) => "goal_incomplete",
        Ok(TurnOutcome::TokenLimit) => "token_limit",
        Ok(TurnOutcome::Compacted) => "compacted",
        Err(_) => "failed",
    };
    span.finish(status, &Value::Null);
    result
}

pub(crate) fn enforce_guardrail(
    receipt: Value,
    trace: &observability::Trace,
    parent: usize,
) -> Result<()> {
    let passed = receipt["passed"]
        .as_bool()
        .context("Invalid guardrail outcome")?;
    let blocked = receipt["blocked"]
        .as_bool()
        .context("Invalid guardrail block outcome")?;
    let name = format!(
        "guardrail.{}.{}.{}.{}",
        receipt["stage"].as_str().unwrap(),
        receipt["action"].as_str().unwrap(),
        if passed { "pass" } else { "fail" },
        receipt["policy_sha256"].as_str().unwrap()
    );
    let mut span = trace.span("tool", &name, Some(parent));
    span.finish(
        if blocked { "failed" } else { "completed" },
        &json!({"guardrail_receipt":receipt}),
    );
    if blocked {
        bail!("guardrail_blocked");
    }
    Ok(())
}

/// The borrowed pieces every stage of one turn reads.
struct Turn<'a> {
    app: &'a App,
    s: &'a SharedSession,
    settings: &'a Settings,
    ctx: &'a TurnContext,
    trace: &'a observability::Trace,
    parent_span: usize,
}

/// What the step loop carries from one step to the next.
#[derive(Default)]
struct StepState {
    failure_loop: progress::FailureLoop,
    goal_continuation_revision: Option<u64>,
    stop_check_done: bool,
}

pub(crate) async fn turn_traced(
    app: App,
    s: SharedSession,
    retry_member: Option<String>,
    trace: &observability::Trace,
    parent_span: usize,
) -> Result<TurnOutcome> {
    let mut settings = s.lock().unwrap().settings.clone();
    validate_settings(&settings, &app)?;
    let mut guardrails = TurnGuardrails::prepare(&app, &s, &mut settings, trace, parent_span)?;
    let ctx = TurnContext::prepare(&app, &s, &settings).await?;
    // Swarm is a different executor: it runs its own wave, merge and optional
    // critic pass, and never touches the single-model step loop below.
    if settings.mode == Mode::Swarm {
        let swarm_turn = swarm::SwarmTurn {
            app: &app,
            shared: &s,
            settings: &settings,
            project: &ctx.project,
            system: &ctx.system,
        };
        return match retry_member {
            Some(label) => swarm::retry(swarm_turn, &label, trace, parent_span).await,
            None => swarm::run(swarm_turn, trace, parent_span).await,
        };
    }
    let turn = Turn {
        app: &app,
        s: &s,
        settings: &settings,
        ctx: &ctx,
        trace,
        parent_span,
    };
    let mut state = StepState::default();
    for step in 1..=settings.max_steps {
        compact::run(&app, &s, false, Some((trace, parent_span))).await?;
        let (history, index) = turn.begin_step(step, &mut state.failure_loop);
        for message in &history {
            context::validate_images(&message.images, &settings.provider)?;
        }
        guardrails.check_input(&history, trace, parent_span)?;
        // A budget stops the turn before the call that would start past it.
        if let Some(spent) = settings.budget.exhausted(&s.lock().unwrap().usage_total) {
            bail!("Conversation budget reached ({spent}). Raise the budget to continue.");
        }
        let buffer_output = guardrails.buffers_output();
        let (mut message, usage) = turn.call_model(&history, index, buffer_output).await?;
        if let Err(blocked) = guardrails.check_output(&message.content, trace, parent_span)? {
            turn.discard_reply(index, &usage);
            return Err(blocked);
        }
        message.provider = Some(settings.provider.clone());
        message.model = Some(settings.model.clone());
        let truncated = message.truncated;
        let calls = message.tool_calls.clone();
        let rag_answer = if calls.is_empty()
            && ctx.rag_plugin_id.is_some()
            && ctx.rag_context.as_ref().is_some_and(|c| !c.is_empty())
            && message.content.chars().take(200).count() >= 200
        {
            Some(message.content.clone())
        } else {
            None
        };
        turn.store_reply(index, message, usage);
        if truncated {
            return Ok(TurnOutcome::TokenLimit);
        }
        if calls.is_empty() {
            match turn.finish(&mut state, rag_answer.as_deref()).await {
                Some(outcome) => return Ok(outcome),
                None => continue,
            }
        }
        let no_progress = turn.run_tools(calls, &mut state.failure_loop).await?;
        if no_progress && s.lock().unwrap().steering.is_empty() {
            return Ok(TurnOutcome::NoProgress);
        }
    }
    if settings.mode == Mode::Goal && !plan::reported_ready(&s.lock().unwrap()) {
        return Ok(TurnOutcome::GoalIncomplete);
    }
    bail!(
        "Step limit reached ({}). Review the result and Resume to continue.",
        settings.max_steps
    )
}

impl Turn<'_> {
    /// A reply without tool calls: end the turn, or return `None` to take
    /// another step (steering arrived, Goal mode's one continuation, or the
    /// stop hooks' one pre-completion check).
    async fn finish(&self, state: &mut StepState, rag_answer: Option<&str>) -> Option<TurnOutcome> {
        let (app, s, ctx) = (self.app, self.s, self.ctx);
        if !s.lock().unwrap().steering.is_empty() {
            return None;
        }
        if self.settings.mode == Mode::Goal {
            let mut session = s.lock().unwrap();
            if !plan::reported_ready(&session) {
                if state.goal_continuation_revision == Some(session.plan_revision) {
                    return Some(TurnOutcome::GoalIncomplete);
                }
                state.goal_continuation_revision = Some(session.plan_revision);
                session.messages.push(Message::text("user", "Goal continuation: the persisted milestone plan is incomplete. Continue the remaining work with available tools and update set_plan using the current revision. Do not repeat completed actions. If blocked, explain the blocker; do not claim the goal is complete. Completion evidence is reported, not independently verified."));
                drop(session);
                persist(app, s);
                return None;
            }
        }
        if !state.stop_check_done && !ctx.harness_hooks.stop.is_empty() {
            state.stop_check_done = true;
            let notes =
                hook_notes(&ctx.project, &ctx.harness_hooks.stop, None, ctx.cmd_enabled).await;
            if !notes.is_empty() {
                self.push_message(Message::text("user", format!("Pre-completion check from the harness:\n{notes}\nIf the result already satisfies it, restate the final answer; otherwise fix what is missing first. Do not repeat completed side effects.")));
                return None;
            }
        }
        if let (Some(rag_id), Some(answer)) = (&ctx.rag_plugin_id, rag_answer) {
            let title = last_user_text(&s.lock().unwrap())
                .map(|text| text.chars().take(80).collect::<String>())
                .unwrap_or_else(|| "allpaka answer".into());
            rag::auto_file_answer(app, rag_id, &title, answer).await;
        }
        Some(TurnOutcome::Complete)
    }
}

/// The text of the latest user message, if there is one.
fn last_user_text(session: &Session) -> Option<&str> {
    session
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
}

/// Run a post_tool `transform`: the tool result goes to stdin and stdout
/// replaces it. A non-zero exit or a timeout keeps the original result.
pub(crate) async fn transform_output(
    cwd: &std::path::Path,
    program: &str,
    tool: &str,
    args: &str,
    input: &str,
) -> Result<String> {
    use tokio::io::AsyncWriteExt;
    let mut child = crate::shell::command(program)
        .current_dir(cwd)
        .env("ALLPAKA_TOOL", tool)
        .env("ALLPAKA_TOOL_ARGS", args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("Failed to start transform")?;
    let mut stdin = child.stdin.take().context("No stdin")?;
    let data = input.as_bytes().to_vec();
    let writer = tokio::spawn(async move {
        let _ = stdin.write_all(&data).await;
    });
    let output = tokio::time::timeout(std::time::Duration::from_secs(30), child.wait_with_output())
        .await
        .context("Transform timed out")??;
    let _ = writer.await;
    if !output.status.success() {
        bail!(
            "exit {:?}: {}",
            output.status.code(),
            truncate_output(&output.stderr)
        );
    }
    Ok(truncate_output(&output.stdout))
}

/// The reason a pre_tool hook refuses this call: a `block` text, or the
/// output of a `command` that exits non-zero. Commands only run when the
/// command plugin is enabled; the call's JSON arguments are in
/// `ALLPAKA_TOOL_ARGS` and its name in `ALLPAKA_TOOL`.
pub(crate) async fn pre_tool_block(
    project: &context::Project,
    hooks: &[harness::Hook],
    tool: &str,
    args: &str,
    cmd_enabled: bool,
) -> Option<String> {
    for hook in hooks.iter().filter(|h| h.matches(tool, args)) {
        if let Some(reason) = &hook.block {
            return Some(reason.trim().to_string());
        }
        if let (Some(command), true, Some(root)) =
            (&hook.command, cmd_enabled, project.roots.first())
        {
            let quote = |s: &str| format!("'{}'", s.replace('\'', "'\\''"));
            let command = format!(
                "ALLPAKA_TOOL={} ALLPAKA_TOOL_ARGS={} /bin/sh -c {}",
                quote(tool),
                quote(args),
                quote(command)
            );
            match execute_command(&root.path, &json!({"command":command,"timeout":30})).await {
                Ok(out) if out["exit_code"] == json!(0) => {}
                Ok(out) => {
                    let text = format!(
                        "{} {}",
                        out["stdout"].as_str().unwrap_or(""),
                        out["stderr"].as_str().unwrap_or("")
                    );
                    return Some(text.trim().chars().take(2000).collect());
                }
                Err(e) => return Some(format!("hook command failed: {e}")),
            }
        }
    }
    None
}

/// Text of the hooks that apply: `inject` verbatim, `command` stdout when the
/// command plugin is enabled. `tool` (name and raw arguments) is None for
/// stop hooks.
pub(crate) async fn hook_notes(
    project: &context::Project,
    hooks: &[harness::Hook],
    tool: Option<(&str, &str)>,
    cmd_enabled: bool,
) -> String {
    let mut notes = vec![];
    for hook in hooks
        .iter()
        .filter(|h| tool.is_none_or(|(name, args)| h.matches(name, args)))
    {
        if let Some(text) = &hook.inject {
            notes.push(text.trim().to_string());
        } else if let (Some(command), true, Some(root)) =
            (&hook.command, cmd_enabled, project.roots.first())
        {
            match execute_command(&root.path, &json!({"command":command,"timeout":30})).await {
                Ok(out) => {
                    let text: String = out["stdout"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(4000)
                        .collect();
                    if !text.is_empty() {
                        notes.push(text);
                    }
                }
                Err(e) => notes.push(format!("hook command failed: {e}")),
            }
        }
    }
    notes.join("\n")
}
