//! The tool-loop stage of a turn step: run the calls of one reply in order,
//! record each result, and stop early on steering or repeated failure.

use super::Turn;
use crate::*;

impl Turn<'_> {
    /// Run every call of one reply. Returns whether repeated failure paused
    /// the turn; calls after that point, or after steering arrives, are
    /// recorded as cancelled instead of run.
    pub(super) async fn run_tools(
        &self,
        calls: Vec<Value>,
        failure_loop: &mut progress::FailureLoop,
    ) -> Result<bool> {
        let mut no_progress = false;
        for call in calls {
            let mut tool_span = self.trace.span(
                "tool",
                call["function"]["name"].as_str().unwrap_or("unknown"),
                Some(self.parent_span),
            );
            // A steering instruction invalidates the unexecuted remainder of this batch.
            let steered = !self.s.lock().unwrap().steering.is_empty();
            let result = if steered {
                Ok(
                    json!({"cancelled":"User steering arrived; re-evaluate this tool after reading it"}),
                )
            } else if no_progress {
                Ok(
                    json!({"cancelled":"Repeated tool failure paused this turn; this call was not executed"}),
                )
            } else {
                self.dispatch_tool(&call).await?
            };
            tool_span.finish(
                if steered || no_progress {
                    "cancelled"
                } else if result.is_ok() {
                    "completed"
                } else {
                    "failed"
                },
                &Value::Null,
            );
            if steered {
                failure_loop.reset();
            } else if !no_progress {
                let error = result.as_ref().err().map(ToString::to_string);
                no_progress = failure_loop.observe(&call, error.as_deref());
            }
            let value = match result {
                Ok(v) => v,
                Err(e) => json!({"error":e.to_string()}),
            };
            let mut content = value.to_string();
            if !steered && !self.ctx.harness_hooks.post_tool.is_empty() {
                content = self.post_tool_hooks(&call, content).await;
            }
            self.push_message(Message {
                role: "tool".into(),
                content,
                tool_call_id: call["id"].as_str().map(str::to_owned),
                ..Message::default()
            });
            tokio::task::yield_now().await;
        }
        Ok(no_progress)
    }

    /// Run one call. The outer error ends the turn (a malformed call or a
    /// missing project root); the inner one is the tool's result.
    async fn dispatch_tool(&self, call: &Value) -> Result<Result<Value>> {
        let (app, s, settings) = (self.app, self.s, self.settings);
        let ctx = self.ctx;
        let project = &ctx.project;
        let harness_tree = &ctx.harness_tree;
        let (cmd_enabled, harness_dynamic, nested, policy, restricted, scripts_runnable) = (
            ctx.cmd_enabled,
            ctx.harness_dynamic,
            ctx.nested,
            ctx.policy,
            ctx.restricted,
            ctx.scripts_runnable,
        );
        let name = call["function"]["name"]
            .as_str()
            .context("Missing tool name")?;
        let raw_args = call["function"]["arguments"]
            .as_str()
            .context("Missing tool arguments")?;
        let args: Value = serde_json::from_str(raw_args)?;
        let blocked = pre_tool_block(
            project,
            &ctx.harness_hooks.pre_tool,
            name,
            raw_args,
            cmd_enabled,
        )
        .await;
        Ok(if let Some(reason) = blocked {
            Err(anyhow::anyhow!("Blocked by a harness hook: {reason}"))
        } else if name == "background" {
            if !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
                || ctx.ephemeral.is_some()
            {
                Err(anyhow::anyhow!(
                    "Background commands require a non-headless Auto/Goal turn with command execution enabled"
                ))
            } else {
                let root = project.roots.first().context("No project root available")?;
                let owner = s.lock().unwrap().id.clone();
                execute_background(app, &owner, &root.path, &args).await
            }
        } else if name == "load_skill" && harness_dynamic {
            harness::load_skill(harness_tree, args["name"].as_str().unwrap_or(""))
        } else if name == "delegate" && harness_dynamic && !nested {
            let parent = SubagentParent {
                settings,
                project,
                tree: harness_tree,
                policy,
            };
            run_subagent(app, parent, &args, self.trace, self.parent_span).await
        } else if name == "run_script" && scripts_runnable {
            let root = project
                .roots
                .first()
                .map(|r| r.path.as_path())
                .context("No project root available")?;
            let script_args: Vec<String> =
                serde_json::from_value(args.get("args").cloned().unwrap_or(json!([])))
                    .context("args must be an array of strings")?;
            match harness::materialize_script(&app.data, harness_tree, args["path"].as_str().unwrap_or("")) {
                Ok(path) => execute_command(root, &json!({"command":harness::script_command(&path,&script_args),"timeout":args.get("timeout").cloned().unwrap_or(json!(30))})).await,
                Err(e) => Err(e),
            }
        } else if name == "memory_recall" {
            app.memory.recall(&settings.project_id, &args)
        } else if name == "conversation_search" {
            let session = s.lock().unwrap();
            conversation_search::search(
                &session.messages,
                session.compaction.as_ref().map_or(0, |c| c.through),
                &args,
            )
        } else if name == "set_plan" && settings.mode != Mode::Chat {
            tools::parse_plan(&args).and_then(|steps| {
                let base = args
                    .get("base_revision")
                    .map(|value| value.as_u64().context("Invalid plan base_revision"))
                    .transpose()?;
                let mut state = s.lock().unwrap();
                let prev_plan = state.plan.clone();
                let prev_rev = state.plan_revision;
                plan::update(&mut state, steps, base, "agent", false)?;
                if let Err(err) = save(app, &state) {
                    state.plan = prev_plan;
                    state.plan_revision = prev_rev;
                    return Err(err);
                }
                Ok(json!({"updated":true,"revision":state.plan_revision,"steps":state.plan,"evidence_basis":"reported"}))
            })
        } else if name == "list_files" && args["path"] == "." {
            Ok(
                json!({"contexts":project.roots.iter().map(|r|json!({"alias":r.alias,"writable":r.writable,"repository":r.repository})).collect::<Vec<_>>()}),
            )
        } else if let Some((client, tool)) = mcp_lookup(app, name).filter(|_| !restricted) {
            client.call(&tool, &args).await
        } else if name == "run_command" && cmd_enabled {
            let root = project
                .roots
                .first()
                .map(|r| r.path.as_path())
                .context("No project root available")?;
            execute_command(root, &args).await
        } else {
            project
                .root(args["path"].as_str().unwrap_or(""), tools::writes(name))
                .and_then(|(root, path)| {
                    let candidate = root.join(&path);
                    let canonical = candidate.canonicalize().unwrap_or(candidate);
                    if canonical.starts_with(app.data.as_ref()) {
                        bail!("Conversation storage is not available to tools");
                    }
                    let mut args = args.clone();
                    args["path"] = json!(path);
                    tools::execute_scoped(root, settings, name, &args, Some(app.data.as_ref()))
                })
        })
    }

    /// Apply the matching post_tool hooks to a result: transforms (only with
    /// command execution), then the output limit, then hook notes.
    async fn post_tool_hooks(&self, call: &Value, mut content: String) -> String {
        let (project, hooks, cmd_enabled) = (
            &self.ctx.project,
            &self.ctx.harness_hooks,
            self.ctx.cmd_enabled,
        );
        let tool = call["function"]["name"].as_str().unwrap_or("");
        let raw_args = call["function"]["arguments"].as_str().unwrap_or("");
        let limit = hooks
            .post_tool
            .iter()
            .filter(|h| h.matches(tool, raw_args))
            .filter_map(|h| h.max_output)
            .min();
        if cmd_enabled {
            let transforms = hooks
                .post_tool
                .iter()
                .filter(|h| h.matches(tool, raw_args))
                .filter_map(|h| h.transform.as_ref());
            for program in transforms {
                let Some(root) = project.roots.first() else {
                    break;
                };
                content =
                    match transform_output(&root.path, program, tool, raw_args, &content).await {
                        Ok(text) => text,
                        Err(e) => {
                            format!("{content}\n\n[harness hook] transform failed: {e}")
                        }
                    };
            }
        }
        if let Some(limit) = limit {
            content = harness::trim_output(&content, limit);
        }
        let notes = hook_notes(
            project,
            &hooks.post_tool,
            Some((tool, raw_args)),
            cmd_enabled,
        )
        .await;
        if !notes.is_empty() {
            content.push_str(&format!("\n\n[harness hook] {notes}"));
        }
        content
    }
}
