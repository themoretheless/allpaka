//! Dispatch for extended Studio parity tools (spill, interactive questions,
//! patch/glob/cwd, LSP, terminals, worktrees, forge PRs, collaborative
//! subagents, agent teams, PTC, workflows, web/documents, browser/computer
//! seams, MCP resources, plugin manager, and schedules).

use super::Turn;
use crate::*;

impl Turn<'_> {
    /// Handle `name` if it is one of the extended parity tools.
    /// Outer `Err` terminates the turn (e.g. missing project root);
    /// `Ok(Some(res))` returns the tool call's result; `Ok(None)` falls through.
    pub(super) async fn try_dispatch_parity(
        &self,
        name: &str,
        args: &Value,
        session_id: &str,
        effective_sandbox: crate::sandbox::SandboxMode,
        writable_roots: &[PathBuf],
    ) -> Result<Option<Result<Value>>> {
        let (app, s, settings, ctx) = (self.app, self.s, self.settings, self.ctx);
        let project = &ctx.project;
        let (cmd_enabled, nested, policy, restricted) =
            (ctx.cmd_enabled, ctx.nested, ctx.policy, ctx.restricted);

        if name == "read_spill" {
            if ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Spill inspection is unavailable in headless turns"
                ))));
            }
            return Ok(Some(spill::execute_read_tool(&app.data, session_id, args)));
        }
        if name == "ask_user_question" {
            if ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Interactive questions require a non-headless session"
                ))));
            }
            return Ok(Some(interaction::ask_user_question(app, s, args).await));
        }
        if name == "glob_files" {
            if restricted {
                return Ok(Some(Err(anyhow::anyhow!(
                    "glob_files is unavailable in restricted sessions"
                ))));
            }
            let wd = s.lock().unwrap().working_dir.clone();
            return Ok(Some(crate::patch::glob_files(
                project,
                wd.as_deref(),
                app.data.as_ref(),
                args,
            )));
        }
        if name == "working_directory" {
            if restricted || ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "working_directory requires a non-ephemeral session"
                ))));
            }
            return Ok(Some(crate::patch::working_directory_tool(
                app, s, project, args,
            )));
        }
        if name == "apply_patch" {
            if restricted || ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "apply_patch requires a non-ephemeral session"
                ))));
            }
            let allow = !settings.mode.is_read_only()
                && settings.allow_writes
                && project.roots.iter().any(|r| r.writable);
            let wd = s.lock().unwrap().working_dir.clone();
            return Ok(Some(crate::patch::apply_patch(
                project,
                wd.as_deref(),
                app.data.as_ref(),
                &app.ext.checkpoints,
                Some(session_id),
                allow,
                args,
            )));
        }
        if name == "lsp" {
            if restricted || ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "lsp requires a non-ephemeral session"
                ))));
            }
            return Ok(Some(
                app.ext.lsp.execute(project, app.data.as_ref(), args).await,
            ));
        }
        if matches!(
            name,
            "terminal_open"
                | "terminal_send"
                | "terminal_read"
                | "terminal_signal"
                | "terminal_close"
                | "terminal_list"
                | "user_terminal_list"
                | "user_terminal_read"
        ) {
            if restricted || ctx.ephemeral.is_some() {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Terminal tools require a non-ephemeral session"
                ))));
            }
            if !matches!(name, "user_terminal_list" | "user_terminal_read")
                && (!matches!(settings.mode, Mode::Auto | Mode::Goal) || !cmd_enabled)
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Model terminals require an Auto/Goal turn with command execution enabled"
                ))));
            }
            let default_cwd = self.default_cwd()?;
            return Ok(Some(
                crate::terminal::execute_tool(
                    app,
                    session_id,
                    &default_cwd,
                    effective_sandbox,
                    writable_roots,
                    name,
                    args,
                )
                .await,
            ));
        }
        if name == "create_worktree" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "create_worktree requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let allow = settings.allow_writes && project.roots.iter().any(|r| r.writable);
            return Ok(Some(crate::worktree::execute(
                app, session_id, s, project, allow, args,
            )));
        }
        if name == "forge_pr" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "forge_pr requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let allow = settings.allow_writes && project.roots.iter().any(|r| r.writable);
            return Ok(Some(
                crate::forge::execute(app, s, project, allow, args).await,
            ));
        }
        if matches!(
            name,
            "subagent_spawn"
                | "list_agents"
                | "send_agent_message"
                | "wait_agent"
                | "interrupt_agent"
        ) {
            if restricted || ctx.ephemeral.is_some() || nested {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Collaborative subagent tree tools require a non-ephemeral root session"
                ))));
            }
            if !matches!(settings.mode, Mode::Auto | Mode::Goal) || !cmd_enabled {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Subagent spawning requires an Auto/Goal turn with command execution enabled"
                ))));
            }
            let res = match name {
                "subagent_spawn" => {
                    let parent = SubagentParent {
                        settings,
                        project,
                        tree: &ctx.harness_tree,
                        policy,
                    };
                    app.ext
                        .agents
                        .spawn(app.clone(), session_id, s, parent, args)
                        .await
                }
                "list_agents" => Ok(json!({ "agents": app.ext.agents.list(session_id) })),
                "send_agent_message" => (|| {
                    let aid = args["agent_id"].as_str().context("agent_id is required")?;
                    let msg = args["message"].as_str().context("message is required")?;
                    let wake = args.get("wake").and_then(Value::as_bool).unwrap_or(true);
                    app.ext.agents.send_message(app, session_id, aid, msg, wake)
                })(),
                "wait_agent" => match args["agent_id"].as_str().context("agent_id is required") {
                    Ok(aid) => {
                        let ms = args
                            .get("timeout_ms")
                            .and_then(Value::as_u64)
                            .unwrap_or(5000);
                        app.ext.agents.wait(session_id, aid, ms).await
                    }
                    Err(e) => Err(e),
                },
                "interrupt_agent" => (|| {
                    let aid = args["agent_id"].as_str().context("agent_id is required")?;
                    app.ext.agents.interrupt(session_id, aid)
                })(),
                _ => unreachable!(),
            };
            return Ok(Some(res));
        }
        if matches!(
            name,
            "team_task_create" | "team_task_list" | "team_task_get" | "team_task_update"
        ) {
            if restricted || ctx.ephemeral.is_some() || !settings.team_mode {
                return Ok(Some(Err(anyhow::anyhow!(
                    "Agent Team tools require a non-ephemeral session with team_mode enabled"
                ))));
            }
            return Ok(Some(crate::agent_team::execute_tool(app, s, name, args)));
        }
        if name == "run_code" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "run_code requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let cwd = self.default_cwd()?;
            return Ok(Some(
                crate::ptc::execute(
                    app,
                    s,
                    project,
                    settings,
                    effective_sandbox,
                    writable_roots,
                    &cwd,
                    args,
                )
                .await,
            ));
        }
        if matches!(name, "ralph" | "workflow") {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "{name} requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let cwd = self.default_cwd()?;
            let res = if name == "ralph" {
                crate::workflow::execute_ralph(
                    app,
                    session_id,
                    settings,
                    project,
                    &ctx.harness_tree,
                    policy,
                    effective_sandbox,
                    writable_roots,
                    &cwd,
                    args,
                )
                .await
            } else {
                crate::workflow::execute_workflow(
                    app,
                    session_id,
                    settings,
                    project,
                    &ctx.harness_tree,
                    policy,
                    effective_sandbox,
                    writable_roots,
                    &cwd,
                    args,
                )
                .await
            };
            return Ok(Some(res));
        }
        if name == "web_fetch" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Plan | Mode::Auto | Mode::Goal)
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "web_fetch requires a non-ephemeral Plan/Auto/Goal turn"
                ))));
            }
            return Ok(Some(
                crate::web_tools::execute_web_fetch(
                    Some(app.data.as_ref()),
                    Some(session_id),
                    args,
                )
                .await,
            ));
        }
        if name == "web_search" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Plan | Mode::Auto | Mode::Goal)
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "web_search requires a non-ephemeral Plan/Auto/Goal turn"
                ))));
            }
            return Ok(Some(crate::web_tools::execute_web_search(args).await));
        }
        if name == "extract_document" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Plan | Mode::Auto | Mode::Goal)
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "extract_document requires a non-ephemeral Plan/Auto/Goal turn"
                ))));
            }
            return Ok(Some(
                crate::documents::execute_tool(app.data.as_ref(), session_id, project, args).await,
            ));
        }
        if matches!(name, "browser_use" | "computer_use") {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "{name} requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let base_cwd = project.roots.first().map(|r| r.path.as_path());
            let res = if name == "browser_use" {
                crate::automation_seams::execute_browser_with_cmd(
                    Some(app.data.as_ref()),
                    session_id,
                    base_cwd,
                    args,
                    None,
                )
                .await
            } else {
                crate::automation_seams::execute_computer_with_cmd(
                    Some(app.data.as_ref()),
                    session_id,
                    base_cwd,
                    args,
                    None,
                )
                .await
            };
            return Ok(Some(res));
        }
        if matches!(
            name,
            "list_mcp_resources" | "list_mcp_resource_templates" | "read_mcp_resource"
        ) {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Plan | Mode::Auto | Mode::Goal)
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "{name} requires a non-ephemeral Plan/Auto/Goal turn"
                ))));
            }
            return Ok(Some(
                crate::mcp::execute_resource_tool(app, name, args).await,
            ));
        }
        if name == "plugin_manager" {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "plugin_manager requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            return Ok(Some(crate::plugins::execute_manager(app, args)));
        }
        if matches!(
            name,
            "schedule_create" | "schedule_edit" | "schedule_list" | "schedule_view"
        ) {
            if restricted
                || ctx.ephemeral.is_some()
                || !matches!(settings.mode, Mode::Auto | Mode::Goal)
                || !cmd_enabled
            {
                return Ok(Some(Err(anyhow::anyhow!(
                    "{name} requires a non-ephemeral Auto/Goal turn with command execution enabled"
                ))));
            }
            let scheduled_origin = s.lock().unwrap().scheduled_origin;
            return Ok(Some(crate::schedule::execute_tool(
                app,
                settings,
                scheduled_origin,
                name,
                args,
            )));
        }
        Ok(None)
    }

    pub(super) fn default_cwd(&self) -> Result<PathBuf> {
        let base_root = self
            .ctx
            .project
            .roots
            .first()
            .map(|r| r.path.clone())
            .context("No project root available")?;
        let wd = self.s.lock().unwrap().working_dir.clone();
        Ok(match wd.as_deref() {
            Some(w) => crate::ext::resolve_project_root(&self.ctx.project, w, false)
                .and_then(|(r, rel)| tools::resolve(&r.path, &rel, false))
                .unwrap_or(base_root),
            None => base_root,
        })
    }
}
