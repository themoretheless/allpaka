use super::*;

// ---- Studio engine -----------------------------------------------------------------

pub(super) struct StudioEngine {
    pub(super) app: App,
    pub(super) executor: Settings,
    pub(super) proposer: Settings,
    /// The proposer as an Auto-mode agent with writes, for agentic stages.
    pub(super) proposer_agent: Settings,
    pub(super) proposer_provider: crate::provider::Provider,
    pub(super) project: context::Project,
    pub(super) workspace: WorkspaceSnapshot,
    pub(super) task: String,
    pub(super) scratch: PathBuf,
    pub(super) external: Option<External>,
    pub(super) execution: crate::ExecutionPolicy,
    pub(super) rollout_timeout: std::time::Duration,
}

#[derive(Clone, Debug)]
pub(super) struct WorkspaceSnapshot {
    pub(super) roots: Vec<SnapshotRoot>,
    pub(super) fingerprint: String,
}
#[derive(Clone, Debug)]
pub(super) struct SnapshotRoot {
    pub(super) alias: String,
    pub(super) original: PathBuf,
    pub(super) snapshot: PathBuf,
    pub(super) writable: bool,
}
impl WorkspaceSnapshot {
    /// Capture writable roots once per task. The hash depends only on stable
    /// root/file identities and bytes, never on directory iteration order.
    pub(super) fn capture(project: &context::Project, directory: &std::path::Path) -> Result<Self> {
        use sha2::{Digest, Sha256};

        if directory.exists() {
            std::fs::remove_dir_all(directory)?;
        }
        std::fs::create_dir_all(directory)?;
        let mut roots = vec![];
        let mut hasher = Sha256::new();
        let (mut files, mut bytes) = (0, 0);
        for root in &project.roots {
            hasher.update((root.alias.len() as u64).to_le_bytes());
            hasher.update(root.alias.as_bytes());
            hasher.update([u8::from(root.writable)]);
            let snapshot = if root.writable {
                let destination = directory.join(&root.alias);
                copy_root_observed(
                    &root.path,
                    &destination,
                    &mut files,
                    &mut bytes,
                    "",
                    &mut |path, content| {
                        hasher.update((path.len() as u64).to_le_bytes());
                        hasher.update(path.as_bytes());
                        hasher.update((content.len() as u64).to_le_bytes());
                        hasher.update(content);
                    },
                )?;
                destination.canonicalize()?
            } else {
                root.path.clone()
            };
            roots.push(SnapshotRoot {
                alias: root.alias.clone(),
                original: root.path.clone(),
                snapshot,
                writable: root.writable,
            });
        }
        Ok(Self {
            roots,
            fingerprint: crate::evaluation::hex_sha256(hasher),
        })
    }

    /// Writable roots resolve to the snapshot; other roots remain where they
    /// are, but callers can downgrade every root to read-only for inspection.
    fn project(&self, template: &context::Project) -> context::Project {
        let mut project = template.clone();
        for root in &mut project.roots {
            if let Some(saved) = self.roots.iter().find(|saved| saved.alias == root.alias) {
                root.path = if saved.writable {
                    saved.snapshot.clone()
                } else {
                    saved.original.clone()
                };
            }
        }
        project
    }
}

impl StudioEngine {
    fn policy_warnings(&self, tree: &Tree) -> Result<Vec<String>> {
        if self.execution.commands != crate::CommandExecution::Disabled {
            return Ok(vec![]);
        }
        let hooks = harness::hooks(tree)?;
        let command_hooks = [&hooks.pre_tool, &hooks.post_tool, &hooks.stop]
            .into_iter()
            .flatten()
            .any(|hook| {
                hook.command.as_ref().is_some_and(|s| !s.trim().is_empty())
                    || hook
                        .transform
                        .as_ref()
                        .is_some_and(|s| !s.trim().is_empty())
            });
        if harness::scripts(tree).is_empty() && !command_hooks {
            Ok(vec![])
        } else {
            Ok(vec![
                "Command scripts, command hooks and transforms were disabled by the files-only execution policy.".into(),
            ])
        }
    }
}

pub(super) const COPY_MAX_FILES: usize = 20_000;
pub(super) const COPY_MAX_BYTES: u64 = 256 * 1024 * 1024;
pub(super) fn skipped(name: &str) -> bool {
    name.starts_with('.') || name == "target" || name == "node_modules"
}
pub(super) fn copy_root(
    from: &std::path::Path,
    to: &std::path::Path,
    files: &mut usize,
    bytes: &mut u64,
) -> Result<()> {
    copy_root_observed(from, to, files, bytes, "", &mut |_, _| {})
}
pub(super) fn copy_root_observed(
    from: &std::path::Path,
    to: &std::path::Path,
    files: &mut usize,
    bytes: &mut u64,
    prefix: &str,
    observe: &mut impl FnMut(&str, &[u8]),
) -> Result<()> {
    std::fs::create_dir_all(to)?;
    let mut entries = std::fs::read_dir(from)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let kind = entry.file_type()?;
        if skipped(&name) || kind.is_symlink() {
            continue;
        }
        let relative = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if kind.is_dir() {
            copy_root_observed(
                &entry.path(),
                &to.join(&name),
                files,
                bytes,
                &relative,
                observe,
            )?;
        } else if kind.is_file() {
            let length = entry.metadata()?.len();
            *files += 1;
            *bytes = bytes.saturating_add(length);
            if *files > COPY_MAX_FILES || *bytes > COPY_MAX_BYTES {
                bail!("Writable context is too large to copy for a probe (over {COPY_MAX_FILES} files or 256 MiB)");
            }
            let content = std::fs::read(entry.path())?;
            observe(&relative, &content);
            std::fs::write(to.join(&name), content)?;
        }
    }
    Ok(())
}
pub(super) fn files_under(root: &std::path::Path) -> BTreeMap<String, PathBuf> {
    fn walk(dir: &std::path::Path, prefix: &str, out: &mut BTreeMap<String, PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if skipped(&name) || kind.is_symlink() {
                continue;
            }
            let rel = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            if kind.is_dir() {
                walk(&entry.path(), &rel, out);
            } else if kind.is_file() {
                out.insert(rel, entry.path());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, "", &mut out);
    out
}
/// What a probe changed in its copy, against the task's frozen snapshot.
pub(super) fn changes(alias: &str, snapshot: &std::path::Path, copy: &std::path::Path) -> String {
    changes_over(alias, snapshot, copy, &BTreeMap::new())
}
/// Like `changes`, but files the harness materialized into the copy count
/// as already present with that content.
pub(super) fn changes_over(
    alias: &str,
    snapshot: &std::path::Path,
    copy: &std::path::Path,
    overlay: &BTreeMap<String, String>,
) -> String {
    let before = files_under(snapshot);
    let after = files_under(copy);
    let mut out = String::new();
    for (rel, path) in &after {
        let new = std::fs::read(path).unwrap_or_default();
        let old = match overlay.get(rel) {
            Some(text) => Some(text.as_bytes().to_vec()),
            None => before
                .get(rel)
                .map(|p| std::fs::read(p).unwrap_or_default()),
        };
        match old {
            Some(old) if old == new => {}
            old => {
                let header = if old.is_some() { "modified" } else { "added" };
                out.push_str(&format!("*** {alias}/{rel} ({header})\n"));
                match (
                    std::str::from_utf8(old.as_deref().unwrap_or(&[])),
                    std::str::from_utf8(&new),
                ) {
                    (Ok(a), Ok(b)) if a.len() + b.len() < 512 * 1024 => {
                        out.push_str(&harness::line_diff(a, b))
                    }
                    _ => out.push_str("(binary or large file)\n"),
                }
            }
        }
        if out.len() > 64_000 {
            out.push_str("… [more changes omitted]\n");
            return out;
        }
    }
    for rel in before
        .keys()
        // Hidden layout directories (.agents, .opencode) are never walked.
        .chain(overlay.keys().filter(|rel| !rel.split('/').any(skipped)))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|rel| !after.contains_key(*rel))
    {
        out.push_str(&format!("*** {alias}/{rel} (removed)\n"));
    }
    out
}

/// An external coding agent (Codex, OpenCode or anything with a CLI) that
/// executes rollouts instead of Studio's own loop, as the Harness²
/// substrates do. The command runs with `/bin/sh -c` in the copy of the
/// first writable root, receives the task on stdin and in `ALLPAKA_TASK`,
/// and its stdout is the answer.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct External {
    pub command: String,
    #[serde(default)]
    pub layout: Layout,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// `AGENTS.md` plus `.agents/skills/<name>/SKILL.md` (Codex).
    #[default]
    Codex,
    /// `AGENTS.md`, `.opencode/skills/<name>/SKILL.md` and
    /// `.opencode/agent/<name>.md` subagents (OpenCode).
    Opencode,
}

/// The files a harness becomes in an external agent's workspace. `existing`
/// is the workspace's own AGENTS.md, which the harness extends.
pub fn materialize(
    tree: &Tree,
    layout: Layout,
    existing: Option<&str>,
) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let skill_dir = match layout {
        Layout::Codex => ".agents/skills",
        Layout::Opencode => ".opencode/skills",
    };
    let mut parts = vec![];
    let base = harness::compose_system(tree, false, false);
    if !base.is_empty() {
        parts.push(base);
    }
    let skills = harness::skills(tree);
    if !skills.is_empty() {
        parts.push(format!(
            "# Available skills (read the file before following one)\n{}",
            skills
                .iter()
                .map(|s| format!(
                    "- {}: {} ({skill_dir}/{}/SKILL.md)",
                    s.name, s.description, s.name
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    let agents = harness::subagents(tree);
    match layout {
        Layout::Opencode if !agents.is_empty() => {
            parts.push(format!(
                "# Available subagents\n{}",
                agents
                    .iter()
                    .map(|a| format!("- {}: {}", a.name, a.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
            for a in &agents {
                files.insert(
                    format!(".opencode/agent/{}.md", a.name),
                    format!(
                        "---\ndescription: {}\nmode: subagent\n---\n{}",
                        a.description,
                        a.prompt.trim()
                    ),
                );
            }
        }
        // Codex has no file-defined subagents: their prompts become playbooks.
        Layout::Codex if !agents.is_empty() => parts.push(format!(
            "# Delegation playbooks\n{}",
            agents
                .iter()
                .map(|a| format!("## {} — {}\n{}", a.name, a.description, a.prompt.trim()))
                .collect::<Vec<_>>()
                .join("\n\n")
        )),
        _ => {}
    }
    let scripts = harness::scripts(tree);
    if !scripts.is_empty() {
        parts.push(format!(
            "# Helper scripts (run from the workspace root with sh or python3)\n{}",
            scripts
                .iter()
                .map(|s| format!("- {}: {}", external_path(&s.name, skill_dir), s.description))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    for (path, text) in tree {
        if matches!(
            harness::classify(path),
            Some(harness::Kind::Skill(_) | harness::Kind::SkillScript(_) | harness::Kind::Script)
        ) {
            files.insert(external_path(path, skill_dir), text.clone());
        }
    }
    if !parts.is_empty() {
        let text = parts.join("\n\n");
        let text = match existing {
            Some(own) if !own.trim().is_empty() => format!("{}\n\n{text}\n", own.trim_end()),
            _ => format!("{text}\n"),
        };
        files.insert("AGENTS.md".into(), text);
    }
    files
}
pub(super) fn external_path(path: &str, skill_dir: &str) -> String {
    match path.strip_prefix("skills/") {
        Some(rest) => format!("{skill_dir}/{rest}"),
        None => path.to_string(),
    }
}
/// The [ACTION]/[OBSERVATION] grammar the proposers read.
pub(crate) fn trajectory(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "user" => out.push_str(&format!("[USER] {}\n", clip(&m.content, 4_000))),
            "assistant" => {
                if let Some(r) = m
                    .reasoning_content
                    .as_deref()
                    .filter(|r| !r.trim().is_empty())
                {
                    out.push_str(&format!("[THINKING] {}\n", clip(r, 2_000)));
                }
                if !m.content.trim().is_empty() {
                    out.push_str(&format!("[ASSISTANT] {}\n", clip(&m.content, 6_000)));
                }
                for call in &m.tool_calls {
                    out.push_str(&format!(
                        "[ACTION] {}({})\n",
                        call["function"]["name"].as_str().unwrap_or("?"),
                        clip(call["function"]["arguments"].as_str().unwrap_or(""), 2_000)
                    ));
                }
            }
            "tool" => {
                let failed = serde_json::from_str::<Value>(&m.content)
                    .ok()
                    .is_some_and(|v| v.get("error").is_some());
                out.push_str(&format!(
                    "[{}] {}\n",
                    if failed { "ERROR" } else { "OBSERVATION" },
                    clip(&m.content, 3_000)
                ));
            }
            _ => {}
        }
    }
    out
}

pub(super) fn elapsed_ms(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
pub(super) fn turn_error_class(message: &str) -> &'static str {
    let lower = message.to_lowercase();
    if lower.contains("step limit reached") {
        "step_limit"
    } else if lower.contains("cancelled") {
        "cancelled"
    } else if lower.contains("provider")
        || lower.contains("stream")
        || lower.contains("connection")
        || lower.contains("sending request")
    {
        "provider_error"
    } else {
        "turn_error"
    }
}

impl Engine for StudioEngine {
    fn rollout<'a>(&'a self, label: &'a str, tree: &'a Tree) -> BoxFuture<'a, Result<RunView>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            if let Some(external) = &self.external {
                return self.external_rollout(external, label, tree).await;
            }
            let workdir = self.scratch.join(label);
            let _ = std::fs::remove_dir_all(&workdir);
            let mut project = self.workspace.project(&self.project);
            let (mut files, mut bytes) = (0, 0);
            let mut copies = vec![];
            for root in &mut project.roots {
                if root.writable {
                    let baseline = root.path.clone();
                    let copy = workdir.join(&root.alias);
                    copy_root(&baseline, &copy, &mut files, &mut bytes)?;
                    copies.push((root.alias.clone(), baseline, copy.canonicalize()?));
                    root.path = copies.last().unwrap().2.clone();
                }
            }
            let warnings = self.policy_warnings(tree)?;
            let session_id = crate::evaluation::new_id();
            let mut session = Session::new(session_id.clone(), self.executor.clone());
            session.title = format!("harness probe {label}");
            session.ephemeral = Some(Arc::new(crate::Ephemeral {
                project,
                harness: Some(tree.clone()),
                system: None,
                nested: false,
                policy: self.execution,
            }));
            session
                .messages
                .push(Message::text("user", self.task.clone()));
            let shared = Arc::new(Mutex::new(session));
            let outcome = tokio::time::timeout(
                self.rollout_timeout,
                crate::turn(self.app.clone(), shared.clone(), None),
            )
            .await;
            let (outcome, mut error_class) = match outcome {
                Ok(Ok(outcome)) => {
                    let class = match outcome {
                        crate::TurnOutcome::Complete => None,
                        crate::TurnOutcome::NoProgress
                        | crate::TurnOutcome::GoalIncomplete
                        | crate::TurnOutcome::TokenLimit
                        | crate::TurnOutcome::Compacted => Some("turn_error".to_string()),
                    };
                    (format!("{outcome:?}"), class)
                }
                Ok(Err(e)) => {
                    let message = format!("{e:#}");
                    let class = turn_error_class(&message).to_string();
                    (format!("error: {message}"), Some(class))
                }
                Err(_) => ("timeout".into(), Some("timeout".into())),
            };
            let state = shared.lock().unwrap().clone();
            let answer = state
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                .map(|m| m.content.clone())
                .unwrap_or_default();
            if outcome == "Complete" {
                error_class = Some(
                    if answer.trim().is_empty() {
                        "empty_answer"
                    } else {
                        "complete"
                    }
                    .into(),
                );
            }
            let delivered = outcome == "Complete" && error_class.as_deref() == Some("complete");
            let changes: String = copies
                .iter()
                .map(|(alias, snapshot, copy)| changes(alias, snapshot, copy))
                .collect();
            let _ = std::fs::remove_dir_all(&workdir);
            Ok(RunView {
                label: label.into(),
                delivered,
                outcome,
                answer,
                changes,
                trajectory: trajectory(&state.messages[1..]),
                workspace_fingerprint: Some(self.workspace.fingerprint.clone()),
                trace_id: Some(session_id),
                duration_ms: Some(elapsed_ms(started)),
                steps: Some(state.step),
                usage: (!state.usage.is_null()).then_some(state.usage.clone()),
                error_class,
                warnings,
                ..RunView::default()
            })
        })
    }
    fn call<'a>(
        &'a self,
        _label: &'a str,
        system: &'a str,
        user: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let history = [Message::text("user", user)];
            let (message, _) = tokio::time::timeout(
                std::time::Duration::from_secs(10 * 60),
                crate::provider::generate(
                    &self.app.client,
                    &self.proposer_provider,
                    &self.proposer,
                    system,
                    &history,
                    &[],
                    |_, _| {},
                ),
            )
            .await
            .context("Proposer timed out")??;
            if message.truncated {
                bail!("Reply hit the output token limit; keep files shorter");
            }
            Ok(message.content)
        })
    }
    fn edit<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        mission: &'a str,
        files: &'a BTreeMap<String, String>,
        current: &'a Tree,
    ) -> BoxFuture<'a, Result<(Tree, String)>> {
        Box::pin(async move {
            let dir = self
                .scratch
                .join(format!("propose-{label}-{}", crate::evaluation::new_id()));
            let result = async {
                let (context_dir, harness_dir) = (dir.join("context"), dir.join("harness"));
                write_files(&context_dir, files)?;
                write_files(&harness_dir, current)?;
                let root =
                    |alias: &str, path: &std::path::Path, writable| -> Result<context::Root> {
                        Ok(context::Root {
                            alias: alias.into(),
                            path: path.canonicalize()?,
                            writable,
                            repository: false,
                        })
                    };
                let project = context::Project {
                    id: self.project.id.clone(),
                    name: "harness proposer".into(),
                    roots: vec![
                        root("context", &context_dir, false)?,
                        root("harness", &harness_dir, true)?,
                    ],
                    instructions: String::new(),
                };
                let mut session =
                    Session::new(crate::evaluation::new_id(), self.proposer_agent.clone());
                session.title = format!("harness proposer {label}");
                session.ephemeral = Some(Arc::new(crate::Ephemeral {
                    project,
                    harness: Some(Tree::new()),
                    system: Some(system.to_string()),
                    nested: true,
                    policy: crate::ExecutionPolicy::files_only(),
                }));
                session.messages.push(Message::text("user", mission));
                let shared = Arc::new(Mutex::new(session));
                // A proposer that runs out of steps has still edited files;
                // validation decides whether they are usable.
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(20 * 60),
                    crate::turn(self.app.clone(), shared.clone(), None),
                )
                .await;
                let notes = shared
                    .lock()
                    .unwrap()
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                Ok((read_files(&harness_dir)?, notes))
            }
            .await;
            let _ = std::fs::remove_dir_all(&dir);
            result
        })
    }
    fn inspect<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        mission: &'a str,
        files: &'a BTreeMap<String, String>,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(self.inspect_session(label, system, mission, files))
    }
}

impl StudioEngine {
    async fn external_rollout(
        &self,
        external: &External,
        label: &str,
        tree: &Tree,
    ) -> Result<RunView> {
        if self.execution != crate::ExecutionPolicy::trusted_unsandboxed() {
            bail!("An external executor requires execution=trusted_unsandboxed. Workspace copies do not sandbox shell commands.");
        }
        let started = std::time::Instant::now();
        let workdir = self.scratch.join(label);
        let _ = std::fs::remove_dir_all(&workdir);
        let result = async {
            let root = self
                .workspace
                .roots
                .iter()
                .find(|r| r.writable)
                .context("An external executor needs a writable context folder to work in")?;
            let copy = workdir.join(&root.alias);
            let (mut files, mut bytes) = (0, 0);
            copy_root(&root.snapshot, &copy, &mut files, &mut bytes)?;
            let existing = std::fs::read_to_string(root.snapshot.join("AGENTS.md")).ok();
            let overlay = materialize(tree, external.layout, existing.as_deref());
            for (path, text) in &overlay {
                let file = copy.join(path);
                std::fs::create_dir_all(file.parent().context("No parent directory")?)?;
                std::fs::write(file, text)?;
            }
            let mut child = tokio::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(&external.command)
                .current_dir(&copy)
                .env("ALLPAKA_TASK", &self.task)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .context("Failed to start the external executor")?;
            let mut stdin = child.stdin.take().context("No stdin")?;
            let task = self.task.clone().into_bytes();
            let writer = tokio::spawn(async move {
                use tokio::io::AsyncWriteExt;
                let _ = stdin.write_all(&task).await;
            });
            let output = tokio::time::timeout(self.rollout_timeout, child.wait_with_output()).await;
            let _ = writer.await;
            let (outcome, stdout, stderr, mut error_class) = match output {
                Ok(Ok(out)) => (
                    if out.status.success() {
                        "Complete".to_string()
                    } else {
                        format!("exit {:?}", out.status.code())
                    },
                    String::from_utf8_lossy(&out.stdout).into_owned(),
                    String::from_utf8_lossy(&out.stderr).into_owned(),
                    (!out.status.success()).then(|| "external_exit".to_string()),
                ),
                Ok(Err(e)) => (
                    format!("error: {e}"),
                    String::new(),
                    String::new(),
                    Some("turn_error".to_string()),
                ),
                Err(_) => (
                    "timeout".into(),
                    String::new(),
                    String::new(),
                    Some("timeout".to_string()),
                ),
            };
            let changes = changes_over(&root.alias, &root.snapshot, &copy, &overlay);
            let answer = clip(stdout.trim(), 64_000);
            if outcome == "Complete" {
                error_class = Some(
                    if answer.is_empty() {
                        "empty_answer"
                    } else {
                        "complete"
                    }
                    .into(),
                );
            }
            Ok(RunView {
                label: label.into(),
                delivered: outcome == "Complete" && error_class.as_deref() == Some("complete"),
                outcome,
                answer,
                changes,
                trajectory: format!(
                    "[EXTERNAL] {}\n[STDOUT]\n{}\n[STDERR]\n{}\n",
                    external.command,
                    clip(&stdout, 60_000),
                    clip(&stderr, 20_000)
                ),
                workspace_fingerprint: Some(self.workspace.fingerprint.clone()),
                duration_ms: Some(elapsed_ms(started)),
                usage: Some(json!({"available": false})),
                error_class,
                ..RunView::default()
            })
        }
        .await;
        let _ = std::fs::remove_dir_all(&workdir);
        result
    }

    async fn inspect_session(
        &self,
        label: &str,
        system: &str,
        mission: &str,
        files: &BTreeMap<String, String>,
    ) -> Result<String> {
        let dir = self
            .scratch
            .join(format!("inspect-{label}-{}", crate::evaluation::new_id()));
        let result = async {
            let evidence = dir.join("harness_evidence");
            write_files(&evidence, files)?;
            let snapshot_project = self.workspace.project(&self.project);
            let mut roots: Vec<context::Root> = snapshot_project
                .roots
                .iter()
                .filter(|r| r.alias != "harness_evidence")
                .map(|r| context::Root {
                    writable: false,
                    ..r.clone()
                })
                .collect();
            roots.insert(
                0,
                context::Root {
                    alias: "harness_evidence".into(),
                    path: evidence.canonicalize()?,
                    writable: false,
                    repository: false,
                },
            );
            let project = context::Project {
                id: self.project.id.clone(),
                name: "harness selector".into(),
                roots,
                instructions: String::new(),
            };
            let mut settings = self.proposer.clone();
            settings.max_steps = 30;
            let mut session = Session::new(crate::evaluation::new_id(), settings);
            session.ephemeral = Some(Arc::new(crate::Ephemeral {
                project,
                harness: Some(Tree::new()),
                system: Some(system.to_string()),
                nested: true,
                policy: crate::ExecutionPolicy::files_only(),
            }));
            session.messages.push(Message::text("user", mission));
            let shared = Arc::new(Mutex::new(session));
            tokio::time::timeout(
                std::time::Duration::from_secs(15 * 60),
                crate::turn(self.app.clone(), shared.clone(), None),
            )
            .await
            .context("Selector timed out")??;
            let answer = shared
                .lock()
                .unwrap()
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                .map(|m| m.content.clone())
                .context("Selector gave no answer")?;
            Ok(answer)
        }
        .await;
        let _ = std::fs::remove_dir_all(&dir);
        result
    }
}

/// Write a flat path->text map under `root`.
pub(super) fn write_files(root: &std::path::Path, files: &BTreeMap<String, String>) -> Result<()> {
    std::fs::create_dir_all(root)?;
    for (path, text) in files {
        if path
            .split('/')
            .any(|part| part.is_empty() || part == ".." || part.starts_with('.'))
        {
            bail!("Unsafe evidence path {path}");
        }
        let file = root.join(path);
        std::fs::create_dir_all(file.parent().context("No parent directory")?)?;
        std::fs::write(file, text)?;
    }
    Ok(())
}
/// Read a directory back into a tree; empty files are deletions.
pub(super) fn read_files(root: &std::path::Path) -> Result<Tree> {
    let mut tree = Tree::new();
    for (path, file) in files_under(root) {
        let text = String::from_utf8(std::fs::read(&file)?)
            .with_context(|| format!("{path} is not UTF-8 text"))?;
        if !text.trim().is_empty() {
            tree.insert(path, text);
        }
    }
    Ok(tree)
}
