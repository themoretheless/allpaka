//! Task-level recursion: Propose -> Probe -> Compose (after Harness²).
//!
//! A job takes the last task of a session and refines the project harness on
//! it without any ground truth:
//!
//! 1. `k` base rollouts run the task under the current harness.
//! 2. Proposers edit the harness on two dimensions, INV (how the agent works)
//!    and DOM (what a practitioner knows); every candidate is probed once.
//! 3. A composer merges, adopts or keeps from the behavioral contrasts.
//!
//! Parallel mode probes `k` INV and `k` DOM candidates and composes `k`
//! finals from the shared pool; sequential mode carries one checkpoint
//! through `k` rounds. Either way the budget is 4k rollouts and 3k proposer
//! stages. A blind selector picks among the finals by deliverable alone, the
//! reflector writes what each edit did, and the chosen tree is stored as a
//! revision that the user adopts explicitly.
//!
//! Rollouts run in headless sessions on copies of the project's writable
//! roots, so a probe never touches the real workspace. Every rollout and
//! stage is saved under the job directory, and a resumed job reuses them.

use crate::{
    context, error, harness, harness::Tree, harness_library, harness_prompts as prompts, types::*,
    ApiResult, App,
};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What the algorithm needs from the outside world. Studio runs real turns
/// and model calls; tests script both.
pub trait Engine: Send + Sync {
    /// Execute the task once under `tree`.
    fn rollout<'a>(&'a self, label: &'a str, tree: &'a Tree) -> BoxFuture<'a, Result<RunView>>;
    /// One model call without tools; returns the reply text.
    fn call<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        user: &'a str,
    ) -> BoxFuture<'a, Result<String>>;
    /// An agent session that edits `current` (its `harness` root) while
    /// reading `files` (its read-only `context` root). Returns the edited
    /// tree and the agent's notes.
    fn edit<'a>(
        &'a self,
        _label: &'a str,
        _system: &'a str,
        _mission: &'a str,
        _files: &'a BTreeMap<String, String>,
        _current: &'a Tree,
    ) -> BoxFuture<'a, Result<(Tree, String)>> {
        Box::pin(async { bail!("This engine cannot run agentic proposers") })
    }
    /// A read-only agent session over `files` (as `harness_evidence`) and
    /// the project's own roots; returns its final answer. Engines without
    /// agents get the files inlined into one call.
    fn inspect<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        mission: &'a str,
        files: &'a BTreeMap<String, String>,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let inlined: String = files
                .iter()
                .map(|(p, t)| format!("=== {p} ===\n{t}\n"))
                .collect();
            self.call(label, system, &format!("{mission}\n\n{inlined}"))
                .await
        })
    }
}

/// One execution. It never carries a score: `delivered` only says that the
/// run finished with an answer.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RunView {
    pub label: String,
    /// Content hash of the harness tree, retained separately from the cache key.
    pub fingerprint: String,
    pub delivered: bool,
    pub outcome: String,
    pub answer: String,
    /// Diff of the files the run changed against the task workspace snapshot.
    pub changes: String,
    pub trajectory: String,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub workspace_fingerprint: Option<String>,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub usage: Option<Value>,
    #[serde(default)]
    pub error_class: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Stage {
    pub label: String,
    pub dimension: String,
    pub tree: Tree,
    pub notes: String,
    /// The proposal was rejected twice and the inherited tree was kept.
    pub fell_back: bool,
    #[serde(default)]
    pub error: Option<String>,
    /// Hash of everything the stage was asked; a resumed job reuses the
    /// saved stage only when its inputs are unchanged.
    #[serde(default)]
    pub key: String,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Recursion {
    #[default]
    Parallel,
    Sequential,
}

pub struct Plan<'a> {
    pub engine: &'a dyn Engine,
    pub dir: PathBuf,
    pub k: usize,
    pub mode: Recursion,
    pub task: String,
    pub base: Tree,
    /// Execution context shared by every rollout of this task. Deliberately
    /// excludes `k` and recursion mode, which do not change a single rollout.
    pub cache_namespace: String,
    /// Content identity of the frozen writable roots, recorded even when the
    /// executor fails before it can return its own RunView.
    pub workspace_fingerprint: String,
    /// Library entries of earlier tasks, already rendered.
    pub library: String,
    pub width: usize,
    /// Proposers and composers run as agents editing files (as in Harness²)
    /// instead of answering with one JSON object.
    pub agentic: bool,
    /// Optional rubric for reporting only: the method never sees scores.
    pub rubric: Option<String>,
    /// Fresh Base/Champion pairs after selection. Zero disables validation.
    pub validation_runs: usize,
    /// Expected output from a dataset sample: given to the rubric judge
    /// only, never to proposers, composers, selectors or comparators.
    pub reference: Option<String>,
    pub progress: &'a (dyn Fn(String) + Send + Sync),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub mode: Recursion,
    pub k: usize,
    pub chosen: String,
    /// `final` when a composed harness was chosen, `base` when every final
    /// failed to deliver and the unmodified harness stays.
    pub chosen_kind: String,
    pub selection_why: String,
    pub tree: Tree,
    pub finals: Vec<String>,
    pub similarity: Option<f64>,
    pub fallbacks: Vec<String>,
    pub rollouts: usize,
    pub stages: usize,
    pub runs: Vec<RunSummary>,
    pub stage_notes: BTreeMap<String, String>,
    pub reflection: Option<Value>,
    /// Judge scores by rollout label, when a rubric was given. Reported
    /// after selection and never fed back into it.
    #[serde(default)]
    pub scores: BTreeMap<String, Option<f64>>,
    /// Independent post-selection validation. Missing only in legacy outcomes.
    #[serde(default)]
    pub validation: Option<ValidationReport>,
    #[serde(default)]
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub label: String,
    pub delivered: bool,
    pub outcome: String,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub error_class: Option<String>,
    /// Fresh validation score, when a rubric was supplied. Search scores stay
    /// in `Outcome::scores` and are never shown to the comparator.
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn summarize(run: &RunView) -> RunSummary {
    RunSummary {
        label: run.label.clone(),
        delivered: run.delivered,
        outcome: run.outcome.clone(),
        trace_id: run.trace_id.clone(),
        duration_ms: run.duration_ms,
        steps: run.steps,
        error_class: run.error_class.clone(),
        score: None,
        warnings: run.warnings.clone(),
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    NotRun,
    Unchanged,
    ImprovementObserved,
    NoObservedDifference,
    RegressionObserved,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PairVerdict {
    Base,
    Champion,
    Tie,
    Inconclusive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationPair {
    pub index: usize,
    pub base: RunSummary,
    pub champion: RunSummary,
    /// `None` means the comparator failed and can be retried on resume.
    pub verdict: Option<PairVerdict>,
    pub why: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub champion: String,
    pub base_tree: String,
    pub champion_tree: String,
    pub requested_runs: usize,
    pub rollout_count: usize,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub pairs: Vec<ValidationPair>,
    pub reason: String,
    pub warnings: Vec<String>,
}

// ---- evidence ----------------------------------------------------------------

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit * 2 / 3).collect();
    let tail: Vec<char> = text.chars().rev().take(limit / 3).collect();
    let tail: String = tail.into_iter().rev().collect();
    format!(
        "{head}\n… [{} characters omitted] …\n{tail}",
        text.chars().count() - limit
    )
}
fn render_tree(tree: &Tree) -> String {
    if tree.is_empty() {
        return "(empty harness: no files)\n".into();
    }
    tree.iter()
        .map(|(path, text)| format!("=== {path} ===\n{}\n", clip(text, 12_000)))
        .collect()
}
fn render_run(run: &RunView, budget: usize) -> String {
    let warnings = if run.warnings.is_empty() {
        String::new()
    } else {
        format!("### execution warnings\n{}\n", run.warnings.join("\n"))
    };
    format!(
        "## rollout {} (delivered: {}, outcome: {})\n{warnings}### trajectory\n{}\n### final answer\n{}\n### file changes\n{}\n",
        run.label,
        if run.delivered { "yes" } else { "NO" },
        run.outcome,
        clip(&run.trajectory, budget),
        clip(&run.answer, budget / 2),
        if run.changes.is_empty() { "(none)".into() } else { clip(&run.changes, budget / 2) },
    )
}
struct Candidate<'a> {
    stage: &'a Stage,
    run: &'a RunView,
}
/// The same evidence as one message (JSON proposers) and as files under
/// `context/` (agentic proposers).
pub struct Evidence {
    text: String,
    files: BTreeMap<String, String>,
}
fn evidence(plan: &Plan, current: &Tree, bases: &[RunView], pool: &[Candidate]) -> Evidence {
    let mut files = BTreeMap::new();
    files.insert("task.md".to_string(), plan.task.clone());
    for (path, text) in &plan.base {
        files.insert(format!("base_harness/{path}"), text.clone());
    }
    for run in bases {
        files.insert(
            format!("rollouts/{}.md", run.label),
            render_run(run, 200_000),
        );
    }
    for c in pool {
        let label = &c.stage.label;
        for (path, text) in &c.stage.tree {
            files.insert(format!("pool/{label}/harness/{path}"), text.clone());
        }
        let d = harness::diff(current, &c.stage.tree);
        files.insert(
            format!("pool/{label}/harness.diff"),
            if d.text.is_empty() {
                "(identical)\n".into()
            } else {
                d.text
            },
        );
        let notes = match &c.stage.error {
            Some(e) => format!(
                "{} candidate; its proposal was rejected twice and it inherited the tree.\n{e}\n",
                c.stage.dimension
            ),
            None => format!("{} candidate\n\n{}\n", c.stage.dimension, c.stage.notes),
        };
        files.insert(format!("pool/{label}/notes.md"), notes);
        files.insert(
            format!("pool/{label}/rollout.md"),
            render_run(c.run, 200_000),
        );
    }
    if !plan.library.is_empty() {
        files.insert("library/LIBRARY.md".into(), plan.library.clone());
    }
    Evidence {
        text: evidence_text(plan, current, bases, pool),
        files,
    }
}
fn evidence_text(plan: &Plan, current: &Tree, bases: &[RunView], pool: &[Candidate]) -> String {
    let runs = bases.len() + pool.len();
    let budget = (120_000 / runs.max(1)).clamp(3_000, 24_000);
    let mut out = format!(
        "# TASK (verbatim)\n{}\n\n# HARNESS YOU ARE EDITING\n{}\n",
        plan.task,
        render_tree(current)
    );
    out.push_str("\n# BASE ROLLOUTS (unmodified harness)\n");
    for run in bases {
        out.push_str(&render_run(run, budget));
    }
    if !pool.is_empty() {
        out.push_str("\n# CANDIDATE POOL (each rolled out once)\n");
        for c in pool {
            let d = harness::diff(&plan.base, &c.stage.tree);
            out.push_str(&format!(
                "## candidate {} ({}{})\n### harness diff against the original harness\n{}\n### proposer notes\n{}\n",
                c.stage.label,
                c.stage.dimension,
                if c.stage.fell_back { ", proposal rejected: inherited tree" } else { "" },
                if d.text.is_empty() { "(identical)".into() } else { clip(&d.text, 16_000) },
                clip(&c.stage.notes, 4_000),
            ));
            out.push_str(&render_run(c.run, budget));
        }
    }
    if !plan.library.is_empty() {
        out.push_str(&format!(
            "\n# LIBRARY (finished harnesses and lessons from earlier tasks in this project)\n{}\n",
            plan.library
        ));
    }
    out
}

// ---- replies -----------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    #[serde(default)]
    files: BTreeMap<String, Option<String>>,
    #[serde(default)]
    adopt: Option<String>,
    #[serde(default)]
    notes: String,
}
/// The first JSON object in a reply, tolerating a code fence around it.
pub(crate) fn json_object(text: &str) -> Result<Value> {
    let start = text.find('{').context("Reply contains no JSON object")?;
    let end = text.rfind('}').context("Reply contains no JSON object")?;
    if end < start {
        bail!("Reply contains no JSON object");
    }
    Ok(crate::strict_json::parse(&text[start..=end])?)
}
fn parse_reply(text: &str) -> Result<Reply> {
    let reply: Reply = serde_json::from_value(json_object(text)?).context(
        "Reply must be {\"files\":{…},\"notes\":\"…\"} or {\"adopt\":\"label\",\"notes\":\"…\"}",
    )?;
    if reply.adopt.is_some() && !reply.files.is_empty() {
        bail!("Use either adopt or files, not both");
    }
    Ok(reply)
}
fn apply(reply: &Reply, current: &Tree, pool: &[Candidate]) -> Result<Tree> {
    let tree = match &reply.adopt {
        Some(label) => pool
            .iter()
            .find(|c| &c.stage.label == label)
            .map(|c| c.stage.tree.clone())
            .with_context(|| format!("Cannot adopt `{label}`: not a candidate label"))?,
        None => {
            let mut tree = current.clone();
            for (path, text) in &reply.files {
                match text {
                    Some(text) => {
                        tree.insert(path.clone(), text.clone());
                    }
                    None => {
                        tree.remove(path);
                    }
                }
            }
            tree
        }
    };
    accept(&tree)
}
fn accept(tree: &Tree) -> Result<Tree> {
    let tree = harness::autotrigger(tree);
    let errors = harness::validate(&tree);
    if !errors.is_empty() {
        bail!("{}", errors.join("\n"));
    }
    Ok(tree)
}

// ---- persistence for resume ----------------------------------------------------

fn read_json<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Option<T> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}
pub(crate) fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    std::fs::create_dir_all(path.parent().context("No parent directory")?)?;
    let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
    std::fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
}

struct Ledger {
    rollouts: Mutex<usize>,
    stages: Mutex<usize>,
    fallbacks: Mutex<Vec<String>>,
}

/// Downstream model calls must not outlive the rollout/context that produced
/// their inputs. Legacy unkeyed records intentionally fail this deserialization
/// and are recomputed once.
#[derive(Debug, Serialize, Deserialize)]
struct Cached<T> {
    cache_key: String,
    value: T,
}

fn downstream_cache_key(plan: &Plan<'_>, kind: &str, input: Value) -> String {
    harness::fingerprint(&Tree::from([
        ("cache_namespace".to_string(), plan.cache_namespace.clone()),
        ("cache_kind".to_string(), kind.to_string()),
        ("input".to_string(), input.to_string()),
    ]))
}

fn rollout_role(label: &str) -> &'static str {
    if label.starts_with("validation-base-") {
        "validation_base"
    } else if label.starts_with("validation-champion-") {
        "validation_champion"
    } else if label.starts_with("base") {
        "base"
    } else if label.starts_with("inv") || label.starts_with("dom") {
        "probe"
    } else if label.starts_with("agg") || label.starts_with('c') {
        "final"
    } else {
        "rollout"
    }
}
fn rollout_attempt(label: &str) -> &str {
    let start = label
        .rfind(|c: char| !c.is_ascii_digit())
        .map_or(0, |index| index + 1);
    if start == label.len() {
        "0"
    } else {
        &label[start..]
    }
}
fn rollout_cache_key(plan: &Plan<'_>, label: &str, tree_fingerprint: &str) -> String {
    harness::fingerprint(&Tree::from([
        ("cache_namespace".to_string(), plan.cache_namespace.clone()),
        ("tree_fingerprint".to_string(), tree_fingerprint.to_string()),
        ("rollout_role".to_string(), rollout_role(label).to_string()),
        ("attempt".to_string(), rollout_attempt(label).to_string()),
    ]))
}

async fn roll(
    plan: &Plan<'_>,
    ledger: &Ledger,
    gate: &tokio::sync::Semaphore,
    label: &str,
    tree: &Tree,
) -> Result<RunView> {
    *ledger.rollouts.lock().unwrap() += 1;
    let path = plan.dir.join("rollouts").join(format!("{label}.json"));
    let fingerprint = harness::fingerprint(tree);
    let cache_key = rollout_cache_key(plan, label, &fingerprint);
    if let Some(run) =
        read_json::<RunView>(&path).filter(|r| r.cache_key.as_deref() == Some(cache_key.as_str()))
    {
        return Ok(run);
    }
    let _permit = gate.acquire().await?;
    (plan.progress)(format!("rollout {label}"));
    let started = std::time::Instant::now();
    let mut run = match plan.engine.rollout(label, tree).await {
        Ok(run) => run,
        // A failed execution is evidence too: it delivered nothing.
        Err(e) => {
            let message = format!("{e:#}");
            RunView {
                outcome: format!("error: {message}"),
                error_class: Some(turn_error_class(&message).into()),
                ..RunView::default()
            }
        }
    };
    run.label = label.into();
    run.fingerprint = fingerprint;
    run.cache_key = Some(cache_key);
    run.duration_ms.get_or_insert_with(|| elapsed_ms(started));
    run.workspace_fingerprint
        .get_or_insert_with(|| plan.workspace_fingerprint.clone());
    run.error_class.get_or_insert_with(|| {
        if run.delivered {
            "complete".into()
        } else if run.answer.trim().is_empty() {
            "empty_answer".into()
        } else {
            "turn_error".into()
        }
    });
    write_json(&path, &run)?;
    Ok(run)
}
async fn roll_many(
    plan: &Plan<'_>,
    ledger: &Ledger,
    gate: &tokio::sync::Semaphore,
    items: &[(String, Tree)],
) -> Result<Vec<RunView>> {
    futures_util::future::join_all(
        items
            .iter()
            .map(|(label, tree)| roll(plan, ledger, gate, label, tree)),
    )
    .await
    .into_iter()
    .collect()
}

#[allow(clippy::too_many_arguments)]
async fn stage(
    plan: &Plan<'_>,
    ledger: &Ledger,
    gate: &tokio::sync::Semaphore,
    label: &str,
    dimension: &str,
    system: &str,
    evidence: &Evidence,
    current: &Tree,
    pool: &[Candidate<'_>],
) -> Result<Stage> {
    *ledger.stages.lock().unwrap() += 1;
    let path = plan.dir.join("stages").join(format!("{label}.json"));
    let key = harness::fingerprint(&Tree::from([
        ("cache_namespace".to_string(), plan.cache_namespace.clone()),
        ("system".to_string(), system.to_string()),
        (
            "evidence".to_string(),
            if plan.agentic {
                serde_json::to_string(&evidence.files)?
            } else {
                evidence.text.clone()
            },
        ),
        ("current".to_string(), harness::fingerprint(current)),
        ("agentic".to_string(), plan.agentic.to_string()),
    ]));
    let user = evidence.text.as_str();
    if let Some(done) = read_json::<Stage>(&path).filter(|d| d.key == key) {
        if done.fell_back {
            ledger.fallbacks.lock().unwrap().push(label.into());
        }
        return Ok(done);
    }
    let _permit = gate.acquire().await?;
    (plan.progress)(format!("stage {label}"));
    let mut prompt = user.to_string();
    let mut last_error = String::new();
    for attempt in 0..2 {
        let result = async {
            if plan.agentic {
                let mission = prompts::mission((attempt > 0).then_some(last_error.as_str()));
                let (tree, notes) = plan
                    .engine
                    .edit(label, system, &mission, &evidence.files, current)
                    .await?;
                return Ok::<_, anyhow::Error>((accept(&tree)?, notes));
            }
            let text = plan.engine.call(label, system, &prompt).await?;
            let reply = parse_reply(&text)?;
            let tree = apply(&reply, current, pool)?;
            Ok::<_, anyhow::Error>((tree, reply.notes))
        }
        .await;
        match result {
            Ok((tree, notes)) => {
                let done = Stage {
                    label: label.into(),
                    dimension: dimension.into(),
                    tree,
                    notes,
                    fell_back: false,
                    error: None,
                    key: key.clone(),
                };
                write_json(&path, &done)?;
                return Ok(done);
            }
            Err(e) => {
                last_error = format!("{e:#}");
                prompt = format!("{user}\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n{last_error}\n\nReply again with one valid JSON object.");
            }
        }
    }
    ledger.fallbacks.lock().unwrap().push(label.into());
    let done = Stage {
        label: label.into(),
        dimension: dimension.into(),
        tree: current.clone(),
        notes: String::new(),
        fell_back: true,
        error: Some(last_error),
        key,
    };
    write_json(&path, &done)?;
    Ok(done)
}

// ---- the two recursion modes ------------------------------------------------------

pub async fn run(plan: Plan<'_>) -> Result<Outcome> {
    if !(1..=4).contains(&plan.k) {
        bail!("k must be between 1 and 4");
    }
    if plan.validation_runs > 4 {
        bail!("validation.runs must be between 0 and 4");
    }
    let ledger = Ledger {
        rollouts: Mutex::new(0),
        stages: Mutex::new(0),
        fallbacks: Mutex::new(vec![]),
    };
    let rollout_gate = tokio::sync::Semaphore::new(plan.width.max(1));
    let call_gate = tokio::sync::Semaphore::new(plan.width.max(1));
    let k = plan.k;
    let base_items: Vec<(String, Tree)> = (1..=k)
        .map(|i| (format!("base{i}"), plan.base.clone()))
        .collect();
    let bases = roll_many(&plan, &ledger, &rollout_gate, &base_items).await?;

    let mut all_stages: Vec<Stage> = vec![];
    let mut all_runs: Vec<RunView> = vec![];
    let finals: Vec<(Stage, RunView)> = match plan.mode {
        Recursion::Parallel => {
            let user = evidence(&plan, &plan.base, &bases, &[]);
            let specs: Vec<(String, &str, String)> = (1..=k)
                .flat_map(|i| {
                    [
                        (
                            format!("inv{i}"),
                            "INV",
                            prompts::propose("INV", k, i, None, plan.agentic),
                        ),
                        (
                            format!("dom{i}"),
                            "DOM",
                            prompts::propose("DOM", k, i, None, plan.agentic),
                        ),
                    ]
                })
                .collect();
            let proposals: Vec<Stage> =
                futures_util::future::join_all(specs.iter().map(|(label, dim, system)| {
                    stage(
                        &plan,
                        &ledger,
                        &call_gate,
                        label,
                        dim,
                        system,
                        &user,
                        &plan.base,
                        &[],
                    )
                }))
                .await
                .into_iter()
                .collect::<Result<_>>()?;
            let probes = roll_many(
                &plan,
                &ledger,
                &rollout_gate,
                &proposals
                    .iter()
                    .map(|s| (s.label.clone(), s.tree.clone()))
                    .collect::<Vec<_>>(),
            )
            .await?;
            let pool: Vec<Candidate> = proposals
                .iter()
                .zip(&probes)
                .map(|(stage, run)| Candidate { stage, run })
                .collect();
            let user = evidence(&plan, &plan.base, &bases, &pool);
            let systems: Vec<(String, String)> = (1..=k)
                .map(|i| (format!("agg{i}"), prompts::aggregate(k, i, plan.agentic)))
                .collect();
            let aggs: Vec<Stage> =
                futures_util::future::join_all(systems.iter().map(|(label, system)| {
                    stage(
                        &plan, &ledger, &call_gate, label, "AGG", system, &user, &plan.base, &pool,
                    )
                }))
                .await
                .into_iter()
                .collect::<Result<_>>()?;
            let runs = roll_many(
                &plan,
                &ledger,
                &rollout_gate,
                &aggs
                    .iter()
                    .map(|s| (s.label.clone(), s.tree.clone()))
                    .collect::<Vec<_>>(),
            )
            .await?;
            all_stages.extend(proposals.iter().cloned());
            all_runs.extend(probes.iter().cloned());
            aggs.into_iter().zip(runs).collect()
        }
        Recursion::Sequential => {
            let mut current = plan.base.clone();
            let mut history: Vec<(Stage, RunView)> = vec![];
            let mut checkpoints = vec![];
            for t in 1..=k {
                let pool: Vec<Candidate> = history
                    .iter()
                    .map(|(stage, run)| Candidate { stage, run })
                    .collect();
                let user = evidence(&plan, &current, &bases, &pool);
                let (inv, dom) = futures_util::future::join(
                    stage(
                        &plan,
                        &ledger,
                        &call_gate,
                        &format!("inv{t}"),
                        "INV",
                        &prompts::propose("INV", 1, 1, Some(t), plan.agentic),
                        &user,
                        &current,
                        &[],
                    ),
                    stage(
                        &plan,
                        &ledger,
                        &call_gate,
                        &format!("dom{t}"),
                        "DOM",
                        &prompts::propose("DOM", 1, 1, Some(t), plan.agentic),
                        &user,
                        &current,
                        &[],
                    ),
                )
                .await;
                let (inv, dom) = (inv?, dom?);
                let runs = roll_many(
                    &plan,
                    &ledger,
                    &rollout_gate,
                    &[
                        (inv.label.clone(), inv.tree.clone()),
                        (dom.label.clone(), dom.tree.clone()),
                    ],
                )
                .await?;
                let mut round: Vec<(Stage, RunView)> = history.clone();
                round.push((inv, runs[0].clone()));
                round.push((dom, runs[1].clone()));
                let pool: Vec<Candidate> = round
                    .iter()
                    .map(|(stage, run)| Candidate { stage, run })
                    .collect();
                let user = evidence(&plan, &current, &bases, &pool);
                let system = if k == 1 {
                    prompts::aggregate(1, 1, plan.agentic)
                } else {
                    prompts::compose_round(t, k, plan.agentic)
                };
                let checkpoint = stage(
                    &plan,
                    &ledger,
                    &call_gate,
                    &format!("c{t}"),
                    "CMP",
                    &system,
                    &user,
                    &current,
                    &pool,
                )
                .await?;
                let run = roll(
                    &plan,
                    &ledger,
                    &rollout_gate,
                    &checkpoint.label,
                    &checkpoint.tree,
                )
                .await?;
                current = checkpoint.tree.clone();
                round.push((checkpoint.clone(), run.clone()));
                checkpoints.push((checkpoint, run));
                history = round;
            }
            for (stage, run) in &history {
                if stage.dimension != "CMP" {
                    all_stages.push(stage.clone());
                    all_runs.push(run.clone());
                }
            }
            checkpoints
        }
    };
    let rollouts = *ledger.rollouts.lock().unwrap();
    let stages = *ledger.stages.lock().unwrap();
    if rollouts != 4 * k || stages != 3 * k {
        bail!("Budget identity broken: {rollouts} rollouts and {stages} stages for k={k}");
    }

    let (chosen, chosen_kind, why) = select(&plan, &finals).await;
    let tree = finals
        .iter()
        .find(|(s, _)| s.label == chosen)
        .map(|(s, _)| s.tree.clone())
        .unwrap_or_else(|| plan.base.clone());
    // Validation has its own ledger and never enters the search evidence,
    // primary selection or reflection below.
    let validation = validate_champion(&plan, &chosen, &tree).await?;
    let similarity = harness::similarity(&finals.iter().map(|(s, _)| &s.tree).collect::<Vec<_>>());
    all_stages.extend(finals.iter().map(|(s, _)| s.clone()));
    all_runs.extend(finals.iter().map(|(_, r)| r.clone()));
    let reflection = reflect(&plan, &bases, &all_stages, &all_runs, &chosen, &why).await;
    let mut runs: Vec<RunSummary> = bases.iter().chain(&all_runs).map(summarize).collect();
    runs.dedup_by(|a, b| a.label == b.label);
    let fallbacks = ledger.fallbacks.lock().unwrap().clone();
    let mut scored: Vec<&RunView> = bases.iter().collect();
    scored.extend(
        finals
            .iter()
            .filter(|(s, _)| s.label == chosen)
            .map(|(_, r)| r),
    );
    let scores = match &plan.rubric {
        Some(rubric) => score(&plan, rubric, &scored).await,
        None => BTreeMap::new(),
    };
    let warnings = warnings(
        &plan,
        &bases,
        &chosen_kind,
        &fallbacks,
        similarity,
        &scores,
        &chosen,
    );
    Ok(Outcome {
        mode: plan.mode,
        k,
        chosen,
        chosen_kind,
        selection_why: why,
        tree,
        finals: finals.iter().map(|(s, _)| s.label.clone()).collect(),
        similarity,
        fallbacks,
        rollouts,
        stages,
        runs,
        stage_notes: all_stages
            .iter()
            .map(|s| {
                (
                    s.label.clone(),
                    s.error
                        .clone()
                        .map_or(s.notes.clone(), |e| format!("rejected twice: {e}")),
                )
            })
            .collect(),
        reflection,
        scores,
        validation: Some(validation),
        warnings,
    })
}

/// Rubric scores for the base rollouts and the chosen final, with Studio's
/// judge instructions. Reporting only: selection and composition never see
/// them. A failed judgement is None, never zero.
async fn score(plan: &Plan<'_>, rubric: &str, runs: &[&RunView]) -> BTreeMap<String, Option<f64>> {
    let path = plan.dir.join("scores.json");
    let cache_key = downstream_cache_key(
        plan,
        "scores",
        json!({
            "judge_system": crate::judge::SYSTEM,
            "rubric": rubric,
            "task": plan.task,
            "reference": plan.reference,
            "runs": runs.iter().map(|run| {
                json!({"label": run.label, "rollout_cache_key": run.cache_key})
            }).collect::<Vec<_>>(),
        }),
    );
    let mut saved = read_json::<Cached<BTreeMap<String, Option<f64>>>>(&path)
        .filter(|cached| cached.cache_key.as_str() == cache_key.as_str())
        .map(|cached| cached.value)
        .unwrap_or_default();
    for run in runs {
        if saved.get(&run.label).is_some_and(Option::is_some) {
            continue;
        }
        let answer = if run.changes.is_empty() {
            run.answer.clone()
        } else {
            format!(
                "{}\n\nFile changes:\n{}",
                run.answer,
                clip(&run.changes, 24_000)
            )
        };
        let input = json!({"rubric": rubric, "input": plan.task, "answer": clip(&answer, 60_000), "reference": plan.reference, "contexts": []});
        let value = async {
            let text = plan
                .engine
                .call(
                    &format!("judge-{}", run.label),
                    crate::judge::SYSTEM,
                    &input.to_string(),
                )
                .await?;
            let score = json_object(&text)?["score"]
                .as_f64()
                .context("Judge reply has no score")?;
            if !(0.0..=1.0).contains(&score) {
                bail!("Judge score out of bounds");
            }
            Ok::<_, anyhow::Error>(score)
        }
        .await
        .ok();
        saved.insert(run.label.clone(), value);
    }
    let _ = write_json(
        &path,
        &Cached {
            cache_key,
            value: &saved,
        },
    );
    saved
}

/// What a reader of the report should not miss (after Harness² report.py).
fn warnings(
    plan: &Plan<'_>,
    bases: &[RunView],
    chosen_kind: &str,
    fallbacks: &[String],
    similarity: Option<f64>,
    scores: &BTreeMap<String, Option<f64>>,
    chosen: &str,
) -> Vec<String> {
    let mut out = vec![];
    if bases.iter().all(|r| !r.delivered) {
        out.push(
            "No base rollout delivered: check the task and the executor before trusting any edit."
                .into(),
        );
    }
    if chosen_kind == "base" {
        out.push("No composed harness delivered; the current harness stays.".into());
    }
    if !fallbacks.is_empty() {
        out.push(format!(
            "Stages fell back to the inherited harness: {}.",
            fallbacks.join(", ")
        ));
    }
    if plan.k > 1 && similarity.is_some_and(|s| s > 0.999) {
        out.push("The finals collapsed into one harness; more budget explored nothing new.".into());
    }
    let base: Vec<f64> = bases
        .iter()
        .filter_map(|r| scores.get(&r.label).copied().flatten())
        .collect();
    if let (Some(chosen), false) = (scores.get(chosen).copied().flatten(), base.is_empty()) {
        let mean = base.iter().sum::<f64>() / base.len() as f64;
        if chosen < mean {
            out.push(format!("The chosen harness scored {chosen:.2} on the rubric, below the base mean {mean:.2}; review before adopting."));
        }
    }
    if scores.values().any(Option::is_none) {
        out.push("Some judgements failed and are left out of the scores.".into());
    }
    out
}

/// Blind selection among delivering finals. Falls back to the first
/// delivering final, and to the base harness when no final delivered.
async fn select(plan: &Plan<'_>, finals: &[(Stage, RunView)]) -> (String, String, String) {
    let delivered: Vec<&(Stage, RunView)> = finals.iter().filter(|(_, r)| r.delivered).collect();
    match delivered.len() {
        0 => {
            return (
                "base".into(),
                "base".into(),
                "No composed harness delivered; the current harness stays.".into(),
            )
        }
        1 => {
            return (
                delivered[0].0.label.clone(),
                "final".into(),
                "Only one composed harness delivered.".into(),
            )
        }
        _ => {}
    }
    let path = plan.dir.join("selection.json");
    // Shuffle deterministically by the task so ids say nothing about labels.
    let mut order: Vec<&(Stage, RunView)> = delivered.clone();
    let seed = harness::fingerprint(&Tree::from([("task".to_string(), plan.task.clone())]));
    order
        .sort_by_key(|(s, _)| harness::fingerprint(&Tree::from([(seed.clone(), s.label.clone())])));
    let ids: Vec<String> = (0..order.len())
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect();
    let mut files = BTreeMap::from([("task.md".to_string(), plan.task.clone())]);
    for (id, (_, run)) in ids.iter().zip(&order) {
        files.insert(
            format!("candidates/{id}.md"),
            format!(
                "# Final answer\n{}\n\n# File changes\n{}\n",
                run.answer,
                if run.changes.is_empty() {
                    "(none)"
                } else {
                    &run.changes
                }
            ),
        );
    }
    let mut user = format!("# REQUEST (verbatim)\n{}\n", plan.task);
    for (id, (_, run)) in ids.iter().zip(&order) {
        user.push_str(&format!(
            "\n# CANDIDATE {id}\n## final answer\n{}\n## file changes\n{}\n",
            clip(&run.answer, 16_000),
            if run.changes.is_empty() {
                "(none)".into()
            } else {
                clip(&run.changes, 16_000)
            }
        ));
    }
    let cache_key = downstream_cache_key(
        plan,
        "selection",
        json!({
            "agentic": plan.agentic,
            "system": prompts::select(order.len(), plan.agentic),
            "mission": plan.agentic.then(prompts::select_mission),
            "user": user,
            "files": files,
            "candidates": order.iter().map(|(stage, run)| {
                json!({"label": stage.label, "rollout_cache_key": run.cache_key})
            }).collect::<Vec<_>>(),
        }),
    );
    if let Some(saved) = read_json::<Cached<(String, String)>>(&path)
        .filter(|cached| cached.cache_key.as_str() == cache_key.as_str())
        .map(|cached| cached.value)
        .filter(|(label, _)| order.iter().any(|(stage, _)| stage.label == *label))
    {
        return (saved.0, "final".into(), saved.1);
    }
    let result = async {
        let text = if plan.agentic {
            plan.engine
                .inspect(
                    "select",
                    &prompts::select(order.len(), true),
                    prompts::select_mission(),
                    &files,
                )
                .await?
        } else {
            plan.engine
                .call("select", &prompts::select(order.len(), false), &user)
                .await?
        };
        let value = json_object(&text)?;
        let choice = value["choice"]
            .as_str()
            .context("Selection has no choice")?
            .trim()
            .to_string();
        let index = ids
            .iter()
            .position(|id| *id == choice)
            .context("Selection chose an unknown id")?;
        Ok::<_, anyhow::Error>((
            order[index].0.label.clone(),
            value["why"].as_str().unwrap_or("").to_string(),
        ))
    }
    .await;
    match result {
        Ok((label, why)) => {
            let _ = write_json(
                &path,
                &Cached {
                    cache_key,
                    value: (label.clone(), why.clone()),
                },
            );
            (label, "final".into(), why)
        }
        Err(e) => (
            delivered[0].0.label.clone(),
            "final".into(),
            format!("Selector failed ({e:#}); first delivering final kept."),
        ),
    }
}

fn validation_report_key(
    plan: &Plan<'_>,
    chosen: &str,
    base_tree: &str,
    champion_tree: &str,
) -> String {
    downstream_cache_key(
        plan,
        "validation",
        json!({
            "champion": chosen,
            "base_tree": base_tree,
            "champion_tree": champion_tree,
            "runs": plan.validation_runs,
            "comparison_system": prompts::compare(plan.agentic),
            "comparison_mission": plan.agentic.then(prompts::compare_mission),
            "agentic": plan.agentic,
            "rubric": plan.rubric,
            "reference": plan.reference,
            "judge_system": crate::judge::SYSTEM,
        }),
    )
}

fn validation_candidate(run: &RunView) -> String {
    format!(
        "# Final answer\n{}\n\n# File changes\n{}\n",
        run.answer,
        if run.changes.is_empty() {
            "(none)"
        } else {
            &run.changes
        }
    )
}

/// A fresh rubric judgement for validation. Unlike `score`, this never writes
/// the search-level scores.json; the value lives in its ValidationPair.
async fn validation_judge(plan: &Plan<'_>, rubric: &str, run: &RunView) -> Result<f64> {
    let answer = if run.changes.is_empty() {
        run.answer.clone()
    } else {
        format!(
            "{}\n\nFile changes:\n{}",
            run.answer,
            clip(&run.changes, 24_000)
        )
    };
    let input = json!({
        "rubric": rubric,
        "input": plan.task,
        "answer": clip(&answer, 60_000),
        "reference": plan.reference,
        "contexts": [],
    });
    let text = plan
        .engine
        .call(
            &format!("judge-{}", run.label),
            crate::judge::SYSTEM,
            &input.to_string(),
        )
        .await?;
    let score = json_object(&text)?["score"]
        .as_f64()
        .context("Judge reply has no score")?;
    if !(0.0..=1.0).contains(&score) {
        bail!("Judge score out of bounds");
    }
    Ok(score)
}

/// Independent post-selection validation. Search rollouts are never reused as
/// validation evidence, and validation itself never becomes search evidence.
async fn validate_champion(
    plan: &Plan<'_>,
    chosen: &str,
    champion_tree: &Tree,
) -> Result<ValidationReport> {
    let selected_tree = champion_tree.clone();
    let base_tree = harness::fingerprint(&plan.base);
    let champion_tree = harness::fingerprint(champion_tree);
    let path = plan.dir.join("validation.json");
    let cache_key = validation_report_key(plan, chosen, &base_tree, &champion_tree);
    if let Some(saved) = read_json::<Cached<ValidationReport>>(&path)
        .filter(|cached| cached.cache_key == cache_key)
        .map(|cached| cached.value)
        .filter(|report| report.pairs.iter().all(|pair| pair.verdict.is_some()))
    {
        return Ok(saved);
    }

    let empty_report = |status, reason: &str| ValidationReport {
        status,
        champion: chosen.into(),
        base_tree: base_tree.clone(),
        champion_tree: champion_tree.clone(),
        requested_runs: plan.validation_runs,
        rollout_count: 0,
        wins: 0,
        losses: 0,
        ties: 0,
        pairs: vec![],
        reason: reason.into(),
        warnings: vec![],
    };
    if plan.validation_runs == 0 {
        let report = empty_report(
            ValidationStatus::NotRun,
            "Additional validation was disabled; search results are unchanged.",
        );
        write_json(
            &path,
            &Cached {
                cache_key,
                value: &report,
            },
        )?;
        return Ok(report);
    }
    if base_tree == champion_tree {
        let report = empty_report(
            ValidationStatus::Unchanged,
            "The selected tree is identical to Base; no additional rollouts were needed.",
        );
        write_json(
            &path,
            &Cached {
                cache_key,
                value: &report,
            },
        )?;
        return Ok(report);
    }

    let ledger = Ledger {
        rollouts: Mutex::new(0),
        stages: Mutex::new(0),
        fallbacks: Mutex::new(vec![]),
    };
    let gate = tokio::sync::Semaphore::new(plan.width.max(1));
    let mut pairs = vec![];
    let mut report_warnings = vec![];
    for index in 1..=plan.validation_runs {
        let labels = (
            format!("validation-base-{index}"),
            format!("validation-champion-{index}"),
        );
        let runs = roll_many(
            plan,
            &ledger,
            &gate,
            &[
                (labels.0.clone(), plan.base.clone()),
                (labels.1.clone(), selected_tree.clone()),
            ],
        )
        .await?;
        let (base, champion) = (&runs[0], &runs[1]);
        let seed = harness::fingerprint(&Tree::from([
            ("task".to_string(), plan.task.clone()),
            ("pair_index".to_string(), index.to_string()),
            ("base_tree".to_string(), base_tree.clone()),
            ("champion_tree".to_string(), champion_tree.clone()),
        ]));
        let champion_is_a = seed.as_bytes()[0] & 1 == 0;
        let (a, b) = if champion_is_a {
            (champion, base)
        } else {
            (base, champion)
        };
        let files = BTreeMap::from([
            ("task.md".to_string(), plan.task.clone()),
            ("candidates/A.md".to_string(), validation_candidate(a)),
            ("candidates/B.md".to_string(), validation_candidate(b)),
        ]);
        let mut user = format!("# REQUEST (verbatim)\n{}\n", plan.task);
        for (id, run) in [("A", a), ("B", b)] {
            user.push_str(&format!(
                "\n# CANDIDATE {id}\n## final answer\n{}\n## file changes\n{}\n",
                clip(&run.answer, 16_000),
                if run.changes.is_empty() {
                    "(none)".into()
                } else {
                    clip(&run.changes, 16_000)
                }
            ));
        }
        let system = prompts::compare(plan.agentic);
        let comparison_key = downstream_cache_key(
            plan,
            "validation-comparison",
            json!({
                "pair_index": index,
                "base_rollout_cache_key": base.cache_key,
                "champion_rollout_cache_key": champion.cache_key,
                "champion_is_a": champion_is_a,
                "system": system,
                "mission": plan.agentic.then(prompts::compare_mission),
                "agentic": plan.agentic,
                "user": user,
                "files": files,
                "rubric": plan.rubric,
                "reference": plan.reference,
                "judge_system": crate::judge::SYSTEM,
            }),
        );
        let comparison_path = plan.dir.join("comparisons").join(format!("{index}.json"));
        let saved = read_json::<Cached<ValidationPair>>(&comparison_path)
            .filter(|cached| cached.cache_key == comparison_key)
            .map(|cached| cached.value);
        if let Some(done) = saved.as_ref().filter(|pair| pair.verdict.is_some()) {
            pairs.push(done.clone());
            continue;
        }
        let mut pair = ValidationPair {
            index,
            base: summarize(base),
            champion: summarize(champion),
            verdict: None,
            why: None,
        };
        if let Some(previous) = saved {
            pair.base.score = previous.base.score;
            pair.champion.score = previous.champion.score;
        }
        if let Some(rubric) = &plan.rubric {
            for (run, summary) in [(base, &mut pair.base), (champion, &mut pair.champion)] {
                if summary.score.is_some() {
                    continue;
                }
                match validation_judge(plan, rubric, run).await {
                    Ok(score) => summary.score = Some(score),
                    Err(e) => report_warnings
                        .push(format!("Validation judge failed for {}: {e:#}", run.label)),
                }
            }
        }

        let result = if !base.delivered && !champion.delivered {
            Ok((
                PairVerdict::Inconclusive,
                "Neither fresh run delivered; no winner is inferred.".to_string(),
            ))
        } else if base.delivered && !champion.delivered {
            Ok((
                PairVerdict::Base,
                "Base delivered and Champion did not.".to_string(),
            ))
        } else if !base.delivered && champion.delivered {
            Ok((
                PairVerdict::Champion,
                "Champion delivered and Base did not.".to_string(),
            ))
        } else {
            async {
                (plan.progress)(format!("validation compare {index}"));
                let text = if plan.agentic {
                    plan.engine
                        .inspect(
                            &format!("compare-{index}"),
                            &system,
                            prompts::compare_mission(),
                            &files,
                        )
                        .await?
                } else {
                    plan.engine
                        .call(&format!("compare-{index}"), &system, &user)
                        .await?
                };
                let reply = json_object(&text)?;
                let winner = reply["winner"]
                    .as_str()
                    .context("Comparison has no winner")?;
                let verdict = match winner {
                    "A" => {
                        if champion_is_a {
                            PairVerdict::Champion
                        } else {
                            PairVerdict::Base
                        }
                    }
                    "B" => {
                        if champion_is_a {
                            PairVerdict::Base
                        } else {
                            PairVerdict::Champion
                        }
                    }
                    "tie" => PairVerdict::Tie,
                    _ => bail!("Comparison winner must be A, B or tie"),
                };
                Ok::<_, anyhow::Error>((verdict, reply["why"].as_str().unwrap_or("").to_string()))
            }
            .await
        };
        match result {
            Ok((verdict, why)) => {
                pair.verdict = Some(verdict);
                pair.why = Some(why);
            }
            Err(e) => {
                let message =
                    format!("Comparator failed for pair {index}: {e:#}; no winner inferred.");
                pair.why = Some(message.clone());
                report_warnings.push(message);
            }
        }
        write_json(
            &comparison_path,
            &Cached {
                cache_key: comparison_key,
                value: &pair,
            },
        )?;
        pairs.push(pair);
    }

    let wins = pairs
        .iter()
        .filter(|pair| pair.verdict == Some(PairVerdict::Champion))
        .count();
    let losses = pairs
        .iter()
        .filter(|pair| pair.verdict == Some(PairVerdict::Base))
        .count();
    let ties = pairs
        .iter()
        .filter(|pair| pair.verdict == Some(PairVerdict::Tie))
        .count();
    let unresolved = pairs
        .iter()
        .any(|pair| pair.verdict.is_none() || pair.verdict == Some(PairVerdict::Inconclusive));
    let (status, reason) = if unresolved {
        (
            ValidationStatus::Inconclusive,
            "Fresh comparisons were inconclusive; no automatic winner is inferred.",
        )
    } else if wins > 0 && losses == 0 {
        (
            ValidationStatus::ImprovementObserved,
            "Champion outperformed Base in the fresh comparisons; improvement is observed, not proven.",
        )
    } else if losses > 0 && wins == 0 {
        (
            ValidationStatus::RegressionObserved,
            "Base outperformed Champion in the fresh comparisons; regression is observed, not proven.",
        )
    } else if wins == 0 && losses == 0 && ties > 0 {
        (
            ValidationStatus::NoObservedDifference,
            "No difference was observed in the fresh comparisons.",
        )
    } else {
        (
            ValidationStatus::Inconclusive,
            "Fresh comparisons mixed wins and losses; the result is inconclusive.",
        )
    };
    let report = ValidationReport {
        status,
        champion: chosen.into(),
        base_tree,
        champion_tree,
        requested_runs: plan.validation_runs,
        rollout_count: *ledger.rollouts.lock().unwrap(),
        wins,
        losses,
        ties,
        pairs,
        reason: reason.into(),
        warnings: report_warnings,
    };
    write_json(
        &path,
        &Cached {
            cache_key,
            value: &report,
        },
    )?;
    Ok(report)
}

/// Harnessing-level evidence: what each edit did, and at most one lesson.
async fn reflect(
    plan: &Plan<'_>,
    bases: &[RunView],
    stages: &[Stage],
    runs: &[RunView],
    chosen: &str,
    why: &str,
) -> Option<Value> {
    let path = plan.dir.join("reflection.json");
    let mut user = format!(
        "# TASK (verbatim)\n{}\n\n# BASE HARNESS\n{}\n# BASE ROLLOUTS\n",
        plan.task,
        render_tree(&plan.base)
    );
    for run in bases {
        user.push_str(&render_run(run, 6_000));
    }
    user.push_str("\n# CANDIDATES\n");
    for (stage, run) in stages.iter().zip(runs) {
        let d = harness::diff(&plan.base, &stage.tree);
        user.push_str(&format!(
            "## {} ({})\n### diff against base\n{}\n### notes\n{}\n",
            stage.label,
            stage.dimension,
            if d.text.is_empty() {
                "(identical)".into()
            } else {
                clip(&d.text, 12_000)
            },
            clip(&stage.notes, 3_000)
        ));
        user.push_str(&render_run(run, 6_000));
    }
    user.push_str(&format!("\n# SELECTION\nchosen: {chosen}\nwhy: {why}\n"));
    let cache_key = downstream_cache_key(
        plan,
        "reflection",
        json!({
            "system": prompts::experience(),
            "user": user,
            "base_rollouts": bases.iter().map(|run| {
                json!({"label": run.label, "rollout_cache_key": run.cache_key})
            }).collect::<Vec<_>>(),
            "rollouts": runs.iter().map(|run| {
                json!({"label": run.label, "rollout_cache_key": run.cache_key})
            }).collect::<Vec<_>>(),
        }),
    );
    if let Some(saved) = read_json::<Cached<Value>>(&path)
        .filter(|cached| cached.cache_key.as_str() == cache_key.as_str())
    {
        return Some(saved.value);
    }
    let text = plan
        .engine
        .call("reflect", &prompts::experience(), &user)
        .await
        .ok()?;
    let value = json_object(&text).ok()?;
    value.get("edits")?.as_array()?;
    let _ = write_json(
        &path,
        &Cached {
            cache_key,
            value: &value,
        },
    );
    Some(value)
}

// ---- Studio engine -----------------------------------------------------------------

struct StudioEngine {
    app: App,
    executor: Settings,
    proposer: Settings,
    /// The proposer as an Auto-mode agent with writes, for agentic stages.
    proposer_agent: Settings,
    proposer_provider: crate::provider::Provider,
    project: context::Project,
    workspace: WorkspaceSnapshot,
    task: String,
    scratch: PathBuf,
    external: Option<External>,
    execution: crate::ExecutionPolicy,
    rollout_timeout: std::time::Duration,
}

#[derive(Clone, Debug)]
struct WorkspaceSnapshot {
    roots: Vec<SnapshotRoot>,
    fingerprint: String,
}
#[derive(Clone, Debug)]
struct SnapshotRoot {
    alias: String,
    original: PathBuf,
    snapshot: PathBuf,
    writable: bool,
}
impl WorkspaceSnapshot {
    /// Capture writable roots once per task. The hash depends only on stable
    /// root/file identities and bytes, never on directory iteration order.
    fn capture(project: &context::Project, directory: &std::path::Path) -> Result<Self> {
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

const COPY_MAX_FILES: usize = 20_000;
const COPY_MAX_BYTES: u64 = 256 * 1024 * 1024;
fn skipped(name: &str) -> bool {
    name.starts_with('.') || name == "target" || name == "node_modules"
}
fn copy_root(
    from: &std::path::Path,
    to: &std::path::Path,
    files: &mut usize,
    bytes: &mut u64,
) -> Result<()> {
    copy_root_observed(from, to, files, bytes, "", &mut |_, _| {})
}
fn copy_root_observed(
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
fn files_under(root: &std::path::Path) -> BTreeMap<String, PathBuf> {
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
fn changes(alias: &str, snapshot: &std::path::Path, copy: &std::path::Path) -> String {
    changes_over(alias, snapshot, copy, &BTreeMap::new())
}
/// Like `changes`, but files the harness materialized into the copy count
/// as already present with that content.
fn changes_over(
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
fn external_path(path: &str, skill_dir: &str) -> String {
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

fn elapsed_ms(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
fn turn_error_class(message: &str) -> &'static str {
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
fn write_files(root: &std::path::Path, files: &BTreeMap<String, String>) -> Result<()> {
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
fn read_files(root: &std::path::Path) -> Result<Tree> {
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

// ---- jobs ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProbeExecution {
    FilesOnly,
    TrustedUnsandboxed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidationRequest {
    #[serde(default = "default_validation_runs")]
    runs: usize,
}

impl Default for ValidationRequest {
    fn default() -> Self {
        Self {
            runs: default_validation_runs(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartRequest {
    session_id: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    mode: Recursion,
    /// Proposer model; defaults to the session's model.
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    /// Proposers edit files as agents (default) or answer with JSON.
    #[serde(default = "default_agentic")]
    agentic: bool,
    /// Optional rubric: the base and chosen rollouts are scored for the
    /// report only.
    #[serde(default)]
    rubric: Option<String>,
    /// Independent fresh Base/Champion comparisons after selection.
    #[serde(default)]
    validation: ValidationRequest,
    /// A stream of tasks refined in order; each sees the library entries of
    /// the ones before it. Empty means the conversation's own task.
    #[serde(default)]
    tasks: Vec<String>,
    /// Take the task stream from a Studio dataset instead: each sample's
    /// input (with its contexts) is a task and its expected output goes to
    /// the rubric judge as the reference.
    #[serde(default)]
    dataset: Option<DatasetRef>,
    /// Internal probes are files-only unless command execution is explicitly
    /// trusted. Workspace copies are not a security sandbox.
    #[serde(default)]
    execution: Option<ProbeExecution>,
    /// Run rollouts with an external agent CLI instead of Studio's loop.
    /// Needs the command plugin and explicit trusted_unsandboxed execution.
    #[serde(default)]
    external: Option<External>,
}

impl StartRequest {
    fn policy(&self) -> crate::ExecutionPolicy {
        match self.execution {
            Some(ProbeExecution::TrustedUnsandboxed) => {
                crate::ExecutionPolicy::trusted_unsandboxed()
            }
            Some(ProbeExecution::FilesOnly) | None => crate::ExecutionPolicy::files_only(),
        }
    }
}

const ROLLOUT_CACHE_SCHEMA: &str = "harness-rollout-cache-v2";
#[allow(clippy::too_many_arguments)]
fn cache_namespace(
    task: &str,
    executor: &Settings,
    proposer_provider: &str,
    proposer_model: &str,
    external: Option<&External>,
    execution: crate::ExecutionPolicy,
    workspace_fingerprint: &str,
    sample_id: Option<&str>,
) -> Result<String> {
    Ok(harness::fingerprint(&Tree::from([
        ("cache_schema".to_string(), ROLLOUT_CACHE_SCHEMA.to_string()),
        ("task".to_string(), task.to_string()),
        (
            "executor_settings".to_string(),
            serde_json::to_string(executor)?,
        ),
        ("executor_provider".to_string(), executor.provider.clone()),
        ("executor_model".to_string(), executor.model.clone()),
        (
            "proposer_provider".to_string(),
            proposer_provider.to_string(),
        ),
        ("proposer_model".to_string(), proposer_model.to_string()),
        ("external".to_string(), serde_json::to_string(&external)?),
        ("execution_policy".to_string(), format!("{execution:?}")),
        (
            "workspace_fingerprint".to_string(),
            workspace_fingerprint.to_string(),
        ),
        (
            "dataset_sample_id".to_string(),
            sample_id.unwrap_or("").to_string(),
        ),
    ])))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetRef {
    id: String,
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    limit: Option<usize>,
}
fn default_k() -> usize {
    1
}
fn default_agentic() -> bool {
    true
}
fn default_validation_runs() -> usize {
    1
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TaskResult {
    index: usize,
    task: String,
    outcome: Outcome,
    revision: Option<String>,
    library_entry: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct JobRecord {
    id: String,
    project_id: String,
    request: StartRequest,
    tasks: Vec<String>,
    /// Dataset references, parallel to `tasks`; empty without a dataset.
    #[serde(default)]
    references: Vec<Option<String>>,
    /// Stable dataset sample IDs, parallel to `tasks`; empty without a dataset.
    #[serde(default)]
    sample_ids: Vec<Option<String>>,
    executor: Settings,
    base_revision: Option<String>,
    created_ms: u64,
    status: String,
    #[serde(default)]
    events: Vec<String>,
    #[serde(default)]
    results: Vec<TaskResult>,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Clone, Default)]
pub(crate) struct Manager {
    running: Arc<Mutex<HashMap<String, tokio::task::AbortHandle>>>,
}
fn jobs_dir(data: &std::path::Path) -> PathBuf {
    data.join("harness/jobs")
}
fn valid_job(id: &str) -> bool {
    !id.is_empty() && id.len() < 80 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}
fn load_job(data: &std::path::Path, id: &str) -> Result<JobRecord> {
    if !valid_job(id) {
        bail!("Invalid job ID");
    }
    read_json(&jobs_dir(data).join(id).join("job.json")).context("Harness job not found")
}
fn update_job(
    data: &std::path::Path,
    id: &str,
    f: impl FnOnce(&mut JobRecord),
) -> Result<JobRecord> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    let mut job = load_job(data, id)?;
    f(&mut job);
    if job.events.len() > 200 {
        job.events.drain(..job.events.len() - 200);
    }
    write_json(&jobs_dir(data).join(id).join("job.json"), &job)?;
    Ok(job)
}
/// Jobs left running by a stopped server can be resumed explicitly.
pub(crate) fn recover(data: &std::path::Path) -> Result<()> {
    let Ok(entries) = std::fs::read_dir(jobs_dir(data)) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let id = entry.file_name().to_string_lossy().to_string();
        if let Ok(job) = load_job(data, &id) {
            if job.status == "running" {
                update_job(data, &id, |j| {
                    j.status = "interrupted".into();
                    j.events.push(
                        "server restarted; resume reuses finished rollouts and stages".into(),
                    );
                })?;
            }
        }
    }
    Ok(())
}

fn spawn(app: App, id: String) {
    let manager = app.harness_jobs.clone();
    let task_app = app.clone();
    let task_id = id.clone();
    let handle = tokio::spawn(async move {
        let data = task_app.data.clone();
        let result = execute(&task_app, &task_id).await;
        let _ = update_job(&data, &task_id, |j| match result {
            Ok(()) => j.status = "completed".into(),
            Err(e) => {
                j.status = "failed".into();
                j.error = Some(format!("{e:#}"));
            }
        });
        task_app
            .harness_jobs
            .running
            .lock()
            .unwrap()
            .remove(&task_id);
    });
    manager
        .running
        .lock()
        .unwrap()
        .insert(id, handle.abort_handle());
}

const LEGACY_EXTERNAL_WARNING: &str = "Legacy external executor job resumed as trusted_unsandboxed; workspace copies do not sandbox shell commands.";

async fn execute(app: &App, id: &str) -> Result<()> {
    let mut legacy_external = false;
    let job = update_job(&app.data, id, |j| {
        legacy_external = j.request.external.is_some() && j.request.execution.is_none();
        if legacy_external {
            j.request.execution = Some(ProbeExecution::TrustedUnsandboxed);
            if !j.warnings.iter().any(|w| w == LEGACY_EXTERNAL_WARNING) {
                j.warnings.push(LEGACY_EXTERNAL_WARNING.into());
            }
            if !j.events.iter().any(|e| e == LEGACY_EXTERNAL_WARNING) {
                j.events.push(LEGACY_EXTERNAL_WARNING.into());
            }
        }
        j.status = "running".into();
        j.error = None;
        j.events.push("started".into());
    })?;
    let project = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == job.project_id)
        .cloned()
        .context("Project not found")?;
    let base = match &job.base_revision {
        Some(id) => harness::load_revision(&app.data, id)?.tree,
        None => Tree::new(),
    };
    let mut proposer = job.executor.clone();
    proposer.mode = Mode::Chat;
    proposer.allow_writes = false;
    proposer.guardrails = None;
    proposer.json_mode = false;
    proposer.max_output_tokens = proposer.max_output_tokens.max(16_384);
    if let Some(provider) = &job.request.provider {
        proposer.provider = provider.clone();
    }
    if let Some(model) = &job.request.model {
        proposer.model = model.clone();
    }
    let proposer_provider = app
        .providers
        .lock()
        .unwrap()
        .iter()
        .find(|p| p.id == proposer.provider)
        .cloned()
        .context("Proposer provider not found")?;
    let mut proposer_agent = proposer.clone();
    proposer_agent.mode = Mode::Auto;
    proposer_agent.allow_writes = true;
    proposer_agent.max_steps = 40;
    for (index, task) in job.tasks.iter().enumerate() {
        if job.results.iter().any(|r| r.index == index) {
            continue;
        }
        let label = if job.tasks.len() > 1 {
            format!("task {}/{}: ", index + 1, job.tasks.len())
        } else {
            String::new()
        };
        let scratch = std::env::temp_dir()
            .join("allpaka-harness")
            .join(id)
            .join(index.to_string());
        let workspace = WorkspaceSnapshot::capture(&project, &scratch.join("snapshot"))?;
        let cache_namespace = cache_namespace(
            task,
            &job.executor,
            &proposer.provider,
            &proposer.model,
            job.request.external.as_ref(),
            job.request.policy(),
            &workspace.fingerprint,
            job.sample_ids.get(index).and_then(Option::as_deref),
        )?;
        let engine = StudioEngine {
            app: app.clone(),
            executor: job.executor.clone(),
            proposer: proposer.clone(),
            proposer_agent: proposer_agent.clone(),
            proposer_provider: proposer_provider.clone(),
            project: project.clone(),
            workspace,
            task: task.clone(),
            scratch: scratch.clone(),
            external: job.request.external.clone(),
            execution: job.request.policy(),
            rollout_timeout: std::time::Duration::from_secs(15 * 60),
        };
        let entry_id = format!("{id}-{index}");
        let library = harness_library::context_for(
            &app.data,
            &job.project_id,
            task,
            Some(&entry_id),
            24_000,
        )?;
        let data = app.data.clone();
        let job_id = id.to_string();
        let prefix = label.clone();
        let progress = move |event: String| {
            let _ = update_job(&data, &job_id, |j| {
                j.events.push(format!("{prefix}{event}"))
            });
        };
        let outcome = run(Plan {
            engine: &engine,
            dir: jobs_dir(&app.data)
                .join(id)
                .join("tasks")
                .join(index.to_string()),
            k: job.request.k,
            mode: job.request.mode,
            task: task.clone(),
            base: base.clone(),
            cache_namespace,
            workspace_fingerprint: engine.workspace.fingerprint.clone(),
            library,
            width: 2,
            agentic: job.request.agentic,
            rubric: job.request.rubric.clone(),
            validation_runs: job.request.validation.runs,
            reference: job.references.get(index).cloned().flatten(),
            progress: &progress,
        })
        .await;
        let _ = std::fs::remove_dir_all(&scratch);
        let outcome = outcome?;
        let revision = if outcome.tree != base {
            Some(
                harness::save_revision(
                    &app.data,
                    &job.project_id,
                    &outcome.tree,
                    job.base_revision.clone(),
                    &format!(
                        "harness job {id}{}: {} ({:?}, k={})",
                        if job.tasks.len() > 1 {
                            format!(" task {}", index + 1)
                        } else {
                            String::new()
                        },
                        outcome.chosen,
                        outcome.mode,
                        outcome.k
                    ),
                )?
                .id,
            )
        } else {
            None
        };
        let library_entry = match &outcome.reflection {
            Some(reflection) => harness_library::add(
                &app.data,
                &job.project_id,
                &entry_id,
                &outcome.tree,
                reflection,
                &outcome.chosen,
                &outcome.selection_why,
            )?,
            None => None,
        };
        update_job(&app.data, id, |j| {
            j.events.push(format!(
                "{label}chosen {} ({})",
                outcome.chosen, outcome.chosen_kind
            ));
            j.results.push(TaskResult {
                index,
                task: task.clone(),
                outcome,
                revision,
                library_entry,
            });
            j.results.sort_by_key(|r| r.index);
            j.warnings = stream_warnings(&j.results, legacy_external);
        })?;
    }
    Ok(())
}

/// Across a stream of tasks (after Harness² report.py): refinement that
/// keeps choosing the base harness is not adapting anything.
fn stream_warnings(results: &[TaskResult], legacy_external: bool) -> Vec<String> {
    let mut out = if legacy_external {
        vec![LEGACY_EXTERNAL_WARNING.into()]
    } else {
        vec![]
    };
    if results.len() >= 2 {
        let base = results
            .iter()
            .filter(|r| r.outcome.chosen_kind == "base")
            .count();
        if base * 5 >= results.len() * 4 {
            out.push(format!(
                "The base harness was kept in {base} of {} tasks.",
                results.len()
            ));
        }
        let fell = results
            .iter()
            .filter(|r| !r.outcome.fallbacks.is_empty())
            .count();
        if fell > 0 {
            out.push(format!(
                "Stages fell back in {fell} of {} tasks.",
                results.len()
            ));
        }
    }
    out
}

fn bad(e: impl ToString) -> (StatusCode, Json<Value>) {
    error(StatusCode::BAD_REQUEST, e)
}
pub(crate) async fn start_api(
    State(app): State<App>,
    Json(mut request): Json<StartRequest>,
) -> ApiResult<Value> {
    if !(1..=4).contains(&request.k) {
        return Err(bad("k must be between 1 and 4"));
    }
    if request.validation.runs > 4 {
        return Err(bad("validation.runs must be between 0 and 4"));
    }
    if request.tasks.len() > 20
        || request
            .tasks
            .iter()
            .any(|t| t.trim().is_empty() || t.len() > 16_000)
    {
        return Err(bad("Give at most 20 non-empty tasks of up to 16000 bytes"));
    }
    if request
        .rubric
        .as_ref()
        .is_some_and(|r| r.trim().is_empty() || r.len() > 16_000)
    {
        return Err(bad("Rubric must be 1–16000 bytes"));
    }
    if app.harness_jobs.running.lock().unwrap().len() >= 2 {
        return Err(error(
            StatusCode::TOO_MANY_REQUESTS,
            "Two harness jobs are already running",
        ));
    }
    let session = app
        .sessions
        .lock()
        .unwrap()
        .get(&request.session_id)
        .map(|(s, _)| s.clone())
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Conversation not found"))?;
    let (mut executor, task) = {
        let s = session.lock().unwrap();
        let task = s
            .goal
            .as_ref()
            .and_then(|g| s.messages.get(g.message_index))
            .or_else(|| {
                s.messages.iter().rev().find(|m| {
                    m.role == "user"
                        && !m.content.starts_with("Continue the task")
                        && !m.content.starts_with("Goal continuation")
                })
            })
            .map(|m| m.content.clone())
            .unwrap_or_default();
        (s.settings.clone(), task)
    };
    let (tasks, references, sample_ids) = match &request.dataset {
        Some(source) => {
            if !request.tasks.is_empty() {
                return Err(bad("Give either tasks or a dataset, not both"));
            }
            let dataset = app
                .evaluation
                .load(&source.id, source.version)
                .map_err(bad)?;
            if dataset.project_id != executor.project_id {
                return Err(bad("The dataset belongs to another project"));
            }
            let limit = source.limit.unwrap_or(20).clamp(1, 20);
            dataset
                .samples
                .iter()
                .take(limit)
                .map(|sample| {
                    let mut task = sample.input.clone();
                    if !sample.contexts.is_empty() {
                        task.push_str(&format!("\n\nContext:\n{}", sample.contexts.join("\n\n")));
                    }
                    (
                        task,
                        sample.expected_output.clone(),
                        Some(sample.id.clone()),
                    )
                })
                .fold(
                    (vec![], vec![], vec![]),
                    |(mut tasks, mut references, mut sample_ids), (task, reference, sample_id)| {
                        tasks.push(task);
                        references.push(reference);
                        sample_ids.push(sample_id);
                        (tasks, references, sample_ids)
                    },
                )
        }
        None if request.tasks.is_empty() => (vec![task], vec![], vec![]),
        None => (request.tasks.clone(), vec![], vec![]),
    };
    if tasks.is_empty() || tasks[0].trim().is_empty() {
        return Err(bad("The conversation has no task to refine on"));
    }
    if tasks.iter().any(|t| t.len() > 16_000) {
        return Err(bad("A task exceeds 16000 bytes"));
    }
    if let Some(external) = &request.external {
        if request.execution != Some(ProbeExecution::TrustedUnsandboxed) {
            return Err(bad(
                "An external executor requires execution=trusted_unsandboxed. Workspace copies do not sandbox shell commands.",
            ));
        }
        if external.command.trim().is_empty() || external.command.len() > 2000 {
            return Err(bad("The external command must be 1–2000 bytes"));
        }
        if !crate::plugins::tool_unlocked(&app.plugins, "run_command") {
            return Err(bad(
                "An external executor runs a command: enable the command plugin first",
            ));
        }
    } else if request.execution.is_none() {
        request.execution = Some(ProbeExecution::FilesOnly);
    }
    match executor.mode {
        Mode::Swarm => {
            return Err(bad(
                "Harness refinement runs single-agent modes; switch away from Swarm",
            ))
        }
        Mode::Goal => executor.mode = Mode::Auto,
        _ => {}
    }
    crate::validate_settings(&executor, &app).map_err(bad)?;
    let base_revision = harness::active_revision(&app.data, &executor.project_id)
        .map_err(bad)?
        .map(|r| r.id);
    let id = crate::evaluation::new_id();
    let job = JobRecord {
        id: id.clone(),
        project_id: executor.project_id.clone(),
        request,
        tasks,
        references,
        sample_ids,
        executor,
        base_revision,
        created_ms: harness::now_ms(),
        status: "queued".into(),
        events: vec![],
        results: vec![],
        warnings: vec![],
        error: None,
    };
    write_json(&jobs_dir(&app.data).join(&id).join("job.json"), &job).map_err(bad)?;
    spawn(app.clone(), id);
    Ok(Json(json!(job)))
}
#[derive(Deserialize)]
pub(crate) struct ProjectQuery {
    project_id: String,
}
pub(crate) async fn list_api(
    State(app): State<App>,
    Query(q): Query<ProjectQuery>,
) -> ApiResult<Value> {
    let mut jobs = vec![];
    if let Ok(entries) = std::fs::read_dir(jobs_dir(&app.data)) {
        for entry in entries.flatten() {
            if let Ok(job) = load_job(&app.data, &entry.file_name().to_string_lossy()) {
                if job.project_id == q.project_id {
                    jobs.push(json!({"id":job.id,"status":job.status,"created_ms":job.created_ms,"k":job.request.k,"mode":job.request.mode,"agentic":job.request.agentic,"tasks":job.tasks.len(),"done":job.results.len(),"task":job.tasks[0].chars().take(120).collect::<String>()}));
                }
            }
        }
    }
    jobs.sort_by_key(|j| std::cmp::Reverse(j["created_ms"].as_u64()));
    jobs.truncate(50);
    Ok(Json(json!({"jobs": jobs})))
}
pub(crate) async fn get_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    let base = job
        .base_revision
        .as_deref()
        .and_then(|r| harness::load_revision(&app.data, r).ok())
        .map(|r| r.tree)
        .unwrap_or_default();
    let diffs: Vec<harness::Diff> = job
        .results
        .iter()
        .map(|r| harness::diff(&base, &r.outcome.tree))
        .collect();
    let active = harness::active_revision(&app.data, &job.project_id)
        .ok()
        .flatten()
        .map(|r| r.id);
    Ok(Json(json!({"job": job, "diffs": diffs, "active": active})))
}
pub(crate) async fn cancel_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let handle = app.harness_jobs.running.lock().unwrap().remove(&id);
    if let Some(handle) = handle {
        handle.abort();
    }
    let job = update_job(&app.data, &id, |j| {
        if matches!(j.status.as_str(), "running" | "queued") {
            j.status = "cancelled".into();
            j.events.push("cancelled".into());
        }
    })
    .map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("allpaka-harness").join(&id));
    Ok(Json(json!(job)))
}
pub(crate) async fn resume_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    if app.harness_jobs.running.lock().unwrap().contains_key(&id) || job.status == "completed" {
        return Err(bad("Job is running or already completed"));
    }
    spawn(app.clone(), id);
    Ok(Json(json!({"resumed": true})))
}
/// Make one task's result (`{"task": index}`, default the last task with a
/// new revision) the project's active harness.
pub(crate) async fn adopt_api(
    State(app): State<App>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    let result = match body.get("task").and_then(Value::as_u64) {
        Some(index) => job.results.iter().find(|r| r.index as u64 == index),
        None => job.results.iter().rev().find(|r| r.revision.is_some()),
    };
    if result.is_some_and(|r| {
        r.outcome
            .validation
            .as_ref()
            .is_some_and(|validation| validation.status == ValidationStatus::RegressionObserved)
    }) && body.get("override_regression").and_then(Value::as_bool) != Some(true)
    {
        return Err(error(
            StatusCode::CONFLICT,
            "Fresh validation observed a regression. Adopt only after reviewing the comparisons; to override, set override_regression=true.",
        ));
    }
    let revision = result
        .and_then(|r| r.revision.clone())
        .ok_or_else(|| bad("The job kept the current harness; nothing to adopt"))?;
    harness::activate(&app.data, &job.project_id, Some(&revision)).map_err(bad)?;
    Ok(Json(json!({"active": revision})))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted engine: proposals add a skill named after the stage, the
    /// composer adopts the first INV candidate, and probes deliver unless the
    /// tree contains a `broken` marker.
    struct Scripted {
        calls: Mutex<Vec<String>>,
        rollouts: Mutex<Vec<String>>,
        reject_first: bool,
    }
    fn skill(name: &str) -> (String, Option<String>) {
        (
            format!("skills/{name}/SKILL.md"),
            Some(format!(
                "---\nname: {name}\ndescription: the task needs {name}\n---\nDo {name}.\n"
            )),
        )
    }
    impl Engine for Scripted {
        fn edit<'a>(
            &'a self,
            label: &'a str,
            system: &'a str,
            mission: &'a str,
            files: &'a BTreeMap<String, String>,
            current: &'a Tree,
        ) -> BoxFuture<'a, Result<(Tree, String)>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(label.to_string());
                assert!(system.contains("YOURS TO EDIT") || system.contains("HOW TO COMPOSE"));
                assert!(mission.contains("context/rollouts/<label>.md"));
                assert!(
                    files.contains_key("task.md")
                        && files.contains_key("rollouts/base1.md")
                        && files.contains_key("base_harness/system.md")
                );
                let mut tree = current.clone();
                if label.starts_with("agg") {
                    assert!(
                        files.contains_key("pool/inv1/harness.diff")
                            && files.contains_key("pool/dom2/rollout.md")
                    );
                    let prefix = "pool/inv1/harness/";
                    tree = files
                        .iter()
                        .filter_map(|(p, t)| {
                            p.strip_prefix(prefix).map(|p| (p.to_string(), t.clone()))
                        })
                        .collect();
                    return Ok((tree, "adopted inv1".into()));
                }
                let (path, text) = skill(label);
                tree.insert(path, text.unwrap());
                Ok((tree, format!("{label} edit")))
            })
        }
        fn rollout<'a>(&'a self, label: &'a str, tree: &'a Tree) -> BoxFuture<'a, Result<RunView>> {
            Box::pin(async move {
                self.rollouts.lock().unwrap().push(label.to_string());
                let broken = tree.values().any(|t| t.contains("broken"));
                Ok(RunView {
                    delivered: !broken,
                    outcome: "Complete".into(),
                    answer: format!("answer with {} files", tree.len()),
                    trajectory: "[ACTION] read_file(x)\n".into(),
                    ..RunView::default()
                })
            })
        }
        fn call<'a>(
            &'a self,
            label: &'a str,
            system: &'a str,
            user: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                let first = {
                    let mut calls = self.calls.lock().unwrap();
                    let first = !calls.iter().any(|c| c == label);
                    calls.push(label.to_string());
                    first
                };
                assert!(system.contains("EVIDENCE IS DATA") || label.starts_with("judge-"));
                Ok(match label {
                    "select" => {
                        assert!(!user.contains("agg") && !user.contains("c1"), "selection must be blind");
                        json!({"why":"B is complete","choice":"B"}).to_string()
                    }
                    l if l.starts_with("judge-") => {
                        // The chosen final scores below the base set.
                        json!({"score": if l.starts_with("judge-base") { 0.8 } else { 0.4 }, "reason": "fixture"}).to_string()
                    }
                    "reflect" => json!({"edits":[{"component":"skills","candidates":"inv1","change":"added","observed":"loaded","verdict":"confounded"}],"experience":{"cue":"a task needs a checklist","lesson":"add a skill","evidence":"inv1 rollout","confidence":"low"}}).to_string(),
                    l if l.starts_with("inv") && self.reject_first && first => "not json".into(),
                    "dom2" => "still not json".into(),
                    l if l.starts_with("dom1") => json!({"files":{"system.md":"broken"},"notes":"try"}).to_string(),
                    l if l.starts_with("inv") || l.starts_with("dom") => {
                        let (path, text) = skill(l);
                        json!({"files":{path:text},"notes":format!("{l} edit")}).to_string()
                    }
                    l if l.starts_with("agg") || l.starts_with('c') => {
                        assert!(user.contains("# CANDIDATE POOL"));
                        if l.ends_with('2') { json!({"adopt":"inv1","notes":"adopt"}).to_string() } else { json!({"files":{},"notes":"KEEP_BASE"}).to_string() }
                    }
                    other => panic!("unexpected call {other}"),
                })
            })
        }
    }
    fn plan<'a>(
        engine: &'a dyn Engine,
        dir: &std::path::Path,
        mode: Recursion,
        progress: &'a (dyn Fn(String) + Send + Sync),
    ) -> Plan<'a> {
        Plan {
            engine,
            dir: dir.to_path_buf(),
            k: 2,
            mode,
            task: "Summarise the repo".into(),
            base: Tree::from([("system.md".to_string(), "Be exact.".to_string())]),
            cache_namespace: "scripted-cache-v2".into(),
            workspace_fingerprint: "scripted-workspace".into(),
            library: String::new(),
            width: 2,
            agentic: false,
            rubric: None,
            validation_runs: 0,
            reference: None,
            progress,
        }
    }
    fn temp() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("allpaka-refine-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct ValidationEngine {
        rollouts: Mutex<Vec<String>>,
        calls: Mutex<Vec<String>>,
        inputs: Mutex<Vec<String>>,
        verdicts: Mutex<Vec<PairVerdict>>,
        fail_comparisons: Mutex<usize>,
        base_delivered: bool,
        champion_delivered: bool,
    }

    fn validator(verdicts: &[PairVerdict]) -> ValidationEngine {
        ValidationEngine {
            rollouts: Mutex::new(vec![]),
            calls: Mutex::new(vec![]),
            inputs: Mutex::new(vec![]),
            verdicts: Mutex::new(verdicts.to_vec()),
            fail_comparisons: Mutex::new(0),
            base_delivered: true,
            champion_delivered: true,
        }
    }

    impl Engine for ValidationEngine {
        fn rollout<'a>(
            &'a self,
            label: &'a str,
            _tree: &'a Tree,
        ) -> BoxFuture<'a, Result<RunView>> {
            Box::pin(async move {
                self.rollouts.lock().unwrap().push(label.to_string());
                let base = label.starts_with("validation-base-");
                Ok(RunView {
                    delivered: if base {
                        self.base_delivered
                    } else {
                        self.champion_delivered
                    },
                    outcome: "Complete".into(),
                    answer: if base {
                        "alpha artifact"
                    } else {
                        "omega artifact"
                    }
                    .into(),
                    trajectory: "secret-trajectory".into(),
                    ..RunView::default()
                })
            })
        }

        fn call<'a>(
            &'a self,
            label: &'a str,
            system: &'a str,
            user: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(label.to_string());
                self.inputs
                    .lock()
                    .unwrap()
                    .push(format!("{label}\n{system}\n{user}"));
                if label.starts_with("judge-validation-") {
                    return Ok(json!({
                        "score": if user.contains("omega artifact") { 1.0 } else { 0.5 },
                        "reason": "validation fixture",
                    })
                    .to_string());
                }
                assert!(label.starts_with("compare-"), "unexpected call {label}");
                for forbidden in [
                    "secret-base-tree",
                    "secret-champion-tree",
                    "secret-trajectory",
                    "secret-rubric",
                    "secret-reference",
                    "validation-base-",
                    "validation-champion-",
                    "agg1",
                    "selection_why",
                ] {
                    assert!(
                        !user.contains(forbidden),
                        "comparator received forbidden input {forbidden}"
                    );
                }
                {
                    let mut failures = self.fail_comparisons.lock().unwrap();
                    if *failures > 0 {
                        *failures -= 1;
                        bail!("comparator offline");
                    }
                }
                let verdict = {
                    let mut verdicts = self.verdicts.lock().unwrap();
                    if verdicts.is_empty() {
                        PairVerdict::Tie
                    } else {
                        verdicts.remove(0)
                    }
                };
                let candidate = |answer: &str| {
                    if user.contains(&format!("# CANDIDATE A\n## final answer\n{answer}")) {
                        "A"
                    } else {
                        assert!(user.contains(&format!("# CANDIDATE B\n## final answer\n{answer}")));
                        "B"
                    }
                };
                let winner = match verdict {
                    PairVerdict::Base => candidate("alpha artifact"),
                    PairVerdict::Champion => candidate("omega artifact"),
                    PairVerdict::Tie => "tie",
                    PairVerdict::Inconclusive => unreachable!(),
                };
                Ok(json!({"winner": winner, "why": "fixture comparison"}).to_string())
            })
        }
    }

    fn changed_tree(plan: &Plan<'_>) -> Tree {
        let mut tree = plan.base.clone();
        tree.insert("memory.md".into(), "secret-champion-tree".into());
        tree
    }

    #[tokio::test]
    async fn validation_disabled_and_unchanged_do_not_roll_out() {
        let dir = temp();
        let engine = validator(&[]);
        let quiet = |_: String| {};
        let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
        let champion = changed_tree(&plan);
        let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
        assert_eq!(report.status, ValidationStatus::NotRun);
        assert_eq!(report.rollout_count, 0);
        assert!(dir.join("validation.json").exists());

        plan.validation_runs = 2;
        let base = plan.base.clone();
        let report = validate_champion(&plan, "agg1", &base).await.unwrap();
        assert_eq!(report.status, ValidationStatus::Unchanged);
        assert_eq!(report.rollout_count, 0);
        assert!(engine.rollouts.lock().unwrap().is_empty());
        assert!(engine.calls.lock().unwrap().is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn validation_aggregates_fresh_pairs_with_an_exact_budget() {
        let quiet = |_: String| {};
        for (verdicts, expected, counts) in [
            (
                vec![PairVerdict::Champion],
                ValidationStatus::ImprovementObserved,
                (1, 0, 0),
            ),
            (
                vec![PairVerdict::Base],
                ValidationStatus::RegressionObserved,
                (0, 1, 0),
            ),
            (
                vec![PairVerdict::Tie],
                ValidationStatus::NoObservedDifference,
                (0, 0, 1),
            ),
            (
                vec![PairVerdict::Champion, PairVerdict::Base],
                ValidationStatus::Inconclusive,
                (1, 1, 0),
            ),
        ] {
            let dir = temp();
            let engine = validator(&verdicts);
            let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
            plan.validation_runs = verdicts.len();
            let champion = changed_tree(&plan);
            let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
            assert_eq!(report.status, expected);
            assert_eq!((report.wins, report.losses, report.ties), counts);
            assert_eq!(report.rollout_count, 2 * verdicts.len());
            assert_eq!(engine.rollouts.lock().unwrap().len(), 2 * verdicts.len());
            assert_eq!(engine.calls.lock().unwrap().len(), verdicts.len());
            assert_eq!(report.pairs.len(), verdicts.len());
            for index in 1..=verdicts.len() {
                assert!(dir.join(format!("comparisons/{index}.json")).exists());
                assert!(dir
                    .join(format!("rollouts/validation-base-{index}.json"))
                    .exists());
                assert!(dir
                    .join(format!("rollouts/validation-champion-{index}.json"))
                    .exists());
            }
            assert!(report.reason.contains("observed") || report.reason.contains("inconclusive"));
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[tokio::test]
    async fn validation_delivery_is_decided_before_the_comparator() {
        let quiet = |_: String| {};
        for (base, champion, status, verdict) in [
            (
                true,
                false,
                ValidationStatus::RegressionObserved,
                PairVerdict::Base,
            ),
            (
                false,
                true,
                ValidationStatus::ImprovementObserved,
                PairVerdict::Champion,
            ),
            (
                false,
                false,
                ValidationStatus::Inconclusive,
                PairVerdict::Inconclusive,
            ),
        ] {
            let dir = temp();
            let mut engine = validator(&[]);
            engine.base_delivered = base;
            engine.champion_delivered = champion;
            let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
            plan.validation_runs = 1;
            let tree = changed_tree(&plan);
            let report = validate_champion(&plan, "agg1", &tree).await.unwrap();
            assert_eq!(report.status, status);
            assert_eq!(report.pairs[0].verdict, Some(verdict));
            assert_eq!(report.rollout_count, 2);
            assert!(engine.calls.lock().unwrap().is_empty());
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[tokio::test]
    async fn validation_retries_only_a_failed_comparator_on_resume() {
        let dir = temp();
        let first = validator(&[PairVerdict::Champion]);
        *first.fail_comparisons.lock().unwrap() = 1;
        let quiet = |_: String| {};
        let mut plan = plan(&first, &dir, Recursion::Parallel, &quiet);
        plan.validation_runs = 1;
        let champion = changed_tree(&plan);
        let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
        assert_eq!(report.status, ValidationStatus::Inconclusive);
        assert_eq!(report.pairs[0].verdict, None);
        assert_eq!(first.rollouts.lock().unwrap().len(), 2);
        assert_eq!(first.calls.lock().unwrap().as_slice(), ["compare-1"]);

        let second = validator(&[PairVerdict::Champion]);
        let mut plan = self::plan(&second, &dir, Recursion::Parallel, &quiet);
        plan.validation_runs = 1;
        let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
        assert_eq!(report.status, ValidationStatus::ImprovementObserved);
        assert!(second.rollouts.lock().unwrap().is_empty());
        assert_eq!(second.calls.lock().unwrap().as_slice(), ["compare-1"]);

        let third = validator(&[]);
        let mut plan = self::plan(&third, &dir, Recursion::Parallel, &quiet);
        plan.validation_runs = 1;
        let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
        assert_eq!(report.status, ValidationStatus::ImprovementObserved);
        assert!(third.rollouts.lock().unwrap().is_empty());
        assert!(third.calls.lock().unwrap().is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn validation_champion_change_reuses_only_the_base_rollout() {
        let dir = temp();
        let first = validator(&[PairVerdict::Champion]);
        let quiet = |_: String| {};
        let mut plan = plan(&first, &dir, Recursion::Parallel, &quiet);
        plan.validation_runs = 1;
        let champion = changed_tree(&plan);
        validate_champion(&plan, "agg1", &champion).await.unwrap();

        let second = validator(&[PairVerdict::Champion]);
        let mut plan = self::plan(&second, &dir, Recursion::Parallel, &quiet);
        plan.validation_runs = 1;
        let mut changed = champion.clone();
        changed.insert("memory.md".into(), "another champion".into());
        validate_champion(&plan, "agg1", &changed).await.unwrap();
        assert_eq!(
            second.rollouts.lock().unwrap().as_slice(),
            ["validation-champion-1"]
        );
        assert_eq!(second.calls.lock().unwrap().as_slice(), ["compare-1"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn validation_scores_fresh_runs_without_leaking_the_rubric() {
        let dir = temp();
        let engine = validator(&[PairVerdict::Tie]);
        let quiet = |_: String| {};
        let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
        plan.base
            .insert("system.md".into(), "secret-base-tree".into());
        plan.validation_runs = 1;
        plan.rubric = Some("secret-rubric".into());
        plan.reference = Some("secret-reference".into());
        let champion = changed_tree(&plan);
        let report = validate_champion(&plan, "agg1", &champion).await.unwrap();
        assert_eq!(report.pairs[0].base.score, Some(0.5));
        assert_eq!(report.pairs[0].champion.score, Some(1.0));
        let calls = engine.calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("judge-validation-"))
                .count(),
            2
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("compare-"))
                .count(),
            1
        );
        let inputs = engine.inputs.lock().unwrap();
        assert!(inputs
            .iter()
            .filter(|input| input.starts_with("judge-validation-"))
            .all(|input| input.contains("secret-rubric") && input.contains("secret-reference")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn parallel_keeps_the_budget_and_selects_blind() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: true,
        };
        let quiet = |_: String| {};
        let outcome = run(plan(&engine, &dir, Recursion::Parallel, &quiet))
            .await
            .unwrap();
        assert_eq!((outcome.rollouts, outcome.stages), (8, 6));
        assert_eq!(engine.rollouts.lock().unwrap().len(), 8);
        assert_eq!(outcome.finals, vec!["agg1", "agg2"]);
        // inv1 and inv2 were rejected once and accepted on retry; dom2 never
        // replied with JSON and inherited the base tree.
        assert_eq!(outcome.fallbacks, vec!["dom2"]);
        assert_eq!(
            engine
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|c| c.starts_with("inv"))
                .count(),
            4
        );
        assert!(["agg1", "agg2"].contains(&outcome.chosen.as_str()));
        assert_eq!(outcome.chosen_kind, "final");
        assert!(outcome.similarity.unwrap() < 1.0);
        assert!(outcome.reflection.is_some());
        // dom1 delivered nothing: it is evidence, not a crash.
        assert!(outcome
            .runs
            .iter()
            .any(|r| r.label == "dom1" && !r.delivered));

        // Resume reuses every rollout and stage.
        let again = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let resumed = run(plan(&again, &dir, Recursion::Parallel, &quiet))
            .await
            .unwrap();
        assert!(again.rollouts.lock().unwrap().is_empty());
        assert!(again.calls.lock().unwrap().is_empty());
        assert_eq!(resumed.chosen, outcome.chosen);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn rollout_cache_reuses_only_the_same_tree_and_execution_context() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let quiet = |_: String| {};
        let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
        let ledger = Ledger {
            rollouts: Mutex::new(0),
            stages: Mutex::new(0),
            fallbacks: Mutex::new(vec![]),
        };
        let gate = tokio::sync::Semaphore::new(1);
        let tree = plan.base.clone();

        let first = roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        assert_eq!(first.fingerprint, harness::fingerprint(&tree));
        assert!(first.cache_key.is_some());
        roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 1);

        // Another rollout is independent, and changing base1 does not rerun it.
        roll(&plan, &ledger, &gate, "inv1", &tree).await.unwrap();
        roll(&plan, &ledger, &gate, "inv1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 2);
        let changed = Tree::from([("system.md".to_string(), "Changed".to_string())]);
        roll(&plan, &ledger, &gate, "base1", &changed)
            .await
            .unwrap();
        roll(&plan, &ledger, &gate, "inv1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 3);

        // Restoring the original tree after a different tree was saved reruns
        // it once; subsequent identical resumes are cached again.
        roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 4);

        // A pre-metrics record deserializes, but its missing cache key never
        // authorizes reuse even when the old tree fingerprint matches.
        let legacy_path = dir.join("rollouts/dom1.json");
        std::fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
        std::fs::write(
            &legacy_path,
            serde_json::to_vec(&json!({
                "label":"dom1",
                "fingerprint":harness::fingerprint(&tree),
                "delivered":true,
                "outcome":"Complete",
                "answer":"legacy",
                "changes":"",
                "trajectory":""
            }))
            .unwrap(),
        )
        .unwrap();
        let legacy: RunView = read_json(&legacy_path).unwrap();
        assert!(legacy.cache_key.is_none() && legacy.duration_ms.is_none());
        assert!(legacy.workspace_fingerprint.is_none() && legacy.usage.is_none());
        roll(&plan, &ledger, &gate, "dom1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 5);
        let rerun: RunView = read_json(&legacy_path).unwrap();
        assert!(rerun.cache_key.is_some());

        plan.cache_namespace = "another-model".into();
        roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 6);
        plan.cache_namespace = "another-execution-policy".into();
        roll(&plan, &ledger, &gate, "base1", &tree).await.unwrap();
        assert_eq!(engine.rollouts.lock().unwrap().len(), 7);
        assert!(
            legacy_path.exists(),
            "cache invalidation does not delete a job"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cache_namespace_covers_every_execution_context_component() {
        let settings: Settings = serde_json::from_value(
            json!({"provider":"executor-provider","model":"executor-model","project_id":"p","mode":"auto"}),
        )
        .unwrap();
        let external = External {
            command: "agent run".into(),
            layout: Layout::Codex,
        };
        let make = |task: &str,
                    executor: &Settings,
                    proposer_provider: &str,
                    proposer_model: &str,
                    external: Option<&External>,
                    execution: crate::ExecutionPolicy,
                    workspace: &str,
                    sample: Option<&str>| {
            cache_namespace(
                task,
                executor,
                proposer_provider,
                proposer_model,
                external,
                execution,
                workspace,
                sample,
            )
            .unwrap()
        };
        let base = make(
            "task",
            &settings,
            "proposer-provider",
            "proposer-model",
            Some(&external),
            crate::ExecutionPolicy::files_only(),
            "workspace-v1",
            Some("sample-1"),
        );
        assert_eq!(
            base,
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            )
        );
        let mut executor_settings = settings.clone();
        executor_settings.max_steps += 1;
        let mut executor_model = settings.clone();
        executor_model.model = "other-model".into();
        let mut executor_provider = settings.clone();
        executor_provider.provider = "other-provider".into();
        let other_layout = External {
            command: external.command.clone(),
            layout: Layout::Opencode,
        };
        let other_command = External {
            command: "other-agent run".into(),
            layout: external.layout,
        };
        let variants = vec![
            make(
                "other task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &executor_settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &executor_model,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &executor_provider,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "other-proposer",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "other-proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&other_layout),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&other_command),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                None,
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::trusted_unsandboxed(),
                "workspace-v1",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v2",
                Some("sample-1"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                Some("sample-2"),
            ),
            make(
                "task",
                &settings,
                "proposer-provider",
                "proposer-model",
                Some(&external),
                crate::ExecutionPolicy::files_only(),
                "workspace-v1",
                None,
            ),
        ];
        assert!(
            variants.iter().all(|variant| variant != &base),
            "a context component did not invalidate the cache: {variants:?}"
        );
        assert_eq!(rollout_role("base1"), "base");
        assert_eq!(rollout_role("inv2"), "probe");
        assert_eq!(rollout_role("agg1"), "final");
        assert_eq!(rollout_attempt("validation-base-12"), "12");
        assert_eq!(turn_error_class("Step limit reached (12)"), "step_limit");
        assert_eq!(turn_error_class("Provider stopped"), "provider_error");
        assert_eq!(turn_error_class("Tool cancelled"), "cancelled");
        assert_eq!(turn_error_class("Invalid tool arguments"), "turn_error");
    }

    #[tokio::test]
    async fn sequential_carries_checkpoints_and_falls_back_on_bad_replies() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let quiet = |_: String| {};
        let outcome = run(plan(&engine, &dir, Recursion::Sequential, &quiet))
            .await
            .unwrap();
        assert_eq!((outcome.rollouts, outcome.stages), (8, 6));
        assert_eq!(outcome.finals, vec!["c1", "c2"]);
        // c1 kept the base; c2 adopted inv1, which adds the inv1 skill.
        let calls = engine.calls.lock().unwrap().clone();
        assert_eq!(calls.iter().filter(|c| *c == "select").count(), 1);
        assert!(outcome.stage_notes["c1"].contains("KEEP_BASE"));
        assert_eq!(outcome.fallbacks, vec!["dom2"]);
        assert!(outcome.stage_notes["dom2"].starts_with("rejected twice"));
        // c2 adopted inv1 wholesale from the history of round one.
        let inv1: Stage = read_json(&dir.join("stages/inv1.json")).unwrap();
        let c2: Stage = read_json(&dir.join("stages/c2.json")).unwrap();
        assert_eq!(c2.tree, inv1.tree);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn agentic_stages_get_evidence_files_and_compose_by_copying() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let quiet = |_: String| {};
        let mut p = plan(&engine, &dir, Recursion::Parallel, &quiet);
        p.agentic = true;
        let outcome = run(p).await.unwrap();
        assert_eq!((outcome.rollouts, outcome.stages), (8, 6));
        let inv1: Stage = read_json(&dir.join("stages/inv1.json")).unwrap();
        let agg1: Stage = read_json(&dir.join("stages/agg1.json")).unwrap();
        assert_eq!(
            agg1.tree, inv1.tree,
            "the composer copied inv1 from the pool"
        );
        assert!(
            inv1.tree.contains_key("skills/inv1/SKILL.md")
                && inv1.tree["guardrails.md"].contains("`inv1`")
        );
        // Changing the mode changes every stage key: nothing is reused.
        let again = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        run(plan(&again, &dir, Recursion::Parallel, &quiet))
            .await
            .unwrap();
        assert!(again.calls.lock().unwrap().iter().any(|c| c == "inv1"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn stream_warnings_flag_a_stream_that_never_adapts() {
        let outcome = |kind: &str, fallbacks: Vec<String>| Outcome {
            mode: Recursion::Parallel,
            k: 1,
            chosen: "agg1".into(),
            chosen_kind: kind.into(),
            selection_why: String::new(),
            tree: Tree::new(),
            finals: vec![],
            similarity: None,
            fallbacks,
            rollouts: 4,
            stages: 3,
            runs: vec![],
            stage_notes: BTreeMap::new(),
            reflection: None,
            scores: BTreeMap::new(),
            validation: None,
            warnings: vec![],
        };
        let mut legacy = serde_json::to_value(outcome("base", vec![])).unwrap();
        legacy.as_object_mut().unwrap().remove("validation");
        let legacy: Outcome = serde_json::from_value(legacy).unwrap();
        assert!(legacy.validation.is_none());
        let result = |i: usize, kind: &str, fallbacks: Vec<String>| TaskResult {
            index: i,
            task: format!("t{i}"),
            outcome: outcome(kind, fallbacks),
            revision: None,
            library_entry: None,
        };
        assert!(stream_warnings(&[result(0, "base", vec![])], false).is_empty());
        assert_eq!(
            stream_warnings(&[], true),
            vec![LEGACY_EXTERNAL_WARNING.to_string()]
        );
        let w = stream_warnings(
            &[
                result(0, "base", vec![]),
                result(1, "base", vec!["dom1".into()]),
                result(2, "base", vec![]),
                result(3, "base", vec![]),
                result(4, "final", vec![]),
            ],
            false,
        );
        assert_eq!(
            w,
            vec![
                "The base harness was kept in 4 of 5 tasks.",
                "Stages fell back in 1 of 5 tasks."
            ]
        );
    }

    #[tokio::test]
    async fn rubric_scores_feed_warnings_but_never_selection() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let quiet = |_: String| {};
        let mut p = plan(&engine, &dir, Recursion::Parallel, &quiet);
        p.rubric = Some("Covers every module.".into());
        let outcome = run(p).await.unwrap();
        let calls = engine.calls.lock().unwrap().clone();
        let select = calls.iter().position(|c| c == "select").unwrap();
        assert!(
            calls.iter().position(|c| c.starts_with("judge-")).unwrap() > select,
            "judging happens after selection"
        );
        assert_eq!(
            outcome.scores.len(),
            3,
            "two base rollouts and the chosen final"
        );
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("below the base mean")),
            "{:?}",
            outcome.warnings
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn downstream_caches_do_not_mix_models_or_rollouts() {
        let dir = temp();
        let engine = Scripted {
            calls: Mutex::new(vec![]),
            rollouts: Mutex::new(vec![]),
            reject_first: false,
        };
        let quiet = |_: String| {};
        let mut plan = plan(&engine, &dir, Recursion::Parallel, &quiet);
        let stage = |label: &str| Stage {
            label: label.into(),
            dimension: "AGG".into(),
            tree: Tree::new(),
            notes: format!("{label} notes"),
            fell_back: false,
            error: None,
            key: format!("{label}-stage-key"),
        };
        let run = |label: &str, cache_key: &str| RunView {
            label: label.into(),
            delivered: true,
            outcome: "Complete".into(),
            answer: "completed answer".into(),
            cache_key: Some(cache_key.into()),
            ..RunView::default()
        };
        let bases = vec![run("base1", "base1-v1"), run("base2", "base2-v1")];
        let mut finals = vec![
            (stage("agg1"), run("agg1", "agg1-v1")),
            (stage("agg2"), run("agg2", "agg2-v1")),
        ];
        let stages = vec![finals[0].0.clone(), finals[1].0.clone()];
        let mut candidate_runs = vec![finals[0].1.clone(), finals[1].1.clone()];
        let chosen = select(&plan, &finals).await;
        reflect(
            &plan,
            &bases,
            &stages,
            &candidate_runs,
            &chosen.0,
            &chosen.2,
        )
        .await
        .unwrap();
        score(
            &plan,
            "Complete the task",
            &[&bases[0], &bases[1], &finals[1].1],
        )
        .await;
        assert_eq!(engine.calls.lock().unwrap().len(), 5);

        // An identical resume reuses all three downstream artifacts.
        select(&plan, &finals).await;
        reflect(
            &plan,
            &bases,
            &stages,
            &candidate_runs,
            &chosen.0,
            &chosen.2,
        )
        .await;
        score(
            &plan,
            "Complete the task",
            &[&bases[0], &bases[1], &finals[1].1],
        )
        .await;
        assert_eq!(engine.calls.lock().unwrap().len(), 5);

        // A model/workspace context change invalidates every model decision.
        plan.cache_namespace = "another-model-or-workspace".into();
        select(&plan, &finals).await;
        reflect(
            &plan,
            &bases,
            &stages,
            &candidate_runs,
            &chosen.0,
            &chosen.2,
        )
        .await;
        score(
            &plan,
            "Complete the task",
            &[&bases[0], &bases[1], &finals[1].1],
        )
        .await;
        assert_eq!(engine.calls.lock().unwrap().len(), 10);

        // A new tree under the same execution context also cannot reuse a
        // selection, reflection or score produced for the previous rollout.
        finals[1].1.cache_key = Some("agg2-v2".into());
        candidate_runs[1].cache_key = Some("agg2-v2".into());
        select(&plan, &finals).await;
        reflect(
            &plan,
            &bases,
            &stages,
            &candidate_runs,
            &chosen.0,
            &chosen.2,
        )
        .await;
        score(
            &plan,
            "Complete the task",
            &[&bases[0], &bases[1], &finals[1].1],
        )
        .await;
        let calls = engine.calls.lock().unwrap();
        assert_eq!(calls.iter().filter(|call| *call == "select").count(), 3);
        assert_eq!(calls.iter().filter(|call| *call == "reflect").count(), 3);
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("judge-"))
                .count(),
            9
        );
        drop(calls);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn hook_model(request: &Value) -> Value {
        let results = results_of(request);
        match results.len() {
            0 => call(
                "write_file",
                json!({"path":"workspace/Cargo.lock","content":"x"}),
            ),
            1 => {
                assert!(
                    results[0].contains("Blocked by a harness hook: Never edit the lockfile."),
                    "{}",
                    results[0]
                );
                call("read_file", json!({"path":"workspace/big.txt"}))
            }
            _ => {
                assert!(
                    results[1].contains("[harness hook trimmed")
                        && results[1].chars().count() < 600,
                    "{}",
                    results[1].len()
                );
                say("done")
            }
        }
    }

    /// pre_tool hooks refuse matching calls before they run; post_tool
    /// max_output trims the result the model sees.
    #[tokio::test]
    async fn pre_tool_blocks_and_post_tool_trims_inside_a_real_turn() {
        let f = fixture(hook_model).await;
        std::fs::write(f.workspace.join("big.txt"), "y".repeat(20_000)).unwrap();
        let tree = Tree::from([("hooks.json".to_string(), r#"{"pre_tool":[{"matcher":"write_file","args_contains":"Cargo.lock","block":"Never edit the lockfile."}],"post_tool":[{"matcher":"read_file","max_output":400}]}"#.to_string())]);
        let mut session = Session::new(crate::evaluation::new_id(), auto_settings());
        session.ephemeral = Some(Arc::new(crate::Ephemeral {
            project: f.project.clone(),
            harness: Some(tree),
            system: None,
            nested: false,
            policy: crate::ExecutionPolicy::trusted_unsandboxed(),
        }));
        session
            .messages
            .push(Message::text("user", "Touch the lockfile"));
        let shared = Arc::new(Mutex::new(session));
        crate::turn(f.app.clone(), shared.clone(), None)
            .await
            .unwrap();
        f.server.abort();
        assert!(!f.workspace.join("Cargo.lock").exists());
        assert_eq!(
            shared.lock().unwrap().messages.last().unwrap().content,
            "done"
        );
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    /// A Studio app wired to a fake OpenAI-compatible model whose replies
    /// come from `model(request)`.
    struct Fixture {
        root: PathBuf,
        app: App,
        provider: crate::provider::Provider,
        project: context::Project,
        workspace: PathBuf,
        data: PathBuf,
        server: tokio::task::JoinHandle<()>,
    }
    fn say(text: &str) -> Value {
        json!({"choices":[{"delta":{"content":text},"finish_reason":"stop"}]})
    }
    fn call(name: &str, args: Value) -> Value {
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":format!("c-{name}-{}", crate::evaluation::new_id()),"type":"function","function":{"name":name,"arguments":args.to_string()}}]},"finish_reason":"tool_calls"}]})
    }
    fn system_of(request: &Value) -> String {
        request["messages"][0]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }
    fn tools_of(request: &Value) -> Vec<String> {
        request["tools"]
            .as_array()
            .map(|t| {
                t.iter()
                    .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
    fn results_of(request: &Value) -> Vec<String> {
        request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["content"].as_str().unwrap().to_string())
            .collect()
    }
    async fn fixture(model: fn(&Value) -> Value) -> Fixture {
        use axum::{response::IntoResponse, routing::post, Router};
        let root = temp();
        std::fs::create_dir_all(root.join("workspace")).unwrap();
        std::fs::create_dir_all(root.join("history")).unwrap();
        std::fs::write(root.join("workspace/readme.txt"), "hello").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let router = Router::new().route(
                "/v1/chat/completions",
                post(move |Json(request): Json<Value>| async move {
                    let event = model(&request);
                    (
                        [("content-type", "text/event-stream")],
                        format!("data: {event}\n\ndata: [DONE]\n\n"),
                    )
                        .into_response()
                }),
            );
            axum::serve(listener, router).await.unwrap();
        });
        let data = root.join("history").canonicalize().unwrap();
        let workspace = root.join("workspace").canonicalize().unwrap();
        let mut provider = crate::provider::defaults().remove(0);
        provider.id = "fixture".into();
        provider.base = format!("http://{address}/v1");
        provider.key.clear();
        provider.custom = true;
        let project = context::Project {
            id: "p".into(),
            name: "fixture".into(),
            roots: vec![context::Root {
                alias: "workspace".into(),
                path: workspace.clone(),
                writable: true,
                repository: false,
            }],
            instructions: String::new(),
        };
        let app = App {
            projects: Arc::new(Mutex::new(vec![project.clone()])),
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
            providers: Arc::new(Mutex::new(vec![provider.clone()])),
            key_mutations: Default::default(),
            client: crate::provider::client().unwrap(),
            workspace: Arc::new(workspace.clone()),
            data: Arc::new(data.clone()),
            origin: "http://127.0.0.1:1".into(),
            plugins: Default::default(),
            harness_jobs: Default::default(),
        };
        Fixture {
            root,
            app,
            provider,
            project,
            workspace,
            data,
            server,
        }
    }
    fn auto_settings() -> Settings {
        serde_json::from_value(json!({"provider":"fixture","model":"tiny","project_id":"p","mode":"auto","allow_writes":true})).unwrap()
    }
    fn engine(f: &Fixture, task: &str) -> StudioEngine {
        let mut agent = auto_settings();
        agent.max_steps = 40;
        let scratch = f.root.join("scratch");
        let workspace = WorkspaceSnapshot::capture(&f.project, &scratch.join("snapshot")).unwrap();
        StudioEngine {
            app: f.app.clone(),
            executor: auto_settings(),
            proposer: auto_settings(),
            proposer_agent: agent,
            proposer_provider: f.provider.clone(),
            project: f.project.clone(),
            workspace,
            task: task.into(),
            scratch,
            external: None,
            execution: crate::ExecutionPolicy::files_only(),
            rollout_timeout: std::time::Duration::from_secs(15 * 60),
        }
    }

    fn probe_model(request: &Value) -> Value {
        let (system, tools, results) = (system_of(request), tools_of(request), results_of(request));
        let last = request["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        if tools.is_empty() {
            if system.contains("choosing which of") {
                say(&json!({"why":"only one","choice":"A"}).to_string())
            } else if system.contains("complete record") {
                say(&json!({"edits":[],"experience":null}).to_string())
            } else if system.starts_with("Evaluate the supplied answer") {
                say(&json!({"score":0.5,"reason":"fixture"}).to_string())
            } else {
                say(&json!({"files":{},"notes":"KEEP_BASE: runs agree"}).to_string())
            }
        } else if results.is_empty() {
            assert!(
                system.contains("# Available skills") && tools.contains(&"load_skill".to_string())
            );
            call("load_skill", json!({"name":"note"}))
        } else if results.len() == 1 {
            assert!(results[0].contains("Write out.txt"));
            call(
                "write_file",
                json!({"path":"workspace/out.txt","content":"written by probe"}),
            )
        } else if results.len() == 2 && !last.contains("Pre-completion") {
            assert!(results[1].contains("[harness hook] Reopen what you wrote."));
            say("wrote out.txt")
        } else {
            say("checked: out.txt is complete")
        }
    }

    /// The real engine against a fake model: a probe loads a skill, writes
    /// into its copy of the workspace, sees the hook note, and leaves the
    /// real workspace and the conversation store untouched.
    #[tokio::test]
    async fn studio_engine_probes_on_copies_with_skills_and_hooks() {
        let f = fixture(probe_model).await;
        let base = harness::autotrigger(&Tree::from([
            ("skills/note/SKILL.md".to_string(), "---\nname: note\ndescription: the task asks for a note file\n---\nWrite out.txt with the note.\n".to_string()),
            ("scripts/mark.sh".to_string(), "# Must not run in files-only mode\nprintf pwned > policy-marker.txt\n".to_string()),
            ("hooks.json".to_string(), r#"{"post_tool":[{"matcher":"write_file","inject":"Reopen what you wrote."}],"stop":[{"inject":"Confirm out.txt exists."}]}"#.to_string()),
        ]));
        let engine = engine(&f, "Leave a note");
        let quiet = |_: String| {};
        let outcome = run(Plan {
            engine: &engine,
            dir: f.root.join("job"),
            k: 1,
            mode: Recursion::Parallel,
            task: "Leave a note".into(),
            base: base.clone(),
            cache_namespace: "studio-fixture-cache-v2".into(),
            workspace_fingerprint: engine.workspace.fingerprint.clone(),
            library: String::new(),
            width: 2,
            agentic: false,
            rubric: Some("The note exists.".into()),
            validation_runs: 0,
            reference: None,
            progress: &quiet,
        })
        .await
        .unwrap();
        f.server.abort();
        assert_eq!((outcome.rollouts, outcome.stages), (4, 3));
        assert_eq!(outcome.chosen, "agg1");
        assert_eq!(outcome.tree, base, "KEEP_BASE everywhere keeps the harness");
        assert_eq!(
            outcome.scores,
            BTreeMap::from([
                ("agg1".to_string(), Some(0.5)),
                ("base1".to_string(), Some(0.5))
            ])
        );
        let base1: RunView = read_json(&f.root.join("job/rollouts/base1.json")).unwrap();
        assert!(base1.delivered, "{base1:?}");
        assert_eq!(base1.answer, "checked: out.txt is complete");
        let policy_warning = "Command scripts, command hooks and transforms were disabled by the files-only execution policy.";
        assert_eq!(base1.warnings, vec![policy_warning.to_string()]);
        assert!(render_run(&base1, 2_000).contains(policy_warning));
        assert!(outcome
            .runs
            .iter()
            .find(|run| run.label == "base1")
            .is_some_and(|run| run.warnings == base1.warnings));
        assert!(
            base1
                .changes
                .contains("*** workspace/out.txt (added)\n+written by probe"),
            "{}",
            base1.changes
        );
        assert!(
            base1.trajectory.contains("[ACTION] load_skill")
                && base1
                    .trajectory
                    .contains("Pre-completion check from the harness")
        );
        assert!(
            !f.workspace.join("out.txt").exists(),
            "probes never touch the real workspace"
        );
        assert!(
            !std::fs::read_dir(&f.data)
                .unwrap()
                .flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "json")),
            "probe sessions are not saved"
        );
        assert!(!f.root.join("scratch/base1").exists());
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    #[test]
    fn workspace_snapshots_are_byte_reproducible() {
        let root = temp();
        let original = root.join("original");
        std::fs::create_dir_all(original.join("src")).unwrap();
        std::fs::write(original.join("src/a.txt"), "one\n").unwrap();
        std::fs::write(original.join("z.txt"), "last\n").unwrap();
        let project = context::Project {
            id: "snapshot".into(),
            name: "snapshot".into(),
            roots: vec![context::Root {
                alias: "workspace".into(),
                path: original.canonicalize().unwrap(),
                writable: true,
                repository: false,
            }],
            instructions: String::new(),
        };
        let first = WorkspaceSnapshot::capture(&project, &root.join("one")).unwrap();
        let second = WorkspaceSnapshot::capture(&project, &root.join("two")).unwrap();
        assert_eq!(first.fingerprint, second.fingerprint);
        assert_eq!(first.roots[0].alias, "workspace");
        assert!(first.roots[0].writable);
        assert_eq!(
            std::fs::read(first.roots[0].snapshot.join("src/a.txt")).unwrap(),
            std::fs::read(second.roots[0].snapshot.join("src/a.txt")).unwrap()
        );

        std::fs::write(original.join("src/a.txt"), "two\n").unwrap();
        std::fs::write(original.join("new.txt"), "new\n").unwrap();
        let changed = WorkspaceSnapshot::capture(&project, &root.join("three")).unwrap();
        assert_ne!(first.fingerprint, changed.fingerprint);
        assert_eq!(
            std::fs::read_to_string(first.roots[0].snapshot.join("src/a.txt")).unwrap(),
            "one\n"
        );
        assert!(!first.roots[0].snapshot.join("new.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rollouts_share_a_frozen_snapshot_and_diff_against_it() {
        let f = fixture(probe_model).await;
        let tree = harness::autotrigger(&Tree::from([
            ("skills/note/SKILL.md".to_string(), "---\nname: note\ndescription: the task asks for a note file\n---\nWrite out.txt with the note.\n".to_string()),
            ("hooks.json".to_string(), r#"{"post_tool":[{"matcher":"write_file","inject":"Reopen what you wrote."}],"stop":[{"inject":"Confirm out.txt exists."}]}"#.to_string()),
        ]));
        let engine = engine(&f, "Leave a note");
        let base = engine.rollout("base1", &tree).await.unwrap();
        std::fs::write(f.workspace.join("readme.txt"), "changed later").unwrap();
        std::fs::write(f.workspace.join("original-only.txt"), "not in snapshot").unwrap();
        let champion = engine.rollout("agg1", &tree).await.unwrap();
        f.server.abort();

        assert_eq!(base.workspace_fingerprint, champion.workspace_fingerprint);
        assert_eq!(
            base.workspace_fingerprint.as_deref(),
            Some(engine.workspace.fingerprint.as_str())
        );
        assert!(base.trace_id.is_some() && base.trace_id != champion.trace_id);
        assert!(base.duration_ms.is_some() && base.steps.is_some_and(|steps| steps > 0));
        assert_eq!(base.error_class.as_deref(), Some("complete"));
        assert_eq!(base.changes, champion.changes);
        assert!(base.changes.contains("*** workspace/out.txt (added)"));
        assert!(!base.changes.contains("readme.txt"));
        assert!(!base.changes.contains("original-only.txt"));
        assert_eq!(
            std::fs::read_to_string(f.workspace.join("readme.txt")).unwrap(),
            "changed later"
        );
        assert!(!f.workspace.join("out.txt").exists());
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn timeout_model(_request: &Value) -> Value {
        static REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if REQUESTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0 {
            // The first answer is already in the session; hold up the stop-hook
            // verification so the timeout must preserve that partial answer.
            std::thread::sleep(std::time::Duration::from_secs(3));
        }
        say("partial answer")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_preserves_partial_answer_and_observable_metrics() {
        let f = fixture(timeout_model).await;
        let mut engine = engine(&f, "Answer, then verify");
        engine.rollout_timeout = std::time::Duration::from_secs(1);
        let tree = Tree::from([(
            "hooks.json".to_string(),
            r#"{"stop":[{"inject":"Verify the answer."}]}"#.to_string(),
        )]);
        let run = engine.rollout("base1", &tree).await.unwrap();
        f.server.abort();

        assert_eq!(run.outcome, "timeout");
        assert_eq!(run.error_class.as_deref(), Some("timeout"));
        assert!(!run.delivered);
        assert_eq!(run.answer, "partial answer");
        assert!(run.trace_id.is_some());
        assert!(run.duration_ms.is_some_and(|duration| duration >= 100));
        assert!(run.steps.is_some_and(|steps| steps > 0));
        assert_eq!(
            run.workspace_fingerprint.as_deref(),
            Some(engine.workspace.fingerprint.as_str())
        );
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn proposer_model(request: &Value) -> Value {
        let (system, tools, results) = (system_of(request), tools_of(request), results_of(request));
        assert!(system.contains("YOUR DIMENSION: INV") && system.contains("YOURS TO EDIT"));
        assert!(!tools
            .iter()
            .any(|t| ["run_command", "background", "delegate", "load_skill"]
                .contains(&t.as_str())
                || t.starts_with("mcp_")));
        match results.len() {
            0 => call("read_file", json!({"path":"context/rollouts/base1.md"})),
            1 => {
                assert!(results[0].contains("searched three times"));
                call(
                    "write_file",
                    json!({"path":"harness/skills/find/SKILL.md","content":"---\nname: find\ndescription: the task needs a file located\n---\nSearch with agentgrep once, then read the hit.\n"}),
                )
            }
            2 => call(
                "write_file",
                json!({"path":"harness/memory.md","content":""}),
            ),
            3 => call(
                "write_file",
                json!({"path":"context/task.md","content":"tampered"}),
            ),
            _ => {
                assert!(results[3].contains("read-only"), "{}", results[3]);
                say("base1 searched three times -> skill `find`")
            }
        }
    }

    /// An agentic proposer edits its `harness` root with file tools, cannot
    /// write the evidence or run commands, and its empty files are deletions.
    #[tokio::test]
    async fn agentic_proposer_edits_files_and_cannot_touch_evidence() {
        let f = fixture(proposer_model).await;
        let engine = engine(&f, "Find the config");
        let files = BTreeMap::from([
            ("task.md".to_string(), "Find the config".to_string()),
            (
                "rollouts/base1.md".to_string(),
                "[ACTION] agentgrep — searched three times".to_string(),
            ),
        ]);
        let current = Tree::from([
            ("memory.md".to_string(), "stale".to_string()),
            ("system.md".to_string(), "Be exact.".to_string()),
        ]);
        let system = prompts::propose("INV", 1, 1, None, true);
        let (tree, notes) = engine
            .edit("inv1", &system, &prompts::mission(None), &files, &current)
            .await
            .unwrap();
        f.server.abort();
        assert_eq!(notes, "base1 searched three times -> skill `find`");
        assert!(
            !tree.contains_key("memory.md"),
            "an emptied file is a deletion"
        );
        assert_eq!(tree["system.md"], "Be exact.");
        assert!(tree["skills/find/SKILL.md"].contains("name: find"));
        assert!(accept(&tree).unwrap()["guardrails.md"].contains("load the `find` skill"));
        assert!(
            !f.root
                .join("scratch")
                .read_dir()
                .unwrap()
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().starts_with("propose-")),
            "proposer workspaces are removed; the task snapshot remains"
        );
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn selector_model(request: &Value) -> Value {
        let (system, tools, results) = (system_of(request), tools_of(request), results_of(request));
        assert!(system.contains("CHECK THE CLAIMS AGAINST THE SOURCE"));
        assert!(
            tools.contains(&"read_file".to_string())
                && !tools
                    .iter()
                    .any(|t| ["write_file", "edit_file", "run_command"].contains(&t.as_str()))
        );
        match results.len() {
            0 => call(
                "read_file",
                json!({"path":"harness_evidence/candidates/A.md"}),
            ),
            1 => {
                assert!(results[0].contains("the readme says goodbye"));
                call("read_file", json!({"path":"workspace/readme.txt"}))
            }
            _ => {
                assert!(results[1].contains("hello"));
                say(
                    &json!({"why":"A contradicts readme.txt; B matches it","choice":"B"})
                        .to_string(),
                )
            }
        }
    }

    /// The agentic selector reads the anonymous candidates and checks them
    /// against the project's own files, read-only.
    #[tokio::test]
    async fn agentic_selector_checks_candidates_against_the_source() {
        let f = fixture(selector_model).await;
        let engine = engine(&f, "What does the readme say?");
        std::fs::write(f.workspace.join("readme.txt"), "changed after snapshot").unwrap();
        let files = BTreeMap::from([
            (
                "task.md".to_string(),
                "What does the readme say?".to_string(),
            ),
            (
                "candidates/A.md".to_string(),
                "# Final answer\nthe readme says goodbye\n".to_string(),
            ),
            (
                "candidates/B.md".to_string(),
                "# Final answer\nthe readme says hello\n".to_string(),
            ),
        ]);
        let answer = engine
            .inspect(
                "select",
                &prompts::select(2, true),
                prompts::select_mission(),
                &files,
            )
            .await
            .unwrap();
        f.server.abort();
        assert_eq!(json_object(&answer).unwrap()["choice"], "B");
        assert_eq!(
            std::fs::read_to_string(f.workspace.join("readme.txt")).unwrap(),
            "changed after snapshot",
            "the selector read the snapshot, while the original stayed untouched"
        );
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn transform_model(request: &Value) -> Value {
        let results = results_of(request);
        if results.is_empty() {
            return call("read_file", json!({"path":"workspace/readme.txt"}));
        }
        assert!(
            results[0].contains("HELLO") && !results[0].contains("hello"),
            "{}",
            results[0]
        );
        say("done")
    }

    /// A post_tool transform is code in the loop: the tool result goes to
    /// its stdin and its stdout is what the model sees.
    #[tokio::test]
    async fn post_tool_transform_rewrites_results_with_code() {
        let f = fixture(transform_model).await;
        let cmd = crate::plugins::defaults()
            .into_iter()
            .find(|p| p.unlocks_tool.as_deref() == Some("run_command"))
            .unwrap();
        f.app.plugins.write().unwrap().insert(
            cmd.id.clone(),
            crate::plugins::PluginState {
                config: cmd,
                client: None,
                error: None,
            },
        );
        let tree = Tree::from([(
            "hooks.json".to_string(),
            r#"{"post_tool":[{"matcher":"read_file","transform":"tr a-z A-Z"}]}"#.to_string(),
        )]);
        let mut session = Session::new(crate::evaluation::new_id(), auto_settings());
        session.ephemeral = Some(Arc::new(crate::Ephemeral {
            project: f.project.clone(),
            harness: Some(tree),
            system: None,
            nested: false,
            policy: crate::ExecutionPolicy::trusted_unsandboxed(),
        }));
        session
            .messages
            .push(Message::text("user", "Read the readme"));
        let shared = Arc::new(Mutex::new(session));
        crate::turn(f.app.clone(), shared.clone(), None)
            .await
            .unwrap();
        f.server.abort();
        assert_eq!(
            shared.lock().unwrap().messages.last().unwrap().content,
            "done"
        );
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn harness_with_everything() -> Tree {
        harness::autotrigger(&Tree::from([
            ("system.md".to_string(), "Be exact.".to_string()),
            ("skills/note/SKILL.md".to_string(), "---\nname: note\ndescription: the task asks for a note\n---\nRun skills/note/scripts/check.sh after writing.\n".to_string()),
            ("skills/note/scripts/check.sh".to_string(), "# Check the note\ntest -s note.txt\n".to_string()),
            ("scripts/count.py".to_string(), "\"\"\"Count lines.\"\"\"\n".to_string()),
            ("subagents/reader/AGENT.md".to_string(), "---\ndescription: reads one file\n---\nRead it.\n".to_string()),
        ]))
    }

    #[test]
    fn harness_materializes_for_codex_and_opencode() {
        let tree = harness_with_everything();
        let codex = materialize(&tree, Layout::Codex, Some("# Repo rules\nUse cargo."));
        assert!(codex["AGENTS.md"].starts_with("# Repo rules\nUse cargo.\n\nBe exact."));
        assert!(codex["AGENTS.md"]
            .contains("- note: the task asks for a note (.agents/skills/note/SKILL.md)"));
        assert!(codex["AGENTS.md"]
            .contains("# Delegation playbooks\n## reader — reads one file\nRead it."));
        assert!(
            codex["AGENTS.md"].contains("- .agents/skills/note/scripts/check.sh: Check the note")
        );
        assert!(
            !codex["AGENTS.md"].contains("load_skill") && !codex["AGENTS.md"].contains("delegate")
        );
        assert!(
            codex.contains_key(".agents/skills/note/SKILL.md")
                && codex.contains_key("scripts/count.py")
        );
        let opencode = materialize(&tree, Layout::Opencode, None);
        assert_eq!(
            opencode[".opencode/agent/reader.md"],
            "---\ndescription: reads one file\nmode: subagent\n---\nRead it."
        );
        assert!(opencode.contains_key(".opencode/skills/note/scripts/check.sh"));
        assert!(opencode["AGENTS.md"].contains("# Available subagents\n- reader: reads one file"));
        assert!(
            materialize(&Tree::new(), Layout::Codex, Some("own")).is_empty(),
            "an empty harness leaves the workspace alone"
        );
    }

    fn unused_model(_: &Value) -> Value {
        panic!("an external executor makes no model calls")
    }

    /// An external agent CLI runs in a copy of the workspace with the
    /// harness laid out for it; only what it changed is reported.
    #[tokio::test]
    async fn external_executor_runs_a_cli_on_the_materialized_harness() {
        let f = fixture(unused_model).await;
        std::fs::write(f.workspace.join("AGENTS.md"), "# Repo rules\n").unwrap();
        let mut engine = engine(&f, "Leave a note");
        engine.external = Some(External {
            command: "cat > task.txt; test -f .agents/skills/note/SKILL.md && grep -q 'Repo rules' AGENTS.md && grep -q 'Be exact' AGENTS.md && echo \"skill ok: $ALLPAKA_TASK\"".into(),
            layout: Layout::Codex,
        });
        let blocked = engine.rollout("blocked", &Tree::new()).await.unwrap_err();
        assert!(blocked.to_string().contains("trusted_unsandboxed"));
        assert!(!f.workspace.join("task.txt").exists());
        engine.execution = crate::ExecutionPolicy::trusted_unsandboxed();
        let run = engine
            .rollout("base1", &harness_with_everything())
            .await
            .unwrap();
        f.server.abort();
        assert_eq!(
            (run.outcome.as_str(), run.answer.as_str(), run.delivered),
            ("Complete", "skill ok: Leave a note", true)
        );
        assert_eq!(
            run.changes, "*** workspace/task.txt (added)\n+Leave a note\n",
            "materialized files are not changes"
        );
        assert!(run.trajectory.starts_with("[EXTERNAL] cat > task.txt"));
        assert!(run.trace_id.is_none() && run.steps.is_none());
        assert!(run.duration_ms.is_some());
        assert_eq!(run.usage.as_ref().unwrap()["available"], false);
        assert_eq!(run.error_class.as_deref(), Some("complete"));
        assert!(!f.workspace.join("task.txt").exists());
        assert_eq!(
            std::fs::read_to_string(f.workspace.join("AGENTS.md")).unwrap(),
            "# Repo rules\n"
        );
        engine.external.as_mut().unwrap().command = "echo partial; exit 3".into();
        let failed = engine.rollout("base2", &Tree::new()).await.unwrap();
        assert_eq!(
            (failed.outcome.as_str(), failed.delivered),
            ("exit Some(3)", false)
        );
        assert_eq!(failed.error_class.as_deref(), Some("external_exit"));
        assert_eq!(failed.answer, "partial");
        engine.external.as_mut().unwrap().command = ":".into();
        let empty = engine.rollout("base3", &Tree::new()).await.unwrap();
        assert_eq!(empty.error_class.as_deref(), Some("empty_answer"));
        assert!(!empty.delivered);
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn delegate_model(request: &Value) -> Value {
        let (system, tools, results) = (system_of(request), tools_of(request), results_of(request));
        if system.contains("You are the `reader` subagent") {
            assert!(
                !tools.contains(&"delegate".to_string()),
                "subagents cannot delegate again"
            );
            assert!(
                !tools.contains(&"write_file".to_string()),
                "reader has writes: false"
            );
            return if results.is_empty() {
                call("read_file", json!({"path":"workspace/readme.txt"}))
            } else {
                say("readme says hello")
            };
        }
        assert!(
            system.contains("# Available subagents\n")
                || system.contains("# Available subagents (")
        );
        if results.is_empty() {
            call(
                "delegate",
                json!({"agent":"reader","task":"Read workspace/readme.txt and summarise it."}),
            )
        } else {
            assert!(results[0].contains("readme says hello"), "{}", results[0]);
            say("the subagent read it")
        }
    }

    /// `delegate` runs the subagent as a nested turn with its own prompt,
    /// without delegation or (for `writes: false`) write tools, and returns
    /// its answer to the parent.
    #[tokio::test]
    async fn delegate_runs_a_nested_subagent_turn() {
        let f = fixture(delegate_model).await;
        let tree = Tree::from([("subagents/reader/AGENT.md".to_string(), "---\ndescription: reads one file and summarises it\nwrites: false\n---\nRead exactly what you are asked to and summarise it in one line.\n".to_string())]);
        let mut session = Session::new(crate::evaluation::new_id(), auto_settings());
        session.ephemeral = Some(Arc::new(crate::Ephemeral {
            project: f.project.clone(),
            harness: Some(tree),
            system: None,
            nested: false,
            policy: crate::ExecutionPolicy::trusted_unsandboxed(),
        }));
        session
            .messages
            .push(Message::text("user", "What does the readme say?"));
        let shared = Arc::new(Mutex::new(session));
        let outcome = crate::turn(f.app.clone(), shared.clone(), None)
            .await
            .unwrap();
        f.server.abort();
        assert!(matches!(outcome, crate::TurnOutcome::Complete));
        let state = shared.lock().unwrap();
        assert_eq!(
            state.messages.last().unwrap().content,
            "the subagent read it"
        );
        let tool = state.messages.iter().find(|m| m.role == "tool").unwrap();
        let value: Value = serde_json::from_str(&tool.content).unwrap();
        assert_eq!(
            (
                value["agent"].as_str(),
                value["outcome"].as_str(),
                value["answer"].as_str()
            ),
            (Some("reader"), Some("Complete"), Some("readme says hello"))
        );
        drop(state);
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    fn files_only_model(request: &Value) -> Value {
        let (system, tools, results) = (system_of(request), tools_of(request), results_of(request));
        let command = json!({"command":"printf pwned > policy-marker.txt"});
        if system.contains("You are the `reader` subagent") {
            assert!(tools.iter().all(|name| ![
                "run_command",
                "run_script",
                "background",
                "delegate"
            ]
            .contains(&name.as_str())));
            return if results.is_empty() {
                call("run_command", command)
            } else {
                assert!(results[0].contains("error"), "{}", results[0]);
                say("child could not execute")
            };
        }
        assert!(tools.contains(&"delegate".to_string()));
        assert!(tools
            .iter()
            .all(|name| !["run_command", "run_script", "background"].contains(&name.as_str())));
        match results.len() {
            0 => call("run_command", command),
            1 => {
                assert!(results[0].contains("error"), "{}", results[0]);
                call(
                    "background",
                    json!({"action":"start","command":"printf pwned > policy-marker.txt"}),
                )
            }
            2 => {
                assert!(
                    results[1].contains("Background commands require"),
                    "{}",
                    results[1]
                );
                call("run_script", json!({"path":"scripts/mark.sh","args":[]}))
            }
            3 => {
                assert!(results[2].contains("error"), "{}", results[2]);
                call(
                    "delegate",
                    json!({"agent":"reader","task":"Try the command and report."}),
                )
            }
            4 => {
                assert!(
                    results[3].contains("child could not execute"),
                    "{}",
                    results[3]
                );
                say("policy enforced")
            }
            _ => panic!("unexpected files-only request"),
        }
    }

    /// Hiding schemas is not a security boundary: direct calls and delegated
    /// turns must enforce the same files-only policy with commands unlocked.
    #[tokio::test]
    async fn files_only_blocks_direct_commands_background_scripts_and_subagent_bypass() {
        let f = fixture(files_only_model).await;
        let cmd = crate::plugins::defaults()
            .into_iter()
            .find(|p| p.unlocks_tool.as_deref() == Some("run_command"))
            .unwrap();
        f.app.plugins.write().unwrap().insert(
            cmd.id.clone(),
            crate::plugins::PluginState {
                config: cmd,
                client: None,
                error: None,
            },
        );
        let tree = Tree::from([
            (
                "scripts/mark.sh".to_string(),
                "printf pwned > policy-marker.txt\n".to_string(),
            ),
            (
                "subagents/reader/AGENT.md".to_string(),
                "---\ndescription: tests inherited restrictions\nwrites: false\n---\nDo what the parent asks.\n".to_string(),
            ),
        ]);
        let mut session = Session::new(crate::evaluation::new_id(), auto_settings());
        session.ephemeral = Some(Arc::new(crate::Ephemeral {
            project: f.project.clone(),
            harness: Some(tree),
            system: None,
            nested: false,
            policy: crate::ExecutionPolicy::files_only(),
        }));
        session
            .messages
            .push(Message::text("user", "Try every execution path"));
        let shared = Arc::new(Mutex::new(session));
        let outcome = crate::turn(f.app.clone(), shared.clone(), None)
            .await
            .unwrap();
        f.server.abort();
        assert!(matches!(outcome, crate::TurnOutcome::Complete));
        assert_eq!(
            shared.lock().unwrap().messages.last().unwrap().content,
            "policy enforced"
        );
        assert!(!f.workspace.join("policy-marker.txt").exists());
        std::fs::remove_dir_all(&f.root).unwrap();
    }

    #[test]
    fn replies_are_parsed_strictly_and_applied_with_validation() {
        let base = Tree::from([("system.md".to_string(), "x".to_string())]);
        let reply = parse_reply(
            "```json\n{\"files\":{\"system.md\":null,\"memory.md\":\"m\"},\"notes\":\"n\"}\n```",
        )
        .unwrap();
        let tree = apply(&reply, &base, &[]).unwrap();
        assert_eq!(
            tree,
            Tree::from([("memory.md".to_string(), "m".to_string())])
        );
        assert!(parse_reply("{\"files\":{},\"adopt\":\"x\",\"extra\":1}").is_err());
        assert!(parse_reply("{\"files\":{\"a\":\"b\"},\"adopt\":\"x\"}").is_err());
        let bad = parse_reply("{\"files\":{\"notes.txt\":\"x\"}}").unwrap();
        assert!(apply(&bad, &base, &[])
            .unwrap_err()
            .to_string()
            .contains("not a harness path"));
        let adopt = parse_reply("{\"adopt\":\"inv9\"}").unwrap();
        assert!(apply(&adopt, &base, &[]).is_err());
    }

    #[test]
    fn trajectory_marks_actions_observations_and_errors() {
        let mut call = Message::text("assistant", "Looking");
        call.tool_calls =
            vec![json!({"id":"1","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}})];
        let text = trajectory(&[
            Message::text("user", "task"),
            call,
            Message {
                role: "tool".into(),
                content: "{\"error\":\"File not found\"}".into(),
                ..Message::default()
            },
            Message::text("assistant", "done"),
        ]);
        assert_eq!(text, "[USER] task\n[ASSISTANT] Looking\n[ACTION] read_file({\"path\":\"a\"})\n[ERROR] {\"error\":\"File not found\"}\n[ASSISTANT] done\n");
        assert!(clip(&"x".repeat(100), 30).contains("omitted"));
    }

    #[test]
    fn probe_changes_are_diffed_against_the_snapshot() {
        let dir = temp();
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::create_dir_all(a.join("src")).unwrap();
        std::fs::write(a.join("src/x.txt"), "one\n").unwrap();
        std::fs::write(a.join("gone.txt"), "g").unwrap();
        std::fs::create_dir_all(a.join(".git")).unwrap();
        let (mut files, mut bytes) = (0, 0);
        copy_root(&a, &b, &mut files, &mut bytes).unwrap();
        assert!(!b.join(".git").exists());
        std::fs::write(b.join("src/x.txt"), "two\n").unwrap();
        std::fs::write(b.join("new.txt"), "n\n").unwrap();
        std::fs::remove_file(b.join("gone.txt")).unwrap();
        let text = changes("ws", &a, &b);
        assert!(
            text.contains("*** ws/src/x.txt (modified)\n-one\n+two\n"),
            "{text}"
        );
        assert!(text.contains("*** ws/new.txt (added)\n+n\n"));
        assert!(text.contains("*** ws/gone.txt (removed)"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
