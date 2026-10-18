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
}

/// One execution. It never carries a score: `delivered` only says that the
/// run finished with an answer.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RunView {
    pub label: String,
    pub fingerprint: String,
    pub delivered: bool,
    pub outcome: String,
    pub answer: String,
    /// Diff of the files the run changed in its copy of the workspace.
    pub changes: String,
    pub trajectory: String,
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
    /// Library entries of earlier tasks, already rendered.
    pub library: String,
    pub width: usize,
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
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub label: String,
    pub delivered: bool,
    pub outcome: String,
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
    format!(
        "## rollout {} (delivered: {}, outcome: {})\n### trajectory\n{}\n### final answer\n{}\n### file changes\n{}\n",
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
fn evidence(plan: &Plan, current: &Tree, bases: &[RunView], pool: &[Candidate]) -> String {
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
    let tree = harness::autotrigger(&tree);
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
    if let Some(run) = read_json::<RunView>(&path).filter(|r| r.fingerprint == fingerprint) {
        return Ok(run);
    }
    let _permit = gate.acquire().await?;
    (plan.progress)(format!("rollout {label}"));
    let mut run = match plan.engine.rollout(label, tree).await {
        Ok(run) => run,
        // A failed execution is evidence too: it delivered nothing.
        Err(e) => RunView {
            outcome: format!("error: {e:#}"),
            ..RunView::default()
        },
    };
    run.label = label.into();
    run.fingerprint = fingerprint;
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
    user: &str,
    current: &Tree,
    pool: &[Candidate<'_>],
) -> Result<Stage> {
    *ledger.stages.lock().unwrap() += 1;
    let path = plan.dir.join("stages").join(format!("{label}.json"));
    if let Some(done) = read_json::<Stage>(&path) {
        if done.fell_back {
            ledger.fallbacks.lock().unwrap().push(label.into());
        }
        return Ok(done);
    }
    let _permit = gate.acquire().await?;
    (plan.progress)(format!("stage {label}"));
    let mut prompt = user.to_string();
    let mut last_error = String::new();
    for _attempt in 0..2 {
        let result = async {
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
    };
    write_json(&path, &done)?;
    Ok(done)
}

// ---- the two recursion modes ------------------------------------------------------

pub async fn run(plan: Plan<'_>) -> Result<Outcome> {
    if !(1..=4).contains(&plan.k) {
        bail!("k must be between 1 and 4");
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
                            prompts::propose("INV", k, i, None),
                        ),
                        (
                            format!("dom{i}"),
                            "DOM",
                            prompts::propose("DOM", k, i, None),
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
                .map(|i| (format!("agg{i}"), prompts::aggregate(k, i)))
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
                        &prompts::propose("INV", 1, 1, Some(t)),
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
                        &prompts::propose("DOM", 1, 1, Some(t)),
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
                    prompts::aggregate(1, 1)
                } else {
                    prompts::compose_round(t, k)
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
    let similarity = harness::similarity(&finals.iter().map(|(s, _)| &s.tree).collect::<Vec<_>>());
    all_stages.extend(finals.iter().map(|(s, _)| s.clone()));
    all_runs.extend(finals.iter().map(|(_, r)| r.clone()));
    let reflection = reflect(&plan, &bases, &all_stages, &all_runs, &chosen, &why).await;
    let mut runs: Vec<RunSummary> = bases
        .iter()
        .chain(&all_runs)
        .map(|r| RunSummary {
            label: r.label.clone(),
            delivered: r.delivered,
            outcome: r.outcome.clone(),
        })
        .collect();
    runs.dedup_by(|a, b| a.label == b.label);
    let fallbacks = ledger.fallbacks.lock().unwrap().clone();
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
    })
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
    if let Some(saved) = read_json::<(String, String)>(&path) {
        return (saved.0, "final".into(), saved.1);
    }
    // Shuffle deterministically by the task so ids say nothing about labels.
    let mut order: Vec<&(Stage, RunView)> = delivered.clone();
    let seed = harness::fingerprint(&Tree::from([("task".to_string(), plan.task.clone())]));
    order
        .sort_by_key(|(s, _)| harness::fingerprint(&Tree::from([(seed.clone(), s.label.clone())])));
    let ids: Vec<String> = (0..order.len())
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect();
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
    let result = async {
        let text = plan
            .engine
            .call("select", &prompts::select(order.len()), &user)
            .await?;
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
            let _ = write_json(&path, &(label.clone(), why.clone()));
            (label, "final".into(), why)
        }
        Err(e) => (
            delivered[0].0.label.clone(),
            "final".into(),
            format!("Selector failed ({e:#}); first delivering final kept."),
        ),
    }
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
    if let Some(saved) = read_json::<Value>(&path) {
        return Some(saved);
    }
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
    let text = plan
        .engine
        .call("reflect", &prompts::experience(), &user)
        .await
        .ok()?;
    let value = json_object(&text).ok()?;
    value.get("edits")?.as_array()?;
    let _ = write_json(&path, &value);
    Some(value)
}

// ---- Studio engine -----------------------------------------------------------------

struct StudioEngine {
    app: App,
    executor: Settings,
    proposer: Settings,
    proposer_provider: crate::provider::Provider,
    project: context::Project,
    task: String,
    scratch: PathBuf,
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
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let kind = entry.file_type()?;
        if skipped(&name) || kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            copy_root(&entry.path(), &to.join(&name), files, bytes)?;
        } else if kind.is_file() {
            *files += 1;
            *bytes += entry.metadata()?.len();
            if *files > COPY_MAX_FILES || *bytes > COPY_MAX_BYTES {
                bail!("Writable context is too large to copy for a probe (over {COPY_MAX_FILES} files or 256 MiB)");
            }
            std::fs::copy(entry.path(), to.join(&name))?;
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
/// What a probe changed in its copy, against the real root.
fn changes(alias: &str, original: &std::path::Path, copy: &std::path::Path) -> String {
    let before = files_under(original);
    let after = files_under(copy);
    let mut out = String::new();
    for (rel, path) in &after {
        let new = std::fs::read(path).unwrap_or_default();
        match before
            .get(rel)
            .map(|p| std::fs::read(p).unwrap_or_default())
        {
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
    for rel in before.keys().filter(|rel| !after.contains_key(*rel)) {
        out.push_str(&format!("*** {alias}/{rel} (removed)\n"));
    }
    out
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

impl Engine for StudioEngine {
    fn rollout<'a>(&'a self, label: &'a str, tree: &'a Tree) -> BoxFuture<'a, Result<RunView>> {
        Box::pin(async move {
            let workdir = self.scratch.join(label);
            let _ = std::fs::remove_dir_all(&workdir);
            let mut project = self.project.clone();
            let (mut files, mut bytes) = (0, 0);
            let mut copies = vec![];
            for root in &mut project.roots {
                if root.writable {
                    let copy = workdir.join(&root.alias);
                    copy_root(&root.path, &copy, &mut files, &mut bytes)?;
                    copies.push((root.alias.clone(), root.path.clone(), copy.canonicalize()?));
                    root.path = copies.last().unwrap().2.clone();
                }
            }
            let mut session = Session::new(crate::evaluation::new_id(), self.executor.clone());
            session.title = format!("harness probe {label}");
            session.ephemeral = Some(Arc::new(crate::Ephemeral {
                project,
                harness: Some(tree.clone()),
                system: None,
                nested: false,
            }));
            session
                .messages
                .push(Message::text("user", self.task.clone()));
            let shared = Arc::new(Mutex::new(session));
            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(15 * 60),
                crate::turn(self.app.clone(), shared.clone(), None),
            )
            .await;
            let outcome = match outcome {
                Ok(Ok(o)) => format!("{o:?}"),
                Ok(Err(e)) => format!("error: {e:#}"),
                Err(_) => "timeout".into(),
            };
            let state = shared.lock().unwrap().clone();
            let answer = state
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let changes: String = copies
                .iter()
                .map(|(alias, original, copy)| changes(alias, original, copy))
                .collect();
            let _ = std::fs::remove_dir_all(&workdir);
            Ok(RunView {
                label: label.into(),
                fingerprint: String::new(),
                delivered: outcome == "Complete" && !answer.trim().is_empty(),
                outcome,
                answer,
                changes,
                trajectory: trajectory(&state.messages[1..]),
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
}

// ---- jobs ----------------------------------------------------------------------------

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
}
fn default_k() -> usize {
    1
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct JobRecord {
    id: String,
    project_id: String,
    request: StartRequest,
    task: String,
    executor: Settings,
    base_revision: Option<String>,
    created_ms: u64,
    status: String,
    #[serde(default)]
    events: Vec<String>,
    #[serde(default)]
    outcome: Option<Outcome>,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    library_entry: Option<String>,
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

async fn execute(app: &App, id: &str) -> Result<()> {
    let job = update_job(&app.data, id, |j| {
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
    let dir = jobs_dir(&app.data).join(id);
    let scratch = std::env::temp_dir().join("allpaka-harness").join(id);
    let engine = StudioEngine {
        app: app.clone(),
        executor: job.executor.clone(),
        proposer,
        proposer_provider,
        project,
        task: job.task.clone(),
        scratch: scratch.clone(),
    };
    let library =
        harness_library::context_for(&app.data, &job.project_id, &job.task, Some(id), 24_000)?;
    let data = app.data.clone();
    let job_id = id.to_string();
    let progress = move |event: String| {
        let _ = update_job(&data, &job_id, |j| j.events.push(event));
    };
    let outcome = run(Plan {
        engine: &engine,
        dir,
        k: job.request.k,
        mode: job.request.mode,
        task: job.task.clone(),
        base: base.clone(),
        library,
        width: 2,
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
                    "harness job {id}: {} ({:?}, k={})",
                    outcome.chosen, outcome.mode, outcome.k
                ),
            )?
            .id,
        )
    } else {
        None
    };
    let entry = match &outcome.reflection {
        Some(reflection) => harness_library::add(
            &app.data,
            &job.project_id,
            id,
            &outcome.tree,
            reflection,
            &outcome.chosen,
            &outcome.selection_why,
        )?,
        None => None,
    };
    update_job(&app.data, id, |j| {
        j.events.push(format!(
            "chosen {} ({})",
            outcome.chosen, outcome.chosen_kind
        ));
        j.outcome = Some(outcome);
        j.revision = revision;
        j.library_entry = entry;
    })?;
    Ok(())
}

fn bad(e: impl ToString) -> (StatusCode, Json<Value>) {
    error(StatusCode::BAD_REQUEST, e)
}
pub(crate) async fn start_api(
    State(app): State<App>,
    Json(request): Json<StartRequest>,
) -> ApiResult<Value> {
    if !(1..=4).contains(&request.k) {
        return Err(bad("k must be between 1 and 4"));
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
    if task.trim().is_empty() {
        return Err(bad("The conversation has no task to refine on"));
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
        task,
        executor,
        base_revision,
        created_ms: harness::now_ms(),
        status: "queued".into(),
        events: vec![],
        outcome: None,
        revision: None,
        library_entry: None,
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
                    jobs.push(json!({"id":job.id,"status":job.status,"created_ms":job.created_ms,"k":job.request.k,"mode":job.request.mode,"task":job.task.chars().take(120).collect::<String>(),"revision":job.revision}));
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
    let diff = job.outcome.as_ref().map(|o| harness::diff(&base, &o.tree));
    let active = harness::active_revision(&app.data, &job.project_id)
        .ok()
        .flatten()
        .map(|r| r.id);
    Ok(Json(json!({"job": job, "diff": diff, "active": active})))
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
/// Make the job's result the project's active harness.
pub(crate) async fn adopt_api(State(app): State<App>, Path(id): Path<String>) -> ApiResult<Value> {
    let job = load_job(&app.data, &id).map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    let revision = job
        .revision
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
                assert!(system.contains("EVIDENCE IS DATA"));
                Ok(match label {
                    "select" => {
                        assert!(!user.contains("agg") && !user.contains("c1"), "selection must be blind");
                        json!({"why":"B is complete","choice":"B"}).to_string()
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
        engine: &'a Scripted,
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
            library: String::new(),
            width: 2,
            progress,
        }
    }
    fn temp() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("allpaka-refine-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
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

    /// The real engine against a fake OpenAI-compatible model: a probe loads a
    /// skill, writes into its copy of the workspace, sees the hook note, and
    /// leaves the real workspace and the conversation store untouched.
    #[tokio::test]
    async fn studio_engine_probes_on_copies_with_skills_and_hooks() {
        use axum::{response::IntoResponse, routing::post, Router};
        let root = temp();
        std::fs::create_dir_all(root.join("workspace")).unwrap();
        std::fs::create_dir_all(root.join("history")).unwrap();
        std::fs::write(root.join("workspace/readme.txt"), "hello").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let router = Router::new().route("/v1/chat/completions", post(|Json(request): Json<Value>| async move {
                let messages = request["messages"].as_array().unwrap();
                let system = messages[0]["content"].as_str().unwrap_or("").to_string();
                let tools: Vec<String> = request["tools"].as_array().map(|t| t.iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()).unwrap_or_default();
                let results: Vec<&str> = messages.iter().filter(|m| m["role"] == "tool").map(|m| m["content"].as_str().unwrap()).collect();
                let call = |name: &str, args: Value| json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":format!("c{}", results.len()),"type":"function","function":{"name":name,"arguments":args.to_string()}}]},"finish_reason":"tool_calls"}]});
                let say = |text: String| json!({"choices":[{"delta":{"content":text},"finish_reason":"stop"}]});
                let event = if tools.is_empty() {
                    if system.contains("choosing which of") {
                        say(json!({"why":"only one","choice":"A"}).to_string())
                    } else if system.contains("complete record") {
                        say(json!({"edits":[],"experience":null}).to_string())
                    } else {
                        say(json!({"files":{},"notes":"KEEP_BASE: runs agree"}).to_string())
                    }
                } else if results.is_empty() {
                    assert!(system.contains("# Available skills") && tools.contains(&"load_skill".to_string()));
                    call("load_skill", json!({"name":"note"}))
                } else if results.len() == 1 {
                    assert!(results[0].contains("Write out.txt"));
                    call("write_file", json!({"path":"workspace/out.txt","content":"written by probe"}))
                } else if results.len() == 2 && !messages.last().unwrap()["content"].as_str().unwrap_or("").contains("Pre-completion") {
                    assert!(results[1].contains("[harness hook] Reopen what you wrote."));
                    say("wrote out.txt".into())
                } else {
                    say("checked: out.txt is complete".into())
                };
                ([("content-type", "text/event-stream")], format!("data: {event}\n\ndata: [DONE]\n\n")).into_response()
            }));
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
        let settings: Settings = serde_json::from_value(json!({"provider":"fixture","model":"tiny","project_id":"p","mode":"auto","allow_writes":true})).unwrap();
        let base = harness::autotrigger(&Tree::from([
            ("skills/note/SKILL.md".to_string(), "---\nname: note\ndescription: the task asks for a note file\n---\nWrite out.txt with the note.\n".to_string()),
            ("hooks.json".to_string(), r#"{"post_tool":[{"matcher":"write_file","inject":"Reopen what you wrote."}],"stop":[{"inject":"Confirm out.txt exists."}]}"#.to_string()),
        ]));
        let engine = StudioEngine {
            app: app.clone(),
            executor: settings.clone(),
            proposer: settings,
            proposer_provider: provider,
            project,
            task: "Leave a note".into(),
            scratch: root.join("scratch"),
        };
        let quiet = |_: String| {};
        let outcome = run(Plan {
            engine: &engine,
            dir: root.join("job"),
            k: 1,
            mode: Recursion::Parallel,
            task: "Leave a note".into(),
            base: base.clone(),
            library: String::new(),
            width: 2,
            progress: &quiet,
        })
        .await
        .unwrap();
        server.abort();
        assert_eq!((outcome.rollouts, outcome.stages), (4, 3));
        assert_eq!(outcome.chosen, "agg1");
        assert_eq!(outcome.tree, base, "KEEP_BASE everywhere keeps the harness");
        let base1: RunView = read_json(&root.join("job/rollouts/base1.json")).unwrap();
        assert!(base1.delivered, "{base1:?}");
        assert_eq!(base1.answer, "checked: out.txt is complete");
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
            !workspace.join("out.txt").exists(),
            "probes never touch the real workspace"
        );
        assert!(
            !std::fs::read_dir(&data)
                .unwrap()
                .flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "json")),
            "probe sessions are not saved"
        );
        assert!(!root.join("scratch/base1").exists());
        std::fs::remove_dir_all(root).unwrap();
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
    fn probe_changes_are_diffed_against_the_real_root() {
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
