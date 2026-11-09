use super::*;

// ---- evidence ----------------------------------------------------------------

pub(super) fn clip(text: &str, limit: usize) -> String {
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
pub(super) fn render_tree(tree: &Tree) -> String {
    if tree.is_empty() {
        return "(empty harness: no files)\n".into();
    }
    tree.iter()
        .map(|(path, text)| format!("=== {path} ===\n{}\n", clip(text, 12_000)))
        .collect()
}
pub(super) fn render_run(run: &RunView, budget: usize) -> String {
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
pub(super) struct Candidate<'a> {
    pub(super) stage: &'a Stage,
    pub(super) run: &'a RunView,
}
/// The same evidence as one message (JSON proposers) and as files under
/// `context/` (agentic proposers).
pub struct Evidence {
    pub(super) text: String,
    pub(super) files: BTreeMap<String, String>,
}
pub(super) fn evidence(plan: &Plan, current: &Tree, bases: &[RunView], pool: &[Candidate]) -> Evidence {
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
pub(super) fn evidence_text(plan: &Plan, current: &Tree, bases: &[RunView], pool: &[Candidate]) -> String {
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
pub(super) struct Reply {
    #[serde(default)]
    pub(super) files: BTreeMap<String, Option<String>>,
    #[serde(default)]
    pub(super) adopt: Option<String>,
    #[serde(default)]
    pub(super) notes: String,
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
pub(super) fn parse_reply(text: &str) -> Result<Reply> {
    let reply: Reply = serde_json::from_value(json_object(text)?).context(
        "Reply must be {\"files\":{…},\"notes\":\"…\"} or {\"adopt\":\"label\",\"notes\":\"…\"}",
    )?;
    if reply.adopt.is_some() && !reply.files.is_empty() {
        bail!("Use either adopt or files, not both");
    }
    Ok(reply)
}
pub(super) fn apply(reply: &Reply, current: &Tree, pool: &[Candidate]) -> Result<Tree> {
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
pub(super) fn accept(tree: &Tree) -> Result<Tree> {
    let tree = harness::autotrigger(tree);
    let errors = harness::validate(&tree);
    if !errors.is_empty() {
        bail!("{}", errors.join("\n"));
    }
    Ok(tree)
}

// ---- persistence for resume ----------------------------------------------------

pub(super) fn read_json<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Option<T> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}
pub(crate) fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    std::fs::create_dir_all(path.parent().context("No parent directory")?)?;
    let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
    std::fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
}

pub(super) struct Ledger {
    pub(super) rollouts: Mutex<usize>,
    pub(super) stages: Mutex<usize>,
    pub(super) fallbacks: Mutex<Vec<String>>,
}

/// Downstream model calls must not outlive the rollout/context that produced
/// their inputs. Legacy unkeyed records intentionally fail this deserialization
/// and are recomputed once.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Cached<T> {
    pub(super) cache_key: String,
    pub(super) value: T,
}

pub(super) fn downstream_cache_key(plan: &Plan<'_>, kind: &str, input: Value) -> String {
    harness::fingerprint(&Tree::from([
        ("cache_namespace".to_string(), plan.cache_namespace.clone()),
        ("cache_kind".to_string(), kind.to_string()),
        ("input".to_string(), input.to_string()),
    ]))
}

pub(super) fn rollout_role(label: &str) -> &'static str {
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
pub(super) fn rollout_attempt(label: &str) -> &str {
    let start = label
        .rfind(|c: char| !c.is_ascii_digit())
        .map_or(0, |index| index + 1);
    if start == label.len() {
        "0"
    } else {
        &label[start..]
    }
}
pub(super) fn rollout_cache_key(plan: &Plan<'_>, label: &str, tree_fingerprint: &str) -> String {
    harness::fingerprint(&Tree::from([
        ("cache_namespace".to_string(), plan.cache_namespace.clone()),
        ("tree_fingerprint".to_string(), tree_fingerprint.to_string()),
        ("rollout_role".to_string(), rollout_role(label).to_string()),
        ("attempt".to_string(), rollout_attempt(label).to_string()),
    ]))
}

pub(super) async fn roll(
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
pub(super) async fn roll_many(
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
pub(super) async fn stage(
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
pub(super) async fn score(plan: &Plan<'_>, rubric: &str, runs: &[&RunView]) -> BTreeMap<String, Option<f64>> {
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
pub(super) fn warnings(
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
pub(super) async fn select(plan: &Plan<'_>, finals: &[(Stage, RunView)]) -> (String, String, String) {
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

