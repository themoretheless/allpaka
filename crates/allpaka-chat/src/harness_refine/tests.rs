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
                let event = tokio::task::spawn_blocking(move || model(&request))
                    .await
                    .unwrap();
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
