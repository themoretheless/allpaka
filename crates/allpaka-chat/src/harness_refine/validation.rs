use super::*;

pub(super) fn validation_report_key(
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

pub(super) fn validation_candidate(run: &RunView) -> String {
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
pub(super) async fn validation_judge(plan: &Plan<'_>, rubric: &str, run: &RunView) -> Result<f64> {
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
pub(super) async fn validate_champion(
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
pub(super) async fn reflect(
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
