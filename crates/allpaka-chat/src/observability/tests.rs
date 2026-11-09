use super::*;
#[test]
fn completed_trace_health_counts_failed_spans_and_rejects_running_or_removed() {
    let data = std::env::temp_dir().join(format!("allpaka-health-{}", crate::evaluation::new_id()));
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("health", "project").unwrap();
    let mut root = trace.span("turn", "agent", None);
    let mut failed = trace.span("tool", "failed-tool", Some(root.id()));
    let mut successful = trace.span("model", "model", Some(root.id()));
    assert!(store
        .completed_trace_health(&trace.id(), "project")
        .is_err());
    root.finish("completed", &Value::Null);
    failed.finish("failed", &Value::Null);
    assert!(store
        .completed_trace_health(&trace.id(), "project")
        .is_err());
    successful.finish("completed", &Value::Null);
    let (hash, score) = store
        .completed_trace_health(&trace.id(), "project")
        .unwrap();
    assert_eq!(score, 2.0 / 3.0);
    assert_eq!(
        hash,
        store
            .completed_trace_fingerprint(&trace.id(), "project")
            .unwrap()
    );
    assert_eq!(
        Store::new(&data)
            .unwrap()
            .completed_trace_health(&trace.id(), "project")
            .unwrap(),
        (hash, score)
    );
    assert!(store
        .completed_trace_health(&trace.id(), "foreign")
        .is_err());
    store.change_visibility(&trace.id(), true).unwrap();
    assert!(store
        .completed_trace_health(&trace.id(), "project")
        .is_err());
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn completed_trace_fingerprint_is_stable_and_detects_changed_metadata() {
    let data =
        std::env::temp_dir().join(format!("allpaka-trace-pin-{}", crate::evaluation::new_id()));
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("source", "project").unwrap();
    let mut root = trace.span("turn", "agent", None);
    assert!(store
        .completed_trace_fingerprint(&trace.id(), "project")
        .is_err());
    root.finish("completed", &Value::Null);
    let hash = store
        .completed_trace_fingerprint(&trace.id(), "project")
        .unwrap();
    assert_eq!(
        Store::new(&data)
            .unwrap()
            .completed_trace_fingerprint(&trace.id(), "project")
            .unwrap(),
        hash
    );
    assert!(store
        .completed_trace_fingerprint(&trace.id(), "other")
        .is_err());
    trace
        .span("tool", "extra", Some(root.id()))
        .finish("completed", &Value::Null);
    assert_ne!(
        store
            .completed_trace_fingerprint(&trace.id(), "project")
            .unwrap(),
        hash
    );
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn callback_summary_separates_evaluator_versions_and_project_scope() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    for (project, version, score, status) in [
        ("p", 1, 0.2, "completed"),
        ("p", 1, 0.8, "completed"),
        ("p", 2, 1.0, "completed"),
        ("other", 1, 0.0, "completed"),
        ("p", 3, 0.0, "failed"),
    ] {
        let usage = if status == "failed" {
            json!({"evaluator_ref":{"id":"check","version":version}})
        } else {
            json!({"evaluation_scores":{"quality":score},"evaluator_ref":{"id":"check","version":version}})
        };
        store.ingest(serde_json::from_value(json!({"project_id":project,"correlation_id":"summary","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":status,"started_ms":1000,"duration_ms":1,"usage":usage}]})).unwrap()).unwrap();
    }
    let query = EvaluationSearch {
        project_id: "p".into(),
        since_ms: None,
        until_ms: None,
    };
    let report = store.evaluation_summary(&query).unwrap();
    assert_eq!(report["trace_count"], 4);
    assert_eq!(report["failed_assessments"], 1);
    assert_eq!(report["completed_assessments"], 3);
    assert_eq!(report["metrics"].as_array().unwrap().len(), 2);
    assert_eq!(report["metrics"][0]["count"], 2);
    assert_eq!(report["metrics"][0]["mean"], 0.5);
    assert_eq!(report["metrics"][1]["evaluator_version"], 2);
    assert_eq!(
        report["evaluators"],
        json!([
            {"evaluator_id":"check","evaluator_version":1,"completed_assessments":2,"failed_assessments":0},
            {"evaluator_id":"check","evaluator_version":2,"completed_assessments":1,"failed_assessments":0},
            {"evaluator_id":"check","evaluator_version":3,"completed_assessments":0,"failed_assessments":1}
        ])
    );
    store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"legacy-failure","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":"failed","started_ms":1000,"duration_ms":1}]})).unwrap()).unwrap();
    let legacy = store.evaluation_summary(&query).unwrap();
    assert_eq!(
        legacy["evaluators"][0],
        json!({"evaluator_id":null,"evaluator_version":null,"completed_assessments":0,"failed_assessments":1})
    );
    assert_eq!(legacy["failed_assessments"], 2);
    for _ in 0..20 {
        store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"rounding","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"evaluation","status":"completed","started_ms":1000,"duration_ms":1,"usage":{"evaluation_scores":{"constant":0.1},"evaluator_ref":{"id":"constant-check","version":1}}}]})).unwrap()).unwrap();
    }
    let constant = store.evaluation_summary(&query).unwrap();
    let group = constant["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["metric"] == "constant")
        .unwrap();
    assert_eq!(group["mean"], group["max"]);
    assert_eq!(group["count"], 20);
    assert_eq!(
        store
            .evaluation_summary(&EvaluationSearch {
                since_ms: Some(1001),
                ..query
            })
            .unwrap()["trace_count"],
        0
    );
    assert!(store
        .evaluation_summary(&EvaluationSearch {
            project_id: "p".into(),
            since_ms: Some(2),
            until_ms: Some(1)
        })
        .is_err());
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn sampling_receipts_reverify_sdk_hash_decisions() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let request = |rate: f64, selected: bool| json!({"project_id":"p","correlation_id":"sampling","started_ms":1000,"spans":[{"parent_id":null,"kind":"tool","name":"task","status":"completed","started_ms":1000,"duration_ms":1,"usage":{"evaluation_sampling":{"method":"sha256_v1","sample_rate":rate,"selected":selected}}}]});
    for (rate, selected) in [(0.0, false), (1.0, true), (0.5, true)] {
        assert!(store
            .ingest(serde_json::from_value(request(rate, selected)).unwrap())
            .is_ok());
        assert!(store
            .ingest(serde_json::from_value(request(rate, !selected)).unwrap())
            .is_err());
    }
    assert!(valid_evaluation_sampling(
        &json!({"method":"sha256_v1","sample_rate":true,"selected":false})
    )
    .is_none());
    assert!(valid_evaluation_sampling(
        &json!({"method":"sha256_v1","sample_rate":0.5,"selected":true,"private":"text"})
    )
    .is_none());
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn evaluation_scores_are_bounded_numeric_metadata() {
    assert!(valid_evaluator_ref(&json!({"id":"check","version":2})).is_some());
    for invalid in [
        json!({"id":"check","version":true}),
        json!({"id":"private text","version":1}),
        json!({"id":"check","version":1,"private":"text"}),
    ] {
        assert!(valid_evaluator_ref(&invalid).is_none());
    }
    assert_eq!(
        public_usage(&json!({"evaluation_scores":{"quality":0.5}}))["evaluation_scores"]["quality"],
        0.5
    );
    for invalid in [
        json!({}),
        json!({"metric":true}),
        json!({"metric":1.1}),
        json!({"private text":0.5}),
        json!({"metric":"PRIVATE"}),
    ] {
        assert!(valid_evaluation_scores(&invalid).is_none());
        assert!(public_usage(&json!({"evaluation_scores":invalid}))["evaluation_scores"].is_null());
    }
}
#[test]
fn guardrail_receipts_strip_private_values_and_reject_inconsistent_results() {
    let mut receipt = json!({"kind":"local_guardrail","schema_version":1,"stage":"input","action":"block","passed":false,"blocked":true,"rules":[{"rule_id":"limit","kind":"max_bytes","passed":false,"value":"PRIVATE"}],"policy_sha256":"a".repeat(64),"provider_calls":0,"content_captured":false,"text":"PRIVATE"});
    let clean = public_usage(&json!({"guardrail_receipt":receipt}));
    assert!(clean["guardrail_receipt"].is_object());
    assert!(!clean.to_string().contains("PRIVATE"));
    receipt["passed"] = json!(true);
    assert!(public_usage(&json!({"guardrail_receipt":receipt}))["guardrail_receipt"].is_null());
}
#[test]
fn same_model_keeps_provider_unknown_and_external_groups_distinct() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("s", "p").unwrap();
    let mut root = trace.span("turn", "agent", None);
    for (provider, tokens) in [(Some("a"), 1), (Some("b"), 2), (None, 3)] {
        let mut model = trace.span("model", "same-model", Some(root.id()));
        if let Some(provider) = provider {
            model.set_provider(provider);
        }
        model.finish("completed", &json!({"input_tokens":tokens}));
    }
    root.finish("completed", &Value::Null);
    store.ingest(serde_json::from_value(json!({"project_id":"p","correlation_id":"ext","started_ms":1000,"spans":[
        {"parent_id":null,"kind":"model","name":"same-model","provider_id":"a","status":"completed","started_ms":1000,"duration_ms":0,"usage":{"input_tokens":4}}]})).unwrap()).unwrap();
    let receipt = store.summary(&SummarySearch::default()).unwrap();
    let groups = receipt["models"].as_array().unwrap();
    assert_eq!(groups.len(), 4);
    assert_eq!(receipt["input_tokens"], 10);
    assert!(groups.iter().any(|group| group["source"] == "native"
        && group["provider_id"].is_null()
        && group["input_tokens"] == 3));
    assert!(groups.iter().any(|group| group["source"] == "external"
        && group["provider_id"] == "a"
        && group["input_tokens"] == 4));
    assert_eq!(
        groups
            .iter()
            .map(|group| group["input_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        10
    );
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn model_quantiles_keep_unknown_durations_and_cost_currencies_separate() {
    let mut totals = ModelTotals::default();
    for (index, duration) in [Some(10), Some(20), Some(100), None]
        .into_iter()
        .enumerate()
    {
        let span:Span=serde_json::from_value(json!({"id":index,"parent_id":null,"kind":"model","name":"m",
            "status":if index==2{"failed"}else{"completed"},"started_ms":0,"duration_ms":duration,
            "usage":if index==0{json!({"prompt_tokens":3,"input_tokens":99,"cost":0.2,"cost_currency":"USD"})}
                else if index==1{json!({"input_tokens":2,"cost":0.3,"cost_currency":"EUR"})}else{json!({})}})).unwrap();
        totals.observe(&span).unwrap();
    }
    let result = totals.receipt("native".into(), None, "m".into());
    assert_eq!(result["calls"], 4);
    assert_eq!(result["input_tokens"], 5);
    assert_eq!(result["input_tokens_unknown_calls"], 2);
    assert_eq!(result["duration_known_calls"], 3);
    assert_eq!(result["duration_unknown_calls"], 1);
    assert_eq!(result["duration_p50_ms"], 20);
    assert_eq!(result["duration_p95_ms"], 100);
    assert_eq!(result["statuses"]["failed"], 1);
    assert_eq!(result["reported_cost_by_currency"]["USD"], 0.2);
    assert_eq!(result["reported_cost_by_currency"]["EUR"], 0.3);
    assert!(
        ModelTotals::default().receipt("external".into(), None, "m".into())["duration_p50_ms"]
            .is_null()
    );
}
#[test]
fn time_series_assigns_trace_start_and_preserves_empty_intervals_and_usage() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    for (started, status) in [(10, "completed"), (19, "failed"), (30, "completed")] {
        let trace = store.begin("session", "project").unwrap();
        trace.record.lock().unwrap().started_ms = started;
        let mut root = trace.span("agent", "root", None);
        let mut model = trace.span("model", "model", Some(0));
        model.finish("completed", &json!({"input_tokens":7,"first_text_ms":0}));
        root.finish(status, &json!({}));
    }
    let query = TimeSearch {
        status: None,
        project_id: Some("project".into()),
        session_id: None,
        since_ms: 10,
        until_ms: 30,
        bucket_ms: 10,
    };
    let result = store.time_series(&query).unwrap();
    assert_eq!(result["bucket_count"], 3);
    let rows = result["buckets"].as_array().unwrap();
    assert_eq!(rows[0]["trace_count"], 2);
    assert_eq!(rows[0]["trace_statuses"]["failed"], 1);
    assert_eq!(rows[0]["model_usage"]["input_tokens"], 14);
    assert_eq!(rows[0]["model_usage"]["first_text"]["known_calls"], 2);
    assert_eq!(rows[1]["trace_count"], 0);
    assert!(rows[1]["model_usage"]["first_text"]["p50_ms"].is_null());
    assert_eq!(rows[2]["trace_count"], 1);
    assert_eq!(rows[2]["end_exclusive_ms"], 31);
    let failed = store
        .time_series(&TimeSearch {
            status: Some("failed".into()),
            project_id: query.project_id.clone(),
            session_id: None,
            since_ms: 10,
            until_ms: 30,
            bucket_ms: 10,
        })
        .unwrap();
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["buckets"][0]["trace_count"], 1);
    assert_eq!(failed["buckets"][0]["model_usage"]["input_tokens"], 7);
    assert_eq!(failed["buckets"][2]["trace_count"], 0);
    assert!(store
        .time_series(&TimeSearch {
            status: Some("INVALID".into()),
            project_id: None,
            session_id: None,
            since_ms: 10,
            until_ms: 30,
            bucket_ms: 10
        })
        .is_err());
    assert!(store
        .time_series(&TimeSearch {
            bucket_ms: 0,
            ..query
        })
        .is_err());
    assert!(store
        .time_series(&TimeSearch {
            status: None,
            project_id: None,
            session_id: None,
            since_ms: 0,
            until_ms: 500,
            bucket_ms: 1
        })
        .is_err());
    assert_eq!(
        store
            .time_series(&TimeSearch {
                status: None,
                project_id: Some("other".into()),
                session_id: None,
                since_ms: 10,
                until_ms: 30,
                bucket_ms: 10
            })
            .unwrap()["buckets"][0]["trace_count"],
        0
    );
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn trace_status_and_time_filters_are_inclusive_and_precede_pagination() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    for (started, status) in [(10, "failed"), (20, "completed"), (30, "failed")] {
        let trace = store.begin("session", "project").unwrap();
        trace.record.lock().unwrap().started_ms = started;
        let mut root = trace.span("agent", "root", None);
        root.finish(status, &json!({}));
    }
    let result = store
        .list(&Search {
            status: Some("failed".into()),
            since_ms: Some(10),
            until_ms: Some(30),
            limit: Some(1),
            offset: Some(1),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result["total"], 2);
    assert_eq!(result["traces"][0]["started_ms"], 10);
    assert_eq!(
        store
            .list(&Search {
                since_ms: Some(20),
                until_ms: Some(20),
                ..Default::default()
            })
            .unwrap()["total"],
        1
    );
    assert_eq!(
        store
            .list(&Search {
                status: Some("failed".into()),
                project_id: Some("other".into()),
                ..Default::default()
            })
            .unwrap()["total"],
        0
    );
    assert!(store
        .list(&Search {
            since_ms: Some(30),
            until_ms: Some(10),
            ..Default::default()
        })
        .is_err());
    assert!(store
        .list(&Search {
            status: Some("PRIVATE STATUS".into()),
            ..Default::default()
        })
        .is_err());
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn first_text_quantiles_preserve_zero_unknowns_and_reject_inconsistent_values() {
    let mut totals = ModelTotals::default();
    for value in [
        json!(0),
        json!(10),
        json!(100),
        json!(201),
        json!(null),
        json!(-1),
    ] {
        let span:Span=serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_model","name":"model","status":"completed","started_ms":0,"duration_ms":200,"usage":{"first_text_ms":value}})).unwrap();
        totals.observe(&span).unwrap();
    }
    let receipt = totals.receipt("external".into(), None, "model".into());
    let first = &receipt["first_text"];
    assert_eq!(first["known_calls"], 3);
    assert_eq!(first["unknown_calls"], 3);
    assert_eq!(first["min_ms"], 0);
    assert_eq!(first["max_ms"], 100);
    assert_eq!(first["p50_ms"], 10);
    assert_eq!(first["p95_ms"], 100);
    let empty = first_text_receipt(vec![], 2);
    assert!(empty["p50_ms"].is_null());
    assert!(empty["min_ms"].is_null());
    assert_eq!(empty["unknown_calls"], 2);
}
#[test]
fn first_text_metadata_is_integer_bounded_and_optional() {
    assert_eq!(
        public_usage(&json!({"first_text_ms":0}))["first_text_ms"],
        0
    );
    assert_eq!(
        public_usage(&json!({"first_text_ms":17}))["first_text_ms"],
        17
    );
    for value in [json!(-1), json!(true), json!(1.5), json!("private")] {
        assert!(public_usage(&json!({"first_text_ms":value}))["first_text_ms"].is_null());
        assert!(serde_json::from_value::<ExternalUsage>(json!({"first_text_ms":value})).is_err());
    }
    assert!(public_usage(&json!({}))["first_text_ms"].is_null());
}
#[test]
fn guardrail_policy_and_outcome_filters_require_the_same_span() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("session", "project").unwrap();
    let a = "a".repeat(64);
    let b = "b".repeat(64);
    let mut observed = trace.span("tool", &format!("guardrail.input.observe.fail.{a}"), None);
    observed.finish("completed", &json!({}));
    let mut blocked = trace.span("tool", &format!("guardrail.output.block.fail.{b}"), None);
    blocked.finish("failed", &json!({}));
    assert_eq!(
        store
            .list(&Search {
                guardrail: Some(GuardrailFilter::Blocked),
                guardrail_policy_sha256: Some(a.clone()),
                ..Default::default()
            })
            .unwrap()["total"],
        0
    );
    assert_eq!(
        store
            .list(&Search {
                guardrail: Some(GuardrailFilter::Blocked),
                guardrail_policy_sha256: Some(b),
                ..Default::default()
            })
            .unwrap()["total"],
        1
    );
    assert_eq!(
        store
            .list(&Search {
                guardrail_policy_sha256: Some(a),
                ..Default::default()
            })
            .unwrap()["total"],
        1
    );
    for hash in ["bad".into(), "A".repeat(64), "a".repeat(65)] {
        assert!(store
            .list(&Search {
                guardrail_policy_sha256: Some(hash),
                ..Default::default()
            })
            .is_err());
    }
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn guardrail_policy_groups_are_bounded_complete_and_order_independent() {
    let make = |index: usize| -> Span {
        serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_tool","name":format!("guardrail.input.block.fail.{index:064x}"),"status":"failed","started_ms":0,"duration_ms":0,"usage":{}})).unwrap()
    };
    let mut ascending = GuardrailSummary::default();
    let mut descending = GuardrailSummary::default();
    for _ in 0..2 {
        for index in 0..105 {
            ascending.observe(&make(index));
        }
        for index in (0..105).rev() {
            descending.observe(&make(index));
        }
    }
    let result = ascending.receipt();
    assert_eq!(result, descending.receipt());
    assert_eq!(result["checks"], 210);
    assert_eq!(result["blocked"], 210);
    assert_eq!(result["policy_groups_truncated"], true);
    let groups = result["policies"].as_array().unwrap();
    assert_eq!(groups.len(), 100);
    assert!(groups
        .iter()
        .all(|row| row["checks"] == 2 && row["blocked"] == 2));
    assert_eq!(groups[0]["policy_sha256"], format!("{:064x}", 0));
    assert_eq!(groups[99]["policy_sha256"], format!("{:064x}", 99));
}
#[test]
fn guardrail_summary_separates_reported_outcomes_and_incomplete_checks() {
    let mut totals = GuardrailTotals::default();
    for (stage, action, outcome, status) in [
        ("input", "block", "pass", "completed"),
        ("output", "observe", "fail", "completed"),
        ("output", "block", "fail", "failed"),
        ("input", "block", "fail", "running"),
        ("input", "block", "pass", "failed"),
    ] {
        let span:Span=serde_json::from_value(json!({"id":0,"parent_id":null,"kind":"external_tool","name":format!("guardrail.{stage}.{action}.{outcome}.{}","a".repeat(64)),"status":status,"started_ms":0,"duration_ms":0,"usage":{}})).unwrap();
        totals.observe(&span);
    }
    let receipt = totals.receipt();
    assert_eq!(receipt["checks"], 5);
    assert_eq!(receipt["passed"], 1);
    assert_eq!(receipt["violations"], 2);
    assert_eq!(receipt["blocked"], 1);
    assert_eq!(receipt["incomplete_or_inconsistent"], 2);
    assert_eq!(receipt["input_checks"], 3);
    assert_eq!(receipt["output_checks"], 2);
}
#[test]
fn guardrail_filters_validate_fingerprints_and_filter_before_pagination() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    for (name, status) in [
        (
            format!("guardrail.input.block.fail.{}", "a".repeat(64)),
            "failed",
        ),
        (
            format!("guardrail.output.observe.fail.{}", "b".repeat(64)),
            "completed",
        ),
        ("guardrail.input.block.fail.invalid".into(), "failed"),
    ] {
        let trace = store.begin("session", "project").unwrap();
        let mut span = trace.span("tool", &name, None);
        span.finish(status, &json!({}));
    }
    assert_eq!(
        store.summary(&SummarySearch::default()).unwrap()["guardrails"]["checks"],
        2
    );
    assert_eq!(
        store
            .summary(&SummarySearch {
                project_id: Some("other".into()),
                ..Default::default()
            })
            .unwrap()["guardrails"]["checks"],
        0
    );
    for (filter, count) in [
        (GuardrailFilter::Any, 2),
        (GuardrailFilter::Failed, 2),
        (GuardrailFilter::Blocked, 1),
    ] {
        let page = store
            .list(&Search {
                guardrail: Some(filter),
                limit: Some(1),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page["total"], count);
        assert_eq!(page["traces"].as_array().unwrap().len(), 1);
    }
    assert_eq!(
        store
            .list(&Search {
                guardrail: Some(GuardrailFilter::Any),
                project_id: Some("other".into()),
                ..Default::default()
            })
            .unwrap()["total"],
        0
    );
    assert!(serde_json::from_value::<Search>(json!({"guardrail":"unknown"})).is_err());
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn removal_preserves_bytes_filters_totals_and_reopens_after_restart() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("session", "project").unwrap();
    let mut root = trace.span("model", "m", None);
    assert!(store.change_visibility(&trace.id(), true).is_err());
    root.finish("completed", &json!({"input_tokens":7}));
    let path = store.root.join(format!("{}.json", trace.id()));
    let original = std::fs::read(&path).unwrap();
    assert_eq!(
        store.change_visibility(&trace.id(), true).unwrap()["changed"],
        true
    );
    assert_eq!(
        store.change_visibility(&trace.id(), true).unwrap()["changed"],
        false
    );
    assert_eq!(
        store.summary(&SummarySearch::default()).unwrap()["trace_count"],
        0
    );
    assert_eq!(
        store
            .list(&Search {
                removed: Some(true),
                ..Default::default()
            })
            .unwrap()["traces"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let reopened = Store::new(&data).unwrap();
    assert!(reopened.validate_target(&trace.id(), None).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    reopened.change_visibility(&trace.id(), false).unwrap();
    reopened.validate_target(&trace.id(), Some(0)).unwrap();
    assert_eq!(
        reopened.summary(&SummarySearch::default()).unwrap()["input_tokens"],
        7
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn keyed_external_retries_are_atomic_project_scoped_and_conflict_checked() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let request = json!({"project_id":"p","correlation_id":"client","idempotency_key":"receipt-key","started_ms":100,
        "spans":[{"parent_id":null,"kind":"agent","name":"pipeline","status":"completed","started_ms":100,"duration_ms":20}]});
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let store = store.clone();
            let request = request.clone();
            std::thread::spawn(move || {
                store
                    .ingest(serde_json::from_value(request).unwrap())
                    .unwrap()
            })
        })
        .collect();
    let receipts: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert!(receipts
        .iter()
        .all(|receipt| receipt["id"] == receipts[0]["id"]));
    assert_eq!(
        receipts
            .iter()
            .filter(|receipt| receipt["deduplicated"] == false)
            .count(),
        1
    );
    let mut conflicting = request.clone();
    conflicting["spans"][0]["duration_ms"] = json!(21);
    assert!(store
        .ingest(serde_json::from_value(conflicting).unwrap())
        .unwrap_err()
        .to_string()
        .contains("idempotency conflict"));
    let mut other = request.clone();
    other["project_id"] = json!("other");
    assert_ne!(
        store
            .ingest(serde_json::from_value(other).unwrap())
            .unwrap()["id"],
        receipts[0]["id"]
    );
    assert_eq!(std::fs::read_dir(store.root.as_ref()).unwrap().count(), 2);
    let stored = std::fs::read_to_string(
        store
            .root
            .join(format!("{}.json", receipts[0]["id"].as_str().unwrap())),
    )
    .unwrap();
    assert!(!stored.contains("receipt-key"));
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn external_tree_bounds_and_immutable_metadata_receipts() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let input = json!({"project_id":"p","correlation_id":"client-1","started_ms":100,
        "spans":[{"parent_id":null,"kind":"agent","name":"pipeline","status":"completed","started_ms":100,"duration_ms":50},
        {"parent_id":0,"kind":"model","name":"model-v1","status":"completed","started_ms":110,"duration_ms":20,
         "usage":{"input_tokens":3,"output_tokens":4,"cost":0.1,"cost_currency":"USD"}}]});
    let receipt = store
        .ingest(serde_json::from_value(input.clone()).unwrap())
        .unwrap();
    let second = store
        .ingest(serde_json::from_value(input.clone()).unwrap())
        .unwrap();
    assert_ne!(receipt["id"], second["id"]);
    let record = store.read_record(receipt["id"].as_str().unwrap()).unwrap();
    assert_eq!(record.spans[1].kind, "external_model");
    let summary = store
        .summary(&SummarySearch {
            project_id: Some("p".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(summary["external_model_calls"], 2);
    for (field, value) in [
        ("parent_id", json!(1)),
        ("started_ms", json!(99)),
        ("duration_ms", json!(u64::MAX)),
        ("status", json!("running")),
        ("name", json!("private prompt text")),
    ] {
        let mut invalid = input.clone();
        invalid["spans"][1][field] = value;
        assert!(store
            .ingest(serde_json::from_value(invalid).unwrap())
            .is_err());
    }
    let mut private = input.clone();
    private["spans"][1]["usage"]["api_key"] = json!("secret");
    assert!(serde_json::from_value::<ExternalTrace>(private).is_err());
    let mut future = input.clone();
    future["spans"][1]["started_ms"] = json!(151);
    future["spans"][1]["duration_ms"] = Value::Null;
    assert!(store
        .ingest(serde_json::from_value(future).unwrap())
        .is_err());
    assert_eq!(std::fs::read_dir(store.root.as_ref()).unwrap().count(), 2);
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn conversation_summary_separates_projects_and_bounds_display_only() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    for index in 0..101 {
        let trace = store.begin(&format!("session-{index}"), "project").unwrap();
        trace
            .span("turn", "agent", None)
            .finish("completed", &Value::Null);
    }
    let other = store.begin("session-0", "other").unwrap();
    other
        .span("turn", "agent", None)
        .finish("failed", &Value::Null);
    let result = store.summary(&SummarySearch::default()).unwrap();
    assert_eq!(result["conversation_count"], 102);
    assert_eq!(result["conversations"].as_array().unwrap().len(), 100);
    assert_eq!(result["conversations_truncated"], true);
    assert_eq!(result["trace_count"], 102);
    let second = store
        .summary(&SummarySearch {
            conversation_offset: Some(100),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(second["conversations"].as_array().unwrap().len(), 2);
    assert_eq!(second["conversation_has_more"], false);
    assert_eq!(second["trace_count"], result["trace_count"]);
    let ids: std::collections::HashSet<_> = result["conversations"]
        .as_array()
        .unwrap()
        .iter()
        .chain(second["conversations"].as_array().unwrap())
        .map(|row| {
            (
                row["project_id"].as_str().unwrap(),
                row["session_id"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(ids.len(), 102);
    for (offset, limit) in [(10_001, 1), (0, 0), (0, 101)] {
        assert!(store
            .summary(&SummarySearch {
                conversation_offset: Some(offset),
                conversation_limit: Some(limit),
                ..Default::default()
            })
            .is_err());
    }
    let failed = store
        .summary(&SummarySearch {
            status: Some("failed".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(failed["trace_count"], 1);
    assert_eq!(failed["conversation_count"], 1);
    assert_eq!(failed["conversations"][0]["project_id"], "other");
    assert_eq!(failed["status"], "failed");
    assert!(store
        .summary(&SummarySearch {
            status: Some("INVALID".into()),
            ..Default::default()
        })
        .is_err());
    let scoped = store
        .summary(&SummarySearch {
            session_id: Some("session-0".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(scoped["conversation_count"], 2);
    assert_eq!(scoped["trace_count"], 2);
    let project = store
        .summary(&SummarySearch {
            project_id: Some("other".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(project["conversations"][0]["trace_statuses"]["failed"], 1);
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn summary_counts_model_leaves_once_and_keeps_unknown_currency_separate() {
    let data =
        std::env::temp_dir().join(format!("allpaka-summary-{}", crate::evaluation::new_id()));
    std::fs::create_dir_all(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("session", "project").unwrap();
    let mut root = trace.span("turn", "agent", None);
    let mut a = trace.span("model", "a", Some(root.id()));
    a.finish("completed",&json!({"prompt_tokens":3,"input_tokens":100,"completion_tokens":4,"cost":0.25,"cost_currency":"USD"}));
    let mut b = trace.span("model", "b", Some(root.id()));
    b.finish(
        "completed",
        &json!({"input_tokens":5,"output_tokens":6,"cost":0.5}),
    );
    let mut c = trace.span("model", "c", Some(root.id()));
    c.finish("failed", &Value::Null);
    root.finish(
        "completed",
        &json!({"prompt_tokens":999,"cost":999,"cost_currency":"USD"}),
    );
    let result = store
        .summary(&SummarySearch {
            project_id: Some("project".into()),
            session_id: None,
            since_ms: None,
            until_ms: None,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result["trace_count"], 1);
    assert_eq!(result["model_calls"], 3);
    assert_eq!(result["input_tokens"], 8);
    assert_eq!(result["output_tokens"], 10);
    assert_eq!(result["input_tokens_unknown_calls"], 1);
    assert_eq!(result["cost_unknown_calls"], 1);
    assert_eq!(result["cost_currency_unknown_calls"], 1);
    assert_eq!(result["reported_cost_by_currency"]["USD"], 0.25);
    assert_eq!(result["conversation_count"], 1);
    let conversation = &result["conversations"][0];
    assert_eq!(conversation["project_id"], "project");
    assert_eq!(conversation["session_id"], "session");
    assert_eq!(conversation["trace_count"], 1);
    assert_eq!(conversation["model_usage"]["calls"], 3);
    assert_eq!(conversation["model_usage"]["input_tokens"], 8);
    assert_eq!(conversation["model_usage"]["input_tokens_unknown_calls"], 1);
    assert_eq!(
        conversation["model_usage"]["reported_cost_by_currency"]["USD"],
        0.25
    );
    assert_eq!(
        conversation["model_usage"]["cost_currency_unknown_calls"],
        1
    );
    let empty = store
        .summary(&SummarySearch {
            project_id: Some("absent".into()),
            session_id: None,
            since_ms: None,
            until_ms: None,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(empty["model_calls"], 0);
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn restart_recovery_keeps_completed_leaves_and_unknown_durations() {
    let data = std::env::temp_dir().join(crate::evaluation::new_id());
    std::fs::create_dir(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let record = Record {
        id: "trace-orphan".into(),
        session_id: "session".into(),
        project_id: "project".into(),
        started_ms: 1,
        status: "running".into(),
        recovered_ms: None,
        spans: vec![
            Span {
                id: 0,
                parent_id: None,
                kind: "turn".into(),
                name: "agent".into(),
                status: "running".into(),
                started_ms: 1,
                duration_ms: None,
                usage: Value::Null,
                linked_trace_id: None,
                provider_id: None,
            },
            Span {
                id: 1,
                parent_id: Some(0),
                kind: "model".into(),
                name: "model".into(),
                status: "completed".into(),
                started_ms: 2,
                duration_ms: Some(3),
                usage: json!({"completion_tokens":4}),
                linked_trace_id: None,
                provider_id: None,
            },
        ],
    };
    atomic_write(&store.root.join("trace-orphan.json"), &record).unwrap();
    assert_eq!(store.recover().unwrap(), 1);
    assert_eq!(store.recover().unwrap(), 0);
    let rows = store.list(&Search::default()).unwrap();
    let recovered = &rows["traces"][0];
    assert_eq!(recovered["status"], "interrupted");
    assert!(recovered["recovered_ms"].as_u64().is_some());
    assert_eq!(recovered["spans"][0]["status"], "interrupted");
    assert!(recovered["spans"][0]["duration_ms"].is_null());
    assert_eq!(recovered["spans"][1]["status"], "completed");
    assert_eq!(recovered["spans"][1]["usage"]["completion_tokens"], 4);
    std::fs::remove_dir_all(data).unwrap();
}
#[test]
fn trace_tree_persists_status_and_whitelisted_usage_and_drop_interrupts() {
    let data = std::env::temp_dir().join(format!("allpaka-trace-{}-{}", std::process::id(), now()));
    std::fs::create_dir(&data).unwrap();
    let data = data.canonicalize().unwrap();
    let store = Store::new(&data).unwrap();
    let trace = store.begin("session", "project").unwrap();
    {
        let mut turn = trace.span("turn", "agent", None);
        let mut model = trace.span("model", "mock", Some(turn.id()));
        model.finish("completed",&json!({"prompt_tokens":3,"completion_tokens":4,"cost":0.1,"api_key":"secret","system":"private"}));
        turn.finish("completed", &Value::Null);
    }
    {
        let trace = store.begin("interrupted-session", "project").unwrap();
        let _turn = trace.span("turn", "agent", None);
    }
    let rows = store.list(&Search::default()).unwrap();
    assert_eq!(rows["total"], 2);
    let text = rows.to_string();
    assert!(!text.contains("secret"));
    assert!(!text.contains("private"));
    let rows = store
        .list(&Search {
            session_id: Some("session".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows["traces"][0]["status"], "completed");
    assert_eq!(rows["traces"][0]["spans"][1]["parent_id"], 0);
    assert_eq!(rows["traces"][0]["spans"][1]["usage"]["cost"], 0.1);
    let rows = store
        .list(&Search {
            session_id: Some("interrupted-session".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows["traces"][0]["status"], "interrupted");
    assert_eq!(
        Store::new(&data).unwrap().list(&Search::default()).unwrap()["total"],
        2
    );
    std::fs::remove_dir_all(data).unwrap();
}
