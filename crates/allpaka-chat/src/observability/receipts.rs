//! Whitelisting of span metadata: usage, evaluator references, sampling and
//! score receipts, and guardrail receipts are rebuilt field by field.

use super::*;

pub(crate) fn public_usage(usage: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for name in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cost",
    ] {
        if let Some(value) = usage
            .get(name)
            .filter(|v| v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0))
        {
            result.insert(name.into(), value.clone());
        }
    }
    if let Some(first) = usage["first_text_ms"].as_u64() {
        result.insert("first_text_ms".into(), json!(first));
    }
    if let Some(cached) = usage["prompt_tokens_details"]["cached_tokens"].as_u64() {
        result.insert("cached_tokens".into(), json!(cached));
    }
    if let Some(currency) = usage["cost_currency"]
        .as_str()
        .filter(|code| code.len() == 3 && code.bytes().all(|b| b.is_ascii_uppercase()))
    {
        result.insert("cost_currency".into(), json!(currency));
    }
    if let Some(reference) = valid_evaluator_ref(&usage["evaluator_ref"]) {
        result.insert("evaluator_ref".into(), reference);
    }
    if let Some(sampling) = valid_evaluation_sampling(&usage["evaluation_sampling"]) {
        result.insert("evaluation_sampling".into(), sampling);
    }
    if let Some(scores) = valid_evaluation_scores(&usage["evaluation_scores"]) {
        result.insert("evaluation_scores".into(), scores);
    }
    if let Some(receipt) = public_guardrail_receipt(&usage["guardrail_receipt"]) {
        result.insert("guardrail_receipt".into(), receipt);
    }
    json!(result)
}
pub(super) fn valid_evaluator_ref(value: &Value) -> Option<Value> {
    if value.as_object()?.len() != 2
        || !value["version"].as_u64().is_some_and(|version| version > 0)
        || !value["id"].as_str().is_some_and(|id| {
            !id.is_empty()
                && id.len() <= 100
                && id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        })
    {
        return None;
    }
    Some(value.clone())
}
pub(super) fn valid_evaluation_sampling(value: &Value) -> Option<Value> {
    let fields = value.as_object()?;
    if fields.len() != 3
        || value["method"] != "sha256_v1"
        || !value["sample_rate"]
            .as_f64()
            .is_some_and(|rate| rate.is_finite() && (0.0..=1.0).contains(&rate))
        || !value["selected"].is_boolean()
    {
        return None;
    }
    Some(value.clone())
}
pub(super) fn valid_evaluation_scores(value: &Value) -> Option<Value> {
    let scores = value.as_object()?;
    if scores.is_empty()
        || scores.len() > 20
        || scores.iter().any(|(metric, value)| {
            metric.is_empty()
                || metric.len() > 100
                || !metric
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                || !value
                    .as_f64()
                    .is_some_and(|score| score.is_finite() && (0.0..=1.0).contains(&score))
        })
    {
        return None;
    }
    Some(value.clone())
}
pub(super) fn public_guardrail_receipt(value: &Value) -> Option<Value> {
    let stage = value["stage"].as_str()?;
    let action = value["action"].as_str()?;
    let hash = value["policy_sha256"].as_str()?;
    if value["kind"] != "local_guardrail"
        || value["schema_version"] != 1
        || !["input", "output"].contains(&stage)
        || !["observe", "block"].contains(&action)
        || !valid_policy_hash(hash)
        || value["provider_calls"] != 0
        || value["content_captured"] != false
    {
        return None;
    }
    let passed = value["passed"].as_bool()?;
    let blocked = value["blocked"].as_bool()?;
    if blocked != (!passed && action == "block") {
        return None;
    }
    let raw = value["rules"].as_array()?;
    if !(1..=32).contains(&raw.len()) {
        return None;
    }
    let mut ids = std::collections::HashSet::new();
    let mut rules = Vec::new();
    for rule in raw {
        let id = rule["rule_id"].as_str()?;
        let kind = rule["kind"].as_str()?;
        let outcome = rule["passed"].as_bool()?;
        if id.is_empty()
            || id.len() > 80
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || !ids.insert(id)
            || ![
                "min_bytes",
                "max_bytes",
                "json_valid",
                "forbidden_substrings",
                "required_substrings",
            ]
            .contains(&kind)
        {
            return None;
        }
        rules.push(json!({"rule_id":id,"kind":kind,"passed":outcome}));
    }
    if rules.iter().all(|rule| rule["passed"] == true) != passed {
        return None;
    }
    Some(
        json!({"kind":"local_guardrail","schema_version":1,"stage":stage,"action":action,"passed":passed,"blocked":blocked,"rules":rules,"policy_sha256":hash,"provider_calls":0,"content_captured":false}),
    )
}
pub(super) fn valid_policy_hash(hash: &str) -> bool {
    crate::storage::valid_sha256_hex(hash)
}
