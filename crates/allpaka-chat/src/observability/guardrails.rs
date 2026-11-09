//! Guardrail filters and aggregates over spans: per-policy totals and the
//! summary split by reported outcome.

use super::*;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GuardrailFilter {
    Any,
    Failed,
    Blocked,
}
impl GuardrailFilter {
    pub(super) fn matches(&self, span: &Span) -> bool {
        if !matches!(span.kind.as_str(), "tool" | "external_tool") {
            return false;
        }
        let parts: Vec<_> = span.name.split('.').collect();
        if parts.len() != 5
            || parts[0] != "guardrail"
            || !matches!(parts[1], "input" | "output")
            || !matches!(parts[2], "observe" | "block")
            || !matches!(parts[3], "pass" | "fail")
            || !valid_policy_hash(parts[4])
        {
            return false;
        }
        match self {
            Self::Any => true,
            Self::Failed => parts[3] == "fail",
            Self::Blocked => parts[2] == "block" && parts[3] == "fail" && span.status == "failed",
        }
    }
}
#[derive(Default)]
pub(super) struct GuardrailTotals {
    pub(super) checks: u64,
    pub(super) passed: u64,
    pub(super) violations: u64,
    pub(super) blocked: u64,
    pub(super) incomplete: u64,
    pub(super) input: u64,
    pub(super) output: u64,
}
impl GuardrailTotals {
    pub(super) fn observe(&mut self, span: &Span) {
        if !GuardrailFilter::Any.matches(span) {
            return;
        }
        let parts: Vec<_> = span.name.split('.').collect();
        self.checks += 1;
        if parts[1] == "input" {
            self.input += 1;
        } else {
            self.output += 1;
        }
        if parts[3] == "pass" && span.status == "completed" {
            self.passed += 1;
        } else if parts[3] == "fail"
            && ((parts[2] == "observe" && span.status == "completed")
                || (parts[2] == "block" && span.status == "failed"))
        {
            self.violations += 1;
            if parts[2] == "block" {
                self.blocked += 1;
            }
        } else {
            self.incomplete += 1;
        }
    }
    pub(super) fn receipt(&self) -> Value {
        json!({"checks":self.checks,"passed":self.passed,"violations":self.violations,"blocked":self.blocked,"incomplete_or_inconsistent":self.incomplete,"input_checks":self.input,"output_checks":self.output,"scope":"reported_guardrail_spans","content_capture":false})
    }
}
#[derive(Default)]
pub(super) struct GuardrailSummary {
    pub(super) totals: GuardrailTotals,
    pub(super) policies: std::collections::BTreeMap<String, GuardrailTotals>,
    pub(super) truncated: bool,
}
impl GuardrailSummary {
    pub(super) fn observe(&mut self, span: &Span) {
        if !GuardrailFilter::Any.matches(span) {
            return;
        }
        self.totals.observe(span);
        let hash = span.name.rsplit('.').next().unwrap();
        if !self.policies.contains_key(hash) && self.policies.len() == 100 {
            self.truncated = true;
            let largest = self.policies.last_key_value().unwrap().0.clone();
            if hash >= largest.as_str() {
                return;
            }
            self.policies.remove(&largest);
        }
        self.policies.entry(hash.into()).or_default().observe(span);
    }
    pub(super) fn receipt(&self) -> Value {
        let mut receipt = self.totals.receipt();
        let policies: Vec<_> = self
            .policies
            .iter()
            .map(|(hash, totals)| {
                let mut row = totals.receipt();
                row["policy_sha256"] = json!(hash);
                row
            })
            .collect();
        receipt["policies"] = json!(policies);
        receipt["policy_groups_limit"] = json!(100);
        receipt["policy_groups_truncated"] = json!(self.truncated);
        receipt["policy_group_order"] = json!("policy_sha256_ascending");
        receipt
    }
}
