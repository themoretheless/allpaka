//! Explicit policy for resolving acceleration declines.

use crate::accel::{AccelOutcome, DeclineReason};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    Embed,
    Norm,
    Matmul,
    Attention,
    Router,
    Experts,
    KvUpdate,
    TokenDecode,
    Prefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackAction {
    Fail,
    CpuOperation,
    CpuLayer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPolicy {
    default: FallbackAction,
    overrides: BTreeMap<Operation, FallbackAction>,
}

impl ExecutionPolicy {
    pub fn fail_closed() -> Self {
        Self {
            default: FallbackAction::Fail,
            overrides: BTreeMap::new(),
        }
    }

    pub fn interactive() -> Self {
        Self {
            default: FallbackAction::CpuOperation,
            overrides: BTreeMap::new(),
        }
    }

    pub fn with_override(mut self, operation: Operation, action: FallbackAction) -> Self {
        self.overrides.insert(operation, action);
        self
    }

    pub fn action(&self, operation: Operation) -> FallbackAction {
        self.overrides
            .get(&operation)
            .copied()
            .unwrap_or(self.default)
    }

    pub fn resolve<T>(
        &self,
        operation: Operation,
        outcome: AccelOutcome<T>,
    ) -> Result<ExecutionDecision<T>, ExecutionDeclined> {
        match outcome {
            AccelOutcome::Executed(value) => Ok(ExecutionDecision::Accelerated(value)),
            AccelOutcome::Declined(reason) => match self.action(operation) {
                FallbackAction::Fail => Err(ExecutionDeclined { operation, reason }),
                action => Ok(ExecutionDecision::Fallback { action, reason }),
            },
        }
    }
}

#[derive(Debug)]
pub enum ExecutionDecision<T> {
    Accelerated(T),
    Fallback {
        action: FallbackAction,
        reason: DeclineReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionDeclined {
    pub operation: Operation,
    pub reason: DeclineReason,
}

impl std::fmt::Display for ExecutionDeclined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} acceleration declined: {}", self.operation, self.reason)
    }
}

impl std::error::Error for ExecutionDeclined {}

#[cfg(test)]
mod tests {
    use super::{ExecutionDecision, ExecutionPolicy, FallbackAction, Operation};
    use crate::accel::{AccelOutcome, DeclineReason};

    #[test]
    fn benchmark_policy_fails_instead_of_hiding_a_cpu_path() {
        let outcome: AccelOutcome<()> = AccelOutcome::Declined(DeclineReason::NoDevice);
        let error = ExecutionPolicy::fail_closed()
            .resolve(Operation::TokenDecode, outcome)
            .unwrap_err();
        assert_eq!(error.reason, DeclineReason::NoDevice);
    }

    #[test]
    fn fallback_scope_is_selected_per_operation() {
        let policy = ExecutionPolicy::interactive()
            .with_override(Operation::Attention, FallbackAction::CpuLayer);
        let outcome: AccelOutcome<()> = AccelOutcome::Declined(DeclineReason::NoDevice);
        match policy.resolve(Operation::Attention, outcome).unwrap() {
            ExecutionDecision::Fallback { action, reason } => {
                assert_eq!(action, FallbackAction::CpuLayer);
                assert_eq!(reason, DeclineReason::NoDevice);
            }
            ExecutionDecision::Accelerated(_) => panic!("unexpected acceleration"),
        }
    }
}
