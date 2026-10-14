//! Explicit policy for resolving acceleration declines.

use crate::accel::{AccelOutcome, DeclineReason};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Operation {
    Embed = 0,
    Norm = 1,
    Matmul = 2,
    Attention = 3,
    Router = 4,
    Experts = 5,
    KvUpdate = 6,
    TokenDecode = 7,
    Prefill = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackAction {
    Fail,
    CpuOperation,
    CpuLayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPolicy {
    default: FallbackAction,
    overrides: [Option<FallbackAction>; 9],
}

impl ExecutionPolicy {
    #[inline]
    pub const fn fail_closed() -> Self {
        Self {
            default: FallbackAction::Fail,
            overrides: [None; 9],
        }
    }

    #[inline]
    pub const fn interactive() -> Self {
        Self {
            default: FallbackAction::CpuOperation,
            overrides: [None; 9],
        }
    }

    #[inline]
    pub const fn with_override(mut self, operation: Operation, action: FallbackAction) -> Self {
        self.overrides[operation as usize] = Some(action);
        self
    }

    #[inline]
    pub const fn action(&self, operation: Operation) -> FallbackAction {
        match self.overrides[operation as usize] {
            Some(action) => action,
            None => self.default,
        }
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
        write!(
            f,
            "{:?} acceleration declined: {}",
            self.operation, self.reason
        )
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
