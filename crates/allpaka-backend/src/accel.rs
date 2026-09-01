//! Backend-neutral acceleration outcomes.
//!
//! A declined acceleration is expected control flow, but it must carry a
//! reason. This keeps callers from confusing an unsupported fast path with a
//! successful CPU/GPU execution or an internal error.

use allpaka_gguf::GgmlType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclineReason {
    NoDevice,
    UnsupportedTensorType {
        operation: &'static str,
        tensor: &'static str,
        ty: GgmlType,
    },
    UnsupportedShape {
        operation: &'static str,
        detail: String,
    },
    ForeignMemory {
        operation: &'static str,
    },
    MissingCapability {
        operation: &'static str,
        capability: &'static str,
    },
    InvalidState {
        operation: &'static str,
        detail: String,
    },
    Backend {
        operation: &'static str,
        detail: String,
    },
}

impl std::fmt::Display for DeclineReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no accelerator device"),
            Self::UnsupportedTensorType { operation, tensor, ty } => {
                write!(f, "{operation}: tensor {tensor} has unsupported type {ty:?}")
            }
            Self::UnsupportedShape { operation, detail }
            | Self::InvalidState { operation, detail }
            | Self::Backend { operation, detail } => write!(f, "{operation}: {detail}"),
            Self::ForeignMemory { operation } => {
                write!(f, "{operation}: weights are outside attached mappings")
            }
            Self::MissingCapability { operation, capability } => {
                write!(f, "{operation}: missing capability {capability}")
            }
        }
    }
}

#[derive(Debug)]
pub enum AccelOutcome<T> {
    Executed(T),
    Declined(DeclineReason),
}

impl<T> AccelOutcome<T> {
    pub fn executed(self) -> Option<T> {
        match self {
            Self::Executed(value) => Some(value),
            Self::Declined(_) => None,
        }
    }

    pub fn decline_reason(&self) -> Option<&DeclineReason> {
        match self {
            Self::Executed(_) => None,
            Self::Declined(reason) => Some(reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AccelOutcome, DeclineReason};

    #[test]
    fn decline_is_not_an_executed_empty_result() {
        let out: AccelOutcome<()> = AccelOutcome::Declined(DeclineReason::NoDevice);
        assert!(out.decline_reason().is_some());
        assert!(out.executed().is_none());
    }
}
