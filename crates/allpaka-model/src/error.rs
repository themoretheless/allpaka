//! Typed error definitions for [`allpaka_model`](crate).

use std::fmt;

#[derive(Debug)]
pub enum ModelError {
    MissingMetadata(String),
    MissingTensor(String),
    ShapeMismatch {
        tensor: String,
        actual: Vec<u64>,
        expected: String,
    },
    ElementCountMismatch {
        tensor: String,
        actual: u64,
        expected: usize,
    },
    InvalidConfig(String),
    InvalidStageRange {
        first: usize,
        end: usize,
    },
    MissingStageEndpoint(&'static str),
    UnsupportedArchitecture {
        architecture: String,
        reason: &'static str,
    },
    OutOfVocab {
        token: u32,
        vocab: u32,
    },
    KvCapacityExhausted {
        needed: usize,
        capacity: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMetadata(key) => write!(f, "GGUF has no {key}"),
            Self::MissingTensor(name) => write!(f, "GGUF has no tensor {name:?}"),
            Self::ShapeMismatch {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "tensor {tensor:?} has shape {actual:?}, expected {expected}"
            ),
            Self::ElementCountMismatch {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "tensor {tensor:?} has {actual} elements, expected {expected}"
            ),
            Self::InvalidConfig(msg) => f.write_str(msg),
            Self::InvalidStageRange { first, end } => {
                write!(f, "invalid stage layer range {first}..{end}")
            }
            Self::MissingStageEndpoint(part) => write!(f, "this stage has no {part}"),
            Self::UnsupportedArchitecture {
                architecture,
                reason,
            } => write!(f, "architecture {architecture} is unsupported: {reason}"),
            Self::OutOfVocab { token, vocab } => {
                write!(f, "token {token} out of vocab (size {vocab})")
            }
            Self::KvCapacityExhausted { needed, capacity } => write!(
                f,
                "kv capacity exhausted: needed {needed}, capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for ModelError {}
