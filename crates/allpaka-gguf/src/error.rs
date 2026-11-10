//! Typed errors for GGUF container parsing, mapping, and dequantisation.

use allpaka_dtype::{DtypeError, GgmlType};
use std::fmt;
use std::path::PathBuf;

/// Errors produced when reading, memory-mapping, or dequantising GGUF files.
#[derive(Debug)]
pub enum GgufError {
    /// Filesystem or stream I/O error with optional operation/path context.
    Io {
        context: Option<String>,
        source: std::io::Error,
    },
    /// Underlying dtype geometry error from `allpaka-dtype`.
    Dtype(DtypeError),
    /// File does not begin with the 4-byte `GGUF` magic.
    InvalidMagic,
    /// GGUF format version predates v2.
    UnsupportedVersion(u32),
    /// Missing required header metadata key (e.g. `<arch>.block_count`).
    MissingField(String),
    /// Missing split GGUF shard on disk.
    MissingSplitPart(PathBuf),
    /// Structural GGUF header or stream violation.
    Format(String),
    /// Tensor geometry, offset, or split-part bounds violation.
    Tensor { name: String, detail: String },
    /// Dequantisation error (byte length mismatch or unsupported dtype).
    Dequant {
        tensor: Option<String>,
        detail: String,
    },
}

pub type Result<T> = std::result::Result<T, GgufError>;

impl GgufError {
    pub(crate) fn io_ctx(ctx: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: Some(ctx.into()),
            source,
        }
    }

    pub(crate) fn with_context(self, ctx: impl Into<String>) -> Self {
        let prefix = ctx.into();
        match self {
            Self::Io {
                context: Some(inner),
                source,
            } => Self::Io {
                context: Some(format!("{prefix}: {inner}")),
                source,
            },
            Self::Io {
                context: None,
                source,
            } => Self::Io {
                context: Some(prefix),
                source,
            },
            Self::Format(msg) => Self::Format(format!("{prefix}: {msg}")),
            other => Self::Format(format!("{prefix}: {other}")),
        }
    }

    pub(crate) fn unsupported_tensor_type(name: &str, ty: GgmlType) -> Self {
        Self::Tensor {
            name: name.to_string(),
            detail: format!("unsupported ggml type {ty:?}"),
        }
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                context: Some(ctx),
                source,
            } => write!(f, "{ctx}: {source}"),
            Self::Io {
                context: None,
                source,
            } => fmt::Display::fmt(source, f),
            Self::Dtype(err) => fmt::Display::fmt(err, f),
            Self::InvalidMagic => f.write_str("not a GGUF file (bad magic)"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "GGUF version {v} is not supported; version 2 or later is required"
                )
            }
            Self::MissingField(key) => write!(f, "GGUF file has no {key}"),
            Self::MissingSplitPart(path) => {
                write!(f, "split GGUF part missing: {}", path.display())
            }
            Self::Format(msg) => f.write_str(msg),
            Self::Tensor { name, detail } => write!(f, "tensor {name:?}: {detail}"),
            Self::Dequant {
                tensor: Some(name),
                detail,
            } => write!(f, "dequantising tensor {name:?}: {detail}"),
            Self::Dequant {
                tensor: None,
                detail,
            } => f.write_str(detail),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Dtype(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(source: std::io::Error) -> Self {
        Self::Io {
            context: None,
            source,
        }
    }
}

impl From<DtypeError> for GgufError {
    fn from(err: DtypeError) -> Self {
        Self::Dtype(err)
    }
}
