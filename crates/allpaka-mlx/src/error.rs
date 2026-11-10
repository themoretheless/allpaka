//! Typed errors for MLX safetensors containers.

use std::fmt;
use std::path::PathBuf;

/// Errors produced when loading or slicing MLX `.safetensors` checkpoints.
#[derive(Debug)]
pub enum MlxError {
    /// Filesystem or memory-mapping I/O error.
    Io {
        path: Option<PathBuf>,
        source: std::io::Error,
    },
    /// Invalid JSON in `model.safetensors.index.json` or a shard header.
    Json(serde_json::Error),
    /// Non-UTF-8 bytes in a safetensors JSON header.
    Utf8(std::str::Utf8Error),
    /// `model.safetensors.index.json` is missing a `"weight_map"` object.
    MissingWeightMap,
    /// Shard file is smaller than the 8-byte header-length prefix.
    EmptySafetensors,
    /// Malformed or overflowing safetensors header geometry.
    InvalidHeader(&'static str),
    /// Container has no mapped shards.
    EmptyContainer,
    /// Tensor byte range overflows `usize`/`u64` or exceeds its shard mapping.
    TensorBounds { name: String, detail: &'static str },
}

pub type Result<T> = std::result::Result<T, MlxError>;

impl MlxError {
    pub(crate) fn io_at(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: Some(path.into()),
            source,
        }
    }
}

impl fmt::Display for MlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                path: Some(p),
                source,
            } => write!(f, "{}: {source}", p.display()),
            Self::Io { path: None, source } => Display::fmt(source, f),
            Self::Json(err) => Display::fmt(err, f),
            Self::Utf8(err) => Display::fmt(err, f),
            Self::MissingWeightMap => f.write_str("no weight_map object in safetensors index"),
            Self::EmptySafetensors => f.write_str("empty safetensors"),
            Self::InvalidHeader(msg) => f.write_str(msg),
            Self::EmptyContainer => f.write_str("no mmaps loaded in container"),
            Self::TensorBounds { name, detail } => write!(f, "tensor {name} {detail}"),
        }
    }
}

use fmt::Display;

impl std::error::Error for MlxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(err) => Some(err),
            Self::Utf8(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MlxError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { path: None, source }
    }
}

impl From<serde_json::Error> for MlxError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<std::str::Utf8Error> for MlxError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Utf8(err)
    }
}
