//! The crate's error type: what failed while reading an MLX checkpoint.

use std::fmt;

#[derive(Debug)]
pub enum MlxError {
    /// Opening, reading or mapping a file failed.
    Io(std::io::Error),
    /// An index or safetensors header is not valid JSON.
    Json(serde_json::Error),
    /// The checkpoint is malformed or points outside its files.
    Format(String),
}

pub type Result<T, E = MlxError> = std::result::Result<T, E>;

impl fmt::Display for MlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlxError::Io(e) => e.fmt(f),
            MlxError::Json(e) => e.fmt(f),
            MlxError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MlxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // The wrapped error is already the message; continue past it.
        match self {
            MlxError::Io(e) => e.source(),
            MlxError::Json(e) => e.source(),
            MlxError::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for MlxError {
    fn from(e: std::io::Error) -> Self {
        MlxError::Io(e)
    }
}

impl From<serde_json::Error> for MlxError {
    fn from(e: serde_json::Error) -> Self {
        MlxError::Json(e)
    }
}

impl From<std::array::TryFromSliceError> for MlxError {
    fn from(e: std::array::TryFromSliceError) -> Self {
        MlxError::Format(e.to_string())
    }
}

impl From<std::str::Utf8Error> for MlxError {
    fn from(e: std::str::Utf8Error) -> Self {
        MlxError::Format(format!("safetensors header is not UTF-8: {e}"))
    }
}

/// A [`MlxError::Format`] built like `format!`.
macro_rules! format_err {
    ($($arg:tt)*) => {
        $crate::error::MlxError::Format(format!($($arg)*))
    };
}
pub(crate) use format_err;
