//! The crate's error type. Like `anyhow`, an error carries the outermost
//! context as its message and the cause as `source()`, so callers that print
//! the chain (`{:#}` through anyhow) see the same text as before; unlike
//! `anyhow`, callers can match on what went wrong.

use allpaka_dtype::DtypeError;
use std::fmt;

#[derive(Debug)]
pub enum GgufError {
    /// Reading, opening or mapping a file failed.
    Io(std::io::Error),
    /// The file is not valid GGUF, or holds something this reader rejects.
    Format(String),
    /// A tensor's element count does not fit its dtype.
    Dtype(DtypeError),
    /// What was being done when `source` happened.
    Context {
        context: String,
        source: Box<GgufError>,
    },
}

pub type Result<T, E = GgufError> = std::result::Result<T, E>;

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => e.fmt(f),
            GgufError::Format(m) => f.write_str(m),
            GgufError::Dtype(e) => e.fmt(f),
            GgufError::Context { context, .. } => f.write_str(context),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Io and Dtype already print the wrapped error, so the chain
        // continues from its own source instead of repeating it.
        match self {
            GgufError::Io(e) => e.source(),
            GgufError::Dtype(e) => e.source(),
            GgufError::Context { source, .. } => Some(source.as_ref()),
            GgufError::Format(_) => None,
        }
    }
}

impl GgufError {
    /// The innermost error, past every layer of context.
    pub fn root(&self) -> &GgufError {
        match self {
            GgufError::Context { source, .. } => source.root(),
            other => other,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

impl From<DtypeError> for GgufError {
    fn from(e: DtypeError) -> Self {
        GgufError::Dtype(e)
    }
}

/// `anyhow::Context` for this crate's error type.
pub trait Context<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T>;
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T, E: Into<GgufError>> Context<T> for std::result::Result<T, E> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|e| GgufError::Context {
            context: context.to_string(),
            source: Box::new(e.into()),
        })
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| GgufError::Context {
            context: f().to_string(),
            source: Box::new(e.into()),
        })
    }
}

impl<T> Context<T> for Option<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.ok_or_else(|| GgufError::Format(context.to_string()))
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| GgufError::Format(f().to_string()))
    }
}

impl From<std::num::TryFromIntError> for GgufError {
    fn from(e: std::num::TryFromIntError) -> Self {
        GgufError::Format(e.to_string())
    }
}

/// Return early with a [`GgufError::Format`] built like `format!`.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::GgufError::Format(format!($($arg)*)))
    };
}
pub(crate) use bail;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_wraps_like_anyhow() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err = Err::<(), _>(io).context("opening x.gguf").unwrap_err();
        assert_eq!(err.to_string(), "opening x.gguf");
        assert!(matches!(err.root(), GgufError::Io(_)));
        let chain = format!("{:#}", anyhow::Error::from(err));
        assert_eq!(chain, "opening x.gguf: gone");
        let none = None::<u8>.context("no split parts").unwrap_err();
        assert!(matches!(none, GgufError::Format(ref m) if m == "no split parts"));
    }
}
