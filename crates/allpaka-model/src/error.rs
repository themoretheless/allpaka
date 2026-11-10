//! Errors from reading a model's config and loading its weights. Like the
//! gguf crate's error, the outermost context is the message and the cause is
//! `source()`, so `{:#}` chains print as they did with anyhow.

use allpaka_gguf::GgufError;
use std::fmt;

#[derive(Debug)]
pub enum ModelError {
    /// The GGUF file could not be read.
    Gguf(GgufError),
    /// The file reads fine but describes a model this engine cannot load.
    Format(String),
    /// A weight could not be prepared by the backend (which reports anyhow).
    Backend(anyhow::Error),
    /// What was being done when `source` happened.
    Context {
        context: String,
        source: Box<ModelError>,
    },
}

pub type Result<T, E = ModelError> = std::result::Result<T, E>;

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Gguf(e) => e.fmt(f),
            ModelError::Format(m) => f.write_str(m),
            ModelError::Backend(e) => e.fmt(f),
            ModelError::Context { context, .. } => f.write_str(context),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Gguf and Backend already print the wrapped error, so the chain
        // continues from its own source instead of repeating it.
        match self {
            ModelError::Gguf(e) => std::error::Error::source(e),
            ModelError::Backend(e) => e.source(),
            ModelError::Context { source, .. } => Some(source.as_ref()),
            ModelError::Format(_) => None,
        }
    }
}

impl ModelError {
    /// The innermost error, past every layer of context.
    pub fn root(&self) -> &ModelError {
        match self {
            ModelError::Context { source, .. } => source.root(),
            other => other,
        }
    }
}

impl From<GgufError> for ModelError {
    fn from(e: GgufError) -> Self {
        ModelError::Gguf(e)
    }
}

impl From<anyhow::Error> for ModelError {
    fn from(e: anyhow::Error) -> Self {
        ModelError::Backend(e)
    }
}

/// `anyhow::Context` for this crate's error type.
pub trait Context<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T>;
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T, E: Into<ModelError>> Context<T> for std::result::Result<T, E> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|e| ModelError::Context {
            context: context.to_string(),
            source: Box::new(e.into()),
        })
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| ModelError::Context {
            context: f().to_string(),
            source: Box::new(e.into()),
        })
    }
}

impl<T> Context<T> for Option<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.ok_or_else(|| ModelError::Format(context.to_string()))
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| ModelError::Format(f().to_string()))
    }
}

/// Return early with a [`ModelError::Format`] built like `format!`.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::ModelError::Format(format!($($arg)*)))
    };
}
pub(crate) use bail;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_keep_their_cause() {
        let gguf = GgufError::Format("bad magic".into());
        let err = Err::<(), _>(gguf).context("parsing m.gguf").unwrap_err();
        assert!(matches!(err.root(), ModelError::Gguf(_)));
        assert_eq!(
            format!("{:#}", anyhow::Error::from(err)),
            "parsing m.gguf: bad magic"
        );
    }
}
