//! Owned intermediate activations. Metal storage is exposed to the CPU only
//! after completion and is detached from the reusable GPU arena while borrowed.
use std::ops::{Deref, DerefMut};

pub enum Activation {
    Host(Vec<f32>),
    #[cfg(target_os = "macos")]
    Shared(crate::gpu::SharedVector),
}

impl Activation {
    pub fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Host(v) => v,
            #[cfg(target_os = "macos")]
            Self::Shared(v) => v.to_vec(),
        }
    }
}

impl From<Vec<f32>> for Activation {
    fn from(v: Vec<f32>) -> Self {
        Self::Host(v)
    }
}

impl Deref for Activation {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            Self::Host(v) => v,
            #[cfg(target_os = "macos")]
            Self::Shared(v) => v,
        }
    }
}

impl DerefMut for Activation {
    fn deref_mut(&mut self) -> &mut [f32] {
        match self {
            Self::Host(v) => v,
            #[cfg(target_os = "macos")]
            Self::Shared(v) => v,
        }
    }
}

/// Cumulative proof that a benchmark exercised shared-buffer handoffs.
#[derive(Debug, Default, Clone, Copy)]
pub struct SharedStats {
    pub outputs: u64,
    pub handoff_bytes: u64,
    pub projection_bytes: u64,
}
