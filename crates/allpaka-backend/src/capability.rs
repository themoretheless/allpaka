//! Backend feature coverage, independent of model-name matching.

use crate::memory::MemoryArchitecture;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Feature {
    DenseFfn = 0,
    SparseMoe = 1,
    GqaAttention = 2,
    QkNorm = 3,
    StandardKv = 4,
    GatedDeltaNet = 5,
    MlaAttention = 6,
    F16Kv = 7,
    PagedKv = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    bits: u16,
}

impl BackendCapabilities {
    #[inline]
    pub fn new(features: impl IntoIterator<Item = Feature>) -> Self {
        let mut bits = 0u16;
        for f in features {
            bits |= 1u16 << (f as u8);
        }
        Self { bits }
    }

    #[inline]
    pub const fn current_metal() -> Self {
        Self {
            bits: (1u16 << (Feature::DenseFfn as u8))
                | (1u16 << (Feature::SparseMoe as u8))
                | (1u16 << (Feature::GqaAttention as u8))
                | (1u16 << (Feature::QkNorm as u8))
                | (1u16 << (Feature::StandardKv as u8))
                | (1u16 << (Feature::GatedDeltaNet as u8))
                | (1u16 << (Feature::F16Kv as u8)),
        }
    }

    /// CUDA runtime coverage (Windows/Linux). Matches Metal for the dense /
    /// MoE / GQA / GDN surface the engine exposes through `gpu::`.
    #[inline]
    pub const fn current_cuda() -> Self {
        Self::current_metal()
    }

    /// Active accelerator profile for this build/host.
    #[inline]
    pub fn current() -> Self {
        static NO_GPU: OnceLock<bool> = OnceLock::new();
        if *NO_GPU.get_or_init(|| std::env::var_os("ALLPAKA_NO_GPU").is_some()) {
            return Self { bits: 0 };
        }
        #[cfg(target_os = "macos")]
        {
            Self::current_metal()
        }
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            Self::current_cuda()
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        {
            Self { bits: 0 }
        }
    }

    #[inline]
    pub const fn supports(&self, feature: Feature) -> bool {
        (self.bits & (1u16 << (feature as u8))) != 0
    }

    /// Physical memory topology of the active accelerator backend (`UMA` on
    /// Apple Silicon Metal, `dUMA` on discrete CUDA GPUs, `HostOnly` when GPU
    /// acceleration is disabled or unavailable).
    #[inline]
    pub fn memory_architecture() -> MemoryArchitecture {
        crate::gpu::memory_architecture()
    }

    pub fn coverage(&self, required: &[Feature]) -> CapabilityCoverage {
        let missing = required
            .iter()
            .copied()
            .filter(|feature| !self.supports(*feature))
            .collect();
        CapabilityCoverage {
            required: required.to_vec(),
            missing,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCoverage {
    pub required: Vec<Feature>,
    pub missing: Vec<Feature>,
}

impl CapabilityCoverage {
    pub fn is_supported(&self) -> bool {
        self.missing.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendCapabilities, Feature};

    #[test]
    fn current_metal_reports_mla_as_missing_instead_of_guessing() {
        let coverage = BackendCapabilities::current_metal()
            .coverage(&[Feature::SparseMoe, Feature::MlaAttention]);
        assert!(!coverage.is_supported());
        assert_eq!(coverage.missing, vec![Feature::MlaAttention]);
    }
}
