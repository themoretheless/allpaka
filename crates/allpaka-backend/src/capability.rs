//! Backend feature coverage, independent of model-name matching.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    DenseFfn,
    SparseMoe,
    GqaAttention,
    QkNorm,
    StandardKv,
    GatedDeltaNet,
    MlaAttention,
    F16Kv,
    PagedKv,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    features: BTreeSet<Feature>,
}

impl BackendCapabilities {
    pub fn new(features: impl IntoIterator<Item = Feature>) -> Self {
        Self {
            features: features.into_iter().collect(),
        }
    }

    pub fn current_metal() -> Self {
        Self::new([
            Feature::DenseFfn,
            Feature::SparseMoe,
            Feature::GqaAttention,
            Feature::QkNorm,
            Feature::StandardKv,
            Feature::GatedDeltaNet,
            Feature::F16Kv,
        ])
    }

    pub fn supports(&self, feature: Feature) -> bool {
        self.features.contains(&feature)
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
