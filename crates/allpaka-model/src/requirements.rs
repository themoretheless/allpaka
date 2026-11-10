//! Declarative model requirements, separate from backend capabilities.

use crate::config::Architecture;
use allpaka_backend::capability::Feature;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3,
    Qwen3Moe,
    Qwen35Moe,
    Llama3,
    Mistral,
    Glm4Moe,
    DeepSeekMla,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRequirements {
    pub family: ModelFamily,
    pub architecture: Architecture,
    pub required: Vec<Feature>,
    pub optional: Vec<Feature>,
}

impl ModelRequirements {
    pub fn for_architecture(architecture: &str) -> Option<Self> {
        Self::from_architecture(Architecture::parse(architecture))
    }

    pub fn from_architecture(architecture: Architecture) -> Option<Self> {
        let (family, required, optional) = match &architecture {
            Architecture::Qwen2 => (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv, Feature::QkNorm],
            ),
            Architecture::Qwen3 => (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::Qwen3Moe => (
                ModelFamily::Qwen3Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::Qwen35Moe => (
                ModelFamily::Qwen35Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                    Feature::GatedDeltaNet,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::Llama => (
                ModelFamily::Llama3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::Mistral => (
                ModelFamily::Mistral,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::Glm4Moe => (
                ModelFamily::Glm4Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            Architecture::DeepSeek2 | Architecture::DeepSeek3 => (
                ModelFamily::DeepSeekMla,
                vec![Feature::SparseMoe, Feature::MlaAttention, Feature::F16Kv],
                vec![Feature::PagedKv],
            ),
            Architecture::Gemma4 | Architecture::Other(_) => return None,
        };
        Some(Self {
            family,
            architecture,
            required,
            optional,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ModelRequirements;
    use crate::config::Architecture;
    use allpaka_backend::capability::{BackendCapabilities, Feature};

    #[test]
    fn qwen_moe_and_deepseek_have_distinct_attention_requirements() {
        let qwen = ModelRequirements::for_architecture("qwen3moe").unwrap();
        let deepseek = ModelRequirements::for_architecture("deepseek_mla").unwrap();
        assert_eq!(qwen.architecture, Architecture::Qwen3Moe);
        assert_eq!(deepseek.architecture, Architecture::DeepSeek2);
        assert!(qwen.required.contains(&Feature::GqaAttention));
        assert!(deepseek.required.contains(&Feature::MlaAttention));
        assert!(BackendCapabilities::current_metal()
            .coverage(&qwen.required)
            .is_supported());
        assert!(!BackendCapabilities::current_metal()
            .coverage(&deepseek.required)
            .is_supported());
    }

    #[test]
    fn glm4moe_is_registered_as_sparse_moe_with_gqa() {
        let glm = ModelRequirements::for_architecture("glm4moe").unwrap();
        assert_eq!(glm.family, super::ModelFamily::Glm4Moe);
        assert_eq!(glm.architecture, Architecture::Glm4Moe);
        assert!(glm.required.contains(&Feature::SparseMoe));
        assert!(glm.required.contains(&Feature::GqaAttention));
        assert!(BackendCapabilities::current_metal()
            .coverage(&glm.required)
            .is_supported());
    }
}
