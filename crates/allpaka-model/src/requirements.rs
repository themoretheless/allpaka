//! Declarative model requirements, separate from backend capabilities.

use allpaka_backend::capability::Feature;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3,
    Qwen3Moe,
    Qwen35Moe,
    Llama3,
    Mistral,
    Gemma3,
    Glm4Moe,
    DeepSeekMla,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRequirements {
    pub family: ModelFamily,
    pub architecture: String,
    pub required: Vec<Feature>,
    pub optional: Vec<Feature>,
}

impl ModelRequirements {
    pub fn for_architecture(architecture: &str) -> Option<Self> {
        use crate::arch::Architecture as A;
        let (family, required, optional) = match A::parse(architecture) {
            A::Qwen2 => (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv, Feature::QkNorm],
            ),
            A::Qwen3 => (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::Qwen3Moe => (
                ModelFamily::Qwen3Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::Qwen35Moe => (
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
            A::Llama => (
                ModelFamily::Llama3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::Gemma3 => (
                ModelFamily::Gemma3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::Mistral => (
                ModelFamily::Mistral,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::Glm4Moe => (
                ModelFamily::Glm4Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            A::DeepSeek2 | A::DeepSeek3 => (
                ModelFamily::DeepSeekMla,
                vec![Feature::SparseMoe, Feature::MlaAttention, Feature::F16Kv],
                vec![Feature::PagedKv],
            ),
            A::Gemma4 | A::Other(_) => return None,
        };
        Some(Self {
            family,
            architecture: architecture.to_string(),
            required,
            optional,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ModelRequirements;
    use allpaka_backend::capability::{BackendCapabilities, Feature};

    #[test]
    fn qwen_moe_and_deepseek_have_distinct_attention_requirements() {
        let qwen = ModelRequirements::for_architecture("qwen3moe").unwrap();
        let deepseek = ModelRequirements::for_architecture("deepseek_mla").unwrap();
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
        assert!(glm.required.contains(&Feature::SparseMoe));
        assert!(glm.required.contains(&Feature::GqaAttention));
        assert!(BackendCapabilities::current_metal()
            .coverage(&glm.required)
            .is_supported());
    }
}
