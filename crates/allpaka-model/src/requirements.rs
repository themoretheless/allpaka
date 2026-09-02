//! Declarative model requirements, separate from backend capabilities.

use allpaka_backend::capability::Feature;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3,
    Qwen3Moe,
    Qwen35Moe,
    Llama3,
    Mistral,
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
        let normalized = architecture.to_ascii_lowercase();
        let (family, required, optional) = match normalized.as_str() {
            "qwen3" => (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            "qwen3moe" | "qwen3_moe" => (
                ModelFamily::Qwen3Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            "qwen35moe" | "qwen3.5moe" | "qwen3_5_moe" => (
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
            "llama" | "llama3" => (
                ModelFamily::Llama3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            "mistral" => (
                ModelFamily::Mistral,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            ),
            "deepseek" | "deepseek2" | "deepseek_mla" => (
                ModelFamily::DeepSeekMla,
                vec![
                    Feature::SparseMoe,
                    Feature::MlaAttention,
                    Feature::F16Kv,
                ],
                vec![Feature::PagedKv],
            ),
            _ => return None,
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
}
