//! Declarative model requirements, separate from backend capabilities.

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
    pub architecture: String,
    pub required: Vec<Feature>,
    pub optional: Vec<Feature>,
}

impl ModelRequirements {
    pub fn for_architecture(architecture: &str) -> Option<Self> {
        let a = architecture;
        let (family, required, optional) = if a.eq_ignore_ascii_case("qwen2") {
            (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv, Feature::QkNorm],
            )
        } else if a.eq_ignore_ascii_case("qwen3") {
            (
                ModelFamily::Qwen3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("qwen3moe") || a.eq_ignore_ascii_case("qwen3_moe") {
            (
                ModelFamily::Qwen3Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("qwen35moe")
            || a.eq_ignore_ascii_case("qwen3.5moe")
            || a.eq_ignore_ascii_case("qwen3_5_moe")
        {
            (
                ModelFamily::Qwen35Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::QkNorm,
                    Feature::StandardKv,
                    Feature::GatedDeltaNet,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("llama") || a.eq_ignore_ascii_case("llama3") {
            (
                ModelFamily::Llama3,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("mistral") {
            (
                ModelFamily::Mistral,
                vec![
                    Feature::DenseFfn,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("glm4moe")
            || a.eq_ignore_ascii_case("glm4_moe")
            || a.eq_ignore_ascii_case("glm_moe")
        {
            (
                ModelFamily::Glm4Moe,
                vec![
                    Feature::SparseMoe,
                    Feature::GqaAttention,
                    Feature::StandardKv,
                ],
                vec![Feature::PagedKv],
            )
        } else if a.eq_ignore_ascii_case("deepseek")
            || a.eq_ignore_ascii_case("deepseek2")
            || a.eq_ignore_ascii_case("deepseek_mla")
        {
            (
                ModelFamily::DeepSeekMla,
                vec![Feature::SparseMoe, Feature::MlaAttention, Feature::F16Kv],
                vec![Feature::PagedKv],
            )
        } else {
            return None;
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
