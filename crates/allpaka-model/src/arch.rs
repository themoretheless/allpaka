//! The one place that turns a GGUF `general.architecture` tag into a model
//! family. `Config` (what the engine can run) and `ModelRequirements` (what
//! the family needs from a backend) both read it, so the two taxonomies
//! cannot drift apart.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// Qwen2.5 GGUF tag; shares the dense Neox + GQA graph with Qwen3.
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen35Moe,
    Llama,
    Mistral,
    Glm4Moe,
    DeepSeek2,
    DeepSeek3,
    Gemma3,
    Gemma4,
    /// Any tag this engine does not know; kept verbatim.
    Other(String),
}

impl Architecture {
    /// Parse a tag case-insensitively. Hugging Face spellings of the same
    /// families (`qwen3_moe`, `glm4_moe`, `llama3`, ...) map to the GGUF tag.
    pub fn parse(tag: &str) -> Self {
        match tag.to_ascii_lowercase().as_str() {
            "qwen2" => Architecture::Qwen2,
            "qwen3" => Architecture::Qwen3,
            "qwen3moe" | "qwen3_moe" => Architecture::Qwen3Moe,
            "qwen35moe" | "qwen3.5moe" | "qwen3_5_moe" => Architecture::Qwen35Moe,
            "llama" | "llama3" => Architecture::Llama,
            "mistral" => Architecture::Mistral,
            "glm4moe" | "glm4_moe" | "glm_moe" => Architecture::Glm4Moe,
            "deepseek2" | "deepseek" | "deepseek_mla" => Architecture::DeepSeek2,
            "deepseek3" => Architecture::DeepSeek3,
            "gemma3" => Architecture::Gemma3,
            "gemma4" => Architecture::Gemma4,
            _ => Architecture::Other(tag.to_string()),
        }
    }

    /// The GGUF tag; an unknown tag comes back as it was given.
    pub fn as_str(&self) -> &str {
        match self {
            Architecture::Qwen2 => "qwen2",
            Architecture::Qwen3 => "qwen3",
            Architecture::Qwen3Moe => "qwen3moe",
            Architecture::Qwen35Moe => "qwen35moe",
            Architecture::Llama => "llama",
            Architecture::Mistral => "mistral",
            Architecture::Glm4Moe => "glm4moe",
            Architecture::DeepSeek2 => "deepseek2",
            Architecture::DeepSeek3 => "deepseek3",
            Architecture::Gemma3 => "gemma3",
            Architecture::Gemma4 => "gemma4",
            Architecture::Other(tag) => tag,
        }
    }

    /// Llama and Mistral rotate adjacent channel pairs; everything else here
    /// uses split halves.
    pub fn rope_style(&self) -> crate::config::RopeStyle {
        match self {
            Architecture::Llama | Architecture::Mistral => crate::config::RopeStyle::Norm,
            _ => crate::config::RopeStyle::Neox,
        }
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Architecture {
    fn from(tag: &str) -> Self {
        Architecture::parse(tag)
    }
}

impl PartialEq<&str> for Architecture {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_and_aliases_round_trip_to_the_gguf_tag() {
        for (tag, want) in [
            ("qwen3", "qwen3"),
            ("Qwen3_MoE", "qwen3moe"),
            ("qwen3.5moe", "qwen35moe"),
            ("llama3", "llama"),
            ("glm_moe", "glm4moe"),
            ("deepseek_mla", "deepseek2"),
            ("gemma3", "gemma3"),
            ("gemma4", "gemma4"),
        ] {
            assert_eq!(Architecture::parse(tag).as_str(), want, "{tag}");
        }
        let unknown = Architecture::parse("Phi3");
        assert_eq!(unknown, Architecture::Other("Phi3".into()));
        assert_eq!(unknown.to_string(), "Phi3");
    }
}
