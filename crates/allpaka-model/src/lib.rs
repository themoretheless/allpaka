//! The Llama-family forward pass, parameterised by GGUF metadata.
//!
//! Qwen3 (dense), Llama 3.x and Mistral all run through the same graph; the
//! genuine differences - RoPE pairing style and Qwen's QK norms - are config
//! switches decided once at load. Mixtures of experts and DeepSeek's MLA are
//! the two known extensions and both have their seams reserved: MoE hangs off
//! the FFN block, MLA lives entirely behind the [`kv`] interface.
//!
//! Correctness story, in order of strength: unit invariants here (causality,
//! cache-vs-recompute equality), then logit comparison against llama.cpp on a
//! real model, which is the acceptance test for the whole engine milestone.

pub mod config;
pub mod error;
pub mod kv;
pub mod model;
pub mod pipeline;
pub mod prefix_cache;
pub mod profile;
pub mod requirements;
pub mod speculate;
pub mod tokenizer;

pub use config::{Architecture, Config, ModelCapability, ModelSupport, RopeStyle};
pub use error::ModelError;
pub use kv::KvCache;
pub use model::{set_trace_hook, Model, Session};
pub use tokenizer::Tokenizer;
