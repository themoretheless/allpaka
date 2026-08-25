//! The Llama-family hyperparameters, read from GGUF metadata.
//!
//! One config type covers Qwen3, Llama 3.x and Mistral because they are the
//! same transformer with different numbers; what genuinely differs between
//! them is captured in two switches (RoPE pairing style, presence of QK
//! norms), and both are decided here, once, from the file - not scattered
//! through the forward pass as architecture ifs.

use allpaka_gguf::GgufFile;
use anyhow::{bail, Context, Result};

/// Which channels rotate together in RoPE. Getting this wrong produces
/// grammatical nonsense rather than an error, which is why it is decided from
/// the architecture string in exactly one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Adjacent pairs `(2i, 2i+1)`: Llama and Mistral.
    Norm,
    /// Split halves `(i, i+d/2)`: Qwen and most newer families.
    Neox,
}

/// How the MoE router turns logits into expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    /// Softmax over all experts, then top-k (Qwen3-MoE).
    Softmax,
    /// Sigmoid per expert, top-k by score, weights renormalised over the
    /// winners (GLM-4.5, `expert_gating_func = 2`).
    Sigmoid,
}

/// Mixture-of-experts hyperparameters, present when the file declares experts.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub n_expert: u32,
    pub n_used: u32,
    /// Hidden width of one expert's FFN (usually much narrower than a dense
    /// model's `ffn_hidden`).
    pub expert_ffn: u32,
    /// How many leading blocks are dense before routing starts (GLM: 1).
    pub leading_dense: u32,
    /// Always-on shared experts (GLM: 1); their FFN is `expert_ffn * shared`
    /// wide and its output is added unweighted.
    pub n_shared: u32,
    pub gating: Gating,
    /// Renormalise the winners' weights to sum to 1 (`expert_weights_norm`).
    pub weights_norm: bool,
    /// Multiplier on the final weights (`expert_weights_scale`).
    pub weights_scale: f32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub architecture: String,
    pub n_layers: u32,
    pub hidden: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// Per-head dimension. Read from `attention.key_length` when present:
    /// Qwen3 uses 128 even where `hidden / n_heads` says otherwise.
    pub head_dim: u32,
    pub ffn_hidden: u32,
    pub vocab: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_style: RopeStyle,
    /// Qwen3 normalises each attention head's q and k before RoPE; Llama and
    /// Mistral do not. Decided by whether the tensors exist, not by name
    /// matching on the architecture.
    pub has_qk_norm: bool,
    /// q/k/v projection biases (GLM: attention_bias). Detected by tensor
    /// presence, like the QK norms.
    pub has_attn_bias: bool,
    /// Rotary width: GLM rotates only the first 64 of 128 head dims
    /// (`rope.dimension_count`); the rest pass through unchanged.
    pub rope_dim: u32,
    /// Present for a mixture of experts; the FFN then routes instead of being
    /// one dense block.
    pub moe: Option<MoeConfig>,
}

impl Config {
    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let arch = f.architecture().to_string();
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let need = |suffix: &str| -> Result<u32> {
            f.meta_u32(&key(suffix)).with_context(|| format!("GGUF has no {}", key(suffix)))
        };

        let n_expert = f.meta_u32(&key("expert_count")).unwrap_or(0);
        let moe = if n_expert > 0 {
            let n_used = f
                .meta_u32(&key("expert_used_count"))
                .with_context(|| format!("{arch} declares experts but no expert_used_count"))?;
            if n_used == 0 || n_used > n_expert {
                bail!("{arch} uses {n_used} of {n_expert} experts, which makes no sense");
            }
            // expert_gating_func: 0/absent = softmax, 2 = sigmoid (llama.cpp's
            // enum; 1 is softmax-with-groups, treated as softmax here).
            let gating = match f.meta_u32(&key("expert_gating_func")).unwrap_or(0) {
                2 => Gating::Sigmoid,
                _ => Gating::Softmax,
            };
            Some(MoeConfig {
                n_expert,
                n_used,
                expert_ffn: need("expert_feed_forward_length")?,
                leading_dense: f.meta_u32(&key("leading_dense_block_count")).unwrap_or(0),
                n_shared: f.meta_u32(&key("expert_shared_count")).unwrap_or(0),
                gating,
                weights_norm: f.meta_bool(&key("expert_weights_norm")).unwrap_or(true),
                weights_scale: f.meta_f32(&key("expert_weights_scale")).unwrap_or(1.0),
            })
        } else {
            None
        };

        let n_heads = need("attention.head_count")?;
        let hidden = need("embedding_length")?;
        let head_dim = f
            .meta_u32(&key("attention.key_length"))
            .unwrap_or_else(|| hidden / n_heads.max(1));

        // Vocabulary size comes from the embedding tensor's shape, so no
        // tokenizer metadata is needed to produce logits.
        let embd = f
            .tensor("token_embd.weight")
            .context("GGUF has no token_embd.weight tensor")?;
        if embd.dims.len() != 2 || embd.dims[0] != hidden as u64 {
            bail!(
                "token_embd.weight has shape {:?}, expected [{hidden}, vocab]",
                embd.dims
            );
        }

        let rope_style = match arch.as_str() {
            "llama" | "mistral" => RopeStyle::Norm,
            _ => RopeStyle::Neox,
        };

        // GLM rotates only `rope.dimension_count` of the head_dim channels.
        let rope_dim = f
            .meta_u32(&key("rope.dimension_count"))
            .unwrap_or(head_dim)
            .min(head_dim);
        if rope_dim == 0 || rope_dim % 2 != 0 {
            bail!("{arch}: rope.dimension_count {rope_dim} must be a positive even number");
        }

        // Some converters count speculative (MTP) blocks in block_count but do
        // not write their tensors (GLM-4.5's nextn layer); trim blocks whose
        // attention weights are absent rather than failing the load.
        let mut n_layers = need("block_count")?;
        while n_layers > 0
            && f.tensor(&format!("blk.{}.attn_q.weight", n_layers - 1)).is_none()
        {
            n_layers -= 1;
        }

        Ok(Config {
            n_layers,
            n_kv_heads: f.meta_u32(&key("attention.head_count_kv")).unwrap_or(n_heads),
            ffn_hidden: need("feed_forward_length")?,
            moe,
            vocab: embd.dims[1] as u32,
            rms_eps: f
                .meta_f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_freq_base: f.meta_f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_style,
            has_qk_norm: f.tensor("blk.0.attn_q_norm.weight").is_some(),
            has_attn_bias: f.tensor("blk.0.attn_q.bias").is_some(),
            rope_dim,
            architecture: arch,
            n_heads,
            hidden,
            head_dim,
        })
    }

    pub fn q_dim(&self) -> usize {
        (self.n_heads * self.head_dim) as usize
    }

    pub fn kv_dim(&self) -> usize {
        (self.n_kv_heads * self.head_dim) as usize
    }

    /// How many query heads share one kv head under grouped-query attention.
    pub fn group_size(&self) -> usize {
        (self.n_heads / self.n_kv_heads.max(1)) as usize
    }
}
