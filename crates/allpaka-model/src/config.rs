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

/// Gated delta net (linear attention) layer parameters, qwen35moe. Layers
/// with `(index + 1) % full_attention_interval != 0` are GDN; the rest are
/// full attention.
#[derive(Debug, Clone)]
pub struct SsmConfig {
    pub d_inner: u32,
    pub d_state: u32,
    pub dt_rank: u32,
    pub n_group: u32,
    pub d_conv: u32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub architecture: String,
    pub n_layers: u32,
    /// An MTP (nextn) speculative block sits past the trunk layers
    /// (qwen35moe: blk.n_layers); the trunk is `n_layers` deep.
    pub nextn: bool,
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
    /// IMROPE frequency-band sizes (t,h,w,e) of qwen35 models; for text the
    /// per-band positions coincide, which collapses the bands to the plain
    /// partial-NEOX table - kept for documentation/verification.
    pub rope_sections: [u32; 4],
    /// Every Nth layer is full attention (qwen35moe: 4); 0 = every layer.
    pub full_attention_interval: u32,
    /// Gated delta net parameters when the model has linear-attention layers.
    pub ssm: Option<SsmConfig>,
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
                // qwen35moe omits expert_shared_count but ships the tensors.
                n_shared: f.meta_u32(&key("expert_shared_count")).unwrap_or_else(|| {
                    u32::from(f.tensor("blk.0.ffn_up_shexp.weight").is_some())
                }),
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
        // attention weights are absent rather than failing the load. When the
        // MTP block IS present (qwen35moe nextn), it is not a trunk layer:
        // subtract it - the trunk is layers 0..n_layers-nextn.
        let nextn = f.meta_u32(&key("nextn_predict_layers")).unwrap_or(0);
        let mut n_layers = need("block_count")?;
        while n_layers > 0
            && f.tensor(&format!("blk.{}.attn_q.weight", n_layers - 1)).is_none()
        {
            n_layers -= 1;
        }
        let nextn = nextn.min(n_layers);
        n_layers -= nextn;

        Ok(Config {
            n_layers,
            nextn: nextn > 0,
            n_kv_heads: f.meta_u32(&key("attention.head_count_kv")).unwrap_or(n_heads),
            // Absent on linear-attention hybrids (qwen35moe has no dense FFN).
            ffn_hidden: f.meta_u32(&key("feed_forward_length")).unwrap_or(0),
            moe,
            vocab: embd.dims[1] as u32,
            rms_eps: f
                .meta_f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_freq_base: f.meta_f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_style,
            rope_sections: f
                .meta_u32_arr(&key("rope.dimension_sections"))
                .map(|s| [s[0], s[1], s[2], s[3]])
                .unwrap_or([0, 0, 0, 0]),
            full_attention_interval: f
                .meta_u32(&key("full_attention_interval"))
                .unwrap_or(0),
            ssm: match f.meta_u32(&key("ssm.inner_size")) {
                Some(d_inner) => Some(SsmConfig {
                    d_inner,
                    d_state: need("ssm.state_size")?,
                    dt_rank: need("ssm.time_step_rank")?,
                    n_group: need("ssm.group_count")?,
                    d_conv: need("ssm.conv_kernel")?,
                }),
                None => None,
            },
            // Probe beyond layer 0: qwen35moe's layer 0 is a gated-delta-net
            // layer, and only its full-attention layers carry q/k norms.
            has_qk_norm: (0..n_layers as usize)
                .any(|i| f.tensor(&format!("blk.{i}.attn_q_norm.weight")).is_some()),
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
