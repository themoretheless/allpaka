//! Shape of the model being served.
//!
//! Everything here is what the planner needs and nothing more: how many layer
//! boundaries exist (that is where a pipeline split can be cut), how many bytes
//! each layer weighs at the chosen quantisation, and how wide the activation
//! tensor is that must cross the wire at a cut point.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    /// Number of transformer blocks. There are `n_layers - 1` places to cut.
    pub n_layers: u32,
    /// Residual stream width. This is what crosses the network at a cut.
    pub hidden_size: u32,
    /// Total weight bytes at the target quantisation, including embeddings.
    pub total_weight_bytes: u64,
    /// Bytes of KV cache per token per layer, at the KV cache dtype.
    pub kv_bytes_per_token_per_layer: u64,
    /// Bytes per activation element on the wire. 2 for f16/bf16.
    #[serde(default = "default_activation_bytes")]
    pub activation_bytes: u32,
    /// Share of the weights that must actually be read to decode one token.
    ///
    /// 1.0 for a dense model. A mixture of experts is far lower, because the
    /// router picks a handful of experts per token and the rest are never
    /// touched. This is the single most important number for a MoE model on
    /// this kind of hardware: **capacity is charged on the whole file, speed
    /// only on the active share**, which is exactly the trade a machine with
    /// lots of slow-ish memory wants.
    #[serde(default = "default_active_fraction")]
    pub active_weight_fraction: f64,
    /// Total parameter count, from the GGUF tensor shapes.
    ///
    /// Needed for the prefill phase, which is compute-bound: its cost is
    /// FLOPs, roughly `2 * active params` per prompt token, and parameters
    /// cannot be recovered from `total_weight_bytes` without knowing the
    /// quantisation (Q2 and F16 differ six-fold in bytes per parameter).
    /// 0 means unknown, and prefill time is then reported as unknown too.
    #[serde(default)]
    pub param_count: u64,
}

fn default_active_fraction() -> f64 {
    1.0
}

fn default_activation_bytes() -> u32 {
    2
}

impl Model {
    /// Average weight bytes attributable to one transformer block.
    ///
    /// Embeddings and the output head are folded in here rather than tracked
    /// separately; for split planning the small error this introduces is far
    /// below the uncertainty in the bandwidth numbers.
    pub fn bytes_per_layer(&self) -> u64 {
        if self.n_layers == 0 {
            return 0;
        }
        self.total_weight_bytes / self.n_layers as u64
    }

    /// Weight bytes for a contiguous run of `layers` blocks. This is the
    /// residency cost: every one of these bytes has to be in memory.
    pub fn bytes_for_layers(&self, layers: u32) -> u64 {
        self.bytes_per_layer() * layers as u64
    }

    /// Weight bytes actually streamed per decode step for those layers. Equal
    /// to `bytes_for_layers` on a dense model, far smaller on a MoE.
    pub fn active_bytes_for_layers(&self, layers: u32) -> u64 {
        (self.bytes_for_layers(layers) as f64 * self.active_weight_fraction.clamp(0.0, 1.0)) as u64
    }

    /// Bytes actually streamed through memory for one decode step over these
    /// layers: the active weights plus the KV cache.
    ///
    /// The KV term is what makes long context slow, not just large. Every step
    /// attends over everything generated so far, and unlike MoE weights the
    /// cache is dense - no router skips any of it. Charged at the full context
    /// rather than the average seen while filling it, for the same reason the
    /// residency budget uses the full context: the plan has to hold at the end
    /// of the conversation, not at its start.
    pub fn streamed_bytes_per_step(&self, layers: u32, context_tokens: u32) -> u64 {
        self.active_bytes_for_layers(layers) + self.kv_bytes(layers, context_tokens)
    }

    /// Parameters actually multiplied per token for a run of `layers` blocks.
    ///
    /// The same sparsity that thins decode streaming thins prefill compute: a
    /// MoE evaluates only the routed experts during prompt processing too.
    pub fn active_params_for_layers(&self, layers: u32) -> f64 {
        if self.n_layers == 0 {
            return 0.0;
        }
        self.param_count as f64 * layers as f64 / self.n_layers as f64
            * self.active_weight_fraction.clamp(0.0, 1.0)
    }

    /// Whether the router leaves most of the weights untouched per token.
    pub fn is_sparse(&self) -> bool {
        self.active_weight_fraction < 0.95
    }

    /// KV cache bytes for `layers` blocks at a given context length.
    pub fn kv_bytes(&self, layers: u32, context_tokens: u32) -> u64 {
        self.kv_bytes_per_token_per_layer * layers as u64 * context_tokens as u64
    }

    /// Bytes that must cross a pipeline cut for a batch of `tokens`.
    ///
    /// One hidden-state tensor of shape `[tokens, hidden_size]` per cut.
    pub fn cut_payload_bytes(&self, tokens: u32) -> u64 {
        self.hidden_size as u64 * tokens as u64 * self.activation_bytes as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model {
            name: "m".into(),
            n_layers: 64,
            hidden_size: 5120,
            total_weight_bytes: 30 << 30,
            kv_bytes_per_token_per_layer: 4096,
            activation_bytes: 2,
            active_weight_fraction: 1.0,
            param_count: 0,
        }
    }

    /// Long context is slow, not just large: every decode step reads the whole
    /// KV cache on top of the weights.
    #[test]
    fn a_longer_context_streams_more_bytes_per_step() {
        let m = model();
        let short = m.streamed_bytes_per_step(m.n_layers, 4096);
        let long = m.streamed_bytes_per_step(m.n_layers, 65536);
        assert!(long > short);
        assert_eq!(long - short, m.kv_bytes(m.n_layers, 65536 - 4096));
    }

    /// Sparsity thins the weight term only. The KV term is dense, so a MoE at
    /// long context is nowhere near `active_weight_fraction` times the cost.
    #[test]
    fn the_kv_term_is_not_reduced_by_sparsity() {
        let mut m = model();
        m.active_weight_fraction = 0.1;
        let streamed = m.streamed_bytes_per_step(m.n_layers, 32768);
        let weights_only = m.active_bytes_for_layers(m.n_layers);
        assert_eq!(streamed - weights_only, m.kv_bytes(m.n_layers, 32768));
        assert!(streamed > weights_only * 2, "KV should dominate a sparse step here");
    }
}
