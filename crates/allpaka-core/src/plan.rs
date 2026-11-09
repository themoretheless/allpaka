//! Deciding where to cut a model, and whether cutting it is worth doing at all.
//!
//! # The model behind the numbers
//!
//! Decoding one token from a dense transformer reads every weight exactly once.
//! Arithmetic intensity is close to 1, so the step time is set by memory
//! bandwidth, not by FLOPS. That gives a usable estimate for a node's share:
//!
//! ```text
//! t_compute(node) = weight_bytes_resident(node) / effective_memory_bandwidth(node)
//! ```
//!
//! Splitting the model does reduce that compute time. Each machine reads only
//! its own layers, through its own memory subsystem, so
//! `Σ(bytes_i / bandwidth_i)` is smaller than `total_bytes / bandwidth_slowest`.
//! Adding a machine with fast memory genuinely speeds up decode.
//!
//! What it costs is network time per forward pass:
//!
//! ```text
//! t_pass = Σ t_compute(node) + Σ t_hop(cut)
//! ```
//!
//! Each hop is charged **one way**, not as a round trip: a stage sends its
//! activations and carries on without waiting for an acknowledgement. A two
//! machine pipeline therefore pays two one-way hops per pass - forward, then
//! the sampled token back to the head - which is one round trip in total, not
//! two. Charging a full round trip per hop doubles the apparent cost of the
//! network and wrongly condemns links that are in fact good enough.
//!
//! For a single interactive request the terms add; they do not overlap.
//! Pipeline parallelism only overlaps stages when several requests are in
//! flight, which raises aggregate throughput but never lowers the latency of
//! any one token.
//!
//! So the whole question is a race between two effects, and it reduces to one
//! comparison:
//!
//! ```text
//! split wins  <=>  network_per_pass  <  compute_saved_per_pass
//! ```
//!
//! The activation crossing a cut is small - hidden size times two bytes, about
//! 10 KB - so throughput barely matters for decode. Tail latency is what
//! matters, because it is paid on every pass.
//!
//! Speculative decoding changes how many tokens come out of a pass, but it
//! scales both sides of that comparison equally, so it never flips the verdict.
//! See `crate::speculation`.
//!
//! # Ordering
//!
//! With unequal links between pairs, which machine holds the head of the
//! pipeline changes the answer, so the planner tries every ordering rather than
//! trusting the order nodes appear in the config.

use crate::speculation::{Speculation, SpeculativeCost};
use crate::{Fabric, Model, Node};

/// One machine's share of the pipeline.
#[derive(Debug, Clone)]
pub struct Stage {
    pub node_index: usize,
    pub node_name: String,
    pub first_layer: u32,
    pub layer_count: u32,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    /// Seconds this stage spends on one decode step.
    pub compute_secs: f64,
    /// Seconds this stage spends processing the prompt, or `None` when the
    /// node's FLOP/s or the model's parameter count is unknown.
    pub prefill_secs: Option<f64>,
}

impl Stage {
    pub fn resident_bytes(&self) -> u64 {
        self.weight_bytes + self.kv_bytes
    }
}

/// A complete assignment of layers to machines, with its predicted cost.
#[derive(Debug, Clone)]
pub struct Plan {
    pub stages: Vec<Stage>,
    /// Sum of per-stage compute for one decode step.
    pub compute_secs_per_token: f64,
    /// Network cost per decode step, using tail latency.
    pub network_secs_per_token: f64,
    /// Seconds to ship the prompt's hidden states across every cut once.
    pub prompt_transfer_secs: f64,
    /// Cost under speculative decoding, when one was requested. Present here
    /// rather than replacing the plain figures so both can be shown side by
    /// side.
    pub speculation: Option<SpeculativeCost>,
}

impl Plan {
    /// Time per output token, under speculation if it was requested.
    pub fn secs_per_token(&self) -> f64 {
        match &self.speculation {
            Some(s) => s.secs_per_token,
            None => self.plain_secs_per_token(),
        }
    }

    /// Time per token with one forward pass per token, ignoring speculation.
    pub fn plain_secs_per_token(&self) -> f64 {
        self.compute_secs_per_token + self.network_secs_per_token
    }

    pub fn tokens_per_sec(&self) -> f64 {
        let t = self.secs_per_token();
        if t <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / t
        }
    }

    pub fn is_split(&self) -> bool {
        self.stages.len() > 1
    }

    /// Network wait per output token. Speculation lowers this by amortising
    /// one round trip over several accepted tokens.
    pub fn network_secs_per_token_effective(&self) -> f64 {
        match &self.speculation {
            Some(s) => s.network_secs_per_token(),
            None => self.network_secs_per_token,
        }
    }

    /// Time to first token: process the prompt on every stage, plus ship its
    /// hidden states across every cut.
    ///
    /// `None` when any stage's prefill is unknown. The unknown case matters
    /// most exactly where the temptation to guess is greatest: layers spilled
    /// to system RAM decode tolerably but prefill at CPU speed, and a plan can
    /// look fine on tok/s while its first token is minutes away.
    pub fn ttft_secs(&self) -> Option<f64> {
        let mut total = self.prompt_transfer_secs;
        for s in &self.stages {
            total += s.prefill_secs?;
        }
        Some(total)
    }

    /// Share of each token's wall clock that is network wait rather than work.
    pub fn network_overhead_fraction(&self) -> f64 {
        let t = self.secs_per_token();
        if t <= 0.0 {
            0.0
        } else {
            self.network_secs_per_token_effective() / t
        }
    }
}

/// What the planner concluded, and why.
#[derive(Debug, Clone)]
pub enum Verdict {
    /// Splitting is both possible and faster than any single machine.
    SplitWins { plan: Plan, single_node: Plan },
    /// A single machine holds the whole model and is faster. Use it.
    UseSingleNode {
        plan: Plan,
        best_split: Option<Plan>,
    },
    /// No single machine can hold the model. Splitting is the only option.
    SplitRequired { plan: Plan },
    /// Nothing fits, split or not.
    Infeasible { reason: String },
}

#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub context_tokens: u32,
    pub prompt_tokens: u32,
    /// Draft model to speculate with, if any.
    pub speculation: Option<Speculation>,
}

impl Default for PlanRequest {
    fn default() -> Self {
        Self {
            context_tokens: 8192,
            prompt_tokens: 2048,
            speculation: None,
        }
    }
}

/// Bytes returned from the tail of the pipeline back to the head each step.
/// A sampled token id, essentially free; included so the return hop's latency
/// is not silently dropped from the estimate.
const RETURN_PAYLOAD_BYTES: u64 = 64;

/// Beyond this many nodes, trying every ordering stops being cheap. Home
/// clusters are far smaller, and the cap keeps a typo in the config from
/// hanging the tool.
const MAX_NODES_FOR_PERMUTATION: usize = 6;

pub fn plan(nodes: &[Node], model: &Model, fabric: &Fabric, req: &PlanRequest) -> Verdict {
    if nodes.is_empty() {
        return Verdict::Infeasible {
            reason: "no nodes configured".into(),
        };
    }
    if model.n_layers == 0 {
        return Verdict::Infeasible {
            reason: "model has zero layers".into(),
        };
    }
    if nodes.len() > MAX_NODES_FOR_PERMUTATION {
        return Verdict::Infeasible {
            reason: format!(
                "{} nodes configured; this planner searches every ordering and is capped at {}",
                nodes.len(),
                MAX_NODES_FOR_PERMUTATION
            ),
        };
    }

    let single = best_single_node(nodes, model, fabric, req);
    let split = best_split(nodes, model, fabric, req);

    match (single, split) {
        (Some(s), Some(sp)) if sp.secs_per_token() < s.secs_per_token() => Verdict::SplitWins {
            plan: sp,
            single_node: s,
        },
        (Some(s), best_split) => Verdict::UseSingleNode {
            plan: s,
            best_split,
        },
        (None, Some(sp)) => Verdict::SplitRequired { plan: sp },
        (None, None) => Verdict::Infeasible {
            reason: format!(
                "no assignment of {} layers fits: total weights {:.1} GiB plus KV cache at {} \
                 context exceeds combined usable memory {:.1} GiB",
                model.n_layers,
                gib(model.total_weight_bytes),
                req.context_tokens,
                gib(nodes.iter().map(|n| n.usable_bytes).sum()),
            ),
        },
    }
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

mod search;
use search::*;

#[cfg(test)]
mod tests;
