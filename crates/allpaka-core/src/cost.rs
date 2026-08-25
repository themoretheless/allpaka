//! The arithmetic every placement mode shares.
//!
//! The three optimisers - split, fleet, replicate - search different spaces,
//! but the cost of a model running whole on one pool is one calculation, and it
//! must be the same calculation everywhere. Fixing what a decode step streams
//! (weights *and* KV cache) in three separate places is how the three modes
//! drift into disagreeing about the same hardware.

use crate::speculation::Speculation;
use crate::{Model, Node};

/// Seconds per token for a model running entirely on one pool, with the KV
/// cache read charged and speculation applied if requested.
pub(crate) fn local_secs_per_token(
    node: &Node,
    model: &Model,
    context_tokens: u32,
    spec: Option<&Speculation>,
) -> f64 {
    let pass = node.stream_time(model.streamed_bytes_per_step(model.n_layers, context_tokens));
    match spec {
        None => pass,
        Some(s) => {
            let draft = node.stream_time(s.draft_weight_bytes) * s.draft_tokens as f64;
            let accepted = s.expected_accepted();
            if accepted <= 0.0 {
                f64::INFINITY
            } else {
                (draft + pass) / accepted
            }
        }
    }
}

/// Seconds to process a prompt through a run of `layers` blocks on this pool.
///
/// Prefill is compute-bound, not bandwidth-bound: the prompt is one batched
/// matmul and the weights are read once regardless of its length. Cost is
/// `2 * active params * prompt_tokens / achieved FLOP/s`. The attention
/// quadratic term is deliberately left out - at the prompt lengths this tool
/// budgets for it is second order next to the FLOP/s uncertainty.
///
/// `None` when the node's FLOP/s or the model's parameter count is unknown.
/// An unknown prefill must surface as unknown, because the plans where it
/// matters most - layers spilled to system RAM - are exactly the plans where
/// a made-up number would hide a time-to-first-token measured in minutes.
pub(crate) fn prefill_secs(
    node: &Node,
    model: &Model,
    layers: u32,
    prompt_tokens: u32,
) -> Option<f64> {
    if node.prefill_flops <= 0.0 || model.param_count == 0 {
        return None;
    }
    Some(2.0 * model.active_params_for_layers(layers) * prompt_tokens as f64 / node.prefill_flops)
}

/// Which entries share a physical machine with another entry, and the factor
/// their speed shrinks by.
///
/// Returns `Some(scale)` for co-resident entries, `None` for entries alone on
/// their machine. Dividing `secs_per_token` by the scale is exact rather than
/// an approximation: contention scales the pool's bandwidth, every term of the
/// local cost is bytes over that bandwidth, and a draft model's acceptance rate
/// does not depend on how fast it ran.
pub(crate) fn co_residency_scale(
    hosts: &[&str],
    node_indices: &[usize],
    nodes: &[Node],
) -> Vec<Option<f64>> {
    hosts
        .iter()
        .enumerate()
        .map(|(i, host)| {
            let shared = hosts.iter().enumerate().any(|(j, h)| j != i && h == host);
            shared.then(|| nodes[node_indices[i]].contention.clamp(0.0, 1.0))
        })
        .collect()
}

/// Apply a co-residency scale to a latency.
pub(crate) fn contended(secs_per_token: f64, scale: f64) -> f64 {
    if scale > 0.0 {
        secs_per_token / scale
    } else {
        f64::INFINITY
    }
}
