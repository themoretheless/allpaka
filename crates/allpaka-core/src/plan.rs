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

fn best_single_node(
    nodes: &[Node],
    model: &Model,
    fabric: &Fabric,
    req: &PlanRequest,
) -> Option<Plan> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(i, _)| build_plan(&[(i, model.n_layers)], nodes, model, fabric, req))
        .min_by(|a, b| a.secs_per_token().total_cmp(&b.secs_per_token()))
}

/// Fastest feasible split, over every ordering of the machines.
/// For a fixed ordering of nodes, stage compute cost is additive across stages
/// and only depends on `(node_index, layer_count)`. We push DP row computation
/// into the permutation recursion tree so each prefix `order[0..=s]` computes
/// row `s` once for all extensions, pruning missing-link subtrees immediately.
fn best_split(nodes: &[Node], model: &Model, fabric: &Fabric, req: &PlanRequest) -> Option<Plan> {
    let total_layers = model.n_layers as usize;
    if nodes.len() < 2 || total_layers < 2 {
        return None;
    }
    // Precompute the maximum feasible layers each node can hold in memory,
    // and the per-step compute cost for (node_idx, layer_count).
    let max_layers_for_node: Vec<usize> = nodes
        .iter()
        .map(|node| {
            let mut lo = 0usize;
            let mut hi = total_layers;
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                let lc = mid as u32;
                if model.bytes_for_layers(lc) + model.kv_bytes(lc, req.context_tokens)
                    <= node.usable_bytes
                {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        })
        .collect();

    if max_layers_for_node.iter().copied().sum::<usize>() < total_layers {
        return None;
    }

    let stride = total_layers + 1;
    let mut node_layer_costs = vec![f64::INFINITY; nodes.len() * stride];
    for (ni, node) in nodes.iter().enumerate() {
        let max_l = max_layers_for_node[ni];
        for l in 1..=max_l {
            node_layer_costs[ni * stride + l] =
                node.stream_time(model.streamed_bytes_per_step(l as u32, req.context_tokens));
        }
    }

    let cut_payload_1 = model.cut_payload_bytes(1);
    let spec_params = req.speculation.as_ref().map(|spec| {
        let vb = spec.verify_batch();
        (
            model.cut_payload_bytes(vb),
            RETURN_PAYLOAD_BYTES * vb as u64,
            spec.expected_accepted(),
            spec.draft_weight_bytes,
            spec.draft_tokens as f64,
        )
    });

    let mut best: Option<Plan> = None;
    let mut best_secs_per_token = f64::INFINITY;
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    let mut dp = vec![(f64::INFINITY, 0u32); nodes.len() * stride];
    let mut dense = Vec::with_capacity(nodes.len());

    #[allow(clippy::too_many_arguments)]
    fn search_tree(
        s: usize,
        cap_sum: usize,
        fwd_net_1: f64,
        fwd_net_spec: f64,
        order: &mut [usize],
        dp: &mut [(f64, u32)],
        dense: &mut Vec<(usize, u32)>,
        best: &mut Option<Plan>,
        best_secs_per_token: &mut f64,
        max_layers_for_node: &[usize],
        node_layer_costs: &[f64],
        stride: usize,
        total_layers: usize,
        cut_payload_1: u64,
        spec_params: Option<(u64, u64, f64, u64, f64)>,
        nodes: &[Node],
        model: &Model,
        fabric: &Fabric,
        req: &PlanRequest,
    ) {
        let n = order.len();
        for i in s..n {
            order.swap(s, i);
            let ni = order[s];
            let max_take = max_layers_for_node[ni];
            if max_take == 0 {
                order.swap(s, i);
                continue;
            }

            if s == 0 {
                let first_limit = max_take.min(total_layers - 1);
                dp[..=first_limit].fill((f64::INFINITY, 0u32));
                let cost_row = ni * stride;
                for take in 1..=first_limit {
                    dp[take] = (node_layer_costs[cost_row + take], take as u32);
                }
                search_tree(
                    1,
                    max_take,
                    0.0,
                    0.0,
                    order,
                    dp,
                    dense,
                    best,
                    best_secs_per_token,
                    max_layers_for_node,
                    node_layer_costs,
                    stride,
                    total_layers,
                    cut_payload_1,
                    spec_params,
                    nodes,
                    model,
                    fabric,
                    req,
                );
            } else {
                let Some(fwd_link) = fabric.between(order[s - 1], ni) else {
                    order.swap(s, i);
                    continue;
                };
                let next_fwd_1 = fwd_net_1 + fwd_link.one_way_p99(cut_payload_1);
                let next_fwd_spec = match spec_params {
                    Some((cut_spec, _, _, _, _)) => fwd_net_spec + fwd_link.one_way_p99(cut_spec),
                    None => 0.0,
                };
                let next_cap = cap_sum + max_take;

                // Compute DP row `s` once for all extensions of `order[..=s]`.
                let prev_row = (s - 1) * stride;
                let curr_row = s * stride;
                let cost_row = ni * stride;
                let max_curr = total_layers.min(next_cap);
                dp[curr_row..=curr_row + max_curr].fill((f64::INFINITY, 0u32));
                let min_prev = s;
                let max_prev = (total_layers - 1).min(cap_sum);
                for prev_l in min_prev..=max_prev {
                    let prev_cost = dp[prev_row + prev_l].0;
                    if !prev_cost.is_finite() {
                        continue;
                    }
                    let hi = max_take.min(total_layers - prev_l);
                    for take in 1..=hi {
                        let cand = prev_cost + node_layer_costs[cost_row + take];
                        let next_l = prev_l + take;
                        if cand < dp[curr_row + next_l].0 {
                            dp[curr_row + next_l] = (cand, take as u32);
                        }
                    }
                }

                let m = s + 1;
                if next_cap >= total_layers && total_layers >= m {
                    let compute = dp[curr_row + total_layers].0;
                    if compute.is_finite() {
                        if let Some(ret_link) = fabric.between(ni, order[0]) {
                            let cand_secs = match spec_params {
                                None => {
                                    compute + next_fwd_1 + ret_link.one_way_p99(RETURN_PAYLOAD_BYTES)
                                }
                                Some((_, ret_spec, expected_accepted, draft_bytes, draft_toks)) => {
                                    if expected_accepted > 0.0 {
                                        let draft_secs =
                                            nodes[order[0]].stream_time(draft_bytes) * draft_toks;
                                        let verify_net =
                                            next_fwd_spec + ret_link.one_way_p99(ret_spec);
                                        (draft_secs + compute + verify_net) / expected_accepted
                                    } else {
                                        f64::INFINITY
                                    }
                                }
                            };
                            if best.is_none() || cand_secs < *best_secs_per_token {
                                dense.clear();
                                dense.resize(m, (0usize, 0u32));
                                let mut rem = total_layers;
                                for stage_idx in (0..m).rev() {
                                    let take = dp[stage_idx * stride + rem].1;
                                    dense[stage_idx] = (order[stage_idx], take);
                                    rem -= take as usize;
                                }
                                if let Some(p) = build_plan(dense, nodes, model, fabric, req) {
                                    let p_secs = p.secs_per_token();
                                    if best.is_none() || p_secs < *best_secs_per_token {
                                        *best_secs_per_token = p_secs;
                                        *best = Some(p);
                                    }
                                }
                            }
                        }
                    }
                }

                if s + 1 < n && total_layers >= s + 2 {
                    search_tree(
                        s + 1,
                        next_cap,
                        next_fwd_1,
                        next_fwd_spec,
                        order,
                        dp,
                        dense,
                        best,
                        best_secs_per_token,
                        max_layers_for_node,
                        node_layer_costs,
                        stride,
                        total_layers,
                        cut_payload_1,
                        spec_params,
                        nodes,
                        model,
                        fabric,
                        req,
                    );
                }
            }

            order.swap(s, i);
        }
    }

    search_tree(
        0,
        0,
        0.0,
        0.0,
        &mut order,
        &mut dp,
        &mut dense,
        &mut best,
        &mut best_secs_per_token,
        &max_layers_for_node,
        &node_layer_costs,
        stride,
        total_layers,
        cut_payload_1,
        spec_params,
        nodes,
        model,
        fabric,
        req,
    );
    best
}

/// Cost a concrete layer assignment, or return `None` if it does not fit in
/// memory or crosses a hop with no measured link.
fn build_plan(
    assignment: &[(usize, u32)],
    nodes: &[Node],
    model: &Model,
    fabric: &Fabric,
    req: &PlanRequest,
) -> Option<Plan> {
    let mut stages = Vec::with_capacity(assignment.len());
    let mut first_layer = 0u32;
    let mut compute = 0.0;

    for &(node_index, layer_count) in assignment {
        let node = nodes.get(node_index)?;
        let weight_bytes = model.bytes_for_layers(layer_count);
        let kv_bytes = model.kv_bytes(layer_count, req.context_tokens);
        if weight_bytes + kv_bytes > node.usable_bytes {
            return None;
        }
        // Residency is charged on every byte; streaming on the active share of
        // the weights plus the KV cache, which is read in full every step.
        let compute_secs =
            node.stream_time(model.streamed_bytes_per_step(layer_count, req.context_tokens));
        compute += compute_secs;
        stages.push(Stage {
            node_index,
            node_name: node.name.clone(),
            first_layer,
            layer_count,
            weight_bytes,
            kv_bytes,
            compute_secs,
            prefill_secs: crate::cost::prefill_secs(node, model, layer_count, req.prompt_tokens),
        });
        first_layer += layer_count;
    }

    // Network cost of one forward pass carrying `batch` positions. Each hop is
    // one way: a stage sends its activations and moves on without waiting for
    // an acknowledgement.
    let pass_network = |batch: u32| -> Option<f64> {
        if stages.len() < 2 {
            return Some(0.0);
        }
        let mut total = 0.0;
        for pair in stages.windows(2) {
            let link = fabric.between(pair[0].node_index, pair[1].node_index)?;
            total += link.one_way_p99(model.cut_payload_bytes(batch));
        }
        // The sampled tokens return from the tail of the pipeline to the head.
        let tail = stages.last().unwrap().node_index;
        let head = stages[0].node_index;
        total += fabric
            .between(tail, head)?
            .one_way_p99(RETURN_PAYLOAD_BYTES * batch as u64);
        Some(total)
    };

    let network = pass_network(1)?;

    let mut prompt = 0.0;
    if stages.len() > 1 {
        for pair in stages.windows(2) {
            let link = fabric.between(pair[0].node_index, pair[1].node_index)?;
            prompt += link.one_way_p50(model.cut_payload_bytes(req.prompt_tokens));
        }
    }

    let speculation = match &req.speculation {
        None => None,
        Some(spec) => {
            // The draft model lives on the head node and never touches the
            // network. Verification is one batched forward over K+1 positions;
            // a memory-bound model reads its weights once regardless of batch
            // size, so the compute is that of a single decode step.
            let head = nodes.get(stages[0].node_index)?;
            let draft_secs_per_cycle =
                head.stream_time(spec.draft_weight_bytes) * spec.draft_tokens as f64;
            let verify_network_secs = pass_network(spec.verify_batch())?;
            let expected_accepted = spec.expected_accepted();
            let cycle = draft_secs_per_cycle + compute + verify_network_secs;
            Some(SpeculativeCost {
                draft_secs_per_cycle,
                verify_compute_secs: compute,
                verify_network_secs,
                expected_accepted,
                secs_per_token: if expected_accepted > 0.0 {
                    cycle / expected_accepted
                } else {
                    f64::INFINITY
                },
            })
        }
    };

    Some(Plan {
        stages,
        compute_secs_per_token: compute,
        network_secs_per_token: network,
        prompt_transfer_secs: prompt,
        speculation,
    })
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests;
