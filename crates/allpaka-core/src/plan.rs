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

use crate::speculation::{SpeculativeCost, Speculation};
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
    UseSingleNode { plan: Plan, best_split: Option<Plan> },
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
        Self { context_tokens: 8192, prompt_tokens: 2048, speculation: None }
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
        return Verdict::Infeasible { reason: "no nodes configured".into() };
    }
    if model.n_layers == 0 {
        return Verdict::Infeasible { reason: "model has zero layers".into() };
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
        (Some(s), Some(sp)) if sp.secs_per_token() < s.secs_per_token() => {
            Verdict::SplitWins { plan: sp, single_node: s }
        }
        (Some(s), best_split) => Verdict::UseSingleNode { plan: s, best_split },
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
fn best_split(nodes: &[Node], model: &Model, fabric: &Fabric, req: &PlanRequest) -> Option<Plan> {
    if nodes.len() < 2 {
        return None;
    }
    let mut best: Option<Plan> = None;
    let mut order: Vec<usize> = (0..nodes.len()).collect();

    permute(&mut order, 0, &mut |order| {
        let mut assignment = Vec::with_capacity(order.len());
        enumerate_splits(order, 0, model.n_layers, &mut assignment, &mut |asg| {
            let dense: Vec<(usize, u32)> = asg.iter().copied().filter(|(_, n)| *n > 0).collect();
            if dense.len() < 2 {
                return;
            }
            if let Some(p) = build_plan(&dense, nodes, model, fabric, req) {
                if best.as_ref().is_none_or(|b| p.secs_per_token() < b.secs_per_token()) {
                    best = Some(p);
                }
            }
        });
    });
    best
}

/// Heap-style permutation by successive swaps.
fn permute(order: &mut Vec<usize>, k: usize, visit: &mut impl FnMut(&[usize])) {
    if k == order.len() {
        visit(order);
        return;
    }
    for i in k..order.len() {
        order.swap(k, i);
        permute(order, k + 1, visit);
        order.swap(k, i);
    }
}

/// Walk every way of dividing `remaining` layers among `order[position..]`.
fn enumerate_splits(
    order: &[usize],
    position: usize,
    remaining: u32,
    assignment: &mut Vec<(usize, u32)>,
    visit: &mut impl FnMut(&[(usize, u32)]),
) {
    if position == order.len() {
        if remaining == 0 {
            visit(assignment);
        }
        return;
    }
    let last = position + 1 == order.len();
    let lo = if last { remaining } else { 0 };
    for take in lo..=remaining {
        assignment.push((order[position], take));
        enumerate_splits(order, position + 1, remaining - take, assignment, visit);
        assignment.pop();
    }
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
        total += fabric.between(tail, head)?.one_way_p99(RETURN_PAYLOAD_BYTES * batch as u64);
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
mod tests {
    use super::*;
    use crate::{Backend, Link};

    fn mac() -> Node {
        Node {
            name: "mac".into(),
            backend: Backend::Metal,
            usable_bytes: 96 << 30,
            mem_bandwidth_bytes_per_sec: 546e9,
            bandwidth_efficiency: 0.7,
            ..Node::default()
        }
    }

    fn pc() -> Node {
        Node {
            name: "pc".into(),
            backend: Backend::Cuda,
            usable_bytes: 30 << 30,
            mem_bandwidth_bytes_per_sec: 1792e9,
            bandwidth_efficiency: 0.7,
            ..Node::default()
        }
    }

    /// 60 GiB of weights: fits the Mac alone, does not fit the PC alone.
    fn model_60gib() -> Model {
        Model {
            name: "test-60".into(),
            n_layers: 64,
            hidden_size: 5120,
            total_weight_bytes: 60 << 30,
            kv_bytes_per_token_per_layer: 4096,
            activation_bytes: 2,
            active_weight_fraction: 1.0,
            param_count: 0,
        }
    }

    fn wifi() -> Link {
        Link { throughput_bytes_per_sec: 40e6, rtt_p50_secs: 0.004, rtt_p99_secs: 0.030 }
    }

    fn ten_gbe() -> Link {
        Link { throughput_bytes_per_sec: 1.1e9, rtt_p50_secs: 0.00015, rtt_p99_secs: 0.0004 }
    }

    #[test]
    fn single_node_has_no_network_cost() {
        let v = plan(&[mac()], &model_60gib(), &Fabric::uniform(wifi()), &PlanRequest::default());
        let Verdict::UseSingleNode { plan, .. } = v else { panic!("expected single node") };
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.network_secs_per_token, 0.0);
    }

    /// A link bad enough that the round trip outweighs the compute the split
    /// saves. Congested Wi-Fi and anything routed over a VPN land here.
    fn congested_wifi() -> Link {
        Link { throughput_bytes_per_sec: 20e6, rtt_p50_secs: 0.030, rtt_p99_secs: 0.120 }
    }

    /// The decision rule in one test: a split wins exactly when the network
    /// cost of a pass is smaller than the compute that pass saves.
    #[test]
    fn a_split_wins_when_the_link_costs_less_than_the_compute_it_saves() {
        let nodes = [mac(), pc()];
        let m = model_60gib();
        let req = PlanRequest::default();

        let Verdict::SplitWins { plan, single_node } =
            plan(&nodes, &m, &Fabric::uniform(ten_gbe()), &req)
        else {
            panic!("10 GbE should win")
        };
        let saving = single_node.compute_secs_per_token - plan.compute_secs_per_token;
        assert!(plan.network_secs_per_token < saving);
    }

    #[test]
    fn a_high_latency_link_makes_the_split_lose() {
        let v = plan(
            &[mac(), pc()],
            &model_60gib(),
            &Fabric::uniform(congested_wifi()),
            &PlanRequest::default(),
        );
        let Verdict::UseSingleNode { plan, best_split } = v else {
            panic!("a 120 ms tail should lose, got {v:?}")
        };
        let split = best_split.expect("still feasible, just slower");
        assert!(split.secs_per_token() > plan.secs_per_token());
    }

    /// Same hardware, same model, only the cable changed.
    #[test]
    fn the_link_alone_flips_the_verdict() {
        let nodes = [mac(), pc()];
        let m = model_60gib();
        let req = PlanRequest::default();

        let slow = plan(&nodes, &m, &Fabric::uniform(congested_wifi()), &req);
        assert!(matches!(slow, Verdict::UseSingleNode { .. }), "got {slow:?}");

        let fast = plan(&nodes, &m, &Fabric::uniform(ten_gbe()), &req);
        let Verdict::SplitWins { plan, single_node } = fast else { panic!("got {fast:?}") };
        assert!(plan.secs_per_token() < single_node.secs_per_token());
        assert!(
            plan.network_overhead_fraction() < 0.05,
            "network should be negligible, was {:.3}",
            plan.network_overhead_fraction()
        );
    }

    /// Decode payloads are ~10 KB, so a fast link wins on latency, not
    /// bandwidth. A high-throughput but laggy path stays a bad path.
    #[test]
    fn throughput_without_low_latency_does_not_help_decode() {
        let laggy =
            Link { throughput_bytes_per_sec: 1.1e9, rtt_p50_secs: 0.100, rtt_p99_secs: 0.150 };
        let v = plan(
            &[mac(), pc()],
            &model_60gib(),
            &Fabric::uniform(laggy),
            &PlanRequest::default(),
        );
        assert!(
            matches!(v, Verdict::UseSingleNode { .. }),
            "a 10 Gb link with a 150 ms tail should not win, got {v:?}"
        );
    }

    #[test]
    fn the_planner_avoids_the_slow_hop_in_a_mixed_fabric() {
        let mut laptop = mac();
        laptop.name = "laptop".into();
        laptop.usable_bytes = 20 << 30;
        let nodes = [mac(), pc(), laptop];

        // mac<->pc is 10 GbE; anything involving the laptop is Wi-Fi.
        let fabric = Fabric::new().connect(0, 1, ten_gbe()).with_fallback(wifi());

        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30; // forces a split
        let v = plan(&nodes, &m, &fabric, &PlanRequest::default());
        let (Verdict::SplitRequired { plan } | Verdict::SplitWins { plan, .. }) = v else {
            panic!("expected a split, got {v:?}")
        };
        let used: Vec<usize> = plan.stages.iter().map(|s| s.node_index).collect();
        assert!(!used.contains(&2), "the Wi-Fi-attached laptop should be left out: {used:?}");
    }

    #[test]
    fn an_unmeasured_hop_blocks_a_split_rather_than_being_guessed() {
        let nodes = [mac(), pc()];
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        // No edges, no fallback: nothing is known about the network.
        let v = plan(&nodes, &m, &Fabric::new(), &PlanRequest::default());
        assert!(matches!(v, Verdict::Infeasible { .. }), "got {v:?}");
    }

    #[test]
    fn split_is_required_when_nothing_holds_the_whole_model() {
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        let v = plan(&[mac(), pc()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::SplitRequired { plan } = v else { panic!("expected a forced split, got {v:?}") };
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn oversized_model_is_reported_infeasible() {
        let mut m = model_60gib();
        m.total_weight_bytes = 400 << 30;
        let v = plan(&[mac(), pc()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        assert!(matches!(v, Verdict::Infeasible { .. }));
    }

    #[test]
    fn layers_are_conserved_and_contiguous_across_stages() {
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        let v = plan(&[mac(), pc()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::SplitRequired { plan } = v else { panic!() };
        let mut next = 0;
        for s in &plan.stages {
            assert_eq!(s.first_layer, next);
            next += s.layer_count;
        }
        assert_eq!(next, m.n_layers);
    }

    #[test]
    fn every_stage_respects_its_memory_budget() {
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        let nodes = [mac(), pc()];
        let v = plan(&nodes, &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::SplitRequired { plan } = v else { panic!() };
        for s in &plan.stages {
            assert!(s.resident_bytes() <= nodes[s.node_index].usable_bytes);
        }
    }

    fn draft(k: u32, a: f64) -> Speculation {
        // A ~2 GiB draft against a 60 GiB target: roughly a thirtieth the size,
        // so a draft step is roughly a thirtieth of a verify step.
        Speculation { draft_weight_bytes: 2 << 30, draft_tokens: k, acceptance_rate: a }
    }

    /// A fire-and-forget hop costs one-way latency, and a two-stage pipeline
    /// has exactly two of them: forward and return. Charging a full round trip
    /// per hop would double this.
    #[test]
    fn a_two_stage_pipeline_costs_one_round_trip_per_token() {
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        let link = ten_gbe();
        let v = plan(&[mac(), pc()], &m, &Fabric::uniform(link.clone()), &PlanRequest::default());
        let Verdict::SplitRequired { plan } = v else { panic!() };
        // Two one-way hops, plus serialisation of ~10 KB each way.
        assert!(plan.network_secs_per_token < link.rtt_p99_secs * 1.2);
        assert!(plan.network_secs_per_token > link.rtt_p99_secs * 0.8);
    }

    /// Pull the split out of whatever verdict came back, for tests that care
    /// about the split's cost rather than about who won.
    fn split_of(v: &Verdict) -> Plan {
        match v {
            Verdict::SplitWins { plan, .. } | Verdict::SplitRequired { plan } => plan.clone(),
            Verdict::UseSingleNode { best_split: Some(p), .. } => p.clone(),
            other => panic!("no split in {other:?}"),
        }
    }

    /// The point of speculation: one round trip carries several tokens, so the
    /// network cost per token falls even though the pass costs the same.
    #[test]
    fn speculation_amortises_the_round_trip_over_several_tokens() {
        let nodes = [mac(), pc()];
        let m = model_60gib();
        let fabric = Fabric::uniform(congested_wifi());

        let plain = split_of(&plan(&nodes, &m, &fabric, &PlanRequest::default()));
        let spec_req = PlanRequest { speculation: Some(draft(4, 0.7)), ..Default::default() };
        let spec = split_of(&plan(&nodes, &m, &fabric, &spec_req));

        assert!(
            spec.network_secs_per_token_effective()
                < plain.network_secs_per_token_effective() / 2.0,
            "speculation should more than halve per-token network wait: {:.5} vs {:.5}",
            spec.network_secs_per_token_effective(),
            plain.network_secs_per_token_effective(),
        );
    }

    /// Speculation does NOT change which placement wins.
    ///
    /// A verification cycle divides both the compute and the network by the
    /// same expected-accepted factor, so the comparison
    /// `network_per_pass < compute_saved_per_pass` is left untouched. It is a
    /// throughput multiplier, not a fix for a bad link.
    #[test]
    fn speculation_does_not_change_the_split_verdict() {
        let nodes = [mac(), pc()];
        let m = model_60gib();

        for fabric in [Fabric::uniform(ten_gbe()), Fabric::uniform(congested_wifi())] {
            let plain = plan(&nodes, &m, &fabric, &PlanRequest::default());
            let spec_req =
                PlanRequest { speculation: Some(draft(4, 0.7)), ..Default::default() };
            let spec = plan(&nodes, &m, &fabric, &spec_req);
            assert_eq!(
                std::mem::discriminant(&plain),
                std::mem::discriminant(&spec),
                "speculation flipped the verdict: {plain:?} became {spec:?}"
            );
        }
    }

    /// What speculation actually buys: more tokens per second, everywhere.
    #[test]
    fn speculation_raises_throughput() {
        let nodes = [mac(), pc()];
        let m = model_60gib();
        let fabric = Fabric::uniform(ten_gbe());

        let Verdict::SplitWins { plan: plain, .. } =
            plan(&nodes, &m, &fabric, &PlanRequest::default())
        else {
            panic!()
        };
        let req = PlanRequest { speculation: Some(draft(4, 0.8)), ..Default::default() };
        let Verdict::SplitWins { plan: spec, .. } = plan(&nodes, &m, &fabric, &req) else {
            panic!()
        };
        assert!(
            spec.tokens_per_sec() > plain.tokens_per_sec() * 1.5,
            "{:.1} vs {:.1} tok/s",
            spec.tokens_per_sec(),
            plain.tokens_per_sec()
        );
    }

    /// A draft nobody accepts is pure overhead. The planner must not present
    /// speculation as free.
    #[test]
    fn a_useless_draft_makes_things_worse_not_better() {
        let nodes = [mac()];
        let m = model_60gib();
        let fabric = Fabric::uniform(ten_gbe());

        let plain = plan(&nodes, &m, &fabric, &PlanRequest::default());
        let Verdict::UseSingleNode { plan: plain, .. } = plain else { panic!() };

        let req = PlanRequest { speculation: Some(draft(8, 0.0)), ..Default::default() };
        let Verdict::UseSingleNode { plan: spec, .. } = plan(&nodes, &m, &fabric, &req) else {
            panic!()
        };
        assert!(
            spec.secs_per_token() > plain.secs_per_token(),
            "a draft with zero acceptance must cost time, not save it"
        );
    }

    /// System RAM on the PC: large, and dozens of times slower than the GPU
    /// sitting beside it. Dual-channel DDR4-3200.
    fn pc_ram() -> Node {
        Node {
            name: "pc-ram".into(),
            backend: Backend::Cpu,
            usable_bytes: 56 << 30,
            mem_bandwidth_bytes_per_sec: 51.2e9,
            bandwidth_efficiency: 0.6,
            ..Node::default()
        }
    }

    /// The PCIe bus inside one box. Microseconds, not milliseconds.
    fn pcie() -> Link {
        Link { throughput_bytes_per_sec: 50e9, rtt_p50_secs: 0.000012, rtt_p99_secs: 0.00003 }
    }

    /// A slow tier must never be used while a faster one has room. Capacity is
    /// what it is for.
    #[test]
    fn slow_memory_is_left_empty_while_fast_memory_has_room() {
        let nodes = [mac(), pc(), pc_ram()];
        let fabric = Fabric::new()
            .connect(1, 2, pcie())
            .connect(0, 1, ten_gbe())
            .connect(0, 2, ten_gbe());

        // 60 GiB fits comfortably in mac + GPU, with no need for system RAM.
        let v = plan(&nodes, &model_60gib(), &fabric, &PlanRequest::default());
        let (Verdict::SplitWins { plan, .. } | Verdict::UseSingleNode { plan, .. }) = v else {
            panic!("got {v:?}")
        };
        assert!(
            !plan.stages.iter().any(|s| s.node_name == "pc-ram"),
            "system RAM should stay empty here: {:?}",
            plan.stages.iter().map(|s| &s.node_name).collect::<Vec<_>>()
        );
    }

    /// But it is exactly what makes an otherwise impossible model runnable.
    #[test]
    fn slow_memory_turns_an_infeasible_model_into_a_feasible_one() {
        let fabric = Fabric::new()
            .connect(1, 2, pcie())
            .connect(0, 1, ten_gbe())
            .connect(0, 2, ten_gbe());
        let mut m = model_60gib();
        m.total_weight_bytes = 170 << 30; // over mac + GPU, under mac + GPU + RAM

        let without = plan(&[mac(), pc()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        assert!(matches!(without, Verdict::Infeasible { .. }), "got {without:?}");

        let with = plan(&[mac(), pc(), pc_ram()], &m, &fabric, &PlanRequest::default());
        let Verdict::SplitRequired { plan } = with else { panic!("got {with:?}") };
        assert!(plan.stages.iter().any(|s| s.node_name == "pc-ram"));
    }

    /// Adding capacity is not the same as adding speed. Layers that land in
    /// system RAM cost roughly eighteen times what they would on the GPU.
    #[test]
    fn layers_in_system_ram_dominate_the_step_time() {
        let fabric = Fabric::new()
            .connect(1, 2, pcie())
            .connect(0, 1, ten_gbe())
            .connect(0, 2, ten_gbe());
        let mut m = model_60gib();
        m.total_weight_bytes = 170 << 30;

        let Verdict::SplitRequired { plan } =
            plan(&[mac(), pc(), pc_ram()], &m, &fabric, &PlanRequest::default())
        else {
            panic!()
        };
        let ram_stage = plan.stages.iter().find(|s| s.node_name == "pc-ram").unwrap();
        assert!(
            ram_stage.compute_secs > plan.compute_secs_per_token * 0.5,
            "the RAM stage should dominate: {:.1} of {:.1} ms",
            ram_stage.compute_secs * 1e3,
            plan.compute_secs_per_token * 1e3,
        );
    }

    /// An in-box hop is orders of magnitude cheaper than any cable, so the
    /// planner should not hesitate to cut between VRAM and system RAM.
    #[test]
    fn the_pcie_hop_costs_almost_nothing() {
        let fabric = Fabric::new()
            .connect(0, 1, pcie())
            .with_fallback(ten_gbe());
        let mut m = model_60gib();
        m.total_weight_bytes = 70 << 30;
        let Verdict::SplitRequired { plan } =
            plan(&[pc(), pc_ram()], &m, &fabric, &PlanRequest::default())
        else {
            panic!()
        };
        assert!(
            plan.network_overhead_fraction() < 0.001,
            "in-box hop should be free, was {:.5}",
            plan.network_overhead_fraction()
        );
    }

    /// A mixture of experts pays full price for memory and a fraction of the
    /// price for bandwidth. That asymmetry is the whole reason a machine with
    /// a lot of memory can punch above its bandwidth.
    #[test]
    fn a_sparse_model_costs_full_memory_but_a_fraction_of_the_time() {
        let dense = model_60gib();
        let mut sparse = model_60gib();
        sparse.active_weight_fraction = 0.1;

        let fabric = Fabric::uniform(ten_gbe());
        let req = PlanRequest::default();

        let Verdict::UseSingleNode { plan: d, .. } = plan(&[mac()], &dense, &fabric, &req) else {
            panic!()
        };
        let Verdict::UseSingleNode { plan: sp, .. } = plan(&[mac()], &sparse, &fabric, &req) else {
            panic!()
        };

        // Same residency ...
        assert_eq!(d.stages[0].weight_bytes, sp.stages[0].weight_bytes);
        // ... and only the weight share of the time shrinks. The KV cache is
        // dense - no router skips it - so the ratio floors above the raw 0.1.
        let expected = sparse.streamed_bytes_per_step(sparse.n_layers, req.context_tokens) as f64
            / dense.streamed_bytes_per_step(dense.n_layers, req.context_tokens) as f64;
        assert!(expected > 0.1, "the KV term should keep this above the weight fraction");
        let ratio = sp.compute_secs_per_token / d.compute_secs_per_token;
        assert!((ratio - expected).abs() < 1e-9, "{ratio} vs {expected}");
    }

    /// Sparsity must not let a model into memory it does not fit in.
    #[test]
    fn sparsity_does_not_relax_the_memory_limit() {
        let mut sparse = model_60gib();
        sparse.total_weight_bytes = 400 << 30;
        sparse.active_weight_fraction = 0.05;
        let v = plan(&[mac(), pc()], &sparse, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        assert!(
            matches!(v, Verdict::Infeasible { .. }),
            "a 400 GiB MoE still needs 400 GiB of memory, got {v:?}"
        );
    }

    /// The practical consequence: a big MoE spread over slow memory can beat a
    /// smaller dense model that fits the fast memory, because it barely touches
    /// the slow bytes.
    #[test]
    fn a_sparse_model_tolerates_slow_memory_far_better_than_a_dense_one() {
        let fabric = Fabric::new().connect(0, 1, pcie()).with_fallback(ten_gbe());
        let mut dense = model_60gib();
        dense.total_weight_bytes = 70 << 30;
        let mut sparse = dense.clone();
        sparse.active_weight_fraction = 0.1;

        let nodes = [pc(), pc_ram()];
        let Verdict::SplitRequired { plan: d } =
            plan(&nodes, &dense, &fabric, &PlanRequest::default())
        else {
            panic!()
        };
        let Verdict::SplitRequired { plan: sp } =
            plan(&nodes, &sparse, &fabric, &PlanRequest::default())
        else {
            panic!()
        };
        assert!(
            sp.tokens_per_sec() > d.tokens_per_sec() * 5.0,
            "sparse {:.1} vs dense {:.1} tok/s",
            sp.tokens_per_sec(),
            d.tokens_per_sec()
        );
    }

    /// An unmodeled prefill must surface as unknown, never as a number.
    #[test]
    fn ttft_is_unknown_until_flops_and_params_are_both_known() {
        let m = model_60gib(); // param_count: 0
        let v = plan(&[mac()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::UseSingleNode { plan: p, .. } = v else { panic!() };
        assert!(p.ttft_secs().is_none());

        let mut node = mac();
        node.prefill_flops = 16e12; // flops alone is still not enough
        let v = plan(&[node], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::UseSingleNode { plan: p, .. } = v else { panic!() };
        assert!(p.ttft_secs().is_none(), "param count is still unknown");
    }

    /// The number prefill modeling exists to expose: layers spilled to a CPU
    /// pool decode tolerably but push the first token out by tens of seconds.
    #[test]
    fn spilling_layers_to_a_cpu_pool_wrecks_time_to_first_token() {
        let mut gpu = pc();
        gpu.prefill_flops = 150e12;
        let mut cpu = pc_ram();
        cpu.prefill_flops = 0.4e12;
        // A MoE: decode reads 11% of the weights, prefill computes 11% of the
        // params - but that is still billions of FLOPs per prompt token.
        let mut m = model_60gib();
        m.param_count = 60_000_000_000; // does not fit the GPU alone
        m.active_weight_fraction = 0.11;

        let v = plan(&[gpu, cpu], &m, &Fabric::uniform(pcie()), &PlanRequest::default());
        let Verdict::SplitRequired { plan: p } = v else { panic!("expected a forced split") };
        let ttft = p.ttft_secs().expect("both pools have flops set");

        // The CPU stage holds only half the layers, yet its prefill dwarfs
        // everything else in the plan.
        let cpu_stage = p.stages.iter().find(|s| s.node_name == "pc-ram").unwrap();
        assert!(cpu_stage.prefill_secs.unwrap() > 30.0);
        assert!(ttft > 30.0, "got {ttft}");
        // Decode alone would never reveal this: it stays interactive.
        assert!(p.tokens_per_sec() > 3.0);
    }

    /// MoE sparsity thins prefill the same way it thins decode streaming.
    #[test]
    fn a_sparse_model_prefills_at_its_active_fraction() {
        let mut node = mac();
        node.prefill_flops = 16e12;
        let mut dense = model_60gib();
        dense.param_count = 60_000_000_000;
        let mut sparse = dense.clone();
        sparse.active_weight_fraction = 0.1;

        let req = PlanRequest::default();
        let fabric = Fabric::uniform(ten_gbe());
        let Verdict::UseSingleNode { plan: d, .. } = plan(&[node.clone()], &dense, &fabric, &req)
        else {
            panic!()
        };
        let Verdict::UseSingleNode { plan: sp, .. } = plan(&[node], &sparse, &fabric, &req)
        else {
            panic!()
        };
        let ratio = sp.ttft_secs().unwrap() / d.ttft_secs().unwrap();
        assert!((ratio - 0.1).abs() < 1e-9, "got {ratio}");
    }

    #[test]
    fn each_node_appears_at_most_once_in_a_plan() {
        let mut m = model_60gib();
        m.total_weight_bytes = 100 << 30;
        let v = plan(&[mac(), pc()], &m, &Fabric::uniform(ten_gbe()), &PlanRequest::default());
        let Verdict::SplitRequired { plan } = v else { panic!() };
        let mut seen: Vec<usize> = plan.stages.iter().map(|s| s.node_index).collect();
        seen.sort_unstable();
        let len = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), len);
    }
}
