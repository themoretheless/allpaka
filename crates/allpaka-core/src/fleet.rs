//! Fleet placement: several different models, one per memory pool.
//!
//! This is the second of the two arrangements the tool supports, and it is not
//! the same thing as [`crate::replicate`]. Replication puts the *same* model on
//! every machine that can hold it. A fleet puts a *different* model on each
//! pool, chosen to suit what that pool is good at:
//!
//! * a small GPU is the fastest thing per byte, so it should run whatever is
//!   called most often and waited on hardest - routing, tool calls, drafts;
//! * a large unified-memory pool holds the biggest model, which is the one
//!   worth waiting for;
//! * slow system RAM still runs something useful when nobody is waiting.
//!
//! Each agent then talks to its own endpoint and the machines never talk to
//! each other, which is the same zero-coupling property replication has, minus
//! the requirement that one model fit everywhere.
//!
//! # The objective
//!
//! Placements are scored lexicographically: first by how many models were
//! placed at all, then by aggregate tokens per second. Placing every model
//! comes first because an unplaced model is a missing capability, not a slow
//! one - no amount of throughput elsewhere replaces the agent that cannot run.
//!
//! # What this deliberately does not do
//!
//! At most one model per pool. Two servers can physically share a GPU, but
//! then they share its memory and its throughput, and the isolation that makes
//! this arrangement worth choosing starts to leak. Keeping it one-to-one also
//! keeps the search a plain assignment problem rather than bin packing.
//!
//! # Pools are not machines
//!
//! One pool per agent does not mean one machine per agent. A desktop with a
//! discrete GPU offers two pools, and two agents placed on them are two
//! processes on one computer: they share its CPU, its PCIe root, its power
//! budget and its uptime. Rebooting that machine takes both agents down, and no
//! amount of pool-level isolation changes that.
//!
//! What it does *not* cost, in steady-state decode, is bandwidth: the CUDA
//! server streams weights out of VRAM and the CPU server out of DDR, and
//! neither is reading the other's memory. That is why [`Node::contention`]
//! defaults to 1.0. Prompt processing is the case where it stops being true,
//! because the GPU pulls the prompt across PCIe out of the same DDR. Measure
//! it, set the number, and the placements below are rescored accordingly.

use crate::speculation::Speculation;
use crate::{Model, Node};

/// One agent's requirements: which model, how much context it needs, and
/// whether it is pinned to a particular pool.
///
/// Context is per member rather than shared across the fleet, because agents
/// differ wildly in what they need: a tool-caller works in a few thousand
/// tokens while a reasoner wants tens of thousands, and reserving the
/// reasoner's KV cache for everyone wastes memory that could have held
/// weights.
#[derive(Debug, Clone)]
pub struct FleetMember {
    pub model: Model,
    pub context_tokens: u32,
    /// Typical prompt length for this agent, used to estimate its time to
    /// first token. Per member for the same reason context is: a tool-caller
    /// re-reads a short scaffold while a reasoner ingests documents.
    pub prompt_tokens: u32,
    /// Pool this agent must run on, by node index. `None` lets the planner
    /// choose.
    pub pin: Option<usize>,
    /// Draft model for speculative decoding, which shares this agent's pool.
    pub speculation: Option<Speculation>,
}

/// One model placed on one pool.
#[derive(Debug, Clone)]
pub struct Placement {
    pub model_index: usize,
    pub model_name: String,
    pub node_index: usize,
    pub node_name: String,
    pub secs_per_token: f64,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub context_tokens: u32,
    /// Whether this placement was forced by the config rather than chosen.
    pub pinned: bool,
    /// Time to first token at this agent's prompt length, or `None` when the
    /// pool's FLOP/s or the model's parameter count is unknown.
    pub ttft_secs: Option<f64>,
    /// Physical machine hosting the pool.
    pub host: String,
    /// Whether another agent in this plan runs on the same machine. When true,
    /// `secs_per_token` already has that pool's `contention` factor applied.
    pub co_resident: bool,
}

impl Placement {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.secs_per_token <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.secs_per_token
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.weight_bytes + self.kv_bytes
    }
}

#[derive(Debug, Clone, Default)]
pub struct FleetPlan {
    pub placements: Vec<Placement>,
    /// Models that fit nowhere left over. Reported rather than dropped.
    pub unplaced: Vec<usize>,
}

impl FleetPlan {
    pub fn aggregate_tokens_per_sec(&self) -> f64 {
        self.placements.iter().map(Placement::tokens_per_sec).sum()
    }

    /// Independent endpoints, which is also the number of requests that can be
    /// in flight without any of them queueing behind another.
    pub fn endpoints(&self) -> usize {
        self.placements.len()
    }

    /// Distinct physical machines the endpoints are spread over.
    ///
    /// Reported separately from [`FleetPlan::endpoints`] because the two are
    /// not the same number and the difference is the one that matters when
    /// something goes down: endpoints are how many requests run at once,
    /// machines are how many independent failure domains there are.
    pub fn machines(&self) -> usize {
        let mut hosts: Vec<&str> = self.placements.iter().map(|p| p.host.as_str()).collect();
        hosts.sort_unstable();
        hosts.dedup();
        hosts.len()
    }

    /// Agents sharing a machine with another agent.
    pub fn co_resident(&self) -> impl Iterator<Item = &Placement> {
        self.placements.iter().filter(|p| p.co_resident)
    }

    fn better_than(&self, other: &FleetPlan) -> bool {
        match self.placements.len().cmp(&other.placements.len()) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => {
                self.aggregate_tokens_per_sec() > other.aggregate_tokens_per_sec()
            }
        }
    }
}

/// Beyond this the exhaustive search stops being instant. Real fleets are far
/// smaller; the cap turns a pathological config into an error rather than a
/// hang.
const MAX_SEARCH: usize = 8;

/// Assign each agent to at most one pool, maximising coverage then throughput.
pub fn fleet(members: &[FleetMember], nodes: &[Node]) -> Result<FleetPlan, String> {
    if members.is_empty() {
        return Err("no agents given".into());
    }
    if nodes.is_empty() {
        return Err("no nodes configured".into());
    }
    if members.len() > MAX_SEARCH || nodes.len() > MAX_SEARCH {
        return Err(format!(
            "fleet placement searches every assignment and is capped at {MAX_SEARCH} agents \
             and {MAX_SEARCH} pools"
        ));
    }
    for (i, m) in members.iter().enumerate() {
        if let Some(pin) = m.pin {
            if pin >= nodes.len() {
                return Err(format!(
                    "agent {:?} is pinned to node index {pin}, which does not exist",
                    m.model.name
                ));
            }
            if place(i, m, pin, &nodes[pin]).is_none() {
                return Err(format!(
                    "agent {:?} is pinned to {:?}, but does not fit there: needs {:.1} GiB \
                     of {:.1} GiB usable",
                    m.model.name,
                    nodes[pin].name,
                    crate::gib(required_bytes(m)),
                    crate::gib(nodes[pin].usable_bytes),
                ));
            }
        }
    }

    // Precompute which (agent, pool) pairs are even possible. A pinned agent
    // has exactly one candidate.
    let mut options: Vec<Vec<Option<Placement>>> = Vec::with_capacity(members.len());
    for (mi, member) in members.iter().enumerate() {
        let mut row = Vec::with_capacity(nodes.len());
        for (ni, node) in nodes.iter().enumerate() {
            let allowed = member.pin.is_none_or(|p| p == ni);
            row.push(if allowed { place(mi, member, ni, node) } else { None });
        }
        options.push(row);
    }

    let mut best = FleetPlan::default();
    let mut current = Vec::new();
    search(&options, nodes, 0, &mut 0u32, &mut current, &mut best);

    let placed: Vec<usize> = best.placements.iter().map(|p| p.model_index).collect();
    best.unplaced = (0..members.len()).filter(|i| !placed.contains(i)).collect();
    Ok(best)
}

/// Memory an agent needs on whichever pool hosts it.
pub fn required_bytes(m: &FleetMember) -> u64 {
    let mut need = m.model.total_weight_bytes
        + m.model.kv_bytes(m.model.n_layers, m.context_tokens);
    if let Some(s) = &m.speculation {
        need += s.draft_weight_bytes;
    }
    need
}

/// Depth-first over models, tracking which pools are taken in a bitmask.
fn search(
    options: &[Vec<Option<Placement>>],
    nodes: &[Node],
    model_index: usize,
    used_nodes: &mut u32,
    current: &mut Vec<Placement>,
    best: &mut FleetPlan,
) {
    if model_index == options.len() {
        let mut placements = current.clone();
        // Co-residency is a property of the whole assignment, not of one pair,
        // so it can only be charged once the candidate is complete.
        charge_co_residency(&mut placements, nodes);
        let candidate = FleetPlan { placements, unplaced: Vec::new() };
        if candidate.better_than(best) {
            *best = candidate;
        }
        return;
    }

    // Leaving a model unplaced is a legal branch: sometimes there is simply no
    // pool left that holds it.
    search(options, nodes, model_index + 1, used_nodes, current, best);

    for (node_index, option) in options[model_index].iter().enumerate() {
        let Some(placement) = option else { continue };
        let bit = 1u32 << node_index;
        if *used_nodes & bit != 0 {
            continue;
        }
        *used_nodes |= bit;
        current.push(placement.clone());
        search(options, nodes, model_index + 1, used_nodes, current, best);
        current.pop();
        *used_nodes &= !bit;
    }
}

/// Mark agents that share a machine and slow them by that pool's measured
/// contention.
///
/// Scaling `secs_per_token` is exact rather than an approximation: contention
/// scales the pool's bandwidth, every term in the cost is bytes over that
/// bandwidth, and the acceptance rate of a draft model does not depend on how
/// fast it ran. So one division reproduces what recomputing from
/// [`Node::contended_stream_time`] would give.
fn charge_co_residency(placements: &mut [Placement], nodes: &[Node]) {
    let hosts: Vec<&str> = placements.iter().map(|p| p.host.as_str()).collect();
    let indices: Vec<usize> = placements.iter().map(|p| p.node_index).collect();
    let scales = crate::cost::co_residency_scale(&hosts, &indices, nodes);
    for (p, scale) in placements.iter_mut().zip(scales) {
        if let Some(s) = scale {
            p.co_resident = true;
            p.secs_per_token = crate::cost::contended(p.secs_per_token, s);
        }
    }
}

/// Cost one agent on one pool, or `None` if it does not fit.
fn place(
    model_index: usize,
    member: &FleetMember,
    node_index: usize,
    node: &Node,
) -> Option<Placement> {
    if required_bytes(member) > node.usable_bytes {
        return None;
    }
    let model = &member.model;
    let kv_bytes = model.kv_bytes(model.n_layers, member.context_tokens);

    // No network term: a fleet member runs entirely within its own pool.
    let secs_per_token = crate::cost::local_secs_per_token(
        node,
        model,
        member.context_tokens,
        member.speculation.as_ref(),
    );

    Some(Placement {
        model_index,
        model_name: model.name.clone(),
        node_index,
        node_name: node.name.clone(),
        secs_per_token,
        weight_bytes: model.total_weight_bytes,
        kv_bytes,
        context_tokens: member.context_tokens,
        pinned: member.pin.is_some(),
        ttft_secs: crate::cost::prefill_secs(node, model, model.n_layers, member.prompt_tokens),
        host: node.host().to_string(),
        co_resident: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

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

    // The GPU and system RAM below are two pools of one desktop, which is why
    // they name the same host.
    fn gpu() -> Node {
        Node {
            name: "pc-gpu".into(),
            backend: Backend::Cuda,
            usable_bytes: 30 << 30,
            mem_bandwidth_bytes_per_sec: 1792e9,
            bandwidth_efficiency: 0.7,
            host: Some("pc".into()),
            ..Node::default()
        }
    }

    fn ram() -> Node {
        Node {
            name: "pc-ram".into(),
            backend: Backend::Cpu,
            usable_bytes: 56 << 30,
            mem_bandwidth_bytes_per_sec: 51.2e9,
            bandwidth_efficiency: 0.6,
            host: Some("pc".into()),
            ..Node::default()
        }
    }

    fn model(name: &str, gib: u64) -> Model {
        Model {
            name: name.into(),
            n_layers: 64,
            hidden_size: 5120,
            total_weight_bytes: gib << 30,
            kv_bytes_per_token_per_layer: 1024,
            activation_bytes: 2,
            active_weight_fraction: 1.0,
            param_count: 0,
        }
    }

    fn member(name: &str, gib: u64) -> FleetMember {
        FleetMember {
            model: model(name, gib),
            context_tokens: 8192,
            prompt_tokens: 2048,
            pin: None,
            speculation: None,
        }
    }

    /// The headline behaviour: a model that fits only one pool goes there, and
    /// the rest arrange themselves around it.
    #[test]
    fn a_constrained_model_takes_the_only_pool_that_holds_it() {
        let plan = fleet(&[member("big", 80), member("small", 10)], &[mac(), gpu(), ram()])
            .unwrap();
        assert_eq!(plan.endpoints(), 2);
        let big = plan.placements.iter().find(|p| p.model_name == "big").unwrap();
        assert_eq!(big.node_name, "mac");
    }

    /// With the big model on the Mac, the small one should land on the GPU
    /// rather than on system RAM, because that is where it is fastest.
    #[test]
    fn a_small_model_is_given_the_fastest_free_pool() {
        let plan = fleet(&[member("big", 80), member("small", 10)], &[mac(), gpu(), ram()])
            .unwrap();
        let small = plan.placements.iter().find(|p| p.model_name == "small").unwrap();
        assert_eq!(small.node_name, "pc-gpu");
    }

    /// Coverage beats speed: three agents on three pools is better than two
    /// agents placed slightly faster.
    #[test]
    fn placing_every_model_wins_over_placing_fewer_of_them_faster() {
        let plan =
            fleet(&[member("a", 10), member("b", 10), member("c", 10)], &[mac(), gpu(), ram()])
                .unwrap();
        assert_eq!(plan.endpoints(), 3, "all three should be placed");
        assert!(plan.unplaced.is_empty());
    }

    #[test]
    fn one_pool_holds_at_most_one_model() {
        let members =
            [member("a", 10), member("b", 10), member("c", 10), member("d", 10)];
        let plan = fleet(&members, &[mac(), gpu()]).unwrap();
        assert_eq!(plan.endpoints(), 2);
        assert_eq!(plan.unplaced.len(), 2);
        let mut nodes: Vec<usize> = plan.placements.iter().map(|p| p.node_index).collect();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes.len(), 2, "a pool must not host two models");
    }

    /// An unplaceable model is named, not silently dropped.
    #[test]
    fn a_model_too_large_for_every_pool_is_reported_as_unplaced() {
        let plan = fleet(&[member("huge", 200), member("small", 10)], &[mac(), gpu()]).unwrap();
        assert_eq!(plan.unplaced, vec![0]);
        assert_eq!(plan.endpoints(), 1);
    }

    /// A fleet member never crosses a wire, so throughput is a clean sum.
    #[test]
    fn aggregate_throughput_is_the_sum_of_independent_endpoints() {
        let plan = fleet(&[member("a", 10), member("b", 10)], &[mac(), gpu()]).unwrap();
        let sum: f64 = plan.placements.iter().map(|p| p.tokens_per_sec()).sum();
        assert!((plan.aggregate_tokens_per_sec() - sum).abs() < 1e-9);
    }

    /// Sparsity helps here exactly as it does for a split: residency is charged
    /// in full, streaming only on the active share.
    #[test]
    fn a_sparse_fleet_member_is_faster_without_needing_less_memory() {
        let mut sparse = member("moe", 80);
        sparse.model.active_weight_fraction = 0.11;
        let dense = member("dense", 80);

        let sp = fleet(&[sparse], &[mac()]).unwrap();
        let de = fleet(&[dense], &[mac()]).unwrap();
        assert_eq!(sp.placements[0].weight_bytes, de.placements[0].weight_bytes);
        assert!(sp.aggregate_tokens_per_sec() > de.aggregate_tokens_per_sec() * 5.0);
    }

    /// Context is per agent, so a tool-caller does not have to reserve the
    /// reasoner's KV cache.
    #[test]
    fn each_agent_reserves_only_the_context_it_asked_for() {
        let mut short = member("tools", 10);
        short.context_tokens = 4096;
        let mut long = member("reasoner", 10);
        long.context_tokens = 65536;

        let plan = fleet(&[short, long], &[mac(), gpu()]).unwrap();
        let t = plan.placements.iter().find(|p| p.model_name == "tools").unwrap();
        let r = plan.placements.iter().find(|p| p.model_name == "reasoner").unwrap();
        assert_eq!(t.context_tokens, 4096);
        assert!(r.kv_bytes > t.kv_bytes * 8, "{} vs {}", r.kv_bytes, t.kv_bytes);
    }

    /// A pin overrides the optimiser, even when it costs throughput.
    #[test]
    fn a_pinned_agent_stays_where_it_was_put() {
        let mut small = member("small", 10);
        small.pin = Some(2); // system RAM, far from optimal
        let plan = fleet(&[small, member("other", 10)], &[mac(), gpu(), ram()]).unwrap();
        let s = plan.placements.iter().find(|p| p.model_name == "small").unwrap();
        assert_eq!(s.node_name, "pc-ram");
        assert!(s.pinned);
    }

    /// Pinning something where it cannot fit is a config error, reported by
    /// name rather than silently ignored or quietly relocated.
    #[test]
    fn pinning_an_agent_somewhere_it_does_not_fit_is_an_error() {
        let mut big = member("big", 80);
        big.pin = Some(1); // the 30 GiB GPU
        let err = fleet(&[big], &[mac(), gpu(), ram()]).unwrap_err();
        assert!(err.contains("big"), "unhelpful error: {err}");
        assert!(err.contains("pc-gpu"), "unhelpful error: {err}");
    }

    #[test]
    fn pinning_to_a_node_that_does_not_exist_is_an_error() {
        let mut m = member("a", 10);
        m.pin = Some(9);
        assert!(fleet(&[m], &[mac()]).is_err());
    }

    /// Two agents pinned to the same pool cannot both run; one is reported
    /// unplaced rather than both being squeezed in.
    #[test]
    fn two_agents_pinned_to_one_pool_leave_one_unplaced() {
        let mut a = member("a", 10);
        a.pin = Some(1);
        let mut b = member("b", 10);
        b.pin = Some(1);
        let plan = fleet(&[a, b], &[mac(), gpu()]).unwrap();
        assert_eq!(plan.endpoints(), 1);
        assert_eq!(plan.unplaced.len(), 1);
    }

    /// A draft model shares its agent's pool, so it counts against the budget.
    #[test]
    fn a_draft_model_is_charged_to_the_pool_that_hosts_it() {
        let mut m = member("target", 26);
        m.speculation = Some(Speculation {
            draft_weight_bytes: 6 << 30,
            draft_tokens: 4,
            acceptance_rate: 0.7,
        });
        // 26 + 6 GiB of draft plus KV does not fit the 30 GiB GPU.
        let plan = fleet(&[m], &[gpu()]).unwrap();
        assert_eq!(plan.endpoints(), 0);
        assert_eq!(plan.unplaced, vec![0]);
    }

    /// Three endpoints on two computers is three endpoints and two computers.
    /// Reporting it as three machines is the lie this exists to stop.
    #[test]
    fn endpoints_and_machines_are_counted_separately() {
        let plan =
            fleet(&[member("a", 10), member("b", 10), member("c", 10)], &[mac(), gpu(), ram()])
                .unwrap();
        assert_eq!(plan.endpoints(), 3);
        assert_eq!(plan.machines(), 2, "pc-gpu and pc-ram are one desktop");
    }

    #[test]
    fn agents_sharing_a_desktop_are_marked_co_resident() {
        let plan =
            fleet(&[member("a", 10), member("b", 10), member("c", 10)], &[mac(), gpu(), ram()])
                .unwrap();
        let names: Vec<&str> =
            plan.co_resident().map(|p| p.node_name.as_str()).collect();
        assert_eq!(names.len(), 2, "got {names:?}");
        assert!(names.contains(&"pc-gpu") && names.contains(&"pc-ram"), "got {names:?}");
        let mac_placement =
            plan.placements.iter().find(|p| p.node_name == "mac").unwrap();
        assert!(!mac_placement.co_resident, "the Mac is on its own");
    }

    /// A single agent on the desktop has the machine to itself, so nothing is
    /// charged and nothing is flagged.
    #[test]
    fn one_agent_on_a_two_pool_machine_is_not_co_resident() {
        let plan = fleet(&[member("a", 10)], &[gpu(), ram()]).unwrap();
        assert_eq!(plan.endpoints(), 1);
        assert_eq!(plan.machines(), 1);
        assert!(plan.co_resident().next().is_none());
    }

    /// The default of 1.0 must leave the numbers exactly where they were, so
    /// declaring a host is free until someone measures the interference.
    #[test]
    fn declaring_a_shared_host_alone_does_not_change_any_timing() {
        let separate = {
            let mut g = gpu();
            g.host = None;
            let mut r = ram();
            r.host = None;
            fleet(&[member("a", 10), member("b", 10)], &[g, r]).unwrap()
        };
        let shared = fleet(&[member("a", 10), member("b", 10)], &[gpu(), ram()]).unwrap();
        assert_eq!(shared.machines(), 1);
        assert_eq!(separate.machines(), 2);
        assert!(
            (shared.aggregate_tokens_per_sec() - separate.aggregate_tokens_per_sec()).abs()
                < 1e-9
        );
    }

    /// Once a real contention figure is recorded, co-resident agents slow down
    /// by exactly it and agents elsewhere do not.
    #[test]
    fn a_measured_contention_slows_only_the_co_resident_agents() {
        let mut g = gpu();
        g.contention = 0.5;
        let free = fleet(&[member("solo", 10)], &[g.clone()]).unwrap();
        let crowded =
            fleet(&[member("solo", 10), member("other", 10)], &[g, ram()]).unwrap();

        let alone = free.placements[0].secs_per_token;
        let shared = crowded
            .placements
            .iter()
            .find(|p| p.node_name == "pc-gpu")
            .unwrap()
            .secs_per_token;
        assert!((shared - alone * 2.0).abs() < 1e-12, "{shared} vs {alone}");
    }

    /// With contention charged, spreading over two machines can beat packing
    /// two agents into one - and the search has to be able to see that.
    #[test]
    fn contention_can_move_an_agent_off_a_crowded_machine() {
        let mut g = gpu();
        g.contention = 0.01;
        let mut r = ram();
        r.contention = 0.01;
        // Two agents, three pools. Packing both onto the desktop is legal but
        // now costs a hundredfold; one of them should take the Mac instead.
        let plan = fleet(&[member("a", 10), member("b", 10)], &[mac(), g, r]).unwrap();
        assert_eq!(plan.endpoints(), 2);
        assert_eq!(plan.machines(), 2, "both agents ended up on one machine");
    }

    #[test]
    fn an_empty_agent_list_is_an_error_rather_than_an_empty_fleet() {
        assert!(fleet(&[], &[mac()]).is_err());
    }
}
