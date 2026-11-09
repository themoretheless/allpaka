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
}

/// Beyond this the exhaustive search stops being instant. Real fleets are far
/// smaller; the cap turns a pathological config into an error rather than a
/// hang.
const MAX_SEARCH: usize = 8;

/// Lightweight stack representation of a candidate `(model_index, node_index)`
/// pair during DFS search, avoiding heap allocations and `String` clones.
#[derive(Clone, Copy, Default)]
struct Cand {
    model_index: u8,
    node_index: u8,
    host_id: u8,
    secs_per_token: f64,
}

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

    // Intern node host names into small `0..MAX_SEARCH` integer IDs so co-residency
    // checks inside the DFS leaf loop are a single byte array lookup.
    let mut host_names: Vec<&str> = Vec::with_capacity(nodes.len());
    let mut node_host_ids = [0u8; MAX_SEARCH];
    let mut node_contention = [1.0f64; MAX_SEARCH];
    for (ni, node) in nodes.iter().enumerate() {
        let h = node.host();
        let hid = match host_names.iter().position(|&existing| existing == h) {
            Some(pos) => pos as u8,
            None => {
                let pos = host_names.len() as u8;
                host_names.push(h);
                pos
            }
        };
        node_host_ids[ni] = hid;
        node_contention[ni] = node.contention.clamp(0.0, 1.0);
    }

    // Precompute which (agent, pool) pairs are even possible. A pinned agent
    // has exactly one candidate.
    let mut options: Vec<Vec<Option<Placement>>> = Vec::with_capacity(members.len());
    let mut cands = [[None::<Cand>; MAX_SEARCH]; MAX_SEARCH];
    for (mi, member) in members.iter().enumerate() {
        let mut row = Vec::with_capacity(nodes.len());
        for (ni, node) in nodes.iter().enumerate() {
            let allowed = member.pin.is_none_or(|p| p == ni);
            let p = if allowed {
                place(mi, member, ni, node)
            } else {
                None
            };
            if let Some(ref placement) = p {
                cands[mi][ni] = Some(Cand {
                    model_index: mi as u8,
                    node_index: ni as u8,
                    host_id: node_host_ids[ni],
                    secs_per_token: placement.secs_per_token,
                });
            }
            row.push(p);
        }
        options.push(row);
    }

    let mut best = FleetPlan::default();
    let mut best_count = 0usize;
    let mut best_tps = 0.0f64;
    let mut current = [Cand::default(); MAX_SEARCH];
    search(
        &options,
        &cands,
        nodes,
        &node_contention,
        0,
        0u32,
        &mut current,
        0,
        &mut best,
        &mut best_count,
        &mut best_tps,
    );

    let placed: Vec<usize> = best.placements.iter().map(|p| p.model_index).collect();
    best.unplaced = (0..members.len()).filter(|i| !placed.contains(i)).collect();
    Ok(best)
}

/// Memory an agent needs on whichever pool hosts it.
pub fn required_bytes(m: &FleetMember) -> u64 {
    let mut need =
        m.model.total_weight_bytes + m.model.kv_bytes(m.model.n_layers, m.context_tokens);
    if let Some(s) = &m.speculation {
        need += s.draft_weight_bytes;
    }
    need
}

/// Depth-first over models, tracking which pools are taken in a bitmask.
#[allow(clippy::too_many_arguments)]
fn search(
    options: &[Vec<Option<Placement>>],
    cands: &[[Option<Cand>; MAX_SEARCH]; MAX_SEARCH],
    nodes: &[Node],
    node_contention: &[f64; MAX_SEARCH],
    model_index: usize,
    used_nodes: u32,
    current: &mut [Cand; MAX_SEARCH],
    current_len: usize,
    best: &mut FleetPlan,
    best_count: &mut usize,
    best_tps: &mut f64,
) {
    if model_index == options.len() {
        if current_len < *best_count {
            return;
        }
        // Co-residency is a property of the whole assignment, not of one pair,
        // so it can only be charged once the candidate is complete.
        let mut host_counts = [0u8; MAX_SEARCH];
        for c in &current[..current_len] {
            host_counts[c.host_id as usize] += 1;
        }
        let mut agg_tps = 0.0f64;
        for c in &current[..current_len] {
            let secs = if host_counts[c.host_id as usize] > 1 {
                crate::cost::contended(c.secs_per_token, node_contention[c.node_index as usize])
            } else {
                c.secs_per_token
            };
            agg_tps += if secs <= 0.0 {
                f64::INFINITY
            } else {
                1.0 / secs
            };
        }
        let is_better =
            current_len > *best_count || (current_len == *best_count && agg_tps > *best_tps);
        if is_better {
            let mut placements: Vec<Placement> = current[..current_len]
                .iter()
                .map(|c| {
                    options[c.model_index as usize][c.node_index as usize]
                        .as_ref()
                        .expect("valid placement option")
                        .clone()
                })
                .collect();
            charge_co_residency(&mut placements, nodes);
            *best_count = current_len;
            *best_tps = placements.iter().map(Placement::tokens_per_sec).sum();
            *best = FleetPlan {
                placements,
                unplaced: Vec::new(),
            };
        }
        return;
    }

    // Prune branch if even placing all remaining models cannot beat best_count.
    let remaining = options.len() - model_index;
    if current_len + remaining < *best_count {
        return;
    }

    for (node_index, cand_opt) in cands[model_index][..nodes.len()].iter().enumerate() {
        let Some(cand) = *cand_opt else { continue };
        let bit = 1u32 << node_index;
        if used_nodes & bit != 0 {
            continue;
        }
        current[current_len] = cand;
        search(
            options,
            cands,
            nodes,
            node_contention,
            model_index + 1,
            used_nodes | bit,
            current,
            current_len + 1,
            best,
            best_count,
            best_tps,
        );
    }

    // Leaving a model unplaced is a legal branch: sometimes there is simply no
    // pool left that holds it. Explored after placement branches so `best_count`
    // reaches high coverage early and prunes inferior unplaced subtrees.
    search(
        options,
        cands,
        nodes,
        node_contention,
        model_index + 1,
        used_nodes,
        current,
        current_len,
        best,
        best_count,
        best_tps,
    );
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
    let node_indices: Vec<usize> = placements.iter().map(|p| p.node_index).collect();
    let scales = crate::cost::co_residency_scale(&hosts, &node_indices, nodes);
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
mod tests;
