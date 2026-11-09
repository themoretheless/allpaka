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

mod search;
use search::*;

#[cfg(test)]
mod tests;
