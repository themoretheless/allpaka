//! Depth-first assignment of fleet members to pools, with co-residency
//! charged to pools that share a host.

use super::*;

/// Depth-first over models, tracking which pools are taken in a bitmask.
#[allow(clippy::too_many_arguments)]
pub(super) fn search(
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
pub(super) fn charge_co_residency(placements: &mut [Placement], nodes: &[Node]) {
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
pub(super) fn place(
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
