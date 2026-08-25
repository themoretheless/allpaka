//! Replication: the placement with no coupling at all.
//!
//! Every other arrangement in this crate splits one request across machines,
//! and therefore pays a hop somewhere inside every forward pass. Replication
//! does the opposite: each machine that can hold the whole model runs its own
//! independent instance, and requests are routed to whichever is free.
//!
//! The machines then never talk during inference. Not less - never. There is no
//! activation to ship, no round trip to wait on, no tail latency to budget for,
//! and no reason to measure the link at all. A machine that goes down takes its
//! own requests with it and nothing else.
//!
//! # What you trade
//!
//! Replication cannot make a single request faster. Each reply runs at whatever
//! one machine can do alone, so the latency a user sees is the latency of the
//! machine that answered - never better, and worse than a split that would have
//! shared the work.
//!
//! What it does buy is aggregate throughput that adds up cleanly, and the
//! ability to serve as many requests at once as there are machines. For serving
//! several people, or an agent loop issuing parallel calls, this beats a split
//! outright. For one person waiting on one answer, it does not.
//!
//! # When it is not available
//!
//! Replication needs the model to fit on a machine *by itself*, including KV
//! cache at the target context length. The moment it does not, the choice
//! collapses back to splitting, because there is nothing to replicate.
//!
//! ```text
//! isolation:   replicate  >  pipeline + speculation  >  pipeline  >  tensor parallel
//! coupling:    never         once per K tokens         every token  every layer
//! ```

use crate::{Model, Node, PlanRequest};

/// One independent model instance on one machine.
#[derive(Debug, Clone)]
pub struct Replica {
    pub node_index: usize,
    pub node_name: String,
    /// Latency this instance delivers on its own, under speculation if one was
    /// requested.
    pub secs_per_token: f64,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    /// Time to first token at the requested prompt length, or `None` when the
    /// pool's FLOP/s or the model's parameter count is unknown.
    pub ttft_secs: Option<f64>,
    /// Physical machine hosting this pool.
    pub host: String,
    /// Whether another instance in this plan runs on the same machine. When
    /// true, `secs_per_token` already carries that pool's contention factor.
    pub co_resident: bool,
}

impl Replica {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.secs_per_token <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.secs_per_token
        }
    }
}

/// A fleet of independent instances.
#[derive(Debug, Clone)]
pub struct ReplicaPlan {
    pub replicas: Vec<Replica>,
}

/// An instance slower than this fraction of the fastest one is not worth
/// routing real traffic to: whoever lands on it waits many times longer than
/// everyone else, and the throughput it contributes is rounding error.
const USEFUL_SHARE_OF_FASTEST: f64 = 0.2;

impl ReplicaPlan {
    /// Tokens per second with every instance busy. These add up exactly,
    /// because nothing is shared.
    pub fn aggregate_tokens_per_sec(&self) -> f64 {
        self.replicas.iter().map(Replica::tokens_per_sec).sum()
    }

    /// Instances fast enough to route traffic to.
    pub fn useful(&self) -> impl Iterator<Item = &Replica> {
        let cutoff = self.fastest_tokens_per_sec() * USEFUL_SHARE_OF_FASTEST;
        self.replicas.iter().filter(move |r| r.tokens_per_sec() >= cutoff)
    }

    /// Whether an instance is worth routing to. Reported per replica so a slow
    /// one is visible rather than silently folded into a flattering total.
    pub fn is_useful(&self, r: &Replica) -> bool {
        r.tokens_per_sec() >= self.fastest_tokens_per_sec() * USEFUL_SHARE_OF_FASTEST
    }

    /// Aggregate over the instances worth using.
    pub fn useful_tokens_per_sec(&self) -> f64 {
        self.useful().map(Replica::tokens_per_sec).sum()
    }

    pub fn useful_concurrency(&self) -> usize {
        self.useful().count()
    }

    fn fastest_tokens_per_sec(&self) -> f64 {
        self.replicas.iter().map(Replica::tokens_per_sec).fold(0.0, f64::max)
    }

    /// Latency of the fastest instance: what one waiting user sees when routed
    /// well.
    pub fn best_secs_per_token(&self) -> f64 {
        self.replicas
            .iter()
            .map(|r| r.secs_per_token)
            .min_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY)
    }

    /// How many requests can be in flight without queueing.
    pub fn concurrency(&self) -> usize {
        self.replicas.len()
    }

    /// Distinct physical machines the instances sit on, which is the number of
    /// independent failure domains. Never larger than `concurrency`, and the
    /// gap is what "independent" quietly overstates.
    pub fn machines(&self) -> usize {
        let mut hosts: Vec<&str> = self.replicas.iter().map(|r| r.host.as_str()).collect();
        hosts.sort_unstable();
        hosts.dedup();
        hosts.len()
    }
}

/// Place an independent instance on every machine that can hold the whole
/// model. Returns `None` if no machine can.
pub fn replicate(nodes: &[Node], model: &Model, req: &PlanRequest) -> Option<ReplicaPlan> {
    let kv_bytes = model.kv_bytes(model.n_layers, req.context_tokens);
    let replicas: Vec<Replica> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            let mut need = model.total_weight_bytes + kv_bytes;
            // A draft model has to live beside the target it drafts for.
            if let Some(s) = &req.speculation {
                need += s.draft_weight_bytes;
            }
            if need > node.usable_bytes {
                return None;
            }
            Some(Replica {
                node_index: i,
                node_name: node.name.clone(),
                secs_per_token: crate::cost::local_secs_per_token(
                    node,
                    model,
                    req.context_tokens,
                    req.speculation.as_ref(),
                ),
                weight_bytes: model.total_weight_bytes,
                kv_bytes,
                ttft_secs: crate::cost::prefill_secs(
                    node,
                    model,
                    model.n_layers,
                    req.prompt_tokens,
                ),
                host: node.host().to_string(),
                co_resident: false,
            })
        })
        .collect();

    if replicas.is_empty() {
        return None;
    }
    let mut plan = ReplicaPlan { replicas };
    charge_co_residency(&mut plan.replicas, nodes);
    Some(plan)
}

/// Instances sharing a machine also share it under load; scale each one by its
/// pool's measured contention.
fn charge_co_residency(replicas: &mut [Replica], nodes: &[Node]) {
    let hosts: Vec<&str> = replicas.iter().map(|r| r.host.as_str()).collect();
    let indices: Vec<usize> = replicas.iter().map(|r| r.node_index).collect();
    let scales = crate::cost::co_residency_scale(&hosts, &indices, nodes);
    for (r, scale) in replicas.iter_mut().zip(scales) {
        if let Some(s) = scale {
            r.co_resident = true;
            r.secs_per_token = crate::cost::contended(r.secs_per_token, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Speculation};

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
            name: "pc-gpu".into(),
            backend: Backend::Cuda,
            usable_bytes: 30 << 30,
            mem_bandwidth_bytes_per_sec: 1792e9,
            bandwidth_efficiency: 0.7,
            ..Node::default()
        }
    }

    fn model(gib: u64) -> Model {
        Model {
            name: "m".into(),
            n_layers: 64,
            hidden_size: 5120,
            total_weight_bytes: gib << 30,
            kv_bytes_per_token_per_layer: 4096,
            activation_bytes: 2,
            active_weight_fraction: 1.0,
            param_count: 0,
        }
    }

    #[test]
    fn only_machines_that_hold_the_whole_model_get_an_instance() {
        // 34 GiB fits the mac, not the 30 GiB GPU.
        let plan = replicate(&[mac(), pc()], &model(34), &PlanRequest::default()).unwrap();
        assert_eq!(plan.concurrency(), 1);
        assert_eq!(plan.replicas[0].node_name, "mac");
    }

    #[test]
    fn a_model_that_fits_everywhere_is_replicated_everywhere() {
        let plan = replicate(&[mac(), pc()], &model(20), &PlanRequest::default()).unwrap();
        assert_eq!(plan.concurrency(), 2);
    }

    /// The defining property: throughput adds up, because nothing is shared.
    #[test]
    fn aggregate_throughput_is_the_exact_sum_of_the_instances() {
        let plan = replicate(&[mac(), pc()], &model(20), &PlanRequest::default()).unwrap();
        let sum: f64 = plan.replicas.iter().map(|r| r.tokens_per_sec()).sum();
        assert!((plan.aggregate_tokens_per_sec() - sum).abs() < 1e-9);
        // And it beats either machine alone.
        assert!(plan.aggregate_tokens_per_sec() > plan.replicas[0].tokens_per_sec());
    }

    /// Replication never improves the latency of one request. That is the
    /// trade, and it should be visible in the numbers rather than glossed over.
    #[test]
    fn replication_does_not_make_a_single_request_faster() {
        let nodes = [mac(), pc()];
        let m = model(20);
        let plan = replicate(&nodes, &m, &PlanRequest::default()).unwrap();
        let req = PlanRequest::default();
        let fastest_alone = nodes
            .iter()
            .map(|n| n.stream_time(m.streamed_bytes_per_step(m.n_layers, req.context_tokens)))
            .min_by(f64::total_cmp)
            .unwrap();
        assert!((plan.best_secs_per_token() - fastest_alone).abs() < 1e-12);
    }

    /// A hopelessly slow instance must not inflate the headline number.
    #[test]
    fn a_far_slower_instance_is_excluded_from_the_useful_total() {
        let mut slow = mac();
        slow.name = "pc-ram".into();
        slow.mem_bandwidth_bytes_per_sec = 51.2e9;
        slow.bandwidth_efficiency = 0.6;

        let plan = replicate(&[pc(), slow], &model(20), &PlanRequest::default()).unwrap();
        assert_eq!(plan.concurrency(), 2);
        assert_eq!(plan.useful_concurrency(), 1, "the DDR4 instance should be excluded");
        assert!(plan.useful_tokens_per_sec() < plan.aggregate_tokens_per_sec());
        assert_eq!(plan.useful().next().unwrap().node_name, "pc-gpu");
    }

    /// Comparable machines both count.
    #[test]
    fn instances_of_similar_speed_all_count_as_useful() {
        let plan = replicate(&[mac(), pc()], &model(20), &PlanRequest::default()).unwrap();
        assert_eq!(plan.useful_concurrency(), 2);
    }

    #[test]
    fn nothing_can_be_replicated_when_the_model_fits_nowhere() {
        assert!(replicate(&[mac(), pc()], &model(120), &PlanRequest::default()).is_none());
    }

    /// A draft model occupies memory on the same machine, so it can push an
    /// instance over the edge.
    #[test]
    fn the_draft_model_counts_against_the_memory_budget() {
        let spec = Speculation {
            draft_weight_bytes: 6 << 30,
            draft_tokens: 4,
            acceptance_rate: 0.7,
        };
        let m = model(26);
        let plain = replicate(&[pc()], &m, &PlanRequest::default());
        assert!(plain.is_some(), "26 GiB alone fits the 30 GiB GPU");

        let with_draft = PlanRequest { speculation: Some(spec), ..Default::default() };
        assert!(
            replicate(&[pc()], &m, &with_draft).is_none(),
            "26 + 6 GiB of draft should not fit"
        );
    }

    #[test]
    fn speculation_speeds_up_every_instance() {
        let m = model(20);
        let plain = replicate(&[mac()], &m, &PlanRequest::default()).unwrap();
        let req = PlanRequest {
            speculation: Some(Speculation {
                draft_weight_bytes: 1 << 30,
                draft_tokens: 4,
                acceptance_rate: 0.8,
            }),
            ..Default::default()
        };
        let spec = replicate(&[mac()], &m, &req).unwrap();
        assert!(spec.aggregate_tokens_per_sec() > plain.aggregate_tokens_per_sec());
    }
}
