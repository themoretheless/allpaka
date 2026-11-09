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
    let plan = fleet(
        &[member("big", 80), member("small", 10)],
        &[mac(), gpu(), ram()],
    )
    .unwrap();
    assert_eq!(plan.endpoints(), 2);
    let big = plan
        .placements
        .iter()
        .find(|p| p.model_name == "big")
        .unwrap();
    assert_eq!(big.node_name, "mac");
}

/// With the big model on the Mac, the small one should land on the GPU
/// rather than on system RAM, because that is where it is fastest.
#[test]
fn a_small_model_is_given_the_fastest_free_pool() {
    let plan = fleet(
        &[member("big", 80), member("small", 10)],
        &[mac(), gpu(), ram()],
    )
    .unwrap();
    let small = plan
        .placements
        .iter()
        .find(|p| p.model_name == "small")
        .unwrap();
    assert_eq!(small.node_name, "pc-gpu");
}

/// Coverage beats speed: three agents on three pools is better than two
/// agents placed slightly faster.
#[test]
fn placing_every_model_wins_over_placing_fewer_of_them_faster() {
    let plan = fleet(
        &[member("a", 10), member("b", 10), member("c", 10)],
        &[mac(), gpu(), ram()],
    )
    .unwrap();
    assert_eq!(plan.endpoints(), 3, "all three should be placed");
    assert!(plan.unplaced.is_empty());
}

#[test]
fn one_pool_holds_at_most_one_model() {
    let members = [
        member("a", 10),
        member("b", 10),
        member("c", 10),
        member("d", 10),
    ];
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
    let t = plan
        .placements
        .iter()
        .find(|p| p.model_name == "tools")
        .unwrap();
    let r = plan
        .placements
        .iter()
        .find(|p| p.model_name == "reasoner")
        .unwrap();
    assert_eq!(t.context_tokens, 4096);
    assert!(
        r.kv_bytes > t.kv_bytes * 8,
        "{} vs {}",
        r.kv_bytes,
        t.kv_bytes
    );
}

/// A pin overrides the optimiser, even when it costs throughput.
#[test]
fn a_pinned_agent_stays_where_it_was_put() {
    let mut small = member("small", 10);
    small.pin = Some(2); // system RAM, far from optimal
    let plan = fleet(&[small, member("other", 10)], &[mac(), gpu(), ram()]).unwrap();
    let s = plan
        .placements
        .iter()
        .find(|p| p.model_name == "small")
        .unwrap();
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
    let plan = fleet(
        &[member("a", 10), member("b", 10), member("c", 10)],
        &[mac(), gpu(), ram()],
    )
    .unwrap();
    assert_eq!(plan.endpoints(), 3);
    assert_eq!(plan.machines(), 2, "pc-gpu and pc-ram are one desktop");
}

#[test]
fn agents_sharing_a_desktop_are_marked_co_resident() {
    let plan = fleet(
        &[member("a", 10), member("b", 10), member("c", 10)],
        &[mac(), gpu(), ram()],
    )
    .unwrap();
    let names: Vec<&str> = plan.co_resident().map(|p| p.node_name.as_str()).collect();
    assert_eq!(names.len(), 2, "got {names:?}");
    assert!(
        names.contains(&"pc-gpu") && names.contains(&"pc-ram"),
        "got {names:?}"
    );
    let mac_placement = plan
        .placements
        .iter()
        .find(|p| p.node_name == "mac")
        .unwrap();
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
        (shared.aggregate_tokens_per_sec() - separate.aggregate_tokens_per_sec()).abs() < 1e-9
    );
}

/// Once a real contention figure is recorded, co-resident agents slow down
/// by exactly it and agents elsewhere do not.
#[test]
fn a_measured_contention_slows_only_the_co_resident_agents() {
    let mut g = gpu();
    g.contention = 0.5;
    let free = fleet(&[member("solo", 10)], &[g.clone()]).unwrap();
    let crowded = fleet(&[member("solo", 10), member("other", 10)], &[g, ram()]).unwrap();

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
