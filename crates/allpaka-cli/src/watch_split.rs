//! `allpaka watch --manifest`: a layer split as a phase plus conditions.
//!
//! The same shape as watching one server, over what a split adds: is every
//! stage listening (`StageReachable/<node>`), do the stages cover the layers
//! without gaps or overlaps (`LayersPlaced`), and was every hop between two
//! machines measured rather than assumed (`LinkVerified`). Each condition is a
//! function of the manifest, the cluster config and this round's probes;
//! nothing is remembered between rounds except to print transitions.

use crate::watch::Phase;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

/// One stage of the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    pub node: String,
    pub endpoint: String,
    pub first: u64,
    pub end: u64,
}

/// What is known about the hop between two adjacent stages.
#[derive(Debug, Clone, PartialEq)]
pub enum Hop {
    /// Both stages run on one node: no network hop.
    SameNode,
    /// The cluster config holds a measurement for this pair.
    Measured,
    /// Only the config's `[link]` fallback covers the pair.
    FallbackOnly,
    /// The config says nothing about the pair.
    Unmeasured,
    /// No cluster config was given, or it names neither node.
    Unknown,
}

/// One round of probes. `connect[i]` is the time to open a TCP connection to
/// stage `i`, or `None` when it did not open within the budget.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Facts {
    pub connect: Vec<Option<Duration>>,
    /// `Some(answered)` when a gateway address was given.
    pub gateway: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub name: String,
    pub held: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub phase: Phase,
    pub conditions: Vec<Condition>,
}

pub fn stages(manifest: &Value) -> Result<Vec<Stage>> {
    manifest["stages"]
        .as_array()
        .context("manifest has no stages array")?
        .iter()
        .map(|stage| {
            Ok(Stage {
                node: stage["node"]
                    .as_str()
                    .context("stage without a node")?
                    .to_owned(),
                endpoint: stage["endpoint"]
                    .as_str()
                    .context("stage without an endpoint")?
                    .to_owned(),
                first: stage["first"].as_u64().context("stage without first")?,
                end: stage["end"].as_u64().context("stage without end")?,
            })
        })
        .collect()
}

/// Why the stages do not tile the layers from 0 upward, or `None` if they do.
fn misplaced(stages: &[Stage]) -> Option<String> {
    if stages.is_empty() {
        return Some("the manifest has no stages".to_owned());
    }
    let mut next = 0;
    for stage in stages {
        if stage.end <= stage.first {
            return Some(format!("{} holds an empty range", stage.node));
        }
        if stage.first > next {
            return Some(format!("layers {next}..{} are on no stage", stage.first));
        }
        if stage.first < next {
            return Some(format!("layers {}..{next} are on two stages", stage.first));
        }
        next = stage.end;
    }
    None
}

/// What the cluster config says about links, by node name.
pub struct Links<'a> {
    /// Whether this pair of nodes has its own measurement.
    pub measured: &'a dyn Fn(&str, &str) -> bool,
    /// Whether the config has a `[link]` default.
    pub fallback: bool,
    /// Whether the config names this node.
    pub known: &'a dyn Fn(&str) -> bool,
}

/// The hops between adjacent stages, judged by the cluster config.
pub fn hops(stages: &[Stage], config: Option<&Links<'_>>) -> Vec<Hop> {
    stages
        .windows(2)
        .map(|pair| {
            let (a, b) = (pair[0].node.as_str(), pair[1].node.as_str());
            if a == b {
                return Hop::SameNode;
            }
            match config {
                Some(links) if (links.known)(a) && (links.known)(b) => {
                    if (links.measured)(a, b) {
                        Hop::Measured
                    } else if links.fallback {
                        Hop::FallbackOnly
                    } else {
                        Hop::Unmeasured
                    }
                }
                _ => Hop::Unknown,
            }
        })
        .collect()
}

pub fn classify(stages: &[Stage], hops: &[Hop], facts: &Facts) -> Status {
    let mut conditions = Vec::new();

    let misplaced = misplaced(stages);
    conditions.push(Condition {
        name: "LayersPlaced".to_owned(),
        held: misplaced.is_none(),
        reason: misplaced.clone().unwrap_or_else(|| {
            format!(
                "layers 0..{} on {} stage(s)",
                stages.last().map_or(0, |s| s.end),
                stages.len()
            )
        }),
    });

    let mut up = 0;
    for (stage, connect) in stages.iter().zip(&facts.connect) {
        up += usize::from(connect.is_some());
        conditions.push(Condition {
            name: format!("StageReachable/{}", stage.node),
            held: connect.is_some(),
            reason: match connect {
                Some(took) => format!("{} connected in {} ms", stage.endpoint, took.as_millis()),
                None => format!("{} did not accept a connection", stage.endpoint),
            },
        });
    }

    // A hop that only the fallback covers is exactly the optimistic guess the
    // planner refuses to cut across, so it does not verify the link.
    let unverified: Vec<String> = stages
        .windows(2)
        .zip(hops)
        .filter_map(|(pair, hop)| {
            let why = match hop {
                Hop::SameNode | Hop::Measured => return None,
                Hop::FallbackOnly => "only the [link] fallback",
                Hop::Unmeasured => "not measured",
                Hop::Unknown => "no cluster config to check",
            };
            Some(format!("{}-{}: {why}", pair[0].node, pair[1].node))
        })
        .collect();
    let crossings = hops.iter().filter(|hop| **hop != Hop::SameNode).count();
    conditions.push(Condition {
        name: "LinkVerified".to_owned(),
        held: unverified.is_empty(),
        reason: if unverified.is_empty() {
            format!("{crossings} hop(s) between machines, each measured")
        } else {
            unverified.join("; ")
        },
    });

    if let Some(answered) = facts.gateway {
        conditions.push(Condition {
            name: "GatewayReachable".to_owned(),
            held: answered,
            reason: String::new(),
        });
    }

    let all_up = up == stages.len() && facts.gateway != Some(false);
    let none_up = up == 0 && facts.gateway != Some(true);
    let phase = if none_up {
        Phase::Stopped
    } else if !all_up {
        Phase::Starting
    } else if misplaced.is_some() || !unverified.is_empty() {
        Phase::Degraded
    } else {
        Phase::Ready
    };
    Status { phase, conditions }
}

/// Lines for what changed: `A -> B` for the phase, `+Name`/`-Name` for a
/// condition that turned.
pub fn diff(previous: &Status, current: &Status) -> Vec<String> {
    let mut lines = Vec::new();
    if previous.phase != current.phase {
        lines.push(format!("{:?} -> {:?}", previous.phase, current.phase));
    }
    for condition in &current.conditions {
        let before = previous
            .conditions
            .iter()
            .find(|c| c.name == condition.name);
        if before.is_none_or(|b| b.held != condition.held) {
            let sign = if condition.held { '+' } else { '-' };
            lines.push(format!("{sign}{} ({})", condition.name, condition.reason));
        }
    }
    lines
}

fn connect(endpoint: &str, budget: Duration) -> Option<Duration> {
    use std::net::ToSocketAddrs;
    let address = endpoint.to_socket_addrs().ok()?.next()?;
    let started = Instant::now();
    std::net::TcpStream::connect_timeout(&address, budget).ok()?;
    Some(started.elapsed())
}

fn status_json(status: &Status) -> Value {
    json!({
        "phase": format!("{:?}", status.phase),
        "conditions": status.conditions.iter().map(|c| json!({
            "name": c.name, "held": c.held, "reason": c.reason,
        })).collect::<Vec<_>>(),
    })
}

pub fn watch(
    manifest_path: &Path,
    config_path: Option<&Path>,
    gateway: Option<&str>,
    interval: Duration,
    as_json: bool,
    once: bool,
) -> Result<()> {
    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_str(&text).context("the manifest is not JSON")?;
    let stages = stages(&manifest)?;

    let hops = match config_path {
        None => hops(&stages, None),
        Some(path) => {
            let config = crate::config::Config::load(path)?;
            let names: Vec<String> = config
                .resolve_nodes()?
                .into_iter()
                .map(|node| node.name)
                .collect();
            let fabric = config.resolve_fabric()?;
            let index = |name: &str| names.iter().position(|n| n == name);
            let measured = |a: &str, b: &str| match (index(a), index(b)) {
                (Some(a), Some(b)) => fabric
                    .edges()
                    .any(|(x, y, _)| (x, y) == (a, b) || (x, y) == (b, a)),
                _ => false,
            };
            let known = |name: &str| index(name).is_some();
            hops(
                &stages,
                Some(&Links {
                    measured: &measured,
                    fallback: fabric.fallback().is_some(),
                    known: &known,
                }),
            )
        }
    };

    let budget = interval.max(Duration::from_millis(50));
    let mut previous: Option<Status> = None;
    loop {
        let tick = Instant::now();
        let facts = Facts {
            connect: stages
                .iter()
                .map(|s| connect(&s.endpoint, budget))
                .collect(),
            gateway: gateway
                .map(|addr| crate::client::get_within(addr, "/v1/models", budget).is_ok()),
        };
        let status = classify(&stages, &hops, &facts);
        let changes = match &previous {
            Some(previous) => diff(previous, &status),
            None => status
                .conditions
                .iter()
                .map(|c| {
                    format!(
                        "{}{} ({})",
                        if c.held { '+' } else { '-' },
                        c.name,
                        c.reason
                    )
                })
                .collect(),
        };
        if previous.is_none() || !changes.is_empty() {
            if as_json {
                let mut line = status_json(&status);
                line["changes"] = json!(changes);
                println!("{line}");
            } else {
                println!("phase={:?}  {}", status.phase, changes.join("  "));
            }
        }
        if once {
            return Ok(());
        }
        previous = Some(status);
        std::thread::sleep(interval.saturating_sub(tick.elapsed()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(node: &str, first: u64, end: u64) -> Stage {
        Stage {
            node: node.to_owned(),
            endpoint: format!("{node}.lan:9798"),
            first,
            end,
        }
    }

    fn two() -> Vec<Stage> {
        vec![stage("mac", 0, 16), stage("pc", 16, 32)]
    }

    fn up(n: usize) -> Facts {
        Facts {
            connect: vec![Some(Duration::from_millis(2)); n],
            gateway: None,
        }
    }

    fn held(status: &Status, name: &str) -> bool {
        status
            .conditions
            .iter()
            .find(|c| c.name == name)
            .unwrap()
            .held
    }

    #[test]
    fn a_measured_split_with_every_stage_up_is_ready() {
        let status = classify(&two(), &[Hop::Measured], &up(2));
        assert_eq!(status.phase, Phase::Ready);
        for name in [
            "LayersPlaced",
            "StageReachable/mac",
            "StageReachable/pc",
            "LinkVerified",
        ] {
            assert!(held(&status, name), "{name}");
        }
    }

    #[test]
    fn a_stage_that_is_not_listening_keeps_the_split_starting() {
        let mut facts = up(2);
        facts.connect[1] = None;
        let status = classify(&two(), &[Hop::Measured], &facts);
        assert_eq!(status.phase, Phase::Starting);
        assert!(held(&status, "StageReachable/mac"));
        assert!(!held(&status, "StageReachable/pc"));

        facts.connect[0] = None;
        assert_eq!(
            classify(&two(), &[Hop::Measured], &facts).phase,
            Phase::Stopped
        );
    }

    #[test]
    fn a_hop_that_was_never_measured_does_not_verify_the_link() {
        for hop in [Hop::FallbackOnly, Hop::Unmeasured, Hop::Unknown] {
            let status = classify(&two(), std::slice::from_ref(&hop), &up(2));
            assert!(!held(&status, "LinkVerified"), "{hop:?}");
            assert_eq!(status.phase, Phase::Degraded);
        }
        // Two stages on one machine cross no link.
        let local = vec![stage("pc", 0, 16), stage("pc", 16, 32)];
        let status = classify(&local, &[Hop::SameNode], &up(2));
        assert!(held(&status, "LinkVerified"));
        assert_eq!(status.phase, Phase::Ready);
    }

    #[test]
    fn gaps_and_overlaps_are_named() {
        let gap = vec![stage("mac", 0, 16), stage("pc", 18, 32)];
        let status = classify(&gap, &[Hop::Measured], &up(2));
        let placed = status
            .conditions
            .iter()
            .find(|c| c.name == "LayersPlaced")
            .unwrap();
        assert!(
            !placed.held && placed.reason.contains("16..18"),
            "{placed:?}"
        );
        assert_eq!(status.phase, Phase::Degraded);

        let overlap = vec![stage("mac", 0, 16), stage("pc", 12, 32)];
        let status = classify(&overlap, &[Hop::Measured], &up(2));
        assert!(!held(&status, "LayersPlaced"));
        let late = vec![stage("mac", 4, 16)];
        assert!(!held(&classify(&late, &[], &up(1)), "LayersPlaced"));
    }

    #[test]
    fn hops_come_from_the_config_not_from_hope() {
        let stages = vec![
            stage("mac", 0, 8),
            stage("pc", 8, 16),
            stage("pc", 16, 24),
            stage("nas", 24, 32),
        ];
        let measured = |a: &str, b: &str| (a, b) == ("mac", "pc");
        let known = |name: &str| name != "nas";
        assert_eq!(
            hops(
                &stages,
                Some(&Links {
                    measured: &measured,
                    fallback: true,
                    known: &known
                })
            ),
            [Hop::Measured, Hop::SameNode, Hop::Unknown]
        );
        let none = |_: &str, _: &str| false;
        let all = |_: &str| true;
        assert_eq!(
            hops(
                &stages,
                Some(&Links {
                    measured: &none,
                    fallback: true,
                    known: &all
                })
            )[0],
            Hop::FallbackOnly
        );
        assert_eq!(
            hops(
                &stages,
                Some(&Links {
                    measured: &none,
                    fallback: false,
                    known: &all
                })
            )[0],
            Hop::Unmeasured
        );
        assert_eq!(
            hops(&stages, None),
            [Hop::Unknown, Hop::SameNode, Hop::Unknown]
        );
    }

    #[test]
    fn a_gateway_that_does_not_answer_holds_the_phase_back() {
        let mut facts = up(2);
        facts.gateway = Some(false);
        let status = classify(&two(), &[Hop::Measured], &facts);
        assert_eq!(status.phase, Phase::Starting);
        assert!(!held(&status, "GatewayReachable"));
        facts.gateway = Some(true);
        let ready = classify(&two(), &[Hop::Measured], &facts);
        assert_eq!(ready.phase, Phase::Ready);
        assert_eq!(
            diff(&status, &ready),
            ["Starting -> Ready", "+GatewayReachable ()"]
        );
    }
}
