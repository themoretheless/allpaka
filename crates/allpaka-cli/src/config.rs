//! The `allpaka.toml` cluster description.

use allpaka_core::{presets, Fabric, Link, Node};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Machines available to hold layers. Order does not matter: the planner
    /// tries every ordering, because with unequal links between pairs the
    /// choice of which machine leads the pipeline changes the answer.
    pub nodes: Vec<NodeConfig>,
    /// Measured links between specific pairs.
    #[serde(default)]
    pub links: Vec<LinkConfig>,
    /// Fallback for pairs with no entry in `links`. Leave it out and an
    /// unmeasured hop blocks the split instead of being optimistically
    /// guessed.
    pub link: Option<Link>,
    /// Agents to place, one model each. This is the record of what runs
    /// where: `allpaka fleet` reads it, and re-reading it later tells you the
    /// same thing it told you the first time.
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
    #[serde(default)]
    pub defaults: Defaults,
}

/// One agent: a name, a model, and how much of the machine it needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// What this agent is for. Used in reports and as the server name.
    pub name: String,
    /// Path to the .gguf this agent runs.
    pub model: PathBuf,
    /// Context length for this agent alone. Falls back to `defaults`.
    pub ctx: Option<u32>,
    /// Typical prompt length, for the time-to-first-token estimate. Falls back
    /// to `defaults`.
    pub prompt: Option<u32>,
    /// Force this agent onto a named pool. Leave unset to let the planner
    /// choose; set it when you know something the cost model does not.
    pub pin: Option<String>,
    /// Port its server listens on. Recorded here so the routing table and the
    /// placement cannot drift apart.
    pub port: Option<u16>,
    /// Draft model for speculative decoding. Lives on the same pool, so it
    /// counts against that pool's memory.
    pub draft: Option<PathBuf>,
    #[serde(default = "default_draft_len")]
    pub draft_len: u32,
    #[serde(default = "default_accept")]
    pub accept: f64,
}

fn default_draft_len() -> u32 {
    4
}

fn default_accept() -> f64 {
    0.7
}

/// A measured link between two named nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkConfig {
    /// The two node names this measurement applies to.
    pub between: [String; 2],
    #[serde(flatten)]
    pub link: Link,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defaults {
    pub context_tokens: u32,
    pub prompt_tokens: u32,
    /// Bytes per KV cache element. 2 for f16, 1 for q8_0.
    pub kv_cache_dtype_bytes: u64,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { context_tokens: 8192, prompt_tokens: 2048, kv_cache_dtype_bytes: 2 }
    }
}

/// A node is described either by naming a hardware preset or by giving the
/// numbers directly. Explicit fields override the preset, so you can start from
/// a preset and correct the one value you measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub preset: Option<String>,
    pub usable_gib: Option<f64>,
    pub mem_bandwidth_gbps: Option<f64>,
    pub bandwidth_efficiency: Option<f64>,
    /// Physical machine this pool lives in. Give two pools the same `host` and
    /// the planner stops treating them as independent machines. Leave it unset
    /// and the pool is assumed to be a machine of its own.
    pub host: Option<String>,
    /// Fraction of this pool's bandwidth left when a co-resident pool is also
    /// serving. Defaults to 1.0; measure before believing anything lower.
    pub contention: Option<f64>,
    /// Achieved TFLOP/s during prompt processing. Overrides the preset; leave
    /// unset on a preset-less node and time-to-first-token reads "unknown".
    pub prefill_tflops: Option<f64>,
}

impl NodeConfig {
    pub fn resolve(&self) -> Result<Node> {
        let mut node = match &self.preset {
            Some(id) => presets::find(id)
                .with_context(|| {
                    format!(
                        "unknown preset {id:?}; run `allpaka presets` to see the available ones"
                    )
                })?
                .to_node(&self.name),
            None => {
                if self.usable_gib.is_none() || self.mem_bandwidth_gbps.is_none() {
                    bail!(
                        "node {:?} has no preset, so it must set both usable_gib and \
                         mem_bandwidth_gbps",
                        self.name
                    );
                }
                Node { name: self.name.clone(), ..Node::default() }
            }
        };
        if let Some(g) = self.usable_gib {
            node.usable_bytes = (g * (1u64 << 30) as f64) as u64;
        }
        if let Some(b) = self.mem_bandwidth_gbps {
            node.mem_bandwidth_bytes_per_sec = b * 1e9;
        }
        if let Some(e) = self.bandwidth_efficiency {
            if !(0.0..=1.0).contains(&e) {
                bail!("node {:?}: bandwidth_efficiency must be in 0..=1, got {e}", self.name);
            }
            node.bandwidth_efficiency = e;
        }
        if let Some(t) = self.prefill_tflops {
            if t < 0.0 {
                bail!("node {:?}: prefill_tflops must not be negative, got {t}", self.name);
            }
            node.prefill_flops = t * 1e12;
        }
        if let Some(h) = &self.host {
            node.host = Some(h.clone());
        }
        if let Some(c) = self.contention {
            if !(0.0..=1.0).contains(&c) {
                bail!("node {:?}: contention must be in 0..=1, got {c}", self.name);
            }
            node.contention = c;
        }
        Ok(node)
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if cfg.nodes.is_empty() {
            bail!("{} defines no nodes", path.display());
        }
        Ok(cfg)
    }

    pub fn resolve_nodes(&self) -> Result<Vec<Node>> {
        self.nodes.iter().map(NodeConfig::resolve).collect()
    }

    /// Resolve an agent's pinned pool name to a node index.
    pub fn pin_index(&self, agent: &AgentConfig) -> Result<Option<usize>> {
        let Some(name) = &agent.pin else { return Ok(None) };
        let idx = self
            .nodes
            .iter()
            .position(|n| &n.name == name)
            .with_context(|| {
                format!("agent {:?} is pinned to unknown node {name:?}", agent.name)
            })?;
        Ok(Some(idx))
    }

    /// Build the network description, resolving node names to indices.
    pub fn resolve_fabric(&self) -> Result<Fabric> {
        let index_of = |name: &str| -> Result<usize> {
            self.nodes
                .iter()
                .position(|n| n.name == name)
                .with_context(|| format!("link refers to unknown node {name:?}"))
        };
        let mut fabric = Fabric::new();
        for l in &self.links {
            let [a, b] = &l.between;
            if a == b {
                bail!("link lists {a:?} on both sides");
            }
            fabric = fabric.connect(index_of(a)?, index_of(b)?, l.link.clone());
        }
        if let Some(default) = &self.link {
            fabric = fabric.with_fallback(default.clone());
        }
        Ok(fabric)
    }
}

/// A starter config for the two-machine setup this tool was built around.
pub const EXAMPLE: &str = r#"# allpaka cluster description.
#
# A "node" is one memory pool, not one machine. A desktop with a discrete GPU
# has two of them: VRAM, fast and small, and system RAM, large and dozens of
# times slower. They hold different weights and stream at different
# speeds, so they are listed separately and joined by a PCIe "link". A Mac has
# unified memory and is therefore genuinely one node.
#
# Node order does not matter. The planner tries every ordering, because with
# unequal links the choice of which machine leads the pipeline changes the
# answer.
#
# Preset values are vendor specifications and therefore optimistic. Override
# any of them once you have measured the real number.

[[nodes]]
name = "mac"
preset = "m4max-128"
# usable_gib = 96
# mem_bandwidth_gbps = 546

# The two pools below share a chassis, which is what `host` records. The
# planner then reports them as one machine (one failure domain, one CPU) rather
# than pretending three pools are three computers.
[[nodes]]
name = "pc-gpu"
preset = "rtx5090"
host = "pc"

# System RAM on the same PC. Buys capacity, not speed: layers placed here are
# computed by the CPU at DDR5 bandwidth. The planner will only use it once the
# faster pools are full - which is exactly what you want.
[[nodes]]
name = "pc-ram"
preset = "ddr4-64"
host = "pc"
# mem_bandwidth_gbps = 51.2  # MEASURE THIS - dual-channel DDR4-3200 assumed
# contention = 0.8           # bandwidth left when pc-gpu also serves; measure
# prefill_tflops = 0.4       # prompt processing is compute-bound; on a CPU it
#                            # is brutally slow, and this is where a plan's
#                            # time-to-first-token dies. Measure prompt tok/s.

# The PCIe bus between the GPU and system RAM of the same machine. It is not a
# network, but it is a boundary activations cross, so it belongs here. Latency
# is microseconds rather than milliseconds, which is why an in-box split is
# almost free compared with any cable.
[[links]]
between = ["pc-gpu", "pc-ram"]
throughput_bytes_per_sec = 50000000000.0
rtt_p50_secs = 0.000012
rtt_p99_secs = 0.000030

# Per-pair measurements from `allpaka bench`. Add one block per hop you have
# actually measured. Latency matters far more than throughput here: a decode
# step ships only ~10 KB across a cut, but pays the round trip on every token.
# [[links]]
# between = ["mac", "pc-gpu"]
# throughput_bytes_per_sec = 1100000000.0
# rtt_p50_secs = 0.00015
# rtt_p99_secs = 0.0004

# Fallback for pairs with no block above, e.g. everything reachable only over
# Wi-Fi. Leave it out entirely and an unmeasured hop blocks the split rather
# than being guessed.
# [link]
# throughput_bytes_per_sec = 40000000.0
# rtt_p50_secs = 0.004
# rtt_p99_secs = 0.030

# Agents: what actually runs where. `allpaka fleet` reads this and places each
# agent on a pool. Leave `pin` unset to let the planner choose; set it when you
# know something the cost model does not. `port` is recorded here so the routing
# table and the placement cannot drift apart.

# [[agents]]
# name = "reasoner"          # the big model, worth waiting for
# model = "models/qwen3-235b-a22b-Q2_K.gguf"
# ctx = 32768
# pin = "mac"
# port = 8081
# draft = "models/qwen3-0.6b-Q8_0.gguf"   # optional, shares this pool
# draft_len = 5
# accept = 0.75              # MEASURE THIS, do not take the default

# [[agents]]
# name = "tools"             # routing and tool calls: short, frequent, latency-bound
# model = "models/qwen3-30b-a3b-Q6_K.gguf"
# ctx = 8192
# pin = "pc-gpu"
# port = 8082

# [[agents]]
# name = "background"        # summarising and indexing: nobody is waiting
# model = "models/qwen3-30b-a3b-Q6_K.gguf"
# ctx = 8192
# pin = "pc-ram"
# port = 8083

[defaults]
context_tokens = 8192
prompt_tokens = 2048
kv_cache_dtype_bytes = 2
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_config_parses_and_resolves() {
        let cfg: Config = toml::from_str(EXAMPLE).expect("example config should parse");
        let nodes = cfg.resolve_nodes().expect("example nodes should resolve");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].name, "mac");
        assert!(nodes.iter().all(|n| n.usable_bytes > 0));
        assert!(nodes.iter().all(|n| n.mem_bandwidth_bytes_per_sec > 0.0));
        // The PCIe link between the two pools of the same PC must resolve.
        let fabric = cfg.resolve_fabric().expect("example fabric should resolve");
        assert!(fabric.between(1, 2).is_some());
    }

    /// System RAM is a capacity tier, not a speed tier. If this stops holding,
    /// the presets have drifted from reality.
    #[test]
    fn system_ram_is_far_slower_but_far_larger_than_vram() {
        let cfg: Config = toml::from_str(EXAMPLE).unwrap();
        let nodes = cfg.resolve_nodes().unwrap();
        let gpu = nodes.iter().find(|n| n.name == "pc-gpu").unwrap();
        let ram = nodes.iter().find(|n| n.name == "pc-ram").unwrap();
        assert!(ram.usable_bytes > gpu.usable_bytes);
        assert!(ram.effective_bandwidth() < gpu.effective_bandwidth() / 10.0);
    }

    #[test]
    fn explicit_fields_override_the_preset() {
        let nc = NodeConfig {
            name: "mac".into(),
            preset: Some("m4max-128".into()),
            usable_gib: Some(110.0),
            mem_bandwidth_gbps: None,
            bandwidth_efficiency: Some(0.62),
            host: None,
            contention: None,
            prefill_tflops: None,
        };
        let node = nc.resolve().unwrap();
        assert_eq!(node.usable_bytes, (110.0 * (1u64 << 30) as f64) as u64);
        assert_eq!(node.bandwidth_efficiency, 0.62);
        // Untouched field still comes from the preset.
        assert_eq!(node.mem_bandwidth_bytes_per_sec, 546e9);
    }

    #[test]
    fn an_unknown_preset_is_rejected() {
        let nc = NodeConfig {
            name: "x".into(),
            preset: Some("m5ultra".into()),
            usable_gib: None,
            mem_bandwidth_gbps: None,
            bandwidth_efficiency: None,
            host: None,
            contention: None,
            prefill_tflops: None,
        };
        assert!(nc.resolve().is_err());
    }

    #[test]
    fn a_node_without_preset_or_numbers_is_rejected() {
        let nc = NodeConfig {
            name: "mystery".into(),
            preset: None,
            usable_gib: None,
            mem_bandwidth_gbps: None,
            bandwidth_efficiency: None,
            host: None,
            contention: None,
            prefill_tflops: None,
        };
        assert!(nc.resolve().is_err());
    }

    fn with_links(body: &str) -> Config {
        let text = format!(
            "[[nodes]]\nname = \"mac\"\npreset = \"m4max-128\"\n\
             [[nodes]]\nname = \"pc\"\npreset = \"rtx5090\"\n{body}"
        );
        toml::from_str(&text).expect("config should parse")
    }

    #[test]
    fn a_named_pair_resolves_to_a_link() {
        let cfg = with_links(
            "[[links]]\nbetween = [\"mac\", \"pc\"]\n\
             throughput_bytes_per_sec = 1100000000.0\n\
             rtt_p50_secs = 0.00015\nrtt_p99_secs = 0.0004\n",
        );
        let fabric = cfg.resolve_fabric().unwrap();
        assert_eq!(fabric.between(0, 1).unwrap().rtt_p99_secs, 0.0004);
    }

    #[test]
    fn a_link_to_an_unknown_node_is_rejected() {
        let cfg = with_links(
            "[[links]]\nbetween = [\"mac\", \"nas\"]\n\
             throughput_bytes_per_sec = 1.0\nrtt_p50_secs = 1.0\nrtt_p99_secs = 1.0\n",
        );
        let err = cfg.resolve_fabric().unwrap_err().to_string();
        assert!(err.contains("nas"), "unhelpful error: {err}");
    }

    #[test]
    fn a_link_from_a_node_to_itself_is_rejected() {
        let cfg = with_links(
            "[[links]]\nbetween = [\"mac\", \"mac\"]\n\
             throughput_bytes_per_sec = 1.0\nrtt_p50_secs = 1.0\nrtt_p99_secs = 1.0\n",
        );
        assert!(cfg.resolve_fabric().is_err());
    }

    #[test]
    fn a_config_with_no_network_at_all_yields_an_empty_fabric() {
        let cfg = with_links("");
        assert!(cfg.resolve_fabric().unwrap().is_empty());
    }

    #[test]
    fn out_of_range_efficiency_is_rejected() {
        let nc = NodeConfig {
            name: "mac".into(),
            preset: Some("m4max-128".into()),
            usable_gib: None,
            mem_bandwidth_gbps: None,
            bandwidth_efficiency: Some(1.4),
            host: None,
            contention: None,
            prefill_tflops: None,
        };
        assert!(nc.resolve().is_err());
    }
}
