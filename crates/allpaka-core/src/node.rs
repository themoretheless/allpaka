//! Description of a machine that can hold model weights and run layers.

use serde::{Deserialize, Serialize};

/// Compute backend a node uses for GGML/GGUF-style inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// NVIDIA CUDA. Weights live in dedicated VRAM.
    Cuda,
    /// Apple Metal. Weights live in unified memory shared with the CPU.
    Metal,
    /// CPU only. Included so a node can be described honestly even if it is slow.
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }
}

/// A machine available to hold part of a model.
///
/// `usable_bytes` is the amount that may actually be filled with weights and KV
/// cache, not the sticker capacity. On a 32 GB RTX 5090 roughly 30 GB is
/// reachable; on a 128 GB Mac the default `iogpu.wired_limit_pct` caps GPU-wired
/// memory well below 128 GB, so the honest number is lower than the spec sheet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub backend: Backend,
    /// Memory available for weights + KV cache, in bytes.
    pub usable_bytes: u64,
    /// Peak theoretical memory bandwidth in bytes/sec.
    ///
    /// Decode of a dense transformer is memory-bandwidth bound, so this number
    /// predicts token rate far better than FLOPS does.
    pub mem_bandwidth_bytes_per_sec: f64,
    /// Fraction of peak bandwidth actually achieved by a real kernel.
    /// Measured llama.cpp decode typically lands around 0.6-0.8 of peak.
    #[serde(default = "default_bw_efficiency")]
    pub bandwidth_efficiency: f64,
    /// Physical machine this pool belongs to.
    ///
    /// A pool is not a machine. A desktop with a discrete GPU has two pools -
    /// VRAM and system RAM - that share one chassis, one CPU, one PCIe root and
    /// one power supply. Naming the host here is what lets the planner stop
    /// calling them independent. Unset means the pool is its own machine.
    #[serde(default)]
    pub host: Option<String>,
    /// Fraction of this pool's bandwidth still available while another pool on
    /// the same host is also serving.
    ///
    /// 1.0 is the default and is roughly right for *decode*: a CUDA server
    /// streams weights from VRAM and a CPU server streams them from DDR, and in
    /// steady state neither touches the other's memory. It is optimistic for
    /// prompt processing, where the GPU pulls the prompt across PCIe out of the
    /// same DDR the CPU server is saturating. Measure it and write the number
    /// down rather than trusting this default.
    #[serde(default = "default_contention")]
    pub contention: f64,
    /// Effective FLOP/s this pool achieves during prompt processing.
    ///
    /// Decode is bandwidth-bound, but prefill is compute-bound: the whole
    /// prompt is one big batched matmul, and weights are read once however
    /// long it is. This is the *achieved* number, not the spec-sheet peak -
    /// real prefill lands at a fraction of peak that varies wildly by backend,
    /// so there is no separate efficiency knob to misjudge; measure tokens/sec
    /// of prompt processing and multiply by `2 * active params`.
    ///
    /// 0 means unmodeled, and time-to-first-token is then reported as unknown
    /// rather than invented.
    #[serde(default)]
    pub prefill_flops: f64,
}

fn default_bw_efficiency() -> f64 {
    0.7
}

fn default_contention() -> f64 {
    1.0
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: String::new(),
            backend: Backend::Cpu,
            usable_bytes: 0,
            mem_bandwidth_bytes_per_sec: 0.0,
            bandwidth_efficiency: default_bw_efficiency(),
            host: None,
            contention: default_contention(),
            prefill_flops: 0.0,
        }
    }
}

impl Node {
    /// Physical machine this pool sits in. A pool with no declared host is
    /// treated as its own machine, which is the safe reading of silence: it
    /// claims no sharing that was not stated.
    pub fn host(&self) -> &str {
        self.host.as_deref().unwrap_or(&self.name)
    }

    /// Whether two pools are the same piece of hardware.
    pub fn shares_host_with(&self, other: &Node) -> bool {
        self.host() == other.host()
    }

    /// Bandwidth we actually expect to see, in bytes/sec.
    pub fn effective_bandwidth(&self) -> f64 {
        self.mem_bandwidth_bytes_per_sec * self.bandwidth_efficiency
    }

    /// Seconds to stream `bytes` of weights once, i.e. one decode step over the
    /// layers resident here.
    pub fn stream_time(&self, bytes: u64) -> f64 {
        if self.effective_bandwidth() <= 0.0 {
            return f64::INFINITY;
        }
        bytes as f64 / self.effective_bandwidth()
    }

    /// Seconds to stream `bytes` while another pool on the same host is also
    /// serving. Equal to [`Node::stream_time`] when `contention` is 1.0.
    pub fn contended_stream_time(&self, bytes: u64) -> f64 {
        let bw = self.effective_bandwidth() * self.contention.clamp(0.0, 1.0);
        if bw <= 0.0 {
            return f64::INFINITY;
        }
        bytes as f64 / bw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(name: &str, host: Option<&str>) -> Node {
        Node {
            name: name.into(),
            host: host.map(str::to_string),
            mem_bandwidth_bytes_per_sec: 100e9,
            bandwidth_efficiency: 1.0,
            ..Node::default()
        }
    }

    /// Silence must not be read as a claim of sharing.
    #[test]
    fn a_pool_with_no_declared_host_is_its_own_machine() {
        let a = pool("mac", None);
        let b = pool("pc-gpu", None);
        assert_eq!(a.host(), "mac");
        assert!(!a.shares_host_with(&b));
    }

    /// The whole point: two pools of one desktop are one machine.
    #[test]
    fn two_pools_naming_the_same_host_are_one_machine() {
        let gpu = pool("pc-gpu", Some("pc"));
        let ram = pool("pc-ram", Some("pc"));
        assert!(gpu.shares_host_with(&ram));
        assert!(!gpu.shares_host_with(&pool("mac", None)));
    }

    #[test]
    fn contention_of_one_leaves_streaming_untouched() {
        let n = pool("x", None);
        assert_eq!(n.contended_stream_time(1 << 30), n.stream_time(1 << 30));
    }

    #[test]
    fn a_measured_contention_slows_a_co_resident_pool() {
        let mut n = pool("x", None);
        n.contention = 0.5;
        assert!((n.contended_stream_time(1 << 30) - 2.0 * n.stream_time(1 << 30)).abs() < 1e-12);
    }
}
