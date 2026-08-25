//! Starting points for hardware description.
//!
//! These are vendor specifications, and specifications are optimistic. Treat
//! them as a first guess to be replaced by measurement: `allpaka bench` reports
//! what a node actually achieves, and the config file is where you record it.

use crate::{Backend, Node};

/// A named hardware preset with the reasoning behind its numbers.
pub struct Preset {
    pub id: &'static str,
    pub description: &'static str,
    pub backend: Backend,
    pub usable_bytes: u64,
    pub mem_bandwidth_bytes_per_sec: f64,
    /// Effective FLOP/s achieved during prompt processing. Even rougher than
    /// the bandwidth numbers: achieved prefill varies severalfold with backend
    /// and quantisation, so these are order-of-magnitude starting points.
    /// Measure prompt tokens/sec and back the real figure out.
    pub prefill_flops: f64,
    /// Why `usable_bytes` is lower than the sticker capacity.
    pub note: &'static str,
}

impl Preset {
    pub fn to_node(&self, name: &str) -> Node {
        Node {
            name: name.to_string(),
            backend: self.backend,
            usable_bytes: self.usable_bytes,
            mem_bandwidth_bytes_per_sec: self.mem_bandwidth_bytes_per_sec,
            // CPU inference reaches a smaller fraction of peak bandwidth than a
            // GPU does: fewer outstanding memory requests, and the cores spend
            // real time on dequantisation.
            bandwidth_efficiency: match self.backend {
                Backend::Cpu => 0.6,
                _ => 0.7,
            },
            prefill_flops: self.prefill_flops,
            // A preset describes a piece of hardware, not where it is plugged
            // in. Which pools share a chassis is a fact about this particular
            // desk, so it comes from the config.
            ..Node::default()
        }
    }
}

pub const PRESETS: &[Preset] = &[
    Preset {
        id: "rtx5090",
        description: "NVIDIA GeForce RTX 5090, 32 GB GDDR7",
        backend: Backend::Cuda,
        // 30 GiB of 32: the display, the CUDA context and fragmentation take
        // the rest. Overcommitting VRAM does not spill gracefully, it fails.
        usable_bytes: 30 << 30,
        // 512-bit bus at 28 Gbps GDDR7.
        mem_bandwidth_bytes_per_sec: 1792e9,
        // Tensor-core prefill in llama.cpp lands in the low hundreds of TFLOP/s.
        prefill_flops: 150e12,
        note: "2 GB held back for display output and CUDA context",
    },
    Preset {
        id: "m4max-128",
        description: "Apple M4 Max, 128 GB unified memory",
        backend: Backend::Metal,
        // macOS will not wire the full 128 GB to the GPU by default. The
        // iogpu.wired_limit_pct sysctl raises this, but leaving headroom for
        // the OS is what keeps the machine responsive instead of swapping.
        usable_bytes: 96 << 30,
        mem_bandwidth_bytes_per_sec: 546e9,
        // Metal prefill achieves a modest slice of the GPU's ~34 TFLOP/s fp16.
        prefill_flops: 16e12,
        note: "raise with `sudo sysctl iogpu.wired_limit_pct=85` if you need more",
    },
    Preset {
        id: "ddr4-64",
        description: "Desktop system RAM, 64 GB dual-channel DDR4",
        backend: Backend::Cpu,
        // 56 GiB of 64: Windows, the page cache and the CUDA host allocations
        // take the rest. Filling system RAM to the brim makes the machine swap,
        // which is far worse than not using the last few gigabytes.
        usable_bytes: 56 << 30,
        // Two 64-bit channels at DDR4-3200: 2 x 8 bytes x 3200 MT/s.
        // MEASURE THIS. DDR4-2400 and DDR4-3600 differ by 50%, and a
        // single-channel configuration halves it again.
        mem_bandwidth_bytes_per_sec: 51.2e9,
        // CPU prefill with AVX2: well under a TFLOP/s achieved. This is where
        // time-to-first-token dies; the planner must see that.
        prefill_flops: 0.4e12,
        note: "roughly 40x slower than the 5090 next to it - capacity, not speed",
    },
    Preset {
        id: "ddr5-64",
        description: "Desktop system RAM, 64 GB dual-channel DDR5",
        backend: Backend::Cpu,
        usable_bytes: 56 << 30,
        // Two 64-bit channels at DDR5-6000.
        mem_bandwidth_bytes_per_sec: 96e9,
        prefill_flops: 0.5e12,
        note: "roughly 22x slower than the 5090 - still capacity, not speed",
    },
    Preset {
        id: "m4pro-64",
        description: "Apple M4 Pro, 64 GB unified memory",
        backend: Backend::Metal,
        usable_bytes: 48 << 30,
        mem_bandwidth_bytes_per_sec: 273e9,
        prefill_flops: 8e12,
        note: "16 GB left to the OS and page cache",
    },
];

pub fn find(id: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.id == id)
}
