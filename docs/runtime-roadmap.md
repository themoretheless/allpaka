# Runtime roadmap

Each implementation increment is tested, committed, and pushed before the next.

| Order | Outcome | Status |
| --- | --- | --- |
| 1 | Comparable benchmarks with exact token replay, KV precision and matched contexts | In progress: workload metadata, strict GPU coverage and alternating context-matched runs; llama token replay pending |
| 2 | Isolated sessions, cancellation, deadlines and explicit backpressure | Pending |
| 3 | Shared memory budget covering weights, KV, scratch and pinned cache leases | Pending |
| 4 | Reusable KV block prefix cache, hit/miss and reused-token telemetry | Pending |
| 5 | Step scheduler and chunked prefill with bounded decode latency | Pending |
| 6 | GPU kernels batching independent sessions | Pending |
| 7 | Workload-aware autotune for latency, throughput and memory | Pending |
| 8 | Structured output and validated tool calls | Pending |

The historical September 2 ratios are observations from different token streams
and context setups. They are not a controlled proof of the 1.10x target. The
comparison harness preserves artifacts and marks comparability explicitly.
For MoE models, identical tensor shapes do not imply identical expert routing.

Current serving batches admission only. Multi-session GPU batching remains an
implementation item, not a completed capability.
