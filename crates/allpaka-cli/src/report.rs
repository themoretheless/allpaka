//! Human-readable output.
//!
//! The reports say what they are confident about and mark what they are not.
//! An estimate presented without its assumptions is worse than no estimate.

use allpaka_gguf::GgufInfo;
use allpaka_core::presets::Preset;
use allpaka_core::fleet::FleetMember;
use allpaka_core::{gib, Fabric, FleetPlan, Link, Model, Node, Plan, PlanRequest, ReplicaPlan, Verdict};
use std::path::Path;

pub fn presets(list: &[Preset]) {
    println!("built-in hardware presets\n");
    for p in list {
        println!("  {:<12} {}", p.id, p.description);
        println!(
            "  {:<12} usable {:.0} GiB, memory bandwidth {:.0} GB/s",
            "",
            gib(p.usable_bytes),
            p.mem_bandwidth_bytes_per_sec / 1e9
        );
        println!("  {:<12} {}\n", "", p.note);
    }
    println!("These are vendor numbers. Override them in allpaka.toml once measured.");
}

pub fn gguf(path: &Path, info: &GgufInfo) {
    println!("{}", path.display());
    println!("  architecture      {}", info.architecture);
    println!("  layers            {}", info.block_count);
    println!("  hidden size       {}", info.embedding_length);
    println!("  kv heads          {}", info.head_count_kv);
    println!("  head dims         k={} v={}", info.key_length, info.value_length);
    println!("  weights on disk   {:.1} GiB", gib(info.file_bytes));
    println!("  parameters        {:.1}B", info.param_count as f64 / 1e9);
    println!(
        "  kv cache          {:.1} MiB per 1k tokens at f16",
        info.kv_bytes_per_token_per_layer(2) as f64 * info.block_count as f64 * 1000.0
            / (1024.0 * 1024.0)
    );
    if info.expert_count > 0 {
        println!(
            "  experts           {} per layer, {} used per token",
            info.expert_count, info.expert_used_count
        );
        println!(
            "  active weights    {:.0}% of the file per token ({:.1} GiB streamed of {:.1} GiB resident)",
            info.active_weight_fraction() * 100.0,
            gib((info.file_bytes as f64 * info.active_weight_fraction()) as u64),
            gib(info.file_bytes),
        );
        println!(
            "\n  Memory is charged on all {:.0} GiB; decode speed only on the active share.\n  \
             That is what lets a model this large run on memory this slow.",
            gib(info.file_bytes)
        );
    }
    println!(
        "\n  A pipeline cut moves {} bytes per token (hidden size x 2 bytes).",
        info.embedding_length * 2
    );
}

pub fn link(l: &Link) {
    println!("\nmeasured link");
    println!("  throughput        {:.1} MB/s", l.throughput_bytes_per_sec / 1e6);
    println!("  round trip p50    {:.2} ms", l.rtt_p50_secs * 1e3);
    println!("  round trip p99    {:.2} ms", l.rtt_p99_secs * 1e3);

    let jitter = if l.rtt_p50_secs > 0.0 { l.rtt_p99_secs / l.rtt_p50_secs } else { f64::NAN };
    println!("  tail / median     {jitter:.1}x");

    // A cut costs at least one round trip per token, so the tail latency puts a
    // hard ceiling on tokens per second regardless of how fast the GPUs are.
    if l.rtt_p99_secs > 0.0 {
        println!(
            "\n  A single pipeline cut over this link caps decode at about {:.0} tok/s,\n  \
             before any compute time is counted.",
            1.0 / l.rtt_p99_secs
        );
    }
    if jitter > 4.0 {
        println!(
            "  The tail is {jitter:.0}x the median, which is the signature of a shared or\n  \
             wireless medium. Per-token stalls will be visible."
        );
    }

    println!("\nadd this to allpaka.toml, with the two node names filled in:\n");
    println!("[[links]]");
    println!("between = [\"nodeA\", \"nodeB\"]");
    println!("throughput_bytes_per_sec = {:.0}", l.throughput_bytes_per_sec);
    println!("rtt_p50_secs = {:.6}", l.rtt_p50_secs);
    println!("rtt_p99_secs = {:.6}", l.rtt_p99_secs);
    println!(
        "\nMeasure every pair separately. A 10 GbE hop and a Wi-Fi hop differ by\n\
         roughly a hundredfold in both directions, and one averaged number would\n\
         pick cuts that are wrong on both."
    );
}

pub fn missing_link(nodes: &[Node], model: &Model, req: &PlanRequest) {
    header(model, req);
    println!("\nno network measurements in the config, so no split can be estimated.\n");
    println!("capacity check at {} token context:", req.context_tokens);
    for n in nodes {
        let need = model.total_weight_bytes + model.kv_bytes(model.n_layers, req.context_tokens);
        let fits = if need <= n.usable_bytes { "fits" } else { "DOES NOT FIT" };
        println!(
            "  {:<10} {:>7.1} GiB usable, needs {:>7.1} GiB  {}",
            n.name,
            gib(n.usable_bytes),
            gib(need),
            fits
        );
    }
    println!("\nrun `allpaka bench` on both machines and record the result to plan a split.");
}

pub fn verdict(
    nodes: &[Node],
    model: &Model,
    req: &PlanRequest,
    fabric: &Fabric,
    v: &Verdict,
) {
    header(model, req);
    fabric_summary(nodes, fabric);

    match v {
        Verdict::SplitWins { plan, single_node } => {
            println!("\nverdict: split it. The link is fast enough to pay for itself.\n");
            placement(plan);
            let speedup = single_node.secs_per_token() / plan.secs_per_token();
            println!("\nthe best single machine, for comparison:\n");
            placement(single_node);
            println!(
                "\n  Splitting is {speedup:.2}x faster. Each machine reads only its own\n  \
                 layers through its own memory, so the compute drops from {:.0} to {:.0} ms.\n  \
                 The link adds {:.1} ms per token, which is {:.0}% of the total.",
                single_node.compute_secs_per_token * 1e3,
                plan.compute_secs_per_token * 1e3,
                plan.network_secs_per_token * 1e3,
                plan.network_overhead_fraction() * 100.0,
            );
        }
        Verdict::UseSingleNode { plan, best_split } => {
            println!("\nverdict: run it on one machine.\n");
            placement(plan);
            if let Some(split) = best_split {
                println!("\nthe best split was considered and rejected:\n");
                placement(split);
                let ratio = split.secs_per_token() / plan.secs_per_token();
                println!(
                    "\n  Splitting cuts compute from {:.0} to {:.0} ms per token, but adds\n  \
                     {:.1} ms of network wait, for a net {ratio:.2}x slowdown. The round trip\n  \
                     is paid on every token and never amortised.",
                    plan.compute_secs_per_token * 1e3,
                    split.compute_secs_per_token * 1e3,
                    split.network_secs_per_token * 1e3,
                );
                suggest_better_link(split, plan);
            } else {
                println!("\nno feasible split exists, but none is needed.");
            }
        }
        Verdict::SplitRequired { plan } => {
            println!("\nverdict: split required. No single machine holds this model.\n");
            placement(plan);
            println!(
                "\n  {:.0}% of each token is network wait.",
                plan.network_overhead_fraction() * 100.0
            );
            if plan.network_overhead_fraction() > 0.5 {
                println!(
                    "  Over half the time is the link. A smaller quantisation that fits one\n  \
                     machine will almost certainly be faster."
                );
            }
        }
        Verdict::Infeasible { reason } => {
            println!("\nverdict: infeasible.\n  {reason}");
            println!(
                "\n  Options: a smaller quantisation, a shorter context, or a quantised KV\n  \
                 cache (kv_cache_dtype_bytes = 1)."
            );
        }
    }
}

/// How much of the gap a better link would close, when the split lost only
/// because of the network.
fn suggest_better_link(split: &Plan, single: &Plan) {
    if split.compute_secs_per_token >= single.secs_per_token() {
        // Compute alone already loses; the network is not the problem.
        return;
    }
    let ceiling = single.secs_per_token() / split.compute_secs_per_token;
    println!(
        "\n  On compute alone this split would be {ceiling:.2}x faster. That is what a\n  \
         low-latency link between these two machines would buy. Measure the 10 GbE\n  \
         path with `allpaka bench` and record it as a [[links]] entry."
    );
}

fn fabric_summary(nodes: &[Node], fabric: &Fabric) {
    println!("\nnetwork:");
    for (a, b, l) in fabric.edges() {
        println!(
            "  {:<16} {:>6.0} MB/s, round trip p99 {:>6.2} ms",
            format!("{} <-> {}", nodes[a].name, nodes[b].name),
            l.throughput_bytes_per_sec / 1e6,
            l.rtt_p99_secs * 1e3,
        );
    }
    if let Some(l) = fabric.fallback() {
        println!(
            "  {:<16} {:>6.0} MB/s, round trip p99 {:>6.2} ms   (assumed for every\n  \
             {:<16} unmeasured pair - measure them instead)",
            "everything else",
            l.throughput_bytes_per_sec / 1e6,
            l.rtt_p99_secs * 1e3,
            "",
        );
    }
}

fn header(model: &Model, req: &PlanRequest) {
    println!("model: {}", model.name);
    println!(
        "  {} layers, hidden {}, {:.1} GiB of weights",
        model.n_layers,
        model.hidden_size,
        gib(model.total_weight_bytes)
    );
    if model.is_sparse() {
        println!(
            "  mixture of experts: {:.0}% of weights read per token, but all of them resident",
            model.active_weight_fraction * 100.0
        );
    }
    println!("  budgeting {} token context, {} token prompt", req.context_tokens, req.prompt_tokens);
}

fn placement(p: &Plan) {
    for s in &p.stages {
        println!(
            "  {:<10} layers {:>3}-{:<3}  {:>6.1} GiB weights + {:>5.1} GiB kv  {:>6.1} ms/token",
            s.node_name,
            s.first_layer,
            s.first_layer + s.layer_count.saturating_sub(1),
            gib(s.weight_bytes),
            gib(s.kv_bytes),
            s.compute_secs * 1e3,
        );
    }
    println!(
        "  {:<10} compute {:.1} ms + network {:.1} ms = {:.1} ms/pass",
        "per pass",
        p.compute_secs_per_token * 1e3,
        p.network_secs_per_token * 1e3,
        p.plain_secs_per_token() * 1e3,
    );
    match &p.speculation {
        None => println!(
            "  {:<10} {:.1} ms/token  ({:.1} tok/s)",
            "total",
            p.secs_per_token() * 1e3,
            p.tokens_per_sec()
        ),
        Some(s) => {
            println!(
                "  {:<10} draft {:.1} ms + verify {:.1} ms = {:.1} ms/cycle, {:.2} tokens accepted",
                "speculate",
                s.draft_secs_per_cycle * 1e3,
                (s.verify_compute_secs + s.verify_network_secs) * 1e3,
                s.cycle_secs() * 1e3,
                s.expected_accepted,
            );
            println!(
                "  {:<10} {:.1} ms/token  ({:.1} tok/s), network {:.2} ms/token",
                "total",
                p.secs_per_token() * 1e3,
                p.tokens_per_sec(),
                p.network_secs_per_token_effective() * 1e3,
            );
        }
    }
    if p.is_split() {
        println!(
            "  {:<10} plus {:.2} s to ship the prompt across the cut",
            "",
            p.prompt_transfer_secs
        );
    }
    match p.ttft_secs() {
        Some(t) => {
            println!("  {:<10} {}", "first token", ttft(t));
            // Decode looking fine while the first token is a coffee break is
            // the failure mode layers-on-CPU invites; flag it in words.
            if t > 30.0 {
                println!(
                    "  {:<10} Prefill dominates this plan: prompt processing runs at CPU\n  \
                     {:<10} speed on the spilled layers. Decode tok/s alone flatters it.",
                    "", ""
                );
            }
        }
        None => println!(
            "  {:<10} unknown - set prefill_tflops on every node to estimate it",
            "first token"
        ),
    }
}

/// Format a time-to-first-token humanely: ms when fast, minutes when not.
fn ttft(secs: f64) -> String {
    if secs < 1.0 {
        format!("~{:.0} ms after the prompt", secs * 1e3)
    } else if secs < 120.0 {
        format!("~{secs:.1} s after the prompt")
    } else {
        format!("~{:.1} min after the prompt", secs / 60.0)
    }
}

/// Replication: independent instances, no coupling between machines at all.
///
/// Reported alongside the split rather than instead of it, because the two
/// answer different questions - one request as fast as possible, or as many
/// requests at once as possible.
pub fn replication(rp: Option<&ReplicaPlan>, split: &Plan) {
    println!("\nisolation option: one independent instance per machine\n");
    let Some(rp) = rp else {
        println!("  Not available: no single machine holds the whole model.");
        println!("  There is nothing to replicate; splitting is the only arrangement left.");
        return;
    };

    for r in &rp.replicas {
        let mark = if !rp.is_useful(r) {
            "  <- too slow to route to"
        } else if r.co_resident {
            "  <- shares its machine"
        } else {
            ""
        };
        let first = match r.ttft_secs {
            Some(t) => format!(", first token {}", ttft(t)),
            None => String::new(),
        };
        println!(
            "  {:<10} own instance, {:>6.1} GiB + {:>4.1} GiB kv  {:>7.1} ms/token  ({:>5.1} tok/s){}{}",
            r.node_name,
            gib(r.weight_bytes),
            gib(r.kv_bytes),
            r.secs_per_token * 1e3,
            r.tokens_per_sec(),
            first,
            mark,
        );
    }
    println!(
        "  {:<10} {} usable instances on {} machine{}, {:.1} tok/s aggregate, 0 bytes on the wire",
        "total",
        rp.useful_concurrency(),
        rp.machines(),
        if rp.machines() == 1 { "" } else { "s" },
        rp.useful_tokens_per_sec(),
    );

    println!();
    println!(
        "  Against the split at {:.1} tok/s for one request:",
        split.tokens_per_sec()
    );
    println!(
        "    aggregate throughput   {:.2}x",
        rp.useful_tokens_per_sec() / split.tokens_per_sec()
    );
    println!(
        "    single-request latency {:.2}x",
        split.secs_per_token() / rp.best_secs_per_token()
    );
    if rp.best_secs_per_token() > split.secs_per_token() {
        println!("  One waiting user is served faster by the split.");
    }
    println!("  Several users, or an agent loop issuing parallel calls, prefer replication.");
    if rp.replicas.iter().any(|r| r.co_resident) {
        println!();
        println!("  Instances marked above share one physical machine: its CPU, PCIe and");
        println!("  power budget. Their speeds include the configured contention factor.");
    }
}

/// Fleet placement: one model per pool, one endpoint per agent.
pub fn fleet(
    members: &[FleetMember],
    nodes: &[Node],
    plan: &FleetPlan,
    ports: &[Option<u16>],
) {
    println!("fleet placement: one agent per memory pool\n");

    if plan.placements.is_empty() {
        println!("  Nothing could be placed. Every agent is larger than every pool.");
        return;
    }

    for p in &plan.placements {
        let node = &nodes[p.node_index];
        let m = &members[p.model_index];
        let how = if p.pinned { "pinned" } else { "chosen" };
        println!(
            "  {:<12} on {:<10} ({})",
            p.model_name, p.node_name, how
        );
        println!(
            "  {:<12}    {:>6.1} GiB weights + {:>4.1} GiB kv at ctx {}  of {:.1} GiB usable",
            "",
            gib(p.weight_bytes),
            gib(p.kv_bytes),
            p.context_tokens,
            gib(node.usable_bytes),
        );
        let sparse = if m.model.is_sparse() {
            format!("  ·  MoE, {:.0}% active", m.model.active_weight_fraction * 100.0)
        } else {
            String::new()
        };
        let draft = if m.speculation.is_some() { "  ·  speculating" } else { "" };
        let shared = if p.co_resident {
            format!("  ·  shares machine \"{}\"", p.host)
        } else {
            String::new()
        };
        println!(
            "  {:<12}    {:>6.1} ms/token  ({:.1} tok/s){}{}{}",
            "",
            p.secs_per_token * 1e3,
            p.tokens_per_sec(),
            sparse,
            draft,
            shared,
        );
        match p.ttft_secs {
            Some(t) => println!("  {:<12}    first token {}", "", ttft(t)),
            None => println!(
                "  {:<12}    first token unknown - set prefill_tflops on this node",
                ""
            ),
        }
        println!();
    }

    println!(
        "  {} endpoints on {} machine{}, {:.1} tok/s aggregate, 0 bytes on the wire",
        plan.endpoints(),
        plan.machines(),
        if plan.machines() == 1 { "" } else { "s" },
        plan.aggregate_tokens_per_sec(),
    );

    if !plan.unplaced.is_empty() {
        println!();
        for &i in &plan.unplaced {
            let m = &members[i];
            println!(
                "  UNPLACED {:<14} needs {:.1} GiB at ctx {}; no free pool is that large",
                m.model.name,
                gib(allpaka_core::fleet::required_bytes(m)),
                m.context_tokens,
            );
        }
        println!("  Options: a smaller quantisation, a shorter ctx, or fewer agents.");
    }

    // The routing table is the point of writing ports down: placement and
    // addressing come from one file, so they cannot drift apart.
    if ports.iter().any(Option::is_some) {
        println!("\nrouting:");
        for p in &plan.placements {
            match ports.get(p.model_index).copied().flatten() {
                Some(port) => println!(
                    "  {:<12} http://{}:{}/v1",
                    p.model_name,
                    host_hint(&p.node_name),
                    port
                ),
                None => println!("  {:<12} no port set in the config", p.model_name),
            }
        }
    }

    let co: Vec<&str> =
        plan.co_resident().map(|p| p.model_name.as_str()).collect();
    if !co.is_empty() {
        println!();
        println!(
            "  {} share one physical machine: one reboot takes them all down,",
            co.join(" and ")
        );
        println!("  and they compete for its CPU. Their speeds above already include the");
        println!("  configured contention factor; measure it under real load.");
    }
}

/// A placeholder host for the routing table. The config knows pool names, not
/// hostnames, so this is a hint to be edited rather than a resolved address.
fn host_hint(node_name: &str) -> String {
    format!("<{node_name}>")
}

/// Print a ready llama-server command for every placed agent, grouped by the
/// physical machine that has to run it.
pub fn launch(cfg: &crate::config::Config, nodes: &[allpaka_core::Node], plan: &FleetPlan) {
    use allpaka_core::Backend;

    println!("launch commands, from the same [[agents]] the planner placed\n");

    let mut hosts: Vec<&str> = plan.placements.iter().map(|p| p.host.as_str()).collect();
    hosts.sort_unstable();
    hosts.dedup();

    for host in hosts {
        println!("on machine \"{host}\":");
        for p in plan.placements.iter().filter(|p| p.host == host) {
            let agent = &cfg.agents[p.model_index];
            let node = &nodes[p.node_index];
            let mut cmd = format!(
                "llama-server -m {} -c {} --port {} --host 0.0.0.0 -a {}",
                agent.model.display(),
                p.context_tokens,
                agent.port.map_or_else(|| "<port>".into(), |v| v.to_string()),
                agent.name,
            );
            match node.backend {
                // Every layer on the accelerator; that is what the plan priced.
                Backend::Cuda | Backend::Metal => cmd.push_str(" -ngl 999"),
                // The CPU pool must NOT touch the GPU, or it silently competes
                // with the agent that owns it.
                Backend::Cpu => cmd.push_str(" -ngl 0"),
            }
            if let Some(d) = &agent.draft {
                cmd.push_str(&format!(" -md {} --draft-max {}", d.display(), agent.draft_len));
            }
            println!("  # {} - {:.1} tok/s expected on {}", agent.name, p.tokens_per_sec(), p.node_name);
            println!("  {cmd}\n");
        }
    }

    if !plan.unplaced.is_empty() {
        println!("not launched (did not fit anywhere): check `allpaka fleet` for details.");
    }

    println!("Windows: same commands with llama-server.exe from a CUDA build of llama.cpp.");
    println!("Model files must be real weights; a placeholder plans fine but will not load.");
}
