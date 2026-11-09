//! Network link and host RAM bandwidth microbenchmarks.
//!
//! Two network characteristics are measured separately because they cost
//! differently in pipeline inference:
//! * **Round-trip latency**, paid once per stage boundary per generated token.
//! * **Throughput**, paid when shipping batched prompt activations.

use allpaka_core::Link;
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Instant;

const OP_PING: u8 = 0;
const OP_SINK: u8 = 1;

/// Payload for latency probes. Small enough that serialisation time is
/// negligible, so what is measured is genuinely latency.
const PING_BYTES: usize = 64;
const PING_ROUNDS: usize = 200;
/// Discard the first few round trips: TCP slow start and the Wi-Fi radio
/// waking from power save would otherwise be reported as steady-state latency.
const PING_WARMUP: usize = 20;

const CHUNK_BYTES: usize = 4 << 20;
const THROUGHPUT_BYTES: usize = 64 << 20;

/// Big enough to defeat every cache level; small enough to allocate anywhere.
const MEM_BUFFER_BYTES: usize = 1 << 30;
const MEM_PASSES: usize = 5;

/// Measure how fast this machine's CPU can stream from its own RAM.
///
/// This is the number the `pc-ram` pool runs at - decode on CPU is a straight
/// read of the weights - and the one preset most likely to be wrong: DDR4-2400
/// vs 3600 differ by 50%, and a single-channel machine halves it again. GPU
/// pools cannot be measured from here; use vendor numbers or llama-bench for
/// those.
///
/// All cores read disjoint slices at once, because one core cannot saturate a
/// memory controller; the sum approximates what a multi-threaded inference
/// runtime actually gets.
pub fn measure_memory() -> Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "streaming a {} MiB buffer with {threads} threads ...",
        MEM_BUFFER_BYTES >> 20
    );

    // u64 words, initialised so the pages are really committed - reading
    // untouched zero pages measures the kernel, not the RAM.
    let words = MEM_BUFFER_BYTES / 8;
    let buf: Vec<u64> = (0..words as u64).collect();

    let mut best_bytes_per_sec = 0.0f64;
    for _ in 0..MEM_PASSES {
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for chunk in buf.chunks(words.div_ceil(threads)) {
                scope.spawn(move || {
                    let mut acc = 0u64;
                    for w in chunk {
                        acc = acc.wrapping_add(*w);
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        let rate = MEM_BUFFER_BYTES as f64 / t0.elapsed().as_secs_f64();
        best_bytes_per_sec = best_bytes_per_sec.max(rate);
    }

    let gbps = best_bytes_per_sec / 1e9;
    println!("\nread bandwidth: {gbps:.1} GB/s (best of {MEM_PASSES} passes)");
    println!("\nfor the CPU-RAM pool of this machine, set in allpaka.toml:");
    println!("mem_bandwidth_gbps = {gbps:.1}");
    println!(
        "\nThis measures CPU-visible RAM only. For a GPU pool keep the preset's\n\
         vendor number, or back one out of measured llama-bench decode tok/s."
    );
    Ok(())
}

/// Serve bench requests until the peer disconnects. Runs on the remote machine.
pub fn serve(bind: &str) -> Result<()> {
    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
    println!(
        "allpaka bench server listening on {}",
        listener.local_addr()?
    );
    println!("run `allpaka bench --connect <this-host>:<port>` on the other machine");

    for stream in listener.incoming() {
        let stream = stream.context("accepting connection")?;
        let peer = stream.peer_addr()?;
        println!("client connected: {peer}");
        if let Err(e) = handle(stream) {
            // A client that finishes its run and closes the socket is normal,
            // not an error worth aborting the server over.
            println!("client {peer} finished: {e}");
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true)?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let mut header = [0u8; 9];
        if stream.read_exact(&mut header).is_err() {
            return Ok(()); // peer hung up
        }
        let op = header[0];
        let len = u64::from_le_bytes(header[1..9].try_into().unwrap()) as usize;
        if len > CHUNK_BYTES {
            bail!("client asked for a {len} byte frame, over the {CHUNK_BYTES} limit");
        }
        stream.read_exact(&mut buf[..len])?;
        match op {
            OP_PING => stream.write_all(&buf[..len])?,
            OP_SINK => stream.write_all(&[0u8; 8])?,
            other => bail!("unknown op {other}"),
        }
        stream.flush()?;
    }
}

/// Connect to a bench server and measure the path to it.
pub fn measure(addr: &str) -> Result<Link> {
    let target = addr
        .to_socket_addrs()
        .with_context(|| format!("resolving {addr}"))?
        .next()
        .with_context(|| format!("no address found for {addr}"))?;
    let mut stream = TcpStream::connect(target).with_context(|| format!("connecting to {addr}"))?;
    // Without this, Nagle's algorithm batches the small ping frames and the
    // latency measurement reports the delayed-ack timer instead of the network.
    stream.set_nodelay(true)?;

    let mut rtts = Vec::with_capacity(PING_ROUNDS);
    let payload = vec![0u8; PING_BYTES];
    let mut echo = vec![0u8; PING_BYTES];
    for i in 0..PING_ROUNDS {
        let t0 = Instant::now();
        send_frame(&mut stream, OP_PING, &payload)?;
        stream.read_exact(&mut echo)?;
        if i >= PING_WARMUP {
            rtts.push(t0.elapsed().as_secs_f64());
        }
    }
    rtts.sort_by(f64::total_cmp);

    let chunk = vec![0u8; CHUNK_BYTES];
    let chunks = THROUGHPUT_BYTES / CHUNK_BYTES;
    let mut ack = [0u8; 8];
    // One untimed chunk so the congestion window is open before we start.
    send_frame(&mut stream, OP_SINK, &chunk)?;
    stream.read_exact(&mut ack)?;

    let t0 = Instant::now();
    for _ in 0..chunks {
        send_frame(&mut stream, OP_SINK, &chunk)?;
        stream.read_exact(&mut ack)?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let throughput = (chunks * CHUNK_BYTES) as f64 / elapsed;

    Ok(Link {
        throughput_bytes_per_sec: throughput,
        rtt_p50_secs: percentile(&rtts, 0.50),
        rtt_p99_secs: percentile(&rtts, 0.99),
    })
}

fn send_frame(stream: &mut TcpStream, op: u8, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; 9];
    header[0] = op;
    header[1..9].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole client/server protocol over loopback: what the two-machine
    /// run does, minus the cable. If this passes, a failure between real
    /// machines is the network or a firewall, not the tool.
    #[test]
    fn the_bench_protocol_round_trips_over_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let _ = handle(stream);
        });

        let link = measure(&addr).unwrap();
        server.join().unwrap();

        assert!(link.rtt_p50_secs > 0.0);
        assert!(link.rtt_p99_secs >= link.rtt_p50_secs);
        // Loopback moves gigabytes per second; anything below 100 MB/s means
        // the throughput phase measured something other than the transport.
        assert!(
            link.throughput_bytes_per_sec > 100e6,
            "{}",
            link.throughput_bytes_per_sec
        );
        // And it is fast: p50 over a millisecond on loopback would mean the
        // Nagle/delayed-ack trap is back.
        assert!(link.rtt_p50_secs < 1e-3, "{}", link.rtt_p50_secs);
    }

    #[test]
    fn percentiles_pick_the_expected_ranks() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 0.50), 50.0);
        assert_eq!(percentile(&v, 0.99), 99.0);
    }

    #[test]
    fn percentile_of_empty_is_infinite_rather_than_panicking() {
        assert!(percentile(&[], 0.5).is_infinite());
    }

    #[test]
    fn percentile_handles_a_single_sample() {
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }
}
