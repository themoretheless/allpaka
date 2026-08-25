//! What one command buffer costs when it does no work.
//!
//! Decode submits three command buffers per layer and blocks on each, so the
//! obvious suspicion is that the round trips themselves are the bill. That is
//! a claim about a constant, and the constant is measurable: submit empty
//! command buffers and time them. Multiply the result by the waits per token
//! reported by `allpaka bench` to get the ceiling on what merging submissions
//! could ever win - if that ceiling is small next to the token's wall time,
//! the time is in the kernels and merging is not the fix.
//!
//! Ignored by default (it is a measurement, not an assertion). Run with
//! `cargo test -p allpaka-backend --test gpu_roundtrip -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions, MTLSize};

const ROUNDS: usize = 2000;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a timing measurement; run explicitly"]
fn an_empty_command_buffer_still_costs_a_round_trip() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();

    // Bare submit and block: no encoder, no dispatch, nothing for the GPU to
    // do. Whatever this costs, every command buffer pays it.
    let mut bare = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        objc::rc::autoreleasepool(|| {
            let t = std::time::Instant::now();
            let cmd = queue.new_command_buffer();
            cmd.commit();
            cmd.wait_until_completed();
            bare.push(t.elapsed().as_secs_f64() * 1e6);
        });
    }

    // The same with an empty compute encoder, which is what the real path
    // always creates.
    let mut encoded = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        objc::rc::autoreleasepool(|| {
            let t = std::time::Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            encoded.push(t.elapsed().as_secs_f64() * 1e6);
        });
    }

    // And with one trivial dispatch, to separate "a command buffer exists"
    // from "the GPU was asked to run something".
    let src = r#"
#include <metal_stdlib>
kernel void nothing(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    y[i] = y[i] + 1.0f;
}
"#;
    let lib = device
        .new_library_with_source(src, &metal::CompileOptions::new())
        .expect("kernel compiles");
    let f = lib.get_function("nothing", None).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&f).unwrap();
    let buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

    let mut dispatched = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        objc::rc::autoreleasepool(|| {
            let t = std::time::Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&buf), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            dispatched.push(t.elapsed().as_secs_f64() * 1e6);
        });
    }

    // The question merging hinges on: is the cost per command buffer or per
    // dispatch? Same trivial kernel, N dispatches inside ONE submission. A
    // flat curve means the submissions are the bill and merging them wins;
    // a linear one means the dispatches are, and merging changes nothing.
    let mut scaling = Vec::new();
    for n in [1usize, 2, 4, 16, 64, 256] {
        let mut times = Vec::with_capacity(ROUNDS / 4);
        for _ in 0..ROUNDS / 4 {
            objc::rc::autoreleasepool(|| {
                let t = std::time::Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for _ in 0..n {
                    enc.set_compute_pipeline_state(&pipe);
                    enc.set_buffer(0, Some(&buf), 0);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                times.push(t.elapsed().as_secs_f64() * 1e6);
            });
        }
        scaling.push((n, median(times)));
    }

    let (b, e, d) = (median(bare), median(encoded), median(dispatched));
    println!("empty command buffer:      {b:>7.1} us median");
    println!("+ empty compute encoder:   {e:>7.1} us median");
    println!("+ one trivial dispatch:    {d:>7.1} us median");
    for (n, us) in &scaling {
        println!("{n:>4} dispatches in one command buffer: {us:>8.1} us  ({:.2} us each)", us / *n as f64);
    }
    // The numbers that decide whether merging submissions is worth writing:
    // waits per token as reported by `allpaka bench --engine`.
    for (model, waits, ms_per_token) in [("30B", 145.0, 62.1), ("235B", 283.0, 219.3)] {
        let floor_ms = d * waits / 1000.0;
        println!(
            "{model}: {waits:.0} waits/token x {d:.1} us = {floor_ms:.1} ms/token of \
             {ms_per_token:.1} ms ({:.0}% is pure round trip)",
            floor_ms / ms_per_token * 100.0
        );
    }
}
