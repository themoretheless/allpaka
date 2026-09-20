//! One-second resource samples for the chat page. Sampling stays off the
//! inference thread: GPU numbers come from nvidia-smi, RAM from the process.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

const HISTORY: usize = 120;

struct Sample {
    vram_mib: u64,
    util: u32,
    rss_mib: u64,
    tok_s: f64,
}

struct State {
    gpu_name: String,
    gpu_used_mib: u64,
    gpu_total_mib: u64,
    gpu_util: u32,
    gpu_temp_c: i32,
    gpu_power_w: f64,
    rss_mib: u64,
    ram_used_mib: u64,
    ram_total_mib: u64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    context_used: usize,
    context_capacity: usize,
    samples: VecDeque<Sample>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

pub fn start() {
    let state = STATE.get_or_init(|| {
        Mutex::new(State {
            gpu_name: String::new(),
            gpu_used_mib: 0,
            gpu_total_mib: 0,
            gpu_util: 0,
            gpu_temp_c: 0,
            gpu_power_w: 0.0,
            rss_mib: 0,
            ram_used_mib: 0,
            ram_total_mib: 0,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
            context_used: 0,
            context_capacity: 0,
            samples: VecDeque::with_capacity(HISTORY),
        })
    });
    let _ = state;
    std::thread::spawn(|| loop {
        sample();
        std::thread::sleep(std::time::Duration::from_secs(1));
    });
}

pub fn note_generation(
    prefill_tok_s: f64,
    decode_tok_s: f64,
    context_used: usize,
    context_capacity: usize,
) {
    let Some(state) = STATE.get() else {
        return;
    };
    let Ok(mut state) = state.lock() else {
        return;
    };
    if prefill_tok_s > 0.0 {
        state.prefill_tok_s = prefill_tok_s;
    }
    if decode_tok_s > 0.0 {
        state.decode_tok_s = decode_tok_s;
    }
    state.context_used = context_used;
    state.context_capacity = context_capacity;
}

pub fn snapshot() -> Value {
    let Some(state) = STATE.get() else {
        return json!({});
    };
    let Ok(state) = state.lock() else {
        return json!({});
    };
    let mut vram = Vec::with_capacity(state.samples.len());
    let mut util = Vec::with_capacity(state.samples.len());
    let mut rss = Vec::with_capacity(state.samples.len());
    let mut tok_s = Vec::with_capacity(state.samples.len());
    for sample in &state.samples {
        vram.push(sample.vram_mib);
        util.push(sample.util);
        rss.push(sample.rss_mib);
        tok_s.push(sample.tok_s);
    }
    json!({
        "gpu_name": state.gpu_name,
        "gpu_used_mib": state.gpu_used_mib,
        "gpu_total_mib": state.gpu_total_mib,
        "gpu_util": state.gpu_util,
        "gpu_temp_c": state.gpu_temp_c,
        "gpu_power_w": state.gpu_power_w,
        "rss_mib": state.rss_mib,
        "ram_used_mib": state.ram_used_mib,
        "ram_total_mib": state.ram_total_mib,
        "prefill_tok_s": state.prefill_tok_s,
        "decode_tok_s": state.decode_tok_s,
        "context_used": state.context_used,
        "context_capacity": state.context_capacity,
        "vram": vram,
        "util": util,
        "rss": rss,
        "tok_s": tok_s,
    })
}

fn sample() {
    let Some(state) = STATE.get() else {
        return;
    };
    let gpu = gpu_query();
    let (rss_mib, ram_used_mib, ram_total_mib) = memory_mib();
    let Ok(mut state) = state.lock() else {
        return;
    };
    if let Some((name, used, total, util, temp, power)) = gpu {
        if !name.is_empty() {
            state.gpu_name = name;
        }
        state.gpu_used_mib = used;
        state.gpu_total_mib = total;
        state.gpu_util = util;
        state.gpu_temp_c = temp;
        state.gpu_power_w = power;
    }
    state.rss_mib = rss_mib;
    state.ram_used_mib = ram_used_mib;
    state.ram_total_mib = ram_total_mib;
    let vram_mib = state.gpu_used_mib;
    let util = state.gpu_util;
    let tok_s = state.decode_tok_s;
    state.samples.push_back(Sample {
        vram_mib,
        util,
        rss_mib,
        tok_s,
    });
    while state.samples.len() > HISTORY {
        state.samples.pop_front();
    }
}

fn gpu_query() -> Option<(String, u64, u64, u32, i32, f64)> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]);
    cmd.stdin(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 6 {
        return None;
    }
    Some((
        parts[0].to_string(),
        parts[1].parse().unwrap_or(0),
        parts[2].parse().unwrap_or(0),
        parts[3].parse().unwrap_or(0),
        parts[4].parse().unwrap_or(0),
        parts[5].parse().unwrap_or(0.0),
    ))
}

fn memory_mib() -> (u64, u64, u64) {
    #[cfg(windows)]
    {
        windows_memory_mib()
    }
    #[cfg(not(windows))]
    {
        (0, 0, 0)
    }
}

#[cfg(windows)]
fn windows_memory_mib() -> (u64, u64, u64) {
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[repr(C)]
    struct MemStat {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
        fn GlobalMemoryStatusEx(status: *mut MemStat) -> i32;
    }
    unsafe {
        let mut counters = std::mem::zeroed::<Counters>();
        counters.cb = std::mem::size_of::<Counters>() as u32;
        let rss = if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0
        {
            counters.working_set_size as u64
        } else {
            0
        };
        let mut status = std::mem::zeroed::<MemStat>();
        status.length = std::mem::size_of::<MemStat>() as u32;
        let (used, total) = if GlobalMemoryStatusEx(&mut status) != 0 {
            (
                status.total_phys.saturating_sub(status.avail_phys),
                status.total_phys,
            )
        } else {
            (0, 0)
        };
        (rss / (1024 * 1024), used / (1024 * 1024), total / (1024 * 1024))
    }
}
