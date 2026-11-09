//! One-second resource samples for the chat page. Sampling stays off the
//! inference thread: GPU numbers come from nvidia-smi or Apple unified memory,
//! RAM from the OS process/host counters.

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

/// Metrics for the resources page, served before the request queue so a
/// busy model cannot starve it. `context_used` / `context_capacity` are the
/// values captured at the end of the last generation, not live ones.
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
    let (rss_mib, ram_used_mib, ram_total_mib) = memory_mib();
    let gpu = gpu_query(rss_mib, ram_total_mib);
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

fn gpu_query(rss_mib: u64, ram_total_mib: u64) -> Option<(String, u64, u64, u32, i32, f64)> {
    #[cfg(target_os = "macos")]
    {
        static GPU_NAME: OnceLock<String> = OnceLock::new();
        let name = GPU_NAME
            .get_or_init(|| {
                macos_sysctl_string(b"machdep.cpu.brand_string\0")
                    .map(|brand| format!("{brand} (Metal)"))
                    .unwrap_or_else(|| "Apple Metal GPU".to_string())
            })
            .clone();
        Some((name, rss_mib, ram_total_mib, 0, 0, 0.0))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (rss_mib, ram_total_mib);
        use std::sync::atomic::{AtomicBool, Ordering};
        static NVIDIA_SMI_MISSING: AtomicBool = AtomicBool::new(false);
        if NVIDIA_SMI_MISSING.load(Ordering::Relaxed) {
            return None;
        }
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
        let out = match cmd.output() {
            Ok(out) => out,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    NVIDIA_SMI_MISSING.store(true, Ordering::Relaxed);
                }
                return None;
            }
        };
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
}

fn memory_mib() -> (u64, u64, u64) {
    #[cfg(windows)]
    {
        windows_memory_mib()
    }
    #[cfg(target_os = "linux")]
    {
        linux_memory_mib()
    }
    #[cfg(target_os = "macos")]
    {
        macos_memory_mib()
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        (0, 0, 0)
    }
}

#[cfg(target_os = "macos")]
fn macos_sysctl_string(name_cstr: &[u8]) -> Option<String> {
    extern "C" {
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    unsafe {
        let mut len: usize = 0;
        if sysctlbyname(
            name_cstr.as_ptr().cast(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null(),
            0,
        ) != 0
            || len == 0
        {
            return None;
        }
        let mut buf = vec![0u8; len];
        if sysctlbyname(
            name_cstr.as_ptr().cast(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null(),
            0,
        ) != 0
        {
            return None;
        }
        if let Some(nul) = buf.iter().position(|&b| b == 0) {
            buf.truncate(nul);
        }
        String::from_utf8(buf).ok().map(|s| s.trim().to_string())
    }
}

#[cfg(target_os = "macos")]
fn macos_memory_mib() -> (u64, u64, u64) {
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: i32,
        suspend_count: i32,
    }

    #[repr(C)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    extern "C" {
        static mach_task_self_: u32;
        fn mach_host_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_outCnt: *mut u32,
        ) -> i32;
        fn host_page_size(host: u32, out_page_size: *mut usize) -> i32;
        fn host_statistics64(
            host: u32,
            flavor: i32,
            host_info64_out: *mut VmStatistics64,
            host_info64_outCnt: *mut u32,
        ) -> i32;
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    const MACH_TASK_BASIC_INFO: u32 = 20;
    const HOST_VM_INFO64: i32 = 4;

    static TOTAL_RAM_BYTES: OnceLock<u64> = OnceLock::new();
    let total_bytes = *TOTAL_RAM_BYTES.get_or_init(|| unsafe {
        let mut memsize: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        if sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut memsize as *mut u64).cast(),
            &mut size,
            std::ptr::null(),
            0,
        ) == 0
        {
            memsize
        } else {
            0
        }
    });

    unsafe {
        let mut info = std::mem::zeroed::<MachTaskBasicInfo>();
        let mut count =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
        let rss_bytes =
            if task_info(mach_task_self_, MACH_TASK_BASIC_INFO, &mut info, &mut count) == 0 {
                info.resident_size
            } else {
                0
            };

        let host = mach_host_self();
        let mut page_size: usize = 16384;
        let _ = host_page_size(host, &mut page_size);
        let mut vm_stat = std::mem::zeroed::<VmStatistics64>();
        let mut vm_count =
            (std::mem::size_of::<VmStatistics64>() / std::mem::size_of::<u32>()) as u32;
        let used_bytes =
            if host_statistics64(host, HOST_VM_INFO64, &mut vm_stat, &mut vm_count) == 0 {
                let used_pages = u64::from(vm_stat.active_count)
                    + u64::from(vm_stat.wire_count)
                    + u64::from(vm_stat.compressor_page_count);
                used_pages.saturating_mul(page_size as u64)
            } else {
                0
            };

        (rss_bytes >> 20, used_bytes >> 20, total_bytes >> 20)
    }
}

#[cfg(target_os = "linux")]
fn linux_memory_mib() -> (u64, u64, u64) {
    let rss_mib = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| (pages * 4096) >> 20)
        .unwrap_or(0);
    let mut total_kib = 0u64;
    let mut avail_kib = 0u64;
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total_kib = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                avail_kib = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }
    }
    let total_mib = total_kib >> 10;
    let used_mib = total_kib.saturating_sub(avail_kib) >> 10;
    (rss_mib, used_mib, total_mib)
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
        let rss = if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
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
        (
            rss / (1024 * 1024),
            used / (1024 * 1024),
            total / (1024 * 1024),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{gpu_query, memory_mib};

    #[test]
    fn host_memory_and_gpu_queries_report_positive_capacity() {
        let (rss_mib, used_mib, total_mib) = memory_mib();
        assert!(total_mib > 0, "expected total_mib > 0, got {total_mib}");
        assert!(used_mib > 0, "expected used_mib > 0, got {used_mib}");
        let _ = rss_mib;
        #[cfg(target_os = "macos")]
        {
            let gpu = gpu_query(rss_mib, total_mib).expect("macos gpu query");
            assert!(!gpu.0.is_empty());
            assert_eq!(gpu.2, total_mib);
        }
    }
}
