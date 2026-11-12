//! Programmatic Dependent Launch (PDL) for Hopper+/Blackwell.
//!
//! Host: `cuLaunchKernelEx` + `PROGRAMMATIC_STREAM_SERIALIZATION`.
//! Device: `griddepcontrol.wait` / `launch_dependents` in kernels.
//!
//! cudarc keeps the raw function handle and the argument vector private, and
//! `cuLaunchKernelEx` needs both. Their place inside `CudaFunction` and
//! `LaunchArgs` is not guessed: the first launch compares the candidate words
//! with the values the types print through their public `Debug` impls, and
//! only a match enables the PDL path. Without a match every launch goes
//! through cudarc's own `launch`.

use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{self, CUresult};
use cudarc::driver::{CudaFunction, CudaStream, LaunchArgs, LaunchConfig};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

thread_local! {
    static TLS_STREAM: Cell<*const CudaStream> = const { Cell::new(std::ptr::null()) };
    static TLS_CU: Cell<sys::CUfunction> = const { Cell::new(std::ptr::null_mut()) };
}

fn pdl_wanted() -> bool {
    static ON: AtomicBool = AtomicBool::new(false);
    static INIT: AtomicBool = AtomicBool::new(false);
    if !INIT.swap(true, Ordering::Relaxed) {
        // Off unless asked for; see docs/benchmarks/2026-11-13-cuda-pdl-rtx5090.md.
        let on = match std::env::var("ALLPAKA_CUDA_PDL") {
            Ok(v) => !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false")),
            Err(_) => false,
        };
        ON.store(on, Ordering::Relaxed);
        if on {
            eprintln!("cuda: PDL launches enabled by ALLPAKA_CUDA_PDL (experimental)");
        }
    }
    ON.load(Ordering::Relaxed)
}

static PDL_DEAD: AtomicBool = AtomicBool::new(false);
static USE_PDL_ATTR: AtomicBool = AtomicBool::new(true);

const WORD: usize = std::mem::size_of::<usize>();
const UNVERIFIED: usize = usize::MAX;
const REJECTED: usize = usize::MAX - 1;
/// Byte offset of `cu_function` inside `CudaFunction`, once verified.
static FUNC_OFF: AtomicUsize = AtomicUsize::new(UNVERIFIED);
/// `ptr_word << 8 | len_word` of the `args` vector inside `LaunchArgs`, once
/// verified; both are word indices from the start of the struct.
static ARGS_AT: AtomicUsize = AtomicUsize::new(UNVERIFIED);
/// The address after `key` in a derived `Debug` rendering (`key0x1f..`).
fn debug_ptr(text: &str, key: &str) -> Option<usize> {
    let rest = &text[text.find(key)? + key.len()..];
    let hex = rest.strip_prefix("0x")?;
    let end = hex
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(hex.len());
    usize::from_str_radix(&hex[..end], 16).ok()
}

/// The pointers of the last `args: [..]` list in a `LaunchArgs` rendering.
fn debug_args(text: &str) -> Option<Vec<usize>> {
    let rest = &text[text.rfind("args: [")? + "args: [".len()..];
    let list = &rest[..rest.find(']')?];
    list.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| usize::from_str_radix(item.strip_prefix("0x")?, 16).ok())
        .collect()
}

fn cu_ok(r: CUresult) -> Result<(), DriverError> {
    if r == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(r))
    }
}

/// The raw handle of `f`, or null when its place in `CudaFunction` could not
/// be verified.
fn cu_function_of(f: &CudaFunction) -> sys::CUfunction {
    const { assert!(std::mem::size_of::<CudaFunction>() == 2 * WORD) };
    let base = f as *const CudaFunction as *const usize;
    // SAFETY: `CudaFunction` is two words (asserted above), both pointers that are
    // always initialised (`cu_function` and an `Arc`), so each word read is in bounds
    // and reads initialised memory.
    let word = |i: usize| unsafe { std::ptr::read(base.add(i)) };
    let mut off = FUNC_OFF.load(Ordering::Relaxed);
    if off == UNVERIFIED {
        let printed = debug_ptr(&format!("{f:?}"), "cu_function: ");
        off = match printed {
            Some(handle) if word(1) == handle && word(0) != handle => WORD,
            Some(handle) if word(0) == handle && word(1) != handle => 0,
            _ => REJECTED,
        };
        FUNC_OFF.store(off, Ordering::Relaxed);
        if off == REJECTED {
            eprintln!("cuda: CudaFunction layout not recognised; PDL launches disabled");
        }
    }
    if off == REJECTED {
        return std::ptr::null_mut();
    }
    word(off / WORD) as sys::CUfunction
}

/// Find `args` inside `launch` by comparing it with what `Debug` prints.
/// Returns the packed position, `REJECTED`, or `UNVERIFIED` when this launch
/// cannot decide (no arguments, or length and capacity coincide).
fn verify_args(launch: &LaunchArgs<'_>) -> usize {
    let Some(printed) = debug_args(&format!("{launch:?}")) else {
        return REJECTED;
    };
    let n = printed.len();
    if n == 0 {
        return UNVERIFIED;
    }
    let words = std::mem::size_of::<LaunchArgs<'_>>() / WORD;
    let base = launch as *const LaunchArgs<'_> as *const usize;
    // SAFETY: every index is below the struct's size in words, so the read is in
    // bounds. All fields but `flags` are references and vectors, whose words are
    // initialised; `flags` is a 4-byte optional enum, and the word holding it is only
    // ever compared against, never used as a pointer or a length.
    let word = |i: usize| unsafe { std::ptr::read(base.add(i)) };
    let mut undecided = false;
    // A `Vec` is three adjacent words: buffer pointer, capacity and length, in an
    // order the compiler chooses.
    for first in 0..words.saturating_sub(2) {
        let triple = [word(first), word(first + 1), word(first + 2)];
        for len_at in 0..3 {
            if triple[len_at] != n {
                continue;
            }
            for cap_at in (0..3).filter(|&i| i != len_at) {
                let ptr_at = 3 - len_at - cap_at;
                let (cap, ptr) = (triple[cap_at], triple[ptr_at]);
                // The capacity is a small count no less than the length; the buffer
                // is a word-aligned heap address far above any such count.
                if !(n..=1024).contains(&cap) || ptr < 0x1_0000 || ptr % WORD != 0 {
                    continue;
                }
                if cap == n {
                    // Length and capacity look alike: a launch with spare capacity
                    // decides which word is the length.
                    undecided = true;
                    continue;
                }
                // SAFETY: `ptr` sits next to a length of `n` and a capacity above it,
                // which among this struct's fields only a non-empty vector provides;
                // its buffer holds at least `n` initialised words. The comparison
                // rejects every vector but `args`.
                let held = unsafe { std::slice::from_raw_parts(ptr as *const usize, n) };
                if held == printed.as_slice() {
                    return (first + ptr_at) << 8 | (first + len_at);
                }
            }
        }
    }
    if undecided {
        UNVERIFIED
    } else {
        REJECTED
    }
}

/// The kernel arguments collected in `launch`, or `None` when their place in
/// `LaunchArgs` is not (yet) verified.
fn args_of<'a>(launch: &'a mut LaunchArgs<'_>) -> Option<&'a mut [*mut std::ffi::c_void]> {
    let mut at = ARGS_AT.load(Ordering::Relaxed);
    if at == UNVERIFIED {
        at = verify_args(launch);
        if at == UNVERIFIED {
            return None;
        }
        ARGS_AT.store(at, Ordering::Relaxed);
        if at == REJECTED {
            eprintln!("cuda: LaunchArgs layout not recognised; PDL launches disabled");
            PDL_DEAD.store(true, Ordering::Relaxed);
        } else {
            eprintln!("cuda: PDL launch path verified against cudarc's layout");
        }
    }
    if at == REJECTED {
        return None;
    }
    let base = launch as *mut LaunchArgs<'_> as *mut usize;
    // SAFETY: `verify_args` matched these two words against the vector's printed
    // contents, and a type's field layout is fixed for the whole program, so they are the
    // buffer pointer and length of `launch.args` for every `LaunchArgs`. The slice borrows
    // `launch` mutably for `'a`, so nothing else touches the vector meanwhile.
    unsafe {
        let ptr = std::ptr::read(base.add(at >> 8)) as *mut *mut std::ffi::c_void;
        let len = std::ptr::read(base.add(at & 0xff));
        (len != 0).then(|| std::slice::from_raw_parts_mut(ptr, len))
    }
}

unsafe fn launch_ex(
    stream: &CudaStream,
    f: sys::CUfunction,
    cfg: LaunchConfig,
    args: &mut [*mut std::ffi::c_void],
    with_pdl: bool,
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    if with_pdl {
        let mut attr = std::mem::zeroed::<sys::CUlaunchAttribute>();
        attr.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
        let config = sys::CUlaunchConfig {
            gridDimX: cfg.grid_dim.0,
            gridDimY: cfg.grid_dim.1,
            gridDimZ: cfg.grid_dim.2,
            blockDimX: cfg.block_dim.0,
            blockDimY: cfg.block_dim.1,
            blockDimZ: cfg.block_dim.2,
            sharedMemBytes: cfg.shared_mem_bytes,
            hStream: stream.cu_stream(),
            attrs: &mut attr,
            numAttrs: 1,
        };
        cu_ok(sys::cuLaunchKernelEx(
            &config,
            f,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    } else {
        cudarc::driver::result::launch_kernel(
            f,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            args,
        )
    }
}

pub trait StreamPdlExt {
    fn launch_builder_pdl<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a>;
}

impl StreamPdlExt for CudaStream {
    #[inline(always)]
    fn launch_builder_pdl<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        TLS_STREAM.with(|c| c.set(self as *const CudaStream));
        let cu = cu_function_of(func);
        TLS_CU.with(|c| c.set(cu));
        self.launch_builder(func)
    }
}

pub unsafe fn launch_pdl_inner(
    launch: &mut LaunchArgs<'_>,
    cfg: LaunchConfig,
) -> Result<Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>, DriverError> {
    if !pdl_wanted() || PDL_DEAD.load(Ordering::Relaxed) {
        return launch.launch(cfg);
    }

    let stream_ptr = TLS_STREAM.with(|c| c.replace(std::ptr::null()));
    let f = TLS_CU.with(|c| c.replace(std::ptr::null_mut()));
    if stream_ptr.is_null() || f.is_null() {
        return launch.launch(cfg);
    }
    // SAFETY: `stream_ptr` was stored by `launch_builder_pdl` on this thread from the
    // `&'a CudaStream` that the `LaunchArgs<'a>` being launched also borrows, so the
    // stream is alive here. Pairing holds because every `launch_builder_pdl(..)` call in
    // this backend is followed by its own `.launch_pdl(..)` before any other
    // `launch_builder_pdl`, and the TLS slots are cleared (`replace(null)`) on each launch;
    // a builder from plain `launch_builder` finds null and takes the cudarc path.
    let stream = &*stream_ptr;

    let Some(args) = args_of(launch) else {
        return launch.launch(cfg);
    };
    let pdl = USE_PDL_ATTR.load(Ordering::Relaxed);
    match launch_ex(stream, f, cfg, args, pdl) {
        Ok(()) => return Ok(None),
        Err(_) if pdl => {
            // Fall back to plain cuLaunchKernelEx without PDL attr.
            if launch_ex(stream, f, cfg, args, false).is_ok() {
                eprintln!("cuda: PDL attr unsupported; using plain launches");
                USE_PDL_ATTR.store(false, Ordering::Relaxed);
                return Ok(None);
            }
        }
        Err(_) => {}
    }

    eprintln!("cuda: PDL launch failed; disabling PDL and falling back to cudarc launch");
    PDL_DEAD.store(true, Ordering::Relaxed);
    launch.launch(cfg)
}

pub trait LaunchArgsPdl {
    unsafe fn launch_pdl(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>, DriverError>;
}

impl LaunchArgsPdl for LaunchArgs<'_> {
    #[inline(always)]
    unsafe fn launch_pdl(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>, DriverError> {
        launch_pdl_inner(self, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::{debug_args, debug_ptr};

    #[test]
    fn debug_renderings_give_back_the_pointers() {
        let func =
            "CudaFunction { cu_function: 0x1f2e3d4c5b60, module: CudaModule { cu_module: 0x77 } }";
        assert_eq!(debug_ptr(func, "cu_function: "), Some(0x1f2e_3d4c_5b60));
        assert_eq!(debug_ptr(func, "missing: "), None);

        let launch = "LaunchArgs { stream: S { args: [0x1] }, waits: [], records: [], args: [0x10, 0xabc, 0x7ffe0008], flags: None }";
        assert_eq!(debug_args(launch), Some(vec![0x10, 0xabc, 0x7ffe_0008]));
        assert_eq!(
            debug_args("LaunchArgs { args: [], flags: None }"),
            Some(vec![])
        );
        assert_eq!(debug_args("LaunchArgs { args: [nonsense] }"), None);
        assert_eq!(debug_args("LaunchArgs { }"), None);
    }
}
