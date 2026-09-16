//! Programmatic Dependent Launch (PDL) for Hopper+/Blackwell.
//!
//! Host: `cuLaunchKernelEx` + `PROGRAMMATIC_STREAM_SERIALIZATION`.
//! Device: `griddepcontrol.wait` / `launch_dependents` in kernels.
//!
//! Layout (cudarc 0.19, 64-bit):
//!   CudaFunction: Arc module @0, CUfunction @8 (slot=2)
//!   LaunchArgs: stream@0, func@8, waits@16 (24), records@40 (24), args@64

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
        let on = match std::env::var("ALLPAKA_CUDA_PDL") {
            // Default on for Blackwell decode (measured ~+0.5–1 tg on 5090).
            Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
            Err(_) => true,
        };
        ON.store(on, Ordering::Relaxed);
        if on {
            eprintln!("cuda: PDL launches enabled (ALLPAKA_CUDA_PDL=0 to disable)");
        }
    }
    ON.load(Ordering::Relaxed)
}

static PDL_DEAD: AtomicBool = AtomicBool::new(false);
static ARGS_OFF: AtomicUsize = AtomicUsize::new(64); // fixed: LaunchArgs.args
static USE_PDL_ATTR: AtomicBool = AtomicBool::new(true);

fn cu_ok(r: CUresult) -> Result<(), DriverError> {
    if r == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(r))
    }
}

unsafe fn cu_function_of(f: &CudaFunction) -> sys::CUfunction {
    // cudarc 0.19: Arc<Module> then CUfunction (or reverse). Prefer offset 8.
    let p = f as *const CudaFunction as *const u8;
    let a = std::ptr::read(p as *const sys::CUfunction);
    let b = std::ptr::read(p.add(8) as *const sys::CUfunction);
    // Heuristic: live CUfunction is non-null and not looking like a small Arc refcount.
    if !b.is_null() {
        b
    } else {
        a
    }
}

unsafe fn args_vec<'a, 'b>(
    launch: &'a mut LaunchArgs<'b>,
    off: usize,
) -> Option<&'a mut Vec<*mut std::ffi::c_void>> {
    let base = launch as *mut LaunchArgs<'b> as *mut u8;
    let v = &mut *(base.add(off) as *mut Vec<*mut std::ffi::c_void>);
    let n = v.len();
    if n == 0 || n > 64 || v.capacity() < n {
        return None;
    }
    Some(v)
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
        let cu = unsafe { cu_function_of(func) };
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

    let stream_ptr = TLS_STREAM.with(|c| c.get());
    let f = TLS_CU.with(|c| c.get());
    if stream_ptr.is_null() || f.is_null() {
        return launch.launch(cfg);
    }
    let stream = &*stream_ptr;
    let off = ARGS_OFF.load(Ordering::Relaxed);

    if let Some(args) = args_vec(launch, off) {
        let pdl = USE_PDL_ATTR.load(Ordering::Relaxed);
        match launch_ex(stream, f, cfg, args.as_mut_slice(), pdl) {
            Ok(()) => return Ok(None),
            Err(_) if pdl => {
                // Fall back to plain cuLaunchKernelEx without PDL attr.
                if let Some(args) = args_vec(launch, off) {
                    if launch_ex(stream, f, cfg, args.as_mut_slice(), false).is_ok() {
                        eprintln!("cuda: PDL attr unsupported; using plain launches");
                        USE_PDL_ATTR.store(false, Ordering::Relaxed);
                        return Ok(None);
                    }
                }
            }
            Err(_) => {}
        }
    }

    // One-shot scan if fixed offset failed (cudarc layout drift).
    const OFFS: &[usize] = &[64, 72, 80, 88, 56, 48, 40, 96];
    for &try_off in OFFS {
        let Some(args) = args_vec(launch, try_off) else {
            continue;
        };
        if launch_ex(stream, f, cfg, args.as_mut_slice(), false).is_err() {
            continue;
        }
        ARGS_OFF.store(try_off, Ordering::Relaxed);
        USE_PDL_ATTR.store(true, Ordering::Relaxed);
        eprintln!("cuda: PDL args offset calibrated to {try_off}");
        return Ok(None);
    }

    eprintln!("cuda: PDL calibration failed; disabling PDL");
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
