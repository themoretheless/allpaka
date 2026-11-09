//! Persistent CPU worker pool for parallel tensor operations.
//!
//! Replaces per-operation `std::thread::scope` calls on hot paths so zero OS
//! threads are spawned during prefill or decode.

use std::any::Any;
use std::cell::Cell;
use std::sync::{Condvar, Mutex, OnceLock};

thread_local! {
    static IN_WORKER: Cell<bool> = const { Cell::new(false) };
}

/// Erased stack-scoped closure pointer (`&dyn Fn(usize) + Sync`).
#[derive(Clone, Copy)]
struct TaskFn {
    data: usize,
    call: unsafe fn(usize, usize),
}

impl TaskFn {
    #[inline]
    unsafe fn invoke<F: Fn(usize) + Sync>(data: usize, idx: usize) {
        let f = &*(data as *const F);
        f(idx);
    }

    #[inline]
    fn from_ref<F: Fn(usize) + Sync>(f: &F) -> Self {
        Self {
            data: f as *const F as usize,
            call: Self::invoke::<F>,
        }
    }

    #[inline]
    unsafe fn run(self, idx: usize) {
        (self.call)(self.data, idx);
    }
}

struct PoolState {
    generation: u64,
    active_workers: usize,
    remaining: usize,
    task: Option<TaskFn>,
    panic_payload: Option<Box<dyn Any + Send + 'static>>,
}

struct SharedPool {
    state: Mutex<PoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct GlobalPool {
    shared: &'static SharedPool,
    dispatch_lock: Mutex<usize>,
}

static POOL: OnceLock<GlobalPool> = OnceLock::new();

fn global_pool() -> &'static GlobalPool {
    POOL.get_or_init(|| {
        let shared: &'static SharedPool = Box::leak(Box::new(SharedPool {
            state: Mutex::new(PoolState {
                generation: 0,
                active_workers: 0,
                remaining: 0,
                task: None,
                panic_payload: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        }));
        let initial_workers = crate::runtime::cpu_threads().saturating_sub(1);
        for worker_id in 1..=initial_workers {
            spawn_worker(shared, worker_id, 0);
        }
        GlobalPool {
            shared,
            dispatch_lock: Mutex::new(initial_workers),
        }
    })
}

fn spawn_worker(shared: &'static SharedPool, worker_id: usize, initial_gen: u64) {
    std::thread::Builder::new()
        .name(format!("allpaka-worker-{worker_id}"))
        .spawn(move || {
            IN_WORKER.with(|flag| flag.set(true));
            let mut seen_gen = initial_gen;
            loop {
                let task = {
                    let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    while st.generation == seen_gen || worker_id > st.active_workers {
                        seen_gen = st.generation;
                        st = shared.work_cv.wait(st).unwrap_or_else(|e| e.into_inner());
                    }
                    seen_gen = st.generation;
                    st.task.expect("active generation must have a task")
                };

                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    task.run(worker_id);
                }));

                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(payload) = res {
                    if st.panic_payload.is_none() {
                        st.panic_payload = Some(payload);
                    }
                }
                st.remaining -= 1;
                if st.remaining == 0 {
                    shared.done_cv.notify_one();
                }
            }
        })
        .expect("failed to spawn allpaka worker thread");
}

#[inline]
pub fn cpu_threads() -> usize {
    crate::runtime::cpu_threads()
}

/// Execute `f(worker_idx)` for `worker_idx in 0..n_workers` across the
/// persistent thread pool. Worker `0` executes on the calling thread.
#[inline]
pub fn par_workers<F>(n_workers: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n_workers == 0 {
        return;
    }
    if n_workers == 1 || IN_WORKER.with(|flag| flag.get()) {
        for i in 0..n_workers {
            f(i);
        }
        return;
    }

    let pool = global_pool();
    let mut spawned = pool
        .dispatch_lock
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let needed_extra = n_workers - 1;

    let task = TaskFn::from_ref(&f);
    {
        let mut st = pool
            .shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while *spawned < needed_extra {
            *spawned += 1;
            spawn_worker(pool.shared, *spawned, st.generation);
        }
        st.active_workers = needed_extra;
        st.remaining = needed_extra;
        st.task = Some(task);
        st.panic_payload = None;
        st.generation = st.generation.wrapping_add(1);
    }
    pool.shared.work_cv.notify_all();

    struct InWorkerGuard;
    impl Drop for InWorkerGuard {
        fn drop(&mut self) {
            IN_WORKER.with(|flag| flag.set(false));
        }
    }
    IN_WORKER.with(|flag| flag.set(true));
    let main_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = InWorkerGuard;
        f(0);
    }));

    let panic_payload = {
        let mut st = pool
            .shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while st.remaining > 0 {
            st = pool
                .shared
                .done_cv
                .wait(st)
                .unwrap_or_else(|e| e.into_inner());
        }
        st.task = None;
        st.active_workers = 0;
        st.panic_payload.take()
    };
    drop(spawned);

    if let Err(payload) = main_res {
        std::panic::resume_unwind(payload);
    }
    if let Some(payload) = panic_payload {
        std::panic::resume_unwind(payload);
    }
}

/// Partition `0..total_items` into up to `n_workers` contiguous chunks and
/// invoke `f(chunk_idx, start, end)` in parallel for each non-empty chunk.
#[inline]
pub fn par_for_chunks<F>(total_items: usize, n_workers: usize, f: F)
where
    F: Fn(usize, usize, usize) + Sync,
{
    if total_items == 0 || n_workers == 0 {
        return;
    }
    let nt = n_workers.min(total_items);
    if nt <= 1 {
        f(0, 0, total_items);
        return;
    }
    let items_per = total_items.div_ceil(nt);
    let actual_chunks = total_items.div_ceil(items_per);
    par_workers(actual_chunks, |tid| {
        let start = tid * items_per;
        let end = (start + items_per).min(total_items);
        if start < end {
            f(tid, start, end);
        }
    });
}

/// Split `slice` into mutable chunks of `chunk_size` elements and invoke
/// `f(chunk_idx, start_offset, chunk_slice)` in parallel on the persistent pool.
#[inline]
pub fn par_chunks_mut<T, F>(slice: &mut [T], chunk_size: usize, f: F)
where
    T: Send,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    let total = slice.len();
    if total == 0 || chunk_size == 0 {
        return;
    }
    let n_chunks = total.div_ceil(chunk_size);
    if n_chunks <= 1 {
        f(0, 0, slice);
        return;
    }
    let base_addr = slice.as_mut_ptr() as usize;
    par_workers(n_chunks, |chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(total);
        if start < end {
            let chunk = unsafe {
                std::slice::from_raw_parts_mut((base_addr as *mut T).add(start), end - start)
            };
            f(chunk_idx, start, chunk);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_chunks_mut_writes_disjoint_ranges() {
        let mut out = vec![0usize; 103];
        par_chunks_mut(&mut out, 16, |chunk_idx, start, sub| {
            for (i, slot) in sub.iter_mut().enumerate() {
                *slot = chunk_idx * 1000 + start + i;
            }
        });
        for (idx, &val) in out.iter().enumerate() {
            let chunk_idx = idx / 16;
            assert_eq!(val, chunk_idx * 1000 + idx);
        }
    }

    #[test]
    fn par_for_chunks_covers_all_elements() {
        let n = 257;
        let mut out = vec![0usize; n];
        let base = out.as_mut_ptr() as usize;
        par_for_chunks(n, 8, |_tid, start, end| {
            let sub =
                unsafe { std::slice::from_raw_parts_mut((base as *mut usize).add(start), end - start) };
            for (i, slot) in sub.iter_mut().enumerate() {
                *slot = start + i + 1;
            }
        });
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i + 1);
        }
    }

    #[test]
    fn nested_par_workers_does_not_deadlock() {
        let mut data = vec![0u32; 64];
        par_chunks_mut(&mut data, 16, |_cid, _start, chunk| {
            par_chunks_mut(chunk, 4, |_inner_cid, _inner_start, sub| {
                for v in sub.iter_mut() {
                    *v += 7;
                }
            });
        });
        assert!(data.iter().all(|&v| v == 7));
    }
}
